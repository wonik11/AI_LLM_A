import os
import io
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from supabase import create_client, Client

# Langchain 컴포넌트
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 환경 변수 로드
load_dotenv()

# 환경 변수 검증 (구글, 오픈AI, Supabase)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not all([GOOGLE_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("필수 환경 변수(구글, 오픈AI, Supabase)가 설정되지 않았습니다.")

# 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

app = FastAPI(title="City Policy RAG API")

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []  # [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF를 텍스트로 변환, 청크 분할, 벡터 임베딩 후 Supabase에 저장합니다."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="업로드된 파일이 PDF 형식이 아닙니다.")
    
    try:
        # 1. PDF 텍스트 추출
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        full_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
                
        if not full_text.strip():
            raise HTTPException(status_code=422, detail="PDF에서 텍스트를 추출할 수 없습니다. (스캔본이거나 텍스트가 없음)")

        # 2. 텍스트 청크 단위 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(full_text)
        
        # 3. OpenAI 임베딩 생성 및 Supabase 적재
        for i, chunk in enumerate(chunks):
            try:
                # 청크 임베딩 변환
                embedding_vector = embeddings.embed_query(chunk)
                # 데이터베이스 삽입
                data = {
                    "content": chunk,
                    "metadata": {"source": file.filename, "chunk_index": i},
                    "embedding": embedding_vector
                }
                supabase.table("document_chunks").insert(data).execute()
            except Exception as e:
                # 개별 청크 삽입 실패 시 상세 에러 처리
                raise HTTPException(status_code=500, detail=f"청크 {i} 임베딩/저장 중 오류 발생: {str(e)}")
                
        return {"status": "success", "message": f"{len(chunks)}개의 청크가 성공적으로 저장되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 예기치 않은 서버 오류가 발생했습니다: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """사용자 질의를 기반으로 문서를 검색하고 스트리밍 답변을 생성합니다."""
    
    # 1. 대화 기록(Chat History) 객체 변환
    history_messages = []
    for msg in request.chat_history:
        if msg.get("role") == "user":
            history_messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "ai":
            history_messages.append(AIMessage(content=msg.get("content")))

    try:
        # 2. Query Augmentation Chain 구성 (질의 확장)
        # LLM에게 부여할 역할 및 이전 대화 내역(MessagesPlaceholder)을 포함하는 시스템 메시지 설계
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
        
        # 질의 확장을 위한 LLM 모델 객체 생성
        condense_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        
        # 프롬프트, LLM, 파서를 하나로 연결
        query_augmentation_chain = condense_prompt | condense_llm | StrOutputParser()
        
        # 체인 실행 및 데이터 주입: 이전 대화 기록이 없으면 원본 질문 그대로 사용
        if history_messages:
            augmented_query = query_augmentation_chain.invoke({
                "chat_history": history_messages,
                "query": request.query
            })
        else:
            augmented_query = request.query

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 확장(Query Augmentation) 처리 중 오류 발생: {str(e)}")

    try:
        # 3. 증강된 질의로 벡터 DB 검색 (Supabase Retriever 역할)
        query_embedding = embeddings.embed_query(augmented_query)
        
        # RPC 호출을 통해 유사도가 높은 문서 청크 반환 (threshold: 0.7, 최대 5개)
        rpc_response = supabase.rpc(
            "match_document_chunks",
            {"query_embedding": query_embedding, "match_threshold": 0.7, "match_count": 5}
        ).execute()
        
        retrieved_docs = rpc_response.data
        context_text = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else "관련된 정책 문서를 찾을 수 없습니다."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase 벡터 검색 중 오류 발생: {str(e)}")

    try:
        # 4. 최종 RAG 답변 생성을 위한 시스템 프롬프트 및 LLM 설정
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 다음 제공된 문서 정보를 바탕으로 사용자의 질문에 전문적으로 답변해. 문서에 없는 내용은 지어내지 마.\n\n[문서 정보]\n{context}"),
            ("human", "{question}")
        ])
        
        # 실시간 스트리밍 핸들러 적용 (ChatGoogleGenerativeAI의 streaming=True 옵션 사용)
        rag_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            streaming=True,
            temperature=0.2
        )
        
        rag_chain = rag_prompt | rag_llm | StrOutputParser()
        
        # 5. 스트리밍 응답 제너레이터 함수
        async def generate_stream():
            try:
                async for chunk in rag_chain.astream({"context": context_text, "question": augmented_query}):
                    yield chunk
            except Exception as stream_e:
                yield f"\n[스트리밍 생성 중 오류 발생: {str(stream_e)}]"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"최종 답변 생성 및 스트리밍 초기화 중 오류 발생: {str(e)}")
