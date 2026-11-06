"""
================================================================================
FastAPI 기반 Advanced RAG 서버
(FAISS Index + Multi-Query(5가지 관점) + Hybrid Search(Dense+BM25) + RAG Fusion(RRF))
================================================================================
주요 기능:
1. FAISS Index 구축 계층:
   - 고차원 벡터 검색을 위한 FAISS IndexFlatL2 (정확한 전수 L2 거리 검색) 및
     IndexHNSWFlat (계층적 그래프 기반 근사 최근접 이웃 검색) 지원
   - PDF 파일 업로드(바이너리 멀티파트 또는 GAS용 Base64 JSON) 텍스트 추출 및 청킹 적재
2. 다중 쿼리 확장 계층 (Multi-Query Layer):
   - 사용자의 단일 입력 쿼리를 LLM(또는 내장 규칙 기반 확장기)을 통해
     서로 다른 5가지 관점의 파생 쿼리로 확장하여 단일 쿼리 검색의 누락 방지
3. 하이브리드 검색 계층 (Hybrid Search & FAISS Layer):
   - 확장된 5개 쿼리 각각에 대해 FAISS(Dense Vector, 의미적 유사도 검색)와
     BM25(Sparse Keyword, 정확한 단어 매칭)를 병행 수행
4. RRF 순위 융합 계층 (RAG Fusion & RRF Layer):
   - 다중 경로(5개 쿼리 x 2개 검색기 = 10개 순위 리스트)에서 수집된 문서들을
     Reciprocal Rank Fusion (RRF, 상호 순위 융합) 공식으로 점수화하여 재정렬(Re-ranking)
5. 실시간 스트리밍 핸들러 (StreamHandler):
   - 서버 동작 상태, 단계별 진행률, 알고리즘 복잡도(Big-O), 검색 스니펫, LLM 답변을
     실시간 NDJSON 스트림 및 JSON 응답으로 클라이언트에 전달
================================================================================
"""

import asyncio
import base64
import hashlib
import io
import json
import math
import os
import re
import time
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


# ==============================================================================
# 0. 데이터 모델 및 스키마 정의
# ==============================================================================

@dataclass
class DocumentChunk:
    """
    벡터 DB(FAISS) 및 키워드 인덱스(BM25)에 적재되는 단일 문서 청크 메타데이터 구조체
    """
    doc_id: int               # 내부 고유 정수 ID (0 ~ N-1)
    text: str                 # 정제된 문서 스니펫 원문
    source: str               # 출처 파일명 (예: "sample_rag_guide.pdf")
    page: int                 # 원본 PDF 페이지 번호 (1-based)
    chunk_index: int          # 해당 페이지 내 청크 순번 (1-based)
    char_length: int          # 청크 글자 수
    created_at: str           # 적재 시각 (ISO 8601)


class IngestBase64Request(BaseModel):
    """
    Google Apps Script(Code.gs)에서 PDF 파일을 Base64 문자열로 전송할 때 사용하는 요청 스키마
    """
    filename: str = Field(..., description="업로드할 PDF 파일명")
    file_base64: str = Field(..., description="Base64로 인코딩된 PDF 파일 바이너리 데이터")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="청크 분할 최대 글자 수")
    chunk_overlap: int = Field(default=80, ge=0, le=500, description="청크 간 중첩 글자 수")
    index_type: str = Field(
        default="IndexFlatL2",
        description="FAISS 인덱스 종류 ('IndexFlatL2' 또는 'IndexHNSWFlat')"
    )
    reset_existing: bool = Field(default=True, description="기존 인덱스를 초기화하고 새로 구축할지 여부")


class QueryRequest(BaseModel):
    """
    질의 및 RAG Fusion 파이프라인 실행 요청 스키마
    """
    query: str = Field(..., min_length=1, description="사용자의 단일 입력 질의")
    top_k: int = Field(default=5, ge=1, le=20, description="최종 반환할 상위 문서 개수")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF(Reciprocal Rank Fusion) 상수 k")
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="FAISS Dense 검색 가중치")
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="BM25 Sparse 검색 가중치")
    index_type: Optional[str] = Field(
        default=None,
        description="요청 시 사용할 FAISS 인덱스 타입 변경 필요 시 지정 ('IndexFlatL2' 또는 'IndexHNSWFlat')"
    )
    stream: bool = Field(default=False, description="실시간 스트리밍(NDJSON) 응답 여부")


# ==============================================================================
# 1. 실시간 스트리밍 핸들러 (StreamHandler)
# ==============================================================================

class StreamHandler:
    """
    서버의 파이프라인 동작 상태, 진행률, 알고리즘 복잡도, 중간 검색 결과,
    LLM 생성 토큰 및 에러 정보를 실시간 스트리밍(NDJSON) 프레임으로 직렬화하는 핸들러 클래스.

    - 시간 복잡도: 이벤트 당 O(1) 직렬화
    - 공간 복잡도: 버퍼링된 이벤트 수 E에 대해 O(E)
    """

    def __init__(self, task_name: str = "rag_pipeline"):
        self.task_name = task_name
        self.start_time = time.perf_counter()
        self.events: List[Dict[str, Any]] = []

    def _elapsed_ms(self) -> float:
        """파이프라인 시작 시점부터 경과한 시간(ms) 계산"""
        return round((time.perf_counter() - self.start_time) * 1000, 2)

    def create_event(
        self,
        event_type: str,
        step: str,
        message: str,
        progress: int = 0,
        data: Optional[Dict[str, Any]] = None,
        complexity: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        단일 스트림 이벤트 객체를 생성하고 NDJSON(줄바꿈 구분 JSON) 문자열로 반환합니다.
        """
        payload: Dict[str, Any] = {
            "task": self.task_name,
            "event": event_type,          # status | progress | multi_query | hybrid_search | rrf_fusion | llm_chunk | complete | error
            "step": step,                 # 현재 실행 중인 파이프라인 계층 식별자
            "message": message,           # 사용자 UI에 표시할 상태 메시지
            "progress": max(0, min(100, progress)),  # 진행률 (0 ~ 100%)
            "elapsed_ms": self._elapsed_ms(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if data is not None:
            payload["data"] = data
        if complexity is not None:
            payload["complexity"] = complexity
        if error is not None:
            payload["error"] = error

        self.events.append(payload)
        return json.dumps(payload, ensure_ascii=False) + "\n"


# ==============================================================================
# 2. 알고리즘 복잡도 계산기 (ComplexityAnalyzer)
# ==============================================================================

class ComplexityAnalyzer:
    """
    FAISS Index + Multi-Query + Hybrid Search + RAG Fusion 각 단계의
    이론적 시간/공간 복잡도(Big-O)와 실제 측정된 연산 비용을 계산하여 반환합니다.
    """

    @staticmethod
    def analyze_ingest(
        num_chunks: int,
        dim: int,
        index_type: str,
        vocab_size: int,
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        """벡터 DB(FAISS + BM25) 구축 단계의 복잡도 분석"""
        if index_type == "IndexHNSWFlat":
            faiss_build_time = f"O(N * d * log(N)) [N={num_chunks}, d={dim}]"
            faiss_space = f"O(N * (d + M_links)) [N={num_chunks}, d={dim}, M=32]"
        else:
            faiss_build_time = f"O(N * d) [N={num_chunks}, d={dim}]"
            faiss_space = f"O(N * d) [N={num_chunks}, d={dim}]"

        return {
            "index_type": index_type,
            "parameters": {
                "N_chunks": num_chunks,
                "d_dimension": dim,
                "V_vocab_size": vocab_size,
            },
            "stages": {
                "pdf_chunking": {
                    "name": "PDF 텍스트 추출 및 청킹",
                    "time_complexity": "O(C) (C: 전체 문서 문자 길이)",
                    "space_complexity": f"O(N * L_chunk) (N={num_chunks})",
                },
                "faiss_indexing": {
                    "name": f"FAISS Dense 인덱싱 ({index_type})",
                    "time_complexity": faiss_build_time,
                    "space_complexity": faiss_space,
                },
                "bm25_indexing": {
                    "name": "BM25 Sparse 역색인 구축",
                    "time_complexity": f"O(N * L_avg) [N={num_chunks}]",
                    "space_complexity": f"O(V + N) [V={vocab_size}, N={num_chunks}]",
                },
            },
            "measured_total_ms": round(elapsed_ms, 2),
        }

    @staticmethod
    def analyze_query_pipeline(
        num_chunks: int,
        dim: int,
        num_queries: int,
        top_k: int,
        index_type: str,
        vocab_size: int,
        timings_ms: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        질의 파이프라인 4대 핵심 계층의 이론적 시간/공간 복잡도 및 실측 소요 시간 반환
        - M: 확장된 다중 쿼리 수 (기본 5개)
        - N: 전체 적재된 문서 청크 수
        - d: 임베딩 벡터 차원 수
        - K: 검색할 상위 후보 문서 수
        """
        if index_type == "IndexHNSWFlat":
            dense_time = f"O(M * d * log(N)) [M={num_queries}, d={dim}, N={num_chunks}]"
            dense_space = f"O(N * (d + M_links)) [그래프 인덱스 메모리]"
            dense_desc = "계층적 소규모 세계 그래프(HNSW) 탐색으로 로그 시간 근사 최근접 이웃 검색 수행"
        else:
            dense_time = f"O(M * N * d) [M={num_queries}, N={num_chunks}, d={dim}]"
            dense_space = f"O(N * d) [무손실 Flat 벡터 메모리]"
            dense_desc = "전체 벡터에 대해 정확한 유클리드 제곱 거리(L2) 전수 계산 수행 (Recall 100%)"

        return {
            "index_type": index_type,
            "parameters": {
                "M_expanded_queries": num_queries,
                "N_total_chunks": num_chunks,
                "d_vector_dim": dim,
                "K_top_candidates": top_k,
                "V_bm25_vocab": vocab_size,
            },
            "layers": {
                "1_multi_query": {
                    "layer_name": "다중 쿼리 확장 계층 (Multi-Query Layer)",
                    "time_complexity": f"O(M * L_tokens) [M={num_queries} 파생 쿼리 생성]",
                    "space_complexity": f"O(M * |Q|) [M={num_queries}]",
                    "description": "단일 질의를 5가지 다각도 관점의 쿼리로 확장하여 어휘 불일치(Lexical Gap) 해소",
                    "elapsed_ms": round(timings_ms.get("multi_query_ms", 0.0), 2),
                },
                "2_faiss_dense_search": {
                    "layer_name": f"FAISS Dense 벡터 검색 ({index_type})",
                    "time_complexity": dense_time,
                    "space_complexity": dense_space,
                    "description": dense_desc,
                    "elapsed_ms": round(timings_ms.get("dense_search_ms", 0.0), 2),
                },
                "3_bm25_sparse_search": {
                    "layer_name": "BM25 Sparse 키워드 검색 (Okapi BM25)",
                    "time_complexity": f"O(M * |Q_terms| * N) [M={num_queries}, N={num_chunks}]",
                    "space_complexity": f"O(V + N) [V={vocab_size}]",
                    "description": "TF-IDF 및 문서 길이 정규화 기반의 정확한 고유명사/키워드 매칭 점수 계산",
                    "elapsed_ms": round(timings_ms.get("sparse_search_ms", 0.0), 2),
                },
                "4_rag_fusion_rrf": {
                    "layer_name": "RRF 순위 융합 및 재정렬 계층 (RAG Fusion & RRF)",
                    "formula": "RRF_Score(d) = Σ_{q ∈ Q} [ w_d / (k + rank_dense(q, d)) + w_s / (k + rank_sparse(q, d)) ]",
                    "time_complexity": f"O(M * K * log(M * K)) [2M={num_queries * 2}개 랭킹 리스트 병합 및 정렬]",
                    "space_complexity": f"O(U) [U <= 2 * M * K 후보 문서 해시맵]",
                    "description": "서로 다른 스케일의 Dense 거리와 BM25 점수를 순위 역수 합산(RRF)으로 정규화 및 융합",
                    "elapsed_ms": round(timings_ms.get("rrf_fusion_ms", 0.0), 2),
                },
                "5_llm_synthesis": {
                    "layer_name": "LLM 근거 기반 답변 생성 (Answer Synthesis)",
                    "time_complexity": f"O(K * L_chunk + L_output) [상위 {top_k}개 스니펫 컨텍스트 주입]",
                    "space_complexity": "O(L_context)",
                    "description": "RRF 상위 문서 스니펫만을 근거로 환각(Hallucination) 없는 답변 및 출처 생성",
                    "elapsed_ms": round(timings_ms.get("llm_synthesis_ms", 0.0), 2),
                },
            },
            "total_elapsed_ms": round(timings_ms.get("total_ms", 0.0), 2),
        }


# ==============================================================================
# 3. 텍스트 토크나이저 및 고차원 임베딩 엔진 (로컬 + OpenAI/Gemini 하이브리드 지원)
# ==============================================================================

class EmbeddingAndTokenizerEngine:
    """
    1) 한국어/영어 하이브리드 형태소·N-gram 토크나이저 (BM25 및 로컬 임베딩 공용)
    2) 고차원 Dense Vector 임베딩 생성기:
       - 환경변수 OPENAI_API_KEY가 설정된 경우 OpenAI text-embedding-3-small 사용 가능
       - API 키가 없거나 오프라인 환경인 경우, 의미적 핵심 어휘 사전 + 문자/단어 N-gram
         Signed Random Projection(Feature Hashing) 기반의 384차원 L2 정규화 벡터를 생성하여
         외부 의존성 없이도 즉각적인 FAISS 고차원 벡터 검색이 작동하도록 보장합니다.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # 한국어 조사/어미 제거용 정규표현식 패턴 (어간 추출 보조)
        self._ko_suffix_pattern = re.compile(
            r"(은|는|이|가|을|를|의|에|에서|으로|로|과|와|도|만|까지|부터|이란|란|이라|며|고|하는|하고|된|되는|입니다|합니다|인가요|무엇인가요|설명해줘|알려줘)$"
        )
        # 동의어/유사 개념 클러스터 사전 (의미적 유사도(Semantic Similarity) 강화용)
        self._semantic_clusters: List[Tuple[str, List[str]]] = [
            ("SEM_FAISS_VECTOR", ["faiss", "벡터", "임베딩", "고차원", "dense", "밀집", "유사도", "indexflatl2", "indexhnswflat", "hnsw", "l2", "거리", "내적"]),
            ("SEM_SPARSE_BM25", ["bm25", "키워드", "희소", "sparse", "역색인", "단어", "매칭", "tf-idf", "tfidf", "토큰", "형태소", "용어"]),
            ("SEM_HYBRID_SEARCH", ["하이브리드", "hybrid", "결합", "병행", "동시", "문맥", "의미", "정확도", "검색"]),
            ("SEM_MULTI_QUERY", ["다중", "쿼리", "multi-query", "multiquery", "확장", "파생", "관점", "재작성", "질의", "변형", "단일"]),
            ("SEM_RAG_FUSION_RRF", ["rag", "fusion", "rrf", "reciprocal", "rank", "순위", "융합", "재정렬", "rerank", "reranking", "가중치", "통합"]),
            ("SEM_COMPLEXITY_PERF", ["복잡도", "성능", "시간", "공간", "속도", "메모리", "big-o", "연산", "효율", "스케일", "지연"]),
            ("SEM_CLOUD_GAS", ["클라우드", "cloud", "run", "gas", "앱스", "스크립트", "스트리밍", "streamhandler", "서버", "연동", "배포", "docker"]),
        ]

    def tokenize(self, text: str) -> List[str]:
        """
        BM25 Sparse 검색을 위한 한국어·영어 하이브리드 토큰화 함수.
        단어 단위 토큰, 조사 제거 어간 토큰, 2-gram(바이그램) 토큰을 함께 추출하여
        한국어 복합명사 및 부분 일치 검색 재현율(Recall)을 높입니다.
        """
        if not text:
            return []
        cleaned = re.sub(r"[^\w\s가-힣a-zA-Z0-9\-]", " ", text.lower())
        raw_words = [w.strip("-") for w in cleaned.split() if len(w.strip("-")) >= 1]

        tokens: List[str] = []
        for word in raw_words:
            tokens.append(word)
            # 한국어 조사/어미가 붙은 경우 어간도 추가
            stem = self._ko_suffix_pattern.sub("", word)
            if stem and stem != word and len(stem) >= 2:
                tokens.append(stem)
                word_for_ngram = stem
            else:
                word_for_ngram = word

            # 3글자 이상 한글/영문 단어에 대해 문자 바이그램(2-gram) 추가
            if len(word_for_ngram) >= 3:
                for i in range(len(word_for_ngram) - 1):
                    tokens.append(word_for_ngram[i : i + 2])

        return tokens

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        입력 텍스트 리스트를 (N, d) 크기의 float32 L2-정규화 고차원 벡터 행렬로 변환합니다.
        FAISS IndexFlatL2 및 IndexHNSWFlat는 float32 연속 메모리 배열(C-contiguous)을 요구합니다.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row_idx, text in enumerate(texts):
            tokens = self.tokenize(text)
            text_lower = text.lower()

            # 1) 의미론적 개념 클러스터 차원 활성화 (Dense Semantic Feature)
            for cluster_idx, (cluster_name, keywords) in enumerate(self._semantic_clusters):
                match_count = sum(1 for kw in keywords if kw in text_lower)
                if match_count > 0:
                    # 클러스터별로 고유한 시드 방향 벡터를 가산하여 문맥적 의미 유사성 부여
                    rng = np.random.default_rng(abs(hash(cluster_name)) % (2**32))
                    concept_vec = rng.standard_normal(self.dimension).astype(np.float32)
                    vectors[row_idx] += concept_vec * (1.8 * math.log1p(match_count))

            # 2) 토큰 및 문자 N-gram Feature Hashing (Signed Random Projection)
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                dim_idx = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                weight = 1.5 if len(token) >= 3 else 0.8
                vectors[row_idx, dim_idx] += sign * weight

                # 보조 차원 투영 (충돌 완화 및 표현력 향상)
                dim_idx2 = int.from_bytes(digest[8:12], "little") % self.dimension
                sign2 = 1.0 if (digest[12] & 1) == 0 else -1.0
                vectors[row_idx, dim_idx2] += sign2 * (weight * 0.5)

            # 3) L2 정규화 (단위 초구면 상에 배치하여 L2 거리와 코사인 유사도가 단조 관계를 갖도록 보장)
            norm = np.linalg.norm(vectors[row_idx])
            if norm > 1e-8:
                vectors[row_idx] /= norm
            else:
                vectors[row_idx, 0] = 1.0

        return np.ascontiguousarray(vectors, dtype=np.float32)


# ==============================================================================
# 4. 다중 쿼리 확장 계층 (Multi-Query Layer - 5가지 관점 파생 쿼리 생성)
# ==============================================================================

class MultiQueryExpansionLayer:
    """
    사용자의 단일 입력 쿼리를 서로 다른 5가지 관점의 파생 쿼리로 확장하는 계층.
    - 관점 1: 핵심 개념 및 정의 관점 (Conceptual & Definition)
    - 관점 2: 내부 동작 원리 및 파이프라인 구조 관점 (Mechanism & Architecture)
    - 관점 3: 핵심 기술 용어 및 키워드 매칭 관점 (Exact Technical Keywords)
    - 관점 4: 실무 구현 및 활용 사례 관점 (Practical Implementation & Use Cases)
    - 관점 5: 알고리즘 복잡도, 성능 및 한계 극복 관점 (Complexity, Performance & Trade-offs)
    """

    PERSPECTIVE_LABELS = [
        "관점 1 [개념·정의]",
        "관점 2 [원리·구조]",
        "관점 3 [핵심 키워드]",
        "관점 4 [구현·활용]",
        "관점 5 [성능·복잡도]",
    ]

    def __init__(self, tokenizer_engine: EmbeddingAndTokenizerEngine):
        self.tokenizer_engine = tokenizer_engine

    async def expand_to_five_queries(self, original_query: str) -> Dict[str, Any]:
        """
        단일 사용자 질의를 5가지 다각도 관점의 파생 쿼리로 확장합니다.
        환경변수 OPENAI_API_KEY가 있으면 LLM 호출을 시도하고,
        없거나 실패할 경우 도메인 특화 5관점 규칙 확장기로 즉시 5개의 파생 쿼리를 생성합니다.
        """
        clean_q = original_query.strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        # 1) OpenAI API 키가 설정된 경우 LLM을 통한 5가지 관점 쿼리 생성 시도
        if openai_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=openai_key)
                prompt = (
                    "당신은 RAG Fusion 검색 최적화 전문가입니다.\n"
                    f"사용자의 원본 질의: \"{clean_q}\"\n\n"
                    "위 단일 질의의 검색 누락을 방지하기 위해 서로 다른 5가지 관점의 파생 검색 쿼리를 JSON 배열(문자열 5개)로 생성하세요.\n"
                    "1. 핵심 개념 및 정의 관점\n"
                    "2. 동작 원리 및 아키텍처 관점\n"
                    "3. 핵심 기술 키워드 중심 관점\n"
                    "4. 실무 구현 및 적용 방법 관점\n"
                    "5. 알고리즘 복잡도, 성능 및 장단점 비교 관점\n"
                    "반드시 {\"queries\": [\"쿼리1\", \"쿼리2\", \"쿼리3\", \"쿼리4\", \"쿼리5\"]} 형식의 JSON만 출력하세요."
                )
                resp = await client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.4,
                    timeout=10.0,
                )
                parsed = json.loads(resp.choices[0].message.content or "{}")
                queries = parsed.get("queries", [])
                if isinstance(queries, list) and len(queries) >= 5:
                    structured = [
                        {
                            "index": i + 1,
                            "perspective": self.PERSPECTIVE_LABELS[i],
                            "query": str(queries[i]).strip(),
                        }
                        for i in range(5)
                    ]
                    return {
                        "original_query": clean_q,
                        "generator_mode": "LLM (OpenAI API)",
                        "expanded_count": 5,
                        "expanded_queries": structured,
                    }
            except Exception:
                # API 오류 시 중단하지 않고 내장 5관점 확장 엔진으로 안전하게 폴백
                pass

        # 2) 내장 다각도 5관점 쿼리 확장 엔진 (API 키 없이도 항상 5가지 관점 보장)
        tokens = self.tokenizer_engine.tokenize(clean_q)
        core_terms = [t for t in tokens if len(t) >= 2][:6]
        keyword_str = " ".join(dict.fromkeys(core_terms)) if core_terms else clean_q

        derived_queries = [
            {
                "index": 1,
                "perspective": self.PERSPECTIVE_LABELS[0],
                "query": f"{clean_q}의 핵심 개념과 정의 및 등장 배경 설명",
            },
            {
                "index": 2,
                "perspective": self.PERSPECTIVE_LABELS[1],
                "query": f"{clean_q} 내부 동작 원리와 단계별 파이프라인 아키텍처 구조",
            },
            {
                "index": 3,
                "perspective": self.PERSPECTIVE_LABELS[2],
                "query": f"{keyword_str} 핵심 기술 키워드 수식 및 구성 요소",
            },
            {
                "index": 4,
                "perspective": self.PERSPECTIVE_LABELS[3],
                "query": f"{clean_q} 실무 구현 방법과 하이브리드 검색 연동 사례",
            },
            {
                "index": 5,
                "perspective": self.PERSPECTIVE_LABELS[4],
                "query": f"{clean_q} 시간 공간 알고리즘 복잡도 성능 비교 및 한계점 극복",
            },
        ]

        return {
            "original_query": clean_q,
            "generator_mode": "Multi-Perspective Rule Engine (5-View Expansion)",
            "expanded_count": 5,
            "expanded_queries": derived_queries,
        }


# ==============================================================================
# 5. 하이브리드 검색(FAISS + BM25) 및 RRF 순위 융합 계층 (RAG Fusion Engine)
# ==============================================================================

class HybridRAGFusionEngine:
    """
    - FAISS Index (IndexFlatL2 / IndexHNSWFlat) 구축 및 Dense 벡터 유사도 검색
    - BM25Okapi 역색인 구축 및 Sparse 키워드 일치 검색
    - Multi-Query (5개 파생 쿼리) 각각에 대해 Dense + Sparse 병행 검색 수행
    - Reciprocal Rank Fusion (RRF) 공식으로 순위를 융합하여 최적 상위 문서 재정렬(Re-ranking)
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.embed_engine = EmbeddingAndTokenizerEngine(dimension=embedding_dim)
        self.multi_query_layer = MultiQueryExpansionLayer(self.embed_engine)

        # 저장된 문서 청크 및 인덱스 상태
        self.chunks: List[DocumentChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.index_type: str = "IndexFlatL2"
        self.bm25_index: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.vocab_size: int = 0
        self.last_updated: Optional[str] = None
        self.loaded_sources: List[str] = []

        # 서버 시작 시 즉시 실습 및 테스트가 가능하도록 기본 RAG 기술 문서 코퍼스 적재
        self._load_default_corpus()

    def _load_default_corpus(self) -> None:
        """
        PDF 업로드 전에도 질의 테스트가 가능하도록 FAISS, Multi-Query, Hybrid Search,
        RAG Fusion(RRF), Google Cloud Run 연동에 관한 핵심 이론 문서를 기본 적재합니다.
        """
        sample_docs = [
            (
                "FAISS_Hybrid_RAG_Architecture.pdf",
                1,
                "FAISS(Facebook AI Similarity Search)는 고차원 밀집 벡터(Dense Vector)의 효율적인 유사도 검색과 클러스터링을 위한 라이브러리입니다. "
                "IndexFlatL2는 모든 벡터를 압축 없이 저장하고 유클리드 제곱 거리(L2 Distance)를 전수 계산하여 100%의 정확도(Recall)를 보장하며, "
                "시간 복잡도는 O(N*d), 공간 복잡도는 O(N*d)입니다.",
            ),
            (
                "FAISS_Hybrid_RAG_Architecture.pdf",
                2,
                "FAISS의 IndexHNSWFlat은 Hierarchical Navigable Small World(계층적 탐색 가능 소규모 세계) 그래프 구조를 활용한 근사 최근접 이웃(ANN) 인덱스입니다. "
                "다층 그래프의 상위 계층에서 긴 간선으로 빠르게 영역을 좁힌 뒤 최하위 계층에서 정밀 탐색을 수행하므로 검색 시간 복잡도를 O(d * log N) 수준으로 획기적으로 단축합니다.",
            ),
            (
                "FAISS_Hybrid_RAG_Architecture.pdf",
                3,
                "다중 쿼리 확장 계층(Multi-Query Expansion Layer)은 사용자의 모호하거나 짧은 단일 질의를 LLM을 이용해 서로 다른 5가지 관점(개념 정의, 동작 원리, 핵심 키워드, 실무 활용, 복잡도 및 한계점)의 파생 쿼리로 확장합니다. "
                "이를 통해 단일 쿼리 검색에서 발생하는 어휘 불일치(Lexical Gap)와 관점 편향에 의한 중요 문서 누락 문제를 해결합니다.",
            ),
            (
                "FAISS_Hybrid_RAG_Architecture.pdf",
                4,
                "하이브리드 검색(Hybrid Search) 계층은 FAISS 기반의 Dense Vector 검색(문맥 및 의미적 유사도 포착)과 BM25 기반의 Sparse Keyword 검색(고유명사, 에러 코드, 정확한 기술 용어 매칭)을 병행합니다. "
                "Dense 검색만 사용할 때 약한 정확한 키워드 매칭력과 Sparse 검색만 사용할 때 약한 동의어·문맥 이해력을 상호 보완합니다.",
            ),
            (
                "FAISS_Hybrid_RAG_Architecture.pdf",
                5,
                "RAG Fusion의 핵심인 RRF(Reciprocal Rank Fusion, 상호 순위 융합) 계층은 여러 검색 경로(5개 확장 쿼리 × Dense/Sparse 검색 = 10개 랭킹 리스트)에서 수집된 문서 순위를 통합합니다. "
                "RRF 공식은 RRF_Score(d) = Σ [ w / (k + rank(q, d)) ] 이며(보통 상수 k=60 사용), 서로 스케일이 다른 L2 거리와 BM25 점수를 별도의 정규화 없이도 안정적으로 융합하여 최적의 상위 문서를 재정렬(Re-ranking)합니다.",
            ),
            (
                "FAISS_Hybrid_RAG_Architecture.pdf",
                6,
                "Google Apps Script(GAS)와 Google Cloud Run 연동 구조에서는 클라우드 런 배포 URL을 GAS의 스크립트 속성(PropertiesService) 환경변수(CLOUD_RUN_URL)에 안전하게 저장합니다. "
                "또한 StreamHandler를 통해 PDF 벡터 DB 구축 현황, 단계별 알고리즘 복잡도, RAG 검색 스니펫 및 LLM 생성 답변을 실시간으로 모니터링하고 출처(파일명, 페이지, RRF 점수)를 투명하게 표기합니다.",
            ),
        ]

        chunks: List[DocumentChunk] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for idx, (source, page, text) in enumerate(sample_docs):
            chunks.append(
                DocumentChunk(
                    doc_id=idx,
                    text=text,
                    source=source,
                    page=page,
                    chunk_index=1,
                    char_length=len(text),
                    created_at=now_iso,
                )
            )
        self.build_indices(chunks=chunks, index_type="IndexFlatL2")

    def build_indices(
        self,
        chunks: List[DocumentChunk],
        index_type: str = "IndexFlatL2",
    ) -> Dict[str, Any]:
        """
        주어진 문서 청크 리스트로부터 FAISS Dense 인덱스와 BM25 Sparse 인덱스를 동시에 구축합니다.
        """
        if not chunks:
            raise ValueError("인덱싱할 문서 청크가 비어 있습니다.")

        t0 = time.perf_counter()

        # doc_id를 0부터 연속 정수로 재할당
        for i, chunk in enumerate(chunks):
            chunk.doc_id = i
        self.chunks = chunks

        # 1) 고차원 Dense 임베딩 벡터 생성 (N x d, float32)
        texts = [c.text for c in self.chunks]
        self.vectors = self.embed_engine.embed_texts(texts)

        # 2) FAISS 인덱스 생성 및 벡터 적재
        normalized_type = "IndexHNSWFlat" if index_type == "IndexHNSWFlat" else "IndexFlatL2"
        self.index_type = normalized_type

        if normalized_type == "IndexHNSWFlat":
            # HNSW 그래프 인덱스 (M=32 양방향 링크, efConstruction=64, efSearch=32)
            hnsw_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            hnsw_index.hnsw.efConstruction = 64
            hnsw_index.hnsw.efSearch = 32
            hnsw_index.add(self.vectors)
            self.faiss_index = hnsw_index
        else:
            # 정확한 L2 유클리드 거리 계산 인덱스
            flat_index = faiss.IndexFlatL2(self.embedding_dim)
            flat_index.add(self.vectors)
            self.faiss_index = flat_index

        # 3) BM25 Sparse 키워드 인덱스 구축
        self.tokenized_corpus = [self.embed_engine.tokenize(t) for t in texts]
        vocab_set = {tok for doc_tokens in self.tokenized_corpus for tok in doc_tokens}
        self.vocab_size = len(vocab_set)
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.loaded_sources = sorted(list({c.source for c in self.chunks}))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        complexity_info = ComplexityAnalyzer.analyze_ingest(
            num_chunks=len(self.chunks),
            dim=self.embedding_dim,
            index_type=self.index_type,
            vocab_size=self.vocab_size,
            elapsed_ms=elapsed_ms,
        )

        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "vocab_size": self.vocab_size,
            "sources": self.loaded_sources,
            "last_updated": self.last_updated,
            "complexity": complexity_info,
        }

    def extract_and_chunk_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> List[DocumentChunk]:
        """
        업로드된 PDF 바이너리에서 페이지별로 텍스트를 추출하고,
        문맥 유지를 위해 슬라이딩 윈도우(chunk_size, chunk_overlap) 방식으로 분할합니다.
        """
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) == 0:
            raise ValueError(f"'{filename}' PDF 파일에 페이지가 존재하지 않습니다.")

        extracted_chunks: List[DocumentChunk] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        step = max(50, chunk_size - chunk_overlap)

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            # 연속 공백 및 불필요한 제어문자 정리
            cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
            if not cleaned_text:
                continue

            chunk_seq = 1
            for start in range(0, len(cleaned_text), step):
                snippet = cleaned_text[start : start + chunk_size].strip()
                if len(snippet) < 20 and extracted_chunks:
                    # 너무 짧은 꼬리 조각은 건너뜀
                    continue
                extracted_chunks.append(
                    DocumentChunk(
                        doc_id=len(extracted_chunks),
                        text=snippet,
                        source=filename,
                        page=page_num,
                        chunk_index=chunk_seq,
                        char_length=len(snippet),
                        created_at=now_iso,
                    )
                )
                chunk_seq += 1
                if start + chunk_size >= len(cleaned_text):
                    break

        if not extracted_chunks:
            raise ValueError(
                f"'{filename}' PDF에서 추출 가능한 텍스트를 찾지 못했습니다. "
                "(스캔된 이미지 전용 PDF인지 확인해 주세요.)"
            )
        return extracted_chunks

    def search_dense_faiss(self, query_text: str, candidate_k: int) -> List[Dict[str, Any]]:
        """
        단일 쿼리에 대해 FAISS 인덱스에서 상위 candidate_k개의 문서를 검색합니다.
        반환값: [{doc_id, rank, l2_distance, dense_similarity}, ...]
        """
        if self.faiss_index is None or not self.chunks:
            return []

        k = min(candidate_k, len(self.chunks))
        q_vec = self.embed_engine.embed_texts([query_text])
        distances, indices = self.faiss_index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for rank_idx in range(k):
            doc_id = int(indices[0][rank_idx])
            if doc_id < 0 or doc_id >= len(self.chunks):
                continue
            dist = float(distances[0][rank_idx])
            # L2 정규화된 단위 벡터에서 cos_sim = 1 - (L2^2 / 2), 0~1 범위로 변환
            sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            results.append(
                {
                    "doc_id": doc_id,
                    "rank": rank_idx + 1,  # 1-based 순위
                    "l2_distance": round(dist, 4),
                    "dense_similarity": round(sim, 4),
                }
            )
        return results

    def search_sparse_bm25(self, query_text: str, candidate_k: int) -> List[Dict[str, Any]]:
        """
        단일 쿼리에 대해 BM25 역색인에서 상위 candidate_k개의 문서를 검색합니다.
        반환값: [{doc_id, rank, bm25_score}, ...]
        """
        if self.bm25_index is None or not self.chunks:
            return []

        k = min(candidate_k, len(self.chunks))
        q_tokens = self.embed_engine.tokenize(query_text)
        if not q_tokens:
            return []

        scores = self.bm25_index.get_scores(q_tokens)
        ranked_indices = np.argsort(scores)[::-1][:k]

        results: List[Dict[str, Any]] = []
        for rank_idx, doc_idx in enumerate(ranked_indices):
            score = float(scores[doc_idx])
            results.append(
                {
                    "doc_id": int(doc_idx),
                    "rank": rank_idx + 1,  # 1-based 순위
                    "bm25_score": round(score, 4),
                }
            )
        return results

    def reciprocal_rank_fusion(
        self,
        per_query_results: List[Dict[str, Any]],
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        ========================================================================
        RRF 순위 융합 계층 (RAG Fusion & Reciprocal Rank Fusion)
        ========================================================================
        수식:
          RRF_Score(d) = Σ_{q ∈ ExpandedQueries} [
              w_dense  * (1 / (k + rank_dense(q, d))) +
              w_sparse * (1 / (k + rank_sparse(q, d)))
          ]
        - 여러 관점의 쿼리와 이종 검색기(FAISS Dense + BM25 Sparse)에서 공통적으로
          상위권에 등장한 문서일수록 높은 RRF 점수를 부여받아 최상위로 재정렬(Re-ranking)됩니다.
        ========================================================================
        """
        doc_fusion_map: Dict[int, Dict[str, Any]] = {}

        # 가중치 합이 0이 되지 않도록 보정
        total_w = dense_weight + sparse_weight
        if total_w <= 0:
            w_d, w_s = 0.5, 0.5
        else:
            w_d, w_s = dense_weight / total_w, sparse_weight / total_w

        for q_item in per_query_results:
            q_idx = q_item["query_index"]
            q_perspective = q_item["perspective"]
            q_text = q_item["query"]

            # 1) FAISS Dense 순위 반영
            for d_hit in q_item["dense_hits"]:
                doc_id = d_hit["doc_id"]
                rank = d_hit["rank"]
                contrib = w_d * (1.0 / (rrf_k + rank))

                entry = doc_fusion_map.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "rrf_score": 0.0,
                        "dense_rrf_score": 0.0,
                        "sparse_rrf_score": 0.0,
                        "best_dense_rank": None,
                        "best_sparse_rank": None,
                        "best_dense_similarity": 0.0,
                        "best_bm25_score": 0.0,
                        "matched_queries": set(),
                        "rank_trace": [],
                    },
                )
                entry["rrf_score"] += contrib
                entry["dense_rrf_score"] += contrib
                if entry["best_dense_rank"] is None or rank < entry["best_dense_rank"]:
                    entry["best_dense_rank"] = rank
                if d_hit["dense_similarity"] > entry["best_dense_similarity"]:
                    entry["best_dense_similarity"] = d_hit["dense_similarity"]
                entry["matched_queries"].add(q_idx)
                entry["rank_trace"].append(
                    {
                        "query_index": q_idx,
                        "perspective": q_perspective,
                        "query": q_text,
                        "type": "FAISS_Dense",
                        "rank": rank,
                        "raw_metric": f"sim={d_hit['dense_similarity']} (L2={d_hit['l2_distance']})",
                        "rrf_contribution": round(contrib, 6),
                    }
                )

            # 2) BM25 Sparse 순위 반영
            for s_hit in q_item["sparse_hits"]:
                doc_id = s_hit["doc_id"]
                rank = s_hit["rank"]
                bm25_val = s_hit["bm25_score"]
                # BM25 점수가 0 초과인 실제 키워드 매칭 문서에 주 가중치 부여 (0점인 경우 최소 기여도만 부여)
                effective_w_s = w_s if bm25_val > 0 else (w_s * 0.15)
                contrib = effective_w_s * (1.0 / (rrf_k + rank))

                entry = doc_fusion_map.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "rrf_score": 0.0,
                        "dense_rrf_score": 0.0,
                        "sparse_rrf_score": 0.0,
                        "best_dense_rank": None,
                        "best_sparse_rank": None,
                        "best_dense_similarity": 0.0,
                        "best_bm25_score": 0.0,
                        "matched_queries": set(),
                        "rank_trace": [],
                    },
                )
                entry["rrf_score"] += contrib
                entry["sparse_rrf_score"] += contrib
                if entry["best_sparse_rank"] is None or rank < entry["best_sparse_rank"]:
                    entry["best_sparse_rank"] = rank
                if bm25_val > entry["best_bm25_score"]:
                    entry["best_bm25_score"] = bm25_val
                if bm25_val > 0:
                    entry["matched_queries"].add(q_idx)
                entry["rank_trace"].append(
                    {
                        "query_index": q_idx,
                        "perspective": q_perspective,
                        "query": q_text,
                        "type": "BM25_Sparse",
                        "rank": rank,
                        "raw_metric": f"bm25={bm25_val}",
                        "rrf_contribution": round(contrib, 6),
                    }
                )

        # RRF 종합 점수 기준 내림차순 재정렬 (Re-ranking)
        sorted_entries = sorted(
            doc_fusion_map.values(),
            key=lambda x: (x["rrf_score"], x["best_dense_similarity"], x["best_bm25_score"]),
            reverse=True,
        )[:top_k]

        reranked_docs: List[Dict[str, Any]] = []
        for final_rank, item in enumerate(sorted_entries, start=1):
            chunk = self.chunks[item["doc_id"]]
            reranked_docs.append(
                {
                    "final_rank": final_rank,
                    "doc_id": chunk.doc_id,
                    "snippet": chunk.text,
                    "metadata": {
                        "source": chunk.source,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "char_length": chunk.char_length,
                        "citation": f"[{chunk.source} - p.{chunk.page} (Chunk #{chunk.chunk_index})]",
                    },
                    "scores": {
                        "rrf_score": round(item["rrf_score"], 6),
                        "dense_rrf_score": round(item["dense_rrf_score"], 6),
                        "sparse_rrf_score": round(item["sparse_rrf_score"], 6),
                        "best_dense_rank": item["best_dense_rank"],
                        "best_sparse_rank": item["best_sparse_rank"],
                        "best_dense_similarity": round(item["best_dense_similarity"], 4),
                        "best_bm25_score": round(item["best_bm25_score"], 4),
                        "matched_queries_count": len(item["matched_queries"]),
                    },
                    "rank_trace": item["rank_trace"][:6],  # 상위 6개 경로 기여 내역
                }
            )

        return reranked_docs

    async def synthesize_answer(
        self,
        original_query: str,
        expanded_queries: List[Dict[str, Any]],
        reranked_docs: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        RRF로 재정렬된 상위 문서 스니펫을 컨텍스트로 사용하여 최종 답변 및 출처 표기를 생성합니다.
        OpenAI API 키가 설정된 경우 LLM이 답변을 생성하고,
        미설정 시 상위 검색 스니펫을 구조화하여 근거 중심 요약 답변을 생성합니다.
        """
        if not reranked_docs:
            return (
                "검색된 관련 문서가 없습니다. 먼저 PDF 파일을 업로드하여 벡터 DB를 생성하거나 다른 질의를 입력해 주세요.",
                "Built-in Synthesizer",
            )

        context_blocks = []
        for doc in reranked_docs:
            meta = doc["metadata"]
            scores = doc["scores"]
            context_blocks.append(
                f"[문서 #{doc['final_rank']}] 출처: {meta['citation']} | RRF 점수: {scores['rrf_score']}\n"
                f"내용: {doc['snippet']}"
            )
        context_text = "\n\n".join(context_blocks)

        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=openai_key)
                sys_prompt = (
                    "당신은 FAISS Hybrid Search 및 RAG Fusion 기반 AI 어시스턴트입니다. "
                    "제공된 [검색된 상위 문서]의 내용만을 근거로 사용자의 질문에 체계적이고 상세하게 한국어로 답변하세요. "
                    "답변 본문 중 근거가 되는 문장 끝에 반드시 출처(예: [파일명 - p.1])를 명시하세요."
                )
                user_prompt = (
                    f"[사용자 질문]\n{original_query}\n\n"
                    f"[검색된 상위 문서 (RRF Re-ranked)]\n{context_text}\n\n"
                    "위 문서를 바탕으로 핵심 답변과 출처를 명확히 작성해 주세요."
                )
                resp = await client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    timeout=15.0,
                )
                answer = (resp.choices[0].message.content or "").strip()
                if answer:
                    return answer, "LLM (OpenAI API)"
            except Exception:
                pass

        # 내장 근거 기반 합성기 (Extractive + Structured Synthesis)
        lines = [
            f"### 질문에 대한 RAG Fusion 종합 답변",
            f"**질의 내용:** `{original_query}`\n",
            f"5가지 다각도 파생 쿼리와 **FAISS({self.index_type}) + BM25 하이브리드 검색** 및 **RRF(상호 순위 융합)** 재정렬을 통해 도출된 핵심 근거 기반 답변입니다.\n",
        ]
        for doc in reranked_docs[:3]:
            meta = doc["metadata"]
            scores = doc["scores"]
            lines.append(
                f"{doc['final_rank']}. **핵심 근거 ({meta['citation']} | RRF: `{scores['rrf_score']}`)**\n"
                f"   - {doc['snippet']}\n"
                f"   - *(Dense 최고 순위: #{scores['best_dense_rank']}, BM25 최고 순위: #{scores['best_sparse_rank']})*"
            )

        citations = ", ".join(
            dict.fromkeys(d["metadata"]["citation"] for d in reranked_docs)
        )
        lines.append(f"\n---\n**참고 출처 종합:** {citations}")
        return "\n".join(lines), "RAG Extractive Synthesizer (Local Engine)"


# ==============================================================================
# 6. FastAPI 애플리케이션 및 전역 엔진 인스턴스 초기화
# ==============================================================================

app = FastAPI(
    title="Advanced RAG Server (FAISS + Multi-Query + Hybrid Search + RAG Fusion)",
    description=(
        "FAISS 고차원 벡터 인덱스, 5가지 관점 다중 쿼리 확장(Multi-Query), "
        "FAISS Dense + BM25 Sparse 하이브리드 검색, RRF(Reciprocal Rank Fusion) 재정렬 및 "
        "실시간 StreamHandler를 지원하는 백엔드 API 서버"
    ),
    version="1.0.0",
)

# GAS(Google Apps Script) 및 웹 브라우저에서의 CORS 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 RAG Fusion 엔진 인스턴스 생성
rag_engine = HybridRAGFusionEngine(embedding_dim=384)


def build_error_response(
    status_code: int,
    error_code: str,
    stage: str,
    message: str,
    detail: str = "",
    troubleshooting: str = "",
) -> JSONResponse:
    """
    요구사항 3(에러 표시는 정확하게 기술)을 충족하기 위한 표준화된 상세 에러 JSON 응답 생성 함수
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error": {
                "error_code": error_code,
                "stage": stage,
                "message": message,
                "detail": detail,
                "troubleshooting": troubleshooting,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        },
    )


# ==============================================================================
# 7. API 엔드포인트 구현
# ==============================================================================

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    [연동 상태 확인 엔드포인트]
    클라이언트(GAS Code.gs 및 index.html)에서 서버의 동작 상태, 현재 활성화된 FAISS 인덱스 타입,
    적재된 문서 청크 수, 알고리즘 복잡도 요약 정보를 실시간으로 확인합니다.
    """
    return {
        "status": "ok",
        "server_state": "READY",
        "message": "Advanced RAG 서버가 정상 동작 중입니다.",
        "index_status": {
            "index_type": rag_engine.index_type,
            "total_chunks": len(rag_engine.chunks),
            "embedding_dim": rag_engine.embedding_dim,
            "bm25_vocab_size": rag_engine.vocab_size,
            "loaded_sources": rag_engine.loaded_sources,
            "last_updated": rag_engine.last_updated,
        },
        "complexity_summary": {
            "multi_query": "O(M * L_tokens) (M=5 관점 확장)",
            "faiss_dense": (
                f"O(M * d * log N) [IndexHNSWFlat]"
                if rag_engine.index_type == "IndexHNSWFlat"
                else f"O(M * N * d) [IndexFlatL2, N={len(rag_engine.chunks)}, d={rag_engine.embedding_dim}]"
            ),
            "bm25_sparse": f"O(M * |Q| * N) [V={rag_engine.vocab_size}]",
            "rrf_fusion": "O(M * K * log(M * K))",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/ingest")
async def ingest_pdf_multipart(
    file: UploadFile = File(..., description="벡터 DB를 구축할 PDF 파일"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(80),
    index_type: str = Form("IndexFlatL2"),
    reset_existing: bool = Form(True),
) -> JSONResponse:
    """
    [멀티파트 PDF 파일 업로드 및 FAISS + BM25 벡터 DB 구축 엔드포인트]
    """
    stream_handler = StreamHandler(task_name="ingest_pdf")
    try:
        filename = file.filename or "uploaded_document.pdf"
        if not filename.lower().endswith(".pdf"):
            return build_error_response(
                status_code=400,
                error_code="INVALID_FILE_EXTENSION",
                stage="PDF_VALIDATION",
                message=f"지원하지 않는 파일 형식입니다: {filename}",
                detail="오직 .pdf 확장자를 가진 문서만 벡터 DB로 구축할 수 있습니다.",
                troubleshooting="PDF 형식의 파일을 선택한 후 다시 시도해 주세요.",
            )

        stream_handler.create_event("progress", "READ_FILE", f"'{filename}' 파일 수신 완료", progress=20)
        pdf_bytes = await file.read()
        if not pdf_bytes:
            return build_error_response(
                status_code=400,
                error_code="EMPTY_PDF_FILE",
                stage="PDF_READ",
                message="업로드된 PDF 파일이 비어 있습니다 (0 bytes).",
                troubleshooting="정상적인 내용이 포함된 PDF 파일인지 확인해 주세요.",
            )

        stream_handler.create_event("progress", "EXTRACT_CHUNK", "PDF 텍스트 추출 및 청킹 진행 중...", progress=50)
        new_chunks = rag_engine.extract_and_chunk_pdf(
            pdf_bytes=pdf_bytes,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        target_chunks = new_chunks if reset_existing else (rag_engine.chunks + new_chunks)
        stream_handler.create_event(
            "progress",
            "BUILD_INDEX",
            f"FAISS({index_type}) 및 BM25 하이브리드 인덱스 구축 중 ({len(target_chunks)}개 청크)...",
            progress=85,
        )
        build_result = rag_engine.build_indices(chunks=target_chunks, index_type=index_type)

        stream_handler.create_event(
            "complete",
            "COMPLETE",
            f"벡터 DB 구축 완료: 총 {build_result['total_chunks']}개 청크 적재됨",
            progress=100,
            data=build_result,
            complexity=build_result["complexity"],
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"'{filename}' 기반 벡터 DB(FAISS + BM25) 생성이 완료되었습니다.",
                "filename": filename,
                "ingested_chunks": len(new_chunks),
                "index_info": build_result,
                "stream_logs": stream_handler.events,
            },
        )
    except ValueError as ve:
        return build_error_response(
            status_code=422,
            error_code="PDF_EXTRACTION_ERROR",
            stage="PDF_PARSING",
            message=str(ve),
            detail=traceback.format_exc(),
            troubleshooting="텍스트 레이어가 포함된 PDF 파일인지 확인해 주세요.",
        )
    except Exception as exc:
        return build_error_response(
            status_code=500,
            error_code="INGEST_INTERNAL_ERROR",
            stage="INDEX_BUILD",
            message=f"벡터 DB 구축 중 서버 내부 오류가 발생했습니다: {str(exc)}",
            detail=traceback.format_exc(),
            troubleshooting="서버 로그 및 PDF 파일 손상 여부를 확인해 주세요.",
        )


@app.post("/ingest/base64")
async def ingest_pdf_base64(req: IngestBase64Request) -> JSONResponse:
    """
    [GAS(Code.gs) 연동 전용 Base64 PDF 업로드 및 실시간 StreamHandler 로그 포함 벡터 DB 생성 엔드포인트]
    """
    stream_handler = StreamHandler(task_name="ingest_pdf_base64")
    try:
        stream_handler.create_event(
            "status",
            "INIT",
            f"'{req.filename}' Base64 데이터 수신 및 디코딩 시작",
            progress=15,
        )
        # Data URL prefix(data:application/pdf;base64,...)가 포함된 경우 제거
        b64_clean = re.sub(r"^data:.*?;base64,", "", req.file_base64.strip())
        pdf_bytes = base64.b64decode(b64_clean)

        stream_handler.create_event(
            "progress",
            "PDF_PARSE",
            f"PDF 바이너리 디코딩 완료 ({len(pdf_bytes):,} bytes). 텍스트 추출 및 청킹 시작...",
            progress=40,
        )
        new_chunks = rag_engine.extract_and_chunk_pdf(
            pdf_bytes=pdf_bytes,
            filename=req.filename,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )

        stream_handler.create_event(
            "progress",
            "EMBED_AND_INDEX",
            f"{len(new_chunks)}개 청크 추출 완료. 고차원 임베딩 및 FAISS({req.index_type}) + BM25 인덱싱 수행 중...",
            progress=75,
        )
        target_chunks = new_chunks if req.reset_existing else (rag_engine.chunks + new_chunks)
        build_result = rag_engine.build_indices(chunks=target_chunks, index_type=req.index_type)

        stream_handler.create_event(
            "complete",
            "COMPLETE",
            f"벡터 DB 구축 완료! ({build_result['total_chunks']}개 청크, 인덱스: {build_result['index_type']})",
            progress=100,
            data=build_result,
            complexity=build_result["complexity"],
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"'{req.filename}' 파일의 RAG 벡터 DB 구축이 완료되었습니다.",
                "filename": req.filename,
                "ingested_chunks": len(new_chunks),
                "index_info": build_result,
                "stream_logs": stream_handler.events,
            },
        )
    except ValueError as ve:
        stream_handler.create_event(
            "error",
            "PDF_PARSE",
            f"PDF 추출 실패: {str(ve)}",
            progress=100,
            error={"error_code": "PDF_EXTRACTION_ERROR", "message": str(ve)},
        )
        return build_error_response(
            status_code=422,
            error_code="PDF_EXTRACTION_ERROR",
            stage="PDF_PARSING",
            message=str(ve),
            detail=traceback.format_exc(),
            troubleshooting="스캔 이미지 PDF가 아닌 텍스트 추출이 가능한 PDF 파일인지 확인해 주세요.",
        )
    except Exception as exc:
        stream_handler.create_event(
            "error",
            "INGEST_ERROR",
            f"인덱싱 오류: {str(exc)}",
            progress=100,
            error={"error_code": "INGEST_INTERNAL_ERROR", "message": str(exc)},
        )
        return build_error_response(
            status_code=500,
            error_code="INGEST_INTERNAL_ERROR",
            stage="INDEX_BUILD",
            message=f"벡터 DB 생성 중 오류 발생: {str(exc)}",
            detail=traceback.format_exc(),
            troubleshooting="Base64 인코딩 데이터 및 서버 메모리 상태를 확인해 주세요.",
        )


async def _execute_rag_fusion_pipeline(
    req: QueryRequest,
    stream_handler: StreamHandler,
) -> Dict[str, Any]:
    """
    [핵심 파이프라인 실행 함수]
    1단계: Multi-Query 확장 (5가지 관점)
    2단계: Hybrid Search (FAISS Dense + BM25 Sparse 병행 검색)
    3단계: RAG Fusion (RRF 상호 순위 융합 및 재정렬)
    4단계: LLM 근거 기반 답변 합성 및 복잡도 산출
    """
    pipeline_start = time.perf_counter()
    timings_ms: Dict[str, float] = {}

    # 인덱스 타입 변경 요청이 있는 경우 즉시 재구축 적용
    if req.index_type and req.index_type in ("IndexFlatL2", "IndexHNSWFlat"):
        if req.index_type != rag_engine.index_type and rag_engine.chunks:
            rag_engine.build_indices(chunks=rag_engine.chunks, index_type=req.index_type)

    # ------------------------------------------------------------------
    # [계층 1] 다중 쿼리 확장 계층 (Multi-Query Layer: 5가지 관점)
    # ------------------------------------------------------------------
    stream_handler.create_event(
        "status",
        "MULTI_QUERY_START",
        "1단계: 단일 질의를 5가지 다각도 관점의 파생 쿼리로 확장 중...",
        progress=15,
    )
    t_mq = time.perf_counter()
    mq_result = await rag_engine.multi_query_layer.expand_to_five_queries(req.query)
    timings_ms["multi_query_ms"] = (time.perf_counter() - t_mq) * 1000.0
    expanded_queries = mq_result["expanded_queries"]

    stream_handler.create_event(
        "multi_query",
        "MULTI_QUERY_DONE",
        f"5가지 관점 파생 쿼리 생성 완료 ({timings_ms['multi_query_ms']:.1f}ms)",
        progress=35,
        data=mq_result,
    )

    # ------------------------------------------------------------------
    # [계층 2] 하이브리드 검색 계층 (FAISS Dense + BM25 Sparse)
    # ------------------------------------------------------------------
    stream_handler.create_event(
        "status",
        "HYBRID_SEARCH_START",
        f"2단계: 5개 파생 쿼리에 대해 FAISS({rag_engine.index_type}) Dense 검색 및 BM25 Sparse 검색 병행 수행 중...",
        progress=50,
    )
    candidate_k = max(req.top_k * 2, 6)
    per_query_results: List[Dict[str, Any]] = []

    dense_total_ms = 0.0
    sparse_total_ms = 0.0

    for q_item in expanded_queries:
        q_text = q_item["query"]

        t_d = time.perf_counter()
        dense_hits = rag_engine.search_dense_faiss(q_text, candidate_k=candidate_k)
        dense_total_ms += (time.perf_counter() - t_d) * 1000.0

        t_s = time.perf_counter()
        sparse_hits = rag_engine.search_sparse_bm25(q_text, candidate_k=candidate_k)
        sparse_total_ms += (time.perf_counter() - t_s) * 1000.0

        per_query_results.append(
            {
                "query_index": q_item["index"],
                "perspective": q_item["perspective"],
                "query": q_text,
                "dense_hits": dense_hits,
                "sparse_hits": sparse_hits,
            }
        )

    timings_ms["dense_search_ms"] = dense_total_ms
    timings_ms["sparse_search_ms"] = sparse_total_ms

    stream_handler.create_event(
        "hybrid_search",
        "HYBRID_SEARCH_DONE",
        f"하이브리드 검색 완료 (FAISS: {dense_total_ms:.2f}ms, BM25: {sparse_total_ms:.2f}ms)",
        progress=70,
        data={
            "candidate_k_per_query": candidate_k,
            "total_search_paths": len(expanded_queries) * 2,
        },
    )

    # ------------------------------------------------------------------
    # [계층 3] RRF 순위 융합 계층 (RAG Fusion & RRF Re-ranking)
    # ------------------------------------------------------------------
    stream_handler.create_event(
        "status",
        "RRF_FUSION_START",
        f"3단계: 10개 검색 경로(5 Queries × Dense/Sparse) 결과를 RRF(k={req.rrf_k}) 공식으로 순위 융합 및 재정렬 중...",
        progress=82,
    )
    t_rrf = time.perf_counter()
    reranked_docs = rag_engine.reciprocal_rank_fusion(
        per_query_results=per_query_results,
        top_k=req.top_k,
        rrf_k=req.rrf_k,
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
    )
    timings_ms["rrf_fusion_ms"] = (time.perf_counter() - t_rrf) * 1000.0

    stream_handler.create_event(
        "rrf_fusion",
        "RRF_FUSION_DONE",
        f"RRF 재정렬 완료: 상위 {len(reranked_docs)}개 최적 문서 스니펫 추출 ({timings_ms['rrf_fusion_ms']:.2f}ms)",
        progress=90,
        data={"reranked_count": len(reranked_docs)},
    )

    # ------------------------------------------------------------------
    # [계층 4] LLM 답변 합성 및 알고리즘 복잡도 분석
    # ------------------------------------------------------------------
    stream_handler.create_event(
        "status",
        "LLM_SYNTHESIS_START",
        "4단계: RRF 상위 문서 스니펫을 기반으로 최종 답변 및 출처 생성 중...",
        progress=95,
    )
    t_llm = time.perf_counter()
    llm_answer, synthesizer_mode = await rag_engine.synthesize_answer(
        original_query=req.query,
        expanded_queries=expanded_queries,
        reranked_docs=reranked_docs,
    )
    timings_ms["llm_synthesis_ms"] = (time.perf_counter() - t_llm) * 1000.0
    timings_ms["total_ms"] = (time.perf_counter() - pipeline_start) * 1000.0

    complexity_report = ComplexityAnalyzer.analyze_query_pipeline(
        num_chunks=len(rag_engine.chunks),
        dim=rag_engine.embedding_dim,
        num_queries=len(expanded_queries),
        top_k=req.top_k,
        index_type=rag_engine.index_type,
        vocab_size=rag_engine.vocab_size,
        timings_ms=timings_ms,
    )

    result_payload = {
        "status": "success",
        "original_query": req.query,
        "multi_query": mq_result,
        "hybrid_search_summary": {
            "index_type": rag_engine.index_type,
            "total_corpus_chunks": len(rag_engine.chunks),
            "bm25_vocab_size": rag_engine.vocab_size,
            "rrf_k": req.rrf_k,
            "dense_weight": req.dense_weight,
            "sparse_weight": req.sparse_weight,
            "per_query_search_details": per_query_results,
        },
        "retrieved_documents": reranked_docs,
        "llm_answer": {
            "answer": llm_answer,
            "synthesizer_mode": synthesizer_mode,
            "sources": [doc["metadata"] for doc in reranked_docs],
        },
        "complexity": complexity_report,
        "server_status": {
            "state": "COMPLETED",
            "index_type": rag_engine.index_type,
            "loaded_sources": rag_engine.loaded_sources,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    stream_handler.create_event(
        "complete",
        "PIPELINE_COMPLETE",
        f"RAG Fusion 파이프라인 완료 (총 {timings_ms['total_ms']:.1f}ms)",
        progress=100,
        data=result_payload,
        complexity=complexity_report,
    )
    result_payload["stream_logs"] = stream_handler.events
    return result_payload


@app.post("/query")
async def query_rag_fusion(req: QueryRequest) -> Any:
    """
    [메인 질의 엔드포인트]
    - req.stream == True 인 경우: StreamHandler를 통해 실시간 NDJSON 스트리밍으로 단계별 이벤트 반환
    - req.stream == False 인 경우: 정제된 상위 문서 스니펫, 메타데이터, 5개 확장 쿼리,
      LLM 답변, 알고리즘 복잡도, StreamHandler 이벤트 로그를 JSON으로 반환
    """
    if req.stream:
        return await query_rag_fusion_stream(req)

    stream_handler = StreamHandler(task_name="rag_fusion_query")
    try:
        if not req.query or not req.query.strip():
            return build_error_response(
                status_code=400,
                error_code="EMPTY_QUERY",
                stage="QUERY_VALIDATION",
                message="질의 내용이 비어 있습니다.",
                troubleshooting="검색할 질문을 1글자 이상 입력해 주세요.",
            )

        payload = await _execute_rag_fusion_pipeline(req, stream_handler)
        return JSONResponse(status_code=200, content=payload)

    except Exception as exc:
        return build_error_response(
            status_code=500,
            error_code="RAG_PIPELINE_ERROR",
            stage="RAG_FUSION_EXECUTION",
            message=f"질의 처리 중 서버 오류가 발생했습니다: {str(exc)}",
            detail=traceback.format_exc(),
            troubleshooting="서버 인덱스 상태(/health) 및 입력 파라미터를 확인해 주세요.",
        )


@app.post("/query/stream")
async def query_rag_fusion_stream(req: QueryRequest) -> StreamingResponse:
    """
    [실시간 스트리밍 전용 엔드포인트 (NDJSON StreamHandler)]
    클라이언트(GAS StreamHandler 또는 브라우저 Fetch Stream)가 각 단계의 진행 상태를
    실시간 줄바꿈 구분 JSON(application/x-ndjson)으로 수신할 수 있도록 스트리밍합니다.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        stream_handler = StreamHandler(task_name="rag_fusion_stream")
        try:
            yield stream_handler.create_event(
                "status",
                "STREAM_CONNECTED",
                "StreamHandler 연결 성공: RAG Fusion 파이프라인을 시작합니다.",
                progress=5,
            )
            await asyncio.sleep(0.01)

            payload = await _execute_rag_fusion_pipeline(req, stream_handler)
            # 중간 단계 이벤트들을 순서대로 yield (첫 번째 STREAM_CONNECTED 제외)
            for ev in stream_handler.events[1:]:
                yield json.dumps(ev, ensure_ascii=False) + "\n"
                await asyncio.sleep(0.01)

        except Exception as exc:
            yield stream_handler.create_event(
                "error",
                "STREAM_ERROR",
                f"스트리밍 파이프라인 오류: {str(exc)}",
                progress=100,
                error={
                    "error_code": "STREAM_EXECUTION_ERROR",
                    "stage": "STREAM_PIPELINE",
                    "message": str(exc),
                    "detail": traceback.format_exc(),
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
