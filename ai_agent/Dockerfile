    # ==============================================================================
    # Google Cloud Run 배포용 Dockerfile
    # FastAPI + FAISS Index + Multi-Query + Hybrid Search + RAG Fusion 서버
    # ==============================================================================

    # 1. 경량 파이썬 3.11 슬림 이미지를 베이스로 사용
    FROM python:3.11-slim

    # 2. 파이썬 출력 버퍼링 해제 (실시간 StreamHandler 로그 출력 보장) 및 바이트코드 생성 방지
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PORT=8080

    # 3. FAISS 구동을 위한 시스템 필수 공유 라이브러리(libgomp1) 설치
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

    # 4. 작업 디렉토리 설정
    WORKDIR /app

    # 5. 의존성 파일 복사 및 패키지 설치
    COPY requirements.txt /app/requirements.txt
    RUN pip install --no-cache-dir --upgrade pip && \
        pip install --no-cache-dir -r /app/requirements.txt

    # 6. 서버 소스코드 복사
    COPY main.py /app/main.py

    # 7. Google Cloud Run 기본 포트 노출
    EXPOSE 8080

    # 8. Uvicorn ASGI 서버 실행 (Cloud Run의 $PORT 환경변수 바인딩)
    CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

