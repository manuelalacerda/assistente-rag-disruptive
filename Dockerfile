FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# camada de dependências primeiro: só reinstala se requirements.txt mudar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data        # contém o index.db gerado por: python -m ingest.build_index

EXPOSE 8000
# $PORT é definido pela plataforma de deploy (Render, Cloud Run...)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
