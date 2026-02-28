FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default local embedding model so container startup is fast.
# Skipped if EMBEDDING_PROVIDER=openai; model is cached in a named volume.
ARG PRELOAD_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${PRELOAD_MODEL}')" \
    || echo "Model pre-download skipped"

COPY src/ ./src/

CMD ["python", "-m", "src.main"]
