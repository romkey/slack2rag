FROM python:3.12-slim

WORKDIR /app

ENV HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install torch from the PyTorch CPU-only index before everything else.
# --index-url (not --extra-index-url) restricts this step to that index so pip
# can never resolve a CUDA wheel here.  The remaining packages come from PyPI.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the default local embedding model so container startup is fast.
# Skipped if EMBEDDING_PROVIDER=openai; model is cached in a named volume.
ARG PRELOAD_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${PRELOAD_MODEL}')" \
    || echo "Model pre-download skipped"

COPY src/ ./src/

CMD ["python", "-m", "src.main"]
