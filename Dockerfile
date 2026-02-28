# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

ENV HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1 (heaviest, changes rarely): torch CPU ────────────────────────────
# Pinned in its own layer so bumping other deps never re-downloads ~300 MB.
# BuildKit cache mount keeps the pip wheel cache across builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch

# ── Layer 2 (medium, changes occasionally): remaining Python deps ────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ── Layer 3 (light, changes often): application code ─────────────────────────
COPY VERSION .
COPY src/ ./src/
COPY entrypoint.sh .

ARG APP_VERSION
LABEL org.opencontainers.image.version="${APP_VERSION}"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "-m", "src.main"]
