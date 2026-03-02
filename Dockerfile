# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1

# curl is used by the Qdrant healthcheck in docker-compose
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY VERSION .
COPY src/ ./src/

ARG APP_VERSION
LABEL org.opencontainers.image.version="${APP_VERSION}"

CMD ["python", "-m", "src.main"]
