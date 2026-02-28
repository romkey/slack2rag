#!/bin/sh
set -e

# Download the embedding model on first run if it isn't already cached
# in the volume. Runs once, then subsequent starts are instant.
if [ "$EMBEDDING_PROVIDER" != "openai" ]; then
    MODEL="${LOCAL_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
    if ! python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL}', cache_folder='${HF_HOME}')" 2>/dev/null; then
        echo "Downloading embedding model: ${MODEL}…"
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL}', cache_folder='${HF_HOME}')"
    fi
fi

exec "$@"
