"""
Generates dense vector embeddings for text.

Two providers are supported:

  local  — sentence-transformers (offline, model must already be cached)
  openai — OpenAI embeddings API (requires OPENAI_API_KEY)

The embedding dimension is exposed via the `dimension` property so the
vector store can create a correctly-sized collection.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, provider: str, local_model: str, openai_api_key: str, openai_model: str) -> None:
        self._provider = provider
        self._dimension: int | None = None

        if provider == "openai":
            self._init_openai(openai_api_key, openai_model)
        else:
            self._init_local(local_model)

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_local(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        hf_home = os.environ.get("HF_HOME", "")
        logger.info("Loading local embedding model: %s", model_name)
        logger.info("  HF_HOME: %s", hf_home or "(not set)")

        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception as exc:
            logger.error("=" * 72)
            logger.error("FATAL: Could not load embedding model %r", model_name)
            logger.error("")
            logger.error("  The model must already be downloaded into the HF cache.")
            logger.error("  HF_HOME is currently: %s", hf_home or "(not set)")
            logger.error("")
            logger.error("  To download it, run:")
            logger.error("")
            logger.error(
                "    docker compose run --rm slack2rag python -c "
                "\"from sentence_transformers import SentenceTransformer; "
                "SentenceTransformer('%s')\"",
                model_name,
            )
            logger.error("")
            logger.error("  This saves the model into the model_cache Docker volume")
            logger.error("  so all subsequent runs can load it offline.")
            logger.error("")
            logger.error("  Underlying error: %s: %s", type(exc).__name__, exc)
            logger.error("=" * 72)
            sys.exit(1)

        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self._dimension)

    def _init_openai(self, api_key: str, model: str) -> None:
        if not api_key:
            logger.error("=" * 72)
            logger.error("FATAL: OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
            logger.error("")
            logger.error("  Set it in your .env file or environment.")
            logger.error("=" * 72)
            sys.exit(1)

        import openai  # type: ignore

        try:
            self._openai_client = openai.OpenAI(api_key=api_key)
            self._openai_model = model
            sample = self._openai_client.embeddings.create(input=["test"], model=model)
            self._dimension = len(sample.data[0].embedding)
        except Exception as exc:
            logger.error("=" * 72)
            logger.error("FATAL: Could not connect to OpenAI embeddings API")
            logger.error("")
            logger.error("  Model: %s", model)
            logger.error("  Verify OPENAI_API_KEY is valid and the model name is correct.")
            logger.error("")
            logger.error("  Underlying error: %s: %s", type(exc).__name__, exc)
            logger.error("=" * 72)
            sys.exit(1)

        logger.info("OpenAI embedding model: %s  dimension: %d", model, self._dimension)

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return a list of float vectors, one per input text."""
        if not texts:
            return []
        if self._provider == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts)

    # ── private ───────────────────────────────────────────────────────────────

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        # OpenAI limits ~8192 tokens per input; stay safe by batching 100 at a time
        results: List[List[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = self._openai_client.embeddings.create(input=batch, model=self._openai_model)
            results.extend([item.embedding for item in resp.data])
        return results
