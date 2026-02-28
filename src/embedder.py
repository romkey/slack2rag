"""
Generates dense vector embeddings for text.

Two providers are supported:

  local  — sentence-transformers (offline, no API key needed)
  openai — OpenAI embeddings API (requires OPENAI_API_KEY)

The embedding dimension is exposed via the `dimension` property so the
vector store can create a correctly-sized collection.
"""

from __future__ import annotations

import logging
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
        logger.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self._dimension)

    def _init_openai(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        import openai  # type: ignore
        self._openai_client = openai.OpenAI(api_key=api_key)
        self._openai_model = model
        # Probe dimension with a dummy request
        sample = self._openai_client.embeddings.create(input=["test"], model=model)
        self._dimension = len(sample.data[0].embedding)
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
