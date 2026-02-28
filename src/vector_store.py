"""
Qdrant vector store wrapper.

The collection schema stores the full message text and rich metadata as the
point payload, so downstream applications can retrieve context without
needing a separate data store.

Collection payload fields
-------------------------
  channel_id   : str   — Slack channel ID (e.g. "C04ABCDEF")
  channel_name : str   — human-readable name (e.g. "general")
  ts           : str   — Slack message timestamp
  date         : str   — ISO-8601 date  (e.g. "2024-01-25")
  user_id      : str
  user_name    : str
  thread_ts    : str | null
  reply_count  : int
  text         : str   — full clean text that was embedded
  permalink    : str
"""

from __future__ import annotations

import logging
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tenacity import retry, stop_after_attempt, wait_exponential

from .processor import Document

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, url: str, collection: str, dimension: int) -> None:
        self._client = QdrantClient(url=url, timeout=30)
        self._collection = collection
        self._dimension = dimension
        self._ensure_collection()

    # ── collection management ─────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            logger.debug("Collection %r already exists", self._collection)
            return

        logger.info(
            "Creating Qdrant collection %r (dim=%d, cosine distance)",
            self._collection,
            self._dimension,
        )
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qmodels.VectorParams(
                size=self._dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )
        # Payload indices for efficient filtering
        for field_name in ("channel_id", "channel_name", "date", "user_id"):
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    # ── write ─────────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upsert(self, documents: List[Document], vectors: List[List[float]]) -> None:
        """Upsert *documents* with their corresponding *vectors*."""
        if not documents:
            return

        points = [
            qmodels.PointStruct(
                id=doc.id,
                vector=vec,
                payload=doc.payload(),
            )
            for doc, vec in zip(documents, vectors)
        ]

        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        logger.debug("Upserted %d points", len(points))

    # ── read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        channel_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[dict]:
        """
        Semantic search.  Returns a list of payload dicts sorted by relevance.

        Optional filters:
          channel_filter — channel name or ID
          date_from / date_to — ISO-8601 dates (inclusive)
        """
        conditions: List[qmodels.Condition] = []

        if channel_filter:
            conditions.append(
                qmodels.FieldCondition(
                    key="channel_id" if channel_filter.startswith("C") else "channel_name",
                    match=qmodels.MatchValue(value=channel_filter),
                )
            )
        if date_from:
            conditions.append(
                qmodels.FieldCondition(key="date", range=qmodels.Range(gte=date_from))
            )
        if date_to:
            conditions.append(
                qmodels.FieldCondition(key="date", range=qmodels.Range(lte=date_to))
            )

        query_filter = qmodels.Filter(must=conditions) if conditions else None

        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {"score": hit.score, **hit.payload}
            for hit in hits
        ]

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
