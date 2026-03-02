"""
Qdrant vector store wrapper.

The collection schema stores the full message text and rich metadata as the
point payload, so downstream applications can retrieve context without
needing a separate data store.

Collection payload fields
-------------------------
  channel_id     : str        — Slack channel ID (e.g. "C04ABCDEF")
  channel_name   : str        — human-readable name (e.g. "general")
  ts             : str        — Slack message timestamp
  date           : str        — ISO-8601 date  (e.g. "2024-01-25")
  datetime       : str        — ISO-8601 datetime (e.g. "2024-01-25T14:30:00Z")
  user_id        : str
  user_name      : str
  thread_ts      : str | null
  reply_count    : int
  text           : str        — full clean text that was embedded
  permalink      : str        — clickable Slack link
  channel_topic  : str        — channel topic or purpose
  reaction_count : int        — total emoji reactions
  reactions      : list[str]  — reaction names
  attachments    : list[str]  — filenames of attached files
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tenacity import retry, stop_after_attempt, wait_exponential

from .processor import Document

logger = logging.getLogger(__name__)

_CHANNEL_ID_RE = re.compile(r"^C[A-Z0-9]{8,}$")


class VectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        dimension: int,
        *,
        hybrid: bool = False,
    ) -> None:
        self._client = QdrantClient(url=url, timeout=30)
        self._collection = collection
        self._dimension = dimension
        self._hybrid = hybrid
        self._ensure_collection()

    # ── collection management ─────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            self._check_dimension()
            self._ensure_payload_indexes()
            logger.debug("Collection %r already exists", self._collection)
            return

        if self._hybrid:
            self._create_hybrid_collection()
        else:
            self._create_simple_collection()

        self._ensure_payload_indexes()

    def _create_simple_collection(self) -> None:
        logger.info(
            "Creating Qdrant collection %r (dim=%d, cosine, simple mode)",
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

    def _create_hybrid_collection(self) -> None:
        logger.info(
            "Creating Qdrant collection %r (dim=%d, cosine, hybrid mode with sparse vectors)",
            self._collection,
            self._dimension,
        )
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": qmodels.VectorParams(
                    size=self._dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(),
            },
        )

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes if they don't already exist (idempotent)."""
        for field_name in ("channel_id", "channel_name", "date", "user_id", "doc_type"):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # already exists

        # Full-text index for keyword-based search/filtering
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="text",
                field_schema=qmodels.TextIndexParams(
                    type="text",
                    tokenizer=qmodels.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=40,
                    lowercase=True,
                ),
            )
        except Exception:
            pass  # already exists

        # Integer index on reaction_count for quality-weighted retrieval
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="reaction_count",
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass

    def _check_dimension(self) -> None:
        """Warn if the existing collection's vector size mismatches the embedder."""
        try:
            info = self._client.get_collection(self._collection)
            vec_cfg = info.config.params.vectors

            if isinstance(vec_cfg, dict):
                actual = vec_cfg.get("dense", vec_cfg.get(next(iter(vec_cfg)))).size
            else:
                actual = vec_cfg.size

            if actual != self._dimension:
                logger.error(
                    "DIMENSION MISMATCH: collection %r has vectors of size %d, "
                    "but the current embedding model produces size %d.  "
                    "Delete the collection and re-index, or switch back to the "
                    "original embedding model.",
                    self._collection,
                    actual,
                    self._dimension,
                )
                raise RuntimeError(
                    f"Vector dimension mismatch: collection={actual}, embedder={self._dimension}"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.debug("Could not verify collection dimension: %s", exc)

    # ── write ─────────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upsert(
        self,
        documents: List[Document],
        vectors: List[List[float]],
        sparse_vectors: Optional[List[dict]] = None,
    ) -> None:
        """Upsert *documents* with their corresponding *vectors*.

        When hybrid search is enabled, *sparse_vectors* must be provided
        as a parallel list of dicts with ``indices`` and ``values`` keys.
        """
        if not documents:
            return

        points: list[qmodels.PointStruct] = []

        for i, (doc, vec) in enumerate(zip(documents, vectors)):
            if self._hybrid and sparse_vectors:
                sv = sparse_vectors[i]
                point = qmodels.PointStruct(
                    id=doc.id,
                    vector={
                        "dense": vec,
                        "sparse": qmodels.SparseVector(
                            indices=sv["indices"],
                            values=sv["values"],
                        ),
                    },
                    payload=doc.payload(),
                )
            else:
                point = qmodels.PointStruct(
                    id=doc.id,
                    vector=vec,
                    payload=doc.payload(),
                )
            points.append(point)

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
        score_threshold: float = 0.0,
        sparse_vector: Optional[dict] = None,
    ) -> List[dict]:
        """
        Semantic search.  Returns a list of payload dicts sorted by relevance.

        Optional filters:
          channel_filter  — channel name or ID
          date_from/date_to — ISO-8601 dates (inclusive)
          score_threshold — minimum similarity score (0.0 disables)

        When *sparse_vector* is provided and the collection is in hybrid mode,
        dense and sparse results are fused via Reciprocal Rank Fusion.
        """
        query_filter = self._build_filter(channel_filter, date_from, date_to)

        if self._hybrid and sparse_vector:
            return self._hybrid_search(
                query_vector, sparse_vector, limit, query_filter, score_threshold,
            )

        return self._dense_search(query_vector, limit, query_filter, score_threshold)

    def _build_filter(
        self,
        channel_filter: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> Optional[qmodels.Filter]:
        conditions: List[qmodels.Condition] = []

        if channel_filter:
            key = "channel_id" if _CHANNEL_ID_RE.match(channel_filter) else "channel_name"
            conditions.append(
                qmodels.FieldCondition(
                    key=key,
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

        return qmodels.Filter(must=conditions) if conditions else None

    def _dense_search(
        self,
        query_vector: List[float],
        limit: int,
        query_filter: Optional[qmodels.Filter],
        score_threshold: float,
    ) -> List[dict]:
        kwargs: dict = {
            "collection_name": self._collection,
            "query": query_vector,
            "limit": limit,
            "query_filter": query_filter,
            "with_payload": True,
        }
        if score_threshold > 0:
            kwargs["score_threshold"] = score_threshold

        response = self._client.query_points(**kwargs)

        return [
            {"score": hit.score, **hit.payload}
            for hit in response.points
        ]

    def _hybrid_search(
        self,
        query_vector: List[float],
        sparse_vector: dict,
        limit: int,
        query_filter: Optional[qmodels.Filter],
        score_threshold: float,
    ) -> List[dict]:
        """Dense + sparse retrieval with Reciprocal Rank Fusion."""
        prefetch_limit = max(limit * 3, 20)

        sparse_qv = qmodels.SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"],
        )

        kwargs: dict = {
            "collection_name": self._collection,
            "prefetch": [
                qmodels.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                qmodels.Prefetch(
                    query=sparse_qv,
                    using="sparse",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            "query": qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            "limit": limit,
            "with_payload": True,
        }
        if score_threshold > 0:
            kwargs["score_threshold"] = score_threshold

        response = self._client.query_points(**kwargs)

        return [
            {"score": hit.score, **hit.payload}
            for hit in response.points
        ]

    def count(self) -> int:
        result = self._client.count(collection_name=self._collection)
        return getattr(result, "count", None) or getattr(result, "points_count", None) or 0

    def count_by_channel(self, channel_id: str) -> int:
        """Return the number of message points with the given channel_id.

        Excludes summary documents.  Points indexed before the doc_type
        field was added (with no doc_type at all) are counted as messages.
        """
        result = self._client.count(
            collection_name=self._collection,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="channel_id",
                        match=qmodels.MatchValue(value=channel_id),
                    ),
                ],
                must_not=[
                    qmodels.FieldCondition(
                        key="doc_type",
                        match=qmodels.MatchValue(value="channel_summary"),
                    ),
                    qmodels.FieldCondition(
                        key="doc_type",
                        match=qmodels.MatchValue(value="workspace_summary"),
                    ),
                    qmodels.FieldCondition(
                        key="doc_type",
                        match=qmodels.MatchValue(value="user_summary"),
                    ),
                    qmodels.FieldCondition(
                        key="doc_type",
                        match=qmodels.MatchValue(value="team_summary"),
                    ),
                ],
            ),
        )
        return getattr(result, "count", None) or 0
