"""
Qdrant collection for member-tier chunks (``hs_member`` by default).

Every point has a dense vector ("dense") and a sparse keyword vector
("sparse"); searches fuse both with reciprocal rank fusion. The embedding
model and dimension are recorded in the collection's metadata, and a
collection made with another model is refused: changing the model means a
full re-embed from staging.

Points follow the shared payload schema (docs/overview.md lists it); this
module only moves them. Payload indexes are created on source, tier,
asset_ids, location_ids, topics, updated_at, section and doc_id, plus
channel_id and kind for this source, and a full-text index on text.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import PUBLIC_COLLECTION
from .embedder import SparseVector

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"

_KEYWORD_FIELDS = (
    "source",
    "tier",
    "asset_ids",
    "location_ids",
    "topics",
    "section",
    "doc_id",
    "channel_id",
    "kind",
)

# A search hit: the point's payload with ``score`` and ``id`` added.
SearchHit = dict[str, Any]


class CollectionMismatchError(RuntimeError):
    """The collection was built with a different embedding model or size."""


@dataclass(frozen=True)
class Point:
    id: str
    dense: list[float]
    sparse: SparseVector
    payload: dict[str, Any]


def _match(key: str, value: Any) -> qmodels.FieldCondition:
    return qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))


def source_filter(source: str, **equals: Any) -> qmodels.Filter:
    return qmodels.Filter(
        must=[_match("source", source), *(_match(k, v) for k, v in equals.items())]
    )


class VectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        dimension: int,
        embedding_model: str,
        client: QdrantClient | None = None,
    ) -> None:
        if collection == PUBLIC_COLLECTION:
            raise ValueError(f"refusing to write member-tier points to {collection}")
        self._client = client or QdrantClient(url=url, timeout=30)
        self.collection = collection
        self._dimension = dimension
        self._model = embedding_model
        self._ensure_collection()

    @property
    def client(self) -> QdrantClient:
        return self._client

    # ── collection management ─────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection in existing:
            self._check_collection()
        else:
            logger.info(
                "Creating Qdrant collection %r (%s, dim=%d, dense + sparse)",
                self.collection,
                self._model,
                self._dimension,
            )
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE: qmodels.VectorParams(
                        size=self._dimension, distance=qmodels.Distance.COSINE
                    )
                },
                sparse_vectors_config={SPARSE: qmodels.SparseVectorParams()},
                metadata={
                    "embedding_model": self._model,
                    "embedding_dimension": self._dimension,
                    "tier": "member",
                },
            )
        self._ensure_payload_indexes()

    def _check_collection(self) -> None:
        info = self._client.get_collection(self.collection)
        vectors = info.config.params.vectors
        params = vectors.get(DENSE) if isinstance(vectors, dict) else None
        if params is None or not info.config.params.sparse_vectors:
            raise CollectionMismatchError(
                f"Collection {self.collection!r} has no named '{DENSE}' and"
                f" '{SPARSE}' vectors; it was made by an older version. Delete it"
                " and re-index from staging."
            )
        metadata = getattr(info.config, "metadata", None) or {}
        model = metadata.get("embedding_model")
        if params.size != self._dimension or (model and model != self._model):
            raise CollectionMismatchError(
                f"Collection {self.collection!r} holds {model or 'unknown model'}"
                f" vectors of size {params.size}, but the embedding model is"
                f" {self._model} (size {self._dimension}). Delete the collection"
                " and re-embed from staging."
            )

    def _ensure_payload_indexes(self) -> None:
        # Creating an index that exists is a no-op or an error, depending on
        # the server version; either way it's the expected case after a run.
        schemas: list[tuple[str, Any]] = [
            (name, qmodels.PayloadSchemaType.KEYWORD) for name in _KEYWORD_FIELDS
        ]
        schemas += [
            ("updated_at", qmodels.PayloadSchemaType.DATETIME),
            ("sync_run_id", qmodels.PayloadSchemaType.INTEGER),
            (
                "text",
                qmodels.TextIndexParams(
                    type=qmodels.TextIndexType.TEXT,
                    tokenizer=qmodels.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=40,
                    lowercase=True,
                ),
            ),
        ]
        for name, schema in schemas:
            with contextlib.suppress(Exception):
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=name,
                    field_schema=schema,
                )

    # ── write ─────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def upsert(self, points: Sequence[Point]) -> None:
        if not points:
            return
        self._client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=p.id,
                    vector={
                        DENSE: p.dense,
                        SPARSE: qmodels.SparseVector(
                            indices=p.sparse["indices"], values=p.sparse["values"]
                        ),
                    },
                    payload=p.payload,
                )
                for p in points
            ],
            wait=True,
        )

    def set_payload(self, ids: Sequence[str], payload: dict[str, Any]) -> None:
        if not ids:
            return
        self._client.set_payload(
            collection_name=self.collection,
            payload=payload,
            points=list(ids),
            wait=True,
        )

    def delete_ids(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=qmodels.PointIdsList(points=list(ids)),
            wait=True,
        )

    def delete_where(self, flt: qmodels.Filter) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(filter=flt),
            wait=True,
        )

    def delete_older_runs(self, source: str, run_id: int) -> None:
        """Delete *source* points last seen by a run before *run_id*."""
        self.delete_where(
            qmodels.Filter(
                must=[_match("source", source)],
                must_not=[
                    qmodels.FieldCondition(
                        key="sync_run_id", range=qmodels.Range(gte=run_id)
                    )
                ],
            )
        )

    # ── read ──────────────────────────────────────────────────────────────

    def count(self, flt: qmodels.Filter | None = None) -> int:
        result = self._client.count(
            collection_name=self.collection, count_filter=flt, exact=True
        )
        return int(result.count)

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        records = self._client.retrieve(
            collection_name=self.collection,
            ids=list(ids),
            with_payload=False,
            with_vectors=False,
        )
        return {str(r.id) for r in records}

    def similar(
        self,
        dense: list[float],
        flt: qmodels.Filter | None,
        limit: int = 5,
    ) -> list[SearchHit]:
        """Dense-only nearest neighbours, for duplicate and wiki matching."""
        response = self._client.query_points(
            collection_name=self.collection,
            query=dense,
            using=DENSE,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": str(p.id), "score": p.score, **(p.payload or {})}
            for p in response.points
        ]

    def search(
        self,
        dense: list[float],
        sparse: SparseVector,
        limit: int = 10,
        flt: qmodels.Filter | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchHit]:
        """Dense + sparse retrieval fused with reciprocal rank fusion."""
        prefetch = max(limit * 3, 20)
        kwargs: dict[str, Any] = {
            "collection_name": self.collection,
            "prefetch": [
                qmodels.Prefetch(query=dense, using=DENSE, limit=prefetch, filter=flt),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=sparse["indices"], values=sparse["values"]
                    ),
                    using=SPARSE,
                    limit=prefetch,
                    filter=flt,
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
            {"id": str(p.id), "score": p.score, **(p.payload or {})}
            for p in response.points
        ]


def count_in_collection(
    client: QdrantClient, collection: str, flt: qmodels.Filter | None = None
) -> int:
    """Points matching *flt* in any collection; 0 if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return 0
    return int(
        client.count(collection_name=collection, count_filter=flt, exact=True).count
    )


def text_matches_in(
    client: QdrantClient, collection: str, needle: str, limit: int = 20
) -> list[str]:
    """Like VectorStore.text_matches, for any collection (read-only)."""
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return []
    # Without a full-text index Qdrant can't match text; scan payloads instead.
    found: list[str] = []
    offset = None
    lowered = needle.lower()
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            if lowered in str(r.payload or {}).lower():
                found.append(str(r.id))
                if len(found) >= limit:
                    return found
        if offset is None:
            return found
