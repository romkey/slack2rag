"""VectorStore tests against in-memory Qdrant."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from slack2rag.vector_store import (
    CollectionMismatchError,
    Point,
    VectorStore,
    count_in_collection,
    source_filter,
    text_matches_in,
)


def _point(vec: list[float], **payload: object) -> Point:
    return Point(
        str(uuid.uuid4()), vec, {"indices": [1], "values": [1.0]}, dict(payload)
    )


@pytest.fixture
def vs(qdrant: QdrantClient) -> VectorStore:
    return VectorStore("", "hs_member", 3, "model-a", client=qdrant)


def test_records_model_and_refuses_another(
    vs: VectorStore, qdrant: QdrantClient
) -> None:
    info = qdrant.get_collection("hs_member")
    assert info.config.metadata == {
        "embedding_model": "model-a",
        "embedding_dimension": 3,
        "tier": "member",
    }
    VectorStore("", "hs_member", 3, "model-a", client=qdrant)  # reopening is fine
    with pytest.raises(CollectionMismatchError, match="re-embed"):
        VectorStore("", "hs_member", 3, "model-b", client=qdrant)
    with pytest.raises(CollectionMismatchError):
        VectorStore("", "hs_member", 4, "model-a", client=qdrant)


def test_refuses_old_unnamed_collection(qdrant: QdrantClient) -> None:
    qdrant.create_collection(
        "old",
        vectors_config=qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE),
    )
    with pytest.raises(CollectionMismatchError, match="older version"):
        VectorStore("", "old", 3, "m", client=qdrant)


def test_refuses_public_collection(qdrant: QdrantClient) -> None:
    with pytest.raises(ValueError, match="member-tier"):
        VectorStore("", "hs_public", 3, "m", client=qdrant)


def test_write_read_delete(vs: VectorStore, qdrant: QdrantClient) -> None:
    a = _point([1, 0, 0], source="slack", sync_run_id=1, channel_id="C1", text="alpha")
    b = _point([0, 1, 0], source="wiki", sync_run_id=1, text="beta")
    c = _point(
        [0.9, 0.1, 0], source="slack", sync_run_id=5, channel_id="C2", text="gamma"
    )
    vs.upsert([a, b, c])
    vs.upsert([])

    hits = vs.search([1, 0, 0], {"indices": [1], "values": [1.0]}, limit=3)
    assert a.id in {h["id"] for h in hits[:2]}
    assert vs.similar([1, 0, 0], source_filter("wiki"))[0]["id"] == b.id
    assert vs.existing_ids([a.id, str(uuid.uuid4())]) == {a.id}
    assert vs.existing_ids([]) == set()

    vs.set_payload([a.id], {"superseded_by": c.id})
    vs.set_payload([], {"x": 1})
    assert (
        vs.similar([1, 0, 0], source_filter("slack", channel_id="C1"))[0][
            "superseded_by"
        ]
        == c.id
    )

    vs.delete_older_runs("slack", 2)
    assert vs.count() == 2
    vs.delete_ids([b.id])
    vs.delete_ids([])
    vs.delete_where(source_filter("slack", channel_id="C2"))
    assert vs.count() == 0


def test_helpers_for_other_collections(vs: VectorStore, qdrant: QdrantClient) -> None:
    vs.upsert([_point([1, 0, 0], source="slack", text="Has CANARY-1 inside")])
    assert count_in_collection(qdrant, "hs_member", source_filter("slack")) == 1
    assert count_in_collection(qdrant, "missing") == 0
    assert len(text_matches_in(qdrant, "hs_member", "canary-1")) == 1
    assert text_matches_in(qdrant, "hs_member", "absent") == []
    assert text_matches_in(qdrant, "missing", "x") == []
