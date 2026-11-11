"""VectorStore tests (QdrantClient is mocked; no network)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from qdrant_client.http import models as qmodels
from tenacity import wait_none

from slack2rag import vector_store
from slack2rag.processor import Document
from slack2rag.vector_store import VectorStore

KEYWORD_FIELDS = ("channel_id", "channel_name", "date", "user_id", "doc_type", "source_system")


def _collections(*names: str) -> SimpleNamespace:
    return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])


def _collection_info(vectors) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
    )


def _doc(i: int = 1, **kw) -> Document:
    base = dict(
        id=f"00000000-0000-0000-0000-00000000000{i}",
        text=f"hello {i}",
        channel_id="C0123456789",
        channel_name="general",
        ts=f"{i}.0",
        date="2024-01-25",
        datetime_str="2024-01-25T14:30:00Z",
        user_id="U1",
        user_name="alice",
        thread_ts=None,
        reply_count=0,
    )
    base.update(kw)
    return Document(**base)


@pytest.fixture
def qc(monkeypatch) -> MagicMock:
    """Patch QdrantClient; default: no existing collections."""
    instance = MagicMock(name="QdrantClient()")
    instance.get_collections.return_value = _collections()
    factory = MagicMock(return_value=instance)
    monkeypatch.setattr(vector_store, "QdrantClient", factory)
    instance._factory = factory
    return instance


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch) -> None:
    monkeypatch.setattr(VectorStore.upsert.retry, "wait", wait_none())


def _store(qc: MagicMock, hybrid: bool = False, dim: int = 4) -> VectorStore:
    store = VectorStore("http://qdrant:6333", "slack", dim, hybrid=hybrid)
    qc.reset_mock()
    return store


# ── collection management ─────────────────────────────────────────────────────


def test_init_creates_simple_collection(qc) -> None:
    VectorStore("http://qdrant:6333", "slack", 768)
    qc._factory.assert_called_once_with(url="http://qdrant:6333", timeout=30)
    kwargs = qc.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "slack"
    assert kwargs["vectors_config"] == qmodels.VectorParams(
        size=768, distance=qmodels.Distance.COSINE
    )
    assert "sparse_vectors_config" not in kwargs
    qc.get_collection.assert_not_called()


def test_init_creates_hybrid_collection(qc) -> None:
    VectorStore("u", "slack", 384, hybrid=True)
    kwargs = qc.create_collection.call_args.kwargs
    assert kwargs["vectors_config"] == {
        "dense": qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
    }
    assert kwargs["sparse_vectors_config"] == {"sparse": qmodels.SparseVectorParams()}


def test_payload_indexes_created(qc) -> None:
    VectorStore("u", "slack", 4)
    calls = qc.create_payload_index.call_args_list
    by_field = {c.kwargs["field_name"]: c.kwargs["field_schema"] for c in calls}
    for f in KEYWORD_FIELDS:
        assert by_field[f] == qmodels.PayloadSchemaType.KEYWORD
    assert by_field["reaction_count"] == qmodels.PayloadSchemaType.INTEGER
    text_schema = by_field["text"]
    assert isinstance(text_schema, qmodels.TextIndexParams)
    assert text_schema.tokenizer == qmodels.TokenizerType.WORD
    assert text_schema.lowercase is True
    assert len(calls) == len(KEYWORD_FIELDS) + 2
    assert all(c.kwargs["collection_name"] == "slack" for c in calls)


def test_payload_index_errors_are_swallowed(qc) -> None:
    qc.create_payload_index.side_effect = RuntimeError("already exists")
    VectorStore("u", "slack", 4)
    assert qc.create_payload_index.call_count == len(KEYWORD_FIELDS) + 2


def test_existing_collection_matching_dimension(qc) -> None:
    qc.get_collections.return_value = _collections("other", "slack")
    qc.get_collection.return_value = _collection_info(qmodels.VectorParams(
        size=4, distance=qmodels.Distance.COSINE))
    VectorStore("u", "slack", 4)
    qc.create_collection.assert_not_called()
    qc.get_collection.assert_called_once_with("slack")
    assert qc.create_payload_index.call_count == len(KEYWORD_FIELDS) + 2


def test_existing_collection_dimension_mismatch_raises(qc) -> None:
    qc.get_collections.return_value = _collections("slack")
    qc.get_collection.return_value = _collection_info(SimpleNamespace(size=1024))
    with pytest.raises(RuntimeError, match="collection=1024, embedder=768"):
        VectorStore("u", "slack", 768)


def test_existing_named_vectors_uses_dense(qc) -> None:
    qc.get_collections.return_value = _collections("slack")
    qc.get_collection.return_value = _collection_info(
        {"other": SimpleNamespace(size=99), "dense": SimpleNamespace(size=4)}
    )
    VectorStore("u", "slack", 4, hybrid=True)  # no raise


def test_existing_named_vectors_without_dense_uses_first(qc) -> None:
    qc.get_collections.return_value = _collections("slack")
    qc.get_collection.return_value = _collection_info({"vec": SimpleNamespace(size=8)})
    with pytest.raises(RuntimeError, match="collection=8"):
        VectorStore("u", "slack", 4)


def test_check_dimension_other_errors_ignored(qc) -> None:
    qc.get_collections.return_value = _collections("slack")
    qc.get_collection.side_effect = ConnectionError("boom")
    VectorStore("u", "slack", 4)
    qc.create_collection.assert_not_called()


# ── upsert ────────────────────────────────────────────────────────────────────


def test_upsert_empty_is_noop(qc) -> None:
    store = _store(qc)
    store.upsert([], [])
    qc.upsert.assert_not_called()


def test_upsert_simple_points(qc) -> None:
    store = _store(qc)
    docs = [_doc(1), _doc(2, reactions=["+1"], reaction_count=1)]
    store.upsert(docs, [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

    kwargs = qc.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "slack"
    assert kwargs["wait"] is True
    points = kwargs["points"]
    assert [p.id for p in points] == [d.id for d in docs]
    assert points[0].vector == [0.1, 0.2, 0.3, 0.4]
    assert points[1].payload == docs[1].payload()
    assert points[1].payload["reactions"] == ["+1"]


def test_upsert_hybrid_points_include_sparse(qc) -> None:
    store = _store(qc, hybrid=True)
    store.upsert(
        [_doc(1)],
        [[1.0, 0.0, 0.0, 0.0]],
        sparse_vectors=[{"indices": [3, 7], "values": [0.5, 0.25]}],
    )
    (point,) = qc.upsert.call_args.kwargs["points"]
    assert point.vector["dense"] == [1.0, 0.0, 0.0, 0.0]
    assert point.vector["sparse"] == qmodels.SparseVector(indices=[3, 7], values=[0.5, 0.25])


def test_upsert_hybrid_without_sparse_falls_back_to_plain_vector(qc) -> None:
    store = _store(qc, hybrid=True)
    store.upsert([_doc(1)], [[1.0, 0.0, 0.0, 0.0]])
    (point,) = qc.upsert.call_args.kwargs["points"]
    assert point.vector == [1.0, 0.0, 0.0, 0.0]


def test_upsert_simple_ignores_sparse(qc) -> None:
    store = _store(qc)
    store.upsert([_doc(1)], [[1.0] * 4], sparse_vectors=[{"indices": [1], "values": [1.0]}])
    (point,) = qc.upsert.call_args.kwargs["points"]
    assert point.vector == [1.0] * 4


def test_upsert_retries_transient_failure(qc) -> None:
    store = _store(qc)
    qc.upsert.side_effect = [ConnectionError("blip"), None]
    store.upsert([_doc(1)], [[1.0] * 4])
    assert qc.upsert.call_count == 2


def test_upsert_gives_up_after_three_attempts(qc) -> None:
    from tenacity import RetryError

    store = _store(qc)
    qc.upsert.side_effect = ConnectionError("down")
    with pytest.raises(RetryError):
        store.upsert([_doc(1)], [[1.0] * 4])
    assert qc.upsert.call_count == 3


# ── filters ───────────────────────────────────────────────────────────────────


def test_build_filter_none_when_empty(qc) -> None:
    assert _store(qc)._build_filter(None, None, None) is None


@pytest.mark.parametrize(
    ("value", "key"),
    [
        ("C0123456789", "channel_id"),
        ("C01234567", "channel_id"),
        ("C0123456", "channel_name"),  # too short for an ID
        ("general", "channel_name"),
        ("c0123456789", "channel_name"),  # lowercase is a name
    ],
)
def test_build_filter_channel_key(qc, value: str, key: str) -> None:
    flt = _store(qc)._build_filter(value, None, None)
    (cond,) = flt.must
    assert cond.key == key
    assert cond.match == qmodels.MatchValue(value=value)


_DATE_RANGE_BUG = pytest.mark.xfail(
    raises=ValidationError,
    strict=True,
    reason="BUG: _build_filter passes ISO date strings to qmodels.Range, which only "
    "accepts numbers; should use qmodels.DatetimeRange",
)


@_DATE_RANGE_BUG
def test_build_filter_date_to_only(qc) -> None:
    flt = _store(qc)._build_filter(None, None, "2024-01-31")
    (cond,) = flt.must
    assert cond.key == "date"


@_DATE_RANGE_BUG
def test_build_filter_dates_and_channel(qc) -> None:
    flt = _store(qc)._build_filter("general", "2024-01-01", "2024-01-31")
    assert len(flt.must) == 3
    _, frm, to = flt.must
    assert frm.key == "date" and frm.range.gte is not None
    assert to.key == "date" and to.range.lte is not None


# ── search ────────────────────────────────────────────────────────────────────


def _hits(*pairs) -> SimpleNamespace:
    return SimpleNamespace(
        points=[SimpleNamespace(score=s, payload=p) for s, p in pairs]
    )


def test_dense_search_basic(qc) -> None:
    store = _store(qc)
    qc.query_points.return_value = _hits(
        (0.9, {"text": "a", "channel_name": "general"}),
        (0.5, {"text": "b"}),
    )
    results = store.search([0.1, 0.2, 0.3, 0.4], limit=5)
    assert results == [
        {"score": 0.9, "text": "a", "channel_name": "general"},
        {"score": 0.5, "text": "b"},
    ]
    assert qc.query_points.call_args.kwargs == {
        "collection_name": "slack",
        "query": [0.1, 0.2, 0.3, 0.4],
        "limit": 5,
        "query_filter": None,
        "with_payload": True,
    }


def test_dense_search_with_filter_and_threshold(qc) -> None:
    store = _store(qc)
    qc.query_points.return_value = _hits()
    assert store.search([0.0] * 4, channel_filter="general", score_threshold=0.3) == []
    kwargs = qc.query_points.call_args.kwargs
    assert kwargs["score_threshold"] == 0.3
    (cond,) = kwargs["query_filter"].must
    assert cond.key == "channel_name"


def test_search_simple_mode_ignores_sparse_vector(qc) -> None:
    store = _store(qc)
    qc.query_points.return_value = _hits()
    store.search([0.0] * 4, sparse_vector={"indices": [1], "values": [1.0]})
    kwargs = qc.query_points.call_args.kwargs
    assert "prefetch" not in kwargs
    assert kwargs["query"] == [0.0] * 4


def test_search_hybrid_without_sparse_uses_dense(qc) -> None:
    store = _store(qc, hybrid=True)
    qc.query_points.return_value = _hits()
    store.search([0.0] * 4)
    assert "prefetch" not in qc.query_points.call_args.kwargs


def test_hybrid_search_uses_rrf_prefetch(qc) -> None:
    store = _store(qc, hybrid=True)
    qc.query_points.return_value = _hits((0.03, {"text": "x"}))
    results = store.search(
        [0.1] * 4,
        limit=4,
        channel_filter="C0123456789",
        sparse_vector={"indices": [2], "values": [0.7]},
    )
    assert results == [{"score": 0.03, "text": "x"}]

    kwargs = qc.query_points.call_args.kwargs
    assert kwargs["query"] == qmodels.FusionQuery(fusion=qmodels.Fusion.RRF)
    assert kwargs["limit"] == 4
    assert "score_threshold" not in kwargs
    assert "query_filter" not in kwargs
    dense, sparse = kwargs["prefetch"]
    assert dense.using == "dense" and dense.query == [0.1] * 4
    assert sparse.using == "sparse"
    assert sparse.query == qmodels.SparseVector(indices=[2], values=[0.7])
    # prefetch_limit = max(limit*3, 20)
    assert dense.limit == sparse.limit == 20
    assert dense.filter.must[0].key == "channel_id"
    assert sparse.filter == dense.filter


def test_hybrid_search_prefetch_limit_scales_and_threshold(qc) -> None:
    store = _store(qc, hybrid=True)
    qc.query_points.return_value = _hits()
    store.search([0.1] * 4, limit=10, score_threshold=0.01,
                 sparse_vector={"indices": [], "values": []})
    kwargs = qc.query_points.call_args.kwargs
    assert kwargs["prefetch"][0].limit == 30
    assert kwargs["prefetch"][0].filter is None
    assert kwargs["score_threshold"] == 0.01


# ── counts ────────────────────────────────────────────────────────────────────


def test_count_uses_count_attr(qc) -> None:
    store = _store(qc)
    qc.count.return_value = SimpleNamespace(count=42)
    assert store.count() == 42
    qc.count.assert_called_once_with(collection_name="slack")


def test_count_falls_back_to_points_count(qc) -> None:
    store = _store(qc)
    qc.count.return_value = SimpleNamespace(points_count=7)
    assert store.count() == 7


def test_count_zero_when_missing(qc) -> None:
    store = _store(qc)
    qc.count.return_value = SimpleNamespace()
    assert store.count() == 0


def test_count_by_channel_excludes_summaries(qc) -> None:
    store = _store(qc)
    qc.count.return_value = SimpleNamespace(count=5)
    assert store.count_by_channel("C0123456789") == 5

    kwargs = qc.count.call_args.kwargs
    assert kwargs["collection_name"] == "slack"
    flt = kwargs["count_filter"]
    (must,) = flt.must
    assert must.key == "channel_id"
    assert must.match == qmodels.MatchValue(value="C0123456789")
    (must_not,) = flt.must_not
    assert must_not.key == "doc_type"
    assert set(must_not.match.any) == {
        "channel_summary",
        "workspace_summary",
        "user_summary",
        "team_summary",
        "thread_summary",
    }


def test_count_by_channel_none_is_zero(qc) -> None:
    store = _store(qc)
    qc.count.return_value = SimpleNamespace(count=None)
    assert store.count_by_channel("C1") == 0


# ── integration with the in-memory Qdrant backend ─────────────────────────────


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
def test_in_memory_roundtrip(monkeypatch) -> None:
    from qdrant_client import QdrantClient

    monkeypatch.setattr(
        vector_store, "QdrantClient", lambda **kw: QdrantClient(":memory:")
    )
    store = VectorStore("ignored", "slack", 4)
    docs = [
        _doc(1, channel_name="general", date="2024-01-10"),
        _doc(2, channel_name="random", date="2024-02-10"),
        _doc(3, channel_name="general", date="2024-03-10", doc_type="channel_summary"),
    ]
    store.upsert(docs, [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]])

    assert store.count() == 3
    assert store.count_by_channel("C0123456789") == 2

    hits = store.search([1, 0, 0, 0], limit=3)
    assert hits[0]["text"] == "hello 1"
    assert hits[0]["score"] == pytest.approx(1.0)

    general = store.search([1, 0, 0, 0], channel_filter="general")
    assert {h["channel_name"] for h in general} == {"general"}

    by_id = store.search([0, 1, 0, 0], limit=1, channel_filter="C0123456789")
    assert by_id[0]["text"] == "hello 2"

    assert store.search([1, 0, 0, 0], score_threshold=0.99)[0]["text"] == "hello 1"
