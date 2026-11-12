"""Additional embedder tests: sparse encoding, truncation, error paths."""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TypedDict
from unittest.mock import MagicMock

import pytest

from slack2rag import embedder as emb_mod
from slack2rag.embedder import (
    SPARSE_VOCAB_SIZE,
    Embedder,
    EmbeddingError,
    SparseEncoder,
    _ContextLengthError,
    _is_context_length_error,
    _is_valid_embedding,
    _token_hash,
    _truncate_at_word,
    tokenize_text,
)


class _Payload(TypedDict):
    model: str
    input: list[str]


class _JsonResponse:
    def __init__(self, body: object) -> None:
        self._body = body

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._body).encode()


def _ok(n: int, vec: list[float] | None = None) -> dict[str, object]:
    return {"data": [{"index": i, "embedding": vec or [1.0, 2.0]} for i in range(n)]}


def _http_error(
    code: int, body: bytes = b"", reason: str = "Bad"
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://llm/v1/embeddings",
        code,
        reason,
        email.message.Message(),
        io.BytesIO(body),
    )


def _make_embedder(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[_Payload], _JsonResponse],
    context_length: int = 0,
    request_batch_size: int = 50,
) -> Embedder:
    """Create an Embedder whose probe succeeds, then route later calls to *handler*."""
    state = {"probed": False}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _JsonResponse:
        assert isinstance(req.data, bytes)
        payload: _Payload = json.loads(req.data)
        if not state["probed"]:
            state["probed"] = True
            return _JsonResponse(_ok(len(payload["input"])))
        return handler(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return Embedder(
        base_url="http://llm/v1/",
        model="m",
        context_length=context_length,
        request_batch_size=request_batch_size,
    )


# ── tokenizer / sparse encoder ───────────────────────────────────────────────


def test_tokenize_text_filters_stopwords_and_short_tokens() -> None:
    assert tokenize_text("The Deploy of K8s is a GO! x y") == ["deploy", "k8s", "go"]
    assert tokenize_text("") == []


def test_token_hash_deterministic_and_in_range() -> None:
    h = _token_hash("deploy")
    assert h == _token_hash("deploy")
    assert 0 <= h < SPARSE_VOCAB_SIZE


def test_sparse_encoder_empty_text() -> None:
    assert SparseEncoder().encode(["", "the a an"]) == [
        {"indices": [0], "values": [0.0]},
        {"indices": [0], "values": [0.0]},
    ]


def test_sparse_encoder_log_tf_weights_sorted() -> None:
    out = SparseEncoder().encode(["deploy deploy deploy kubernetes"])[0]
    assert out["indices"] == sorted(out["indices"])
    weights = dict(zip(out["indices"], out["values"], strict=True))
    assert weights[_token_hash("kubernetes")] == pytest.approx(1.0)
    assert weights[_token_hash("deploy")] == pytest.approx(1.0 + 1.0986122886681098)


def test_sparse_encoder_hash_collision_accumulates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(emb_mod, "_token_hash", lambda t: 7)
    out = SparseEncoder().encode(["alpha beta"])[0]
    assert out == {"indices": [7], "values": [2.0]}


# ── helpers ──────────────────────────────────────────────────────────────────


def test_truncate_at_word() -> None:
    assert _truncate_at_word("short", 10) == "short"
    assert _truncate_at_word("hello world foo", 13) == "hello world"
    # no space in the back half -> hard cut
    assert _truncate_at_word("a" + "b" * 20, 10) == "a" + "b" * 9


@pytest.mark.parametrize(
    ("code", "body", "expected"),
    [
        (400, b"input exceeds maximum context length", True),
        (413, b"Request too long", True),
        (422, b'{"error":"context_length_exceeded"}', True),
        (400, b"invalid model", False),
        (500, b"context length", False),
    ],
)
def test_is_context_length_error(code: int, body: bytes, expected: bool) -> None:
    assert _is_context_length_error(_http_error(code, body)) is expected


def test_is_context_length_error_unreadable_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = _http_error(400)
    monkeypatch.setattr(exc, "read", MagicMock(side_effect=OSError("closed")))
    assert _is_context_length_error(exc) is False


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ([0.1, 2, -3.5], True),
        ([], False),
        (None, False),
        ("abc", False),
        ([1.0, None], False),
        ([True, 1.0], False),
        ([1.0, "2"], False),
    ],
)
def test_is_valid_embedding(value: object, ok: bool) -> None:
    assert _is_valid_embedding(value) is ok


# ── constructor ──────────────────────────────────────────────────────────────


def test_constructor_wraps_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _JsonResponse:
        raise ValueError("weird")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(EmbeddingError, match="Failed to connect to LLM at http://x/v1"):
        Embedder(base_url="http://x/v1", model="m")


@pytest.mark.parametrize("base_url", ["file:///etc/passwd", "ftp://x/v1", "llm/v1"])
def test_constructor_rejects_non_http_scheme(base_url: str) -> None:
    with pytest.raises(EmbeddingError, match="must start with http:// or https://"):
        Embedder(base_url=base_url, model="m")


def test_constructor_propagates_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(req: urllib.request.Request, timeout: float) -> _JsonResponse:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(
        EmbeddingError, match="Could not reach LLM at http://x/v1: refused"
    ):
        Embedder(base_url="http://x/v1", model="m")


def test_constructor_dimension_and_context_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def handler(payload: _Payload) -> _JsonResponse:
        seen.append(payload["input"])
        return _JsonResponse(_ok(len(payload["input"])))

    e = _make_embedder(monkeypatch, handler, context_length=4)  # 12 chars
    assert e.dimension == 2
    assert e.embed([]) == []
    assert seen == []
    e.embed(["hello world and more words"])
    assert seen == [["hello world"]]


def test_request_batch_size_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []

    def handler(payload: _Payload) -> _JsonResponse:
        sizes.append(len(payload["input"]))
        return _JsonResponse(_ok(len(payload["input"])))

    e = _make_embedder(monkeypatch, handler, request_batch_size=2)
    assert len(e.embed(["a", "b", "c", "d", "e"])) == 5
    assert sizes == [2, 2, 1]


def test_request_batch_size_minimum_one(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []

    def handler(payload: _Payload) -> _JsonResponse:
        sizes.append(len(payload["input"]))
        return _JsonResponse(_ok(len(payload["input"])))

    e = _make_embedder(monkeypatch, handler, request_batch_size=0)
    e.embed(["a", "b"])
    assert sizes == [1, 1]


# ── HTTP error handling ──────────────────────────────────────────────────────


def test_http_error_includes_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(payload: _Payload) -> _JsonResponse:
        raise _http_error(500, b"internal kaboom", reason="Server Error")

    e = _make_embedder(monkeypatch, handler)
    with pytest.raises(EmbeddingError) as exc:
        e.embed(["x"])
    msg = str(exc.value)
    assert "HTTP 500 (Server Error)" in msg
    assert "POST http://llm/v1/embeddings" in msg
    assert "Response: internal kaboom" in msg
    assert not isinstance(exc.value, _ContextLengthError)


def test_http_error_unreadable_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(payload: _Payload) -> _JsonResponse:
        exc = _http_error(503)
        monkeypatch.setattr(exc, "read", MagicMock(side_effect=OSError("closed")))
        raise exc

    e = _make_embedder(monkeypatch, handler)
    with pytest.raises(EmbeddingError) as exc:
        e.embed(["x"])
    assert "HTTP 503" in str(exc.value)
    assert "Response:" not in str(exc.value)


def test_url_error_after_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(payload: _Payload) -> _JsonResponse:
        raise urllib.error.URLError("timed out")

    e = _make_embedder(monkeypatch, handler)
    with pytest.raises(EmbeddingError, match="timed out"):
        e.embed(["x"])


# ── context-length fallback ──────────────────────────────────────────────────


def test_context_error_falls_back_to_single_and_halves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_text = ("word " * 100).strip()  # 499 chars
    inputs: list[list[str]] = []

    def handler(payload: _Payload) -> _JsonResponse:
        inputs.append(payload["input"])
        batch = payload["input"]
        if len(batch) > 1 or len(batch[0]) > 200:
            raise _http_error(400, b"input is too long for context window")
        return _JsonResponse(_ok(1, [3.0]))

    e = _make_embedder(monkeypatch, handler)
    out = e.embed([long_text, "short"])
    assert out == [[3.0], [3.0]]
    # batch attempt, then long text: 499 -> fail, ~249 -> fail, ~124 -> ok, then "short"
    assert len(inputs[0]) == 2
    single_lengths = [len(b[0]) for b in inputs[1:]]
    assert single_lengths[0] == 499
    assert single_lengths[1] < 250
    assert single_lengths[2] < 125
    assert inputs[-1] == ["short"]


def test_context_error_gives_up_below_min_len(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(payload: _Payload) -> _JsonResponse:
        raise _http_error(413, b"payload exceeds limit")

    e = _make_embedder(monkeypatch, handler)
    with pytest.raises(EmbeddingError, match="Cannot embed text even at"):
        e.embed(["x" * 100])


# ── response-shape validation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ([1, 2], "Unexpected response shape"),
        ({"embeddings": []}, "Unexpected response shape"),
        ({"data": "nope"}, "Unexpected response shape"),
        (
            {"data": [{"index": 0, "embedding": [1.0]}] * 2},
            r"returned 2 embedding\(s\) for 1",
        ),
    ],
)
def test_bad_response_shapes(
    monkeypatch: pytest.MonkeyPatch, body: object, match: str
) -> None:
    e = _make_embedder(monkeypatch, lambda p: _JsonResponse(body))
    with pytest.raises(EmbeddingError, match=match):
        e.embed(["x"])


def test_non_dict_item_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    e = _make_embedder(monkeypatch, lambda p: _JsonResponse({"data": ["oops"]}))
    with pytest.raises(EmbeddingError, match="invalid embedding for a single input"):
        e.embed(["x"])


def test_bad_index_falls_back_to_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "data": [
            {"index": 99, "embedding": [1.0]},
            {"index": "zero", "embedding": [2.0]},
        ]
    }
    e = _make_embedder(monkeypatch, lambda p: _JsonResponse(body))
    assert e.embed(["a", "b"]) == [[1.0], [2.0]]


def test_missing_index_uses_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]}
    e = _make_embedder(monkeypatch, lambda p: _JsonResponse(body))
    assert e.embed(["a", "b"]) == [[1.0], [2.0]]


def test_duplicate_index_leaves_hole_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def handler(payload: _Payload) -> _JsonResponse:
        n = len(payload["input"])
        calls.append(n)
        if n == 2:
            return _JsonResponse(
                {
                    "data": [
                        {"index": 0, "embedding": [1.0]},
                        {"index": 0, "embedding": [2.0]},
                    ]
                }
            )
        return _JsonResponse(_ok(1, [5.0]))

    e = _make_embedder(monkeypatch, handler)
    assert e.embed(["a", "b"]) == [[5.0], [5.0]]
    assert calls == [2, 1, 1]
    # single-input mode persists for subsequent calls
    e.embed(["c", "d"])
    assert calls == [2, 1, 1, 1, 1]


@pytest.mark.xfail(
    strict=True,
    reason="src bug: _is_context_length_error consumes the HTTPError body, so the "
    "subsequent EmbeddingError for a non-context 400/413/422 has no 'Response:'",
)
def test_non_context_400_keeps_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(payload: _Payload) -> _JsonResponse:
        raise _http_error(400, b"model 'm' is not an embedding model")

    e = _make_embedder(monkeypatch, handler)
    with pytest.raises(EmbeddingError) as exc:
        e.embed(["x"])
    assert "not an embedding model" in str(exc.value)
