"""Embedder tests."""

from __future__ import annotations

import json
import urllib.request

import pytest

from src.embedder import Embedder, EmbeddingError


class _JsonResponse:
    def __init__(self, body: dict) -> None:
        self._body = body

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._body).encode()


def _embedding_list(vectors: list[list[float]]) -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def test_embedder_uses_openai_embeddings_endpoint(monkeypatch) -> None:
    payloads: list[dict] = []
    urls: list[str] = []
    auth_headers: list[str | None] = []

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _JsonResponse:
        assert timeout == 120
        urls.append(req.full_url)
        auth_headers.append(req.get_header("Authorization"))
        payload = json.loads(req.data.decode())
        payloads.append(payload)
        return _JsonResponse(_embedding_list([[1.0]] * len(payload["input"])))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    embedder = Embedder(
        base_url="http://localhost:11434/v1",
        model="nomic-embed-text",
        api_key="sk-test",
        input_prefix="search_document:",
    )

    assert embedder.embed(["hello", "world"]) == [[1.0], [1.0]]

    assert urls == [
        "http://localhost:11434/v1/embeddings",
        "http://localhost:11434/v1/embeddings",
    ]
    assert auth_headers == ["Bearer sk-test", "Bearer sk-test"]
    assert payloads[0] == {
        "model": "nomic-embed-text",
        "input": ["search_document:dimension probe"],
    }
    assert payloads[1] == {
        "model": "nomic-embed-text",
        "input": ["search_document:hello", "search_document:world"],
    }


def test_embedder_omits_auth_header_when_no_key(monkeypatch) -> None:
    auth_headers: list[str | None] = []

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _JsonResponse:
        auth_headers.append(req.get_header("Authorization"))
        payload = json.loads(req.data.decode())
        return _JsonResponse(_embedding_list([[1.0]] * len(payload["input"])))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    Embedder(base_url="http://localhost:11434/v1", model="nomic-embed-text")
    assert auth_headers == [None]


def test_embedder_orders_by_index(monkeypatch) -> None:
    """Servers may return items in any order; we sort by `index`."""

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _JsonResponse:
        payload = json.loads(req.data.decode())
        n = len(payload["input"])
        if n == 1:
            return _JsonResponse(_embedding_list([[0.0]]))
        # Return in reverse order with explicit indexes
        return _JsonResponse({
            "object": "list",
            "data": [
                {"index": 2, "embedding": [0.3]},
                {"index": 0, "embedding": [0.1]},
                {"index": 1, "embedding": [0.2]},
            ],
            "model": "m",
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    embedder = Embedder(base_url="http://x/v1", model="m")
    assert embedder.embed(["a", "b", "c"]) == [[0.1], [0.2], [0.3]]


def test_embedder_falls_back_to_single_input_on_null_batch(monkeypatch) -> None:
    """Ollama's /v1/embeddings returns null embeddings for array inputs;
    after the first bad batch we drop to one-input-per-request."""
    calls: list[int] = []

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _JsonResponse:
        payload = json.loads(req.data.decode())
        n = len(payload["input"])
        calls.append(n)
        if n == 1:
            # single-input requests work fine
            return _JsonResponse(_embedding_list([[float(len(calls))]]))
        # array input: server returns nulls (the bug we're working around)
        return _JsonResponse({
            "object": "list",
            "data": [{"index": i, "embedding": None} for i in range(n)],
            "model": "m",
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    embedder = Embedder(base_url="http://x/v1", model="m")
    # probe call already happened (n=1, succeeds)
    assert calls == [1]

    out = embedder.embed(["a", "b", "c"])

    # vectors come back from the single-input retries, so each is a 1-d list
    assert len(out) == 3
    assert all(len(v) == 1 for v in out)
    # request sequence: bad batch of 3, then 3 single-input retries
    assert calls == [1, 3, 1, 1, 1]

    # Subsequent embeds skip the bad-batch attempt entirely
    out2 = embedder.embed(["d", "e"])
    assert len(out2) == 2
    assert calls == [1, 3, 1, 1, 1, 1, 1]


def test_embedder_raises_when_single_input_returns_null(monkeypatch) -> None:
    """If even single-input mode produces a null embedding, we raise clearly."""
    counter = {"n": 0}

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _JsonResponse:
        counter["n"] += 1
        if counter["n"] == 1:
            # probe succeeds so we get past __init__
            return _JsonResponse(_embedding_list([[1.0]]))
        return _JsonResponse({
            "object": "list",
            "data": [{"index": 0, "embedding": None}],
            "model": "m",
        })

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    embedder = Embedder(base_url="http://x/v1", model="m")
    with pytest.raises(EmbeddingError, match="invalid embedding"):
        embedder.embed(["a"])
