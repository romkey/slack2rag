"""Embedder tests."""

from __future__ import annotations

import json
import urllib.request

from src.embedder import Embedder


class _FakeResponse:
    def __init__(self, embedding_count: int) -> None:
        self._embedding_count = embedding_count

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        body = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": [1.0]}
                for i in range(self._embedding_count)
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        return json.dumps(body).encode()


def test_embedder_uses_openai_embeddings_endpoint(monkeypatch) -> None:
    payloads: list[dict] = []
    urls: list[str] = []
    auth_headers: list[str | None] = []

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _FakeResponse:
        assert timeout == 120
        urls.append(req.full_url)
        auth_headers.append(req.get_header("Authorization"))
        payload = json.loads(req.data.decode())
        payloads.append(payload)
        return _FakeResponse(len(payload["input"]))

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

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _FakeResponse:
        auth_headers.append(req.get_header("Authorization"))
        payload = json.loads(req.data.decode())
        return _FakeResponse(len(payload["input"]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    Embedder(base_url="http://localhost:11434/v1", model="nomic-embed-text")
    assert auth_headers == [None]
