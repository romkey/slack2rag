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
        return json.dumps({"embeddings": [[1.0] for _ in range(self._embedding_count)]}).encode()


def test_embedder_prefixes_ollama_inputs(monkeypatch) -> None:
    payloads: list[dict] = []

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _FakeResponse:
        assert timeout == 120
        payload = json.loads(req.data.decode())
        payloads.append(payload)
        return _FakeResponse(len(payload["input"]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    embedder = Embedder(
        url="http://ollama:11434",
        model="nomic-embed-text",
        input_prefix="search_document:",
    )

    assert embedder.embed(["hello", "world"]) == [[1.0], [1.0]]
    assert payloads[0]["input"] == ["search_document:dimension probe"]
    assert payloads[1]["input"] == ["search_document:hello", "search_document:world"]
