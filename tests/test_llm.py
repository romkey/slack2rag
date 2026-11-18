"""Chat-completions client tests."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any, cast

import pytest

from slack2rag import llm
from slack2rag.llm import ChatModel, LLMError, chat_completion


class _Response:
    def __init__(self, body: object) -> None:
        self.body = json.dumps(body).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _answer(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ChatModel.chat_json.retry, "sleep", lambda _: None)


def test_sends_system_and_user_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[urllib.request.Request] = []

    def fake(req: urllib.request.Request, timeout: int) -> _Response:
        sent.append(req)
        return _Response(_answer('{"ok": true}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert (
        chat_completion("http://gw/v1/", "k", "m", "hi", system="sys") == '{"ok": true}'
    )
    body = json.loads(cast(bytes, sent[0].data))
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert sent[0].full_url == "http://gw/v1/chat/completions"
    assert sent[0].get_header("Authorization") == "Bearer k"


def test_refuses_non_http(caplog: pytest.LogCaptureFixture) -> None:
    assert chat_completion("file:///etc", "", "m", "x") is None


def test_http_error_and_bad_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def http_error(req: object, timeout: int) -> None:
        raise urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"err"))  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", http_error)
    assert chat_completion("http://gw", "", "m", "x") is None

    def unreachable(req: object, timeout: int) -> None:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", unreachable)
    assert chat_completion("http://gw", "", "m", "x") is None

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda r, timeout: _Response({"x": 1})
    )
    assert chat_completion("http://gw", "", "m", "x") is None


def test_chat_json(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(['```json\n{"a": 1}\n```'])
    monkeypatch.setattr(llm, "chat_completion", lambda *a, **k: next(answers))
    assert ChatModel("http://gw", "m").chat_json("s", "u") == {"a": 1}


@pytest.mark.parametrize("content", [None, "not json", "[1, 2]"])
def test_chat_json_failures_retry_then_raise(
    monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    calls = []

    def fake(*args: object, **kwargs: object) -> str | None:
        calls.append(1)
        return content

    monkeypatch.setattr(llm, "chat_completion", fake)
    with pytest.raises(LLMError):
        ChatModel("http://gw", "m").chat_json("s", "u")
    assert len(calls) == 3
