"""Evaluator tests (LLM chat-completions scoring)."""

from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
from unittest.mock import MagicMock

import pytest

from slack2rag import evaluator
from slack2rag.evaluator import _chat_completion, _format_line, run_eval, score_message


class _JsonResponse:
    def __init__(self, body: object) -> None:
        self._body = body

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self._body, bytes):
            return self._body
        return json.dumps(self._body).encode()


def _chat_body(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://llm/v1/chat/completions", code, "err", {}, io.BytesIO(body),
    )


# ── _chat_completion ─────────────────────────────────────────────────────────


def test_chat_completion_request_shape(monkeypatch) -> None:
    captured: dict = {}

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> _JsonResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["ctype"] = req.get_header("Content-type")
        captured["payload"] = json.loads(req.data.decode())
        return _JsonResponse(_chat_body('{"score": 8}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = _chat_completion("http://llm/v1/", "sk-x", "m1", "hello", timeout=7)

    assert out == '{"score": 8}'
    assert captured["url"] == "http://llm/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["auth"] == "Bearer sk-x"
    assert captured["ctype"] == "application/json"
    p = captured["payload"]
    assert p["model"] == "m1"
    assert p["messages"] == [{"role": "user", "content": "hello"}]
    assert p["response_format"] == {"type": "json_object"}
    assert p["temperature"] == 0
    assert p["stream"] is False


def test_chat_completion_no_auth_header_without_key(monkeypatch) -> None:
    seen: list = []

    def fake_urlopen(req, timeout):
        seen.append(req.get_header("Authorization"))
        return _JsonResponse(_chat_body("x"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _chat_completion("http://llm/v1", "", "m", "hi") == "x"
    assert seen == [None]


def test_chat_completion_http_error_returns_none_and_logs_body(monkeypatch, caplog) -> None:
    def fake_urlopen(req, timeout):
        raise _http_error(500, b"model exploded")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with caplog.at_level(logging.ERROR, logger="slack2rag.evaluator"):
        assert _chat_completion("http://llm/v1", "", "m", "hi") is None
    assert "HTTP 500" in caplog.text
    assert "model exploded" in caplog.text


def test_chat_completion_http_error_unreadable_body(monkeypatch, caplog) -> None:
    exc = _http_error(502)
    exc.read = MagicMock(side_effect=OSError("boom"))

    def fake_urlopen(req, timeout):
        raise exc

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with caplog.at_level(logging.ERROR, logger="slack2rag.evaluator"):
        assert _chat_completion("http://llm/v1", "", "m", "hi") is None
    assert "HTTP 502" in caplog.text


def test_chat_completion_url_error_returns_none(monkeypatch, caplog) -> None:
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with caplog.at_level(logging.ERROR, logger="slack2rag.evaluator"):
        assert _chat_completion("http://llm/v1", "", "m", "hi") is None
    assert "connection refused" in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": "nope"},
    ],
)
def test_chat_completion_unexpected_shape_returns_none(monkeypatch, caplog, body) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: _JsonResponse(body))
    with caplog.at_level(logging.ERROR, logger="slack2rag.evaluator"):
        assert _chat_completion("http://llm/v1", "", "m", "hi") is None
    assert "Unexpected chat-completions response shape" in caplog.text


# ── score_message ────────────────────────────────────────────────────────────


def _patch_completion(monkeypatch, content):
    calls: list = []

    def fake(base_url, api_key, model, user_content, timeout=120):
        calls.append((base_url, api_key, model, user_content))
        return content

    monkeypatch.setattr(evaluator, "_chat_completion", fake)
    return calls


def test_score_message_parses_json_and_builds_prompt(monkeypatch) -> None:
    calls = _patch_completion(monkeypatch, '  {"score": 8, "reason": "useful"}  ')
    score, reason = score_message("http://b", "m", "Rate it", "hello world", api_key="k")
    assert (score, reason) == (8, "useful")
    base_url, api_key, model, content = calls[0]
    assert (base_url, api_key, model) == ("http://b", "k", "m")
    assert content.startswith("Rate it\n\nMessage:\nhello world\n\n")
    assert '"score": <1-10>' in content


def test_score_message_string_score_and_missing_reason(monkeypatch) -> None:
    _patch_completion(monkeypatch, '{"score": "6"}')
    assert score_message("u", "m", "p", "msg") == (6, "")


def test_score_message_none_response(monkeypatch) -> None:
    _patch_completion(monkeypatch, None)
    assert score_message("u", "m", "p", "msg") == (None, "")


def test_score_message_regex_fallback(monkeypatch) -> None:
    _patch_completion(monkeypatch, "I would give this a 9 out of ten")
    assert score_message("u", "m", "p", "msg") == (9, "")


def test_score_message_regex_fallback_on_missing_key(monkeypatch) -> None:
    # Valid JSON, but no "score" key -> KeyError -> regex over raw text.
    _patch_completion(monkeypatch, '{"rating": 4}')
    assert score_message("u", "m", "p", "msg") == (4, "")


def test_score_message_non_numeric_score_falls_back(monkeypatch) -> None:
    _patch_completion(monkeypatch, '{"score": "high", "reason": "r"}')
    # int("high") -> ValueError; no digits anywhere -> None
    assert score_message("u", "m", "p", "msg") == (None, "")


def test_score_message_unparseable(monkeypatch) -> None:
    _patch_completion(monkeypatch, "no idea")
    assert score_message("u", "m", "p", "msg") == (None, "")


def test_score_message_clamps_above_ten(monkeypatch) -> None:
    _patch_completion(monkeypatch, '{"score": 42, "reason": "wow"}')
    assert score_message("u", "m", "p", "msg") == (10, "wow")


def test_score_message_regex_fallback_clamps(monkeypatch) -> None:
    _patch_completion(monkeypatch, "score: 15")
    assert score_message("u", "m", "p", "msg") == (10, "")


# ── _format_line ─────────────────────────────────────────────────────────────


def test_format_line_with_reason() -> None:
    line = _format_line(7, "good", "general", "alice", "1.0", "hi\nthere")
    assert line == "[ 7] #general  @alice  ts=1.0  hi there\n     reason: good\n"


def test_format_line_without_reason_truncates() -> None:
    line = _format_line(10, "", "c", "u", "2.0", "x" * 400)
    assert line.endswith("x" * 300 + "…\n")
    assert "reason" not in line
    assert line.startswith("[10] #c")


# ── run_eval ─────────────────────────────────────────────────────────────────


def _fake_slack(messages_by_channel: dict) -> MagicMock:
    slack = MagicMock()
    slack.get_channel_messages.side_effect = lambda cid, oldest_ts=None: iter(
        messages_by_channel.get(cid, [])
    )
    slack.resolve_text.side_effect = lambda t: t.upper()
    slack.get_user_name.side_effect = lambda uid: f"name-{uid}"
    slack.get_user_record.side_effect = lambda uid: {"id": uid} if uid != "U_NONE" else None
    return slack


def test_run_eval_writes_good_and_bad(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(evaluator, "RESULTS_DIR", str(tmp_path / "results"))
    scores = {"GREAT": (9, "great"), "MEH": (3, "meh"), "BROKEN": (None, ""), "EDGE": (7, "")}
    seen_keys: list = []

    def fake_score(base_url, model, prompt, message, api_key=""):
        seen_keys.append(api_key)
        return scores[message]

    monkeypatch.setattr(evaluator, "score_message", fake_score)

    slack = _fake_slack({
        "C1": [
            {"ts": "1.0", "user": "U1", "text": "great"},
            {"ts": "2.0", "user": "U2", "text": "meh"},
            {"ts": "3.0", "user": "U1", "text": "   "},
            {"ts": "4.0", "text": "broken"},
        ],
        "C2": [{"ts": "5.0", "user": "U3", "text": "edge"}],
    })
    channels = [{"id": "C1", "name": "general"}, {"id": "C2"}]

    run_eval("http://b", "m", "prompt", channels, slack, api_key="k")

    good = (tmp_path / "results" / "good.txt").read_text()
    bad = (tmp_path / "results" / "bad.txt").read_text()
    assert "[ 9] #general  @name-U1  ts=1.0  GREAT" in good
    assert "reason: great" in good
    # channel without name falls back to id; score == threshold is "good"
    assert "[ 7] #C2  @name-U3  ts=5.0  EDGE" in good
    assert "[ 3] #general  @name-U2  ts=2.0  MEH" in bad
    assert "BROKEN" not in good + bad
    # whitespace-only message skipped (never scored); 4 scored calls
    assert seen_keys == ["k"] * 4
    slack.get_user_name.assert_any_call("unknown")


def test_run_eval_persists_to_pg(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(evaluator, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(evaluator, "score_message", lambda *a, **k: (5, ""))
    msgs = [
        {"ts": "1.0", "user": "U1", "text": "a"},
        {"ts": "2.0", "user": "U_NONE", "text": "b"},
        {"ts": "3.0", "user": {"weird": 1}, "text": ""},
        {"ts": "4.0", "text": "c"},
    ]
    slack = _fake_slack({"C1": msgs})
    pg = MagicMock()
    ch = {"id": "C1", "name": "x"}

    run_eval("u", "m", "p", [ch], slack, pg=pg)

    pg.upsert_channel.assert_called_once_with(ch)
    assert [c.args for c in pg.upsert_message.call_args_list] == [("C1", m) for m in msgs]
    # only U1 had a record; U_NONE returned None; dict user / missing user skipped
    pg.upsert_user.assert_called_once_with({"id": "U1"})
    assert [c.args[0] for c in slack.get_user_record.call_args_list] == ["U1", "U_NONE"]


def test_run_eval_progress_logging(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setattr(evaluator, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(evaluator, "score_message", lambda *a, **k: (8, ""))
    msgs = [{"ts": f"{i}.0", "user": "U", "text": f"m{i}"} for i in range(100)]
    slack = _fake_slack({"C": msgs})
    with caplog.at_level(logging.INFO, logger="slack2rag.evaluator"):
        run_eval("u", "m", "p", [{"id": "C"}], slack)
    assert "Eval progress: 100 scored (100 good, 0 bad, 0 errors)" in caplog.text
    assert "Eval complete: 100 messages scored" in caplog.text
    assert len((tmp_path / "good.txt").read_text().splitlines()) == 100
