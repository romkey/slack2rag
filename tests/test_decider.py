"""Tests for the decision model (slack2rag/decider.py)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from slack2rag import decider
from slack2rag.decider import (
    MAX_INPUT_CHARS,
    Decision,
    DecisionError,
    DecisionModel,
    parse_decision,
)


@pytest.fixture
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries without the 2-second pauses."""
    monkeypatch.setattr(DecisionModel._ask.retry, "wait", lambda *a, **k: 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"useful": true, "reason": " where things are "}',
            Decision(True, "where things are"),
        ),
        ('{"useful": false}', Decision(False, "")),
        ('  {"useful": "yes", "reason": 3}\n', Decision(True, "3")),
        ('{"useful": "No"}', Decision(False, "")),
        ('{"useful": "maybe"}', None),
        ('{"reason": "x"}', None),
        ("[true]", None),
        ("not json", None),
    ],
)
def test_parse_decision(text: str, expected: Decision | None) -> None:
    assert parse_decision(text) == expected


def test_build_prompt_adds_channel_message_and_format() -> None:
    model = DecisionModel("http://llm/v1", "m", "Is it useful?")
    prompt = model.build_prompt("[alice]: drill bits are in bin 4", "tools")
    assert prompt.startswith("Is it useful?\n\nSlack channel: #tools\n\n")
    assert "Message:\n[alice]: drill bits are in bin 4\n\n" in prompt
    assert prompt.endswith(
        '{"useful": true or false, "reason": "<one short sentence>"}'
    )
    assert "Slack channel" not in model.build_prompt("x")


def test_build_prompt_truncates_long_text() -> None:
    prompt = DecisionModel("u", "m", "p").build_prompt("a" * (MAX_INPUT_CHARS + 50))
    assert "a" * MAX_INPUT_CHARS + "\n[…truncated]" in prompt
    assert "a" * (MAX_INPUT_CHARS + 1) not in prompt


def test_decide_sends_to_configured_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = MagicMock(return_value='{"useful": true, "reason": "how-to"}')
    monkeypatch.setattr(decider, "_chat_completion", chat)
    model = DecisionModel("http://llm/v1", "qwen", "p", api_key="k", timeout=9)
    assert model.decide("text", "general") == Decision(True, "how-to")
    args, kwargs = chat.call_args
    assert args[:3] == ("http://llm/v1", "k", "qwen")
    assert "text" in args[3]
    assert kwargs == {"timeout": 9}


def test_decide_unparseable_answer_counts_as_useful(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(decider, "_chat_completion", MagicMock(return_value="hmm"))
    with caplog.at_level(logging.WARNING, logger="slack2rag"):
        result = DecisionModel("u", "m", "p").decide("x")
    assert result.useful is True
    assert "unparseable answer; indexing anyway: hmm" in caplog.text


@pytest.mark.usefixtures("no_wait")
def test_decide_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = MagicMock(return_value=None)
    monkeypatch.setattr(decider, "_chat_completion", chat)
    with pytest.raises(DecisionError, match="'m' at u gave no answer"):
        DecisionModel("u", "m", "p").decide("x")
    assert chat.call_count == 3


@pytest.mark.usefixtures("no_wait")
def test_decide_recovers_after_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = MagicMock(side_effect=[None, '{"useful": false, "reason": "chat"}'])
    monkeypatch.setattr(decider, "_chat_completion", chat)
    assert DecisionModel("u", "m", "p").decide("x") == Decision(False, "chat")
