"""Extractor tests, with scripted model answers."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from slack2rag.extractor import (
    EXCLUSION_SYSTEM,
    RELEVANCE_SYSTEM,
    Extractor,
    Fact,
    fact_id,
    load_topics,
    point_id,
    stale_after,
)
from slack2rag.llm import ChatModel, LLMError


class Scripted:
    def __init__(self, *answers: dict[str, Any], model: str = "m") -> None:
        self.answers = list(answers)
        self.model = model
        self.prompts: list[tuple[str, str]] = []

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        self.prompts.append((system, user))
        return self.answers.pop(0)


def _extractor(*answers: dict[str, Any], topics: tuple[str, ...] = ()) -> Extractor:
    return Extractor(cast(ChatModel, Scripted(*answers)), topics, ["Alice"])


def _fact(**extra: Any) -> dict[str, Any]:
    return {
        "kind": "howto",
        "question": "How do I start the lathe?",
        "answer": "Press the green button.",
        "confidence": "high",
        **extra,
    }


def test_not_relevant_stops_after_one_call() -> None:
    ex = _extractor({"relevant": False, "reason": "chit chat"})
    result = ex.process("hi", "general")
    assert not result.relevant
    assert result.facts == []
    assert len(cast(Scripted, ex.model).prompts) == 1


def test_full_pipeline_validates_and_scrubs() -> None:
    ex = _extractor(
        {"relevant": "yes"},
        {
            "facts": [
                _fact(
                    entities=["lathe", "lathe"],
                    links=["https://x.org", "https://made.up"],
                ),
                _fact(kind="gossip"),
                _fact(answer=""),
                "not a dict",
                _fact(answer="Ask Alice, she knows.", confidence="sure"),
                _fact(answer="Ask the shop lead.", topics=["woodworking", "nope"]),
            ]
        },
        {
            "verdicts": [
                {"index": 0, "exclude": False, "weakness": True},
                {"index": 1, "exclude": False},
                {"index": 2, "exclude": "false"},
            ]
        },
        topics=("woodworking",),
    )
    result = ex.process("See https://x.org for the lathe", "shop", boost=True)

    assert result.relevant
    first, second, third = result.facts
    assert first.entities == ("lathe",)
    assert first.links == ("https://x.org",)  # the made-up link is dropped
    assert first.tags == ("weakness",)
    assert second.answer == "Ask [member], she knows."
    assert second.confidence == "low"
    assert third.topics == ("woodworking",)
    relevance_prompt = cast(Scripted, ex.model).prompts[0][1]
    assert "pinned or bookmarked" in relevance_prompt


def test_unruled_facts_are_dropped() -> None:
    ex = _extractor(
        {"relevant": True},
        {"facts": [_fact(), _fact(question="Other?")]},
        {"verdicts": [{"index": 0, "exclude": False}, {"index": "1"}]},
    )
    assert len(ex.process("x", "c").facts) == 1


def test_secret_in_a_fact_drops_it() -> None:
    ex = _extractor(
        {"relevant": True},
        {"facts": [_fact(answer="The door code is 1234."), _fact()]},
        {"verdicts": [{"index": 0, "exclude": False}, {"index": 1, "exclude": False}]},
    )
    result = ex.process("x", "c")
    assert [f.answer for f in result.facts] == ["Press the green button."]
    assert result.excluded == 1


def test_secrets_are_redacted_before_the_model_sees_them() -> None:
    ex = _extractor({"relevant": False})
    result = ex.process("password: hunter2", "c")
    assert result.secret_rules == ("password-assignment",)
    assert "hunter2" not in cast(Scripted, ex.model).prompts[0][1]


@pytest.mark.parametrize(
    "answers",
    [
        ({"reason": "no field"},),
        ({"relevant": True}, {"facts": "nope"}),
        ({"relevant": True}, {"facts": [_fact()]}, {"verdicts": None}),
    ],
)
def test_malformed_answers_raise(answers: tuple[dict[str, Any], ...]) -> None:
    with pytest.raises(LLMError):
        _extractor(*answers).process("x", "c")


def test_long_input_is_truncated() -> None:
    ex = _extractor({"relevant": False})
    ex.process("x" * 20_000, "c")
    assert "[…truncated]" in cast(Scripted, ex.model).prompts[0][1]


def test_version_tracks_model_and_topics() -> None:
    a = Extractor(cast(ChatModel, Scripted(model="a")))
    b = Extractor(cast(ChatModel, Scripted(model="b")))
    c = Extractor(cast(ChatModel, Scripted(model="a")), ("lasers",))
    assert len({a.version, b.version, c.version}) == 3
    assert "lasers" in c.extraction_system
    assert '"topics"' in c.extraction_system
    assert '"topics"' not in a.extraction_system


def test_prompts_are_distinct() -> None:
    assert RELEVANCE_SYSTEM != EXCLUSION_SYSTEM


def test_stale_after() -> None:
    day = datetime.date(2026, 1, 1)
    assert stale_after("status", day) == datetime.datetime(
        2026, 1, 8, tzinfo=datetime.UTC
    )
    assert stale_after("announcement", day) == datetime.datetime(
        2026, 1, 31, tzinfo=datetime.UTC
    )
    assert stale_after("policy", day) is None
    assert stale_after("history", day) is None


def test_ids_are_stable() -> None:
    assert fact_id("C1", "1.0", 0) == fact_id("C1", "1.0", 0)
    assert fact_id("C1", "1.0", 0) != fact_id("C1", "1.0", 1)
    assert fact_id("C1", "1.0", 0) == point_id("slack", "C1:1.0:0")


def test_fact_text_and_hash() -> None:
    fact = Fact("howto", "Q?", "A.", "high")
    assert fact.text == "Q: Q?\nA: A."
    assert fact.content_hash == Fact("howto", "Q?", "A.", "low").content_hash


def test_load_topics(tmp_path: Path) -> None:
    path = tmp_path / "topics.txt"
    path.write_text("# taxonomy\nlasers\n\n3d-printing  # printers\n")
    assert load_topics(str(path)) == ("lasers", "3d-printing")
    assert load_topics("") == ()
