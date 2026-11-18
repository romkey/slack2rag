"""Conversation unit tests."""

from __future__ import annotations

from slack2rag.units import (
    Message,
    bookmark_unit,
    pin_unit,
    sessions,
    thread_unit,
    to_message,
)

from .conftest import ME

NONE: frozenset[str] = frozenset()


def _m(ts: float, text: str = "hi", speaker: str = "U1") -> Message:
    return Message(ts=f"{ts:.6f}", text=text, speaker=speaker)


def test_to_message_excludes_noise_and_bots() -> None:
    assert (
        to_message({"ts": "1", "text": "x", "subtype": "channel_join"}, ME, NONE)
        is None
    )
    assert to_message({"ts": "1", "text": "x", "user": "UBOT"}, ME, NONE) is None
    assert to_message({"ts": "1", "text": "x", "bot_id": "BBOT"}, ME, NONE) is None
    assert to_message({"ts": "1", "text": "x", "bot_id": "B9"}, ME, NONE) is None
    assert to_message({"ts": "1", "text": "   ", "user": "U1"}, ME, NONE) is None
    allowed = to_message(
        {"ts": "1", "text": "x", "bot_id": "B9", "subtype": "bot_message"},
        ME,
        frozenset({"B9"}),
    )
    assert allowed is not None
    assert allowed.integration


def test_to_message_anonymizes_and_records_edits() -> None:
    m = to_message(
        {
            "ts": "5.0",
            "text": "ask <@U9|bob> in <#C1|laser> <https://x.org|docs>",
            "user": "U1",
            "edited": {"ts": "9.0"},
        },
        ME,
        NONE,
    )
    assert m is not None
    assert m.text == "ask @member in #laser docs (https://x.org)"
    assert m.activity == 9.0


def test_thread_unit_renders_numbered_speakers() -> None:
    raw = [
        {"ts": "100.0", "text": "Q?", "user": "UA", "thread_ts": "100.0"},
        {"ts": "160.0", "text": "A.", "user": "UB", "thread_ts": "100.0"},
        {"ts": "170.0", "text": "Thanks", "user": "UA", "thread_ts": "100.0"},
    ]
    unit = thread_unit("C1", raw, ME, NONE)
    assert unit is not None
    rendered = unit.render()
    assert "Person 1: Q?" in rendered
    assert "Person 2: A." in rendered
    assert "UA" not in rendered
    assert "UB" not in rendered
    assert unit.key == "100.0"
    assert unit.message_ts == ["100.0", "160.0", "170.0"]
    assert len(unit.content_hash) == 64


def test_thread_to_the_bot_is_excluded() -> None:
    raw = [{"ts": "1.0", "text": "<@UBOT> help", "user": "UA"}]
    assert thread_unit("C1", raw, ME, NONE) is None
    assert thread_unit("C1", [], ME, NONE) is None
    assert thread_unit("C1", [{"ts": "1.0", "text": "", "user": "U"}], ME, NONE) is None


def test_sessions_split_on_gap_and_cap_with_overlap() -> None:
    msgs = [_m(0), _m(60), _m(120), _m(180), _m(3000), _m(100_000)]
    units = sessions("C1", msgs, gap_minutes=30, max_messages=3, overlap=1)
    # cap after 3; the next starts with the last one as overlap; a 47-minute
    # gap (under 4x) overlaps too; a day-long gap does not.
    assert [u.message_ts for u in units] == [
        [msgs[0].ts, msgs[1].ts, msgs[2].ts],
        [msgs[2].ts, msgs[3].ts],
        [msgs[3].ts, msgs[4].ts],
        [msgs[5].ts],
    ]
    assert [u.key for u in units] == [
        m.ts for m in (msgs[0], msgs[3], msgs[4], msgs[5])
    ]
    assert units[1].overlap == 1
    assert units[1].first_ts == msgs[3].ts
    assert sessions("C1", [], 30, 3, 1) == []


def test_quiet() -> None:
    unit = sessions("C1", [_m(1000)], 30, 10, 0)[0]
    assert unit.is_quiet(1000 + 3 * 3600, 3)
    assert not unit.is_quiet(1000 + 3600, 3)


def test_pin_and_bookmark_units() -> None:
    pin = pin_unit("C1", _m(5, "rule"))
    assert pin.key == pin.first_ts.join(["pin:", ""])
    assert pin.boost
    mark = bookmark_unit(
        "C1", {"id": "Bk", "title": "Manual", "link": "https://m", "date_updated": 7}
    )
    assert mark is not None
    assert mark.render() == "Channel bookmark: Manual https://m"
    assert mark.last_activity == 7.0
    assert mark.key == "bookmark:Bk"
    assert bookmark_unit("C1", {"id": "x"}) is None


def test_integration_render_and_edits() -> None:
    unit = sessions(
        "C1",
        [
            Message("1.0", "Open house", "integration", integration=True),
            Message("2.0", "ok", "U", edited_ts="3.0"),
        ],
        30,
        10,
        0,
    )[0]
    assert "Integration: Open house" in unit.render()
    assert "(edited)" in unit.render()
