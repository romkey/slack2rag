"""Processor (message -> Document) tests."""

from __future__ import annotations

import datetime
import uuid
from collections import Counter

import pytest

from src import processor
from src.processor import (
    MAX_CHARS,
    OVERLAP,
    REPLY_PREVIEW_CHARS,
    ChannelStats,
    Document,
    build_channel_summary,
    build_documents,
    build_team_summary,
    build_user_summary,
    build_workspace_summary,
)

# 2024-01-25T14:30:00Z
TS = "1706193000.000100"
CHANNEL = {"id": "C1", "name": "general"}
NAMES = {"U1": "alice", "U2": "bob", "U3": "carol"}


def _identity(text: str) -> str:
    return text


def _name(uid: str) -> str:
    return NAMES.get(uid, uid)


def _now() -> float:
    return datetime.datetime.now(datetime.timezone.utc).timestamp()


def _build(root: dict, replies: list[dict] | None = None, **kwargs) -> list[Document]:
    return build_documents(root, replies or [], CHANNEL, _identity, _name, **kwargs)


# ── helpers ───────────────────────────────────────────────────────────────────


def test_ts_conversions() -> None:
    assert processor._ts_to_date(TS) == "2024-01-25"
    assert processor._ts_to_datetime(TS) == "2024-01-25T14:30:00Z"
    assert processor._ts_to_iso_date(TS) == "2024-01-25"


@pytest.mark.parametrize("bad", ["", "not-a-number"])
def test_ts_conversions_invalid_return_empty(bad: str) -> None:
    assert processor._ts_to_date(bad) == ""
    assert processor._ts_to_datetime(bad) == ""
    assert processor._ts_to_iso_date(bad) == ""


@pytest.mark.xfail(
    raises=OverflowError,
    strict=True,
    reason="BUG: _ts_to_* helpers do not catch OverflowError for huge timestamps",
)
def test_ts_conversions_out_of_range_timestamp() -> None:
    assert processor._ts_to_date("1e30") == ""


def test_ts_to_iso_date_handles_none() -> None:
    assert processor._ts_to_iso_date(None) == ""  # type: ignore[arg-type]


def test_make_id_is_stable_and_distinct() -> None:
    a = processor._make_id("C1", TS, 0)
    assert a == processor._make_id("C1", TS, 0)
    assert uuid.UUID(a).version == 5
    assert a != processor._make_id("C1", TS, 1)
    assert a != processor._make_id("C2", TS, 0)
    assert processor._make_summary_id("x") == processor._make_summary_id("x")
    assert processor._make_summary_id("x") != processor._make_summary_id("y")


def test_chunk_short_text_is_single_chunk() -> None:
    assert processor._chunk("hello") == ["hello"]
    exact = "a" * MAX_CHARS
    assert processor._chunk(exact) == [exact]


def test_chunk_splits_on_line_boundaries_with_overlap() -> None:
    lines = [f"line {i:03d} " + "x" * 90 for i in range(40)]  # ~99 chars each
    text = "\n".join(lines)
    chunks = processor._chunk(text)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= MAX_CHARS
        # every chunk is made of whole original lines
        for ln in c.split("\n"):
            assert ln in lines
    # consecutive chunks share trailing/leading lines within the overlap budget
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_last = prev.split("\n")[-1]
        assert nxt.split("\n")[0] in prev.split("\n")
        assert prev_last in nxt.split("\n")
        shared = set(prev.split("\n")) & set(nxt.split("\n"))
        assert sum(len(s) + 1 for s in shared) <= OVERLAP
    # all content is covered
    covered = set()
    for c in chunks:
        covered.update(c.split("\n"))
    assert covered == set(lines)


def test_chunk_hard_splits_single_long_line() -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(4000))
    chunks = processor._chunk(text)
    step = MAX_CHARS - OVERLAP
    assert chunks[0] == text[:MAX_CHARS]
    assert chunks[1] == text[step : step + MAX_CHARS]
    assert all(len(c) <= MAX_CHARS for c in chunks)
    # last chunk reaches the end of the text
    assert text.endswith(chunks[-1])
    assert len(chunks) == len(range(0, len(text), step))


def test_chunk_custom_limits() -> None:
    chunks = processor._chunk("aaaa\nbbbb\ncccc", max_chars=9, overlap=0)
    assert chunks == ["aaaa\nbbbb", "cccc"]


def test_format_message() -> None:
    assert processor._format_message({"text": " hi "}, "alice", _identity) == "[alice]: hi"
    assert processor._format_message({"text": "   "}, "alice", _identity) == ""
    assert processor._format_message({}, "alice", _identity) == ""
    assert (
        processor._format_message({"text": "<@U2>"}, "alice", lambda t: t.replace("<@U2>", "@bob"))
        == "[alice]: @bob"
    )


# ── build_documents ───────────────────────────────────────────────────────────


def test_standalone_message_document_fields() -> None:
    docs = _build(
        {"ts": TS, "user": "U1", "text": "hello world"},
        permalink="https://slack/p1",
        channel_topic="chit chat",
        reaction_count=2,
        reactions=["+1", "tada"],
        attachments=["file.pdf"],
    )
    assert len(docs) == 1
    d = docs[0]
    assert d.text == "[alice]: hello world"
    assert d.id == processor._make_id("C1", TS, 0)
    assert d.channel_id == "C1"
    assert d.channel_name == "general"
    assert d.ts == TS
    assert d.thread_ts == TS  # falls back to ts for standalone messages
    assert d.date == "2024-01-25"
    assert d.datetime_str == "2024-01-25T14:30:00Z"
    assert d.user_id == "U1"
    assert d.user_name == "alice"
    assert d.reply_count == 0
    assert d.doc_type == "message"

    payload = d.payload()
    assert payload["datetime"] == "2024-01-25T14:30:00Z"
    assert payload["permalink"] == "https://slack/p1"
    assert payload["channel_topic"] == "chit chat"
    assert payload["reaction_count"] == 2
    assert payload["reactions"] == ["+1", "tada"]
    assert payload["attachments"] == ["file.pdf"]
    assert payload["source_system"] == "slack"
    assert payload["text"] == d.text
    assert set(payload) == {
        "channel_id", "channel_name", "ts", "date", "datetime", "user_id",
        "user_name", "thread_ts", "reply_count", "text", "permalink",
        "channel_topic", "reaction_count", "reactions", "attachments",
        "doc_type", "source_system",
    }


def test_defaults_for_missing_user_and_channel_name() -> None:
    docs = build_documents(
        {"ts": TS, "text": "hi"}, [], {"id": "C7"}, _identity, _name,
    )
    assert docs[0].user_id == "unknown"
    assert docs[0].user_name == "unknown"
    assert docs[0].channel_name == "C7"
    assert docs[0].reactions == []
    assert docs[0].attachments == []


def test_explicit_thread_ts_is_preserved() -> None:
    docs = _build({"ts": TS, "thread_ts": "1.0", "user": "U1", "text": "hi"})
    assert docs[0].thread_ts == "1.0"


def test_empty_messages_produce_no_documents() -> None:
    assert _build({"ts": TS, "user": "U1", "text": ""}) == []
    assert _build({"ts": TS, "user": "U1", "text": "  "}, [{"user": "U2", "text": ""}]) == []


def test_resolve_text_is_applied() -> None:
    docs = build_documents(
        {"ts": TS, "user": "U1", "text": "ping <@U2>"},
        [], CHANNEL, lambda t: t.replace("<@U2>", "@bob"), _name,
    )
    assert docs[0].text == "[alice]: ping @bob"


def test_min_message_length_filters_short_standalone() -> None:
    root = {"ts": TS, "user": "U1", "text": "ok"}  # "[alice]: ok" == 11 chars
    assert _build(root, min_message_length=50) == []
    assert len(_build(root, min_message_length=11)) == 1
    assert len(_build(root, min_message_length=0)) == 1


def test_min_message_length_does_not_filter_threads() -> None:
    docs = _build(
        {"ts": TS, "user": "U1", "text": "ok"},
        [{"user": "U2", "text": "k"}],
        min_message_length=1000,
    )
    assert len(docs) == 1
    assert docs[0].text == "[alice]: ok\n[bob]: k"
    assert docs[0].reply_count == 1


def test_thread_with_empty_root_still_indexes_replies() -> None:
    docs = _build(
        {"ts": TS, "user": "U1", "text": ""},
        [{"user": "U2", "text": "a"}, {"user": "U3", "text": "b"}, {"text": "c"}],
    )
    # No thread summary because root text is empty.
    assert [d.doc_type for d in docs] == ["message"]
    assert docs[0].text == "[bob]: a\n[carol]: b\n[unknown]: c"
    assert docs[0].user_name == "alice"
    assert docs[0].reply_count == 3


@pytest.mark.parametrize(
    ("threshold", "count", "boosted"),
    [(0, 100, False), (3, 2, False), (3, 3, True), (3, 10, True)],
)
def test_reaction_boost(threshold: int, count: int, boosted: bool) -> None:
    docs = _build(
        {"ts": TS, "user": "U1", "text": "great idea"},
        reaction_count=count,
        reaction_boost_threshold=threshold,
    )
    expected = "[alice]: great idea"
    if boosted:
        expected = "[highlighted by team] " + expected
    assert docs[0].text == expected


def test_long_message_is_chunked_with_part_labels() -> None:
    replies = [{"user": "U2", "text": f"reply {i} " + "z" * 100} for i in range(2)]
    root = {"ts": TS, "user": "U1", "text": "\n".join("w" * 100 for _ in range(30))}
    docs = _build(root, replies)
    msgs = [d for d in docs if d.doc_type == "message"]
    assert len(msgs) > 1
    n = len(msgs)
    for i, d in enumerate(msgs):
        assert d.text.startswith(f"(part {i + 1}/{n}) ")
        assert d.id == processor._make_id("C1", TS, i)
        assert d.ts == TS
    assert len({d.id for d in msgs}) == n
    # only 2 replies -> no thread summary
    assert all(d.doc_type == "message" for d in docs)


def test_thread_summary_created_for_three_or_more_replies() -> None:
    long_reply = "r" * (REPLY_PREVIEW_CHARS + 50)
    replies = [
        {"user": "U2", "text": "short reply"},
        {"user": "U3", "text": long_reply},
        {"user": "U2", "text": "third"},
    ]
    docs = _build(
        {"ts": TS, "user": "U1", "text": "what should we do?"},
        replies,
        permalink="p",
        channel_topic="t",
        reaction_count=1,
        reactions=["eyes"],
        attachments=["a.txt"],
    )
    assert [d.doc_type for d in docs] == ["message", "thread_summary"]
    summary = docs[1]
    lines = summary.text.split("\n")
    assert lines[0] == "Thread in #general (3 replies, 2024-01-25):"
    assert lines[1] == "[alice]: what should we do?"
    assert lines[2] == "[bob]: short reply"
    full_long = f"[carol]: {long_reply}"
    assert lines[3] == full_long[:REPLY_PREVIEW_CHARS] + "…"
    assert lines[4] == "[bob]: third"
    assert summary.id == processor._make_summary_id(f"thread:C1:{TS}")
    assert summary.id != docs[0].id
    assert summary.reply_count == 3
    assert summary.permalink == "p"
    assert summary.channel_topic == "t"
    assert summary.reactions == ["eyes"]
    assert summary.attachments == ["a.txt"]
    assert summary.thread_ts == TS


def test_thread_summary_requires_three_replies() -> None:
    replies = [{"user": "U2", "text": "a"}, {"user": "U3", "text": "b"}]
    docs = _build({"ts": TS, "user": "U1", "text": "q"}, replies)
    assert [d.doc_type for d in docs] == ["message"]


def test_thread_summary_skipped_when_all_replies_empty() -> None:
    replies = [{"user": "U2", "text": ""}] * 3
    docs = _build({"ts": TS, "user": "U1", "text": "q"}, replies)
    assert [d.doc_type for d in docs] == ["message"]
    assert docs[0].reply_count == 3


def test_thread_summary_is_truncated_to_max_chars() -> None:
    replies = [{"user": "U2", "text": "y" * 200} for _ in range(5)]
    root = {"ts": TS, "user": "U1", "text": "x" * 2000}
    docs = _build(root, replies)
    summary = docs[-1]
    assert summary.doc_type == "thread_summary"
    assert len(summary.text) == MAX_CHARS
    assert summary.text.endswith("…")


# ── ChannelStats ──────────────────────────────────────────────────────────────


def test_channel_stats_top_terms_and_posters() -> None:
    cs = ChannelStats(
        token_counts=Counter({"deploy": 5, "bug": 3, "lunch": 1}),
        user_counts=Counter({"U1": 4, "U2": 7}),
    )
    assert cs.top_terms(2) == ["deploy", "bug"]
    assert cs.top_posters() == ["U2", "U1"]
    assert cs.top_posters(1, get_name=_name) == ["bob"]


def test_channel_stats_cadence_empty() -> None:
    assert ChannelStats().cadence() == {
        "messages_last_7d": 0,
        "messages_last_30d": 0,
        "last_message_date": "",
    }


def test_channel_stats_cadence_counts_windows() -> None:
    now = _now()
    cs = ChannelStats(timestamps=[now - 3600, now - 10 * 86400, now - 60 * 86400])
    c = cs.cadence()
    assert c["messages_last_7d"] == 1
    assert c["messages_last_30d"] == 2
    expected = datetime.datetime.fromtimestamp(now - 3600, tz=datetime.timezone.utc)
    assert c["last_message_date"] == expected.strftime("%Y-%m-%d")


def test_channel_stats_round_trip_and_truncation() -> None:
    cs = ChannelStats(
        token_counts=Counter({f"t{i}": i + 1 for i in range(600)}),
        user_counts=Counter({"U1": 2}),
        timestamps=[float(i) for i in range(10_050)],
    )
    data = cs.to_dict()
    assert len(data["token_counts"]) == 500
    assert "t599" in data["token_counts"] and "t0" not in data["token_counts"]
    assert data["user_counts"] == {"U1": 2}
    assert len(data["timestamps"]) == 10_000
    assert data["timestamps"][0] == 50.0

    restored = ChannelStats.from_dict(data)
    assert restored.token_counts["t599"] == 600
    assert restored.user_counts == Counter({"U1": 2})
    assert restored.timestamps == data["timestamps"]


def test_channel_stats_from_dict_empty() -> None:
    cs = ChannelStats.from_dict({})
    assert cs.token_counts == Counter()
    assert cs.user_counts == Counter()
    assert cs.timestamps == []


def test_channel_stats_merge_adds_and_prunes_old_timestamps() -> None:
    now = _now()
    a = ChannelStats(
        token_counts=Counter({"x": 1}),
        user_counts=Counter({"U1": 1}),
        timestamps=[now - 100 * 86400, now - 86400],
    )
    b = ChannelStats(
        token_counts=Counter({"x": 2, "y": 1}),
        user_counts=Counter({"U2": 3}),
        timestamps=[now - 10],
    )
    a.merge(b)
    assert a.token_counts == Counter({"x": 3, "y": 1})
    assert a.user_counts == Counter({"U1": 1, "U2": 3})
    assert a.timestamps == [now - 86400, now - 10]


# ── build_channel_summary ─────────────────────────────────────────────────────


def test_channel_summary_minimal() -> None:
    doc = build_channel_summary({"id": "C5"}, 0)
    assert doc.text == (
        "Channel #C5 information:\n"
        "Indexed messages: 0\n"
        "This channel has no indexed messages yet."
    )
    assert doc.doc_type == "channel_summary"
    assert doc.id == processor._make_summary_id("channel:C5")
    assert doc.channel_name == "C5"
    assert doc.ts == "0"
    assert doc.thread_ts is None
    assert doc.channel_topic == ""


def test_channel_summary_full_with_stats() -> None:
    now = _now()
    stats = ChannelStats(
        token_counts=Counter({"deploy": 3, "infra": 2}),
        user_counts=Counter({"U1": 5, "U2": 1}),
        timestamps=[now - 60],
    )
    channel = {
        "id": "C1",
        "name": "eng",
        "topic": {"value": "Engineering"},
        "purpose": {"value": "Build things"},
        "num_members": 12,
        "created": 1706193000,
    }
    doc = build_channel_summary(channel, 1234, stats, get_user_name=_name)
    lines = doc.text.split("\n")
    assert lines[0] == "Channel #eng information:"
    assert "Topic: Engineering" in lines
    assert "Purpose: Build things" in lines
    assert "Members: 12" in lines
    assert "Created: 2024-01-25" in lines
    assert "Indexed messages: 1,234" in lines
    assert "This channel has no indexed messages yet." not in lines
    today = datetime.datetime.fromtimestamp(now - 60, tz=datetime.timezone.utc)
    assert f"Last message: {today:%Y-%m-%d}" in lines
    assert "Messages in last 7 days: 1" in lines
    assert "Messages in last 30 days: 1" in lines
    assert "Common topics: deploy, infra" in lines
    assert "Most active members: alice, bob" in lines
    assert doc.channel_topic == "Engineering"


def test_channel_summary_purpose_same_as_topic_and_empty_stats() -> None:
    channel = {
        "id": "C1",
        "name": "eng",
        "topic": {"value": ""},
        "purpose": {"value": "Build"},
    }
    doc = build_channel_summary(channel, 3, ChannelStats())
    lines = doc.text.split("\n")
    assert "Purpose: Build" in lines
    assert not any(ln.startswith("Last message") for ln in lines)
    assert not any(ln.startswith("Common topics") for ln in lines)
    assert not any(ln.startswith("Most active") for ln in lines)
    assert "Messages in last 7 days: 0" in lines
    assert doc.channel_topic == "Build"  # falls back to purpose

    same = build_channel_summary(
        {"id": "C1", "topic": {"value": "X"}, "purpose": {"value": "X"}}, 1
    )
    assert "Topic: X" in same.text
    assert "Purpose:" not in same.text


# ── build_workspace_summary ───────────────────────────────────────────────────


def test_workspace_summary_sorts_by_count() -> None:
    channels = [
        {"id": "C1", "name": "quiet", "num_members": 3},
        {"id": "C2", "name": "busy", "topic": {"value": "Main"}, "purpose": {"value": "P"}},
        {"id": "C3", "purpose": {"value": "Only purpose"}},
    ]
    doc = build_workspace_summary(channels, {"C1": 5, "C2": 2000})
    lines = doc.text.split("\n")
    assert lines[0] == (
        "This Slack workspace has 3 indexed public channels with 2,005 total messages."
    )
    assert lines[1:3] == ["", "Channels:"]
    assert lines[3] == "  #busy (2,000 messages) — Main"
    assert lines[4] == "  #quiet (5 messages, 3 members) — no description"
    assert lines[5] == "  #C3 (0 messages) — Only purpose"
    assert doc.doc_type == "workspace_summary"
    assert doc.id == processor._make_summary_id("workspace")
    assert doc.channel_id == "workspace"


def test_workspace_summary_singular() -> None:
    doc = build_workspace_summary([{"id": "C1", "name": "a"}], {"C1": 1})
    assert doc.text.startswith("This Slack workspace has 1 indexed public channel with 1 total")


# ── build_user_summary ────────────────────────────────────────────────────────


def test_user_summary_full_profile() -> None:
    profile = {
        "user_id": "U1",
        "display_name": "Ally",
        "real_name": "Alice Smith",
        "username": "alice",
        "title": "Engineer",
        "pronouns": "she/her",
        "timezone_label": "Pacific Standard Time",
        "timezone": "America/Los_Angeles",
        "status_text": "On vacation",
        "status_emoji": ":palm_tree:",
        "is_owner": True,
        "is_admin": True,
        "is_bot": True,
    }
    doc = build_user_summary(profile, active_channels=["eng", "random"])
    assert doc.text.split("\n") == [
        "Team member: Ally",
        "Full name: Alice Smith",
        "Username: @alice",
        "Title: Engineer",
        "Pronouns: she/her",
        "Timezone: Pacific Standard Time (America/Los_Angeles)",
        "Status: On vacation :palm_tree:",
        "Role: workspace owner",
        "This is a bot account.",
        "Active in: #eng, #random",
    ]
    assert doc.user_id == "U1"
    assert doc.user_name == "Ally"
    assert doc.doc_type == "user_summary"
    assert doc.id == processor._make_summary_id("user:U1")


def test_user_summary_admin_and_no_extras() -> None:
    doc = build_user_summary(
        {
            "user_id": "U2",
            "real_name": "Bob",
            "timezone_label": "UTC",
            "status_text": "busy",
            "is_admin": True,
        }
    )
    assert doc.text.split("\n") == [
        "Team member: Bob",
        "Timezone: UTC",
        "Status: busy",
        "Role: workspace admin",
    ]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ({"user_id": "U3", "username": "carol"}, "carol"),
        ({"user_id": "U4"}, "U4"),
    ],
)
def test_user_summary_display_fallbacks(profile: dict, expected: str) -> None:
    doc = build_user_summary(profile)
    assert doc.text.split("\n")[0] == f"Team member: {expected}"
    assert doc.user_name == expected


# ── build_team_summary ────────────────────────────────────────────────────────


def test_team_summary_with_bots_and_deleted() -> None:
    profiles = [
        {"user_id": "U1", "real_name": "zed", "title": "CTO", "pronouns": "he/him"},
        {"user_id": "U2", "display_name": "amy"},
        {"user_id": "U3", "username": "ghost", "deleted": True},
        {"user_id": "U4", "real_name": "Botty", "is_bot": True},
        {"user_id": "U5", "real_name": "Abot", "is_bot": True},
        {"user_id": "U6", "real_name": "Dead bot", "is_bot": True, "deleted": True},
        {"user_id": "U7"},
    ]
    doc = build_team_summary(profiles)
    lines = doc.text.split("\n")
    assert lines[0] == "This Slack workspace has 3 active team members and 2 bots."
    assert lines[1:3] == ["", "Team members:"]
    # sorted by (real_name or display_name or "") lowercase; U7 sorts first ("")
    assert lines[3:6] == ["  U7", "  amy", "  zed (CTO) [he/him]"]
    assert lines[6:] == ["", "Bots:", "  Abot", "  Botty"]
    assert "ghost" not in doc.text
    assert "Dead bot" not in doc.text
    assert doc.doc_type == "team_summary"
    assert doc.id == processor._make_summary_id("team")


def test_team_summary_singular_no_bots() -> None:
    doc = build_team_summary([{"user_id": "U1", "real_name": "Solo"}])
    assert doc.text.split("\n") == [
        "This Slack workspace has 1 active team member.",
        "",
        "Team members:",
        "  Solo",
    ]


def test_team_summary_single_bot() -> None:
    doc = build_team_summary([{"user_id": "B1", "username": "bot", "is_bot": True}])
    assert doc.text.split("\n")[0] == (
        "This Slack workspace has 0 active team members and 1 bot."
    )
    assert doc.text.split("\n")[-1] == "  bot"
