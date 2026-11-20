"""End-to-end tests of the sync jobs: fake Slack and models, real PostgreSQL,
in-memory Qdrant. Many are the Slack plan's acceptance checks."""

from __future__ import annotations

from typing import Any

import psycopg
from qdrant_client import QdrantClient

from slack2rag.config import PUBLIC_COLLECTION
from slack2rag.evaluation import canaries
from slack2rag.pg_store import PgStore
from slack2rag.pipeline import Pipeline
from slack2rag.vector_store import Point, count_in_collection, source_filter

from .conftest import DAY, Clock, FakeChat, FakeSlack, build

HOUR = 3600.0
OLD = 2 * DAY  # well past QUIET_HOURS


def _points(
    qdrant: QdrantClient, collection: str = "hs_member"
) -> list[dict[str, Any]]:
    records, _ = qdrant.scroll(collection, limit=1000, with_payload=True)
    return [dict(r.payload or {}, id=str(r.id)) for r in records]


def _facts(qdrant: QdrantClient) -> list[dict[str, Any]]:
    return [p for p in _points(qdrant) if p.get("kind") != "channel"]


def _thread(slack: FakeSlack, cid: str, ts: float, *replies: str) -> str:
    parent = slack.post(cid, ts, "Where is the laser key?", user="U1")
    for i, text in enumerate(replies, start=1):
        slack.post(cid, ts + i * 60, text, user=f"U{i + 1}", thread_ts=ts)
    return parent


def test_thread_becomes_member_tier_facts(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    _thread(
        slack,
        "C1",
        clock.now - OLD,
        "FACT location|Where is the laser key?|In the blue box by the laser.",
    )

    counts = pipeline.catchup()

    assert counts.added == 1
    [fact] = _facts(qdrant)
    assert fact["source"] == "slack"
    assert fact["tier"] == "member"
    assert fact["authority"] == 1
    assert fact["kind"] == "location"
    assert (
        fact["text"] == "Q: Where is the laser key?\nA: In the blue box by the laser."
    )
    assert fact["label"].startswith("Slack discussion, ")
    assert fact["url"].startswith("https://ws.slack.com/archives/C1/p")
    assert fact["stale_after"] is not None  # location: 12 months
    assert fact["entities"] == ["laser"]
    assert fact["topics"] == ["lasers"]


def test_no_raw_text_or_user_ids_are_persisted(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    pg_url: str,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    raw = "FACT howto|How to focus?|Use the focus gauge on the laser."
    _thread(slack, "C1", clock.now - OLD, raw)
    slack.post("C1", clock.now - OLD + 9000, "chit chat RAWCHAT", user="USECRET")

    pipeline.catchup()

    with psycopg.connect(pg_url) as conn:
        dump = conn.execute(
            "SELECT string_agg(t::text, ' ') FROM ("
            " SELECT row_to_json(u)::text AS t FROM slack_units u"
            " UNION ALL SELECT row_to_json(f)::text FROM slack_facts f"
            " UNION ALL SELECT row_to_json(c)::text FROM slack_channels c"
            " UNION ALL SELECT row_to_json(q)::text FROM slack_quarantine q) x"
        ).fetchone()
    assert dump is not None
    dump = dump[0]
    everything = dump + str(_points(qdrant))
    for leaked in ("RAWCHAT", "USECRET", "U1", "U2", raw, "Where is the laser key?\n"):
        assert leaked not in everything


def test_unit_waits_for_quiet_then_is_processed(
    pipeline: Pipeline,
    slack: FakeSlack,
    chat: FakeChat,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    _thread(slack, "C1", clock.now - HOUR, "FACT howto|How?|Like this.")

    pipeline.catchup()
    assert chat.calls == []
    assert _facts(qdrant) == []

    clock.now += 4 * HOUR
    pipeline.catchup()
    assert chat.calls == ["relevance", "extraction", "exclusion"]
    assert len(_facts(qdrant)) == 1


def test_unchanged_units_are_not_sent_to_the_model_again(
    pipeline: Pipeline, slack: FakeSlack, chat: FakeChat, clock: Clock
) -> None:
    _thread(slack, "C1", clock.now - OLD, "FACT howto|How?|Like this.")
    slack.post("C1", clock.now - OLD + 9000, "lunch anyone?")

    pipeline.catchup()
    calls = len(chat.calls)
    clock.now += HOUR
    pipeline.catchup()

    assert len(chat.calls) == calls


def test_deleting_a_message_removes_its_facts(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    parent = _thread(slack, "C1", clock.now - OLD, "FACT howto|How?|Like this.")
    pipeline.catchup()
    assert len(_facts(qdrant)) == 1

    reply = next(m["ts"] for m in slack.messages["C1"] if m["ts"] != parent)
    slack.delete("C1", reply)
    clock.now += HOUR
    pipeline.catchup()

    assert _facts(qdrant) == []
    assert store.facts() == []


def test_live_deletion_is_processed_without_waiting(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    slack.post("C1", clock.now - OLD, "FACT policy|Can I stay overnight?|No.")
    pipeline.catchup()
    assert len(_facts(qdrant)) == 1

    [row] = store.units(channel_id="C1")
    slack.delete("C1", row.message_ts[0])
    store.mark_dirty("C1", row.unit_key)
    pipeline.process_dirty()

    assert _facts(qdrant) == []


def test_edit_reprocesses_immediately(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    ts = slack.post("C1", clock.now - OLD, "FACT policy|Food in the shop?|Not allowed.")
    pipeline.catchup()
    slack.edit("C1", ts, "FACT policy|Food in the shop?|Only sealed drinks.", clock.now)
    clock.now += 60
    pipeline.catchup()

    [fact] = _facts(qdrant)
    assert "sealed drinks" in fact["text"]


def test_questions_to_the_bot_and_its_answers_never_become_facts(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    t = clock.now - OLD
    slack.post("C1", t, "<@UBOT> where is the laser key?")
    slack.post("C1", t + 60, "FACT location|Key?|Blue box.", user="UBOT", thread_ts=t)
    slack.post("C1", t + 4000, "FACT location|Key?|Blue box.", user="", bot_id="BBOT")
    slack.post("C1", t + 9000, "<@UBOT> FACT location|Key?|Blue box.")

    pipeline.catchup()

    assert _facts(qdrant) == []


def test_other_bots_are_ignored_unless_allowlisted(
    store: PgStore,
    slack: FakeSlack,
    chat: FakeChat,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    t = clock.now - OLD
    slack.post(
        "C1", t, "FACT announcement|New laser?|Arrives Monday.", user="", bot_id="BX"
    )
    slack.post(
        "C1",
        t + 9000,
        "FACT announcement|Open house?|Saturday.",
        user="",
        bot_id="BANN",
        subtype="bot_message",
        username="announcer",
    )

    build(
        store, slack, chat, qdrant, clock, slack_bot_allowlist=frozenset({"announcer"})
    ).catchup()

    [fact] = _facts(qdrant)
    assert "Saturday" in fact["text"]


def test_join_messages_are_excluded(
    pipeline: Pipeline, slack: FakeSlack, chat: FakeChat, clock: Clock
) -> None:
    slack.post("C1", clock.now - OLD, "FACT x|y|z joined", subtype="channel_join")
    pipeline.catchup()
    assert chat.calls == []


def test_removed_channel_loses_everything(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How?|Like this.")
    pipeline.catchup()
    assert _points(qdrant)

    del slack.channels["C1"]
    pipeline.catchup()

    assert _points(qdrant) == []
    assert store.units() == []
    assert store.channel_names() == {}


def test_denylisted_channel_is_never_read(
    store: PgStore, slack: FakeSlack, chat: FakeChat, qdrant: QdrantClient, clock: Clock
) -> None:
    slack.add_channel("C2", "board")
    slack.post("C2", clock.now - OLD, "FACT policy|Dues?|Secret board stuff.")

    build(
        store, slack, chat, qdrant, clock, slack_channel_denylist=frozenset({"board"})
    ).catchup()

    assert all(cid != "C2" for _, cid in slack.calls)
    assert "C2" not in store.channel_names()


def test_private_channel_canary_is_nowhere(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    # A channel the bot is not in is invisible to it, like a private one.
    slack.messages["G1"] = []
    slack.post("G1", clock.now - OLD, "FACT howto|Canary?|CANARY-PRIVATE-7731")
    slack.post("C1", clock.now - OLD, "FACT howto|How?|Like this.")

    pipeline.catchup()
    pipeline.reconcile()

    assert canaries(store, qdrant, ["CANARY-PRIVATE-7731"]) == []
    assert canaries(store, qdrant, ["Like this"])  # the check does find things


def test_nothing_is_written_to_the_public_collection(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How?|Like this.")
    pipeline.catchup()
    assert count_in_collection(qdrant, PUBLIC_COLLECTION, source_filter("slack")) == 0


def test_channel_directory_is_indexed_and_scrubbed(
    pipeline: Pipeline, qdrant: QdrantClient
) -> None:
    pipeline.catchup()
    [entry] = [p for p in _points(qdrant) if p.get("kind") == "channel"]
    assert entry["title"] == "#laser"
    assert "Laser cutter help" in entry["text"]
    assert "Alice" not in entry["text"]


def test_exclusion_check_and_weakness_tag(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    slack.post(
        "C1",
        clock.now - OLD,
        "FACT history|Who broke it?|EXCLUDE someone broke the lathe."
        " FACT status|Is the door sensor ok?|The sensor is unreliable.",
    )
    pipeline.catchup()

    [fact] = _facts(qdrant)
    assert fact["kind"] == "status"
    assert fact["tags"] == ["weakness"]
    assert fact["stale_after"] is not None  # status: 7 days


def test_secrets_are_quarantined_and_never_indexed(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    ts = slack.post(
        "C1",
        clock.now - OLD,
        "FACT howto|Wifi?|The password: hunter2 works."
        " FACT howto|Lights?|Switch by door.",
    )
    pipeline.catchup()

    assert store.quarantine_count() == 1
    texts = " ".join(p["text"] for p in _facts(qdrant))
    assert "hunter2" not in texts
    with psycopg.connect(store._conninfo) as conn:
        row = conn.execute("SELECT ts, rules FROM slack_quarantine").fetchone()
    assert row == (ts, ["password-assignment"])


def test_names_are_scrubbed_from_facts(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    slack.post(
        "C1", clock.now - OLD, "FACT contact|Who to ask?|Ask Alice Smith in #laser."
    )
    pipeline.catchup()
    [fact] = _facts(qdrant)
    assert "Alice" not in fact["text"]
    assert "[member]" in fact["text"]


def test_similar_newer_fact_supersedes_older(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    text = "FACT location|Where is the laser key?|In the blue box by the laser."
    slack.post("C1", clock.now - 10 * DAY, text)
    slack.post("C1", clock.now - OLD, text + " FACT howto|Unrelated?|Something else.")

    pipeline.catchup()

    facts = {f.id: f for f in store.facts()}
    superseded = [f for f in facts.values() if f.superseded_by]
    assert len(superseded) == 1
    older = superseded[0]
    assert older.superseded_by is not None
    assert facts[older.superseded_by].said_on > older.said_on
    point = next(p for p in _facts(qdrant) if p["id"] == older.id)
    assert point["superseded_by"] == older.superseded_by


def test_pinned_message_and_bookmark_units(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    ts = slack.post("C1", clock.now - 40 * DAY, "FACT policy|Shop hours?|Always open.")
    slack.pinned["C1"] = [ts]
    slack.marks["C1"] = [
        {
            "id": "Bk1",
            "title": "FACT resource|Laser manual?|https://example.org/manual",
            "link": "https://example.org/manual",
            "date_updated": clock.now - OLD,
        }
    ]

    pipeline.catchup()
    kinds = {r.kind for r in store.units(channel_id="C1")}
    assert {"pin", "bookmark"} <= kinds
    assert {f["kind"] for f in _facts(qdrant)} >= {"policy", "resource"}

    slack.pinned["C1"] = []
    slack.marks["C1"] = []
    pipeline.catchup()
    assert {r.kind for r in store.units(channel_id="C1")} & {"pin", "bookmark"} == set()


def test_model_failure_leaves_unit_for_next_run(
    pipeline: Pipeline, slack: FakeSlack, store: PgStore, clock: Clock
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How?|LLMFAIL")
    counts = pipeline.catchup()
    assert counts.errors == 1
    assert store.facts() == []


def test_sessions_group_unthreaded_messages(
    pipeline: Pipeline, slack: FakeSlack, store: PgStore, clock: Clock
) -> None:
    t = clock.now - OLD
    slack.post("C1", t, "Anyone know where the drill bits are?")
    slack.post(
        "C1", t + 300, "FACT location|Drill bits?|Top drawer of the red cabinet."
    )
    slack.post("C1", t + 5 * HOUR, "Unrelated later chat")

    pipeline.catchup()

    sessions = store.units(channel_id="C1", kind="session")
    assert [len(s.message_ts) for s in sessions] == [2, 1]
    status = store.status()
    assert status is not None
    assert status["last_status"] == "ok"


def test_reconcile_deletes_points_no_run_saw(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How?|Like this.")
    pipeline.reconcile()
    pipeline.vectors.upsert(
        [
            Point(
                id="00000000-0000-0000-0000-000000000001",
                dense=[1.0] + [0.0] * 63,
                sparse={"indices": [1], "values": [1.0]},
                payload={"source": "slack", "sync_run_id": 0, "text": "orphan"},
            )
        ]
    )

    clock.now += DAY
    pipeline.reconcile()

    texts = [p["text"] for p in _points(qdrant)]
    assert "orphan" not in texts
    assert any("Like this" in t for t in texts)


def test_reconcile_safety_valve(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    for i in range(4):
        slack.post("C1", clock.now - OLD - i * DAY, f"FACT howto|How {i}?|Way {i}.")
    pipeline.reconcile()
    assert len(_facts(qdrant)) == 4

    # Slack suddenly returns almost nothing (an outage, a scope change...).
    real_history = slack.history
    slack.history = lambda *a, **k: iter(())  # type: ignore[method-assign]
    slack.thread = lambda *a, **k: []  # type: ignore[method-assign]
    clock.now += DAY
    pipeline.reconcile()
    slack.history = real_history  # type: ignore[method-assign]

    status = store.status()
    assert status is not None
    assert status["last_status"] == "alert"
    assert "Safety valve" in status["last_error"]


def test_prompt_change_reprocesses_at_reconcile(
    store: PgStore, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How?|Like this.")
    first = FakeChat("model-a")
    build(store, slack, first, qdrant, clock).reconcile()

    second = FakeChat("model-b")
    clock.now += DAY
    build(store, slack, second, qdrant, clock).reconcile()

    assert "relevance" in second.calls


def test_reconcile_refetches_old_units_outside_the_window(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    t = clock.now - 20 * DAY
    first = slack.post("C1", t, "FACT howto|Glue?|Use wood glue on the lathe.")
    slack.post("C1", t + 60, "FACT location|Clamps?|Wall rack.")
    pin = slack.post("C1", t - 5 * DAY, "FACT policy|Guests?|Members sign them in.")
    slack.pinned["C1"] = [pin]
    slack.marks["C1"] = [
        {
            "id": "Bk",
            "title": "FACT resource|Manual?|https://m.org",
            "link": "https://m.org",
            "date_updated": t,
        }
    ]
    pipeline.reconcile()
    assert len(_facts(qdrant)) == 4

    # Changes far outside the hourly window: only the reconcile sees them.
    slack.delete("C1", first)
    slack.edit("C1", pin, "FACT policy|Guests?|Guests must sign the waiver.", clock.now)
    slack.marks["C1"][0]["title"] = "FACT resource|Manual?|https://m.org now v2"
    clock.now += DAY
    slack.pinned["C1"] = [pin]
    pipeline.catchup()
    pipeline.reconcile()

    texts = sorted(f["text"] for f in _facts(qdrant))
    assert not any("wood glue" in t for t in texts)
    assert any("waiver" in t for t in texts)
    assert any("Clamps" in t for t in texts)
    [session] = store.units(channel_id="C1", kind="session")
    assert session.unit_key != first  # re-keyed to its first remaining message


def test_refetch_of_vanished_pin_and_bookmark(
    pipeline: Pipeline, slack: FakeSlack, store: PgStore, clock: Clock
) -> None:
    pin = slack.post("C1", clock.now - 30 * DAY, "FACT policy|Pets?|No pets.")
    slack.pinned["C1"] = [pin]
    slack.marks["C1"] = [
        {
            "id": "Bk",
            "title": "FACT resource|M?|https://m.org",
            "link": "https://m.org",
            "date_updated": clock.now - OLD,
        }
    ]
    pipeline.catchup()
    rows = {r.kind: r for r in store.units(channel_id="C1")}
    slack.delete("C1", pin)
    slack.marks["C1"] = []
    assert pipeline.refetch(rows["pin"]) is None
    assert pipeline.refetch(rows["bookmark"]) is None


def test_channel_gone_mid_sync_is_forgotten(
    pipeline: Pipeline, slack: FakeSlack, store: PgStore, clock: Clock
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How?|Like this.")
    pipeline.catchup()
    real = slack.history

    def gone(cid: str, *a: object, **k: object):  # type: ignore[no-untyped-def]
        raise __import__("slack2rag.slack_client", fromlist=["x"]).ChannelGoneError(cid)

    slack.history = gone  # type: ignore[method-assign]
    pipeline.catchup()
    slack.history = real  # type: ignore[method-assign]
    assert store.facts() == []


def test_new_reply_to_old_thread_via_live_event(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    t = clock.now - 30 * DAY
    slack.post("C1", t, "Where is the soldering iron?")
    slack.post("C1", t + 60, "No idea", user="U2", thread_ts=t)
    pipeline.catchup()
    assert _facts(qdrant) == []

    slack.post(
        "C1",
        clock.now - 4 * HOUR,
        "FACT location|Iron?|Bench 3.",
        user="U3",
        thread_ts=t,
    )
    store.ensure_dirty("C1", f"{t:.6f}", "thread")
    pipeline.process_dirty()

    [fact] = _facts(qdrant)
    assert "Bench 3" in fact["text"]


# ── rebuilding the index ──────────────────────────────────────────────────


def _wipe_slack_points(qdrant: QdrantClient) -> None:
    qdrant.delete("hs_member", points_selector=source_filter("slack"))


def test_reconcile_writes_back_points_missing_from_qdrant(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How to focus?|Use the gauge.")
    pipeline.catchup()
    before = {p["id"] for p in _points(qdrant)}
    _wipe_slack_points(qdrant)

    clock.now += DAY
    pipeline.reconcile()

    assert {p["id"] for p in _points(qdrant)} == before
    [fact] = _facts(qdrant)
    assert "gauge" in fact["text"]


def test_rebuild_reembeds_stored_facts_without_the_model(
    pipeline: Pipeline,
    slack: FakeSlack,
    chat: FakeChat,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How to focus?|Use the gauge.")
    pipeline.catchup()
    before = {p["id"]: p["text"] for p in _points(qdrant)}
    _wipe_slack_points(qdrant)
    chat.calls.clear()

    pipeline.rebuild()

    assert {p["id"]: p["text"] for p in _points(qdrant)} == before
    assert chat.calls == []  # nothing was sent to the model again
    run = store.last_run("rebuild")
    assert run is not None
    assert {p["sync_run_id"] for p in _points(qdrant)} == {run["id"]}


def test_rebuild_drops_slack_points_staging_does_not_explain(
    pipeline: Pipeline, slack: FakeSlack, qdrant: QdrantClient, clock: Clock
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How to focus?|Use the gauge.")
    pipeline.catchup()
    [fact] = _facts(qdrant)
    orphan = dict(fact, doc_id="C9:1.0:0", channel_id="C9")
    orphan.pop("id")
    pipeline.vectors.upsert(
        [
            Point(
                id="00000000-0000-0000-0000-000000000001",
                dense=pipeline.embedder.embed(["orphan"])[0],
                sparse=pipeline.sparse.encode(["orphan"])[0],
                payload=orphan,
            )
        ]
    )

    pipeline.rebuild()

    assert {p["channel_id"] for p in _points(qdrant)} == {"C1"}


def test_rebuild_reextract_and_reprocess_rerun_the_model(
    pipeline: Pipeline,
    slack: FakeSlack,
    chat: FakeChat,
    store: PgStore,
    qdrant: QdrantClient,
    clock: Clock,
) -> None:
    slack.post("C1", clock.now - OLD, "FACT howto|How to focus?|Use the gauge.")
    pipeline.catchup()
    chat.calls.clear()

    pipeline.rebuild(reextract=True)
    assert "extraction" in chat.calls
    [fact] = _facts(qdrant)
    assert "gauge" in fact["text"]

    chat.calls.clear()
    store.mark_all_dirty()  # slack2rag reprocess
    pipeline.catchup()
    assert "extraction" in chat.calls
    assert len(_facts(qdrant)) == 1
