"""Staging store tests (require TEST_DATABASE_URL)."""

from __future__ import annotations

import datetime

import psycopg
import pytest
from alembic import command as alembic_command

from slack2rag.extractor import Fact, fact_id
from slack2rag.pg_store import (
    FactChanges,
    PgStore,
    RunCounts,
    SchemaError,
    _alembic_config,
    head_revision,
    known_revision,
    migrate,
    sqlalchemy_url,
)

from .conftest import reset_database

F1 = Fact("howto", "How?", "Like this.", "high", ("laser",))
F2 = Fact("location", "Where?", "There.", "medium")


def _save(
    store: PgStore, key: str = "1.0", facts: list[Fact] | None = None, **kw: object
) -> FactChanges:
    args: dict[str, object] = {
        "channel_id": "C1",
        "unit_key": key,
        "kind": "session",
        "message_ts": [key],
        "content_hash": "h",
        "extractor_version": "v",
        "relevant": True,
        "boost": False,
        "last_activity": float(key.removeprefix("pin:")),
        "facts": facts,
        "channel_name": "laser",
        "said_on": datetime.date(2026, 5, 1),
    }
    args.update(kw)
    return store.save_unit(**args)  # type: ignore[arg-type]


def test_sqlalchemy_url() -> None:
    assert sqlalchemy_url("postgresql://a/b") == "postgresql+psycopg://a/b"
    assert sqlalchemy_url("postgres://a/b") == "postgresql+psycopg://a/b"
    assert sqlalchemy_url("sqlite://") == "sqlite://"


def test_migrate_is_idempotent(pg_url: str) -> None:
    reset_database(pg_url)
    store = PgStore(pg_url)
    assert store.schema_revision() is None
    migrate(pg_url)
    migrate(pg_url)
    assert store.schema_revision() == head_revision() == "0002"


def test_ensure_schema_migrates_an_empty_or_older_database(pg_url: str) -> None:
    reset_database(pg_url)
    store = PgStore(pg_url)
    store.ensure_schema()
    assert store.schema_revision() == head_revision()
    store.ensure_schema()  # already current: nothing to do
    reset_database(pg_url)
    alembic_command.upgrade(_alembic_config(pg_url), "0001")
    assert store.schema_revision() == "0001"
    store.ensure_schema()
    assert store.schema_revision() == head_revision()
    assert store.member_channel_ids() == set()


def test_ensure_schema_refuses_a_newer_schema(pg_url: str) -> None:
    reset_database(pg_url)
    store = PgStore(pg_url)
    migrate(pg_url)
    with psycopg.connect(pg_url) as conn:
        conn.execute("UPDATE alembic_version SET version_num = '9999'")
    assert not known_revision("9999")
    with pytest.raises(SchemaError, match="newer than this version"):
        store.ensure_schema()
    assert store.schema_revision() == "9999"


def test_no_column_can_hold_raw_text_or_user_ids(store: PgStore, pg_url: str) -> None:
    with psycopg.connect(pg_url) as conn:
        columns = {
            r[0]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_schema = 'public'"
            ).fetchall()
        }
    assert not columns & {"text", "payload", "user_id", "user", "user_name", "raw"}
    with psycopg.connect(pg_url) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables"
                " WHERE table_schema = 'public'"
            ).fetchall()
        }
    assert "slack_messages" not in tables
    assert "slack_users" not in tables


def test_save_unit_tracks_changes(store: PgStore) -> None:
    changes = _save(store, facts=[F1, F2])
    assert len(changes.added) == 2
    assert not changes.updated

    changes = _save(store, facts=[F1, Fact("location", "Where?", "Here.", "high")])
    assert changes.unchanged == [fact_id("C1", "1.0", 0)]
    assert changes.updated == [fact_id("C1", "1.0", 1)]

    changes = _save(store, facts=[F1])
    assert changes.removed == [fact_id("C1", "1.0", 1)]

    changes = _save(store, facts=None, relevant=None, content_hash="h2")
    assert changes.added == changes.removed == []
    row = store.get_unit("C1", "1.0")
    assert row is not None
    assert row.relevant is None
    assert row.fact_count == 1
    assert [u.unit_key for u in store.dirty_units()] == ["1.0"]

    [fact] = store.facts(unit_of=("C1", "1.0"))
    assert fact.entities == ["laser"]
    assert fact.fact_index == 0
    assert fact.stale_after is not None


def test_supersede_is_cleared_when_the_newer_fact_goes(store: PgStore) -> None:
    _save(store, "1.0", [F1])
    _save(store, "2.0", [F2])
    old, new = fact_id("C1", "1.0", 0), fact_id("C1", "2.0", 0)
    store.supersede(old, new)
    assert store.facts(ids=[old])[0].superseded_by == new
    assert store.facts(current_only=True) == store.facts(ids=[new])
    assert store.delete_unit("C1", "2.0") == [new]
    assert store.facts(ids=[old])[0].superseded_by is None


def test_queries(store: PgStore) -> None:
    _save(store, "100.0", [F1], message_ts=["90.0", "100.0", "150.0"])
    _save(store, "pin:5.0", [F2], kind="pin", message_ts=["5.0"], last_activity=5.0)
    assert store.session_start_covering("C1", "120.0") == "100.0"
    assert store.session_start_covering("C1", "200.0") is None
    assert [u.unit_key for u in store.units_containing("C1", "150.0")] == ["100.0"]
    assert [u.unit_key for u in store.units(kind="pin")] == ["pin:5.0"]
    since = datetime.datetime.fromtimestamp(50, tz=datetime.UTC)
    assert [u.unit_key for u in store.units(active_since=since)] == ["100.0"]
    assert len(store.units(active_since=since, with_facts_or_active=True)) == 2
    assert store.facts(kinds=["howto"], confidence="high")[0].question == "How?"
    assert len(store.facts(limit=1, random_order=True)) == 1

    assert store.mark_dirty("C1", "100.0")
    assert not store.mark_dirty("C1", "nope")
    store.clear_dirty("C1", "100.0")
    store.ensure_dirty("C1", "7.0", "thread")
    assert {u.unit_key for u in store.dirty_units()} == {"7.0"}
    assert store.mark_all_dirty() == 3


def test_channels_cursors_quarantine_and_forgetting(store: PgStore) -> None:
    assert store.upsert_channel("C1", "laser", "t", "p")
    assert not store.upsert_channel("C1", "laser", "t", "p")
    assert store.upsert_channel("C1", "laser", "new topic", "p")
    assert store.member_channel_ids() == {"C1"}
    store.set_cursor("C1", "5.0")
    store.set_cursor("C1", "6.0")
    assert store.get_cursor("C1") == "6.0"
    assert store.get_cursor("C2") is None
    store.record_quarantine("C1", "5.0", ["jwt", "jwt", "aws-access-key"])
    store.record_quarantine("C1", "5.0", ["jwt"])
    assert store.quarantine_count() == 1
    _save(store, "1.0", [F1])
    assert store.facts_containing(["like THIS", "new topic", "absent"]) == [
        ("like THIS", fact_id("C1", "1.0", 0)),
        ("new topic", "channel:C1"),
    ]

    assert store.forget_channel("C1") == [fact_id("C1", "1.0", 0)]
    assert store.units() == []
    assert store.channel_names() == {}
    assert store.get_cursor("C1") is None
    assert store.quarantine_count() == 0


def test_runs_status_and_sweep(store: PgStore) -> None:
    assert store.status() is None
    assert store.last_run("reconcile") is None
    run = store.start_run("reconcile")
    _save(store, "1.0", [F1], sync_run_id=run)
    _save(store, "2.0", [F2], sync_run_id=None)
    store.stamp_facts([], run)
    store.finish_run(run, RunCounts(seen=1, added=2), "ok")
    last = store.last_run("reconcile")
    assert last is not None
    assert last["seen"] == 1
    status = store.status()
    assert status is not None
    assert status["last_status"] == "ok"
    assert status["added"] == 2

    later = store.start_run("reconcile")
    store.stamp_facts([fact_id("C1", "1.0", 0)], later)
    assert store.delete_facts_before_run(later) == [fact_id("C1", "2.0", 0)]
    store.finish_run(later, RunCounts(errors=1), "error", "boom")
    final = store.status()
    assert final is not None
    assert final["last_error"] == "boom"


def test_lock(store: PgStore) -> None:
    with store.lock() as first:
        assert first
        with PgStore(store._conninfo).lock() as second:
            assert not second
    with store.lock() as again:
        assert again
