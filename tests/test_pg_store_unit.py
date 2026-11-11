"""Unit tests for PgStore with psycopg mocked (no database needed)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from psycopg.types.json import Jsonb

from src import pg_store
from src.pg_store import DDL_STATEMENTS, PgStore


@pytest.fixture
def conn(monkeypatch):
    """Patch psycopg.connect; return the mock connection used inside `with`."""
    connection = MagicMock(name="conn")
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    cursor = MagicMock(name="cursor")
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    connection.cursor.return_value = cursor
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(pg_store.psycopg, "connect", connect)
    connection.connect = connect
    connection.cur = cursor
    return connection


def _executed(conn) -> list[tuple[str, tuple]]:
    return [(c.args[0], c.args[1]) for c in conn.execute.call_args_list]


def _jsonb_obj(value) -> object:
    assert isinstance(value, Jsonb)
    return value.obj


# ── from_url ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", ["", "   ", None])
def test_from_url_empty_returns_none(url) -> None:
    assert PgStore.from_url(url) is None


def test_from_url_strips(conn) -> None:
    store = PgStore.from_url("  postgresql://u@h/db \n")
    assert isinstance(store, PgStore)
    store.upsert_channel({"id": "C1"})
    conn.connect.assert_called_once_with("postgresql://u@h/db")


# ── init_schema ──────────────────────────────────────────────────────────────


def test_init_schema_runs_all_ddl_and_commits(conn, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="src.pg_store"):
        PgStore("dsn").init_schema()
    assert [c.args[0] for c in conn.execute.call_args_list] == DDL_STATEMENTS
    conn.commit.assert_called_once()
    assert "PostgreSQL schema ready" in caplog.text


def test_ddl_creates_expected_tables() -> None:
    joined = "\n".join(DDL_STATEMENTS)
    for table in ("slack_channels", "slack_users", "slack_messages"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in joined
    # messages must be created after channels (FK reference)
    assert DDL_STATEMENTS.index(
        next(s for s in DDL_STATEMENTS if "slack_messages" in s)
    ) > DDL_STATEMENTS.index(next(s for s in DDL_STATEMENTS if "slack_channels (" in s))


# ── single upserts ───────────────────────────────────────────────────────────


def test_upsert_channel(conn) -> None:
    ch = {"id": "C1", "name": "eng", "topic": {"value": "t"}}
    PgStore("dsn").upsert_channel(ch)
    [(sql, params)] = _executed(conn)
    assert "INSERT INTO slack_channels" in sql
    assert "ON CONFLICT (id) DO UPDATE" in sql
    assert params[0] == "C1"
    assert _jsonb_obj(params[1]) == ch
    conn.commit.assert_called_once()


def test_upsert_channel_missing_id(conn, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="src.pg_store"):
        PgStore("dsn").upsert_channel({"name": "noid"})
    conn.connect.assert_not_called()
    assert "upsert_channel: missing id" in caplog.text


def test_upsert_user(conn) -> None:
    user = {"id": "U1", "name": "alice"}
    PgStore("dsn").upsert_user(user)
    [(sql, params)] = _executed(conn)
    assert "INSERT INTO slack_users" in sql
    assert params[0] == "U1"
    assert _jsonb_obj(params[1]) == user
    conn.commit.assert_called_once()


def test_upsert_user_missing_id(conn, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="src.pg_store"):
        PgStore("dsn").upsert_user({"id": ""})
    conn.connect.assert_not_called()
    assert "upsert_user: missing id" in caplog.text


def test_upsert_message(conn) -> None:
    msg = {"ts": "1.0", "text": "hi"}
    PgStore("dsn").upsert_message("C1", msg)
    [(sql, params)] = _executed(conn)
    assert "INSERT INTO slack_messages" in sql
    assert "ON CONFLICT (channel_id, ts)" in sql
    assert params[:2] == ("C1", "1.0")
    assert _jsonb_obj(params[2]) == msg
    conn.commit.assert_called_once()


def test_upsert_message_missing_ts(conn, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="src.pg_store"):
        PgStore("dsn").upsert_message("C1", {"text": "no ts"})
    conn.connect.assert_not_called()
    assert "upsert_message: missing ts" in caplog.text


def test_connection_errors_propagate(conn) -> None:
    conn.connect.side_effect = RuntimeError("db down")
    with pytest.raises(RuntimeError, match="db down"):
        PgStore("dsn").upsert_user({"id": "U1"})


# ── bulk_upsert_users ────────────────────────────────────────────────────────


def test_bulk_upsert_users_filters_missing_ids(conn) -> None:
    users = [{"id": "U1"}, {"name": "noid"}, {"id": ""}, {"id": "U2", "x": 1}]
    PgStore("dsn").bulk_upsert_users(users)
    conn.cur.executemany.assert_called_once()
    sql, rows = conn.cur.executemany.call_args.args
    assert "INSERT INTO slack_users" in sql
    assert [r[0] for r in rows] == ["U1", "U2"]
    assert _jsonb_obj(rows[1][1]) == {"id": "U2", "x": 1}
    conn.commit.assert_called_once()


@pytest.mark.parametrize("users", [[], [{"name": "a"}, {"id": None}]])
def test_bulk_upsert_users_noop(conn, users) -> None:
    PgStore("dsn").bulk_upsert_users(users)
    conn.connect.assert_not_called()


# ── persist_channel_messages_and_users ───────────────────────────────────────


def test_persist_channel_messages_and_users(monkeypatch) -> None:
    store = PgStore("dsn")
    calls: list = []
    monkeypatch.setattr(store, "upsert_channel", lambda ch: calls.append(("channel", ch["id"])))
    monkeypatch.setattr(store, "upsert_message", lambda cid, m: calls.append(("msg", cid, m["ts"])))
    monkeypatch.setattr(store, "upsert_user", lambda u: calls.append(("user", u["id"])))

    slack = MagicMock()
    slack.get_user_record.side_effect = lambda uid: None if uid == "U_GONE" else {"id": uid}

    root = {"ts": "1.0", "user": "U1"}
    replies = [
        {"ts": "1.1", "user": "U2"},
        {"ts": "1.2", "user": "U1"},  # duplicate user
        {"ts": "1.3", "user": "U_GONE"},
        {"ts": "1.4", "user": {"id": "weird"}},  # non-string user ignored
        {"ts": "1.5"},
    ]
    store.persist_channel_messages_and_users(slack, {"id": "C1"}, root, replies)

    assert calls[0] == ("channel", "C1")
    assert calls[1:7] == [("msg", "C1", t) for t in ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5")]
    users = sorted(c[1] for c in calls if c[0] == "user")
    assert users == ["U1", "U2"]
    assert sorted(c.args[0] for c in slack.get_user_record.call_args_list) == [
        "U1", "U2", "U_GONE",
    ]


def test_persist_without_channel_id_is_noop(conn) -> None:
    slack = MagicMock()
    PgStore("dsn").persist_channel_messages_and_users(slack, {"name": "x"}, {"ts": "1"}, [])
    conn.connect.assert_not_called()
    slack.get_user_record.assert_not_called()


def test_persist_end_to_end_with_mocked_psycopg(conn) -> None:
    slack = MagicMock()
    slack.get_user_record.return_value = {"id": "U1"}
    PgStore("dsn").persist_channel_messages_and_users(
        slack, {"id": "C1"}, {"ts": "1.0", "user": "U1"}, [],
    )
    tables = [sql.split("INSERT INTO ")[1].split()[0] for sql, _ in _executed(conn)]
    assert tables == ["slack_channels", "slack_messages", "slack_users"]
    assert conn.commit.call_count == 3
