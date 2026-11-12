"""Unit tests for PgStore with psycopg mocked (no database needed)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import psycopg
import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from psycopg.types.json import Jsonb

from slack2rag import pg_store
from slack2rag.pg_store import PgStore, SchemaError


@pytest.fixture
def conn(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch psycopg.connect; return the mock connection used inside `with`."""
    connection = MagicMock(name="conn")
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    cursor = MagicMock(name="cursor")
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    connection.cursor.return_value = cursor
    connect = MagicMock(return_value=connection)
    monkeypatch.setattr(psycopg, "connect", connect)
    connection.connect = connect
    connection.cur = cursor
    return connection


def _executed(conn: MagicMock) -> list[tuple[str, tuple[object, ...]]]:
    return [(c.args[0], c.args[1]) for c in conn.execute.call_args_list]


def _jsonb_obj(value: object) -> object:
    assert isinstance(value, Jsonb)
    return value.obj


# ── from_url ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", ["", "   ", None])
def test_from_url_empty_returns_none(url: str | None) -> None:
    assert PgStore.from_url(url) is None


def test_from_url_strips(conn: MagicMock) -> None:
    store = PgStore.from_url("  postgresql://u@h/db \n")
    assert isinstance(store, PgStore)
    store.upsert_channel({"id": "C1"})
    conn.connect.assert_called_once_with("postgresql://u@h/db")


# ── migrations ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        ("postgres://u@h/db", "postgresql+psycopg://u@h/db"),
        ("postgresql+psycopg://u@h/db", "postgresql+psycopg://u@h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_sqlalchemy_url(url: str, expected: str) -> None:
    assert pg_store.sqlalchemy_url(url) == expected


def test_alembic_config_uses_package_migrations_and_escapes_percent() -> None:
    cfg = pg_store._alembic_config("postgresql://u:p%40ss@h/db")
    assert cfg.get_main_option("script_location") == "slack2rag:migrations"
    # Alembic interpolates % on read, so the stored value round-trips unchanged.
    assert cfg.get_main_option("sqlalchemy.url") == "postgresql+psycopg://u:p%40ss@h/db"


def test_head_revision_is_initial_migration() -> None:
    assert pg_store.head_revision() == "0001"


def test_migrate_upgrades_to_head(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    upgrade = MagicMock()
    monkeypatch.setattr(alembic_command, "upgrade", upgrade)
    with caplog.at_level(logging.INFO, logger="slack2rag.pg_store"):
        pg_store.migrate("postgresql://u@h/db")
    [(cfg, target)] = [c.args for c in upgrade.call_args_list]
    assert target == "head"
    assert cfg.get_main_option("sqlalchemy.url") == "postgresql+psycopg://u@h/db"
    assert "schema is at revision 0001" in caplog.text


def test_migrations_render_offline_sql(capsys: pytest.CaptureFixture[str]) -> None:
    """Run env.py and 0001 in Alembic's offline mode: SQL is printed, no connection."""
    cfg = pg_store._alembic_config("postgresql://u@h/db")
    alembic_command.upgrade(cfg, "head", sql=True)
    out = capsys.readouterr().out
    for table in ("slack_channels", "slack_users", "slack_messages"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in out
    assert out.index("slack_channels (") < out.index("slack_messages (")  # FK order
    assert "INSERT INTO alembic_version (version_num) VALUES ('0001')" in out


def test_migrations_refuse_to_run_without_url() -> None:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", "slack2rag:migrations")
    with pytest.raises(RuntimeError, match=r"sqlalchemy\.url is not set"):
        alembic_command.upgrade(cfg, "head", sql=True)


# ── schema_revision / check_schema ───────────────────────────────────────────


def _results(conn: MagicMock, *rows: tuple[str | None] | None) -> None:
    """Make successive conn.execute(...).fetchone() calls return *rows*."""
    conn.execute.return_value.fetchone.side_effect = list(rows)


@pytest.mark.parametrize("regclass_row", [None, (None,)])
def test_schema_revision_none_when_never_migrated(
    conn: MagicMock, regclass_row: tuple[None] | None
) -> None:
    _results(conn, regclass_row)
    assert PgStore("dsn").schema_revision() is None
    assert conn.execute.call_count == 1
    assert "to_regclass('alembic_version')" in conn.execute.call_args.args[0]


def test_schema_revision_none_when_version_table_empty(conn: MagicMock) -> None:
    _results(conn, ("alembic_version",), None)
    assert PgStore("dsn").schema_revision() is None


def test_schema_revision_returns_version(conn: MagicMock) -> None:
    _results(conn, ("alembic_version",), ("0001",))
    assert PgStore("dsn").schema_revision() == "0001"
    assert "SELECT version_num FROM alembic_version" in conn.execute.call_args.args[0]


def test_check_schema_passes_at_head(monkeypatch: pytest.MonkeyPatch) -> None:
    store = PgStore("dsn")
    monkeypatch.setattr(store, "schema_revision", lambda: pg_store.head_revision())
    store.check_schema()


@pytest.mark.parametrize(("revision", "shown"), [(None, "none"), ("0000", "0000")])
def test_check_schema_raises_when_not_at_head(
    monkeypatch: pytest.MonkeyPatch, revision: str | None, shown: str
) -> None:
    store = PgStore("dsn")
    monkeypatch.setattr(store, "schema_revision", lambda: revision)
    with pytest.raises(
        SchemaError, match=f"at revision {shown}, not 0001; run `slack2rag migrate`"
    ):
        store.check_schema()


# ── single upserts ───────────────────────────────────────────────────────────


def test_upsert_channel(conn: MagicMock) -> None:
    ch = {"id": "C1", "name": "eng", "topic": {"value": "t"}}
    PgStore("dsn").upsert_channel(ch)
    [(sql, params)] = _executed(conn)
    assert "INSERT INTO slack_channels" in sql
    assert "ON CONFLICT (id) DO UPDATE" in sql
    assert params[0] == "C1"
    assert _jsonb_obj(params[1]) == ch
    conn.commit.assert_called_once()


def test_upsert_channel_missing_id(
    conn: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="slack2rag.pg_store"):
        PgStore("dsn").upsert_channel({"name": "noid"})
    conn.connect.assert_not_called()
    assert "upsert_channel: missing id" in caplog.text


def test_upsert_user(conn: MagicMock) -> None:
    user = {"id": "U1", "name": "alice"}
    PgStore("dsn").upsert_user(user)
    [(sql, params)] = _executed(conn)
    assert "INSERT INTO slack_users" in sql
    assert params[0] == "U1"
    assert _jsonb_obj(params[1]) == user
    conn.commit.assert_called_once()


def test_upsert_user_missing_id(
    conn: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="slack2rag.pg_store"):
        PgStore("dsn").upsert_user({"id": ""})
    conn.connect.assert_not_called()
    assert "upsert_user: missing id" in caplog.text


def test_upsert_message(conn: MagicMock) -> None:
    msg = {"ts": "1.0", "text": "hi"}
    PgStore("dsn").upsert_message("C1", msg)
    [(sql, params)] = _executed(conn)
    assert "INSERT INTO slack_messages" in sql
    assert "ON CONFLICT (channel_id, ts)" in sql
    assert params[:2] == ("C1", "1.0")
    assert _jsonb_obj(params[2]) == msg
    conn.commit.assert_called_once()


def test_upsert_message_missing_ts(
    conn: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="slack2rag.pg_store"):
        PgStore("dsn").upsert_message("C1", {"text": "no ts"})
    conn.connect.assert_not_called()
    assert "upsert_message: missing ts" in caplog.text


def test_connection_errors_propagate(conn: MagicMock) -> None:
    conn.connect.side_effect = RuntimeError("db down")
    with pytest.raises(RuntimeError, match="db down"):
        PgStore("dsn").upsert_user({"id": "U1"})


# ── bulk_upsert_users ────────────────────────────────────────────────────────


def test_bulk_upsert_users_filters_missing_ids(conn: MagicMock) -> None:
    users: list[dict[str, object]] = [
        {"id": "U1"},
        {"name": "noid"},
        {"id": ""},
        {"id": "U2", "x": 1},
    ]
    PgStore("dsn").bulk_upsert_users(users)
    conn.cur.executemany.assert_called_once()
    sql, rows = conn.cur.executemany.call_args.args
    assert "INSERT INTO slack_users" in sql
    assert [r[0] for r in rows] == ["U1", "U2"]
    assert _jsonb_obj(rows[1][1]) == {"id": "U2", "x": 1}
    conn.commit.assert_called_once()


@pytest.mark.parametrize("users", [[], [{"name": "a"}, {"id": None}]])
def test_bulk_upsert_users_noop(
    conn: MagicMock, users: list[dict[str, object]]
) -> None:
    PgStore("dsn").bulk_upsert_users(users)
    conn.connect.assert_not_called()


# ── persist_channel_messages_and_users ───────────────────────────────────────


def test_persist_channel_messages_and_users(monkeypatch: pytest.MonkeyPatch) -> None:
    store = PgStore("dsn")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        store, "upsert_channel", lambda ch: calls.append(("channel", ch["id"]))
    )
    monkeypatch.setattr(
        store, "upsert_message", lambda cid, m: calls.append(("msg", cid, m["ts"]))
    )
    monkeypatch.setattr(store, "upsert_user", lambda u: calls.append(("user", u["id"])))

    slack = MagicMock()
    slack.get_user_record.side_effect = lambda uid: (
        None if uid == "U_GONE" else {"id": uid}
    )

    root = {"ts": "1.0", "user": "U1"}
    replies: list[dict[str, object]] = [
        {"ts": "1.1", "user": "U2"},
        {"ts": "1.2", "user": "U1"},  # duplicate user
        {"ts": "1.3", "user": "U_GONE"},
        {"ts": "1.4", "user": {"id": "weird"}},  # non-string user ignored
        {"ts": "1.5"},
    ]
    store.persist_channel_messages_and_users(slack, {"id": "C1"}, root, replies)

    assert calls[0] == ("channel", "C1")
    assert calls[1:7] == [
        ("msg", "C1", t) for t in ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5")
    ]
    users = sorted(c[1] for c in calls if c[0] == "user")
    assert users == ["U1", "U2"]
    assert sorted(c.args[0] for c in slack.get_user_record.call_args_list) == [
        "U1",
        "U2",
        "U_GONE",
    ]


def test_persist_without_channel_id_is_noop(conn: MagicMock) -> None:
    slack = MagicMock()
    PgStore("dsn").persist_channel_messages_and_users(
        slack, {"name": "x"}, {"ts": "1"}, []
    )
    conn.connect.assert_not_called()
    slack.get_user_record.assert_not_called()


def test_persist_end_to_end_with_mocked_psycopg(conn: MagicMock) -> None:
    slack = MagicMock()
    slack.get_user_record.return_value = {"id": "U1"}
    PgStore("dsn").persist_channel_messages_and_users(
        slack,
        {"id": "C1"},
        {"ts": "1.0", "user": "U1"},
        [],
    )
    tables = [sql.split("INSERT INTO ")[1].split()[0] for sql, _ in _executed(conn)]
    assert tables == ["slack_channels", "slack_messages", "slack_users"]
    assert conn.commit.call_count == 3
