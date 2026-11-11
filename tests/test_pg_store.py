"""Integration tests for PostgreSQL persistence (requires TEST_DATABASE_URL)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import psycopg
import pytest

from slack2rag.pg_store import PgStore, SchemaError, head_revision, migrate


@pytest.fixture
def pg_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (PostgreSQL required)")
    return url


def _drop_everything(url: str) -> None:
    with psycopg.connect(url) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS slack_messages, slack_users, slack_channels, alembic_version"
        )
        conn.commit()


@pytest.fixture
def empty_db(pg_url: str) -> str:
    """A database with none of slack2rag's tables (as if never migrated)."""
    _drop_everything(pg_url)
    return pg_url


@pytest.fixture
def pg(empty_db: str) -> PgStore:
    migrate(empty_db)
    return PgStore(empty_db)


def test_check_schema_fails_before_migrate_and_passes_after(empty_db: str) -> None:
    store = PgStore(empty_db)
    assert store.schema_revision() is None
    with pytest.raises(SchemaError, match="revision none"):
        store.check_schema()

    migrate(empty_db)

    assert store.schema_revision() == head_revision()
    store.check_schema()


def test_migrate_is_idempotent(pg: PgStore, pg_url: str) -> None:
    pg.upsert_channel({"id": "C_KEEP"})
    migrate(pg_url)
    pg.check_schema()
    with psycopg.connect(pg_url) as conn:
        assert conn.execute("SELECT count(*) FROM slack_channels").fetchone()[0] == 1


def test_downgrade_to_base_drops_tables(pg: PgStore, pg_url: str) -> None:
    from alembic import command

    from slack2rag.pg_store import _alembic_config

    command.downgrade(_alembic_config(pg_url), "base")
    assert pg.schema_revision() is None
    with psycopg.connect(pg_url) as conn:
        assert (
            conn.execute("SELECT to_regclass('slack_channels')").fetchone()[0] is None
        )


def test_migrate_adopts_tables_created_before_migrations(empty_db: str) -> None:
    """Databases from before Alembic already have the tables; migrate must not fail on them."""
    with psycopg.connect(empty_db) as conn:
        conn.execute(
            "CREATE TABLE slack_channels (id TEXT PRIMARY KEY, payload JSONB NOT NULL)"
        )
        conn.commit()
    migrate(empty_db)
    PgStore(empty_db).check_schema()


def test_upsert_channel_user_message_roundtrip(pg: PgStore) -> None:
    channel = {"id": "C_TEST", "name": "test-channel", "is_channel": True}
    user = {"id": "U_TEST", "name": "alice", "is_bot": False}
    message = {"ts": "1234567890.000001", "user": "U_TEST", "text": "hello"}

    pg.upsert_channel(channel)
    pg.upsert_user(user)
    pg.upsert_message("C_TEST", message)

    with psycopg.connect(pg._conninfo) as conn:
        ch = conn.execute(
            "SELECT payload FROM slack_channels WHERE id = %s",
            ("C_TEST",),
        ).fetchone()
        assert ch is not None
        assert ch[0]["name"] == "test-channel"

        u = conn.execute(
            "SELECT payload FROM slack_users WHERE id = %s",
            ("U_TEST",),
        ).fetchone()
        assert u is not None
        assert u[0]["name"] == "alice"

        m = conn.execute(
            "SELECT payload FROM slack_messages WHERE channel_id = %s AND ts = %s",
            ("C_TEST", "1234567890.000001"),
        ).fetchone()
        assert m is not None
        assert m[0]["text"] == "hello"


def test_persist_channel_messages_and_users_calls_slack(pg: PgStore) -> None:
    slack = MagicMock()
    slack.get_user_record.return_value = {"id": "U1", "name": "bob"}

    channel = {"id": "C1", "name": "general"}
    root = {"ts": "1.0", "user": "U1", "text": "root"}
    replies = [{"ts": "1.1", "user": "U1", "text": "reply"}]

    pg.persist_channel_messages_and_users(slack, channel, root, replies)

    slack.get_user_record.assert_called_with("U1")

    with psycopg.connect(pg._conninfo) as conn:
        row = conn.execute(
            "SELECT count(*) FROM slack_messages WHERE channel_id = %s",
            ("C1",),
        ).fetchone()
        assert row is not None
        assert row[0] == 2


def test_from_url_empty_returns_none() -> None:
    assert PgStore.from_url("") is None
    assert PgStore.from_url("   ") is None


def test_bulk_upsert_users(pg: PgStore) -> None:
    users = [
        {"id": "U_A", "name": "a"},
        {"id": "U_B", "name": "b"},
    ]
    pg.bulk_upsert_users(users)

    with psycopg.connect(pg._conninfo) as conn:
        n = conn.execute(
            "SELECT count(*) FROM slack_users WHERE id IN ('U_A','U_B')"
        ).fetchone()[0]
        assert n == 2
