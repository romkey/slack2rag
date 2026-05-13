"""Integration tests for PostgreSQL persistence (requires TEST_DATABASE_URL)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from src.pg_store import PgStore


@pytest.fixture
def pg_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (PostgreSQL required)")
    return url


@pytest.fixture
def pg(pg_url: str) -> PgStore:
    store = PgStore(pg_url)
    store.init_schema()
    return store


def test_upsert_channel_user_message_roundtrip(pg: PgStore) -> None:
    channel = {"id": "C_TEST", "name": "test-channel", "is_channel": True}
    user = {"id": "U_TEST", "name": "alice", "is_bot": False}
    message = {"ts": "1234567890.000001", "user": "U_TEST", "text": "hello"}

    pg.upsert_channel(channel)
    pg.upsert_user(user)
    pg.upsert_message("C_TEST", message)

    import psycopg

    with psycopg.connect(pg._conninfo) as conn:
        ch = conn.execute(
            "SELECT payload FROM slack_channels WHERE id = %s", ("C_TEST",),
        ).fetchone()
        assert ch is not None
        assert ch[0]["name"] == "test-channel"

        u = conn.execute(
            "SELECT payload FROM slack_users WHERE id = %s", ("U_TEST",),
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

    import psycopg

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

    import psycopg

    with psycopg.connect(pg._conninfo) as conn:
        n = conn.execute("SELECT count(*) FROM slack_users WHERE id IN ('U_A','U_B')").fetchone()[0]
        assert n == 2
