"""
PostgreSQL persistence for Slack channels, users, and raw messages.

Each row stores the full Slack API object as JSONB so no metadata is lost.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import psycopg
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS slack_channels (
        id TEXT PRIMARY KEY,
        payload JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS slack_users (
        id TEXT PRIMARY KEY,
        payload JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS slack_messages (
        channel_id TEXT NOT NULL REFERENCES slack_channels(id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        payload JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (channel_id, ts)
    )
    """,
]


class PgStore:
    """Upsert Slack API payloads into PostgreSQL."""

    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo

    @staticmethod
    def from_url(url: str) -> Optional["PgStore"]:
        if not (url or "").strip():
            return None
        return PgStore(url.strip())

    def init_schema(self) -> None:
        with psycopg.connect(self._conninfo) as conn:
            for stmt in DDL_STATEMENTS:
                conn.execute(stmt)
            conn.commit()
        logger.info("PostgreSQL schema ready")

    def upsert_channel(self, channel: dict[str, Any]) -> None:
        cid = channel.get("id")
        if not cid:
            logger.warning("upsert_channel: missing id, skipping")
            return
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(
                """
                INSERT INTO slack_channels (id, payload)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                """,
                (cid, Jsonb(channel)),
            )
            conn.commit()

    def upsert_user(self, user: dict[str, Any]) -> None:
        uid = user.get("id")
        if not uid:
            logger.warning("upsert_user: missing id, skipping")
            return
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(
                """
                INSERT INTO slack_users (id, payload)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                """,
                (uid, Jsonb(user)),
            )
            conn.commit()

    def upsert_message(self, channel_id: str, message: dict[str, Any]) -> None:
        ts = message.get("ts")
        if not ts:
            logger.warning("upsert_message: missing ts, skipping")
            return
        with psycopg.connect(self._conninfo) as conn:
            conn.execute(
                """
                INSERT INTO slack_messages (channel_id, ts, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (channel_id, ts) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                """,
                (channel_id, ts, Jsonb(message)),
            )
            conn.commit()

    def bulk_upsert_users(self, users: list[dict[str, Any]]) -> None:
        if not users:
            return
        rows = [(u["id"], Jsonb(u)) for u in users if u.get("id")]
        if not rows:
            return
        with psycopg.connect(self._conninfo) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO slack_users (id, payload)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        payload = EXCLUDED.payload,
                        updated_at = NOW()
                    """,
                    rows,
                )
            conn.commit()

    def persist_channel_messages_and_users(
        self,
        slack: Any,
        channel: dict[str, Any],
        root_msg: dict[str, Any],
        replies: list[dict[str, Any]],
    ) -> None:
        """Store channel row, each message JSON, and any referenced users."""
        cid = channel.get("id")
        if not cid:
            return
        self.upsert_channel(channel)
        self.upsert_message(cid, root_msg)
        for reply in replies:
            self.upsert_message(cid, reply)
        user_ids: set[str] = set()
        for m in (root_msg, *replies):
            uid = m.get("user")
            if uid and isinstance(uid, str):
                user_ids.add(uid)
        for uid in user_ids:
            rec = slack.get_user_record(uid)
            if rec:
                self.upsert_user(rec)
