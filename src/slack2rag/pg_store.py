"""
PostgreSQL persistence for Slack channels, users, and raw messages.

Each row stores the full Slack API object as JSONB so no metadata is lost.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import psycopg
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


class SchemaError(RuntimeError):
    """The database schema isn't at the revision this version needs."""


def _alembic_config(url: str) -> AlembicConfig:
    """Alembic settings for DATABASE_URL, using the migrations in this package."""
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", "slack2rag:migrations")
    # SQLAlchemy picks psycopg 3 only when asked; its default is psycopg2.
    # set_main_option interpolates %, so a %-encoded password needs %%.
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_url(url).replace("%", "%%"))
    return cfg


def sqlalchemy_url(url: str) -> str:
    """postgresql://... (libpq form) to postgresql+psycopg://... for SQLAlchemy."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def head_revision() -> str:
    """The newest migration in this package."""
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", "slack2rag:migrations")
    head = ScriptDirectory.from_config(cfg).get_current_head()
    if head is None:  # pragma: no cover - the package always ships migrations
        raise SchemaError("no migrations found in slack2rag.migrations")
    return head


def migrate(url: str) -> None:
    """Bring the database schema up to date (`slack2rag migrate`)."""
    alembic_command.upgrade(_alembic_config(url), "head")
    logger.info("PostgreSQL schema is at revision %s", head_revision())


class PgStore:
    """Upsert Slack API payloads into PostgreSQL."""

    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo

    @staticmethod
    def from_url(url: str) -> Optional["PgStore"]:
        if not (url or "").strip():
            return None
        return PgStore(url.strip())

    def schema_revision(self) -> str | None:
        """The migration the database is at, or None if it has never been migrated."""
        with psycopg.connect(self._conninfo) as conn:
            exists = conn.execute("SELECT to_regclass('alembic_version')").fetchone()
            if exists is None or exists[0] is None:
                return None
            row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
            return None if row is None else str(row[0])

    def check_schema(self) -> None:
        """Raise SchemaError unless `slack2rag migrate` has brought the schema up to date."""
        revision = self.schema_revision()
        head = head_revision()
        if revision != head:
            raise SchemaError(
                f"PostgreSQL schema is at revision {revision or 'none'}, not "
                f"{head}; run `slack2rag migrate` first"
            )

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
