"""
PostgreSQL staging store for the Slack source.

Stored: channel IDs and names (the channel directory), the message
timestamps that make up each conversation unit, the unit's content hash
and relevance verdict, extracted facts, quarantine entries (channel and
timestamp only), cursors and sync status. Never stored: raw message text,
user IDs or names. Raw text is refetched from Slack whenever a unit is
reprocessed.
"""

from __future__ import annotations

import contextlib
import datetime
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import psycopg
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from alembic.util import CommandError as AlembicCommandError
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .extractor import Fact, fact_id, stale_after

logger = logging.getLogger(__name__)

SOURCE = "slack"


class SchemaError(RuntimeError):
    """The database schema isn't at the revision this version needs."""


def _alembic_config(url: str) -> AlembicConfig:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", "slack2rag:migrations")
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
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", "slack2rag:migrations")
    head = ScriptDirectory.from_config(cfg).get_current_head()
    if head is None:  # pragma: no cover - the package always ships migrations
        raise SchemaError("no migrations found in slack2rag.migrations")
    return head


def known_revision(revision: str) -> bool:
    """Whether this version of slack2rag ships the migration `revision`."""
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", "slack2rag:migrations")
    try:
        return ScriptDirectory.from_config(cfg).get_revision(revision) is not None
    except AlembicCommandError:
        return False


def migrate(url: str) -> None:
    """Bring the database schema up to date (`slack2rag migrate`).

    Serialized with an advisory lock, so processes starting together
    don't race to run the same migration.
    """
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(hashtext('slack2rag:migrate'))")
        try:
            alembic_command.upgrade(_alembic_config(url), "head")
        finally:
            conn.execute("SELECT pg_advisory_unlock(hashtext('slack2rag:migrate'))")
    logger.info("PostgreSQL schema is at revision %s", head_revision())


def _utc(ts: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)


@dataclass(frozen=True)
class UnitRow:
    channel_id: str
    unit_key: str
    kind: str
    message_ts: list[str]
    content_hash: str
    extractor_version: str
    relevant: bool | None
    boost: bool
    dirty: bool
    last_activity: datetime.datetime
    processed_at: datetime.datetime | None = None
    fact_count: int = 0


@dataclass(frozen=True)
class FactRow:
    id: str
    channel_id: str
    unit_key: str
    fact_index: int
    kind: str
    question: str
    answer: str
    confidence: str
    entities: list[str]
    links: list[str]
    topics: list[str]
    tags: list[str]
    channel_name: str
    permalink: str
    said_on: datetime.date
    content_hash: str
    stale_after: datetime.datetime | None
    superseded_by: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime


@dataclass
class FactChanges:
    """What saving a unit did to its facts, by fact ID."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


@dataclass
class RunCounts:
    seen: int = 0
    added: int = 0
    updated: int = 0
    deleted: int = 0
    quarantined: int = 0
    errors: int = 0

    def add(self, changes: FactChanges) -> None:
        self.added += len(changes.added)
        self.updated += len(changes.updated)
        self.deleted += len(changes.removed)


def _fact_row(row: dict[str, Any]) -> FactRow:
    return FactRow(
        id=str(row["id"]),
        channel_id=row["channel_id"],
        unit_key=row["unit_key"],
        fact_index=int(row["fact_index"]),
        kind=row["kind"],
        question=row["question"],
        answer=row["answer"],
        confidence=row["confidence"],
        entities=list(row["entities"]),
        links=list(row["links"]),
        topics=list(row["topics"]),
        tags=list(row["tags"]),
        channel_name=row["channel_name"],
        permalink=row["permalink"],
        said_on=row["said_on"],
        content_hash=row["content_hash"],
        stale_after=row["stale_after"],
        superseded_by=str(row["superseded_by"]) if row["superseded_by"] else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PgStore:
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo

    @contextlib.contextmanager
    def _conn(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        with psycopg.connect(self._conninfo, row_factory=dict_row) as conn:
            yield conn

    # ── schema ───────────────────────────────────────────────────────────

    def schema_revision(self) -> str | None:
        with self._conn() as conn:
            exists = conn.execute(
                "SELECT to_regclass('alembic_version') AS t"
            ).fetchone()
            if exists is None or exists["t"] is None:
                return None
            row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
            return None if row is None else str(row["version_num"])

    def ensure_schema(self) -> None:
        """Migrate the schema to this version's head if it is behind.

        A schema newer than this version (after a downgrade) is refused,
        not touched: SchemaError.
        """
        revision = self.schema_revision()
        head = head_revision()
        if revision == head:
            return
        if revision is not None and not known_revision(revision):
            raise SchemaError(
                f"PostgreSQL schema is at revision {revision}, newer than this "
                f"version of slack2rag knows ({head}); upgrade slack2rag"
            )
        logger.info(
            "Migrating the PostgreSQL schema from revision %s to %s",
            revision or "none",
            head,
        )
        migrate(self._conninfo)

    @contextlib.contextmanager
    def lock(self, name: str = SOURCE) -> Iterator[bool]:
        """A session advisory lock; yields False if another process holds it."""
        with psycopg.connect(self._conninfo) as conn:
            row = conn.execute(
                "SELECT pg_try_advisory_lock(hashtext(%s))", (f"slack2rag:{name}",)
            ).fetchone()
            got = bool(row and row[0])
            try:
                yield got
            finally:
                if got:
                    conn.execute(
                        "SELECT pg_advisory_unlock(hashtext(%s))",
                        (f"slack2rag:{name}",),
                    )

    # ── channels ─────────────────────────────────────────────────────────

    def upsert_channel(
        self, channel_id: str, name: str, topic: str, purpose: str
    ) -> bool:
        """Store a directory entry; True if it is new or changed."""
        with self._conn() as conn:
            old = conn.execute(
                "SELECT name, topic, purpose, is_member FROM slack_channels"
                " WHERE id = %s",
                (channel_id,),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO slack_channels (id, name, topic, purpose, is_member)
                VALUES (%s, %s, %s, %s, TRUE)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, topic = EXCLUDED.topic,
                    purpose = EXCLUDED.purpose, is_member = TRUE,
                    updated_at = NOW()
                """,
                (channel_id, name, topic, purpose),
            )
        return old != {
            "name": name,
            "topic": topic,
            "purpose": purpose,
            "is_member": True,
        }

    def member_channel_ids(self) -> set[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM slack_channels WHERE is_member"
            ).fetchall()
        return {r["id"] for r in rows}

    def channel_names(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, name FROM slack_channels").fetchall()
        return {r["id"]: r["name"] for r in rows}

    def forget_channel(self, channel_id: str) -> list[str]:
        """Delete everything derived from a channel; returns removed fact IDs."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM slack_facts WHERE channel_id = %s", (channel_id,)
            ).fetchall()
            conn.execute(
                "UPDATE slack_facts SET superseded_by = NULL WHERE superseded_by"
                " IN (SELECT id FROM slack_facts WHERE channel_id = %s)",
                (channel_id,),
            )
            conn.execute("DELETE FROM slack_units WHERE channel_id = %s", (channel_id,))
            conn.execute(
                "DELETE FROM slack_quarantine WHERE channel_id = %s", (channel_id,)
            )
            conn.execute(
                "DELETE FROM slack_cursors WHERE channel_id = %s", (channel_id,)
            )
            conn.execute("DELETE FROM slack_channels WHERE id = %s", (channel_id,))
        return [str(r["id"]) for r in rows]

    # ── cursors ──────────────────────────────────────────────────────────

    def get_cursor(self, channel_id: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT latest_ts FROM slack_cursors WHERE channel_id = %s",
                (channel_id,),
            ).fetchone()
        return None if row is None else str(row["latest_ts"])

    def set_cursor(self, channel_id: str, ts: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO slack_cursors (channel_id, latest_ts) VALUES (%s, %s)
                ON CONFLICT (channel_id) DO UPDATE SET
                    latest_ts = EXCLUDED.latest_ts, updated_at = NOW()
                """,
                (channel_id, ts),
            )

    # ── units ────────────────────────────────────────────────────────────

    _UNIT_SELECT = """
        SELECT u.*, (SELECT count(*) FROM slack_facts f
                     WHERE f.channel_id = u.channel_id AND f.unit_key = u.unit_key)
               AS fact_count
        FROM slack_units u
    """

    @staticmethod
    def _unit_row(row: dict[str, Any]) -> UnitRow:
        return UnitRow(
            channel_id=row["channel_id"],
            unit_key=row["unit_key"],
            kind=row["kind"],
            message_ts=list(row["message_ts"]),
            content_hash=row["content_hash"],
            extractor_version=row["extractor_version"],
            relevant=row["relevant"],
            boost=row["boost"],
            dirty=row["dirty"],
            last_activity=row["last_activity"],
            processed_at=row["processed_at"],
            fact_count=int(row["fact_count"]),
        )

    def get_unit(self, channel_id: str, unit_key: str) -> UnitRow | None:
        with self._conn() as conn:
            row = conn.execute(
                self._UNIT_SELECT + " WHERE u.channel_id = %s AND u.unit_key = %s",
                (channel_id, unit_key),
            ).fetchone()
        return None if row is None else self._unit_row(row)

    def units(
        self,
        channel_id: str | None = None,
        kind: str | None = None,
        active_since: datetime.datetime | None = None,
        with_facts_or_active: bool = False,
    ) -> list[UnitRow]:
        """Units matching every filter given.

        with_facts_or_active: units that have facts OR were active since
        *active_since* (for the nightly reconcile).
        """
        where = []
        params: list[Any] = []
        if channel_id is not None:
            where.append("u.channel_id = %s")
            params.append(channel_id)
        if kind is not None:
            where.append("u.kind = %s")
            params.append(kind)
        if with_facts_or_active:
            where.append(
                "(EXISTS (SELECT 1 FROM slack_facts f WHERE f.channel_id ="
                " u.channel_id AND f.unit_key = u.unit_key) OR u.last_activity >= %s)"
            )
            params.append(active_since or datetime.datetime.now(tz=datetime.UTC))
        elif active_since is not None:
            where.append("u.last_activity >= %s")
            params.append(active_since)
        sql = self._UNIT_SELECT + (" WHERE " + " AND ".join(where) if where else "")
        with self._conn() as conn:
            rows = conn.execute(sql + " ORDER BY u.unit_key", params).fetchall()
        return [self._unit_row(r) for r in rows]

    def session_start_covering(self, channel_id: str, ts: str) -> str | None:
        """Key of the earliest stored session that ends at or after *ts*."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT s.unit_key AS k FROM (
                    SELECT unit_key,
                           CASE WHEN kind = 'session' THEN unit_key::numeric END
                               AS start,
                           CASE WHEN kind = 'session'
                               THEN message_ts[array_upper(message_ts, 1)]::numeric
                           END AS finish
                    FROM slack_units WHERE channel_id = %s AND kind = 'session'
                ) s
                WHERE s.finish >= %s::numeric AND s.start < %s::numeric
                ORDER BY s.start LIMIT 1
                """,
                (channel_id, ts, ts),
            ).fetchone()
        return None if row is None else str(row["k"])

    def mark_dirty(self, channel_id: str, unit_key: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE slack_units SET dirty = TRUE"
                " WHERE channel_id = %s AND unit_key = %s",
                (channel_id, unit_key),
            )
            return cur.rowcount > 0

    def units_containing(self, channel_id: str, ts: str) -> list[UnitRow]:
        """Units whose messages include *ts* (or whose key is *ts*)."""
        with self._conn() as conn:
            rows = conn.execute(
                self._UNIT_SELECT
                + " WHERE u.channel_id = %s AND (%s = ANY(u.message_ts)"
                " OR u.unit_key = %s)",
                (channel_id, ts, ts),
            ).fetchall()
        return [self._unit_row(r) for r in rows]

    def clear_dirty(self, channel_id: str, unit_key: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE slack_units SET dirty = FALSE"
                " WHERE channel_id = %s AND unit_key = %s",
                (channel_id, unit_key),
            )

    def ensure_dirty(self, channel_id: str, unit_key: str, kind: str) -> None:
        """Mark a unit for reprocessing, creating a placeholder if it's new."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO slack_units (channel_id, unit_key, kind, content_hash,
                                         dirty, last_activity)
                VALUES (%s, %s, %s, '', TRUE, NOW())
                ON CONFLICT (channel_id, unit_key) DO UPDATE SET dirty = TRUE
                """,
                (channel_id, unit_key, kind),
            )

    def dirty_units(self) -> list[UnitRow]:
        """Units marked by live events, and units still waiting for quiet."""
        with self._conn() as conn:
            rows = conn.execute(
                self._UNIT_SELECT + " WHERE u.dirty OR u.relevant IS NULL"
            ).fetchall()
        return [self._unit_row(r) for r in rows]

    def mark_all_dirty(self) -> int:
        """Reprocess everything (after a prompt or model change)."""
        with self._conn() as conn:
            cur = conn.execute("UPDATE slack_units SET dirty = TRUE")
            return cur.rowcount

    def save_unit(
        self,
        *,
        channel_id: str,
        unit_key: str,
        kind: str,
        message_ts: Sequence[str],
        content_hash: str,
        extractor_version: str,
        relevant: bool | None,
        boost: bool,
        last_activity: float,
        facts: Sequence[Fact] | None,
        channel_name: str = "",
        permalink: str = "",
        said_on: datetime.date | None = None,
        sync_run_id: int | None = None,
    ) -> FactChanges:
        """Upsert a unit; when *facts* is not None, they replace its facts.

        facts=None (not processed yet, still waiting for quiet) leaves any
        existing facts alone.
        """
        changes = FactChanges()
        processed = facts is not None
        with self._conn() as conn, conn.transaction():
            conn.execute(
                """
                INSERT INTO slack_units (channel_id, unit_key, kind, message_ts,
                    content_hash, extractor_version, relevant, boost, dirty,
                    last_activity, processed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s,
                        CASE WHEN %s THEN NOW() END)
                ON CONFLICT (channel_id, unit_key) DO UPDATE SET
                    kind = EXCLUDED.kind, message_ts = EXCLUDED.message_ts,
                    content_hash = EXCLUDED.content_hash,
                    extractor_version = EXCLUDED.extractor_version,
                    relevant = EXCLUDED.relevant, boost = EXCLUDED.boost,
                    dirty = FALSE, last_activity = EXCLUDED.last_activity,
                    processed_at = COALESCE(EXCLUDED.processed_at,
                                            slack_units.processed_at)
                """,
                (
                    channel_id,
                    unit_key,
                    kind,
                    list(message_ts),
                    content_hash,
                    extractor_version,
                    relevant,
                    boost,
                    _utc(last_activity),
                    processed,
                ),
            )
            if facts is None:
                return changes
            old = {
                str(r["id"]): r["content_hash"]
                for r in conn.execute(
                    "SELECT id, content_hash FROM slack_facts"
                    " WHERE channel_id = %s AND unit_key = %s",
                    (channel_id, unit_key),
                ).fetchall()
            }
            day = said_on or datetime.datetime.now(tz=datetime.UTC).date()
            new_ids = []
            for index, fact in enumerate(facts):
                fid = fact_id(channel_id, unit_key, index)
                new_ids.append(fid)
                if fid not in old:
                    changes.added.append(fid)
                elif old[fid] != fact.content_hash:
                    changes.updated.append(fid)
                else:
                    changes.unchanged.append(fid)
                conn.execute(
                    """
                    INSERT INTO slack_facts (id, channel_id, unit_key, fact_index,
                        kind, question, answer, confidence, entities, links,
                        topics, tags, channel_name, permalink, said_on,
                        content_hash, stale_after, sync_run_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        kind = EXCLUDED.kind, question = EXCLUDED.question,
                        answer = EXCLUDED.answer,
                        confidence = EXCLUDED.confidence,
                        entities = EXCLUDED.entities, links = EXCLUDED.links,
                        topics = EXCLUDED.topics, tags = EXCLUDED.tags,
                        channel_name = EXCLUDED.channel_name,
                        permalink = EXCLUDED.permalink,
                        said_on = EXCLUDED.said_on,
                        stale_after = EXCLUDED.stale_after,
                        sync_run_id = EXCLUDED.sync_run_id,
                        superseded_by = CASE
                            WHEN slack_facts.content_hash = EXCLUDED.content_hash
                            THEN slack_facts.superseded_by END,
                        updated_at = CASE
                            WHEN slack_facts.content_hash = EXCLUDED.content_hash
                            THEN slack_facts.updated_at ELSE NOW() END,
                        content_hash = EXCLUDED.content_hash
                    """,
                    (
                        fid,
                        channel_id,
                        unit_key,
                        index,
                        fact.kind,
                        fact.question,
                        fact.answer,
                        fact.confidence,
                        Jsonb(list(fact.entities)),
                        Jsonb(list(fact.links)),
                        Jsonb(list(fact.topics)),
                        Jsonb(list(fact.tags)),
                        channel_name,
                        permalink,
                        day,
                        fact.content_hash,
                        stale_after(fact.kind, day),
                        sync_run_id,
                    ),
                )
            changes.removed = [fid for fid in old if fid not in new_ids]
            self._delete_facts(conn, changes.removed)
        return changes

    @staticmethod
    def _delete_facts(conn: psycopg.Connection[dict[str, Any]], ids: list[str]) -> None:
        if not ids:
            return
        conn.execute(
            "UPDATE slack_facts SET superseded_by = NULL"
            " WHERE superseded_by = ANY(%s::uuid[])",
            (ids,),
        )
        conn.execute("DELETE FROM slack_facts WHERE id = ANY(%s::uuid[])", (ids,))

    def delete_unit(self, channel_id: str, unit_key: str) -> list[str]:
        """Delete a unit and its facts; returns the removed fact IDs."""
        with self._conn() as conn, conn.transaction():
            ids = [
                str(r["id"])
                for r in conn.execute(
                    "SELECT id FROM slack_facts"
                    " WHERE channel_id = %s AND unit_key = %s",
                    (channel_id, unit_key),
                ).fetchall()
            ]
            self._delete_facts(conn, ids)
            conn.execute(
                "DELETE FROM slack_units WHERE channel_id = %s AND unit_key = %s",
                (channel_id, unit_key),
            )
        return ids

    # ── quarantine ───────────────────────────────────────────────────────

    def record_quarantine(self, channel_id: str, ts: str, rules: Sequence[str]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO slack_quarantine (channel_id, ts, rules)
                VALUES (%s, %s, %s)
                ON CONFLICT (channel_id, ts) DO UPDATE SET
                    rules = EXCLUDED.rules, found_at = NOW()
                """,
                (channel_id, ts, sorted(set(rules))),
            )

    def quarantine_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT count(*) AS n FROM slack_quarantine").fetchone()
        return int(row["n"]) if row else 0

    # ── facts ────────────────────────────────────────────────────────────

    def facts(
        self,
        ids: Sequence[str] | None = None,
        unit_of: tuple[str, str] | None = None,
        kinds: Sequence[str] | None = None,
        confidence: str | None = None,
        current_only: bool = False,
        limit: int | None = None,
        random_order: bool = False,
    ) -> list[FactRow]:
        where = []
        params: list[Any] = []
        if ids is not None:
            where.append("id = ANY(%s::uuid[])")
            params.append(list(ids))
        if unit_of is not None:
            where.append("channel_id = %s AND unit_key = %s")
            params.extend(unit_of)
        if kinds is not None:
            where.append("kind = ANY(%s)")
            params.append(list(kinds))
        if confidence is not None:
            where.append("confidence = %s")
            params.append(confidence)
        if current_only:
            where.append(
                "superseded_by IS NULL AND (stale_after IS NULL OR stale_after > NOW())"
            )
        # S608: the WHERE clauses are fixed strings; every value is a parameter.
        sql = "SELECT * FROM slack_facts" + (  # noqa: S608
            " WHERE " + " AND ".join(where) if where else ""
        )
        sql += " ORDER BY random()" if random_order else " ORDER BY said_on, id"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_fact_row(r) for r in rows]

    def delete_facts_before_run(self, run_id: int) -> list[str]:
        """Delete facts no run since *run_id* has seen; returns their IDs."""
        with self._conn() as conn, conn.transaction():
            ids = [
                str(r["id"])
                for r in conn.execute(
                    "SELECT id FROM slack_facts"
                    " WHERE sync_run_id IS NULL OR sync_run_id < %s",
                    (run_id,),
                ).fetchall()
            ]
            self._delete_facts(conn, ids)
        return ids

    def supersede(self, old_id: str, new_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE slack_facts SET superseded_by = %s WHERE id = %s",
                (new_id, old_id),
            )

    def stamp_facts(self, ids: Sequence[str], run_id: int) -> None:
        if not ids:
            return
        with self._conn() as conn:
            conn.execute(
                "UPDATE slack_facts SET sync_run_id = %s WHERE id = ANY(%s::uuid[])",
                (run_id, list(ids)),
            )

    def facts_containing(self, needles: Sequence[str]) -> list[tuple[str, str]]:
        """(needle, ID) for every fact or directory entry containing a needle."""
        found: list[tuple[str, str]] = []
        with self._conn() as conn:
            for needle in needles:
                pattern = (
                    "%"
                    + needle.replace("\\", "\\\\")
                    .replace("%", r"\%")
                    .replace("_", r"\_")
                    + "%"
                )
                rows = conn.execute(
                    "SELECT id FROM slack_facts WHERE question ILIKE %s"
                    " OR answer ILIKE %s OR entities::text ILIKE %s"
                    " OR links::text ILIKE %s",
                    (pattern, pattern, pattern, pattern),
                ).fetchall()
                found.extend((needle, str(r["id"])) for r in rows)
                rows = conn.execute(
                    "SELECT id FROM slack_channels WHERE name ILIKE %s"
                    " OR topic ILIKE %s OR purpose ILIKE %s",
                    (pattern, pattern, pattern),
                ).fetchall()
                found.extend((needle, f"channel:{r['id']}") for r in rows)
        return found

    # ── runs and status ──────────────────────────────────────────────────

    def start_run(self, kind: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO sync_runs (source, kind) VALUES (%s, %s) RETURNING id",
                (SOURCE, kind),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO sync_status (source, last_run_id, last_kind, last_status,
                                         last_started_at)
                VALUES (%s, %s, %s, 'running', NOW())
                ON CONFLICT (source) DO UPDATE SET
                    last_run_id = EXCLUDED.last_run_id,
                    last_kind = EXCLUDED.last_kind, last_status = 'running',
                    last_started_at = NOW(), updated_at = NOW()
                """,
                (SOURCE, row["id"] if row else None, kind),
            )
        if row is None:  # pragma: no cover - RETURNING always returns a row
            raise RuntimeError("could not start a sync run")
        return int(row["id"])

    def finish_run(
        self, run_id: int, counts: RunCounts, status: str, message: str = ""
    ) -> None:
        quarantine = self.quarantine_count()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE sync_runs SET finished_at = NOW(), status = %s, seen = %s,
                    added = %s, updated = %s, deleted = %s, quarantined = %s,
                    errors = %s, message = %s
                WHERE id = %s
                """,
                (
                    status,
                    counts.seen,
                    counts.added,
                    counts.updated,
                    counts.deleted,
                    counts.quarantined,
                    counts.errors,
                    message[:2000],
                    run_id,
                ),
            )
            conn.execute(
                """
                UPDATE sync_status SET last_status = %s, last_finished_at = NOW(),
                    added = %s, updated = %s, deleted = %s, quarantine_count = %s,
                    errors = %s, last_error = %s, updated_at = NOW()
                WHERE source = %s
                """,
                (
                    status,
                    counts.added,
                    counts.updated,
                    counts.deleted,
                    quarantine,
                    counts.errors,
                    message[:2000] if status != "ok" else "",
                    SOURCE,
                ),
            )

    def last_run(self, kind: str, status: str = "ok") -> dict[str, Any] | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM sync_runs WHERE source = %s AND kind = %s"
                " AND status = %s ORDER BY id DESC LIMIT 1",
                (SOURCE, kind, status),
            ).fetchone()

    def status(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM sync_status WHERE source = %s", (SOURCE,)
            ).fetchone()
