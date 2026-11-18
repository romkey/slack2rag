"""Fact staging: units, facts, quarantine, cursors and sync status.

Drops the raw message, user and channel tables: Slack is now mined for
facts, and raw message text, user IDs and names are never stored. Existing
indexed data is disposable, so nothing is carried over.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-17
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS slack_messages")
    op.execute("DROP TABLE IF EXISTS slack_users")
    op.execute("DROP TABLE IF EXISTS slack_channels")
    op.execute(
        """
        CREATE TABLE slack_channels (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT '',
            purpose TEXT NOT NULL DEFAULT '',
            is_member BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE slack_units (
            channel_id TEXT NOT NULL,
            unit_key TEXT NOT NULL,
            kind TEXT NOT NULL,
            message_ts TEXT[] NOT NULL DEFAULT '{}',
            content_hash TEXT NOT NULL,
            extractor_version TEXT NOT NULL DEFAULT '',
            relevant BOOLEAN,
            boost BOOLEAN NOT NULL DEFAULT FALSE,
            dirty BOOLEAN NOT NULL DEFAULT FALSE,
            last_activity TIMESTAMPTZ NOT NULL,
            processed_at TIMESTAMPTZ,
            PRIMARY KEY (channel_id, unit_key)
        )
        """
    )
    op.execute("CREATE INDEX slack_units_dirty ON slack_units (dirty) WHERE dirty")
    op.execute(
        """
        CREATE TABLE slack_facts (
            id UUID PRIMARY KEY,
            channel_id TEXT NOT NULL,
            unit_key TEXT NOT NULL,
            fact_index INTEGER NOT NULL,
            kind TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            confidence TEXT NOT NULL,
            entities JSONB NOT NULL DEFAULT '[]',
            links JSONB NOT NULL DEFAULT '[]',
            topics JSONB NOT NULL DEFAULT '[]',
            tags JSONB NOT NULL DEFAULT '[]',
            channel_name TEXT NOT NULL,
            permalink TEXT NOT NULL DEFAULT '',
            said_on DATE NOT NULL,
            content_hash TEXT NOT NULL,
            stale_after TIMESTAMPTZ,
            superseded_by UUID,
            sync_run_id BIGINT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (channel_id, unit_key, fact_index),
            FOREIGN KEY (channel_id, unit_key)
                REFERENCES slack_units (channel_id, unit_key) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX slack_facts_kind ON slack_facts (kind)")
    op.execute(
        """
        CREATE TABLE slack_quarantine (
            id BIGSERIAL PRIMARY KEY,
            channel_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            rules TEXT[] NOT NULL,
            found_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (channel_id, ts)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE slack_cursors (
            channel_id TEXT PRIMARY KEY,
            latest_ts TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE sync_runs (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            kind TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            status TEXT NOT NULL DEFAULT 'running',
            seen INTEGER NOT NULL DEFAULT 0,
            added INTEGER NOT NULL DEFAULT 0,
            updated INTEGER NOT NULL DEFAULT 0,
            deleted INTEGER NOT NULL DEFAULT 0,
            quarantined INTEGER NOT NULL DEFAULT 0,
            errors INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT ''
        )
        """
    )
    op.execute(
        """
        CREATE TABLE sync_status (
            source TEXT PRIMARY KEY,
            last_run_id BIGINT,
            last_kind TEXT,
            last_status TEXT,
            last_started_at TIMESTAMPTZ,
            last_finished_at TIMESTAMPTZ,
            added INTEGER NOT NULL DEFAULT 0,
            updated INTEGER NOT NULL DEFAULT 0,
            deleted INTEGER NOT NULL DEFAULT 0,
            quarantine_count INTEGER NOT NULL DEFAULT 0,
            errors INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def downgrade() -> None:
    for table in (
        "sync_status",
        "sync_runs",
        "slack_cursors",
        "slack_quarantine",
        "slack_facts",
        "slack_units",
        "slack_channels",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table}")
    # Back to revision 0001's (empty) tables.
    op.execute(
        "CREATE TABLE slack_channels (id TEXT PRIMARY KEY, payload JSONB NOT NULL,"
        " updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
    )
    op.execute(
        "CREATE TABLE slack_users (id TEXT PRIMARY KEY, payload JSONB NOT NULL,"
        " updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
    )
    op.execute(
        "CREATE TABLE slack_messages (channel_id TEXT NOT NULL REFERENCES"
        " slack_channels(id) ON DELETE CASCADE, ts TEXT NOT NULL, payload JSONB"
        " NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
        " PRIMARY KEY (channel_id, ts))"
    )
