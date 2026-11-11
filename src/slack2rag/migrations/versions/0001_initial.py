"""Initial schema: Slack channels, users and messages as JSONB.

IF NOT EXISTS, because before migrations existed slack2rag created these
tables itself at startup; databases from then upgrade without changes.

Revision ID: 0001
Revises:
Create Date: 2026-11-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS slack_channels (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS slack_users (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS slack_messages (
            channel_id TEXT NOT NULL REFERENCES slack_channels(id) ON DELETE CASCADE,
            ts TEXT NOT NULL,
            payload JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (channel_id, ts)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS slack_messages")
    op.execute("DROP TABLE IF EXISTS slack_users")
    op.execute("DROP TABLE IF EXISTS slack_channels")
