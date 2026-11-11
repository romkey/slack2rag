"""
Persists the latest indexed Slack timestamp per channel so incremental
syncs only fetch new messages.

Crash-safety note
-----------------
The cursor for a channel is only updated *after* all messages in that
channel have been fetched and flushed to Qdrant (see ``sync_channel``
in ``main.py``).  If the process crashes mid-batch, some messages that
were fetched but not yet upserted will be re-fetched on the next run
because the cursor hasn't advanced yet.  This is the intended
behaviour — failing safe by re-fetching is better than advancing the
cursor and silently skipping messages.  Qdrant upserts are idempotent
(same UUID → overwrite with identical data), so re-indexing is
harmless.
"""

import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SyncState:
    def __init__(self, path: str) -> None:
        self._path = path
        self._data: Dict[str, str] = {}
        self._load()

    # ── public ────────────────────────────────────────────────────────────────

    def get_cursor(self, channel_id: str) -> Optional[str]:
        """Return the latest timestamp synced for *channel_id*, or None."""
        return self._data.get(channel_id)

    def set_cursor(self, channel_id: str, ts: str) -> None:
        """Persist *ts* as the latest synced timestamp for *channel_id*."""
        self._data[channel_id] = ts
        self._save()

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path) as fh:
                    self._data = json.load(fh)
                logger.debug("Loaded sync state from %s", self._path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read state file %s: %s", self._path, exc)
                self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        try:
            with open(self._path, "w") as fh:
                json.dump(self._data, fh, indent=2)
        except OSError as exc:
            logger.warning("Could not save state to %s: %s", self._path, exc)
