"""
Persists the latest indexed Slack timestamp per channel so incremental
syncs only fetch new messages.
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
