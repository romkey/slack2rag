"""
Live Slack events over Socket Mode (no inbound endpoint needed).

Events only mark units for reprocessing; the worker's main thread does the
work. A new reply marks its thread; an edit or a deletion marks every unit
holding that message, so a deleted message's facts go within minutes. When
the bot leaves or is removed from a channel, or the channel is archived or
deleted, the channel is queued to be forgotten.

Event subscriptions (bot events): message.channels, member_left_channel,
channel_left, channel_archive, channel_deleted.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .pg_store import PgStore
from .slack_client import BotIdentity

logger = logging.getLogger(__name__)

_NEW_CONTENT = frozenset({None, "", "thread_broadcast", "file_share", "me_message"})


class LiveEvents:
    def __init__(
        self,
        store: PgStore,
        me: BotIdentity,
        denylist: frozenset[str] = frozenset(),
    ) -> None:
        self.store = store
        self.me = me
        self.denylist = denylist
        self.wake = threading.Event()
        self._lock = threading.Lock()
        self._forget: set[str] = set()
        self._client: Any = None

    # ── connection ────────────────────────────────────────────────────────

    def start(self, app_token: str, web_client: Any) -> None:
        from slack_sdk.socket_mode import SocketModeClient
        from slack_sdk.socket_mode.client import BaseSocketModeClient
        from slack_sdk.socket_mode.request import SocketModeRequest
        from slack_sdk.socket_mode.response import SocketModeResponse

        client = SocketModeClient(app_token=app_token, web_client=web_client)

        def listener(sm: BaseSocketModeClient, req: SocketModeRequest) -> None:
            sm.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id)
            )
            if req.type == "events_api":
                try:
                    self.handle(dict(req.payload.get("event") or {}))
                except Exception:
                    logger.exception("Could not handle a live Slack event")

        client.socket_mode_request_listeners.append(listener)
        client.connect()
        self._client = client
        logger.info("Listening for live Slack events over Socket Mode")

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── events ────────────────────────────────────────────────────────────

    def take_forgotten(self) -> set[str]:
        with self._lock:
            channels, self._forget = self._forget, set()
        return channels

    def handle(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        channel = str(event.get("channel") or "")
        if isinstance(event.get("channel"), dict):  # channel_deleted etc.
            channel = str(event["channel"].get("id") or "")
        if not channel:
            return
        if kind in ("channel_left", "channel_archive", "channel_deleted") or (
            kind == "member_left_channel" and event.get("user") == self.me.user_id
        ):
            with self._lock:
                self._forget.add(channel)
            self.wake.set()
            return
        if kind != "message" or channel in self.denylist:
            return
        if event.get("channel_type") not in (None, "channel"):
            return  # never anything but public channels
        subtype = event.get("subtype")
        if subtype == "message_changed":
            msg = event.get("message") or {}
            self._touch(channel, str(msg.get("ts") or ""), msg.get("thread_ts"))
        elif subtype == "message_deleted":
            previous = event.get("previous_message") or {}
            self._touch(
                channel, str(event.get("deleted_ts") or ""), previous.get("thread_ts")
            )
        elif subtype in _NEW_CONTENT:
            ts, thread_ts = str(event.get("ts") or ""), event.get("thread_ts")
            if thread_ts and thread_ts != ts:
                self.store.ensure_dirty(channel, str(thread_ts), "thread")
                self.wake.set()
            # A new top-level message waits for the hourly catch-up: its
            # session has to go quiet before it is processed anyway.

    def _touch(self, channel: str, ts: str, thread_ts: Any) -> None:
        if not ts:
            return
        for row in self.store.units_containing(channel, ts):
            self.store.mark_dirty(channel, row.unit_key)
        if thread_ts:
            self.store.ensure_dirty(channel, str(thread_ts), "thread")
        self.wake.set()
