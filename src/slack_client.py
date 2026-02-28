"""
Thin wrapper around the Slack Web API.

Handles pagination, rate-limit retries (via slack-sdk's built-in handler),
and user-name caching.
"""

import logging
import re
from typing import Dict, Generator, List, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

_MENTION_RE = re.compile(r"<@([A-Z0-9]+)>")
_CHANNEL_RE = re.compile(r"<#([A-Z0-9]+)\|([^>]*)>")
_URL_RE = re.compile(r"<(https?://[^|>]+)(?:\|([^>]*))?>" )
_SPECIAL_RE = re.compile(r"<!(here|channel|everyone)>")


class SlackClient:
    def __init__(self, token: str) -> None:
        # slack-sdk automatically retries on HTTP 429 (rate limited)
        self._client = WebClient(token=token)
        self._user_cache: Dict[str, str] = {}

    # ── channels ──────────────────────────────────────────────────────────────

    def get_public_channels(self, channel_names: Optional[List[str]] = None) -> List[dict]:
        """
        Return public channels the bot can see.

        If *channel_names* is provided, only those channels are returned
        (matching by name or ID).
        """
        channels: List[dict] = []
        cursor = None

        while True:
            kwargs: dict = {
                "types": "public_channel",
                "exclude_archived": True,
                "limit": 200,
            }
            if cursor:
                kwargs["cursor"] = cursor

            try:
                resp = self._client.conversations_list(**kwargs)
            except SlackApiError as exc:
                logger.error("conversations.list error: %s", exc.response["error"])
                break

            for ch in resp.get("channels", []):
                channels.append(ch)

            meta = resp.get("response_metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        if channel_names:
            needle = {n.lstrip("#") for n in channel_names}
            channels = [
                ch for ch in channels
                if ch["name"] in needle or ch["id"] in needle
            ]
            found = {ch["name"] for ch in channels} | {ch["id"] for ch in channels}
            for name in needle:
                if name not in found:
                    logger.warning("Channel %r not found or not accessible", name)

        logger.info("Found %d channels to index", len(channels))
        return channels

    def _join_channel(self, channel_id: str) -> bool:
        """Auto-join a public channel. Requires the channels:join scope."""
        try:
            self._client.conversations_join(channel=channel_id)
            logger.info("Joined channel %s", channel_id)
            return True
        except SlackApiError as exc:
            logger.warning(
                "Could not join channel %s: %s  (add channels:join scope to your Slack app)",
                channel_id,
                exc.response["error"],
            )
            return False

    # ── messages ──────────────────────────────────────────────────────────────

    def get_channel_messages(
        self,
        channel_id: str,
        oldest_ts: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Yield every top-level message in *channel_id*, newest-first.

        Pass *oldest_ts* to fetch only messages newer than that timestamp
        (incremental sync).
        """
        cursor = None

        while True:
            kwargs: dict = {"channel": channel_id, "limit": 200}
            if oldest_ts:
                kwargs["oldest"] = oldest_ts
            if cursor:
                kwargs["cursor"] = cursor

            try:
                resp = self._client.conversations_history(**kwargs)
            except SlackApiError as exc:
                error = exc.response["error"]
                if error == "not_in_channel":
                    if self._join_channel(channel_id):
                        continue  # retry after joining
                    return
                else:
                    logger.error("conversations.history error for %s: %s", channel_id, error)
                return

            for msg in resp.get("messages", []):
                # Skip channel join/leave noise and bot messages without text
                if msg.get("subtype") in ("channel_join", "channel_leave", "channel_purpose", "channel_topic"):
                    continue
                yield msg

            meta = resp.get("response_metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

    def get_thread_replies(self, channel_id: str, thread_ts: str) -> List[dict]:
        """Return all replies in a thread (excludes the root message)."""
        replies: List[dict] = []
        cursor = None

        while True:
            kwargs: dict = {"channel": channel_id, "ts": thread_ts, "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor

            try:
                resp = self._client.conversations_replies(**kwargs)
            except SlackApiError as exc:
                logger.warning(
                    "conversations.replies error for thread %s/%s: %s",
                    channel_id, thread_ts, exc.response["error"],
                )
                return []

            msgs = resp.get("messages", [])
            # First message is the root; skip it when appending
            replies.extend(msgs[1:] if not replies else msgs)

            meta = resp.get("response_metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        return replies

    # ── users ─────────────────────────────────────────────────────────────────

    def get_user_name(self, user_id: str) -> str:
        """Return a display name for *user_id*, with caching."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            resp = self._client.users_info(user=user_id)
            profile = resp["user"].get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or resp["user"].get("name")
                or user_id
            )
        except SlackApiError:
            name = user_id

        self._user_cache[user_id] = name
        return name

    # ── text helpers ──────────────────────────────────────────────────────────

    def resolve_text(self, text: str) -> str:
        """
        Convert Slack mrkdwn syntax into plain readable text:
          <@U123>          → @username
          <#C123|general>  → #general
          <https://…|txt>  → txt (https://…)
          <https://…>      → https://…
          <!here>          → @here
        """
        text = _MENTION_RE.sub(lambda m: f"@{self.get_user_name(m.group(1))}", text)
        text = _CHANNEL_RE.sub(lambda m: f"#{m.group(2)}", text)
        text = _URL_RE.sub(
            lambda m: f"{m.group(2)} ({m.group(1)})" if m.group(2) else m.group(1),
            text,
        )
        text = _SPECIAL_RE.sub(lambda m: f"@{m.group(1)}", text)
        return text
