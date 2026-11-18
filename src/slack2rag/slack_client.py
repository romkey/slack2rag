"""
Thin wrapper around the Slack Web API, read with a bot token.

The bot reads only public channels it has been invited to: channels come
from users.conversations (the bot's own memberships), and it never joins a
channel itself. Raw message text is returned to the caller for processing
in memory; nothing here stores it.

Scopes used: channels:read, channels:history, pins:read, bookmarks:read,
and optionally users:read (member names, held in memory for scrubbing).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

_MENTION_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]*)?>")
_CHANNEL_RE = re.compile(r"<#([A-Z0-9]+)(?:\|([^>]*))?>")
_URL_RE = re.compile(r"<((?:https?|mailto):[^|>]+)(?:\|([^>]*))?>")
_SPECIAL_RE = re.compile(r"<!(here|channel|everyone)(?:\|[^>]*)?>")
_SUBTEAM_RE = re.compile(r"<!subteam\^[A-Z0-9]+(?:\|([^>]*))?>")

# Slack's Tier 3 methods allow ~50 requests a minute.
DEFAULT_API_PAUSE = 1.2

# A raw Slack Web API object; its shape varies by method, and slack_sdk
# returns it untyped, so the values are Any.
SlackObject = dict[str, Any]

# Errors meaning the bot can no longer read a channel.
_GONE = frozenset(
    {"not_in_channel", "channel_not_found", "is_archived", "missing_scope_channel"}
)


class ChannelGoneError(RuntimeError):
    """The bot can no longer read this channel (removed, archived, deleted)."""


@dataclass(frozen=True)
class BotIdentity:
    workspace_url: str
    user_id: str  # the bot's user ID (U...), for spotting questions to it
    bot_id: str  # the bot's bot ID (B...), on messages it posts


def resolve_text(text: str) -> str:
    """Slack mrkdwn to plain text, with every person mention anonymized."""
    text = _MENTION_RE.sub("@member", text)
    text = _SUBTEAM_RE.sub(lambda m: "@" + (m.group(1) or "group").lstrip("@"), text)
    text = _CHANNEL_RE.sub(lambda m: f"#{m.group(2) or m.group(1)}", text)
    text = _URL_RE.sub(
        lambda m: (
            f"{m.group(2)} ({m.group(1)})"
            if m.group(2) and m.group(2) != m.group(1)
            else m.group(1)
        ),
        text,
    )
    return _SPECIAL_RE.sub(lambda m: f"@{m.group(1)}", text)


class SlackClient:
    def __init__(
        self,
        token: str,
        api_pause: float = DEFAULT_API_PAUSE,
        client: WebClient | None = None,
    ) -> None:
        self._client = client or WebClient(token=token)
        self._api_pause = api_pause
        self._identity: BotIdentity | None = None

    @property
    def web_client(self) -> WebClient:
        return self._client

    def _pace(self) -> None:
        if self._api_pause > 0:
            time.sleep(self._api_pause)

    def _paged(
        self,
        method: Callable[..., Any],
        key: str,
        what: str,
        **kwargs: Any,
    ) -> Generator[SlackObject]:
        """Yield items under *key* from every page of a cursor-paginated method.

        Raises ChannelGoneError for errors meaning the channel can't be read, and
        SlackApiError for anything else.
        """
        cursor = None
        while True:
            if cursor:
                kwargs["cursor"] = cursor
            try:
                resp = method(**kwargs)
            except SlackApiError as exc:
                error = exc.response.get("error", "")
                if error in _GONE:
                    raise ChannelGoneError(f"{what}: {error}") from exc
                raise
            finally:
                self._pace()
            yield from resp.get(key, []) or []
            meta: SlackObject = resp.get("response_metadata") or {}
            cursor = meta.get("next_cursor")
            if not cursor:
                return

    # ── workspace ─────────────────────────────────────────────────────────────

    def identify(self) -> BotIdentity:
        """auth.test, once: the workspace URL and who the bot is."""
        if self._identity is None:
            resp = self._client.auth_test()
            self._pace()
            self._identity = BotIdentity(
                workspace_url=str(resp.get("url", "")).rstrip("/"),
                user_id=str(resp.get("user_id", "")),
                bot_id=str(resp.get("bot_id", "")),
            )
            logger.info("Workspace URL: %s", self._identity.workspace_url)
        return self._identity

    def permalink(self, channel_id: str, ts: str, thread_ts: str | None = None) -> str:
        base = self.identify().workspace_url
        if not base:
            return ""
        link = f"{base}/archives/{channel_id}/p{ts.replace('.', '')}"
        if thread_ts and thread_ts != ts:
            link += f"?thread_ts={thread_ts}&cid={channel_id}"
        return link

    def member_names(self) -> set[str]:
        """Display and real names of everyone in the workspace, for scrubbing.

        Kept in memory only. Without the users:read scope this returns an
        empty set and scrubbing relies on the other rules.
        """
        names: set[str] = set()
        try:
            for user in self._paged(
                self._client.users_list, "members", "users.list", limit=200
            ):
                profile: SlackObject = user.get("profile") or {}
                for key in ("display_name", "real_name", "first_name", "last_name"):
                    value = str(profile.get(key) or "").strip()
                    if value:
                        names.add(value)
                if user.get("name"):
                    names.add(str(user["name"]))
        except SlackApiError as exc:
            logger.warning(
                "users.list failed (%s); member names won't be scrubbed by name",
                exc.response.get("error", exc),
            )
        return names

    # ── channels ──────────────────────────────────────────────────────────────

    def member_channels(self) -> list[SlackObject]:
        """Public, unarchived channels the bot has been invited to."""
        channels = list(
            self._paged(
                self._client.users_conversations,
                "channels",
                "users.conversations",
                types="public_channel",
                exclude_archived=True,
                limit=200,
            )
        )
        logger.info("The bot is in %d public channels", len(channels))
        return channels

    def channel_info(self, channel_id: str) -> SlackObject:
        try:
            resp = self._client.conversations_info(channel=channel_id)
        except SlackApiError as exc:
            if exc.response.get("error", "") in _GONE:
                raise ChannelGoneError(f"conversations.info: {channel_id}") from exc
            raise
        finally:
            self._pace()
        channel: SlackObject = resp.get("channel") or {}
        return channel

    # ── messages ──────────────────────────────────────────────────────────────

    def history(
        self,
        channel_id: str,
        oldest: str | None = None,
        latest: str | None = None,
        inclusive: bool = False,
    ) -> Generator[SlackObject]:
        """Top-level messages in a channel, newest first."""
        kwargs: dict[str, Any] = {"channel": channel_id, "limit": 200}
        if oldest:
            kwargs["oldest"] = oldest
        if latest:
            kwargs["latest"] = latest
        if inclusive:
            kwargs["inclusive"] = True
        yield from self._paged(
            self._client.conversations_history,
            "messages",
            f"conversations.history {channel_id}",
            **kwargs,
        )

    def thread(self, channel_id: str, thread_ts: str) -> list[SlackObject]:
        """Every message in a thread, the parent first. Empty if it's gone."""
        try:
            return list(
                self._paged(
                    self._client.conversations_replies,
                    "messages",
                    f"conversations.replies {channel_id}",
                    channel=channel_id,
                    ts=thread_ts,
                    limit=200,
                )
            )
        except SlackApiError as exc:
            if exc.response.get("error", "") == "thread_not_found":
                return []
            raise

    def pins(self, channel_id: str) -> list[SlackObject]:
        """Pinned messages in a channel."""
        try:
            resp = self._client.pins_list(channel=channel_id)
        except SlackApiError as exc:
            if exc.response.get("error", "") in _GONE:
                raise ChannelGoneError(f"pins.list: {channel_id}") from exc
            logger.warning(
                "pins.list failed for %s: %s", channel_id, exc.response.get("error")
            )
            return []
        finally:
            self._pace()
        items: list[SlackObject] = resp.get("items", []) or []
        return [
            item["message"]
            for item in items
            if item.get("type") == "message" and item.get("message")
        ]

    def bookmarks(self, channel_id: str) -> list[SlackObject]:
        """Bookmarks (links pinned to the channel header)."""
        try:
            resp = self._client.bookmarks_list(channel_id=channel_id)
        except SlackApiError as exc:
            if exc.response.get("error", "") in _GONE:
                raise ChannelGoneError(f"bookmarks.list: {channel_id}") from exc
            logger.warning(
                "bookmarks.list failed for %s: %s",
                channel_id,
                exc.response.get("error"),
            )
            return []
        finally:
            self._pace()
        bookmarks: list[SlackObject] = resp.get("bookmarks", []) or []
        return bookmarks
