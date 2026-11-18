"""
Conversation units: the pieces of Slack that facts are extracted from.

* A thread (parent plus replies) is one unit, keyed by the parent's ts.
* Unthreaded channel messages are grouped into sessions: consecutive
  messages less than SESSION_GAP_MINUTES apart, at most
  SESSION_MAX_MESSAGES long. A session starts with the last few messages of
  the one before it when they were split by the length cap or by a short
  gap, so a question and its answer aren't separated. A session is keyed by
  the ts of its first message of its own (not counting the overlap).
* A pinned message that isn't a thread parent is a unit of its own, and a
  channel bookmark is a unit; both get a relevance boost.

Messages are held in memory only. Speakers are numbered per unit
("Person 1", "Person 2") so the model can follow the conversation without
ever seeing a name or user ID.
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .slack_client import BotIdentity, resolve_text

UnitKind = Literal["thread", "session", "pin", "bookmark"]

# Message subtypes that carry content. Everything else (joins, leaves, topic
# changes, pins, reminders, tombstones...) is excluded.
_CONTENT_SUBTYPES = frozenset(
    {None, "", "thread_broadcast", "file_share", "me_message"}
)

# A short gap still overlaps sessions; a longer one starts fresh.
_OVERLAP_GAP_FACTOR = 4


@dataclass(frozen=True)
class Message:
    ts: str
    text: str  # resolved, mentions anonymized
    speaker: str  # opaque, only for numbering speakers within a unit
    thread_ts: str | None = None
    reply_count: int = 0
    latest_reply: str | None = None
    edited_ts: str | None = None
    integration: bool = False  # an allowlisted bot or integration

    @property
    def activity(self) -> float:
        """The most recent thing that happened to this message."""
        return max(
            float(self.ts),
            float(self.latest_reply or 0),
            float(self.edited_ts or 0),
        )


def _bot_names(msg: dict[str, Any]) -> set[str]:
    profile = msg.get("bot_profile") or {}
    return {
        str(v)
        for v in (msg.get("bot_id"), msg.get("username"), profile.get("name"))
        if v
    }


def mentions_bot(msg: dict[str, Any], me: BotIdentity) -> bool:
    return bool(me.user_id) and f"<@{me.user_id}" in str(msg.get("text", ""))


def is_own(msg: dict[str, Any], me: BotIdentity) -> bool:
    return bool(
        (me.user_id and msg.get("user") == me.user_id)
        or (me.bot_id and msg.get("bot_id") == me.bot_id)
    )


def to_message(
    msg: dict[str, Any], me: BotIdentity, bot_allowlist: frozenset[str]
) -> Message | None:
    """A Slack message as a Message, or None if it must be excluded."""
    if msg.get("subtype") not in _CONTENT_SUBTYPES and msg.get("subtype") != (
        "bot_message"
    ):
        return None
    if is_own(msg, me):
        return None
    is_bot = bool(msg.get("bot_id")) or msg.get("subtype") == "bot_message"
    if is_bot and not (_bot_names(msg) & bot_allowlist):
        return None
    text = resolve_text(str(msg.get("text") or "")).strip()
    if not text:
        return None
    edited = msg.get("edited") or {}
    return Message(
        ts=str(msg["ts"]),
        text=text,
        speaker="integration" if is_bot else str(msg.get("user") or "unknown"),
        thread_ts=str(msg["thread_ts"]) if msg.get("thread_ts") else None,
        reply_count=int(msg.get("reply_count") or 0),
        latest_reply=str(msg["latest_reply"]) if msg.get("latest_reply") else None,
        edited_ts=str(edited["ts"]) if edited.get("ts") else None,
        integration=is_bot,
    )


def format_ts(ts: str | float) -> str:
    dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.UTC)
    return dt.strftime("%Y-%m-%d %H:%M")


def ts_date(ts: str | float) -> datetime.date:
    return datetime.datetime.fromtimestamp(float(ts), tz=datetime.UTC).date()


@dataclass
class Unit:
    channel_id: str
    key: str
    kind: UnitKind
    messages: list[Message] = field(default_factory=list)
    boost: bool = False
    # For bookmarks, which have no messages: the text and when it changed.
    extra_text: str = ""
    extra_activity: float = 0.0
    overlap: int = 0  # leading messages borrowed from the previous session

    @property
    def message_ts(self) -> list[str]:
        return [m.ts for m in self.messages]

    @property
    def first_ts(self) -> str:
        own = self.messages[self.overlap :] or self.messages
        return own[0].ts if own else self.key

    @property
    def last_activity(self) -> float:
        times = [m.activity for m in self.messages]
        return max([*times, self.extra_activity])

    def render(self) -> str:
        """The unit as the model sees it: dated lines, speakers numbered."""
        if self.kind == "bookmark":
            return self.extra_text
        numbers: dict[str, int] = {}
        lines = []
        for m in self.messages:
            if m.integration:
                who = "Integration"
            else:
                who = f"Person {numbers.setdefault(m.speaker, len(numbers) + 1)}"
            edited = " (edited)" if m.edited_ts else ""
            lines.append(f"[{format_ts(m.ts)}] {who}{edited}: {m.text}")
        return "\n".join(lines)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.render().encode()).hexdigest()

    def is_quiet(self, now: float, quiet_hours: float) -> bool:
        return now - self.last_activity >= quiet_hours * 3600


def thread_unit(
    channel_id: str,
    raw: Sequence[dict[str, Any]],
    me: BotIdentity,
    bot_allowlist: frozenset[str],
    boost: bool = False,
) -> Unit | None:
    """A thread as a unit, or None if it is excluded or empty.

    A thread whose parent is a question to the bot (or posted by it) is
    excluded entirely, so the bot's answers never come back as facts.
    """
    if not raw:
        return None
    parent = raw[0]
    if mentions_bot(parent, me) or is_own(parent, me):
        return None
    messages = [
        m for m in (to_message(msg, me, bot_allowlist) for msg in raw) if m is not None
    ]
    if not messages:
        return None
    key = str(parent.get("thread_ts") or parent["ts"])
    return Unit(
        channel_id=channel_id, key=key, kind="thread", messages=messages, boost=boost
    )


def sessions(
    channel_id: str,
    messages: Iterable[Message],
    gap_minutes: int,
    max_messages: int,
    overlap: int,
) -> list[Unit]:
    """Group unthreaded messages (any order) into sessions."""
    ordered = sorted(messages, key=lambda m: float(m.ts))
    gap = gap_minutes * 60
    units: list[Unit] = []
    current: list[Message] = []
    borrowed = 0

    def close() -> None:
        if len(current) > borrowed:
            units.append(
                Unit(
                    channel_id=channel_id,
                    key=current[borrowed].ts,
                    kind="session",
                    messages=list(current),
                    overlap=borrowed,
                )
            )

    for msg in ordered:
        if current:
            since = float(msg.ts) - float(current[-1].ts)
            if since >= gap or len(current) >= max_messages:
                close()
                keep = overlap if since < gap * _OVERLAP_GAP_FACTOR else 0
                current = current[-keep:] if keep else []
                borrowed = len(current)
        current.append(msg)
    if current:
        close()
    return units


def pin_unit(channel_id: str, message: Message) -> Unit:
    return Unit(
        channel_id=channel_id,
        key=f"pin:{message.ts}",
        kind="pin",
        messages=[message],
        boost=True,
    )


def bookmark_unit(channel_id: str, bookmark: dict[str, Any]) -> Unit | None:
    title = str(bookmark.get("title") or "").strip()
    link = str(bookmark.get("link") or "").strip()
    if not link and not title:
        return None
    updated = float(bookmark.get("date_updated") or bookmark.get("date_created") or 0)
    return Unit(
        channel_id=channel_id,
        key=f"bookmark:{bookmark.get('id') or link}",
        kind="bookmark",
        boost=True,
        extra_text=f"Channel bookmark: {title} {link}".strip(),
        extra_activity=updated,
    )
