"""
Converts raw Slack API message dicts into indexable Document objects.

Strategy
--------
* Each *thread* is one document: the root message plus all replies are
  concatenated so retrieval surfaces the full conversation context.
* Standalone messages (no replies) are individual documents.
* Documents longer than MAX_CHARS are split into overlapping chunks
  using line-aware splitting so no message is cut mid-sentence.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

MAX_CHARS = 1_500   # target document size before chunking
OVERLAP = 200       # character overlap between consecutive chunks


@dataclass
class Document:
    """Unit of text that will be embedded and stored in the vector DB."""

    id: str                       # stable UUID derived from channel+ts
    text: str                     # clean, human-readable content
    channel_id: str
    channel_name: str
    ts: str                       # Slack timestamp of the root message
    date: str                     # ISO-8601 date, e.g. "2024-01-25"
    datetime_str: str             # ISO-8601 datetime, e.g. "2024-01-25T14:30:00Z"
    user_id: str
    user_name: str
    thread_ts: Optional[str]      # None for standalone messages
    reply_count: int
    permalink: str = ""
    channel_topic: str = ""
    reaction_count: int = 0
    reactions: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    doc_type: str = "message"     # "message" | "channel_summary" | "workspace_summary"

    def payload(self) -> dict:
        """Return metadata dict suitable for Qdrant point payload."""
        return {
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "ts": self.ts,
            "date": self.date,
            "datetime": self.datetime_str,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "thread_ts": self.thread_ts,
            "reply_count": self.reply_count,
            "text": self.text,
            "permalink": self.permalink,
            "channel_topic": self.channel_topic,
            "reaction_count": self.reaction_count,
            "reactions": self.reactions,
            "attachments": self.attachments,
            "doc_type": self.doc_type,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts_to_date(ts: str) -> str:
    try:
        dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return ""


def _ts_to_datetime(ts: str) -> str:
    try:
        dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OSError):
        return ""


def _make_id(channel_id: str, ts: str, chunk_idx: int = 0) -> str:
    """Stable UUID-5 from (channel_id, ts, chunk_idx)."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    name = f"{channel_id}:{ts}:{chunk_idx}"
    return str(uuid.uuid5(namespace, name))


def _chunk(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
    """Split *text* into overlapping chunks, preferring line boundaries.

    Lines are kept whole whenever possible.  Only lines longer than
    *max_chars* fall back to a hard character split.
    """
    if len(text) <= max_chars:
        return [text]

    lines = text.split("\n")
    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + (1 if current_lines else 0)  # +1 for \n separator

        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))

            # Overlap: keep trailing lines that fit within the overlap budget
            overlap_lines: List[str] = []
            overlap_len = 0
            for prev_line in reversed(current_lines):
                if overlap_len + len(prev_line) + 1 > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_len += len(prev_line) + 1

            current_lines = overlap_lines
            current_len = sum(len(l) for l in current_lines) + max(len(current_lines) - 1, 0)

        # Handle single lines longer than max_chars with hard split
        if len(line) > max_chars and not current_lines:
            start = 0
            while start < len(line):
                chunks.append(line[start : start + max_chars])
                start += max_chars - overlap
            continue

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def _format_message(msg: dict, user_name: str, resolve_text) -> str:
    text = resolve_text(msg.get("text", "")).strip()
    if not text:
        return ""
    return f"[{user_name}]: {text}"


# ── public API ────────────────────────────────────────────────────────────────

def build_documents(
    root_msg: dict,
    replies: List[dict],
    channel: dict,
    resolve_text,
    get_user_name,
    *,
    permalink: str = "",
    channel_topic: str = "",
    reaction_count: int = 0,
    reactions: Optional[List[str]] = None,
    attachments: Optional[List[str]] = None,
    min_message_length: int = 0,
) -> List[Document]:
    """
    Build one or more Documents from a root message and its replies.

    Returns multiple Documents when the combined text exceeds MAX_CHARS.
    Returns an empty list for messages shorter than *min_message_length*
    (standalone messages only — threads are always kept).
    """
    channel_id = channel["id"]
    channel_name = channel.get("name", channel_id)
    ts = root_msg["ts"]
    thread_ts = root_msg.get("thread_ts") or ts
    reply_count = len(replies)
    root_user_id = root_msg.get("user", "unknown")
    root_user_name = get_user_name(root_user_id)
    date = _ts_to_date(ts)
    datetime_str = _ts_to_datetime(ts)

    lines: List[str] = []

    root_text = _format_message(root_msg, root_user_name, resolve_text)
    if root_text:
        lines.append(root_text)

    for reply in replies:
        user_id = reply.get("user", "unknown")
        user_name = get_user_name(user_id)
        reply_text = _format_message(reply, user_name, resolve_text)
        if reply_text:
            lines.append(reply_text)

    full_text = "\n".join(lines).strip()
    if not full_text:
        return []

    # Filter short standalone messages (threads are always kept)
    if reply_count == 0 and min_message_length > 0 and len(full_text) < min_message_length:
        return []

    chunks = _chunk(full_text)
    docs: List[Document] = []

    for idx, chunk_text in enumerate(chunks):
        doc_text = chunk_text
        if len(chunks) > 1:
            doc_text = f"(part {idx + 1}/{len(chunks)}) {chunk_text}"

        docs.append(
            Document(
                id=_make_id(channel_id, ts, idx),
                text=doc_text,
                channel_id=channel_id,
                channel_name=channel_name,
                ts=ts,
                date=date,
                datetime_str=datetime_str,
                user_id=root_user_id,
                user_name=root_user_name,
                thread_ts=thread_ts,
                reply_count=reply_count,
                permalink=permalink,
                channel_topic=channel_topic,
                reaction_count=reaction_count,
                reactions=reactions or [],
                attachments=attachments or [],
            )
        )

    return docs


# ── Summary documents ─────────────────────────────────────────────────────────

def _make_summary_id(key: str) -> str:
    """Stable UUID-5 for a summary document."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(namespace, f"summary:{key}"))


def _ts_to_iso_date(ts_str: str) -> str:
    """Convert a Slack-style unix timestamp string to YYYY-MM-DD."""
    try:
        dt = datetime.datetime.fromtimestamp(float(ts_str), tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OSError, TypeError):
        return ""


def build_channel_summary(
    channel: dict,
    message_count: int,
) -> Document:
    """Build a summary document describing a single channel.

    Designed to surface when users ask questions like "what is #engineering
    for?", "how active is #random?", or "which channel handles incidents?"
    """
    name = channel.get("name", channel.get("id", "unknown"))
    channel_id = channel["id"]
    topic = channel.get("topic", {}).get("value", "")
    purpose = channel.get("purpose", {}).get("value", "")
    num_members = channel.get("num_members", 0)
    created = _ts_to_iso_date(str(channel.get("created", "")))

    lines = [f"Channel #{name} information:"]
    if topic:
        lines.append(f"Topic: {topic}")
    if purpose and purpose != topic:
        lines.append(f"Purpose: {purpose}")
    if num_members:
        lines.append(f"Members: {num_members}")
    if created:
        lines.append(f"Created: {created}")
    lines.append(f"Indexed messages: {message_count:,}")
    if message_count == 0:
        lines.append("This channel has no indexed messages yet.")

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return Document(
        id=_make_summary_id(f"channel:{channel_id}"),
        text="\n".join(lines),
        channel_id=channel_id,
        channel_name=name,
        ts="0",
        date="",
        datetime_str=now,
        user_id="system",
        user_name="system",
        thread_ts=None,
        reply_count=0,
        channel_topic=topic or purpose,
        doc_type="channel_summary",
    )


def build_workspace_summary(
    channels: List[dict],
    channel_counts: dict[str, int],
) -> Document:
    """Build a summary document listing all indexed channels.

    Designed to surface when users ask "what channels do we have?",
    "where should I ask about X?", or "how many channels are there?"
    """
    total_messages = sum(channel_counts.values())
    total_members = sum(ch.get("num_members", 0) for ch in channels)

    lines = [
        f"This Slack workspace has {len(channels)} indexed public channel{'s' if len(channels) != 1 else ''} "
        f"with {total_messages:,} total messages.",
        "",
        "Channels:",
    ]

    sorted_channels = sorted(channels, key=lambda c: channel_counts.get(c["id"], 0), reverse=True)
    for ch in sorted_channels:
        name = ch.get("name", ch.get("id", "?"))
        topic = ch.get("topic", {}).get("value", "")
        purpose = ch.get("purpose", {}).get("value", "")
        desc = topic or purpose or "no description"
        members = ch.get("num_members", 0)
        count = channel_counts.get(ch["id"], 0)

        member_str = f", {members} members" if members else ""
        lines.append(f"  #{name} ({count:,} messages{member_str}) — {desc}")

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return Document(
        id=_make_summary_id("workspace"),
        text="\n".join(lines),
        channel_id="workspace",
        channel_name="workspace",
        ts="0",
        date="",
        datetime_str=now,
        user_id="system",
        user_name="system",
        thread_ts=None,
        reply_count=0,
        doc_type="workspace_summary",
    )


def build_user_summary(profile: dict) -> Document:
    """Build a summary document for a single Slack user.

    Surfaces when users ask "who is Alice?", "what does Bob do?",
    "what timezone is Carol in?", or "what are Alice's pronouns?"
    """
    uid = profile["user_id"]
    display = profile.get("display_name") or profile.get("real_name") or profile.get("username") or uid
    real = profile.get("real_name", "")
    username = profile.get("username", "")

    lines = [f"Team member: {display}"]
    if real and real != display:
        lines.append(f"Full name: {real}")
    if username:
        lines.append(f"Username: @{username}")
    if profile.get("title"):
        lines.append(f"Title: {profile['title']}")
    if profile.get("pronouns"):
        lines.append(f"Pronouns: {profile['pronouns']}")
    if profile.get("timezone_label"):
        tz = profile.get("timezone", "")
        lines.append(f"Timezone: {profile['timezone_label']}" + (f" ({tz})" if tz else ""))
    if profile.get("status_text"):
        emoji = profile.get("status_emoji", "")
        lines.append(f"Status: {profile['status_text']} {emoji}".strip())
    if profile.get("is_owner"):
        lines.append("Role: workspace owner")
    elif profile.get("is_admin"):
        lines.append("Role: workspace admin")
    if profile.get("is_bot"):
        lines.append("This is a bot account.")

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return Document(
        id=_make_summary_id(f"user:{uid}"),
        text="\n".join(lines),
        channel_id="users",
        channel_name="users",
        ts="0",
        date="",
        datetime_str=now,
        user_id=uid,
        user_name=display,
        thread_ts=None,
        reply_count=0,
        doc_type="user_summary",
    )


def build_team_summary(profiles: List[dict]) -> Document:
    """Build a summary document listing all team members.

    Surfaces when users ask "who is on the team?", "how many people
    do we have?", or "list all team members."
    """
    active = [p for p in profiles if not p.get("deleted") and not p.get("is_bot")]
    bots = [p for p in profiles if p.get("is_bot") and not p.get("deleted")]

    lines = [f"This Slack workspace has {len(active)} active team member{'s' if len(active) != 1 else ''}"]
    if bots:
        lines[0] += f" and {len(bots)} bot{'s' if len(bots) != 1 else ''}."
    else:
        lines[0] += "."

    lines.append("")
    lines.append("Team members:")
    for p in sorted(active, key=lambda p: (p.get("real_name") or p.get("display_name") or "").lower()):
        display = p.get("display_name") or p.get("real_name") or p.get("username") or p["user_id"]
        parts = [f"  {display}"]
        if p.get("title"):
            parts.append(f"({p['title']})")
        if p.get("pronouns"):
            parts.append(f"[{p['pronouns']}]")
        lines.append(" ".join(parts))

    if bots:
        lines.append("")
        lines.append("Bots:")
        for p in sorted(bots, key=lambda p: (p.get("real_name") or p.get("display_name") or "").lower()):
            display = p.get("display_name") or p.get("real_name") or p.get("username") or p["user_id"]
            lines.append(f"  {display}")

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return Document(
        id=_make_summary_id("team"),
        text="\n".join(lines),
        channel_id="users",
        channel_name="users",
        ts="0",
        date="",
        datetime_str=now,
        user_id="system",
        user_name="system",
        thread_ts=None,
        reply_count=0,
        doc_type="team_summary",
    )
