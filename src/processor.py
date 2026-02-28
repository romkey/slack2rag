"""
Converts raw Slack API message dicts into indexable Document objects.

Strategy
--------
* Each *thread* is one document: the root message plus all replies are
  concatenated so retrieval surfaces the full conversation context.
* Standalone messages (no replies) are individual documents.
* Documents longer than MAX_CHARS are split into overlapping chunks so
  no embedding window is overwhelmed.
"""

from __future__ import annotations

import datetime
import textwrap
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
    user_id: str
    user_name: str
    thread_ts: Optional[str]      # None for standalone messages
    reply_count: int
    permalink: str = ""

    def payload(self) -> dict:
        """Return metadata dict suitable for Qdrant point payload."""
        return {
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "ts": self.ts,
            "date": self.date,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "thread_ts": self.thread_ts,
            "reply_count": self.reply_count,
            "text": self.text,
            "permalink": self.permalink,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts_to_date(ts: str) -> str:
    try:
        return datetime.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return ""


def _make_id(channel_id: str, ts: str, chunk_idx: int = 0) -> str:
    """Stable UUID-5 from (channel_id, ts, chunk_idx)."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    name = f"{channel_id}:{ts}:{chunk_idx}"
    return str(uuid.uuid5(namespace, name))


def _chunk(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
    """Split *text* into overlapping chunks of at most *max_chars*."""
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
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
) -> List[Document]:
    """
    Build one or more Documents from a root message and its replies.

    Returns multiple Documents when the combined text exceeds MAX_CHARS.
    """
    channel_id = channel["id"]
    channel_name = channel.get("name", channel_id)
    ts = root_msg["ts"]
    thread_ts = root_msg.get("thread_ts") or ts
    reply_count = len(replies)
    root_user_id = root_msg.get("user", "unknown")
    root_user_name = get_user_name(root_user_id)
    date = _ts_to_date(ts)

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
                user_id=root_user_id,
                user_name=root_user_name,
                thread_ts=thread_ts,
                reply_count=reply_count,
            )
        )

    return docs
