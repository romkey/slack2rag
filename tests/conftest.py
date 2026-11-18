"""Shared fakes and fixtures.

FakeSlack stands in for the Slack Web API; FakeChat for the chat model
(facts are written into message text as "FACT kind|question|answer", so a
test controls what gets extracted); FakeEmbedder makes bag-of-words vectors
so similar texts are similar. Database tests need TEST_DATABASE_URL.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterator
from typing import Any, cast

import psycopg
import pytest
from qdrant_client import QdrantClient

from slack2rag.config import Config
from slack2rag.embedder import Embedder, SparseEncoder
from slack2rag.extractor import (
    EXCLUSION_SYSTEM,
    RELEVANCE_SYSTEM,
    Extractor,
)
from slack2rag.llm import ChatModel, LLMError
from slack2rag.pg_store import PgStore, migrate
from slack2rag.pipeline import Pipeline
from slack2rag.slack_client import BotIdentity, ChannelGoneError, SlackClient
from slack2rag.vector_store import VectorStore

ME = BotIdentity(workspace_url="https://ws.slack.com", user_id="UBOT", bot_id="BBOT")
DAY = 86400.0
T0 = 1_780_000_000.0  # a fixed "now" for tests: 2026-05-28


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeSlack:
    """Channels and messages in memory, read the way the Web API reads them."""

    def __init__(self) -> None:
        self.channels: dict[str, dict[str, Any]] = {}
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.pinned: dict[str, list[str]] = {}
        self.marks: dict[str, list[dict[str, Any]]] = {}
        self.names = {"Alice Smith", "alice", "Bob"}
        self.calls: list[tuple[str, str]] = []
        self.web_client = None

    # set-up helpers

    def add_channel(
        self, cid: str, name: str, topic: str = "", purpose: str = ""
    ) -> None:
        self.channels[cid] = {
            "id": cid,
            "name": name,
            "topic": {"value": topic},
            "purpose": {"value": purpose},
        }
        self.messages.setdefault(cid, [])

    def post(
        self,
        cid: str,
        ts: float | str,
        text: str,
        user: str = "U1",
        thread_ts: float | str | None = None,
        **extra: Any,
    ) -> str:
        msg: dict[str, Any] = {
            "type": "message",
            "ts": f"{float(ts):.6f}",
            "text": text,
        }
        if user:
            msg["user"] = user
        if thread_ts is not None:
            msg["thread_ts"] = f"{float(thread_ts):.6f}"
        msg.update(extra)
        self.messages.setdefault(cid, []).append(msg)
        return str(msg["ts"])

    def delete(self, cid: str, ts: str) -> None:
        self.messages[cid] = [m for m in self.messages[cid] if m["ts"] != ts]

    def edit(self, cid: str, ts: str, text: str, edited_ts: float) -> None:
        for m in self.messages[cid]:
            if m["ts"] == ts:
                m["text"] = text
                m["edited"] = {"ts": f"{edited_ts:.6f}"}

    # the SlackClient interface

    def identify(self) -> BotIdentity:
        return ME

    def permalink(self, cid: str, ts: str, thread_ts: str | None = None) -> str:
        return SlackClient.permalink(cast(SlackClient, self), cid, ts, thread_ts)

    def member_names(self) -> set[str]:
        return set(self.names)

    def member_channels(self) -> list[dict[str, Any]]:
        self.calls.append(("member_channels", ""))
        return list(self.channels.values())

    def _check(self, method: str, cid: str) -> None:
        self.calls.append((method, cid))
        if cid not in self.channels:
            raise ChannelGoneError(f"{method}: not_in_channel")

    def _top_level(self, cid: str) -> list[dict[str, Any]]:
        out = []
        for m in self.messages.get(cid, []):
            if m.get("thread_ts") and m["thread_ts"] != m["ts"]:
                if m.get("subtype") == "thread_broadcast":
                    out.append(dict(m))
                continue
            replies = [
                r
                for r in self.messages[cid]
                if r.get("thread_ts") == m["ts"] and r["ts"] != m["ts"]
            ]
            top = dict(m)
            if replies:
                top["thread_ts"] = m["ts"]
                top["reply_count"] = len(replies)
                top["latest_reply"] = max(r["ts"] for r in replies)
            out.append(top)
        return out

    def history(
        self,
        cid: str,
        oldest: str | None = None,
        latest: str | None = None,
        inclusive: bool = False,
    ) -> Iterator[dict[str, Any]]:
        self._check("history", cid)
        lo = float(oldest) if oldest else -math.inf
        hi = float(latest) if latest else math.inf
        for m in sorted(self._top_level(cid), key=lambda m: -float(m["ts"])):
            ts = float(m["ts"])
            if (inclusive and lo <= ts <= hi) or (not inclusive and lo < ts < hi):
                yield m

    def thread(self, cid: str, ts: str) -> list[dict[str, Any]]:
        self._check("thread", cid)
        parent = [m for m in self._top_level(cid) if m["ts"] == ts]
        replies = sorted(
            (
                dict(r)
                for r in self.messages[cid]
                if r.get("thread_ts") == ts and r["ts"] != ts
            ),
            key=lambda r: float(r["ts"]),
        )
        return parent + replies if parent else replies

    def pins(self, cid: str) -> list[dict[str, Any]]:
        self._check("pins", cid)
        tops = {m["ts"]: m for m in self._top_level(cid)}
        return [tops[ts] for ts in self.pinned.get(cid, []) if ts in tops]

    def bookmarks(self, cid: str) -> list[dict[str, Any]]:
        self._check("bookmarks", cid)
        return list(self.marks.get(cid, []))


_FACT = re.compile(r"FACT (\w+)\|([^|]+)\|([^|\n]+?)(?:\|(\w+))?(?=$|\n| FACT)")


class FakeChat:
    """Answers the three prompts from markers in the text it is sent."""

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.calls: list[str] = []

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        if "LLMFAIL" in user:
            raise LLMError("fake outage")
        if system == RELEVANCE_SYSTEM:
            self.calls.append("relevance")
            return {"relevant": "FACT" in user, "reason": "fake"}
        if system == EXCLUSION_SYSTEM:
            self.calls.append("exclusion")
            lines = [ln for ln in user.splitlines() if re.match(r"^\d+\. ", ln)]
            return {
                "verdicts": [
                    {
                        "index": i,
                        "exclude": "EXCLUDE" in ln,
                        "category": "none",
                        "weakness": "unreliable" in ln,
                    }
                    for i, ln in enumerate(lines)
                ]
            }
        self.calls.append("extraction")
        facts = []
        for kind, question, answer, confidence in _FACT.findall(user):
            links = re.findall(r"https?://\S+", answer)
            facts.append(
                {
                    "kind": kind,
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "entities": re.findall(r"\b(laser|printer|lathe|wiki)\b", answer),
                    "links": [link.rstrip(".") for link in links],
                    "confidence": confidence or "high",
                    "topics": ["lasers"] if "laser" in answer else [],
                }
            )
        return {"facts": facts}


class FakeEmbedder:
    dimension = 64

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        self.calls += 1
        vectors = []
        for text in texts:
            v = [0.0] * self.dimension
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(
                    hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16
                )
                v[h % self.dimension] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


def make_config(**overrides: Any) -> Config:
    values: dict[str, Any] = {
        "slack_bot_token": "xoxb-test",
        "database_url": os.environ.get("TEST_DATABASE_URL", "postgresql://unused"),
        "chat_model": "fake-model",
        "report_dir": "/tmp/unused",  # noqa: S108 - tests override where they write
    }
    values.update(overrides)
    return Config(**values)


@pytest.fixture
def pg_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (PostgreSQL required)")
    return url


def reset_database(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")


@pytest.fixture
def store(pg_url: str) -> PgStore:
    reset_database(pg_url)
    migrate(pg_url)
    return PgStore(pg_url)


@pytest.fixture
def qdrant() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def slack() -> FakeSlack:
    fake = FakeSlack()
    fake.add_channel("C1", "laser", topic="Laser cutter help", purpose="Ask Alice")
    return fake


@pytest.fixture
def chat() -> FakeChat:
    return FakeChat()


@pytest.fixture
def clock() -> Clock:
    return Clock()


def build(
    store: PgStore,
    slack: FakeSlack,
    chat: FakeChat,
    qdrant: QdrantClient,
    clock: Clock,
    **cfg: Any,
) -> Pipeline:
    embedder = FakeEmbedder()
    config = make_config(**cfg)
    vectors = VectorStore(
        "", config.qdrant_collection, embedder.dimension, "fake-embed", client=qdrant
    )
    extractor = Extractor(
        cast(ChatModel, chat), ("lasers", "woodworking"), slack.member_names()
    )
    return Pipeline(
        config,
        cast(SlackClient, slack),
        store,
        vectors,
        cast(Embedder, embedder),
        extractor,
        SparseEncoder(),
        clock=clock,
    )


@pytest.fixture
def pipeline(
    store: PgStore,
    slack: FakeSlack,
    chat: FakeChat,
    qdrant: QdrantClient,
    clock: Clock,
) -> Pipeline:
    return build(store, slack, chat, qdrant, clock)
