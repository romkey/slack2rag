"""
slack2rag — entry point

Flow per sync cycle
-------------------
1. List public channels (or those specified in SLACK_CHANNELS).
2. For each channel, fetch messages newer than the last indexed timestamp.
3. For each message, fetch thread replies (if any).
4. Build Documents from the messages.
5. Embed in batches and upsert into Qdrant.
6. Advance the channel cursor.
7. Sleep until the next cycle (or exit if RUN_ONCE=true).
"""

import logging
import os
import time
from typing import List

from dotenv import load_dotenv

from .config import Config
from .embedder import Embedder
from .processor import Document, build_documents
from .slack_client import SlackClient
from .state import SyncState
from .vector_store import VectorStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("slack2rag")


def sync_channel(
    channel: dict,
    slack: SlackClient,
    store: VectorStore,
    embedder: Embedder,
    state: SyncState,
    batch_size: int,
) -> int:
    """
    Sync one channel.  Returns the number of new documents indexed.
    """
    channel_id = channel["id"]
    channel_name = channel.get("name", channel_id)
    oldest_ts = state.get_cursor(channel_id)

    logger.info(
        "Syncing #%s  (oldest_ts=%s)",
        channel_name,
        oldest_ts or "beginning",
    )

    pending_docs: List[Document] = []
    latest_ts: str | None = oldest_ts
    total_indexed = 0

    for msg in slack.get_channel_messages(channel_id, oldest_ts=oldest_ts):
        ts = msg["ts"]

        # Track the newest ts we've seen so we can advance the cursor later
        if latest_ts is None or float(ts) > float(latest_ts):
            latest_ts = ts

        # Fetch replies for threaded messages
        replies: List[dict] = []
        if msg.get("reply_count", 0) > 0 and msg.get("thread_ts") == ts:
            replies = slack.get_thread_replies(channel_id, ts)

        docs = build_documents(
            root_msg=msg,
            replies=replies,
            channel=channel,
            resolve_text=slack.resolve_text,
            get_user_name=slack.get_user_name,
        )
        pending_docs.extend(docs)

        if len(pending_docs) >= batch_size:
            total_indexed += _flush(pending_docs, store, embedder)
            pending_docs = []

    if pending_docs:
        total_indexed += _flush(pending_docs, store, embedder)

    if latest_ts and latest_ts != oldest_ts:
        state.set_cursor(channel_id, latest_ts)
        logger.info("#%s  indexed %d documents (cursor→%s)", channel_name, total_indexed, latest_ts)
    else:
        logger.info("#%s  no new messages", channel_name)

    return total_indexed


def _flush(docs: List[Document], store: VectorStore, embedder: Embedder) -> int:
    texts = [d.text for d in docs]
    vectors = embedder.embed(texts)
    store.upsert(docs, vectors)
    return len(docs)


def run_once(cfg: Config, slack: SlackClient, store: VectorStore, embedder: Embedder, state: SyncState) -> None:
    channels = slack.get_public_channels(cfg.channel_list or None)
    if not channels:
        logger.warning("No accessible public channels found")
        return

    total = 0
    for ch in channels:
        total += sync_channel(ch, slack, store, embedder, state, cfg.batch_size)

    logger.info("Sync complete — %d documents indexed  (total in store: %d)", total, store.count())


def main() -> None:
    cfg = Config.from_env()

    from . import __version__
    logger.info("Starting slack2rag v%s", __version__)
    logger.info("  Qdrant:     %s / %s", cfg.qdrant_url, cfg.qdrant_collection)
    logger.info("  Embeddings: %s", cfg.embedding_provider)
    logger.info("  Channels:   %s", cfg.channel_list or "all public")
    logger.info("  Run once:   %s", cfg.run_once)

    slack = SlackClient(cfg.slack_bot_token)
    embedder = Embedder(
        provider=cfg.embedding_provider,
        local_model=cfg.local_embedding_model,
        openai_api_key=cfg.openai_api_key,
        openai_model=cfg.openai_embedding_model,
    )
    store = VectorStore(
        url=cfg.qdrant_url,
        collection=cfg.qdrant_collection,
        dimension=embedder.dimension,
    )
    state = SyncState(cfg.state_file)

    if cfg.run_once:
        run_once(cfg, slack, store, embedder, state)
        return

    interval = cfg.sync_interval_minutes * 60
    while True:
        try:
            run_once(cfg, slack, store, embedder, state)
        except Exception:
            logger.exception("Sync cycle failed — will retry next interval")
        logger.info("Sleeping %d minutes until next sync…", cfg.sync_interval_minutes)
        time.sleep(interval)


if __name__ == "__main__":
    main()
