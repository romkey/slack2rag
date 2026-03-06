"""
slack2rag — entry point

Flow per sync cycle
-------------------
1. List public channels (or those specified in SLACK_CHANNELS).
2. For each channel, fetch messages newer than the last indexed timestamp.
3. For each message, fetch thread replies (if any).
4. Build Documents from the messages (with rich metadata).
5. Embed in batches and upsert into Qdrant.
6. Advance the channel cursor.
7. Optionally refresh recently-active threads that gained new replies.
8. Generate summary documents (channels, workspace, users, team).
9. Sleep until the next cycle (or exit if RUN_ONCE=true).
"""

import json
import logging
import os
import time
from collections import Counter
from typing import List, Optional

from dotenv import load_dotenv

from .config import Config
from .embedder import Embedder, EmbeddingError, SparseEncoder, tokenize_text
from .processor import (
    ChannelStats,
    Document,
    build_documents,
    build_channel_summary,
    build_workspace_summary,
    build_user_summary,
    build_team_summary,
)
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


def _build_docs_for_message(
    msg: dict,
    replies: List[dict],
    channel: dict,
    slack: SlackClient,
    cfg: Config,
) -> List[Document]:
    """Build Documents from a single message, enriched with full metadata."""
    ts = msg["ts"]
    thread_ts = msg.get("thread_ts") or ts

    reaction_count, reactions = slack.get_reactions(msg)
    attachments = slack.get_attachments(msg)

    # Aggregate reactions from replies too
    for reply in replies:
        rc, rnames = slack.get_reactions(reply)
        reaction_count += rc
        reactions.extend(rnames)
        attachments.extend(slack.get_attachments(reply))

    permalink = slack.make_permalink(channel["id"], ts)
    channel_topic = slack.get_channel_topic(channel)

    return build_documents(
        root_msg=msg,
        replies=replies,
        channel=channel,
        resolve_text=slack.resolve_text,
        get_user_name=slack.get_user_name,
        permalink=permalink,
        channel_topic=channel_topic,
        reaction_count=reaction_count,
        reactions=reactions,
        attachments=attachments,
        min_message_length=cfg.min_message_length,
        reaction_boost_threshold=cfg.reaction_boost_threshold,
    )


def sync_channel(
    channel: dict,
    slack: SlackClient,
    store: VectorStore,
    embedder: Embedder,
    sparse_encoder: Optional[SparseEncoder],
    state: SyncState,
    cfg: Config,
) -> tuple[int, ChannelStats]:
    """Sync one channel.  Returns (documents_indexed, channel_stats)."""
    channel_id = channel["id"]
    channel_name = channel.get("name", channel_id)
    oldest_ts = state.get_cursor(channel_id)

    logger.info("Syncing #%s  (oldest_ts=%s)", channel_name, oldest_ts or "beginning")

    pending_docs: List[Document] = []
    latest_ts: str | None = oldest_ts
    total_indexed = 0
    stats = ChannelStats()

    for msg in slack.get_channel_messages(channel_id, oldest_ts=oldest_ts):
        ts = msg["ts"]

        if latest_ts is None or float(ts) > float(latest_ts):
            latest_ts = ts

        replies: List[dict] = []
        if msg.get("reply_count", 0) > 0 and msg.get("thread_ts") == ts:
            replies = slack.get_thread_replies(channel_id, ts)

        # Accumulate channel statistics
        raw_text = msg.get("text", "")
        stats.token_counts.update(Counter(tokenize_text(raw_text)))
        uid = msg.get("user", "unknown")
        if uid != "unknown":
            stats.user_counts[uid] += 1
        try:
            stats.timestamps.append(float(ts))
        except (ValueError, TypeError):
            pass
        for reply in replies:
            reply_text = reply.get("text", "")
            stats.token_counts.update(Counter(tokenize_text(reply_text)))
            ruid = reply.get("user", "unknown")
            if ruid != "unknown":
                stats.user_counts[ruid] += 1

        docs = _build_docs_for_message(msg, replies, channel, slack, cfg)
        pending_docs.extend(docs)

        if len(pending_docs) >= cfg.batch_size:
            total_indexed += _flush(pending_docs, store, embedder, sparse_encoder)
            pending_docs = []

    if pending_docs:
        total_indexed += _flush(pending_docs, store, embedder, sparse_encoder)

    if latest_ts and latest_ts != oldest_ts:
        state.set_cursor(channel_id, latest_ts)
        logger.info("#%s  indexed %d documents (cursor→%s)", channel_name, total_indexed, latest_ts)
    else:
        logger.info("#%s  no new messages", channel_name)

    return total_indexed, stats


def refresh_threads(
    channel: dict,
    slack: SlackClient,
    store: VectorStore,
    embedder: Embedder,
    sparse_encoder: Optional[SparseEncoder],
    cfg: Config,
    lookback_hours: int,
) -> int:
    """Re-index threads in *channel* that received new replies recently.

    Looks back *lookback_hours* from now, finds threaded messages, and
    re-indexes them.  Upsert is idempotent so unchanged threads are
    harmless (same UUID → overwrite with identical data).
    """
    channel_id = channel["id"]
    channel_name = channel.get("name", channel_id)

    lookback_ts = str(time.time() - lookback_hours * 3600)

    logger.info(
        "#%s  refreshing threads active in the last %d hours",
        channel_name,
        lookback_hours,
    )

    pending_docs: List[Document] = []
    refreshed = 0

    for msg in slack.get_channel_messages(channel_id, oldest_ts=lookback_ts):
        if msg.get("reply_count", 0) == 0:
            continue
        ts = msg["ts"]
        if msg.get("thread_ts") != ts:
            continue

        replies = slack.get_thread_replies(channel_id, ts)
        docs = _build_docs_for_message(msg, replies, channel, slack, cfg)
        pending_docs.extend(docs)
        refreshed += 1

        if len(pending_docs) >= cfg.batch_size:
            _flush(pending_docs, store, embedder, sparse_encoder)
            pending_docs = []

    if pending_docs:
        _flush(pending_docs, store, embedder, sparse_encoder)

    logger.info("#%s  refreshed %d threads", channel_name, refreshed)
    return refreshed


def _flush(
    docs: List[Document],
    store: VectorStore,
    embedder: Embedder,
    sparse_encoder: Optional[SparseEncoder],
) -> int:
    texts = [d.text for d in docs]
    vectors = embedder.embed(texts)
    sparse_vectors = sparse_encoder.encode(texts) if sparse_encoder else None
    store.upsert(docs, vectors, sparse_vectors=sparse_vectors)
    return len(docs)


def _load_channel_stats(stats_file: str) -> dict[str, ChannelStats]:
    """Load persistent channel stats from disk."""
    if not os.path.exists(stats_file):
        return {}
    try:
        with open(stats_file) as f:
            raw = json.load(f)
        return {k: ChannelStats.from_dict(v) for k, v in raw.items()}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


def _save_channel_stats(stats_file: str, all_stats: dict[str, ChannelStats]) -> None:
    """Persist accumulated channel stats to disk."""
    os.makedirs(os.path.dirname(stats_file) or ".", exist_ok=True)
    raw = {k: v.to_dict() for k, v in all_stats.items()}
    try:
        with open(stats_file, "w") as f:
            json.dump(raw, f)
    except OSError as exc:
        logger.warning("Could not save channel stats: %s", exc)


def _index_summaries(
    channels: List[dict],
    channel_stats: dict[str, ChannelStats],
    user_profiles: dict[str, dict],
    get_user_name,
    store: VectorStore,
    embedder: Embedder,
    sparse_encoder: Optional[SparseEncoder],
) -> None:
    """Generate and index workspace, channel, user, and team summary documents."""
    logger.info("Generating channel, workspace, and user summaries…")

    channel_counts: dict[str, int] = {}
    summary_docs: List[Document] = []

    for ch in channels:
        cid = ch["id"]
        count = store.count_by_channel(cid)
        channel_counts[cid] = count
        summary_docs.append(
            build_channel_summary(
                ch, count,
                stats=channel_stats.get(cid),
                get_user_name=get_user_name,
            )
        )

    summary_docs.append(build_workspace_summary(channels, channel_counts))

    # Build user→channel activity map from accumulated stats
    user_channels: dict[str, Counter] = {}
    channel_names = {ch["id"]: ch.get("name", ch["id"]) for ch in channels}
    for cid, cstats in channel_stats.items():
        cname = channel_names.get(cid, cid)
        for uid, msg_count in cstats.user_counts.items():
            if uid not in user_channels:
                user_channels[uid] = Counter()
            user_channels[uid][cname] += msg_count

    profiles = list(user_profiles.values())
    for p in profiles:
        if not p.get("deleted"):
            uid = p["user_id"]
            active = [ch for ch, _ in user_channels.get(uid, Counter()).most_common(10)]
            summary_docs.append(build_user_summary(p, active_channels=active or None))
    summary_docs.append(build_team_summary(profiles))

    _flush(summary_docs, store, embedder, sparse_encoder)
    logger.info("Indexed %d summary documents", len(summary_docs))


def run_once(
    cfg: Config,
    slack: SlackClient,
    store: VectorStore,
    embedder: Embedder,
    sparse_encoder: Optional[SparseEncoder],
    state: SyncState,
) -> None:
    channels = slack.get_public_channels(cfg.channel_list or None)
    if not channels:
        logger.warning("No accessible public channels found")
        return

    stats_file = os.path.join(os.path.dirname(cfg.state_file), "channel_stats.json")
    all_stats = _load_channel_stats(stats_file)

    total = 0
    for ch in channels:
        count, new_stats = sync_channel(ch, slack, store, embedder, sparse_encoder, state, cfg)
        total += count
        cid = ch["id"]
        if cid in all_stats:
            all_stats[cid].merge(new_stats)
        else:
            all_stats[cid] = new_stats

    _save_channel_stats(stats_file, all_stats)

    if cfg.thread_update_lookback_hours > 0:
        logger.info("Starting thread-update refresh pass (%dh lookback)…",
                     cfg.thread_update_lookback_hours)
        for ch in channels:
            refresh_threads(ch, slack, store, embedder, sparse_encoder, cfg,
                            cfg.thread_update_lookback_hours)

    _index_summaries(
        channels, all_stats, slack.get_user_profiles(),
        slack.get_user_name, store, embedder, sparse_encoder,
    )
    logger.info("Sync complete — %d documents indexed  (total in store: %d)", total, store.count())


def main() -> None:
    cfg = Config.from_env()

    from . import __version__
    logger.info("Starting slack2rag v%s", __version__)
    logger.info("  Qdrant:         %s / %s", cfg.qdrant_url, cfg.qdrant_collection)
    logger.info("  Ollama:         %s / %s", cfg.ollama_url, cfg.ollama_embedding_model)
    logger.info("  Hybrid search:  %s", cfg.hybrid_search)
    logger.info("  Channels:       %s", cfg.channel_list or "all public")
    logger.info("  Min msg length: %d chars", cfg.min_message_length)
    logger.info("  Thread refresh: %s",
                f"{cfg.thread_update_lookback_hours}h lookback"
                if cfg.thread_update_lookback_hours else "disabled")
    logger.info("  Run once:       %s", cfg.run_once)

    slack = SlackClient(cfg.slack_bot_token, api_pause=cfg.api_pause)
    slack.prefetch_users()
    slack.fetch_workspace_url()

    try:
        embedder = Embedder(url=cfg.ollama_url, model=cfg.ollama_embedding_model,
                            context_length=cfg.ollama_context_length)
    except EmbeddingError as exc:
        logger.error("Embedding setup failed:\n  %s", exc)
        raise SystemExit(1) from exc

    sparse_encoder = SparseEncoder() if cfg.hybrid_search else None

    store = VectorStore(
        url=cfg.qdrant_url,
        collection=cfg.qdrant_collection,
        dimension=embedder.dimension,
        hybrid=cfg.hybrid_search,
    )
    state = SyncState(cfg.state_file)

    if cfg.run_once:
        run_once(cfg, slack, store, embedder, sparse_encoder, state)
        return

    interval = cfg.sync_interval_minutes * 60
    while True:
        try:
            run_once(cfg, slack, store, embedder, sparse_encoder, state)
        except EmbeddingError as exc:
            logger.error("Sync cycle failed (embedding error): %s", exc)
        except Exception:
            logger.exception("Sync cycle failed — will retry next interval")
        logger.info("Sleeping %d minutes until next sync…", cfg.sync_interval_minutes)
        time.sleep(interval)


if __name__ == "__main__":
    main()
