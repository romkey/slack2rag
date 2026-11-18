"""
slack2rag worker: mines Slack for facts and keeps them in Qdrant.

Schedule
--------
* Hourly (SYNC_INTERVAL_MINUTES): catch-up of every channel the bot is in.
* Nightly (RECONCILE_HOUR, UTC): full reconcile with the safety valve.
* Weekly (REPORT_WEEKDAY): the wiki promotion queue, in REPORT_DIR.
* Live (SLACK_APP_TOKEN set): Socket Mode events mark units; they are
  processed within a minute.

Each job holds a PostgreSQL advisory lock, so two workers never run at
once, and writes its status row. One failed run is logged and retried; it
never stops the service.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import signal
import threading
import time
from dataclasses import dataclass
from types import FrameType

from .config import Config
from .embedder import Embedder, EmbeddingError, SparseEncoder
from .extractor import Extractor, load_topics
from .llm import ChatModel, LLMError
from .pg_store import PgStore, SchemaError
from .pipeline import Pipeline
from .realtime import LiveEvents
from .report import write_report
from .slack_client import SlackClient
from .vector_store import CollectionMismatchError, VectorStore

logger = logging.getLogger(__name__)

# Set by SIGTERM (`docker stop`) or Ctrl-C to end the loop between runs.
stop_event = threading.Event()

LIVE_DEBOUNCE_SECONDS = 30


def make_embedder(cfg: Config) -> Embedder:
    return Embedder(
        base_url=cfg.llm_base_url,
        model=cfg.embedding_model,
        context_length=cfg.embedding_context_length,
        api_key=cfg.llm_api_key,
        query_prefix=cfg.embedding_query_prefix,
        document_prefix=cfg.embedding_document_prefix,
    )


def make_chat(cfg: Config) -> ChatModel:
    return ChatModel(
        cfg.effective_chat_base_url, cfg.chat_model, cfg.effective_chat_api_key
    )


def build_pipeline(cfg: Config, store: PgStore | None = None) -> Pipeline:
    """Connect everything. Raises on a missing service or bad schema."""
    store = store or PgStore(cfg.database_url)
    store.check_schema()
    slack = SlackClient(cfg.slack_bot_token, api_pause=cfg.api_pause)
    slack.identify()
    embedder = make_embedder(cfg)
    vectors = VectorStore(
        cfg.qdrant_url, cfg.qdrant_collection, embedder.dimension, cfg.embedding_model
    )
    extractor = Extractor(
        make_chat(cfg), load_topics(cfg.topics_file), slack.member_names()
    )
    return Pipeline(cfg, slack, store, vectors, embedder, extractor, SparseEncoder())


def _check(cfg: Config) -> bool:
    """Probe the embedding model, the chat model and Slack; log each result."""
    ok = True
    try:
        vector = make_embedder(cfg).embed(["slack2rag check"], kind="query")[0]
        logger.info(
            "  ✓ Embedding model %s (dimension %d)", cfg.embedding_model, len(vector)
        )
    except EmbeddingError as exc:
        logger.error("  ✗ Embedding model: %s", exc)
        ok = False
    try:
        answer = make_chat(cfg).chat_json(
            'Reply with JSON only: {"ok": true}.', "This is a connectivity check."
        )
        logger.info("  ✓ Chat model %s (%s)", cfg.chat_model, answer)
    except LLMError as exc:
        logger.error("  ✗ Chat model: %s", exc)
        ok = False
    try:
        me = SlackClient(cfg.slack_bot_token, api_pause=0).identify()
        logger.info("  ✓ Slack (%s, bot user %s)", me.workspace_url, me.user_id)
    except Exception as exc:
        logger.error("  ✗ Slack: %s", exc)
        ok = False
    return ok


@dataclass
class Schedule:
    """When each job last ran, and whether it is due."""

    cfg: Config
    last_catchup: float = 0.0
    last_reconcile: datetime.date | None = None
    last_report: datetime.date | None = None

    def catchup_due(self, now: float) -> bool:
        return now - self.last_catchup >= self.cfg.sync_interval_minutes * 60

    def reconcile_due(self, now: datetime.datetime) -> bool:
        return now.hour == self.cfg.reconcile_hour and self.last_reconcile != now.date()

    def report_due(self, now: datetime.datetime) -> bool:
        return (
            now.weekday() == self.cfg.report_weekday
            and now.hour >= self.cfg.reconcile_hour
            and self.last_report != now.date()
        )


def run_job(pipeline: Pipeline, job: str) -> bool:
    """Run one job under the source lock; False if it failed or was skipped."""
    with pipeline.store.lock() as got:
        if not got:
            logger.warning("Another slack2rag worker holds the lock; skipping %s", job)
            return False
        try:
            if job == "catchup":
                pipeline.catchup()
            elif job == "reconcile":
                pipeline.extractor.member_names = frozenset(
                    pipeline.slack.member_names()
                )
                pipeline.reconcile()
            elif job == "live":
                pipeline.process_dirty()
            elif job == "report":
                write_report(
                    pipeline.store,
                    pipeline.vectors,
                    pipeline.cfg.report_dir,
                    datetime.datetime.now(tz=datetime.UTC).date(),
                )
        except Exception:
            # One bad run (a Slack or model outage) mustn't stop the service.
            logger.exception("The %s run failed; it will be retried", job)
            return False
    return True


def _request_stop(signum: int, frame: FrameType | None) -> None:
    logger.info("Received signal %d; stopping after the current run", signum)
    stop_event.set()


def loop(pipeline: Pipeline, live: LiveEvents | None) -> None:
    cfg = pipeline.cfg
    schedule = Schedule(cfg)
    if pipeline.store.last_run("reconcile") is None:
        logger.info("No full reconcile has run yet; starting with one")
        if run_job(pipeline, "reconcile"):
            schedule.last_reconcile = datetime.datetime.now(tz=datetime.UTC).date()
            schedule.last_catchup = time.time()
    while not stop_event.is_set():
        now = time.time()
        today = datetime.datetime.now(tz=datetime.UTC)
        if live is not None:
            for channel in live.take_forgotten():
                with pipeline.store.lock() as got:
                    if got:
                        pipeline.forget_channel(channel)
            if live.wake.is_set():
                live.wake.clear()
                stop_event.wait(LIVE_DEBOUNCE_SECONDS)
                run_job(pipeline, "live")
        if schedule.reconcile_due(today):
            if run_job(pipeline, "reconcile"):
                schedule.last_reconcile = today.date()
                schedule.last_catchup = now
        elif schedule.catchup_due(now):
            run_job(pipeline, "catchup")
            schedule.last_catchup = now
        if schedule.report_due(today) and run_job(pipeline, "report"):
            schedule.last_report = today.date()
        if live is not None:
            live.wake.wait(timeout=60)
        else:
            stop_event.wait(60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="slack2rag", description="Mine Slack for facts and index them in Qdrant."
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the embedding model, chat model and Slack token, then exit "
        "(0 if all work, 1 if not).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, cfg: Config | None = None) -> int:
    """Run the worker; returns the process exit code."""
    args = _parse_args(argv)
    if cfg is None:
        cfg = Config.from_env()

    from . import __version__

    logger.info("Starting slack2rag v%s", __version__)
    for warning in cfg.missing_api_key_warnings():
        logger.warning("%s", warning)

    if args.check:
        if _check(cfg):
            logger.info("Check: SUCCESS")
            return 0
        logger.error("Check: FAILURE")
        return 1

    logger.info("  Qdrant:        %s / %s", cfg.qdrant_url, cfg.qdrant_collection)
    logger.info("  Gateway:       %s", cfg.llm_base_url)
    logger.info("  Embeddings:    %s", cfg.embedding_model)
    logger.info(
        "  Chat model:    %s at %s", cfg.chat_model, cfg.effective_chat_base_url
    )
    logger.info("  Live events:   %s", "on" if cfg.slack_app_token else "off")
    logger.info(
        "  Denylist:      %s", ", ".join(sorted(cfg.slack_channel_denylist)) or "none"
    )

    try:
        pipeline = build_pipeline(cfg)
    except SchemaError as exc:
        logger.error("%s", exc)
        return 1
    except (EmbeddingError, CollectionMismatchError) as exc:
        logger.error("%s", exc)
        return 1

    if cfg.run_once:
        ok = run_job(pipeline, "catchup")
        return 0 if ok else 1

    live = None
    if cfg.slack_app_token:
        live = LiveEvents(pipeline.store, pipeline.me, cfg.slack_channel_denylist)
        live.start(cfg.slack_app_token, pipeline.slack.web_client)

    signal.signal(signal.SIGTERM, _request_stop)
    try:
        loop(pipeline, live)
    finally:
        if live is not None:
            live.stop()
    logger.info("Stopped")
    return 0
