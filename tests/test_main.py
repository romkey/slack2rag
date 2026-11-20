"""Worker, CLI and query command tests."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient

from slack2rag import __main__ as cli
from slack2rag import main as worker
from slack2rag import query
from slack2rag.embedder import EmbeddingError
from slack2rag.llm import LLMError
from slack2rag.pg_store import PgStore, SchemaError
from slack2rag.pipeline import Pipeline

from .conftest import DAY, Clock, FakeSlack, make_config

ENV = {
    "SLACK_BOT_TOKEN": "xoxb-1",
    "DATABASE_URL": "postgresql://x@db/x",
    "CHAT_MODEL": "qwen",
}


# ── schedule and jobs ─────────────────────────────────────────────────────


def test_schedule() -> None:
    s = worker.Schedule(make_config(reconcile_hour=3, report_weekday=0))
    assert s.catchup_due(10_000)
    s.last_catchup = 10_000
    assert not s.catchup_due(10_100)
    monday_3am = datetime.datetime(2026, 5, 25, 3, tzinfo=datetime.UTC)
    assert s.reconcile_due(monday_3am)
    assert not s.reconcile_due(monday_3am.replace(hour=4))
    s.last_reconcile = monday_3am.date()
    assert not s.reconcile_due(monday_3am)
    assert s.report_due(monday_3am)
    assert not s.report_due(monday_3am + datetime.timedelta(days=1))


def test_run_job_each_kind(
    pipeline: Pipeline, slack: FakeSlack, clock: Clock, tmp_path: Path
) -> None:
    slack.post("C1", clock.now - 2 * DAY, "FACT howto|How?|Like this.")
    pipeline.cfg = make_config(report_dir=str(tmp_path))
    for job in (
        "catchup",
        "reconcile",
        "live",
        "report",
        "rebuild",
        "rebuild-reextract",
    ):
        assert worker.run_job(pipeline, job)
    assert list(tmp_path.glob("wiki-promotion-*.md"))


def test_run_job_skips_when_locked_and_survives_failure(pipeline: Pipeline) -> None:
    with pipeline.store.lock():
        assert not worker.run_job(cast(Pipeline, PipelineProxy(pipeline)), "catchup")
    with patch.object(pipeline, "catchup", side_effect=RuntimeError("down")):
        assert not worker.run_job(pipeline, "catchup")


class PipelineProxy:
    """The same pipeline over a second connection, to contend for the lock."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.store = PgStore(pipeline.store._conninfo)
        self.cfg = pipeline.cfg

    def catchup(self) -> None:  # pragma: no cover - must not run
        raise AssertionError("ran while locked")


def test_loop_reconciles_first_then_stops(pipeline: Pipeline) -> None:
    calls: list[str] = []

    def fake_run(p: Any, job: str) -> bool:
        calls.append(job)
        worker.stop_event.set()
        return True

    live = MagicMock()
    live.take_forgotten.return_value = {"C9"}
    live.wake.is_set.return_value = True
    with patch.object(worker, "run_job", fake_run):
        worker.stop_event.clear()
        worker.loop(pipeline, None)
        worker.stop_event.clear()
        with patch.object(worker, "LIVE_DEBOUNCE_SECONDS", 0):
            worker.stop_event.set()
            worker.loop(pipeline, live)
    worker.stop_event.clear()
    assert calls[0] == "reconcile"


def test_loop_body_runs_live_and_catchup(pipeline: Pipeline) -> None:
    pipeline.store.start_run("reconcile")
    run = pipeline.store.last_run("reconcile", status="running")
    assert run is not None
    pipeline.store.finish_run(
        int(run["id"]),
        __import__("slack2rag.pg_store", fromlist=["RunCounts"]).RunCounts(),
        "ok",
    )
    jobs: list[str] = []
    live = MagicMock()
    live.take_forgotten.return_value = {"C9"}
    live.wake.is_set.return_value = True

    def fake_run(p: Any, job: str) -> bool:
        jobs.append(job)
        if len(jobs) >= 2:
            worker.stop_event.set()
        return True

    worker.stop_event.clear()
    with (
        patch.object(worker, "run_job", fake_run),
        patch.object(worker, "LIVE_DEBOUNCE_SECONDS", 0),
        patch.object(pipeline, "forget_channel") as forget,
    ):
        worker.loop(pipeline, live)
    worker.stop_event.clear()
    forget.assert_called_once_with("C9")
    assert jobs[0] == "live"


def test_check(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_config()
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2]]
    chat = MagicMock()
    chat.chat_json.return_value = {"ok": True}
    slack = MagicMock()
    with (
        patch.object(worker, "make_embedder", return_value=embedder),
        patch.object(worker, "make_chat", return_value=chat),
        patch.object(worker, "SlackClient", return_value=slack),
    ):
        assert worker._check(cfg)
        assert worker.main(["--check"], cfg=cfg) == 0
        embedder.embed.side_effect = EmbeddingError("no")
        chat.chat_json.side_effect = LLMError("no")
        slack.identify.side_effect = RuntimeError("no")
        assert not worker._check(cfg)
        assert worker.main(["--check"], cfg=cfg) == 1


def test_main_startup_failures_and_run_once() -> None:
    cfg = make_config(run_once=True)
    with patch.object(worker, "build_pipeline", side_effect=SchemaError("old")):
        assert worker.main([], cfg=cfg) == 1
    with patch.object(worker, "build_pipeline", side_effect=EmbeddingError("down")):
        assert worker.main([], cfg=cfg) == 1
    with (
        patch.object(worker, "build_pipeline", return_value=MagicMock()),
        patch.object(worker, "run_job", return_value=True),
    ):
        assert worker.main([], cfg=cfg) == 0


def test_main_runs_loop_with_live_events() -> None:
    cfg = make_config(slack_app_token="xapp-1-x")
    with (
        patch.object(worker, "build_pipeline", return_value=MagicMock()),
        patch.object(worker, "LiveEvents") as live_cls,
        patch.object(worker, "loop") as loop,
        patch("slack2rag.main.signal.signal"),
    ):
        assert worker.main([], cfg=cfg) == 0
    loop.assert_called_once()
    live_cls.return_value.start.assert_called_once()
    live_cls.return_value.stop.assert_called_once()


def test_build_pipeline(store: PgStore, qdrant: QdrantClient) -> None:
    cfg = make_config()
    embedder = MagicMock(dimension=4)
    slack = MagicMock()
    slack.identify.return_value = __import__("tests.conftest", fromlist=["ME"]).ME
    slack.member_names.return_value = {"Alice"}
    with (
        patch.object(worker, "make_embedder", return_value=embedder),
        patch.object(worker, "SlackClient", return_value=slack),
        patch("slack2rag.vector_store.QdrantClient", return_value=qdrant),
    ):
        pipeline = worker.build_pipeline(cfg, store)
    assert pipeline.extractor.member_names == frozenset({"Alice"})
    assert worker.make_chat(cfg).model == "fake-model"


# ── command line ──────────────────────────────────────────────────────────


def test_cli_help_version_and_config_errors(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["--help"]) == 0
    assert "slack2rag canary" in capsys.readouterr().out
    assert cli.main(["--version"]) == 0
    monkeypatch.setattr(cli, "load_dotenv", lambda path: None)
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    assert cli.main(["status"]) == 2
    assert "SLACK_BOT_TOKEN is required" in capsys.readouterr().err


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, pg_url: str) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda path: None)
    monkeypatch.setattr(cli, "_configure_logging", lambda level: None)
    for key, value in {**ENV, "DATABASE_URL": pg_url}.items():
        monkeypatch.setenv(key, value)


def test_cli_store_commands(
    env: None, store: PgStore, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["status"]) == 0
    assert "No run has finished yet" in capsys.readouterr().out
    run = store.start_run("catchup")
    store.finish_run(
        run,
        __import__("slack2rag.pg_store", fromlist=["RunCounts"]).RunCounts(added=2),
        "ok",
    )
    assert cli.main(["status"]) == 0
    assert "added" in capsys.readouterr().out
    assert cli.main(["reprocess"]) == 0
    assert cli.main(["migrate"]) == 0
    assert cli.main(["migrate", "extra"]) == 2
    assert cli.main(["spot-check", "x"]) == 2
    with patch("slack2rag.slack_client.SlackClient.member_names", return_value=set()):
        assert cli.main(["spot-check", "5"]) == 0


def test_cli_pipeline_commands(
    env: None,
    pipeline: Pipeline,
    slack: FakeSlack,
    clock: Clock,
    qdrant: QdrantClient,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    slack.post("C1", clock.now - 2 * DAY, "FACT howto|How?|Like this.")
    pipeline.catchup()
    with patch.object(worker, "build_pipeline", return_value=pipeline):
        assert cli.main(["report"]) == 0
        assert "Wiki promotion queue" in capsys.readouterr().out
        assert cli.main(["label-template", "5"]) == 0
        line = capsys.readouterr().out.splitlines()[0]
        label = json.loads(line) | {"relevant": True}
        labels = tmp_path / "l.jsonl"
        labels.write_text(json.dumps(label))
        assert cli.main(["eval-classifier", str(labels)]) == 0
        assert "precision" in capsys.readouterr().out
        assert cli.main(["eval-classifier"]) == 2
        assert cli.main(["reconcile"]) == 0
        assert cli.main(["rebuild"]) == 0
        assert cli.main(["rebuild", "--reextract"]) == 0
        assert cli.main(["rebuild", "--everything"]) == 2
        assert cli.main(["report", "x"]) == 2
    with patch("qdrant_client.QdrantClient", return_value=qdrant):
        assert cli.main(["canary", "NOT-THERE"]) == 0
        assert cli.main(["canary", "Like this"]) == 1
        assert cli.main(["canary"]) == 2


def test_cli_dispatches_run_and_query(env: None) -> None:
    with patch.object(worker, "main", return_value=0) as run:
        assert cli.main([]) == 0
        assert cli.main(["run", "--check"]) == 0
    assert run.call_args.args[0] == ["--check"]
    with patch.object(query, "main", side_effect=KeyboardInterrupt):
        assert cli.main(["query", "x"]) == 130


# ── query ─────────────────────────────────────────────────────────────────


def test_query_filter() -> None:
    flt = query._filter("C123ABC", "howto", current=True)
    keys = [c.key for c in flt.must]  # type: ignore[union-attr]
    assert keys == ["source", "channel_id", "kind"]
    assert flt.should
    flt = query._filter("#laser", None, current=False)
    assert [c.key for c in flt.must] == ["source", "channel_name"]  # type: ignore[union-attr]
    assert flt.should is None


def test_query_main(capsys: pytest.CaptureFixture[str]) -> None:
    store = MagicMock()
    store.search.return_value = [
        {
            "score": 0.5,
            "kind": "howto",
            "label": "Slack discussion, 2026-05-01",
            "channel_name": "laser",
            "text": "Q: How?\nA: Like this.",
            "links": ["https://x.org"],
            "url": "https://ws/p1",
            "superseded_by": "abc",
            "confidence": "high",
        }
    ]
    embedder = MagicMock(dimension=4)
    embedder.embed.return_value = [[0.1] * 4]
    with (
        patch.object(query, "Embedder", return_value=embedder),
        patch.object(query, "VectorStore", return_value=store),
    ):
        assert query.main(["how", "to"], cfg=make_config()) == 0
        out = capsys.readouterr().out
        assert "(superseded)" in out
        assert "link: https://x.org" in out
        store.search.return_value = []
        assert query.main(["x", "--all"], cfg=make_config()) == 0
        assert "No results" in capsys.readouterr().out
    with patch.object(query, "Embedder", side_effect=EmbeddingError("down")):
        assert query.main(["x"], cfg=make_config()) == 1
