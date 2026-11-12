"""Tests for the `slack2rag` command line (slack2rag/__main__.py)."""

from __future__ import annotations

import dataclasses
import logging
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import slack2rag.__main__ as cli
from slack2rag import __version__, query
from slack2rag import main as indexer
from slack2rag.config import Config

_CONFIG_VARS = [f.name.upper() for f in dataclasses.fields(Config)] + [
    "SLACK_API_PAUSE"
]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """A clean environment in an empty working directory (so no stray .env is read)."""
    for var in _CONFIG_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    return monkeypatch


@pytest.fixture
def commands(env: pytest.MonkeyPatch) -> MagicMock:
    """Replace everything main() dispatches to with recorders."""
    m = MagicMock()
    m.indexer.return_value = 0
    m.query.return_value = 0
    env.setattr(indexer, "main", m.indexer)
    env.setattr(query, "main", m.query)
    env.setattr(cli, "migrate", m.migrate)
    env.setattr(cli, "_configure_logging", m.configure_logging)
    return m


# ── help and version (no settings needed) ────────────────────────────────────


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help(
    flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    assert cli.main([flag]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Command line: `slack2rag`")
    assert "slack2rag migrate" in out


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_version(
    flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    assert cli.main([flag]) == 0
    assert capsys.readouterr().out == f"slack2rag {__version__}\n"


def test_argv_defaults_to_sys_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["slack2rag", "--version"])
    assert cli.main() == 0
    assert __version__ in capsys.readouterr().out


# ── settings ─────────────────────────────────────────────────────────────────


def test_config_error_returns_two(
    commands: MagicMock, env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.delenv("SLACK_BOT_TOKEN")
    assert cli.main([]) == 2
    assert capsys.readouterr().err == "slack2rag: SLACK_BOT_TOKEN is required\n"
    commands.configure_logging.assert_not_called()
    commands.indexer.assert_not_called()


def test_bad_setting_returns_two(
    commands: MagicMock, env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.setenv("BATCH_SIZE", "lots")
    assert cli.main(["query", "x"]) == 2
    assert "BATCH_SIZE must be a whole number" in capsys.readouterr().err
    commands.query.assert_not_called()


def test_dotenv_in_working_directory_is_loaded(
    commands: MagicMock, env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "QDRANT_COLLECTION=from-dotenv\nSLACK_BOT_TOKEN=xoxb-dotenv\n"
    )
    # Register both so monkeypatch removes whatever load_dotenv sets.
    env.setenv("QDRANT_COLLECTION", "placeholder")
    env.delenv("QDRANT_COLLECTION")

    assert cli.main([]) == 0

    cfg = commands.indexer.call_args.kwargs["cfg"]
    assert cfg.qdrant_collection == "from-dotenv"
    # The real environment wins over .env.
    assert cfg.slack_bot_token == "xoxb-test"  # noqa: S105 dummy test token


def test_logging_configured_from_log_level(
    commands: MagicMock, env: pytest.MonkeyPatch
) -> None:
    env.setenv("LOG_LEVEL", "debug")
    assert cli.main([]) == 0
    commands.configure_logging.assert_called_once_with("DEBUG")


def test_configure_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    basic_config = MagicMock()
    monkeypatch.setattr(logging, "basicConfig", basic_config)
    cli._configure_logging("WARNING")
    kwargs = basic_config.call_args.kwargs
    assert kwargs["level"] == "WARNING"
    assert kwargs["stream"] is sys.stdout
    assert "%(levelname)" in kwargs["format"]


# ── run (the default) ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "passed"),
    [
        ([], []),
        (["--check"], ["--check"]),
        (["run"], []),
        (["run", "--check"], ["--check"]),
    ],
)
def test_run_dispatches_to_indexer(
    commands: MagicMock, argv: list[str], passed: list[str]
) -> None:
    commands.indexer.return_value = 7
    assert cli.main(argv) == 7
    args, kwargs = commands.indexer.call_args
    assert args == (passed,)
    assert isinstance(kwargs["cfg"], Config)
    assert kwargs["cfg"].slack_bot_token == "xoxb-test"  # noqa: S105 dummy test token
    commands.query.assert_not_called()
    commands.migrate.assert_not_called()


def test_keyboard_interrupt_returns_130(
    commands: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    commands.indexer.side_effect = KeyboardInterrupt
    with caplog.at_level(logging.INFO, logger="slack2rag"):
        assert cli.main([]) == 130
    assert "Interrupted" in caplog.text


# ── query ────────────────────────────────────────────────────────────────────


def test_query_dispatches_with_remaining_args(commands: MagicMock) -> None:
    commands.query.return_value = 1
    assert cli.main(["query", "-n", "3", "deploy", "process"]) == 1
    args, kwargs = commands.query.call_args
    assert args == (["-n", "3", "deploy", "process"],)
    assert isinstance(kwargs["cfg"], Config)
    commands.indexer.assert_not_called()


# ── migrate ──────────────────────────────────────────────────────────────────


def test_migrate(commands: MagicMock, env: pytest.MonkeyPatch) -> None:
    env.setenv("DATABASE_URL", "postgresql://u@db/x")
    assert cli.main(["migrate"]) == 0
    commands.migrate.assert_called_once_with("postgresql://u@db/x")
    commands.indexer.assert_not_called()


def test_migrate_rejects_arguments(
    commands: MagicMock, env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.setenv("DATABASE_URL", "postgresql://u@db/x")
    assert cli.main(["migrate", "head"]) == 2
    assert "takes no arguments" in capsys.readouterr().err
    commands.migrate.assert_not_called()


def test_migrate_needs_database_url(
    commands: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["migrate"]) == 2
    assert "DATABASE_URL is not set" in capsys.readouterr().err
    commands.migrate.assert_not_called()


# ── python -m slack2rag ──────────────────────────────────────────────────────


def test_module_entry_point_exits_with_main_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["slack2rag", "--version"])
    monkeypatch.delitem(sys.modules, "slack2rag.__main__")
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("slack2rag", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out == f"slack2rag {__version__}\n"
