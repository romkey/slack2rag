"""Config loading tests."""

from __future__ import annotations

from src.config import Config


def test_database_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/app")
    cfg = Config.from_env()
    assert cfg.database_url == "postgresql://u:p@db:5432/app"


def test_database_url_default_empty(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    cfg = Config.from_env()
    assert cfg.database_url == ""
