"""Config tests."""

from __future__ import annotations

import pytest

from slack2rag.config import RETIRED, Config, ConfigError

BASE = {
    "SLACK_BOT_TOKEN": "xoxb-1",
    "DATABASE_URL": "postgresql://x@db/x",
    "CHAT_MODEL": "qwen",
}


def _cfg(**env: str) -> Config:
    return Config.from_env({**BASE, **env})


def test_defaults() -> None:
    cfg = _cfg()
    assert cfg.qdrant_collection == "hs_member"
    assert cfg.slack_app_token == ""
    assert cfg.effective_chat_base_url == cfg.llm_base_url
    assert cfg.session_gap_minutes == 30
    assert cfg.quiet_hours == 3.0
    assert cfg.slack_channel_denylist == frozenset()


@pytest.mark.parametrize("missing", ["SLACK_BOT_TOKEN", "DATABASE_URL", "CHAT_MODEL"])
def test_required(missing: str) -> None:
    env = {k: v for k, v in BASE.items() if k != missing}
    with pytest.raises(ConfigError, match=missing):
        Config.from_env(env)


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_retired_variables_are_refused(name: str) -> None:
    with pytest.raises(ConfigError, match=f"{name} is no longer used"):
        _cfg(**{name: ""})


def test_user_token_is_refused() -> None:
    with pytest.raises(ConfigError, match="user token"):
        _cfg(SLACK_BOT_TOKEN="xoxp-1")


def test_app_token_must_be_app_level() -> None:
    with pytest.raises(ConfigError, match="xapp"):
        _cfg(SLACK_APP_TOKEN="xoxb-2")
    assert _cfg(SLACK_APP_TOKEN="xapp-1-abc").slack_app_token == "xapp-1-abc"


def test_public_collection_is_refused() -> None:
    with pytest.raises(ConfigError, match="member tier"):
        _cfg(QDRANT_COLLECTION="hs_public")
    with pytest.raises(ConfigError, match="empty"):
        _cfg(QDRANT_COLLECTION=" ")


def test_lists_strip_hashes() -> None:
    cfg = _cfg(SLACK_CHANNEL_DENYLIST="#board, C123 ,", SLACK_BOT_ALLOWLIST="announcer")
    assert cfg.slack_channel_denylist == frozenset({"board", "C123"})
    assert cfg.slack_bot_allowlist == frozenset({"announcer"})


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("SESSION_GAP_MINUTES", "0", "at least 1"),
        ("RECONCILE_HOUR", "24", "at most 23"),
        ("REPORT_WEEKDAY", "x", "whole number"),
        ("SUPERSEDE_SIMILARITY", "1.5", "at most 1.0"),
        ("QUIET_HOURS", "-1", "at least 0"),
        ("SLACK_API_PAUSE", "fast", "a number"),
        ("RUN_ONCE", "maybe", "true or false"),
        ("LOG_LEVEL", "LOUD", "logging level"),
    ],
)
def test_bad_values(name: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        _cfg(**{name: value})


def test_overlap_must_be_smaller_than_cap() -> None:
    with pytest.raises(ConfigError, match="less than"):
        _cfg(SESSION_MAX_MESSAGES="3", SESSION_OVERLAP_MESSAGES="3")


def test_numbers_and_booleans_parse() -> None:
    cfg = _cfg(RUN_ONCE="yes", BACKFILL_DAYS="90", QUIET_HOURS="1.5", LOG_LEVEL="debug")
    assert cfg.run_once is True
    assert cfg.backfill_days == 90
    assert cfg.quiet_hours == 1.5
    assert cfg.log_level == "DEBUG"


def test_openai_without_key_warns() -> None:
    cfg = _cfg(LLM_BASE_URL="https://api.openai.com/v1")
    warnings = cfg.missing_api_key_warnings()
    assert len(warnings) == 2
    cfg = _cfg(LLM_BASE_URL="https://api.openai.com/v1", LLM_API_KEY="k")
    assert cfg.missing_api_key_warnings() == []
    chat = _cfg(CHAT_BASE_URL="https://api.openai.com/v1", CHAT_API_KEY="k")
    assert chat.effective_chat_api_key == "k"
    assert chat.missing_api_key_warnings() == []
