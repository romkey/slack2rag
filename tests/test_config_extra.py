"""Additional Config tests: derived properties, defaults, env parsing."""

from __future__ import annotations

import pytest

from src.config import Config, _bool_env

_ALL_VARS = (
    "QDRANT_URL", "QDRANT_COLLECTION", "DATABASE_URL", "SLACK_CHANNELS",
    "SLACK_CHANNEL_BLACKLIST", "LLM_BASE_URL", "LLM_API_KEY", "EMBEDDING_MODEL",
    "EMBEDDING_PREFIX", "EMBEDDING_CONTEXT_LENGTH", "SYNC_INTERVAL_MINUTES", "RUN_ONCE",
    "STATE_FILE", "BATCH_SIZE", "SLACK_API_PAUSE", "MIN_MESSAGE_LENGTH", "SCORE_THRESHOLD",
    "HYBRID_SEARCH", "THREAD_UPDATE_LOOKBACK_HOURS", "REACTION_BOOST_THRESHOLD",
    "EVAL_TEST", "EVAL_PROMPT", "EVAL_MODEL", "EVAL_BASE_URL", "EVAL_API_KEY",
)


@pytest.fixture
def clean_env(monkeypatch):
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    return monkeypatch


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("TRUE", True), ("Yes", True),
     ("0", False), ("false", False), ("no", False), ("", False), ("on", False)],
)
def test_bool_env(value, expected) -> None:
    assert _bool_env(value) is expected


def test_missing_token_raises(monkeypatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(EnvironmentError, match="SLACK_BOT_TOKEN is required"):
        Config.from_env()


def test_empty_token_raises(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "")
    with pytest.raises(EnvironmentError):
        Config.from_env()


def test_defaults(clean_env) -> None:
    cfg = Config.from_env()
    assert cfg == Config(slack_bot_token="xoxb-test")
    assert cfg.channel_list == []
    assert cfg.channel_blacklist == set()
    assert cfg.run_once is False
    assert cfg.hybrid_search is False
    assert cfg.eval_test is False


def test_numeric_and_bool_parsing(clean_env) -> None:
    env = {
        "EMBEDDING_CONTEXT_LENGTH": "0",
        "SYNC_INTERVAL_MINUTES": "15",
        "RUN_ONCE": "yes",
        "BATCH_SIZE": "7",
        "SLACK_API_PAUSE": "0.25",
        "MIN_MESSAGE_LENGTH": "3",
        "SCORE_THRESHOLD": "0.42",
        "HYBRID_SEARCH": "1",
        "THREAD_UPDATE_LOOKBACK_HOURS": "48",
        "REACTION_BOOST_THRESHOLD": "5",
        "STATE_FILE": "/tmp/s.json",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION": "c2",
    }
    for k, v in env.items():
        clean_env.setenv(k, v)
    cfg = Config.from_env()
    assert cfg.embedding_context_length == 0
    assert cfg.sync_interval_minutes == 15
    assert cfg.run_once is True
    assert cfg.batch_size == 7
    assert cfg.api_pause == 0.25
    assert cfg.min_message_length == 3
    assert cfg.score_threshold == 0.42
    assert cfg.hybrid_search is True
    assert cfg.thread_update_lookback_hours == 48
    assert cfg.reaction_boost_threshold == 5
    assert cfg.state_file == "/tmp/s.json"
    assert cfg.qdrant_url == "http://localhost:6333"
    assert cfg.qdrant_collection == "c2"


def test_invalid_int_raises(clean_env) -> None:
    clean_env.setenv("BATCH_SIZE", "lots")
    with pytest.raises(ValueError):
        Config.from_env()


def test_channel_list_parsing(clean_env) -> None:
    clean_env.setenv("SLACK_CHANNELS", " eng, ,ops ,C123,")
    assert Config.from_env().channel_list == ["eng", "ops", "C123"]


def test_channel_blacklist_strips_hash(clean_env) -> None:
    clean_env.setenv("SLACK_CHANNEL_BLACKLIST", "#random, social ,, C999")
    assert Config.from_env().channel_blacklist == {"random", "social", "C999"}


def test_eval_settings_and_effective_fallbacks(clean_env) -> None:
    clean_env.setenv("LLM_BASE_URL", "http://llm/v1")
    clean_env.setenv("LLM_API_KEY", "llm-key")
    clean_env.setenv("EVAL_TEST", "1")
    clean_env.setenv("EVAL_PROMPT", "rate")
    clean_env.setenv("EVAL_MODEL", "judge")
    cfg = Config.from_env()
    assert cfg.eval_test is True
    assert cfg.eval_prompt == "rate"
    assert cfg.eval_model == "judge"
    assert cfg.effective_eval_base_url == "http://llm/v1"
    assert cfg.effective_eval_api_key == "llm-key"

    clean_env.setenv("EVAL_BASE_URL", "http://eval/v1")
    clean_env.setenv("EVAL_API_KEY", "eval-key")
    cfg = Config.from_env()
    assert cfg.effective_eval_base_url == "http://eval/v1"
    assert cfg.effective_eval_api_key == "eval-key"


@pytest.mark.xfail(
    strict=True,
    reason="src bug: EVAL_TEST uses bool(str) instead of _bool_env, so 'false'/'0' enable eval mode",
)
@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_eval_test_false_strings_disable(clean_env, value) -> None:
    clean_env.setenv("EVAL_TEST", value)
    assert Config.from_env().eval_test is False
