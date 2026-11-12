"""Additional Config tests: derived properties, defaults, env parsing."""

from __future__ import annotations

import dataclasses

import pytest

from slack2rag.config import Config, ConfigError

_ALL_VARS = (
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "DATABASE_URL",
    "SLACK_CHANNELS",
    "SLACK_CHANNEL_BLACKLIST",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_PREFIX",
    "EMBEDDING_CONTEXT_LENGTH",
    "SYNC_INTERVAL_MINUTES",
    "RUN_ONCE",
    "STATE_FILE",
    "BATCH_SIZE",
    "SLACK_API_PAUSE",
    "MIN_MESSAGE_LENGTH",
    "SCORE_THRESHOLD",
    "HYBRID_SEARCH",
    "THREAD_UPDATE_LOOKBACK_HOURS",
    "REACTION_BOOST_THRESHOLD",
    "EVAL_TEST",
    "EVAL_PROMPT",
    "EVAL_MODEL",
    "EVAL_BASE_URL",
    "EVAL_API_KEY",
    "LOG_LEVEL",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    return monkeypatch


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" Yes ", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("", False),
        ("Off", False),
    ],
)
def test_bool_parsing(value: str, expected: bool) -> None:
    env = {"SLACK_BOT_TOKEN": "xoxb-test", "RUN_ONCE": value, "HYBRID_SEARCH": value}
    cfg = Config.from_env(env)
    assert cfg.run_once is expected
    assert cfg.hybrid_search is expected


def test_invalid_bool_raises_naming_variable() -> None:
    with pytest.raises(
        ConfigError, match="HYBRID_SEARCH must be true or false, not 'maybe'"
    ):
        Config.from_env({"SLACK_BOT_TOKEN": "xoxb-test", "HYBRID_SEARCH": "maybe"})


def test_config_error_is_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


def test_from_env_reads_os_environ_by_default(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("QDRANT_COLLECTION", "from-os")
    assert Config.from_env().qdrant_collection == "from-os"


def test_config_is_frozen() -> None:
    cfg = Config(slack_bot_token="xoxb-test")  # noqa: S106 dummy test token
    with pytest.raises(dataclasses.FrozenInstanceError):
        # Assigning to a frozen dataclass is the point of the test.
        cfg.batch_size = 1  # type: ignore[misc]


def test_missing_token_raises() -> None:
    with pytest.raises(ConfigError, match="SLACK_BOT_TOKEN is required"):
        Config.from_env({})


def test_empty_token_raises() -> None:
    with pytest.raises(ConfigError, match="SLACK_BOT_TOKEN"):
        Config.from_env({"SLACK_BOT_TOKEN": ""})


def test_defaults(clean_env: pytest.MonkeyPatch) -> None:
    cfg = Config.from_env()
    assert cfg == Config(slack_bot_token="xoxb-test")  # noqa: S106 dummy test token
    assert cfg.channel_list == []
    assert cfg.channel_blacklist == set()
    assert cfg.run_once is False
    assert cfg.hybrid_search is False
    assert cfg.eval_test is False


def test_numeric_and_bool_parsing(clean_env: pytest.MonkeyPatch) -> None:
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
        "STATE_FILE": "/var/lib/slack2rag/s.json",
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
    assert cfg.state_file == "/var/lib/slack2rag/s.json"
    assert cfg.qdrant_url == "http://localhost:6333"
    assert cfg.qdrant_collection == "c2"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("BATCH_SIZE", "lots", "BATCH_SIZE must be a whole number, not 'lots'"),
        ("MIN_MESSAGE_LENGTH", "1.5", "MIN_MESSAGE_LENGTH must be a whole number"),
        ("BATCH_SIZE", "0", "BATCH_SIZE must be at least 1, not 0"),
        ("SYNC_INTERVAL_MINUTES", "0", "SYNC_INTERVAL_MINUTES must be at least 1"),
        (
            "REACTION_BOOST_THRESHOLD",
            "-1",
            "REACTION_BOOST_THRESHOLD must be at least 0",
        ),
        ("SLACK_API_PAUSE", "fast", "SLACK_API_PAUSE must be a number, not 'fast'"),
        ("SCORE_THRESHOLD", "-0.1", "SCORE_THRESHOLD must be at least 0.0, not -0.1"),
    ],
)
def test_invalid_numbers_raise_naming_variable(
    name: str, value: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        Config.from_env({"SLACK_BOT_TOKEN": "xoxb-test", name: value})


def test_blank_numbers_use_defaults() -> None:
    cfg = Config.from_env(
        {"SLACK_BOT_TOKEN": "xoxb-test", "BATCH_SIZE": "  ", "SLACK_API_PAUSE": ""}
    )
    assert cfg.batch_size == Config.batch_size
    assert cfg.api_pause == Config.api_pause


@pytest.mark.parametrize(
    ("value", "expected"), [("debug", "DEBUG"), (" Warning ", "WARNING")]
)
def test_log_level_normalised(value: str, expected: str) -> None:
    cfg = Config.from_env({"SLACK_BOT_TOKEN": "xoxb-test", "LOG_LEVEL": value})
    assert cfg.log_level == expected


def test_log_level_default() -> None:
    assert Config.from_env({"SLACK_BOT_TOKEN": "xoxb-test"}).log_level == "INFO"


def test_invalid_log_level_raises() -> None:
    with pytest.raises(ConfigError, match="LOG_LEVEL 'CHATTY' is not a logging level"):
        Config.from_env({"SLACK_BOT_TOKEN": "xoxb-test", "LOG_LEVEL": "chatty"})


def test_channel_list_parsing(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("SLACK_CHANNELS", " eng, ,ops ,C123,")
    assert Config.from_env().channel_list == ["eng", "ops", "C123"]


def test_channel_blacklist_strips_hash(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("SLACK_CHANNEL_BLACKLIST", "#random, social ,, C999")
    assert Config.from_env().channel_blacklist == {"random", "social", "C999"}


def test_eval_settings_and_effective_fallbacks(clean_env: pytest.MonkeyPatch) -> None:
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


@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_eval_test_false_strings_disable(
    clean_env: pytest.MonkeyPatch, value: str
) -> None:
    clean_env.setenv("EVAL_TEST", value)
    assert Config.from_env().eval_test is False
