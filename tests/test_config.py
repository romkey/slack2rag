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


def test_embedding_prefix_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("EMBEDDING_PREFIX", "search_document:")
    cfg = Config.from_env()
    assert cfg.embedding_prefix == "search_document:"


def test_llm_base_url_and_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_CONTEXT_LENGTH", "8192")
    cfg = Config.from_env()
    assert cfg.llm_base_url == "https://api.openai.com/v1"
    assert cfg.llm_api_key == "sk-secret"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.embedding_context_length == 8192


def test_eval_endpoint_falls_back_to_llm_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "local-key")
    monkeypatch.delenv("EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("EVAL_API_KEY", raising=False)
    cfg = Config.from_env()
    assert cfg.effective_eval_base_url == "http://localhost:11434/v1"
    assert cfg.effective_eval_api_key == "local-key"


def test_eval_endpoint_overrides_llm_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "local-key")
    monkeypatch.setenv("EVAL_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("EVAL_API_KEY", "sk-cloud")
    cfg = Config.from_env()
    assert cfg.effective_eval_base_url == "https://api.openai.com/v1"
    assert cfg.effective_eval_api_key == "sk-cloud"


def test_llm_defaults_to_openai(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    cfg = Config.from_env()
    assert cfg.llm_base_url == "https://api.openai.com/v1"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.llm_api_key == ""


def test_missing_api_key_warns_for_openai() -> None:
    cfg = Config(slack_bot_token="xoxb-test")
    assert len(cfg.missing_api_key_warnings()) == 1
    cfg.eval_model = "gpt-4o-mini"
    assert len(cfg.missing_api_key_warnings()) == 2


def test_no_api_key_warning_when_key_set_or_local() -> None:
    assert Config(slack_bot_token="x", llm_api_key="sk-x", eval_model="m").missing_api_key_warnings() == []
    local = Config(slack_bot_token="x", llm_base_url="http://ollama:11434/v1", eval_model="m")
    assert local.missing_api_key_warnings() == []


def test_eval_key_warning_respects_separate_eval_endpoint() -> None:
    cfg = Config(
        slack_bot_token="x",
        llm_base_url="http://ollama:11434/v1",
        eval_model="gpt-4o-mini",
        eval_base_url="https://api.openai.com/v1",
    )
    warnings = cfg.missing_api_key_warnings()
    assert len(warnings) == 1 and "Evaluator" in warnings[0]
