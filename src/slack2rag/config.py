"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

OPENAI_HOST = "api.openai.com"


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("", "0", "false", "no", "off")


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    text = env.get(name)
    if text is None:
        return default
    value = text.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false, not {text!r}")


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    text = env.get(name, "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, not {text!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {value}")
    return value


def _float(
    env: Mapping[str, str], name: str, default: float, minimum: float = 0.0
) -> float:
    text = env.get(name, "").strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError:
        raise ConfigError(f"{name} must be a number, not {text!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {value}")
    return value


@dataclass(frozen=True)
class Config:
    slack_bot_token: str

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "slack_messages"

    database_url: str = ""  # optional PostgreSQL (DATABASE_URL); empty disables

    slack_channels: str = ""  # comma-separated names/IDs; empty = all public
    slack_channel_blacklist: str = ""  # comma-separated names/IDs to skip entirely

    # Inference endpoint: by default a local Hugging Face Text Embeddings
    # Inference server, spoken to over the OpenAI API. Any OpenAI-compatible
    # server (OpenAI, Ollama's /v1 layer, vLLM, llama.cpp...) also works.
    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = ""  # optional Bearer token; OpenAI requires one

    # ModernBERT, trained with these task prefixes (see its model card).
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    embedding_query_prefix: str = "search_query: "  # put before search queries
    embedding_document_prefix: str = "search_document: "  # before indexed text
    # Tokens per input, prefix included; 0 = no truncation. ModernBERT takes
    # 8192, but on CPU the compose files' TEI server runs out of memory past
    # about 2048 (attention memory grows with the square of the length).
    embedding_context_length: int = 2048

    sync_interval_minutes: int = 60
    run_once: bool = False
    state_file: str = "/data/state.json"
    batch_size: int = 50
    api_pause: float = 1.2  # seconds between Slack API calls

    min_message_length: int = 20
    score_threshold: float = 0.0
    hybrid_search: bool = False
    thread_update_lookback_hours: int = 0
    reaction_boost_threshold: int = (
        3  # prepend "[highlighted by team]" at this many reactions
    )

    eval_test: bool = False
    eval_prompt: str = ""
    eval_model: str = ""
    # Optional separate endpoint for the evaluator LLM.  When empty, reuses
    # llm_base_url / llm_api_key.
    eval_base_url: str = ""
    eval_api_key: str = ""

    log_level: str = "INFO"

    @property
    def channel_list(self) -> list[str]:
        if self.slack_channels:
            return [c.strip() for c in self.slack_channels.split(",") if c.strip()]
        return []

    @property
    def channel_blacklist(self) -> set[str]:
        if self.slack_channel_blacklist:
            return {
                c.strip().lstrip("#")
                for c in self.slack_channel_blacklist.split(",")
                if c.strip()
            }
        return set()

    @property
    def effective_eval_base_url(self) -> str:
        return self.eval_base_url or self.llm_base_url

    @property
    def effective_eval_api_key(self) -> str:
        return self.eval_api_key or self.llm_api_key

    def missing_api_key_warnings(self) -> list[str]:
        """Warnings for OpenAI endpoints configured without an API key.

        The key stays optional so local servers work without one, but
        OpenAI itself rejects unauthenticated requests.
        """
        warnings = []
        if OPENAI_HOST in self.llm_base_url and not self.llm_api_key:
            warnings.append(
                f"LLM_BASE_URL is {self.llm_base_url} but LLM_API_KEY is not "
                "set; OpenAI will refuse the embedding requests"
            )
        if (
            self.eval_model
            and OPENAI_HOST in self.effective_eval_base_url
            and not self.effective_eval_api_key
        ):
            warnings.append(
                f"The evaluator uses {self.effective_eval_base_url} but neither "
                "EVAL_API_KEY nor LLM_API_KEY is set; OpenAI will refuse its requests"
            )
        return warnings

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Config:
        token = env.get("SLACK_BOT_TOKEN", "")
        if not token:
            raise ConfigError("SLACK_BOT_TOKEN is required")

        if "EMBEDDING_PREFIX" in env:
            raise ConfigError(
                "EMBEDDING_PREFIX was replaced by EMBEDDING_QUERY_PREFIX and "
                "EMBEDDING_DOCUMENT_PREFIX; set those instead"
            )

        log_level = env.get("LOG_LEVEL", cls.log_level).strip().upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        return cls(
            slack_bot_token=token,
            qdrant_url=env.get("QDRANT_URL", cls.qdrant_url),
            qdrant_collection=env.get("QDRANT_COLLECTION", cls.qdrant_collection),
            database_url=env.get("DATABASE_URL", ""),
            slack_channels=env.get("SLACK_CHANNELS", ""),
            slack_channel_blacklist=env.get("SLACK_CHANNEL_BLACKLIST", ""),
            llm_base_url=env.get("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=env.get("LLM_API_KEY", ""),
            embedding_model=env.get("EMBEDDING_MODEL", cls.embedding_model),
            embedding_query_prefix=env.get(
                "EMBEDDING_QUERY_PREFIX", cls.embedding_query_prefix
            ),
            embedding_document_prefix=env.get(
                "EMBEDDING_DOCUMENT_PREFIX", cls.embedding_document_prefix
            ),
            embedding_context_length=_int(
                env, "EMBEDDING_CONTEXT_LENGTH", cls.embedding_context_length
            ),
            sync_interval_minutes=_int(
                env, "SYNC_INTERVAL_MINUTES", cls.sync_interval_minutes, minimum=1
            ),
            run_once=_bool(env, "RUN_ONCE", cls.run_once),
            state_file=env.get("STATE_FILE", cls.state_file),
            batch_size=_int(env, "BATCH_SIZE", cls.batch_size, minimum=1),
            api_pause=_float(env, "SLACK_API_PAUSE", cls.api_pause),
            min_message_length=_int(env, "MIN_MESSAGE_LENGTH", cls.min_message_length),
            score_threshold=_float(env, "SCORE_THRESHOLD", cls.score_threshold),
            hybrid_search=_bool(env, "HYBRID_SEARCH", cls.hybrid_search),
            thread_update_lookback_hours=_int(
                env, "THREAD_UPDATE_LOOKBACK_HOURS", cls.thread_update_lookback_hours
            ),
            reaction_boost_threshold=_int(
                env, "REACTION_BOOST_THRESHOLD", cls.reaction_boost_threshold
            ),
            eval_test=_bool(env, "EVAL_TEST", cls.eval_test),
            eval_prompt=env.get("EVAL_PROMPT", ""),
            eval_model=env.get("EVAL_MODEL", ""),
            eval_base_url=env.get("EVAL_BASE_URL", ""),
            eval_api_key=env.get("EVAL_API_KEY", ""),
            log_level=log_level,
        )
