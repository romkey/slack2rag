"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

OPENAI_HOST = "api.openai.com"

# The public tier's collection. Everything derived from Slack is member tier,
# so slack2rag refuses to write to a collection with this name.
PUBLIC_COLLECTION = "hs_public"


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("", "0", "false", "no", "off")

# Variables earlier versions read, with what replaced them. Setting one is an
# error rather than silently ignored, so an upgrade can't quietly change what
# is read or indexed.
RETIRED = {
    "EMBEDDING_PREFIX": "EMBEDDING_QUERY_PREFIX and EMBEDDING_DOCUMENT_PREFIX",
    "SLACK_CHANNELS": "inviting the bot to the channels it should read",
    "SLACK_CHANNEL_BLACKLIST": "SLACK_CHANNEL_DENYLIST",
    "DECISION_MODEL": "CHAT_MODEL",
    "DECISION_PROMPT": "the built-in relevance prompt",
    "DECISION_BASE_URL": "CHAT_BASE_URL",
    "DECISION_API_KEY": "CHAT_API_KEY",
    "EVAL_TEST": "`slack2rag eval-classifier`",
    "EVAL_MODEL": "CHAT_MODEL",
    "EVAL_PROMPT": "`slack2rag eval-classifier`",
    "EVAL_BASE_URL": "CHAT_BASE_URL",
    "EVAL_API_KEY": "CHAT_API_KEY",
    "HYBRID_SEARCH": "nothing: dense and sparse vectors are always stored",
    "MIN_MESSAGE_LENGTH": "the relevance check",
    "REACTION_BOOST_THRESHOLD": "nothing: pinned messages are boosted instead",
    "THREAD_UPDATE_LOOKBACK_HOURS": "QUIET_HOURS and CATCHUP_LOOKBACK_HOURS",
    "STATE_FILE": "the slack_cursors table in PostgreSQL",
}


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


def _int(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    text = env.get(name, "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, not {text!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be at most {maximum}, not {value}")
    return value


def _float(
    env: Mapping[str, str],
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float | None = None,
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
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be at most {maximum}, not {value}")
    return value


def _names(text: str) -> frozenset[str]:
    """A comma-separated list of channel or bot names/IDs, without '#'."""
    return frozenset(c.strip().lstrip("#") for c in text.split(",") if c.strip())


@dataclass(frozen=True)
class Config:
    slack_bot_token: str
    database_url: str
    chat_model: str

    # Socket Mode app-level token (xapp-...). Empty disables live events;
    # the hourly catch-up still finds new messages, edits and deletions.
    slack_app_token: str = ""
    slack_channel_denylist: frozenset[str] = frozenset()
    slack_bot_allowlist: frozenset[str] = frozenset()
    api_pause: float = 1.2  # seconds between Slack API calls

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "hs_member"

    # One OpenAI-compatible gateway for every model. The chat endpoint can be
    # separate because the bundled embeddings server (TEI) has no chat models.
    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = ""
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    embedding_context_length: int = 2048
    chat_base_url: str = ""
    chat_api_key: str = ""
    topics_file: str = ""  # one taxonomy slug per line; empty = no topics

    # Conversation units.
    session_gap_minutes: int = 30
    session_max_messages: int = 40
    session_overlap_messages: int = 2
    quiet_hours: float = 3.0

    # Sync.
    sync_interval_minutes: int = 60
    catchup_lookback_hours: int = 72
    backfill_days: int = 0  # 0 = all the history Slack returns
    reconcile_hour: int = 3  # UTC hour of the nightly full reconcile
    report_weekday: int = 0  # 0 = Monday; day the promotion report is written
    report_dir: str = "/data/reports"
    supersede_similarity: float = 0.9
    batch_size: int = 50
    run_once: bool = False

    score_threshold: float = 0.0
    log_level: str = "INFO"

    @property
    def effective_chat_base_url(self) -> str:
        return self.chat_base_url or self.llm_base_url

    @property
    def effective_chat_api_key(self) -> str:
        return self.chat_api_key or self.llm_api_key

    def missing_api_key_warnings(self) -> list[str]:
        """Warnings for OpenAI endpoints configured without an API key."""
        warnings = []
        if OPENAI_HOST in self.llm_base_url and not self.llm_api_key:
            warnings.append(
                f"LLM_BASE_URL is {self.llm_base_url} but LLM_API_KEY is not "
                "set; OpenAI will refuse the embedding requests"
            )
        if (
            OPENAI_HOST in self.effective_chat_base_url
            and not self.effective_chat_api_key
        ):
            warnings.append(
                f"The chat model uses {self.effective_chat_base_url} but neither "
                "CHAT_API_KEY nor LLM_API_KEY is set; OpenAI will refuse its requests"
            )
        return warnings

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Config:
        for old, new in RETIRED.items():
            if old in env:
                raise ConfigError(f"{old} is no longer used; it was replaced by {new}")

        token = env.get("SLACK_BOT_TOKEN", "").strip()
        if not token:
            raise ConfigError("SLACK_BOT_TOKEN is required")
        if token.startswith("xoxp-"):
            raise ConfigError(
                "SLACK_BOT_TOKEN is a user token (xoxp-); use the bot token (xoxb-)"
            )
        app_token = env.get("SLACK_APP_TOKEN", "").strip()
        if app_token and not app_token.startswith("xapp-"):
            raise ConfigError("SLACK_APP_TOKEN must be an app-level token (xapp-...)")

        database_url = env.get("DATABASE_URL", "").strip()
        if not database_url:
            raise ConfigError(
                "DATABASE_URL is required: units, facts and sync state live there"
            )
        chat_model = env.get("CHAT_MODEL", "").strip()
        if not chat_model:
            raise ConfigError("CHAT_MODEL is required: it classifies and extracts")

        collection = env.get("QDRANT_COLLECTION", cls.qdrant_collection).strip()
        if not collection:
            raise ConfigError("QDRANT_COLLECTION must not be empty")
        if collection == PUBLIC_COLLECTION:
            raise ConfigError(
                f"QDRANT_COLLECTION must not be {PUBLIC_COLLECTION}: everything"
                " derived from Slack is member tier"
            )

        log_level = env.get("LOG_LEVEL", cls.log_level).strip().upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        cfg = cls(
            slack_bot_token=token,
            database_url=database_url,
            chat_model=chat_model,
            slack_app_token=app_token,
            slack_channel_denylist=_names(env.get("SLACK_CHANNEL_DENYLIST", "")),
            slack_bot_allowlist=_names(env.get("SLACK_BOT_ALLOWLIST", "")),
            api_pause=_float(env, "SLACK_API_PAUSE", cls.api_pause),
            qdrant_url=env.get("QDRANT_URL", cls.qdrant_url),
            qdrant_collection=collection,
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
            chat_base_url=env.get("CHAT_BASE_URL", "").strip(),
            chat_api_key=env.get("CHAT_API_KEY", ""),
            topics_file=env.get("TOPICS_FILE", "").strip(),
            session_gap_minutes=_int(
                env, "SESSION_GAP_MINUTES", cls.session_gap_minutes, minimum=1
            ),
            session_max_messages=_int(
                env, "SESSION_MAX_MESSAGES", cls.session_max_messages, minimum=2
            ),
            session_overlap_messages=_int(
                env, "SESSION_OVERLAP_MESSAGES", cls.session_overlap_messages
            ),
            quiet_hours=_float(env, "QUIET_HOURS", cls.quiet_hours),
            sync_interval_minutes=_int(
                env, "SYNC_INTERVAL_MINUTES", cls.sync_interval_minutes, minimum=1
            ),
            catchup_lookback_hours=_int(
                env, "CATCHUP_LOOKBACK_HOURS", cls.catchup_lookback_hours, minimum=1
            ),
            backfill_days=_int(env, "BACKFILL_DAYS", cls.backfill_days),
            reconcile_hour=_int(env, "RECONCILE_HOUR", cls.reconcile_hour, maximum=23),
            report_weekday=_int(env, "REPORT_WEEKDAY", cls.report_weekday, maximum=6),
            report_dir=env.get("REPORT_DIR", cls.report_dir),
            supersede_similarity=_float(
                env, "SUPERSEDE_SIMILARITY", cls.supersede_similarity, maximum=1.0
            ),
            batch_size=_int(env, "BATCH_SIZE", cls.batch_size, minimum=1),
            run_once=_bool(env, "RUN_ONCE", cls.run_once),
            score_threshold=_float(env, "SCORE_THRESHOLD", cls.score_threshold),
            log_level=log_level,
        )
        if cfg.session_overlap_messages >= cfg.session_max_messages:
            raise ConfigError(
                "SESSION_OVERLAP_MESSAGES must be less than SESSION_MAX_MESSAGES"
            )
        return cfg
