import os
from dataclasses import dataclass
from typing import List


DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_HOST = "api.openai.com"


def _bool_env(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


@dataclass
class Config:
    slack_bot_token: str

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "slack_messages"

    database_url: str = ""  # optional PostgreSQL (DATABASE_URL); empty disables

    slack_channels: str = ""  # comma-separated names/IDs; empty = all public
    slack_channel_blacklist: str = ""  # comma-separated names/IDs to skip entirely

    # Inference endpoint.  Defaults to OpenAI; any OpenAI-compatible server
    # (Ollama's /v1 layer, llama.cpp server, vLLM, LM Studio, etc.) also works.
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: str = ""  # optional; sent as Bearer token when set (OpenAI requires it)

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_prefix: str = ""  # optional prefix prepended to each embedding input
    embedding_context_length: int = 8192  # model context window in tokens; 0 = no truncation

    sync_interval_minutes: int = 60
    run_once: bool = False
    state_file: str = "/data/state.json"
    batch_size: int = 50
    api_pause: float = 1.2  # seconds between Slack API calls

    min_message_length: int = 20
    score_threshold: float = 0.0
    hybrid_search: bool = False
    thread_update_lookback_hours: int = 0
    reaction_boost_threshold: int = 3  # prepend "[highlighted by team]" at this many reactions

    eval_test: bool = False
    eval_prompt: str = ""
    eval_model: str = ""
    # Optional separate endpoint for the evaluator LLM.  When empty, reuses
    # llm_base_url / llm_api_key.
    eval_base_url: str = ""
    eval_api_key: str = ""

    @property
    def channel_list(self) -> List[str]:
        if self.slack_channels:
            return [c.strip() for c in self.slack_channels.split(",") if c.strip()]
        return []

    @property
    def channel_blacklist(self) -> set[str]:
        if self.slack_channel_blacklist:
            return {c.strip().lstrip("#") for c in self.slack_channel_blacklist.split(",") if c.strip()}
        return set()

    @property
    def effective_eval_base_url(self) -> str:
        return self.eval_base_url or self.llm_base_url

    @property
    def effective_eval_api_key(self) -> str:
        return self.eval_api_key or self.llm_api_key

    def missing_api_key_warnings(self) -> List[str]:
        """Warnings for OpenAI endpoints configured without an API key.

        The key stays optional so local OpenAI-compatible servers work
        without one, but OpenAI itself rejects unauthenticated requests.
        """
        warnings: List[str] = []
        if OPENAI_HOST in self.llm_base_url and not self.llm_api_key:
            warnings.append(
                f"LLM_BASE_URL is {self.llm_base_url} but LLM_API_KEY is not set; "
                "OpenAI requests will fail with 401"
            )
        if (
            self.eval_model
            and OPENAI_HOST in self.effective_eval_base_url
            and not self.effective_eval_api_key
        ):
            warnings.append(
                f"Evaluator endpoint is {self.effective_eval_base_url} but no "
                "EVAL_API_KEY or LLM_API_KEY is set; OpenAI requests will fail with 401"
            )
        return warnings

    @classmethod
    def from_env(cls) -> "Config":
        token = os.environ.get("SLACK_BOT_TOKEN", "")
        if not token:
            raise EnvironmentError("SLACK_BOT_TOKEN is required")
        return cls(
            slack_bot_token=token,
            qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "slack_messages"),
            database_url=os.environ.get("DATABASE_URL", ""),
            slack_channels=os.environ.get("SLACK_CHANNELS", ""),
            slack_channel_blacklist=os.environ.get("SLACK_CHANNEL_BLACKLIST", ""),
            llm_base_url=os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            llm_api_key=os.environ.get("LLM_API_KEY", ""),
            embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_prefix=os.environ.get("EMBEDDING_PREFIX", ""),
            embedding_context_length=int(os.environ.get("EMBEDDING_CONTEXT_LENGTH", "8192")),
            sync_interval_minutes=int(os.environ.get("SYNC_INTERVAL_MINUTES", "60")),
            run_once=_bool_env(os.environ.get("RUN_ONCE", "false")),
            state_file=os.environ.get("STATE_FILE", "/data/state.json"),
            batch_size=int(os.environ.get("BATCH_SIZE", "50")),
            api_pause=float(os.environ.get("SLACK_API_PAUSE", "1.2")),
            min_message_length=int(os.environ.get("MIN_MESSAGE_LENGTH", "20")),
            score_threshold=float(os.environ.get("SCORE_THRESHOLD", "0.0")),
            hybrid_search=_bool_env(os.environ.get("HYBRID_SEARCH", "false")),
            thread_update_lookback_hours=int(os.environ.get("THREAD_UPDATE_LOOKBACK_HOURS", "0")),
            reaction_boost_threshold=int(os.environ.get("REACTION_BOOST_THRESHOLD", "3")),
            eval_test=bool(os.environ.get("EVAL_TEST", "")),
            eval_prompt=os.environ.get("EVAL_PROMPT", ""),
            eval_model=os.environ.get("EVAL_MODEL", ""),
            eval_base_url=os.environ.get("EVAL_BASE_URL", ""),
            eval_api_key=os.environ.get("EVAL_API_KEY", ""),
        )
