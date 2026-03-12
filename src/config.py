import os
from dataclasses import dataclass
from typing import List


def _bool_env(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


@dataclass
class Config:
    slack_bot_token: str

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "slack_messages"

    slack_channels: str = ""  # comma-separated names/IDs; empty = all public

    ollama_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_context_length: int = 8192  # model context window in tokens; 0 = no truncation

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

    @property
    def channel_list(self) -> List[str]:
        if self.slack_channels:
            return [c.strip() for c in self.slack_channels.split(",") if c.strip()]
        return []

    @classmethod
    def from_env(cls) -> "Config":
        token = os.environ.get("SLACK_BOT_TOKEN", "")
        if not token:
            raise EnvironmentError("SLACK_BOT_TOKEN is required")
        return cls(
            slack_bot_token=token,
            qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "slack_messages"),
            slack_channels=os.environ.get("SLACK_CHANNELS", ""),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            ollama_embedding_model=os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            ollama_context_length=int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "8192")),
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
        )
