import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    slack_bot_token: str

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "slack_messages"

    slack_channels: str = ""  # comma-separated names/IDs; empty = all public

    embedding_provider: str = "local"  # "local" | "openai"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    sync_interval_minutes: int = 60
    run_once: bool = False
    state_file: str = "/data/state.json"
    batch_size: int = 50

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
            embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "local"),
            local_embedding_model=os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            sync_interval_minutes=int(os.environ.get("SYNC_INTERVAL_MINUTES", "60")),
            run_once=os.environ.get("RUN_ONCE", "false").lower() in ("1", "true", "yes"),
            state_file=os.environ.get("STATE_FILE", "/data/state.json"),
            batch_size=int(os.environ.get("BATCH_SIZE", "50")),
        )
