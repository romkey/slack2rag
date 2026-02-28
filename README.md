# slack2rag

Indexes Slack public-channel messages into a [Qdrant](https://qdrant.tech/) vector database for use with Retrieval-Augmented Generation (RAG).

```
Slack API  →  slack2rag  →  Qdrant
                              ↑
               any app can query the REST API
```

The Qdrant service is the reusable component — any application that can make HTTP requests can query it without touching Slack at all.

---

## Requirements

* Docker + Docker Compose
* A Slack app with a **Bot Token** (`xoxb-…`)

### Slack app setup

1. Go to <https://api.slack.com/apps> and create a new app ("From scratch").
2. Under **OAuth & Permissions → Bot Token Scopes** add:
   | Scope | Used for |
   |---|---|
   | `channels:read` | list public channels |
   | `channels:history` | read messages |
   | `users:read` | resolve user mentions to names |
3. Install the app to your workspace and copy the **Bot User OAuth Token**.
4. Invite the bot to any private channels you also want indexed:
   `/invite @your-bot-name`  (public channels are accessible without this).

> **Free-plan note:** Slack's free plan only retains 90 days of message history.  Only those messages will be indexed.

---

## Pre-built image

Every push to `main` (and every semver tag) publishes a Docker image to the GitHub Container Registry:

```
ghcr.io/<owner>/slack2rag:latest
ghcr.io/<owner>/slack2rag:v1.2.3
```

To use the pre-built image instead of building from source:

```bash
IMAGE=ghcr.io/<owner>/slack2rag:latest docker compose up
```

---

## Quick start

```bash
cp .env.example .env
# Edit .env and set SLACK_BOT_TOKEN
docker compose up --build
```

The first run downloads the embedding model (~90 MB for the default `all-MiniLM-L6-v2`) and indexes all accessible public channels.  Subsequent runs are incremental.

---

## Configuration

All configuration is via environment variables (or `.env`).

| Variable | Default | Description |
|---|---|---|
| `SLACK_BOT_TOKEN` | **required** | Bot token from api.slack.com |
| `SLACK_CHANNELS` | *(empty)* | Comma-separated channel names/IDs to index.  Empty = all public channels |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `slack_messages` | Collection name |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model |
| `OPENAI_API_KEY` | *(empty)* | Required when `EMBEDDING_PROVIDER=openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model |
| `SYNC_INTERVAL_MINUTES` | `60` | Minutes between syncs |
| `RUN_ONCE` | `false` | Exit after one sync (for cron/k8s jobs) |
| `BATCH_SIZE` | `50` | Messages embedded per batch |

---

## Querying from another application

Qdrant exposes a REST API on port **6333** (and gRPC on 6334).

### Python (qdrant-client)

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient("http://localhost:6333")

query = "how do I set up SSO?"
vector = model.encode([query], normalize_embeddings=True)[0].tolist()

results = client.search(
    collection_name="slack_messages",
    query_vector=vector,
    limit=5,
    with_payload=True,
)

for hit in results:
    p = hit.payload
    print(f"[{p['date']}] #{p['channel_name']}  {p['user_name']}")
    print(p["text"])
    print()
```

### Filtering by channel or date

```python
from qdrant_client.http import models

results = client.search(
    collection_name="slack_messages",
    query_vector=vector,
    limit=10,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="channel_name", match=models.MatchValue(value="engineering")),
            models.FieldCondition(key="date", range=models.Range(gte="2024-01-01")),
        ]
    ),
)
```

### REST (curl)

```bash
curl -s http://localhost:6333/collections/slack_messages/points/search \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [/* your query vector */],
    "limit": 5,
    "with_payload": true
  }'
```

---

## Payload schema

Each stored point carries this payload:

```json
{
  "channel_id":   "C04ABCDEF",
  "channel_name": "general",
  "ts":           "1706234567.123456",
  "date":         "2024-01-25",
  "user_id":      "U04GHIJKL",
  "user_name":    "alice",
  "thread_ts":    "1706234567.123456",
  "reply_count":  3,
  "text":         "[alice]: How do I reset my password?\n[bob]: Go to /account/settings",
  "permalink":    ""
}
```

---

## Architecture

```
docker-compose.yml
├── qdrant          — vector DB, persistent volume, REST API on :6333
└── slack2rag
    ├── slack_client.py   — Slack Web API (pagination, rate-limit retries)
    ├── processor.py      — message → Document (thread grouping, chunking)
    ├── embedder.py       — sentence-transformers or OpenAI
    ├── vector_store.py   — Qdrant upsert / search
    ├── state.py          — per-channel sync cursor (JSON on disk)
    └── main.py           — sync loop
```

## Running one-shot (e.g. cron)

```bash
RUN_ONCE=true docker compose run --rm slack2rag
```
