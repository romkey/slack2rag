# slack2rag

Indexes Slack public-channel messages into a [Qdrant](https://qdrant.tech/) vector database for use with Retrieval-Augmented Generation (RAG).  Embeddings (and the optional message-quality evaluator) are generated through any **OpenAI-compatible** inference endpoint — OpenAI itself, [Ollama](https://ollama.com/) (`/v1`), llama.cpp's server, vLLM, LM Studio, etc.

```
Slack API  →  slack2rag  →  Qdrant
                  ↕
       OpenAI-compatible LLM
                  ↑
    any app can query the Qdrant REST API
```

---

## Requirements

* Docker + Docker Compose
* A Slack app with a **Bot Token** (`xoxb-…`)
* An OpenAI-compatible inference endpoint with an embedding model available
  (the bundled `docker-compose.yml` ships an Ollama service for this)

### Slack app setup

1. Go to <https://api.slack.com/apps> and create a new app ("From scratch").
2. Under **OAuth & Permissions → Bot Token Scopes** add:
   | Scope | Used for |
   |---|---|
   | `channels:read` | list public channels |
   | `channels:history` | read messages |
   | `channels:join` | auto-join public channels (no manual `/invite` needed) |
   | `users:read` | resolve user mentions to names |
3. Install the app to your workspace and copy the **Bot User OAuth Token**.
4. Invite the bot to any private channels you also want indexed:
   `/invite @your-bot-name`  (public channels are accessible without this).

> **Free-plan note:** Slack's free plan only retains 90 days of message history.  Only those messages will be indexed.

---

## Quick start

```bash
cp .env.example .env
# Edit .env and set SLACK_BOT_TOKEN

# Start Qdrant, Ollama, and slack2rag
docker compose up --build -d

# Pull the embedding model (first time only)
docker compose exec ollama ollama pull nomic-embed-text
```

The first sync indexes all accessible public channels.  Subsequent runs are incremental.

> If you already have Ollama running on the host, comment out the `ollama` service in `docker-compose.yml` and set `LLM_BASE_URL=http://host.docker.internal:11434/v1` in `.env`.  To use OpenAI (or another OpenAI-compatible provider) instead, set `LLM_BASE_URL=https://api.openai.com/v1` and `LLM_API_KEY=sk-…`.

### Verifying connectivity

Before kicking off a full sync, confirm slack2rag can talk to the configured embedding and evaluator models:

```bash
docker compose run --rm slack2rag python -m src.main --check
```

This embeds a probe string and (if `EVAL_MODEL` is set) sends a tiny chat-completions request, then exits 0 on success or 1 on any failure.

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

## Configuration

All configuration is via environment variables (or `.env`).

### Core settings

| Variable | Default | Description |
|---|---|---|
| `SLACK_BOT_TOKEN` | **required** | Bot token from api.slack.com |
| `SLACK_CHANNELS` | *(empty)* | Comma-separated channel names/IDs to index.  Empty = all public channels |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `slack_messages` | Collection name |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL.  `/embeddings` and `/chat/completions` are appended automatically |
| `LLM_API_KEY` | *(empty)* | Bearer token for the LLM endpoint.  Required for OpenAI; optional for most local servers |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model name for embeddings (must exist on the LLM server) |
| `EMBEDDING_PREFIX` | *(empty)* | Optional prefix prepended to every embedding input, e.g. `search_document:` |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Max tokens per embedding input.  Texts are truncated at a word boundary.  `0` disables truncation |
| `SYNC_INTERVAL_MINUTES` | `60` | Minutes between syncs |
| `RUN_ONCE` | `false` | Exit after one sync (for cron/k8s jobs) |
| `BATCH_SIZE` | `50` | Messages embedded per batch |
| `SLACK_API_PAUSE` | `1.2` | Seconds between Slack API calls |

### Eval mode (optional)

| Variable | Default | Description |
|---|---|---|
| `EVAL_TEST` | *(empty)* | Set to any value to enable eval mode (scans messages, scores them, skips indexing) |
| `EVAL_MODEL` | *(empty)* | Chat-completions model used to score messages |
| `EVAL_PROMPT` | *(empty)* | Prompt prepended to each message before scoring |
| `EVAL_BASE_URL` | falls back to `LLM_BASE_URL` | Override to point the evaluator at a different OpenAI-compatible endpoint |
| `EVAL_API_KEY` | falls back to `LLM_API_KEY` | Override key for the evaluator endpoint |

### Retrieval quality

| Variable | Default | Description |
|---|---|---|
| `SCORE_THRESHOLD` | `0.0` | Minimum cosine similarity (0.0–1.0). Results below this are discarded. |
| `HYBRID_SEARCH` | `false` | Enable dense + sparse hybrid search with RRF fusion |
| `MIN_MESSAGE_LENGTH` | `20` | Skip standalone messages shorter than this (threads always kept) |
| `THREAD_UPDATE_LOOKBACK_HOURS` | `0` | Hours to look back for thread updates (0 = disabled) |

### Choosing an embedding model

Any embedding model your inference server exposes will work.  For Ollama, pull it first with `ollama pull <model>`.

| Model | Dims | Notes |
|---|---|---|
| `nomic-embed-text` (default) | 768 | Strong general-purpose Ollama model (recommended for local) |
| `mxbai-embed-large` | 1024 | High quality (Ollama) |
| `all-minilm` | 384 | Lightweight (Ollama) |
| `text-embedding-3-small` | 1536 | OpenAI, fast and cheap |
| `text-embedding-3-large` | 3072 | OpenAI, highest quality |

> **Note:** Changing the embedding model requires deleting the Qdrant collection and re-indexing (the vector dimensions change).

### Hybrid search

When `HYBRID_SEARCH=true`, each document is stored with both a dense embedding and a sparse (BM25-like) keyword vector.  At query time, both are searched and results are fused with Reciprocal Rank Fusion (RRF).  This catches keyword/exact-match queries (project names, error codes, ticket IDs) that pure semantic search misses.

> **Note:** Enabling hybrid search requires a fresh Qdrant collection (the schema changes from a single vector to named dense + sparse vectors).

---

## Querying from the command line

A built-in CLI tool is included in the image:

```bash
# basic search
docker compose run --rm query "how do I reset my password?"

# limit results
docker compose run --rm query --limit 10 "deploy process"

# filter by channel
docker compose run --rm query --channel engineering "incident postmortem"

# filter by date range
docker compose run --rm query --date-from 2024-06-01 --date-to 2024-06-30 "outage"

# hide relevance score bars
docker compose run --rm query --no-score "onboarding"
```

Example output:
```
Searching for: "how do I reset my password?"

Top 3 of 14,823 indexed messages

#1  #it-help  2024-03-12T09:15:00Z  @alice  [2 replies]  3 reactions  ████████████████░░░░  0.891
  topic: IT support and troubleshooting
────────────────────────────────────────────────────────────────────────────
  [alice]: To reset your password go to /account/settings and click "Forgot
  password". You'll get an email within a few minutes.
  🔗 https://myworkspace.slack.com/archives/C04ABCDEF/p1710234900123456
```

---

## Querying from another application

Qdrant exposes a REST API on port **6333** (and gRPC on 6334).

### Python (qdrant-client)

```python
import json, urllib.request
from qdrant_client import QdrantClient

# Get embedding from any OpenAI-compatible endpoint (Ollama's /v1 shown here)
payload = json.dumps({"model": "nomic-embed-text", "input": ["how do I set up SSO?"]}).encode()
req = urllib.request.Request("http://localhost:11434/v1/embeddings",
                             data=payload, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    vector = json.loads(resp.read())["data"][0]["embedding"]

# Search Qdrant
client = QdrantClient("http://localhost:6333")
response = client.query_points(
    collection_name="slack_messages",
    query=vector,
    limit=5,
    with_payload=True,
)

for hit in response.points:
    p = hit.payload
    print(f"[{p['datetime']}] #{p['channel_name']}  {p['user_name']}")
    print(p["text"])
    if p.get("permalink"):
        print(p["permalink"])
    print()
```

### Filtering by channel or date

```python
from qdrant_client.http import models

response = client.query_points(
    collection_name="slack_messages",
    query=vector,
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
  "channel_id":     "C04ABCDEF",
  "channel_name":   "general",
  "ts":             "1706234567.123456",
  "date":           "2024-01-25",
  "datetime":       "2024-01-25T14:30:00Z",
  "user_id":        "U04GHIJKL",
  "user_name":      "alice",
  "thread_ts":      "1706234567.123456",
  "reply_count":    3,
  "text":           "[alice]: How do I reset my password?\n[bob]: Go to /account/settings",
  "permalink":      "https://myworkspace.slack.com/archives/C04ABCDEF/p1706234567123456",
  "channel_topic":  "General discussion and announcements",
  "reaction_count": 5,
  "reactions":      ["+1", "white_check_mark", "eyes"],
  "attachments":    ["Q4-report.pdf"]
}
```

### Payload indexes

| Field | Index type | Use |
|---|---|---|
| `channel_id` | keyword | Filter by channel ID |
| `channel_name` | keyword | Filter by channel name |
| `date` | keyword | Date range queries |
| `user_id` | keyword | Filter by author |
| `text` | full-text | Keyword search and text matching |
| `reaction_count` | integer | Quality-weighted retrieval |

---

## Architecture

```
docker-compose.yml
├── qdrant          — vector DB, persistent volume, REST API on :6333
├── ollama          — OpenAI-compatible LLM server on :11434 (optional; swap for any OpenAI-compatible endpoint)
└── slack2rag
    ├── slack_client.py   — Slack Web API (pagination, rate-limit retries)
    ├── processor.py      — message → Document (thread grouping, chunking)
    ├── embedder.py       — OpenAI-compatible /embeddings client + sparse encoder
    ├── evaluator.py      — OpenAI-compatible /chat/completions message scorer
    ├── vector_store.py   — Qdrant upsert / search / hybrid search
    ├── state.py          — per-channel sync cursor (JSON on disk)
    ├── config.py         — configuration from environment variables
    └── main.py           — sync loop with thread-update refresh
```

## Running one-shot (e.g. cron)

```bash
RUN_ONCE=true docker compose run --rm slack2rag
```
