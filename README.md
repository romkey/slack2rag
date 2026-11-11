# slack2rag

[![CI](https://github.com/romkey/slack2rag/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/slack2rag/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/slack2rag/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/slack2rag/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

slack2rag indexes the messages in your Slack channels into a
[Qdrant](https://qdrant.tech/) vector database, so a Retrieval-Augmented
Generation (RAG) chatbot or any other app can search your team's
conversations by meaning. It keeps the index current with incremental syncs,
groups threads, and stores rich metadata (channel, author, date, reactions,
permalinks). Embeddings, and an optional message-quality evaluator, come from
any **OpenAI-compatible** inference endpoint: OpenAI itself,
[Ollama](https://ollama.com/) (`/v1`), llama.cpp's server, vLLM, LM Studio.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

```
Slack API  →  slack2rag  →  Qdrant  ←  any app, via Qdrant's REST API
                  ↕
       OpenAI-compatible LLM
```

## What's here

| Path | What it is |
|------|------------|
| `src/slack2rag/__main__.py` | The `slack2rag` command: `run` (default), `query`, `migrate` |
| `src/slack2rag/main.py` | The sync loop, with thread-update refresh and summary documents |
| `src/slack2rag/slack_client.py` | Slack Web API: channels, history, threads, users, pagination |
| `src/slack2rag/processor.py` | Slack messages → `Document`s: thread grouping, chunking, summaries |
| `src/slack2rag/embedder.py` | OpenAI-compatible `/embeddings` client and the sparse (BM25-like) encoder |
| `src/slack2rag/vector_store.py` | Qdrant collection setup, upsert, search and hybrid search |
| `src/slack2rag/evaluator.py` | Eval mode: scores messages with an OpenAI-compatible `/chat/completions` model |
| `src/slack2rag/pg_store.py`, `migrations/` | Optional PostgreSQL copy of raw Slack payloads, with Alembic migrations |
| `src/slack2rag/state.py` | Per-channel sync cursors, as JSON on disk |
| `src/slack2rag/config.py` | Settings from environment variables |
| `src/slack2rag/query.py` | The `slack2rag query` command-line search |
| `tests/` | pytest tests, one file per module |
| `docs/configuration.md` | Every environment variable |

## Technical details

### How a sync works

1. List public channels (or those in `SLACK_CHANNELS`), skipping `SLACK_CHANNEL_BLACKLIST`.
2. For each channel, fetch messages newer than its saved cursor, joining the
   channel if needed, and their thread replies.
3. Turn each message and its thread into documents: long text is chunked,
   short standalone messages are skipped, and well-reacted messages are
   marked `[highlighted by team]`.
4. Embed in batches and upsert into Qdrant; advance the cursor.
5. Optionally re-index threads that gained replies in the last
   `THREAD_UPDATE_LOOKBACK_HOURS`.
6. Write summary documents for each channel, the workspace, users and the team.
7. Sleep `SYNC_INTERVAL_MINUTES`, or exit if `RUN_ONCE=true`. `docker stop`
   (SIGTERM) ends the loop cleanly between cycles.

The first sync indexes everything the bot can see; later ones are
incremental. Slack's free plan only keeps 90 days of history, so only those
messages can be indexed.

With `HYBRID_SEARCH=true`, each document also gets a sparse keyword vector,
and searches fuse dense and sparse results with Reciprocal Rank Fusion. That
catches exact matches (project names, error codes, ticket IDs) that pure
semantic search misses. Turning it on, or changing the embedding model,
needs a fresh Qdrant collection.

With `DATABASE_URL` set, every channel, user and raw message payload is also
upserted into PostgreSQL as JSONB.

### Command line

```
slack2rag [--check]              index Slack into Qdrant (the default; same as `slack2rag run`)
slack2rag query [options] TEXT   semantic search over the index
slack2rag migrate                create or upgrade the PostgreSQL schema
slack2rag --version
```

`--check` embeds a probe string and, if `EVAL_MODEL` is set, sends a tiny
chat request, then exits 0 if both worked or 1 if not. Use it before a first
full sync.

`slack2rag query` options: `-n/--limit N`, `-c/--channel NAME_OR_ID`,
`--date-from YYYY-MM-DD`, `--date-to YYYY-MM-DD`, `--no-score`. Example
output:

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

Eval mode (`EVAL_TEST=true`) skips indexing and instead scores every message
with `EVAL_MODEL` and `EVAL_PROMPT`, writing messages scoring 7 or more to
`/results/good.txt` and the rest to `/results/bad.txt`.

### Querying from another application

Qdrant serves a REST API on port **6333** (gRPC on 6334). Embed the query
with the same model slack2rag uses, then search:

```python
import json, urllib.request
from qdrant_client import QdrantClient

payload = json.dumps({"model": "nomic-embed-text", "input": ["how do I set up SSO?"]}).encode()
req = urllib.request.Request("http://localhost:11434/v1/embeddings",
                             data=payload, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    vector = json.loads(resp.read())["data"][0]["embedding"]

client = QdrantClient("http://localhost:6333")
response = client.query_points(
    collection_name="slack_messages", query=vector, limit=5, with_payload=True,
)
for hit in response.points:
    p = hit.payload
    print(f"[{p['datetime']}] #{p['channel_name']}  {p['user_name']}")
    print(p["text"])
```

Filter by channel with
`models.FieldCondition(key="channel_name", match=models.MatchValue(value="engineering"))`.

### Payload schema

Each stored point carries:

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

Indexed fields: `channel_id`, `channel_name`, `date`, `user_id` (keyword),
`text` (full-text), `reaction_count` (integer).

### Choosing an embedding model

Any embedding model your inference server offers works; with Ollama, pull it
first (`ollama pull <model>`). Set `EMBEDDING_CONTEXT_LENGTH` to match.

| Model | Dims | Notes |
|---|---|---|
| `nomic-embed-text` (default) | 768 | Strong general-purpose Ollama model |
| `mxbai-embed-large` | 1024 | High quality (Ollama) |
| `all-minilm` | 384 | Lightweight (Ollama) |
| `text-embedding-3-small` | 1536 | OpenAI, fast and cheap |
| `text-embedding-3-large` | 3072 | OpenAI, highest quality |

## Running it

### Slack app setup

1. Create an app ("From scratch") at <https://api.slack.com/apps>.
2. Under **OAuth & Permissions → Bot Token Scopes** add `channels:read`
   (list channels), `channels:history` (read messages), `channels:join`
   (join public channels without an `/invite`) and `users:read` (resolve
   names).
3. Install the app to your workspace and copy the **Bot User OAuth Token**
   (`xoxb-…`).
4. `/invite @your-bot-name` to any private channels you also want indexed.

### With Docker

Requires Docker. Configuration comes from environment variables, kept in
`.env` (see [docs/configuration.md](docs/configuration.md)):

```bash
cp .env.example .env   # then set SLACK_BOT_TOKEN
docker compose -f docker-compose.prod.yaml up -d
docker compose -f docker-compose.prod.yaml exec ollama ollama pull nomic-embed-text   # first time
docker compose -f docker-compose.prod.yaml run --rm app slack2rag --check
docker compose -f docker-compose.prod.yaml run --rm app slack2rag query "deploy process"
```

That runs the latest release from `ghcr.io/romkey/slack2rag` with Qdrant
and Ollama; set `APP_VERSION=X.Y.Z` to pin one. To use an Ollama already
running on the host, remove the `ollama` service and set
`LLM_BASE_URL=http://host.docker.internal:11434/v1`; for OpenAI, set
`LLM_BASE_URL=https://api.openai.com/v1` and `LLM_API_KEY`.

For the optional PostgreSQL store, set `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DB` and `DATABASE_URL` (host `db`) in `.env` and
`COMPOSE_PROFILES=postgres`; a `migrate` service upgrades the schema before
the app starts.

For a one-shot sync (cron, Kubernetes jobs), set `RUN_ONCE=true` and
`docker compose -f docker-compose.prod.yaml run --rm app`.

### With pip

Needs Python 3.15 or newer. Install into a virtual environment from a
release's wheel:

```bash
python3 -m venv ~/.venvs/slack2rag
~/.venvs/slack2rag/bin/pip install https://github.com/romkey/slack2rag/releases/download/vX.Y.Z/slack2rag-X.Y.Z-py3-none-any.whl
~/.venvs/slack2rag/bin/slack2rag --version
```

Set its environment variables yourself, or put them in a `.env` file in the
directory you run it from (see [docs/configuration.md](docs/configuration.md)).
You also need Qdrant and an embedding server it can reach.

## Development

With Docker, nothing else to install:

```bash
docker compose up --build                                  # app, Qdrant, Ollama, PostgreSQL, from the working tree
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage, against a real PostgreSQL
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions
CI tests (`uv.lock`):

```bash
uv sync
uv run pytest       # the PostgreSQL tests run when TEST_DATABASE_URL is set
uv run mypy
uv run slack2rag --check
```

With pip (25.1 or newer, for `--group`). pip doesn't read `uv.lock`, so it
may pick newer dependency versions than CI tested:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . --group dev
pytest
mypy
slack2rag --check
```

Either way, `uvx pre-commit install` (or `pip install pre-commit &&
pre-commit install`) runs the lint checks on every commit.

## License

[MIT](LICENSE)
