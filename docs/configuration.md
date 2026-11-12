# Configuration

slack2rag reads all of its settings from environment variables, once at
startup. With Docker they come from `.env` (copy `.env.example`); natively,
a `.env` file in the working directory is read too. An invalid value stops
the app with a message naming the variable.

## Required

| Variable | Meaning |
|---|---|
| `SLACK_BOT_TOKEN` | Bot User OAuth Token (`xoxb-…`) with `channels:read`, `channels:history`, `channels:join` and `users:read` |

## Slack

| Variable | Default | Meaning |
|---|---|---|
| `SLACK_CHANNELS` | *(empty: all public channels)* | Comma-separated channel names or IDs to index |
| `SLACK_CHANNEL_BLACKLIST` | *(empty)* | Comma-separated channel names (with or without `#`) or IDs never to index or evaluate |
| `SLACK_API_PAUSE` | `1.2` | Seconds between Slack API calls, ≥ 0. 1.2 stays under the free plan's rate limit; 0 for paid plans |

## Qdrant

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant REST endpoint. The compose files set it |
| `QDRANT_COLLECTION` | `slack_messages` | Collection name |

## Embeddings (any OpenAI-compatible endpoint)

slack2rag sends `POST {LLM_BASE_URL}/embeddings` requests in the OpenAI
format. By default the server is a local
[Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
(TEI), which the compose files run as the `embeddings` service, serving
ModernBERT. `.env.example` has a table of common models with their
dimensions, context lengths and prompts.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | Base URL; `/embeddings` and `/chat/completions` are appended. The compose files ignore it; see `CONTAINER_LLM_BASE_URL` |
| `CONTAINER_LLM_BASE_URL` | `http://embeddings/v1` | Compose files only: the `LLM_BASE_URL` they give the app container. Its own name, so a host-only `LLM_BASE_URL` in `.env` doesn't leak into the container. `https://api.openai.com/v1` for OpenAI, `http://ollama:11434/v1` for the bundled Ollama (`COMPOSE_PROFILES=ollama`) |
| `TEI_IMAGE_TAG` | `cpu-1.9.4` | Compose files only: the TEI image tag for the `embeddings` service. `cpu-arm64-1.9.4` on ARM hosts (Apple silicon) |
| `LLM_API_KEY` | *(empty)* | Optional; sent as a Bearer token when set. Required by OpenAI (a warning is logged at startup if it is missing); most local servers ignore it |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Embedding model name on that server; the compose files' `embeddings` service also downloads and serves it. Changing it needs a fresh collection |
| `EMBEDDING_QUERY_PREFIX` | `search_query: ` | Put before search queries (`slack2rag query`). Must match the model; `""` for models without prompts, such as OpenAI's |
| `EMBEDDING_DOCUMENT_PREFIX` | `search_document: ` | Put before every message and thread being indexed. Must match the model; changing it needs a fresh collection |
| `EMBEDDING_CONTEXT_LENGTH` | `2048` | Longest input in tokens, ≥ 0, at most the model's context. Longer texts, prefix included, are truncated at a word boundary; 0 disables truncation. The compose files' `embeddings` service uses it as its own limit; on CPU, values past about 2048 need several GB of memory |
| `BATCH_SIZE` | `50` | Messages embedded per request, ≥ 1 |

## What gets indexed

| Variable | Default | Meaning |
|---|---|---|
| `MIN_MESSAGE_LENGTH` | `20` | Standalone messages shorter than this many characters are skipped; threads are always kept. 0 disables |
| `REACTION_BOOST_THRESHOLD` | `3` | Messages with at least this many reactions get `[highlighted by team]` prepended. 0 disables |
| `THREAD_UPDATE_LOOKBACK_HOURS` | `0` | Re-index threads that gained replies within this many hours, after each sync. 0 disables; 24–72 is a good start |
| `HYBRID_SEARCH` | `false` | `true` stores sparse keyword vectors too and searches with RRF fusion. Changing it needs a fresh collection |
| `SCORE_THRESHOLD` | `0.0` | Minimum cosine similarity for `slack2rag query` results, ≥ 0. 0 disables |

## Sync

| Variable | Default | Meaning |
|---|---|---|
| `SYNC_INTERVAL_MINUTES` | `60` | Minutes between syncs, ≥ 1 |
| `RUN_ONCE` | `false` | `true` syncs once and exits, for cron or Kubernetes jobs |
| `STATE_FILE` | `/data/state.json` | Per-channel sync cursors; `channel_stats.json` is kept beside it |

Booleans accept `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`.

## PostgreSQL (optional)

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | *(empty: disabled)* | `postgresql://user:password@host:5432/db`. Raw channels, users and messages are upserted there. Run `slack2rag migrate` before the first start and after upgrades; the app refuses to start on an out-of-date schema |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | — | Not read by the app: they configure the `db` service in `docker-compose.prod.yaml` and must match `DATABASE_URL` |

## Eval mode (optional)

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_TEST` | `false` | `true` scores every message instead of indexing, writing `/results/good.txt` (score ≥ 7) and `/results/bad.txt` |
| `EVAL_MODEL` | *(empty)* | Chat-completions model that scores messages; required in eval mode |
| `EVAL_PROMPT` | *(empty)* | Instructions put before each message; required in eval mode |
| `EVAL_BASE_URL` | `LLM_BASE_URL` | A different OpenAI-compatible endpoint for the evaluator |
| `EVAL_API_KEY` | `LLM_API_KEY` | Its API key |

## Logging

| Variable | Default | Meaning |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

## Tooling (not read by the app)

| Variable | Meaning |
|---|---|
| `APP_VERSION` | Release `docker-compose.prod.yaml` runs; default `latest` |
| `COMPOSE_PROFILES` | `postgres` starts the optional PostgreSQL services |
| `UV_PUBLISH_TOKEN` | PyPI token for publishing by hand; not needed for releases |
| `TEST_DATABASE_URL` | PostgreSQL the tests use; the test compose file and CI set it, and the database tests are skipped without it |
