# Configuration

slack2rag reads all of its settings from environment variables, once at
startup. With Docker they come from `.env` (copy `.env.example`); natively,
a `.env` file in the working directory is read too. An invalid value stops
the app with a message naming the variable.

## Required

| Variable | Meaning |
|---|---|
| `SLACK_BOT_TOKEN` | Bot User OAuth Token (`xoxb-…`). A user token (`xoxp-`) is refused. Scopes are listed in [slack-app.md](slack-app.md) |
| `DATABASE_URL` | `postgresql://user:password@host:5432/db`, the staging store. Run `slack2rag migrate` before the first start and after upgrades; the app refuses to start on an out-of-date schema |
| `CHAT_MODEL` | Chat-completions model that runs the relevance check, fact extraction and exclusion check |

## Slack

| Variable | Default | Meaning |
|---|---|---|
| `SLACK_APP_TOKEN` | *(empty: no live events)* | App-level token (`xapp-…`) with `connections:write`, for Socket Mode. Live events reprocess edited and deleted messages within a minute; without it the hourly catch-up does, within the hour |
| `SLACK_CHANNEL_DENYLIST` | *(empty)* | Comma-separated channel names (with or without `#`) or IDs never read, even if the bot is invited. Their data is deleted on the next run |
| `SLACK_BOT_ALLOWLIST` | *(empty)* | Bot IDs, usernames or bot names whose messages are read, such as an announcements integration. Every other bot is ignored |
| `SLACK_API_PAUSE` | `1.2` | Seconds between Slack API calls, ≥ 0 |

The bot reads only public channels it has been invited to; it never joins
a channel itself.

## Qdrant

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant REST endpoint. The compose files set it |
| `QDRANT_COLLECTION` | `hs_member` | The member-tier collection. `hs_public` is refused: everything from Slack is member tier |

## Model gateway

All models are reached over the OpenAI API. By default embeddings come
from a local [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
server (the compose files' `embeddings` service), which has no chat
models, so the chat model usually needs `CHAT_BASE_URL`.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | Gateway base URL; `/embeddings` is appended. The compose files ignore it; see `CONTAINER_LLM_BASE_URL` |
| `CONTAINER_LLM_BASE_URL` | `http://embeddings/v1` | Compose files only: the `LLM_BASE_URL` they give the app container |
| `TEI_IMAGE_TAG` | `cpu-1.9.4` | Compose files only: the TEI image tag. `cpu-arm64-1.9.4` on ARM hosts |
| `TEI_MAX_BATCH_TOKENS` | `2048` | Compose files only: TEI's input limit in tokens |
| `LLM_API_KEY` | *(empty)* | Optional Bearer token. OpenAI requires one (a warning is logged if it is missing) |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Embedding model. Recorded in the collection's metadata; a collection made with another model is refused until it is deleted and re-embedded |
| `EMBEDDING_QUERY_PREFIX` | `search_query: ` | Put before search queries. Must match the model |
| `EMBEDDING_DOCUMENT_PREFIX` | `search_document: ` | Put before every fact being indexed. Must match the model |
| `EMBEDDING_CONTEXT_LENGTH` | `2048` | Longest input in tokens; longer ones are truncated. 0 disables |
| `BATCH_SIZE` | `50` | Facts embedded per request, ≥ 1 |
| `CHAT_BASE_URL` | `LLM_BASE_URL` | A different endpoint for the chat model |
| `CHAT_API_KEY` | `LLM_API_KEY` | Its API key |
| `TOPICS_FILE` | *(empty: no topics)* | The shared topic taxonomy, one slug per line (`#` comments allowed). Each fact gets up to three |

Changing `CHAT_MODEL`, the prompts (a new release) or the taxonomy changes
the extractor version; the next nightly reconcile refetches and reprocesses
every unit.

## Conversation units

| Variable | Default | Meaning |
|---|---|---|
| `SESSION_GAP_MINUTES` | `30` | Unthreaded messages closer than this form one session, ≥ 1 |
| `SESSION_MAX_MESSAGES` | `40` | Longest session, ≥ 2 |
| `SESSION_OVERLAP_MESSAGES` | `2` | Messages a session borrows from the one before it, when they were split by the cap or a gap under four times `SESSION_GAP_MINUTES`. Less than `SESSION_MAX_MESSAGES` |
| `QUIET_HOURS` | `3` | A unit is processed only once it has been quiet this long, so late answers count. A unit with facts whose message is edited or deleted is reprocessed at once |

## Sync

| Variable | Default | Meaning |
|---|---|---|
| `SYNC_INTERVAL_MINUTES` | `60` | Minutes between catch-up runs, ≥ 1 |
| `CATCHUP_LOOKBACK_HOURS` | `72` | How far back each catch-up rereads history for late replies, edits and deletions, ≥ 1 |
| `BACKFILL_DAYS` | `0` | Days of history the first run reads per channel; 0 reads everything Slack returns (90 days on the free plan) |
| `RECONCILE_HOUR` | `3` | UTC hour of the nightly full reconcile, 0–23 |
| `REPORT_WEEKDAY` | `0` | Day the wiki promotion report is written, 0 (Monday) – 6 |
| `REPORT_DIR` | `/data/reports` | Where it is written, as `wiki-promotion-YYYY-MM-DD.md` |
| `SUPERSEDE_SIMILARITY` | `0.9` | Cosine similarity, 0–1, at which a newer fact about the same things supersedes an older one |
| `RUN_ONCE` | `false` | `true` runs one catch-up and exits, for cron or Kubernetes jobs |

Booleans accept `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off`.

## Search and logging

| Variable | Default | Meaning |
|---|---|---|
| `SCORE_THRESHOLD` | `0.0` | Minimum fused score for `slack2rag query`; 0 disables |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

## Retired

These were read by 0.6 and earlier. Setting one now stops the app with a
message naming its replacement, so an upgrade can't silently change what is
read or indexed: `SLACK_CHANNELS` (invite the bot instead),
`SLACK_CHANNEL_BLACKLIST` (`SLACK_CHANNEL_DENYLIST`), `DECISION_MODEL`,
`DECISION_PROMPT`, `DECISION_BASE_URL`, `DECISION_API_KEY` (`CHAT_*`),
`EVAL_TEST`, `EVAL_MODEL`, `EVAL_PROMPT`, `EVAL_BASE_URL`, `EVAL_API_KEY`
(`slack2rag eval-classifier`), `HYBRID_SEARCH` (always on),
`MIN_MESSAGE_LENGTH`, `REACTION_BOOST_THRESHOLD`,
`THREAD_UPDATE_LOOKBACK_HOURS`, `STATE_FILE` and `EMBEDDING_PREFIX`.

## Tooling (not read by the app)

| Variable | Meaning |
|---|---|
| `APP_VERSION` | Release `docker-compose.prod.yaml` runs; default `latest` |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Configure the `db` service in `docker-compose.prod.yaml`; must match `DATABASE_URL` |
| `UV_PUBLISH_TOKEN` | PyPI token for publishing by hand; not needed for releases |
| `TEST_DATABASE_URL` | PostgreSQL the tests use; the test compose file and CI set it, and the database tests are skipped without it |
