# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The PostgreSQL schema is migrated automatically on start: the worker
  and every command that uses the database upgrade it when it is behind,
  under an advisory lock so instances starting together don't race. A
  schema newer than the running version (after a downgrade) is refused,
  not touched. `slack2rag migrate` still works by hand.
- `docker-compose.prod.yaml` no longer has a `migrate` service; the app
  waits for `db` to be healthy instead.

## [0.7.0] - 2026-11-18

slack2rag is now the Slack source of the hackerspace RAG pipeline: it
mines Slack for facts instead of archiving it. Existing indexes are
disposable; start the Qdrant collection and the database over.

### Changed (breaking)

- Messages are no longer indexed. Threads, sessions of unthreaded
  messages, pinned messages and channel bookmarks are screened by a
  relevance check, turned into standalone facts (kind, question, answer,
  entities, links, confidence) by the chat model, checked for excluded
  content (individuals, incidents, credentials, admin procedures,
  access-path internals) and scrubbed of names, handles, emails and phone
  numbers. Only the facts are stored and indexed.
- Raw message text, user IDs and names are never stored. Migration 0002
  drops the `slack_messages`, `slack_users` and old `slack_channels`
  tables and adds the staging tables (units, facts, quarantine, cursors,
  sync runs and status). User and team summaries, channel statistics and
  `channel_stats.json` are gone.
- PostgreSQL (`DATABASE_URL`) and a chat model (`CHAT_MODEL`, with
  optional `CHAT_BASE_URL`/`CHAT_API_KEY`) are required. The production
  compose file always runs `db` and `migrate`; the `postgres` profile is
  gone.
- The bot reads only public channels it has been invited to and never
  joins one: `channels:join` is no longer used, and `SLACK_CHANNELS` is
  retired. `SLACK_CHANNEL_BLACKLIST` is now `SLACK_CHANNEL_DENYLIST`. New
  scopes: `pins:read`, `bookmarks:read` (see docs/slack-app.md).
- The default collection is `hs_member`, with named dense and sparse
  vectors, the shared payload schema (`source`, `doc_id`, `tier` member,
  `authority` 1, `stale_after`, `superseded_by`, `sync_run_id`...) and the
  embedding model recorded in its metadata. `hs_public` is refused.
  `HYBRID_SEARCH` is retired: hybrid search is always on.
- Eval mode (`EVAL_*`) and the decision model (`DECISION_*`) are replaced
  by the relevance check and `slack2rag eval-classifier`.
  `MIN_MESSAGE_LENGTH`, `REACTION_BOOST_THRESHOLD`,
  `THREAD_UPDATE_LOOKBACK_HOURS` and `STATE_FILE` are retired too. Setting
  any retired variable stops the app with a message naming its
  replacement.
- Logs go to stderr, so command output on stdout can be redirected.

### Added

- Live events over Socket Mode (`SLACK_APP_TOKEN`): edits and deletions
  are reprocessed within a minute; leaving a channel deletes everything
  derived from it.
- An hourly catch-up, a nightly full reconcile (`RECONCILE_HOUR`) that
  deletes what no run saw, with a safety valve when a run sees less than
  half of the previous one, PostgreSQL advisory locks, and a status row.
- Facts go stale by kind (status 7 days, announcement 30, location, howto
  and contact 12 months), and a newer, highly similar fact supersedes an
  older one (`SUPERSEDE_SIMILARITY`).
- The channel directory (name, topic, purpose) is indexed.
- A secret scanner with a quarantine log (channel and timestamp only).
- Topics from the shared taxonomy (`TOPICS_FILE`).
- A weekly wiki promotion report (`REPORT_WEEKDAY`, `REPORT_DIR`,
  `slack2rag report`).
- Commands: `status`, `reconcile`, `reprocess`, `report`,
  `label-template`, `eval-classifier`, `spot-check` and `canary`.
- docs/slack-app.md and a draft member notice, docs/member-notice.md.

## [0.6.0] - 2026-11-13

The first published release with the 0.5.0 changes below: v0.5.0 was
tagged, but neither its image nor its GitHub Release was published.

### Fixed

- The release workflow no longer fails to push the image when the Actions
  cache can't be written (as happened for v0.5.0).

## [0.5.0] - 2026-11-13

The project now follows
[python-ai-rules](https://github.com/romkey/python-ai-rules). Releases
before this one have no changelog entries.

### Changed (breaking)

- The command is now `slack2rag` (`slack2rag run`, `slack2rag query …`,
  `slack2rag migrate`) instead of `python -m src.main` / `python -m src.query`;
  the code moved to the `src/slack2rag/` package.
- `docker-compose.yml` is replaced by `docker-compose.yaml` (development),
  `docker-compose.prod.yaml` (production), `docker-compose.test.yaml` and
  `docker-compose.lint.yaml`. The app service is `app`, not `slack2rag`; the
  `query` service is gone (use `run --rm app slack2rag query …`); the
  `IMAGE` variable is replaced by `APP_VERSION`.
- The production image runs as a non-root user (uid 10001) with a read-only
  filesystem. An existing state volume written by the old root container
  needs its owner changed once:
  `docker compose -f docker-compose.prod.yaml run --rm --user root app chown -R 10001:10001 /data`.
- PostgreSQL's schema is now managed by Alembic migrations, applied by
  `slack2rag migrate` (the production compose file's `migrate` service);
  the app no longer creates tables and won't start on an unmigrated
  database. Existing databases migrate in place without changes.
- PostgreSQL in the compose files is 18 (was 16), in a new
  `postgres18_data` volume. To keep data from the old `postgres_data`
  volume, `pg_dump` it with a PostgreSQL 16 container and restore the dump
  into the new one.
- Images are published to ghcr.io only for releases (`X.Y.Z`, `X.Y`,
  `latest`), built for amd64 and arm64, instead of on every push to `main`.
- Invalid numbers or booleans in the environment now stop the app with a
  message naming the variable, instead of a traceback or being ignored.
- `LLM_BASE_URL` and `EVAL_BASE_URL` must be http:// or https:// URLs;
  anything else is refused before a request is made.
- Embeddings default to ModernBERT (`nomic-ai/modernbert-embed-base`, 768
  dimensions) served locally by Hugging Face Text Embeddings Inference over
  the OpenAI API: the compose files run it as the `embeddings` service, and
  `LLM_BASE_URL` defaults to `http://localhost:8080/v1`. The bundled Ollama
  service is now behind the `ollama` compose profile. `LLM_API_KEY` stays
  optional. To keep an existing Ollama-built index, set
  `COMPOSE_PROFILES=ollama`, `CONTAINER_LLM_BASE_URL=http://ollama:11434/v1`,
  `EMBEDDING_MODEL=nomic-embed-text` and the two prefixes below in `.env`;
  otherwise delete the Qdrant collection and re-index.
- `EMBEDDING_CONTEXT_LENGTH` now defaults to 2048 (was 8192). The bundled
  embeddings service has its own limit, `TEI_MAX_BATCH_TOKENS` (default
  2048), so an 8192 left in `.env` doesn't exhaust its memory.
- `EMBEDDING_PREFIX` is replaced by `EMBEDDING_QUERY_PREFIX` and
  `EMBEDDING_DOCUMENT_PREFIX` (defaults `search_query: ` and
  `search_document: `, ModernBERT's prompts). Setting the old variable stops
  the app with a message.

### Added

- Optional decision model (`DECISION_MODEL`, `DECISION_PROMPT`, and
  `DECISION_BASE_URL` / `DECISION_API_KEY` for a separate chat server): a
  chat-completions model decides whether each
  message or thread is a useful resource for members; only those go to
  Qdrant, while PostgreSQL still stores every message. `--check` tests it.
- `LOG_LEVEL` setting.
- A startup warning when an OpenAI endpoint has no API key.
- Separate query and document prefixes for embedding models trained with
  prompts: indexed messages get the document prefix, `slack2rag query` the
  query prefix, and the prefix counts toward `EMBEDDING_CONTEXT_LENGTH`.
- `docker stop` (SIGTERM) ends the sync loop cleanly between cycles.
- Tests for every module (coverage 12% → 99%), including PostgreSQL tests
  against a real database; strict-xfail tests record known bugs.
- Strict mypy, an expanded ruff rule set and ruff formatting; pre-commit
  hooks for them plus yamllint, actionlint, shellcheck, hadolint and secret
  scanning (gitleaks and a check for `.env` values in committed files).
- Release workflow, Renovate, `docs/configuration.md`, `docs/overview.md`,
  `AGENTS.md`, `SECURITY.md`, MIT `LICENSE`.

### Fixed

- The production compose file gives eval mode a `slack2rag_results`
  volume for `/results`. Both compose files take the container's LLM URL
  from the new `CONTAINER_LLM_BASE_URL` (default: the bundled Ollama), so
  they no longer hardcode it and an old `LLM_BASE_URL=http://localhost…`
  in `.env` doesn't break them.
- `EVAL_TEST=false` (or `0`) turned eval mode on; it is now read as a
  boolean.

### Updated

- Python 3.15; uv with `uv.lock` replaces `requirements*.txt`; current
  versions of every dependency, GitHub Action, Qdrant and Ollama image.
