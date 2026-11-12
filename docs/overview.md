# slack2rag overview

**What:** indexes Slack channel messages into Qdrant for RAG search.
**Runs as:** a long-running worker (`slack2rag`), plus `query` and `migrate` CLI commands. No HTTP port.
**Talks to:** Slack Web API, Qdrant, an OpenAI-compatible embeddings/chat endpoint, optionally PostgreSQL.

## Files

- `src/slack2rag/__main__.py`: command line (`run`/`query`/`migrate`), `.env` loading, logging.
- `src/slack2rag/main.py`: sync loop, `--check`, SIGTERM handling, summaries.
- `src/slack2rag/config.py`: frozen `Config` from environment variables; `ConfigError`.
- `src/slack2rag/slack_client.py`: Slack API wrapper (pagination, auto-join, user cache).
- `src/slack2rag/processor.py`: messages → `Document`s, chunking, channel/user/team summaries.
- `src/slack2rag/embedder.py`: `/embeddings` client with context-length fallback; `SparseEncoder`.
- `src/slack2rag/vector_store.py`: Qdrant collection, upsert, (hybrid) search.
- `src/slack2rag/evaluator.py`: eval mode, scoring with `/chat/completions`.
- `src/slack2rag/pg_store.py`: optional PostgreSQL JSONB store; `migrate`, `check_schema`.
- `src/slack2rag/migrations/`: Alembic migrations, raw SQL via `op.execute` (no ORM models).
- `src/slack2rag/state.py`: per-channel cursors in a JSON file.
- `src/slack2rag/query.py`: `slack2rag query` search output.
- `tests/`: pytest; one file per module, `*_extra.py`/`*_unit.py` for more cases. `test_pg_store.py` needs `TEST_DATABASE_URL`.
- `Dockerfile`: `dev` and `production` stages; `/data` and `/results` writable by uid 10001.
- `docker-compose.yaml` (dev: app, qdrant, db; `ollama` profile adds ollama), `.test` (test, db), `.lint`, `.prod` (app, qdrant; `ollama` profile adds ollama, `postgres` profile adds migrate, db).

## Conventions

- All configuration from environment variables; see docs/configuration.md and .env.example.
- Strict xfail tests mark known, unfixed bugs; fixing one makes its test fail until the marker goes.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
