# slack2rag overview

**What:** the Slack source of the hackerspace RAG pipeline. Mines public
channels the bot is in for standalone facts, stages them in PostgreSQL and
indexes them, member tier only, in Qdrant. Slack is mined for facts, not
archived: the index can never play a conversation back.
**Runs as:** a long-running worker (`slack2rag`) plus CLI commands. No HTTP port.
**Talks to:** Slack Web API and Socket Mode, PostgreSQL (required), Qdrant,
an OpenAI-compatible gateway for embeddings and the chat model.

## Flow

```
Slack (bot token, invited public channels only)
  → units: threads, sessions of unthreaded messages, pins, bookmarks
  → wait until quiet (QUIET_HOURS)
  → secret scan (deterministic; quarantine log keeps channel + ts only)
  → relevance check (model) → fact extraction (model)
  → exclusion check (model) → personal-data scrub (deterministic)
  → staging (slack_units, slack_facts) → embed (dense + sparse) → hs_member
```

Jobs, each under a PostgreSQL advisory lock with a `sync_runs` row and the
`sync_status` row for source `slack`:

- **catch-up** (hourly): history since the cursor minus
  `CATCHUP_LOOKBACK_HOURS`, threads whose replies changed, pins, bookmarks,
  the channel directory, units marked dirty, units waiting for quiet.
- **reconcile** (nightly): catch-up, plus every unit with facts (or active
  in the last 30 days, or processed by an older extractor) refetched by its
  stored timestamps. Facts seen are stamped with the run ID; older points
  are deleted unless the run saw less than half of the last one (safety
  valve: status `alert`, nothing deleted).
- **live** (Socket Mode): replies, edits and deletions mark units dirty;
  they are processed within a minute. Leaving or losing a channel deletes
  everything derived from it.
- **report** (weekly): the wiki promotion queue.
- **rebuild** (`slack2rag rebuild`, by hand): deletes every Slack point,
  re-embeds every stored fact without the chat model, then catches up;
  `--reextract` also runs every unit through the model again. Any run
  that sees a stored fact whose point is missing writes it back.

## Stored

PostgreSQL: channel directory (ID, name, scrubbed topic and purpose),
units (channel, key, message timestamps, content hash, extractor version,
verdict), facts, quarantine (channel, ts, rule names), cursors, runs and
status. Never: raw message text, user IDs, names.

Qdrant (`hs_member`): one point per fact and per channel directory entry,
in the shared payload schema: `id` = uuid5(NAMESPACE_URL,
`slack:{doc_id}:0`), `source` `slack`, `doc_id`
`{channel}:{unit}:{index}` (or `channel:{id}`), `title` (the question),
`heading_path`, `text` (`Q: …\nA: …`), `context` (embedded with the text),
`url` (permalink), `tier` `member`, `authority` 1, `created_at`,
`updated_at`, `indexed_at`, `stale_after` (status 7 days, announcement 30,
location/howto/contact 12 months, others none), `superseded_by`,
`content_hash`, `sync_run_id`, empty `asset_ids`/`location_ids`/
`work_ids`/`event_ids`/`training_ids` for the linker, `topics`, `section`,
`page`, `lang`; plus `kind`, `confidence`, `entities` (names for the
linker), `links`, `tags` (`weakness`), `channel_id`, `channel_name`,
`said_on` and `label` ("Slack discussion, DATE", for bots to cite).
The collection's metadata records the embedding model and dimension.

## Files

- `src/slack2rag/__main__.py`: command line and dispatch.
- `src/slack2rag/main.py`: worker loop, schedule, locks, `--check`.
- `src/slack2rag/pipeline.py`: catch-up, reconcile, live processing, indexing, superseding, safety valve.
- `src/slack2rag/units.py`: messages → threads, sessions, pin and bookmark units; exclusions; anonymized rendering.
- `src/slack2rag/extractor.py`: the five extraction steps, prompts, `Fact`, IDs, staleness.
- `src/slack2rag/sanitize.py`: secret scanner and personal-data scrub.
- `src/slack2rag/llm.py`: chat-completions client.
- `src/slack2rag/embedder.py`: `/embeddings` client; `SparseEncoder`.
- `src/slack2rag/slack_client.py`: Slack Web API (no joins; mentions anonymized).
- `src/slack2rag/realtime.py`: Socket Mode events.
- `src/slack2rag/pg_store.py`, `migrations/`: staging store, Alembic migrations (raw SQL).
- `src/slack2rag/vector_store.py`: Qdrant collection, payload indexes, hybrid search.
- `src/slack2rag/report.py`: wiki promotion queue.
- `src/slack2rag/evaluation.py`: labeling template, classifier eval, spot check, canaries.
- `src/slack2rag/query.py`: `slack2rag query`.
- `tests/`: pytest; `conftest.py` has FakeSlack, FakeChat, FakeEmbedder. Database tests need `TEST_DATABASE_URL`; Qdrant runs in memory.
- `docs/`: configuration, Slack app setup, the member notice.

## Not in this repo

The retrieval service, realtime tools service, entity registry and linker,
the other sources, and the shared evaluation harness live elsewhere; this
source leaves `asset_ids` and the like empty and puts raw `entities` and
`links` in the payload for the linker. Not implemented yet: parsing PDFs
shared in an allowlist of channels (the plan's optional "files"), and the
hand-labeled set of ~200 units (people make it with
`slack2rag label-template`).

## Conventions

- All configuration from environment variables; see docs/configuration.md and .env.example.
- Strict xfail tests mark known, unfixed bugs; fixing one makes its test fail until the marker goes.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
