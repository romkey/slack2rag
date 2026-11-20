# slack2rag

[![CI](https://github.com/romkey/slack2rag/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/slack2rag/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/slack2rag/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/slack2rag/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

slack2rag is the Slack source of a hackerspace's retrieval pipeline. It
mines the public channels its bot has been invited to for **facts**: where
things are, how to do things, policies, resources, who (which role or
channel) to ask. Each fact is a standalone question and answer with no
names, quotes or handles, staged in PostgreSQL and indexed, member tier
only, in a [Qdrant](https://qdrant.tech/) collection shared with the
pipeline's other sources. Raw messages are never stored: the index can't
play a conversation back.

All inference is local by default, over the **OpenAI API**: embeddings
from a bundled
[Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
server, and a chat model on any OpenAI-compatible server (Ollama, vLLM, a
gateway) for the relevance check, extraction and exclusion check.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

```
Slack (bot token + Socket Mode)  →  slack2rag  →  PostgreSQL (staging)
                                         ↓
            OpenAI-API gateway  ←→  embed  →  Qdrant hs_member  ←  retrieval service
```

## How it works

1. **Units.** A thread is one unit. Unthreaded messages are grouped into
   sessions (gaps under 30 minutes, capped, slightly overlapping). Pinned
   messages and channel bookmarks are units with a relevance boost. A unit
   is processed only after it has been quiet for a few hours, so late
   answers count.
2. **Excluded** before anything else: the bot's own messages, threads that
   start with a question to it (so its answers never come back as facts),
   other bots unless allowlisted, joins, leaves and the like, and any
   denylisted channel.
3. **Extraction**, per unit: a deterministic secret scan (hits are
   redacted; the quarantine log keeps only channel and timestamp); a
   relevance check; extraction into standalone facts (kind, question,
   answer, entities, links, confidence); an exclusion check that drops
   facts about individuals, incidents, credentials, admin procedures and
   access-path internals (known weaknesses stay, tagged `weakness`); and a
   personal-data scrub.
4. **Indexing.** Facts are embedded with a short context sentence, as
   dense and sparse vectors, into `hs_member` with the shared payload
   schema (`tier` member, `authority` 1, `stale_after` by kind, a
   "Slack discussion, DATE" label). A newer, highly similar fact about the
   same things supersedes an older one. The channel directory (name,
   topic, purpose) is indexed too, so bots can say "ask in #laser".
5. **Keeping current.** Socket Mode events reprocess edited and deleted
   messages within a minute; an hourly catch-up covers anything missed; a
   nightly reconcile refetches every unit with facts and deletes what no
   longer exists, with a safety valve. Removing the bot from a channel
   deletes everything derived from it. Changing the model or prompts
   reprocesses everything from stored timestamps.
6. **Wiki promotion.** A weekly report lists high-confidence howto,
   location and policy facts no wiki page covers, grouped by topic, with
   permalinks. Facts the wiki covers are superseded by it.

[docs/overview.md](docs/overview.md) has the details and the full payload
schema.

## Command line

```
slack2rag [--check]              the worker (same as `slack2rag run`)
slack2rag query [options] TEXT   search the facts (-n N, -c CHANNEL, -k KIND, --all)
slack2rag migrate                create or upgrade the PostgreSQL schema (automatic on start)
slack2rag status                 the last run: counts, quarantine size, errors
slack2rag reconcile              run the nightly full reconcile now
slack2rag reprocess              refetch every unit and run it through the model again, on the next run
slack2rag rebuild [--reextract]  delete the Slack points and re-embed every stored fact now
slack2rag report                 print the wiki promotion queue
slack2rag label-template [N]     JSONL of N units (permalinks, no text) to hand-label
slack2rag eval-classifier FILE   relevance-check misses and false keeps on labeled units
slack2rag spot-check [N]         N random facts, with leaked names and non-standalone facts flagged
slack2rag canary STRING...       exit 1 if a planted string is in staging or any collection
```

`--check` probes the embedding model, the chat model and the Slack token.

### Rebuilding the index

`slack2rag rebuild` deletes every Slack point from the collection (other
sources' points are left alone), re-embeds every fact in PostgreSQL, then
re-indexes the channel directory and catches up. It sends nothing to the
chat model, so it is quick, and it keeps facts from messages Slack no
longer shows (older than 90 days on the free plan). Use it after changing
`EMBEDDING_MODEL` (delete the collection first) or if the collection was
lost. `--reextract` also refetches every unit and runs it through the
chat model again; that can take hours, so run it detached
(`docker compose -f docker-compose.prod.yaml run -d --rm app slack2rag rebuild --reextract`).
Facts in PostgreSQL missing from Qdrant are also written back by every
run that sees them, so a nightly reconcile repairs a lost collection by
itself.

To start over completely instead, stop the app, delete the Slack points
(or the collection, if nothing else uses it) and the PostgreSQL data, and
start it again: it reads the history Slack still returns.

### Evaluation before trusting it

1. `slack2rag label-template 200 > labels.jsonl`, then open each permalink
   in Slack and set `"relevant"` (and, for relevant ones, the expected
   facts' questions in `"facts"`).
2. `slack2rag eval-classifier labels.jsonl` reports precision, recall,
   every miss and false keep, and expected against extracted facts.
3. `slack2rag spot-check 50` for leaked names and facts that don't stand alone.
4. Plant a canary string in a channel the bot is not in, then
   `slack2rag canary THE-STRING`: it also fails if any Slack-derived point
   is in `hs_public`.

## Running it

Set up the Slack app as in [docs/slack-app.md](docs/slack-app.md), and send
members the notice in [docs/member-notice.md](docs/member-notice.md) before
launch.

### With Docker

```bash
cp .env.example .env   # set SLACK_BOT_TOKEN, POSTGRES_*, DATABASE_URL, CHAT_MODEL, CHAT_BASE_URL
docker compose -f docker-compose.prod.yaml up -d
docker compose -f docker-compose.prod.yaml run --rm app slack2rag --check
docker compose -f docker-compose.prod.yaml run --rm app slack2rag query "where is the laser key"
```

That runs the latest release from `ghcr.io/romkey/slack2rag` with
PostgreSQL (the app creates and upgrades its schema on start), Qdrant and the
`embeddings` service; `APP_VERSION=X.Y.Z` pins a release. On ARM hosts set
`TEI_IMAGE_TAG=cpu-arm64-1.9.4`. `COMPOSE_PROFILES=ollama` adds a local
Ollama for the chat model (`CHAT_BASE_URL=http://ollama:11434/v1`).

### With pip

Needs Python 3.15 or newer:

```bash
python3 -m venv ~/.venvs/slack2rag
~/.venvs/slack2rag/bin/pip install https://github.com/romkey/slack2rag/releases/download/vX.Y.Z/slack2rag-X.Y.Z-py3-none-any.whl
~/.venvs/slack2rag/bin/slack2rag --version
```

You also need PostgreSQL, Qdrant, an embedding server and a chat model it
can reach. Settings come from the environment or a `.env` file in the
working directory ([docs/configuration.md](docs/configuration.md)).

## Development

```bash
docker compose up --build                                  # app, Qdrant, embeddings, PostgreSQL, from the working tree
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage, against a real PostgreSQL
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest       # the PostgreSQL tests run when TEST_DATABASE_URL is set
uv run mypy
```

`uvx pre-commit install` runs the lint checks on every commit.

## License

[MIT](LICENSE)
