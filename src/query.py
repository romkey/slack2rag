"""
slack2rag query CLI

Usage (inside Docker):
  docker compose run --rm query "how do I set up SSO?"
  docker compose run --rm query --limit 10 --channel engineering "deploy process"
  docker compose run --rm query --date-from 2024-06-01 "incident postmortem"

Usage (local, against a running Qdrant):
  QDRANT_URL=http://localhost:6333 python -m src.query "your question here"
"""

import argparse
import os
import sys
import textwrap

from dotenv import load_dotenv

from .config import Config
from .embedder import Embedder
from .vector_store import VectorStore

load_dotenv()

_SCORE_BAR_WIDTH = 20


def _score_bar(score: float) -> str:
    filled = round(score * _SCORE_BAR_WIDTH)
    return "█" * filled + "░" * (_SCORE_BAR_WIDTH - filled)


def _print_result(idx: int, hit: dict, show_score: bool) -> None:
    score = hit.get("score", 0.0)
    channel = hit.get("channel_name", hit.get("channel_id", "?"))
    date = hit.get("date", "")
    user = hit.get("user_name", hit.get("user_id", "?"))
    text = hit.get("text", "")
    reply_count = hit.get("reply_count", 0)
    thread_note = f"  [{reply_count} repl{'y' if reply_count == 1 else 'ies'}]" if reply_count else ""

    header = f"#{idx + 1}  #{channel}  {date}  @{user}{thread_note}"
    if show_score:
        header += f"  {_score_bar(score)}  {score:.3f}"

    print(header)
    print("─" * len(header))
    for line in textwrap.wrap(text, width=90):
        print("  " + line)
    print()


def main(argv: list[str] | None = None) -> None:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="query",
        description="Semantic search over indexed Slack messages.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "query",
        nargs="+",
        help="Search query (all positional args are joined into one query string)",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=5,
        metavar="N",
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "-c", "--channel",
        default=None,
        metavar="NAME_OR_ID",
        help="Restrict results to a specific channel",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only return messages on or after this date",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only return messages on or before this date",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Hide relevance score bars",
    )

    args = parser.parse_args(argv)
    query_text = " ".join(args.query)

    cfg = Config.from_env()

    embedder = Embedder(
        provider=cfg.embedding_provider,
        local_model=cfg.local_embedding_model,
        openai_api_key=cfg.openai_api_key,
        openai_model=cfg.openai_embedding_model,
    )
    store = VectorStore(
        url=cfg.qdrant_url,
        collection=cfg.qdrant_collection,
        dimension=embedder.dimension,
    )

    print(f'\nSearching for: "{query_text}"\n')

    vector = embedder.embed([query_text])[0]
    results = store.search(
        query_vector=vector,
        limit=args.limit,
        channel_filter=args.channel,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    if not results:
        print("No results found.")
        sys.exit(0)

    total = store.count()
    print(f"Top {len(results)} of {total:,} indexed messages\n")

    for idx, hit in enumerate(results):
        _print_result(idx, hit, show_score=not args.no_score)


if __name__ == "__main__":
    main()
