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
import sys
import textwrap

from dotenv import load_dotenv

from .config import Config
from .embedder import Embedder, EmbeddingError, SparseEncoder
from .vector_store import VectorStore

load_dotenv()

_SCORE_BAR_WIDTH = 20


def _score_bar(score: float) -> str:
    clamped = max(0.0, min(1.0, score))
    filled = round(clamped * _SCORE_BAR_WIDTH)
    return "█" * filled + "░" * (_SCORE_BAR_WIDTH - filled)


def _print_result(idx: int, hit: dict, show_score: bool) -> None:
    score = hit.get("score", 0.0)
    channel = hit.get("channel_name", hit.get("channel_id", "?"))
    date = hit.get("date", "")
    dt = hit.get("datetime", "")
    user = hit.get("user_name", hit.get("user_id", "?"))
    text = hit.get("text", "")
    reply_count = hit.get("reply_count", 0)
    reaction_count = hit.get("reaction_count", 0)
    permalink = hit.get("permalink", "")
    channel_topic = hit.get("channel_topic", "")
    attachments = hit.get("attachments", [])

    thread_note = ""
    if reply_count:
        thread_note = f"  [{reply_count} repl{'y' if reply_count == 1 else 'ies'}]"

    reaction_note = ""
    if reaction_count:
        reaction_note = f"  {reaction_count} reaction{'s' if reaction_count != 1 else ''}"

    display_date = dt if dt else date
    header = f"#{idx + 1}  #{channel}  {display_date}  @{user}{thread_note}{reaction_note}"
    if show_score:
        header += f"  {_score_bar(score)}  {score:.3f}"

    print(header)
    if channel_topic:
        print(f"  topic: {channel_topic}")
    print("─" * max(len(header), 60))
    for line in textwrap.wrap(text, width=90):
        print("  " + line)
    if attachments:
        print(f"  📎 {', '.join(attachments)}")
    if permalink:
        print(f"  🔗 {permalink}")
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

    try:
        embedder = Embedder(url=cfg.ollama_url, model=cfg.ollama_embedding_model,
                            context_length=cfg.ollama_context_length)
    except EmbeddingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    sparse_encoder = SparseEncoder() if cfg.hybrid_search else None

    store = VectorStore(
        url=cfg.qdrant_url,
        collection=cfg.qdrant_collection,
        dimension=embedder.dimension,
        hybrid=cfg.hybrid_search,
    )

    print(f'\nSearching for: "{query_text}"')
    if cfg.hybrid_search:
        print("  (hybrid: dense + sparse)")
    print()

    vector = embedder.embed([query_text])[0]
    sparse_vector = sparse_encoder.encode([query_text])[0] if sparse_encoder else None

    results = store.search(
        query_vector=vector,
        limit=args.limit,
        channel_filter=args.channel,
        date_from=args.date_from,
        date_to=args.date_to,
        score_threshold=cfg.score_threshold,
        sparse_vector=sparse_vector,
    )

    if not results:
        print("No results found.")
        if cfg.score_threshold > 0:
            print(f"  (score threshold is set to {cfg.score_threshold} — try lowering it)")
        sys.exit(0)

    total = store.count()
    print(f"Top {len(results)} of {total:,} indexed messages\n")

    for idx, hit in enumerate(results):
        _print_result(idx, hit, show_score=not args.no_score)


if __name__ == "__main__":
    main()
