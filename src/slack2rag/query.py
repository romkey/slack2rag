"""
slack2rag query: hybrid search over the Slack facts in the collection.

  slack2rag query "where is the laser cutter manual?"
  slack2rag query --kind howto --channel laser "focus the lens"
"""

import argparse
import sys
import textwrap

from qdrant_client.http import models as qmodels

from .config import Config
from .embedder import Embedder, EmbeddingError, SparseEncoder
from .pg_store import SOURCE
from .vector_store import SearchHit, VectorStore


def _filter(channel: str | None, kind: str | None, current: bool) -> qmodels.Filter:
    must: list[qmodels.Condition] = [
        qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=SOURCE))
    ]
    if channel:
        key = (
            "channel_id"
            if channel.startswith("C") and channel.isupper()
            else ("channel_name")
        )
        must.append(
            qmodels.FieldCondition(
                key=key, match=qmodels.MatchValue(value=channel.lstrip("#"))
            )
        )
    if kind:
        must.append(
            qmodels.FieldCondition(key="kind", match=qmodels.MatchValue(value=kind))
        )
    if not current:
        return qmodels.Filter(must=must)
    # Current facts have superseded_by null (or, from other sources, unset).
    superseded = qmodels.PayloadField(key="superseded_by")
    return qmodels.Filter(
        must=must,
        should=[
            qmodels.IsNullCondition(is_null=superseded),
            qmodels.IsEmptyCondition(is_empty=superseded),
        ],
    )


def _print_hit(index: int, hit: SearchHit) -> None:
    label = hit.get("label") or hit.get("source", "")
    kind = hit.get("kind", "")
    header = f"#{index + 1}  [{kind}] {label}  #{hit.get('channel_name', '')}"
    header += f"  score {hit.get('score', 0.0):.3f}"
    if hit.get("superseded_by"):
        header += "  (superseded)"
    if hit.get("confidence"):
        header += f"  confidence {hit['confidence']}"
    print(header)
    print("─" * max(len(header), 60))
    for line in str(hit.get("text", "")).splitlines():
        for wrapped in textwrap.wrap(line, width=90) or [""]:
            print("  " + wrapped)
    for link in hit.get("links") or []:
        print(f"  link: {link}")
    if hit.get("url"):
        print(f"  from: {hit['url']}")
    print()


def main(argv: list[str] | None = None, cfg: Config | None = None) -> int:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="slack2rag query", description="Search the facts mined from Slack."
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("query", nargs="+", help="what to search for")
    parser.add_argument("-n", "--limit", type=int, default=5, metavar="N")
    parser.add_argument("-c", "--channel", metavar="NAME_OR_ID")
    parser.add_argument("-k", "--kind", help="location, howto, policy, resource, ...")
    parser.add_argument(
        "--all", action="store_true", help="include superseded facts too"
    )
    args = parser.parse_args(argv)
    text = " ".join(args.query)
    if cfg is None:
        cfg = Config.from_env()

    try:
        embedder = Embedder(
            base_url=cfg.llm_base_url,
            model=cfg.embedding_model,
            context_length=cfg.embedding_context_length,
            api_key=cfg.llm_api_key,
            query_prefix=cfg.embedding_query_prefix,
            document_prefix=cfg.embedding_document_prefix,
        )
    except EmbeddingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    store = VectorStore(
        cfg.qdrant_url, cfg.qdrant_collection, embedder.dimension, cfg.embedding_model
    )
    hits = store.search(
        embedder.embed([text], kind="query")[0],
        SparseEncoder().encode([text])[0],
        limit=args.limit,
        flt=_filter(args.channel, args.kind, current=not args.all),
        score_threshold=cfg.score_threshold,
    )
    print(f'\nSearching for: "{text}"\n')
    if not hits:
        print("No results found.")
        return 0
    for index, hit in enumerate(hits):
        _print_hit(index, hit)
    return 0
