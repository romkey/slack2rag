"""
The wiki promotion queue: the main path for Slack knowledge to become
durable.

Lists the current high-confidence howto, location and policy facts that no
wiki page covers, grouped by topic, with permalinks. A fact is covered when
a wiki point in the same collection (source "wiki") is at least
WIKI_MATCH_SCORE similar; covered facts are marked superseded by that wiki
point, since the wiki always outranks Slack.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from collections import defaultdict
from pathlib import Path

from .pg_store import FactRow, PgStore
from .vector_store import DENSE, VectorStore, source_filter

logger = logging.getLogger(__name__)

PROMOTE_KINDS = ("howto", "location", "policy")
WIKI_MATCH_SCORE = 0.85
UNCATEGORIZED = "uncategorized"


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _dense(vectors: VectorStore, fact_id: str) -> list[float] | None:
    records = vectors.client.retrieve(
        collection_name=vectors.collection,
        ids=[fact_id],
        with_payload=False,
        with_vectors=[DENSE],
    )
    if not records:
        return None
    vector = records[0].vector
    if isinstance(vector, dict):
        dense = vector.get(DENSE)
        if isinstance(dense, list):
            return [float(x) for x in dense if isinstance(x, int | float)]
    return None


def promotion_candidates(store: PgStore, vectors: VectorStore) -> list[FactRow]:
    """Facts to promote; marks the ones the wiki already covers superseded."""
    wiki = source_filter("wiki")
    candidates = []
    for fact in store.facts(kinds=PROMOTE_KINDS, confidence="high", current_only=True):
        dense = _dense(vectors, fact.id)
        if dense is None:
            continue
        hits = vectors.similar(dense, wiki, limit=1)
        if hits and hits[0]["score"] >= WIKI_MATCH_SCORE:
            if _is_uuid(hits[0]["id"]):
                store.supersede(fact.id, hits[0]["id"])
                vectors.set_payload([fact.id], {"superseded_by": hits[0]["id"]})
            continue
        candidates.append(fact)
    return candidates


def render(facts: list[FactRow], today: datetime.date) -> str:
    groups: dict[str, list[FactRow]] = defaultdict(list)
    for fact in facts:
        groups[fact.topics[0] if fact.topics else UNCATEGORIZED].append(fact)
    lines = [
        f"# Wiki promotion queue, week of {today.isoformat()}",
        "",
        f"{len(facts)} high-confidence facts from Slack with no matching wiki"
        " content. Once one is on the wiki, its Slack fact is superseded.",
    ]
    for topic in sorted(groups, key=lambda t: (t == UNCATEGORIZED, t)):
        lines += ["", f"## {topic}", ""]
        for fact in sorted(groups[topic], key=lambda f: (f.kind, f.question)):
            link = f" [Slack]({fact.permalink})" if fact.permalink else ""
            lines.append(
                f"- **{fact.question}** ({fact.kind}) {fact.answer}"
                f" — #{fact.channel_name}, {fact.said_on.isoformat()}.{link}"
            )
    return "\n".join(lines) + "\n"


def write_report(
    store: PgStore, vectors: VectorStore, report_dir: str, today: datetime.date
) -> Path:
    text = render(promotion_candidates(store, vectors), today)
    path = Path(report_dir) / f"wiki-promotion-{today.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    logger.info("Wrote the wiki promotion queue to %s", path)
    return path
