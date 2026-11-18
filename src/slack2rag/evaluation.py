"""
Evaluation and safety checks for the Slack source.

label_template()  a JSONL file of stored units (channel, unit key,
                  permalink) for people to label in Slack itself: no
                  message text is written.
eval_classifier() runs the relevance check (and extraction) on labeled
                  units, refetched from Slack, and reports what it misses
                  and what it wrongly keeps.
spot_check()      a random sample of stored facts with leaked names,
                  handles, emails and phone numbers flagged, and facts that
                  lean on the conversation ("as mentioned above").
canaries()        finds planted strings anywhere in staging or any Qdrant
                  collection, and any Slack point in the public collection.
"""

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from .config import PUBLIC_COLLECTION
from .llm import LLMError
from .pg_store import SOURCE, FactRow, PgStore, UnitRow
from .pipeline import Pipeline
from .sanitize import find_personal
from .vector_store import count_in_collection, source_filter, text_matches_in

_NOT_STANDALONE = re.compile(
    r"(?i)\b(?:he|she|his|hers|him|this thread|the thread|as mentioned|"
    r"mentioned above|see above|said above|earlier in|the person|someone said|"
    r"the poster|the asker)\b"
)


def _permalink(pipeline: Pipeline, row: UnitRow) -> str:
    key = row.unit_key.removeprefix("pin:")
    if row.kind == "bookmark":
        return ""
    return pipeline.slack.permalink(row.channel_id, key)


# ── labeling ──────────────────────────────────────────────────────────────


def label_template(pipeline: Pipeline, count: int, seed: int = 0) -> list[str]:
    """JSONL lines for *count* stored units, for hand labeling."""
    import random

    rows = [r for r in pipeline.store.units() if r.kind != "bookmark"]
    random.Random(seed).shuffle(rows)  # noqa: S311 - sampling, not security
    return [
        json.dumps(
            {
                "channel": r.channel_id,
                "unit": r.unit_key,
                "kind": r.kind,
                "permalink": _permalink(pipeline, r),
                "relevant": None,
                "facts": [],
            }
        )
        for r in rows[:count]
    ]


@dataclass
class ClassifierReport:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    skipped: int = 0
    misses: list[str] = field(default_factory=list)  # permalinks
    false_keeps: list[str] = field(default_factory=list)
    fact_review: list[dict[str, Any]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        kept = self.true_positive + self.false_positive
        return self.true_positive / kept if kept else 0.0

    @property
    def recall(self) -> float:
        relevant = self.true_positive + self.false_negative
        return self.true_positive / relevant if relevant else 0.0

    def render(self) -> str:
        lines = [
            f"labeled units: {self.true_positive + self.false_positive}"
            f" kept, {self.false_negative + self.true_negative} dropped,"
            f" {self.skipped} skipped (gone from Slack or unlabeled)",
            f"true positives {self.true_positive}, false positives"
            f" {self.false_positive}, false negatives {self.false_negative},"
            f" true negatives {self.true_negative}",
            f"precision {self.precision:.2f}, recall {self.recall:.2f}",
        ]
        if self.misses:
            lines += ["", "Missed (labeled relevant, classifier dropped):"]
            lines += [f"  {m}" for m in self.misses]
        if self.false_keeps:
            lines += ["", "Wrongly kept (labeled not relevant, classifier kept):"]
            lines += [f"  {m}" for m in self.false_keeps]
        for review in self.fact_review:
            lines += ["", f"Facts for {review['permalink']}:", "  expected:"]
            lines += [f"    - {q}" for q in review["expected"]]
            lines += ["  extracted:"]
            lines += [f"    - {q}" for q in review["extracted"]] or ["    (none)"]
        return "\n".join(lines)


def _labels(path: Path) -> list[dict[str, Any]]:
    labels = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not JSON: {exc}") from None
        if not isinstance(label, dict) or "channel" not in label or "unit" not in label:
            raise ValueError(f"{path}:{number}: needs 'channel' and 'unit'")
        labels.append(label)
    return labels


def eval_classifier(pipeline: Pipeline, path: Path) -> ClassifierReport:
    report = ClassifierReport()
    names = pipeline.store.channel_names()
    for label in _labels(path):
        expected = label.get("relevant")
        if not isinstance(expected, bool):
            report.skipped += 1
            continue
        cid, key = str(label["channel"]), str(label["unit"])
        row = pipeline.store.get_unit(cid, key) or UnitRow(
            channel_id=cid,
            unit_key=key,
            kind=str(label.get("kind") or "thread"),
            message_ts=[],
            content_hash="",
            extractor_version="",
            relevant=None,
            boost=False,
            dirty=False,
            last_activity=datetime.datetime.now(tz=datetime.UTC),
        )
        unit = pipeline.refetch(row)
        if unit is None:
            report.skipped += 1
            continue
        link = _permalink(pipeline, row)
        try:
            result = pipeline.extractor.process(
                unit.render(), names.get(cid, cid), unit.boost
            )
        except LLMError:
            report.skipped += 1
            continue
        if result.relevant and expected:
            report.true_positive += 1
        elif result.relevant:
            report.false_positive += 1
            report.false_keeps.append(link)
        elif expected:
            report.false_negative += 1
            report.misses.append(link)
        else:
            report.true_negative += 1
        wanted = label.get("facts") or []
        if expected and wanted:
            report.fact_review.append(
                {
                    "permalink": link,
                    "expected": [
                        str(f.get("question", f)) if isinstance(f, dict) else str(f)
                        for f in wanted
                    ],
                    "extracted": [f"{f.question} — {f.answer}" for f in result.facts],
                }
            )
    return report


# ── spot check ───────────────────────────────────────────────────────────


def fact_problems(fact: FactRow, names: Iterable[str]) -> list[str]:
    text = f"{fact.question}\n{fact.answer}"
    problems = [f"leaked {kind}" for kind in find_personal(text, names)]
    if _NOT_STANDALONE.search(text):
        problems.append("may not stand alone")
    return problems


def spot_check(
    store: PgStore, names: Iterable[str], count: int
) -> list[tuple[FactRow, list[str]]]:
    names = list(names)
    return [
        (fact, fact_problems(fact, names))
        for fact in store.facts(limit=count, random_order=True)
    ]


# ── canaries ─────────────────────────────────────────────────────────────


def canaries(
    store: PgStore,
    client: QdrantClient,
    needles: Sequence[str],
    public_collection: str = PUBLIC_COLLECTION,
) -> list[str]:
    """Every place a canary was found; empty means the check passed."""
    found = [
        f"staging {where}: {needle!r}"
        for needle, where in store.facts_containing(needles)
    ]
    for collection in sorted(c.name for c in client.get_collections().collections):
        for needle in needles:
            for pid in text_matches_in(client, collection, needle):
                found.append(f"Qdrant {collection} point {pid}: {needle!r}")
    slack_public = count_in_collection(client, public_collection, source_filter(SOURCE))
    if slack_public:
        found.append(f"Qdrant {public_collection}: {slack_public} Slack-derived points")
    return found
