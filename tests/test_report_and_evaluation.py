"""Wiki promotion report and evaluation command tests."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from slack2rag.evaluation import (
    ClassifierReport,
    canaries,
    eval_classifier,
    fact_problems,
    label_template,
    spot_check,
)
from slack2rag.extractor import point_id
from slack2rag.pg_store import PgStore
from slack2rag.pipeline import Pipeline
from slack2rag.report import _dense, promotion_candidates, render, write_report
from slack2rag.vector_store import Point

from .conftest import DAY, Clock, FakeSlack

OLD = 2 * DAY


def _seed(pipeline: Pipeline, slack: FakeSlack, clock: Clock) -> None:
    slack.post(
        "C1",
        clock.now - OLD,
        "FACT howto|How do I focus the laser?|Use the gauge on the laser.",
    )
    slack.post(
        "C1",
        clock.now - OLD - DAY,
        "FACT location|Where are the clamps?|On the wall rack.",
    )
    slack.post("C1", clock.now - OLD - 2 * DAY, "FACT howto|Is it guessed?|Maybe.|low")
    pipeline.catchup()


def test_promotion_report(
    pipeline: Pipeline,
    slack: FakeSlack,
    store: PgStore,
    clock: Clock,
    tmp_path: Path,
) -> None:
    _seed(pipeline, slack, clock)
    # The wiki already covers the clamps.
    clamps = next(f for f in store.facts() if "clamps" in f.question)
    dense = _dense(pipeline.vectors, clamps.id)
    assert dense is not None
    wiki_id = point_id("wiki", "page-clamps")
    pipeline.vectors.upsert(
        [Point(wiki_id, dense, {"indices": [1], "values": [1.0]}, {"source": "wiki"})]
    )

    facts = promotion_candidates(store, pipeline.vectors)

    assert [f.question for f in facts] == ["How do I focus the laser?"]
    assert store.facts(ids=[clamps.id])[0].superseded_by == wiki_id
    text = render(facts, datetime.date(2026, 5, 25))
    assert "## lasers" in text
    assert "[Slack](https://ws.slack.com" in text
    path = write_report(
        store, pipeline.vectors, str(tmp_path), datetime.date(2026, 5, 25)
    )
    assert path.name == "wiki-promotion-2026-05-25.md"
    assert render([], datetime.date(2026, 5, 25)).startswith("# Wiki promotion queue")


def test_render_groups_uncategorized_last(
    pipeline: Pipeline, slack: FakeSlack, store: PgStore, clock: Clock
) -> None:
    _seed(pipeline, slack, clock)
    text = render(store.facts(), datetime.date(2026, 5, 25))
    assert text.index("## lasers") < text.index("## uncategorized")


def test_label_template_and_classifier_eval(
    pipeline: Pipeline, slack: FakeSlack, store: PgStore, clock: Clock, tmp_path: Path
) -> None:
    slack.post("C1", clock.now - 10 * DAY, "just chatting")
    _seed(pipeline, slack, clock)

    lines = label_template(pipeline, 10)
    labels = [json.loads(line) for line in lines]
    assert all("text" not in label for label in labels)
    assert all(label["permalink"].startswith("https://") for label in labels)

    for label in labels:
        unit = store.get_unit(label["channel"], label["unit"])
        assert unit is not None
        # Label the chat relevant (a miss) and the fact units not (false keeps).
        label["relevant"] = unit.fact_count == 0
    labels[0]["facts"] = ["How do I focus?"]
    labels.append(
        {"channel": "C1", "unit": "999.0", "kind": "thread", "relevant": True}
    )
    labels.append({"channel": "C1", "unit": "1.0"})
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(label) for label in labels) + "\n\n")

    report = eval_classifier(pipeline, path)

    assert report.false_negative == 1  # the chat, labeled relevant
    assert report.skipped == 2
    assert report.false_positive == 3
    rendered = report.render()
    assert "precision" in rendered
    assert "Missed" in rendered


def test_classifier_report_math() -> None:
    report = ClassifierReport(true_positive=3, false_positive=1, false_negative=1)
    assert report.precision == 0.75
    assert report.recall == 0.75
    assert ClassifierReport().precision == 0.0 == ClassifierReport().recall
    report.false_keeps.append("link")
    report.fact_review.append({"permalink": "p", "expected": ["q"], "extracted": []})
    assert "(none)" in report.render()
    assert "Wrongly kept" in report.render()


def test_bad_labels_file(pipeline: Pipeline, tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{nope\n")
    with pytest.raises(ValueError, match="not JSON"):
        eval_classifier(pipeline, path)
    path.write_text('{"channel": "C1"}\n')
    with pytest.raises(ValueError, match="needs 'channel' and 'unit'"):
        eval_classifier(pipeline, path)


def test_spot_check(
    pipeline: Pipeline, slack: FakeSlack, store: PgStore, clock: Clock
) -> None:
    _seed(pipeline, slack, clock)
    results = spot_check(store, ["Alice"], 10)
    assert len(results) == 3
    assert all(problems == [] for _, problems in results)
    fact = store.facts()[0]
    leaky = fact.__class__(
        **{**fact.__dict__, "answer": "As mentioned above, ask Alice."}
    )
    assert fact_problems(leaky, ["Alice"]) == ["leaked name", "may not stand alone"]


def test_canaries_find_planted_strings(store: PgStore, qdrant: QdrantClient) -> None:
    qdrant.create_collection("hs_public", vectors_config={})
    qdrant.upsert(
        "hs_public",
        points=[
            __import__(
                "qdrant_client.http.models", fromlist=["PointStruct"]
            ).PointStruct(
                id=1, vector={}, payload={"source": "slack", "text": "CANARY-ADMIN"}
            )
        ],
    )
    found = canaries(store, qdrant, ["CANARY-ADMIN", "CANARY-NONE"])
    assert any("hs_public point 1" in line for line in found)
    assert any("Slack-derived points" in line for line in found)
