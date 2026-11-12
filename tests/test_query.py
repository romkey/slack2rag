"""Query CLI tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from slack2rag import __version__, query
from slack2rag.config import Config, ConfigError
from slack2rag.embedder import EmbeddingError
from slack2rag.query import _print_result, _score_bar


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    for var in ("HYBRID_SEARCH", "SCORE_THRESHOLD", "EMBEDDING_PREFIX", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("QDRANT_URL", "http://q:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "coll")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "emb")
    return monkeypatch


@pytest.fixture
def mocks(env: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    embedder = MagicMock()
    embedder.dimension = 3
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    embedder_cls = MagicMock(return_value=embedder)

    store = MagicMock()
    store.search.return_value = []
    store.count.return_value = 1234
    store_cls = MagicMock(return_value=store)

    sparse = MagicMock()
    sparse.encode.return_value = [{"indices": [1], "values": [1.0]}]
    sparse_cls = MagicMock(return_value=sparse)

    env.setattr(query, "Embedder", embedder_cls)
    env.setattr(query, "VectorStore", store_cls)
    env.setattr(query, "SparseEncoder", sparse_cls)
    return {
        "embedder_cls": embedder_cls,
        "embedder": embedder,
        "store_cls": store_cls,
        "store": store,
        "sparse_cls": sparse_cls,
        "sparse": sparse,
    }


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("score", "filled"),
    [(0.0, 0), (1.0, 20), (0.5, 10), (-3.0, 0), (7.0, 20), (0.26, 5)],
)
def test_score_bar(score: float, filled: int) -> None:
    bar = _score_bar(score)
    assert len(bar) == 20
    assert bar.count("█") == filled
    assert bar.count("░") == 20 - filled


def test_print_result_full(capsys: pytest.CaptureFixture[str]) -> None:
    hit = {
        "score": 0.75,
        "channel_name": "eng",
        "date": "2024-01-01",
        "datetime": "2024-01-01 12:00",
        "user_name": "alice",
        "text": "word " * 40,
        "reply_count": 1,
        "reaction_count": 1,
        "permalink": "https://slack/p1",
        "channel_topic": "Engineering talk",
        "attachments": ["a.png", "b.pdf"],
    }
    _print_result(0, hit, show_score=True)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith(
        "#1  #eng  2024-01-01 12:00  @alice  [1 reply]  1 reaction  "
    )
    assert lines[0].endswith("0.750")
    assert "█" in lines[0]
    assert lines[1] == "  topic: Engineering talk"
    assert set(lines[2]) == {"─"}
    assert len(lines[2]) >= 60
    # text wrapped at 90 chars and indented
    body = [ln for ln in lines[3:] if ln.startswith("  word")]
    assert len(body) >= 2
    assert all(len(ln) <= 92 for ln in body)
    assert "  📎 a.png, b.pdf" in lines
    assert "  🔗 https://slack/p1" in lines
    assert out.endswith("\n\n")


def test_print_result_minimal_fallbacks(capsys: pytest.CaptureFixture[str]) -> None:
    hit = {
        "channel_id": "C9",
        "user_id": "U9",
        "date": "2024-02-02",
        "text": "hi",
        "reply_count": 3,
        "reaction_count": 2,
    }
    _print_result(4, hit, show_score=False)
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    assert header == "#5  #C9  2024-02-02  @U9  [3 replies]  2 reactions"
    assert "topic:" not in out
    assert "📎" not in out
    assert "🔗" not in out


def test_print_result_missing_everything(capsys: pytest.CaptureFixture[str]) -> None:
    _print_result(0, {}, show_score=True)
    header = capsys.readouterr().out.splitlines()[0]
    assert header.startswith("#1  #?    @?")
    assert header.endswith("0.000")


# ── main() ───────────────────────────────────────────────────────────────────


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        query.main(["--version"])
    assert exc.value.code == 0
    assert f"slack2rag query {__version__}" in capsys.readouterr().out


def test_requires_query(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        query.main([])
    assert exc.value.code == 2
    assert "usage: slack2rag query" in capsys.readouterr().err


def test_main_with_results_and_args(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    mocks["store"].search.return_value = [
        {"score": 0.9, "channel_name": "eng", "user_name": "a", "text": "first"},
        {"score": 0.4, "channel_name": "ops", "user_name": "b", "text": "second"},
    ]

    assert (
        query.main(
            [
                "-n",
                "2",
                "-c",
                "eng",
                "--date-from",
                "2024-01-01",
                "--date-to",
                "2024-12-31",
                "how",
                "do",
                "I",
                "deploy",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert 'Searching for: "how do I deploy"' in out
    assert "hybrid" not in out
    assert "Top 2 of 1,234 indexed messages" in out
    assert "#1  #eng" in out
    assert "#2  #ops" in out
    assert "0.900" in out

    mocks["embedder_cls"].assert_called_once_with(
        base_url="http://llm/v1",
        model="emb",
        context_length=8192,
        api_key="",
        input_prefix="",
    )
    mocks["store_cls"].assert_called_once_with(
        url="http://q:6333",
        collection="coll",
        dimension=3,
        hybrid=False,
    )
    mocks["embedder"].embed.assert_called_once_with(["how do I deploy"])
    mocks["sparse_cls"].assert_not_called()
    mocks["store"].search.assert_called_once_with(
        query_vector=[0.1, 0.2, 0.3],
        limit=2,
        channel_filter="eng",
        date_from="2024-01-01",
        date_to="2024-12-31",
        score_threshold=0.0,
        sparse_vector=None,
    )


def test_main_defaults_and_no_score(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    mocks["store"].search.return_value = [{"score": 0.9, "text": "x"}]
    assert query.main(["--no-score", "hello"]) == 0
    out = capsys.readouterr().out
    assert "0.900" not in out
    assert "█" not in out
    kwargs = mocks["store"].search.call_args.kwargs
    assert kwargs["limit"] == 5
    assert kwargs["channel_filter"] is None
    assert kwargs["date_from"] is None
    assert kwargs["date_to"] is None


def test_main_hybrid_uses_sparse_encoder(
    mocks: dict[str, MagicMock],
    env: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env.setenv("HYBRID_SEARCH", "true")
    mocks["store"].search.return_value = [{"score": 0.5, "text": "x"}]
    assert query.main(["hello"]) == 0
    out = capsys.readouterr().out
    assert "(hybrid: dense + sparse)" in out
    mocks["sparse"].encode.assert_called_once_with(["hello"])
    assert mocks["store_cls"].call_args.kwargs["hybrid"] is True
    assert mocks["store"].search.call_args.kwargs["sparse_vector"] == {
        "indices": [1],
        "values": [1.0],
    }


def test_main_no_results_returns_zero(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    assert query.main(["nothing"]) == 0
    out = capsys.readouterr().out
    assert "No results found." in out
    assert "score threshold" not in out
    mocks["store"].count.assert_not_called()


def test_main_no_results_mentions_threshold(
    mocks: dict[str, MagicMock],
    env: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env.setenv("SCORE_THRESHOLD", "0.6")
    assert query.main(["nothing"]) == 0
    out = capsys.readouterr().out
    assert "score threshold is set to 0.6" in out
    assert mocks["store"].search.call_args.kwargs["score_threshold"] == 0.6


def test_main_embedder_error_returns_one(
    mocks: dict[str, MagicMock], capsys: pytest.CaptureFixture[str]
) -> None:
    mocks["embedder_cls"].side_effect = EmbeddingError("cannot reach LLM")
    assert query.main(["hello"]) == 1
    assert "Error: cannot reach LLM" in capsys.readouterr().err
    mocks["store_cls"].assert_not_called()


def test_main_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="SLACK_BOT_TOKEN"):
        query.main(["hello"])


def test_main_uses_given_config_instead_of_env(
    mocks: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    cfg = Config(
        slack_bot_token="xoxb-given",  # noqa: S106 - fake token for a test Config
        qdrant_url="http://given:6333",
        qdrant_collection="given",
        llm_base_url="http://given/v1",
        embedding_model="m",
    )
    assert query.main(["hello"], cfg=cfg) == 0
    mocks["store_cls"].assert_called_once_with(
        url="http://given:6333",
        collection="given",
        dimension=3,
        hybrid=False,
    )
    assert mocks["embedder_cls"].call_args.kwargs["base_url"] == "http://given/v1"
