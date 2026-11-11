"""Tests for the slack2rag entry point (src/main.py)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src import __version__, main
from src.config import Config
from src.embedder import EmbeddingError
from src.processor import ChannelStats
from src.state import SyncState


# ── shared fixtures ──────────────────────────────────────────────────────────


def _cfg(tmp_path, **overrides) -> Config:
    base = dict(
        slack_bot_token="xoxb-test",
        state_file=str(tmp_path / "data" / "state.json"),
        batch_size=50,
        api_pause=0.0,
    )
    base.update(overrides)
    return Config(**base)


def _doc(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


@pytest.fixture
def fake_build(monkeypatch):
    """Replace processor.build_documents with a recorder that returns one doc per message."""
    calls: list[dict] = []

    def fake_build_documents(**kwargs):
        calls.append(kwargs)
        return [_doc(f"doc-{kwargs['root_msg']['ts']}")]

    monkeypatch.setattr(main, "build_documents", fake_build_documents)
    return calls


def _slack(messages: list[dict] | None = None, replies: dict | None = None) -> MagicMock:
    slack = MagicMock()
    msgs = messages or []
    slack.get_channel_messages.side_effect = lambda cid, oldest_ts=None: iter(list(msgs))
    slack.get_thread_replies.side_effect = lambda cid, ts: list((replies or {}).get(ts, []))
    slack.get_reactions.side_effect = lambda m: (
        len(m.get("reactions", [])), [r["name"] for r in m.get("reactions", [])]
    )
    slack.get_attachments.side_effect = lambda m: list(m.get("files", []))
    slack.make_permalink.side_effect = lambda cid, ts: f"https://x/{cid}/{ts}"
    slack.get_channel_topic.side_effect = lambda ch: ch.get("topic", "")
    return slack


def _embedder() -> MagicMock:
    emb = MagicMock()
    emb.embed.side_effect = lambda texts: [[0.5]] * len(texts)
    emb.dimension = 1
    return emb


# ── _build_docs_for_message ──────────────────────────────────────────────────


def test_build_docs_aggregates_reactions_and_attachments(tmp_path, fake_build) -> None:
    cfg = _cfg(tmp_path, min_message_length=5, reaction_boost_threshold=2)
    slack = _slack()
    msg = {"ts": "1.0", "reactions": [{"name": "+1"}], "files": ["a.png"]}
    replies = [
        {"ts": "1.1", "reactions": [{"name": "tada"}, {"name": "eyes"}], "files": ["b.pdf"]},
        {"ts": "1.2"},
    ]
    channel = {"id": "C1", "name": "eng", "topic": "T"}

    docs = main._build_docs_for_message(msg, replies, channel, slack, cfg)

    assert [d.text for d in docs] == ["doc-1.0"]
    kw = fake_build[0]
    assert kw["reaction_count"] == 3
    assert kw["reactions"] == ["+1", "tada", "eyes"]
    assert kw["attachments"] == ["a.png", "b.pdf"]
    assert kw["permalink"] == "https://x/C1/1.0"
    assert kw["channel_topic"] == "T"
    assert kw["min_message_length"] == 5
    assert kw["reaction_boost_threshold"] == 2
    assert kw["resolve_text"] is slack.resolve_text
    assert kw["get_user_name"] is slack.get_user_name
    assert kw["replies"] is replies


# ── sync_channel ─────────────────────────────────────────────────────────────


def test_sync_channel_indexes_batches_and_advances_cursor(tmp_path, fake_build) -> None:
    cfg = _cfg(tmp_path, batch_size=2)
    msgs = [
        {"ts": "100.0", "user": "U1", "text": "deploy pipeline broken"},
        {"ts": "300.0", "user": "U2", "text": "kubernetes deploy", "reply_count": 1,
         "thread_ts": "300.0"},
        {"ts": "200.0", "text": "anonymous"},
        # reply_count but not a thread root -> replies not fetched
        {"ts": "400.0", "user": "U1", "text": "x", "reply_count": 2, "thread_ts": "1.0"},
        {"ts": "500.0", "user": "U1", "text": "tail"},
    ]
    replies = {"300.0": [{"ts": "301.0", "user": "U3", "text": "deploy fixed"},
                         {"ts": "302.0", "text": "anon reply"}]}
    slack = _slack(msgs, replies)
    store = MagicMock()
    emb = _embedder()
    sparse = MagicMock()
    sparse.encode.side_effect = lambda texts: [{"i": t} for t in texts]
    state = SyncState(cfg.state_file)
    pg = MagicMock()

    count, stats = main.sync_channel(
        {"id": "C1", "name": "eng"}, slack, store, emb, sparse, state, cfg, pg=pg,
    )

    assert count == 5
    # batch_size=2 -> flush at 2, 4, then remainder 1
    assert [len(c.args[0]) for c in store.upsert.call_args_list] == [2, 2, 1]
    first = store.upsert.call_args_list[0]
    assert first.args[1] == [[0.5], [0.5]]
    assert first.kwargs["sparse_vectors"] == [{"i": "doc-100.0"}, {"i": "doc-300.0"}]
    slack.get_thread_replies.assert_called_once_with("C1", "300.0")
    assert state.get_cursor("C1") == "500.0"
    assert json.loads(open(cfg.state_file).read()) == {"C1": "500.0"}

    assert stats.user_counts == Counter({"U1": 3, "U2": 1, "U3": 1})
    assert stats.token_counts["deploy"] == 3
    assert sorted(stats.timestamps) == [100.0, 200.0, 300.0, 400.0, 500.0]
    assert pg.persist_channel_messages_and_users.call_count == 5
    _, _, root, rep = pg.persist_channel_messages_and_users.call_args_list[1].args
    assert root["ts"] == "300.0" and len(rep) == 2


def test_sync_channel_uses_cursor_and_no_new_messages(tmp_path, fake_build, caplog) -> None:
    cfg = _cfg(tmp_path)
    state = SyncState(cfg.state_file)
    state.set_cursor("C1", "50.0")
    slack = _slack([])
    store = MagicMock()

    with caplog.at_level(logging.INFO, logger="slack2rag"):
        count, stats = main.sync_channel({"id": "C1"}, slack, store, _embedder(), None, state, cfg)

    assert count == 0
    assert stats.timestamps == []
    slack.get_channel_messages.assert_called_once_with("C1", oldest_ts="50.0")
    store.upsert.assert_not_called()
    assert state.get_cursor("C1") == "50.0"
    assert "#C1  no new messages" in caplog.text


def test_sync_channel_without_sparse_or_pg(tmp_path, fake_build) -> None:
    cfg = _cfg(tmp_path)
    state = SyncState(cfg.state_file)
    slack = _slack([{"ts": "10.0", "user": "U1", "text": "hello"}])
    store = MagicMock()
    count, _ = main.sync_channel({"id": "C1"}, slack, store, _embedder(), None, state, cfg)
    assert count == 1
    assert store.upsert.call_args.kwargs["sparse_vectors"] is None
    assert state.get_cursor("C1") == "10.0"


# ── refresh_threads ──────────────────────────────────────────────────────────


def test_refresh_threads(tmp_path, fake_build, monkeypatch) -> None:
    monkeypatch.setattr(main.time, "time", lambda: 10_000.0)
    cfg = _cfg(tmp_path, batch_size=1)
    msgs = [
        {"ts": "1.0", "reply_count": 0},
        {"ts": "2.0", "reply_count": 3, "thread_ts": "2.0"},
        {"ts": "3.0", "reply_count": 1, "thread_ts": "1.5"},  # reply, not root
        {"ts": "4.0", "reply_count": 1, "thread_ts": "4.0"},
    ]
    slack = _slack(msgs, {"2.0": [{"ts": "2.1"}], "4.0": [{"ts": "4.1"}]})
    store = MagicMock()
    pg = MagicMock()

    n = main.refresh_threads({"id": "C1", "name": "eng"}, slack, store, _embedder(), None,
                             cfg, lookback_hours=2, pg=pg)

    assert n == 2
    slack.get_channel_messages.assert_called_once_with("C1", oldest_ts=str(10_000.0 - 7200))
    assert [c.args[1] for c in slack.get_thread_replies.call_args_list] == ["2.0", "4.0"]
    assert store.upsert.call_count == 2
    assert pg.persist_channel_messages_and_users.call_count == 2


def test_refresh_threads_flushes_remainder(tmp_path, fake_build) -> None:
    cfg = _cfg(tmp_path, batch_size=10)
    slack = _slack([{"ts": "2.0", "reply_count": 1, "thread_ts": "2.0"}])
    store = MagicMock()
    assert main.refresh_threads({"id": "C1"}, slack, store, _embedder(), None, cfg, 1) == 1
    store.upsert.assert_called_once()


def test_refresh_threads_nothing(tmp_path, fake_build) -> None:
    cfg = _cfg(tmp_path)
    store = MagicMock()
    assert main.refresh_threads({"id": "C1"}, _slack([]), store, _embedder(), None, cfg, 1) == 0
    store.upsert.assert_not_called()


# ── channel stats persistence ────────────────────────────────────────────────


def test_channel_stats_roundtrip(tmp_path) -> None:
    path = str(tmp_path / "nested" / "stats.json")
    cs = ChannelStats()
    cs.token_counts.update({"deploy": 3})
    cs.user_counts.update({"U1": 2})
    cs.timestamps = [1.0, 2.0]
    main._save_channel_stats(path, {"C1": cs})
    loaded = main._load_channel_stats(path)
    assert set(loaded) == {"C1"}
    assert loaded["C1"].token_counts == Counter({"deploy": 3})
    assert loaded["C1"].user_counts == Counter({"U1": 2})
    assert loaded["C1"].timestamps == [1.0, 2.0]


def test_load_channel_stats_missing_and_corrupt(tmp_path) -> None:
    assert main._load_channel_stats(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert main._load_channel_stats(str(bad)) == {}


def test_save_channel_stats_oserror_is_logged(tmp_path, monkeypatch, caplog) -> None:
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    with caplog.at_level(logging.WARNING, logger="slack2rag"):
        main._save_channel_stats(str(tmp_path / "s.json"), {"C1": ChannelStats()})
    assert "Could not save channel stats: disk full" in caplog.text


# ── _index_summaries ─────────────────────────────────────────────────────────


def test_index_summaries(monkeypatch) -> None:
    calls: dict[str, list] = {"channel": [], "workspace": [], "user": [], "team": []}

    def chan(ch, count, stats=None, get_user_name=None):
        calls["channel"].append((ch["id"], count, stats, get_user_name))
        return _doc(f"chan-{ch['id']}")

    def ws(channels, counts):
        calls["workspace"].append(dict(counts))
        return _doc("ws")

    def user(profile, active_channels=None):
        calls["user"].append((profile["user_id"], active_channels))
        return _doc(f"user-{profile['user_id']}")

    def team(profiles):
        calls["team"].append(len(profiles))
        return _doc("team")

    monkeypatch.setattr(main, "build_channel_summary", chan)
    monkeypatch.setattr(main, "build_workspace_summary", ws)
    monkeypatch.setattr(main, "build_user_summary", user)
    monkeypatch.setattr(main, "build_team_summary", team)

    channels = [{"id": "C1", "name": "eng"}, {"id": "C2", "name": "ops"}]
    s1 = ChannelStats(user_counts=Counter({"U1": 5, "U2": 1}))
    s2 = ChannelStats(user_counts=Counter({"U1": 9}))
    # stats for a channel no longer in the list -> name falls back to id
    s3 = ChannelStats(user_counts=Counter({"U2": 2}))
    stats = {"C1": s1, "C2": s2, "COLD": s3}
    profiles = {
        "U1": {"user_id": "U1"},
        "U2": {"user_id": "U2"},
        "U3": {"user_id": "U3"},
        "U4": {"user_id": "U4", "deleted": True},
    }
    store = MagicMock()
    store.count_by_channel.side_effect = lambda cid: {"C1": 10, "C2": 20}[cid]
    get_name = MagicMock()

    main._index_summaries(channels, stats, profiles, get_name, store, _embedder(), None)

    assert calls["channel"] == [("C1", 10, s1, get_name), ("C2", 20, s2, get_name)]
    assert calls["workspace"] == [{"C1": 10, "C2": 20}]
    assert dict(calls["user"]) == {
        "U1": ["ops", "eng"],
        "U2": ["COLD", "eng"],
        "U3": None,
    }
    assert calls["team"] == [4]
    texts = [d.text for d in store.upsert.call_args.args[0]]
    assert texts == ["chan-C1", "chan-C2", "ws", "user-U1", "user-U2", "user-U3", "team"]


# ── _get_channels ────────────────────────────────────────────────────────────


def test_get_channels_include_list_and_blacklist(tmp_path, caplog) -> None:
    cfg = _cfg(tmp_path, slack_channels="eng, ops,random",
               slack_channel_blacklist="#random, C_OPS")
    slack = MagicMock()
    slack.get_public_channels.return_value = [
        {"id": "C_ENG", "name": "eng"},
        {"id": "C_OPS", "name": "ops"},
        {"id": "C_RND", "name": "random"},
        {"id": "C_X"},
    ]
    with caplog.at_level(logging.INFO, logger="slack2rag"):
        result = main._get_channels(cfg, slack)
    slack.get_public_channels.assert_called_once_with(["eng", "ops", "random"])
    assert [c["id"] for c in result] == ["C_ENG", "C_X"]
    assert "Blacklist: skipping 2 channel(s): C_OPS, random" in caplog.text


def test_get_channels_all_public_no_blacklist(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    slack = MagicMock()
    slack.get_public_channels.return_value = [{"id": "C1"}]
    assert main._get_channels(cfg, slack) == [{"id": "C1"}]
    slack.get_public_channels.assert_called_once_with(None)


def test_get_channels_blacklist_no_match(tmp_path, caplog) -> None:
    cfg = _cfg(tmp_path, slack_channel_blacklist="zzz")
    slack = MagicMock()
    slack.get_public_channels.return_value = [{"id": "C1", "name": "a"}]
    with caplog.at_level(logging.INFO, logger="slack2rag"):
        assert len(main._get_channels(cfg, slack)) == 1
    assert "Blacklist" not in caplog.text


# ── run_once ─────────────────────────────────────────────────────────────────


def test_run_once_no_channels(tmp_path, monkeypatch, caplog) -> None:
    cfg = _cfg(tmp_path)
    slack = MagicMock()
    slack.get_public_channels.return_value = []
    sync = MagicMock()
    monkeypatch.setattr(main, "sync_channel", sync)
    with caplog.at_level(logging.WARNING, logger="slack2rag"):
        main.run_once(cfg, slack, MagicMock(), MagicMock(), None, MagicMock())
    sync.assert_not_called()
    assert "No accessible public channels found" in caplog.text


def test_run_once_full_cycle(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, thread_update_lookback_hours=6)
    stats_file = tmp_path / "data" / "channel_stats.json"
    existing = ChannelStats(user_counts=Counter({"U1": 1}))
    main._save_channel_stats(str(stats_file), {"C1": existing})

    slack = MagicMock()
    slack.get_public_channels.return_value = [{"id": "C1"}, {"id": "C2"}]
    slack.get_user_profiles.return_value = {"U1": {"user_id": "U1"}}

    def fake_sync(ch, *args, pg=None):
        return 3, ChannelStats(user_counts=Counter({"U9": 2}))

    refresh = MagicMock(return_value=0)
    summaries = MagicMock()
    monkeypatch.setattr(main, "sync_channel", fake_sync)
    monkeypatch.setattr(main, "refresh_threads", refresh)
    monkeypatch.setattr(main, "_index_summaries", summaries)
    store = MagicMock()
    store.count.return_value = 99
    pg = MagicMock()

    main.run_once(cfg, slack, store, MagicMock(), None, MagicMock(), pg=pg)

    saved = main._load_channel_stats(str(stats_file))
    assert saved["C1"].user_counts == Counter({"U1": 1, "U9": 2})  # merged
    assert saved["C2"].user_counts == Counter({"U9": 2})  # new
    assert refresh.call_count == 2
    assert refresh.call_args.args[6] == 6
    assert refresh.call_args.kwargs["pg"] is pg
    ch_arg, stats_arg, profiles_arg = summaries.call_args.args[:3]
    assert [c["id"] for c in ch_arg] == ["C1", "C2"]
    assert set(stats_arg) == {"C1", "C2"}
    assert profiles_arg == {"U1": {"user_id": "U1"}}


def test_run_once_skips_thread_refresh_when_disabled(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    slack = MagicMock()
    slack.get_public_channels.return_value = [{"id": "C1"}]
    monkeypatch.setattr(main, "sync_channel", lambda *a, **k: (0, ChannelStats()))
    refresh = MagicMock()
    monkeypatch.setattr(main, "refresh_threads", refresh)
    monkeypatch.setattr(main, "_index_summaries", MagicMock())
    main.run_once(cfg, slack, MagicMock(), MagicMock(), None, MagicMock())
    refresh.assert_not_called()


# ── _check_llm ───────────────────────────────────────────────────────────────


def test_check_llm_embedding_ok_no_eval(tmp_path, monkeypatch, caplog) -> None:
    emb = MagicMock()
    emb.embed.return_value = [[0.25, 0.5]]
    emb_cls = MagicMock(return_value=emb)
    monkeypatch.setattr(main, "Embedder", emb_cls)
    chat = MagicMock()
    monkeypatch.setattr(main, "_chat_completion", chat)
    cfg = _cfg(tmp_path, llm_base_url="http://llm/v1", llm_api_key="k",
               embedding_model="e", embedding_prefix="p:", embedding_context_length=10)

    with caplog.at_level(logging.INFO, logger="slack2rag"):
        assert main._check_llm(cfg) is True

    emb_cls.assert_called_once_with(base_url="http://llm/v1", model="e", context_length=10,
                                    api_key="k", input_prefix="p:")
    chat.assert_not_called()
    assert "Embedding model OK (dimension=2, sample[0]=0.2500)" in caplog.text
    assert "Evaluator: skipped" in caplog.text


def test_check_llm_embedding_empty_vector(tmp_path, monkeypatch, caplog) -> None:
    emb = MagicMock()
    emb.embed.return_value = [[]]
    monkeypatch.setattr(main, "Embedder", MagicMock(return_value=emb))
    with caplog.at_level(logging.INFO, logger="slack2rag"):
        assert main._check_llm(_cfg(tmp_path)) is True
    assert "dimension=0, sample[0]=0.0000" in caplog.text


def test_check_llm_embedding_failure_and_eval_ok(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(main, "Embedder", MagicMock(side_effect=EmbeddingError("down")))
    chat = MagicMock(return_value='  {"ok": true}\n')
    monkeypatch.setattr(main, "_chat_completion", chat)
    cfg = _cfg(tmp_path, llm_base_url="http://llm/v1", llm_api_key="k1",
               eval_model="judge", eval_base_url="http://eval/v1", eval_api_key="k2")

    with caplog.at_level(logging.INFO, logger="slack2rag"):
        assert main._check_llm(cfg) is False

    assert "Embedding model FAILED: down" in caplog.text
    assert 'Evaluator model OK (response: {"ok": true})' in caplog.text
    args, kwargs = chat.call_args
    assert args[0] == "http://eval/v1"
    assert args[1] == "k2"
    assert args[2] == "judge"
    assert kwargs["timeout"] == 30


def test_check_llm_eval_failure_uses_llm_fallback(tmp_path, monkeypatch, caplog) -> None:
    emb = MagicMock()
    emb.embed.return_value = [[1.0]]
    monkeypatch.setattr(main, "Embedder", MagicMock(return_value=emb))
    chat = MagicMock(return_value=None)
    monkeypatch.setattr(main, "_chat_completion", chat)
    cfg = _cfg(tmp_path, llm_base_url="http://llm/v1", llm_api_key="k1", eval_model="judge")

    with caplog.at_level(logging.INFO, logger="slack2rag"):
        assert main._check_llm(cfg) is False
    assert chat.call_args.args[:3] == ("http://llm/v1", "k1", "judge")
    assert "Evaluator model FAILED" in caplog.text


# ── CLI / main() ─────────────────────────────────────────────────────────────


def test_parse_args() -> None:
    assert main._parse_args([]).check is False
    assert main._parse_args(["--check"]).check is True


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main._parse_args(["--version"])
    assert exc.value.code == 0
    assert f"slack2rag {__version__}" in capsys.readouterr().out


def test_unknown_arg_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        main._parse_args(["--bogus"])
    assert exc.value.code == 2


class _StopLoop(Exception):
    pass


@pytest.fixture
def app(monkeypatch, tmp_path):
    """Patch all external collaborators used by main.main()."""
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("STATE_FILE", str(tmp_path / "state.json"))
    for var in ("DATABASE_URL", "EVAL_TEST", "EVAL_PROMPT", "EVAL_MODEL", "RUN_ONCE",
                "HYBRID_SEARCH", "SYNC_INTERVAL_MINUTES", "EMBEDDING_PREFIX",
                "THREAD_UPDATE_LOOKBACK_HOURS", "EVAL_BASE_URL", "EVAL_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    m = SimpleNamespace()
    m.slack = MagicMock()
    m.slack_cls = MagicMock(return_value=m.slack)
    m.pg = MagicMock()
    m.pg_from_url = MagicMock(return_value=None)
    m.embedder = MagicMock(dimension=768)
    m.embedder_cls = MagicMock(return_value=m.embedder)
    m.sparse_cls = MagicMock()
    m.store = MagicMock()
    m.store_cls = MagicMock(return_value=m.store)
    m.state_cls = MagicMock()
    m.run_once = MagicMock()
    m.run_eval = MagicMock()
    m.check = MagicMock(return_value=True)
    m.sleep = MagicMock()

    monkeypatch.setattr(main, "SlackClient", m.slack_cls)
    monkeypatch.setattr(main.PgStore, "from_url", m.pg_from_url)
    monkeypatch.setattr(main, "Embedder", m.embedder_cls)
    monkeypatch.setattr(main, "SparseEncoder", m.sparse_cls)
    monkeypatch.setattr(main, "VectorStore", m.store_cls)
    monkeypatch.setattr(main, "SyncState", m.state_cls)
    monkeypatch.setattr(main, "run_once", m.run_once)
    monkeypatch.setattr(main, "run_eval", m.run_eval)
    monkeypatch.setattr(main, "_check_llm", m.check)
    monkeypatch.setattr(main.time, "sleep", m.sleep)
    m.env = monkeypatch
    return m


@pytest.mark.parametrize("ok,code", [(True, 0), (False, 1)])
def test_main_check_flag_exit_codes(app, ok, code) -> None:
    app.check.return_value = ok
    with pytest.raises(SystemExit) as exc:
        main.main(["--check"])
    assert exc.value.code == code
    app.check.assert_called_once()
    app.slack_cls.assert_not_called()


def test_main_missing_token(monkeypatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(EnvironmentError):
        main.main([])


def test_main_run_once_mode(app) -> None:
    app.env.setenv("RUN_ONCE", "true")
    app.env.setenv("SLACK_API_PAUSE", "0.3")
    main.main([])

    app.slack_cls.assert_called_once_with("xoxb-test", api_pause=0.3)
    app.slack.prefetch_users.assert_called_once()
    app.slack.fetch_workspace_url.assert_called_once()
    app.pg_from_url.assert_called_once_with("")
    app.sparse_cls.assert_not_called()
    app.store_cls.assert_called_once_with(
        url="http://qdrant:6333", collection="slack_messages", dimension=768, hybrid=False,
    )
    app.run_once.assert_called_once()
    args, kwargs = app.run_once.call_args
    assert args[1] is app.slack and args[2] is app.store and args[3] is app.embedder
    assert args[4] is None
    assert kwargs["pg"] is None
    app.sleep.assert_not_called()


def test_main_with_postgres_and_hybrid(app) -> None:
    app.env.setenv("RUN_ONCE", "1")
    app.env.setenv("HYBRID_SEARCH", "yes")
    app.env.setenv("DATABASE_URL", "postgresql://db/x")
    app.pg_from_url.return_value = app.pg
    app.slack.user_records.return_value = [{"id": "U1"}]

    main.main([])

    app.pg_from_url.assert_called_once_with("postgresql://db/x")
    app.pg.init_schema.assert_called_once()
    app.pg.bulk_upsert_users.assert_called_once_with([{"id": "U1"}])
    app.sparse_cls.assert_called_once()
    assert app.store_cls.call_args.kwargs["hybrid"] is True
    assert app.run_once.call_args.args[4] is app.sparse_cls.return_value
    assert app.run_once.call_args.kwargs["pg"] is app.pg


def test_main_embedder_failure_exits(app, caplog) -> None:
    app.embedder_cls.side_effect = EmbeddingError("no model")
    with caplog.at_level(logging.ERROR, logger="slack2rag"):
        with pytest.raises(SystemExit) as exc:
            main.main([])
    assert exc.value.code == 1
    assert isinstance(exc.value.__cause__, EmbeddingError)
    assert "Embedding setup failed" in caplog.text
    app.run_once.assert_not_called()


def test_main_loop_survives_errors_and_sleeps(app, caplog) -> None:
    app.env.setenv("SYNC_INTERVAL_MINUTES", "2")
    app.run_once.side_effect = [EmbeddingError("embed down"), RuntimeError("qdrant down"), None]
    app.sleep.side_effect = [None, None, _StopLoop()]

    with caplog.at_level(logging.INFO, logger="slack2rag"):
        with pytest.raises(_StopLoop):
            main.main([])

    assert app.run_once.call_count == 3
    assert [c.args[0] for c in app.sleep.call_args_list] == [120, 120, 120]
    assert "Sync cycle failed (embedding error): embed down" in caplog.text
    assert "Sync cycle failed — will retry next interval" in caplog.text
    assert "Sleeping 2 minutes until next sync" in caplog.text


def test_main_eval_mode(app) -> None:
    app.env.setenv("EVAL_TEST", "1")
    app.env.setenv("EVAL_PROMPT", "rate it")
    app.env.setenv("EVAL_MODEL", "judge")
    app.env.setenv("LLM_BASE_URL", "http://llm/v1")
    app.env.setenv("EVAL_API_KEY", "ek")
    app.env.setenv("SLACK_CHANNEL_BLACKLIST", "random")
    app.slack.get_public_channels.return_value = [
        {"id": "C1", "name": "eng"}, {"id": "C2", "name": "random"},
    ]

    main.main([])

    app.run_eval.assert_called_once()
    args, kwargs = app.run_eval.call_args
    assert args[:3] == ("http://llm/v1", "judge", "rate it")
    assert args[3] == [{"id": "C1", "name": "eng"}]
    assert args[4] is app.slack
    assert kwargs == {"api_key": "ek", "pg": None}
    app.embedder_cls.assert_not_called()
    app.run_once.assert_not_called()


@pytest.mark.parametrize(
    "prompt,model",
    [("", "judge"), ("rate it", "")],
)
def test_main_eval_mode_requires_prompt_and_model(app, prompt, model) -> None:
    app.env.setenv("EVAL_TEST", "1")
    app.env.setenv("EVAL_PROMPT", prompt)
    app.env.setenv("EVAL_MODEL", model)
    with pytest.raises(SystemExit) as exc:
        main.main([])
    assert exc.value.code == 1
    app.run_eval.assert_not_called()
