"""SlackClient tests (WebClient is fully mocked; no network)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from slack2rag import slack_client
from slack2rag.slack_client import SlackClient


def _err(code: str) -> SlackApiError:
    return SlackApiError(f"error: {code}", {"ok": False, "error": code})


def _page(key: str, items: list, next_cursor: str = "") -> dict:
    return {key: items, "response_metadata": {"next_cursor": next_cursor}}


@pytest.fixture
def web(monkeypatch) -> MagicMock:
    """Replace slack_sdk.WebClient with a mock and return the instance."""
    instance = MagicMock(name="WebClient()")
    factory = MagicMock(return_value=instance)
    monkeypatch.setattr(slack_client, "WebClient", factory)
    instance._factory = factory
    return instance


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(slack_client.time, "sleep", calls.append)
    return calls


@pytest.fixture
def client(web, sleeps) -> SlackClient:
    return SlackClient("xoxb-test", api_pause=0)


# ── construction / pacing ─────────────────────────────────────────────────────


def test_init_passes_token_to_webclient(web) -> None:
    SlackClient("xoxb-abc")
    web._factory.assert_called_once_with(token="xoxb-abc")


def test_pace_sleeps_configured_amount(web, sleeps) -> None:
    sc = SlackClient("t", api_pause=0.5)
    web.auth_test.return_value = {"url": "https://x.slack.com/"}
    sc.fetch_workspace_url()
    assert sleeps == [0.5]


def test_pace_disabled_when_zero(client, web, sleeps) -> None:
    web.auth_test.return_value = {"url": "https://x.slack.com/"}
    client.fetch_workspace_url()
    assert sleeps == []


def test_default_api_pause(web, sleeps) -> None:
    sc = SlackClient("t")
    sc._pace()
    assert sleeps == [slack_client.DEFAULT_API_PAUSE]


# ── workspace / permalinks ────────────────────────────────────────────────────


def test_fetch_workspace_url_strips_slash_and_caches(client, web) -> None:
    web.auth_test.return_value = {"url": "https://acme.slack.com/"}
    assert client.fetch_workspace_url() == "https://acme.slack.com"
    assert client.fetch_workspace_url() == "https://acme.slack.com"
    web.auth_test.assert_called_once()


def test_fetch_workspace_url_missing_url(client, web) -> None:
    web.auth_test.return_value = {}
    assert client.fetch_workspace_url() == ""


def test_fetch_workspace_url_error_caches_empty(client, web) -> None:
    web.auth_test.side_effect = _err("invalid_auth")
    assert client.fetch_workspace_url() == ""
    assert client.fetch_workspace_url() == ""
    web.auth_test.assert_called_once()


def test_make_permalink_top_level(client, web) -> None:
    web.auth_test.return_value = {"url": "https://acme.slack.com/"}
    assert (
        client.make_permalink("C1", "1700000000.123456")
        == "https://acme.slack.com/archives/C1/p1700000000123456"
    )


def test_make_permalink_same_thread_ts_has_no_query(client, web) -> None:
    web.auth_test.return_value = {"url": "https://acme.slack.com"}
    link = client.make_permalink("C1", "1.2", thread_ts="1.2")
    assert link == "https://acme.slack.com/archives/C1/p12"


def test_make_permalink_reply_includes_thread(client, web) -> None:
    web.auth_test.return_value = {"url": "https://acme.slack.com"}
    link = client.make_permalink("C1", "1.3", thread_ts="1.2")
    assert link == "https://acme.slack.com/archives/C1/p13?thread_ts=1.2&cid=C1"


def test_make_permalink_without_workspace_url(client, web) -> None:
    web.auth_test.side_effect = _err("invalid_auth")
    assert client.make_permalink("C1", "1.2") == ""


# ── channels ──────────────────────────────────────────────────────────────────


def test_get_public_channels_paginates(client, web) -> None:
    web.conversations_list.side_effect = [
        _page("channels", [{"id": "C1", "name": "general"}], "cur1"),
        _page("channels", [{"id": "C2", "name": "random"}]),
    ]
    chans = client.get_public_channels()
    assert [c["id"] for c in chans] == ["C1", "C2"]

    first, second = web.conversations_list.call_args_list
    assert first.kwargs == {
        "types": "public_channel",
        "exclude_archived": True,
        "limit": 200,
    }
    assert second.kwargs["cursor"] == "cur1"


def test_get_public_channels_no_response_metadata(client, web) -> None:
    web.conversations_list.return_value = {"channels": [{"id": "C1", "name": "a"}]}
    assert len(client.get_public_channels()) == 1
    web.conversations_list.assert_called_once()


def test_get_public_channels_filters_by_name_and_id(client, web, caplog) -> None:
    web.conversations_list.return_value = _page(
        "channels",
        [
            {"id": "C1", "name": "general"},
            {"id": "C2", "name": "random"},
            {"id": "C3", "name": "dev"},
        ],
    )
    with caplog.at_level("WARNING"):
        chans = client.get_public_channels(["#general", "C3", "missing"])
    assert {c["id"] for c in chans} == {"C1", "C3"}
    warnings = [r.getMessage() for r in caplog.records if "not found" in r.getMessage()]
    assert warnings == ["Channel 'missing' not found or not accessible"]


def test_get_public_channels_error_returns_partial(client, web, caplog) -> None:
    web.conversations_list.side_effect = [
        _page("channels", [{"id": "C1", "name": "general"}], "cur1"),
        _err("ratelimited"),
    ]
    with caplog.at_level("ERROR"):
        chans = client.get_public_channels()
    assert [c["id"] for c in chans] == ["C1"]
    assert "ratelimited" in caplog.text


# ── messages ──────────────────────────────────────────────────────────────────


def test_get_channel_messages_paginates_and_filters_subtypes(client, web) -> None:
    web.conversations_history.side_effect = [
        _page(
            "messages",
            [
                {"ts": "3", "text": "hi"},
                {"ts": "2", "subtype": "channel_join"},
                {"ts": "1", "subtype": "bot_message", "text": "bot"},
            ],
            "next",
        ),
        _page(
            "messages",
            [
                {"ts": "0.5", "subtype": "channel_leave"},
                {"ts": "0.4", "subtype": "channel_purpose"},
                {"ts": "0.3", "subtype": "channel_topic"},
                {"ts": "0.2", "text": "old"},
            ],
        ),
    ]
    msgs = list(client.get_channel_messages("C1", oldest_ts="0.1"))
    assert [m["ts"] for m in msgs] == ["3", "1", "0.2"]

    first, second = web.conversations_history.call_args_list
    assert first.kwargs == {"channel": "C1", "limit": 200, "oldest": "0.1"}
    assert second.kwargs == {
        "channel": "C1",
        "limit": 200,
        "oldest": "0.1",
        "cursor": "next",
    }


def test_get_channel_messages_without_oldest(client, web) -> None:
    web.conversations_history.return_value = {"messages": [{"ts": "1"}]}
    assert len(list(client.get_channel_messages("C1"))) == 1
    assert "oldest" not in web.conversations_history.call_args.kwargs


def test_get_channel_messages_joins_and_retries(client, web) -> None:
    web.conversations_history.side_effect = [
        _err("not_in_channel"),
        _page("messages", [{"ts": "1", "text": "hello"}]),
    ]
    msgs = list(client.get_channel_messages("C1"))
    assert msgs == [{"ts": "1", "text": "hello"}]
    web.conversations_join.assert_called_once_with(channel="C1")
    assert web.conversations_history.call_count == 2


def test_get_channel_messages_join_failure_stops(client, web, caplog) -> None:
    web.conversations_history.side_effect = _err("not_in_channel")
    web.conversations_join.side_effect = _err("missing_scope")
    with caplog.at_level("WARNING"):
        assert list(client.get_channel_messages("C1")) == []
    assert "missing_scope" in caplog.text
    web.conversations_history.assert_called_once()


def test_get_channel_messages_other_error_stops(client, web, caplog) -> None:
    web.conversations_history.side_effect = _err("channel_not_found")
    with caplog.at_level("ERROR"):
        assert list(client.get_channel_messages("C1")) == []
    assert "channel_not_found" in caplog.text
    web.conversations_join.assert_not_called()


def test_get_channel_messages_error_mid_pagination_keeps_prior(client, web) -> None:
    web.conversations_history.side_effect = [
        _page("messages", [{"ts": "2"}], "c"),
        _err("ratelimited"),
    ]
    assert [m["ts"] for m in client.get_channel_messages("C1")] == ["2"]


def test_join_channel_success(client, web) -> None:
    assert client._join_channel("C1") is True
    web.conversations_join.assert_called_once_with(channel="C1")


# ── threads ───────────────────────────────────────────────────────────────────


def test_get_thread_replies_skips_root(client, web) -> None:
    web.conversations_replies.return_value = _page(
        "messages", [{"ts": "1"}, {"ts": "1.1"}, {"ts": "1.2"}]
    )
    replies = client.get_thread_replies("C1", "1")
    assert [r["ts"] for r in replies] == ["1.1", "1.2"]
    assert web.conversations_replies.call_args.kwargs == {
        "channel": "C1",
        "ts": "1",
        "limit": 200,
    }


def test_get_thread_replies_paginates(client, web) -> None:
    web.conversations_replies.side_effect = [
        _page("messages", [{"ts": "1"}, {"ts": "1.1"}], "cur"),
        _page("messages", [{"ts": "1.2"}, {"ts": "1.3"}]),
    ]
    replies = client.get_thread_replies("C1", "1")
    assert [r["ts"] for r in replies] == ["1.1", "1.2", "1.3"]
    assert web.conversations_replies.call_args_list[1].kwargs["cursor"] == "cur"


def test_get_thread_replies_root_only(client, web) -> None:
    web.conversations_replies.return_value = {"messages": [{"ts": "1"}]}
    assert client.get_thread_replies("C1", "1") == []


def test_get_thread_replies_error_returns_empty(client, web, caplog) -> None:
    web.conversations_replies.side_effect = [
        _page("messages", [{"ts": "1"}, {"ts": "1.1"}], "cur"),
        _err("thread_not_found"),
    ]
    with caplog.at_level("WARNING"):
        assert client.get_thread_replies("C1", "1") == []
    assert "thread_not_found" in caplog.text


# ── users ─────────────────────────────────────────────────────────────────────


def _user(uid: str, **kw) -> dict:
    return {"id": uid, **kw}


def test_prefetch_users_paginates_and_picks_names(client, web) -> None:
    web.users_list.side_effect = [
        _page(
            "members",
            [
                _user("U1", name="alice", profile={"display_name": "Ali", "real_name": "Alice A"}),
                _user("U2", name="bob", profile={"display_name": "", "real_name": "Bob B"}),
            ],
            "cur",
        ),
        _page(
            "members",
            [
                _user("U3", name="carol"),
                _user("U4", profile={}),
            ],
        ),
    ]
    client.prefetch_users()

    assert web.users_list.call_args_list[0].kwargs == {"limit": 200}
    assert web.users_list.call_args_list[1].kwargs == {"limit": 200, "cursor": "cur"}
    assert client.get_user_name("U1") == "Ali"
    assert client.get_user_name("U2") == "Bob B"
    assert client.get_user_name("U3") == "carol"
    assert client.get_user_name("U4") == "U4"
    web.users_info.assert_not_called()
    assert {u["id"] for u in client.user_records()} == {"U1", "U2", "U3", "U4"}
    assert set(client.get_user_profiles()) == {"U1", "U2", "U3", "U4"}


def test_prefetch_users_error_stops(client, web, caplog) -> None:
    web.users_list.side_effect = _err("missing_scope")
    with caplog.at_level("WARNING"):
        client.prefetch_users()
    assert "missing_scope" in caplog.text
    assert client.user_records() == []


def test_get_user_record_fetches_and_caches(client, web) -> None:
    user = _user("U9", name="zed", profile={"real_name": "Zed Z"})
    web.users_info.return_value = {"user": user}
    assert client.get_user_record("U9") is user
    assert client.get_user_record("U9") is user
    web.users_info.assert_called_once_with(user="U9")
    assert client.get_user_name("U9") == "Zed Z"
    assert client.get_user_profiles()["U9"]["real_name"] == "Zed Z"


def test_get_user_record_falls_back_to_id(client, web) -> None:
    web.users_info.return_value = {"user": {"id": "U7"}}
    client.get_user_record("U7")
    assert client.get_user_name("U7") == "U7"


def test_get_user_record_error_returns_none(client, web) -> None:
    web.users_info.side_effect = _err("user_not_found")
    assert client.get_user_record("UX") is None


def test_get_user_name_unknown_user_cached_as_id(client, web) -> None:
    web.users_info.side_effect = _err("user_not_found")
    assert client.get_user_name("UX") == "UX"
    assert client.get_user_name("UX") == "UX"
    web.users_info.assert_called_once()


def test_get_user_name_fetches_via_users_info(client, web) -> None:
    web.users_info.return_value = {"user": _user("U5", name="eve")}
    assert client.get_user_name("U5") == "eve"


def test_extract_profile_fields_and_defaults() -> None:
    full = SlackClient._extract_profile(
        {
            "id": "U1",
            "name": "alice",
            "tz": "America/Los_Angeles",
            "tz_label": "Pacific",
            "is_admin": True,
            "is_owner": True,
            "is_bot": False,
            "deleted": True,
            "profile": {
                "display_name": "Ali",
                "real_name": "Alice A",
                "first_name": "Alice",
                "last_name": "A",
                "title": "Eng",
                "pronouns": "she/her",
                "status_text": "busy",
                "status_emoji": ":x:",
                "email": "alice@example.com",
            },
        }
    )
    assert full["user_id"] == "U1"
    assert full["display_name"] == "Ali"
    assert full["timezone"] == "America/Los_Angeles"
    assert full["is_admin"] is True and full["deleted"] is True
    assert "email" not in full
    assert all("example.com" not in str(v) for v in full.values())

    bare = SlackClient._extract_profile({"id": "U2"})
    assert bare["username"] == ""
    assert bare["title"] == ""
    assert bare["is_bot"] is False


# ── metadata helpers ──────────────────────────────────────────────────────────


def test_get_reactions() -> None:
    msg = {"reactions": [{"name": "+1", "count": 3}, {"name": "tada"}]}
    assert SlackClient.get_reactions(msg) == (4, ["+1", "tada"])
    assert SlackClient.get_reactions({}) == (0, [])


def test_get_attachments_fallbacks() -> None:
    msg = {
        "files": [
            {"name": "a.pdf", "title": "A"},
            {"title": "Title Only"},
            {"id": "F123"},
            {},
        ]
    }
    assert SlackClient.get_attachments(msg) == [
        "a.pdf",
        "Title Only",
        "F123",
        "unknown_file",
    ]
    assert SlackClient.get_attachments({}) == []


def test_get_channel_topic_prefers_topic_then_purpose() -> None:
    assert SlackClient.get_channel_topic(
        {"topic": {"value": "T"}, "purpose": {"value": "P"}}
    ) == "T"
    assert SlackClient.get_channel_topic(
        {"topic": {"value": ""}, "purpose": {"value": "P"}}
    ) == "P"
    assert SlackClient.get_channel_topic({}) == ""


# ── text resolution ───────────────────────────────────────────────────────────


def test_resolve_text_all_forms(client, web) -> None:
    web.users_info.return_value = {"user": _user("U1", name="alice")}
    text = (
        "hey <@U1> see <#C123|general> and <https://ex.com/a|the doc> "
        "or <https://ex.com/b> <!here> <!channel> <!everyone>"
    )
    assert client.resolve_text(text) == (
        "hey @alice see #general and the doc (https://ex.com/a) "
        "or https://ex.com/b @here @channel @everyone"
    )


def test_resolve_text_unknown_user_and_plain_text(client, web) -> None:
    web.users_info.side_effect = _err("user_not_found")
    assert client.resolve_text("ping <@UNKNOWN1>") == "ping @UNKNOWN1"
    assert client.resolve_text("nothing special") == "nothing special"
