"""Slack client tests against a mocked WebClient."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from slack2rag.slack_client import ChannelGoneError, SlackClient, resolve_text


def _error(code: str) -> SlackApiError:
    return SlackApiError("failed", {"ok": False, "error": code})  # type: ignore[no-untyped-call]


def _client(**methods: Any) -> tuple[SlackClient, MagicMock]:
    web = MagicMock()
    web.auth_test.return_value = {
        "url": "https://ws.slack.com/",
        "user_id": "UBOT",
        "bot_id": "BBOT",
    }
    for name, value in methods.items():
        setattr(web, name, value)
    return SlackClient("xoxb", api_pause=0, client=web), web


def test_identify_and_permalink() -> None:
    client, web = _client()
    me = client.identify()
    client.identify()
    assert me.user_id == "UBOT"
    assert me.workspace_url == "https://ws.slack.com"
    assert web.auth_test.call_count == 1
    assert client.permalink("C1", "1.5") == "https://ws.slack.com/archives/C1/p15"
    assert client.permalink("C1", "2.5", "1.5").endswith("?thread_ts=1.5&cid=C1")
    assert client.web_client is web


def test_permalink_without_workspace_url() -> None:
    client, web = _client()
    web.auth_test.return_value = {}
    assert client.permalink("C1", "1.5") == ""


def test_member_channels_pages_and_never_joins() -> None:
    pages = [
        {"channels": [{"id": "C1"}], "response_metadata": {"next_cursor": "n"}},
        {"channels": [{"id": "C2"}], "response_metadata": {"next_cursor": ""}},
    ]
    client, web = _client(users_conversations=MagicMock(side_effect=pages))
    assert [c["id"] for c in client.member_channels()] == ["C1", "C2"]
    kwargs = web.users_conversations.call_args_list[0].kwargs
    assert kwargs["types"] == "public_channel"
    assert web.users_conversations.call_args_list[1].kwargs["cursor"] == "n"
    web.conversations_join.assert_not_called()


def test_history_and_gone_channels() -> None:
    client, web = _client(
        conversations_history=MagicMock(return_value={"messages": [{"ts": "1"}]})
    )
    assert list(client.history("C1", oldest="0", latest="2", inclusive=True)) == [
        {"ts": "1"}
    ]
    assert web.conversations_history.call_args.kwargs["inclusive"] is True
    web.conversations_history.side_effect = _error("not_in_channel")
    with pytest.raises(ChannelGoneError):
        list(client.history("C1"))
    web.conversations_history.side_effect = _error("ratelimited")
    with pytest.raises(SlackApiError):
        list(client.history("C1"))


def test_thread() -> None:
    client, web = _client(
        conversations_replies=MagicMock(
            return_value={"messages": [{"ts": "1"}, {"ts": "2"}]}
        )
    )
    assert len(client.thread("C1", "1")) == 2
    web.conversations_replies.side_effect = _error("thread_not_found")
    assert client.thread("C1", "1") == []
    web.conversations_replies.side_effect = _error("fatal_error")
    with pytest.raises(SlackApiError):
        client.thread("C1", "1")


def test_pins_and_bookmarks() -> None:
    client, web = _client(
        pins_list=MagicMock(
            return_value={
                "items": [
                    {"type": "message", "message": {"ts": "1"}},
                    {"type": "file", "file": {}},
                ]
            }
        ),
        bookmarks_list=MagicMock(return_value={"bookmarks": [{"id": "Bk"}]}),
    )
    assert client.pins("C1") == [{"ts": "1"}]
    assert client.bookmarks("C1") == [{"id": "Bk"}]
    web.pins_list.side_effect = _error("missing_scope")
    web.bookmarks_list.side_effect = _error("missing_scope")
    assert client.pins("C1") == []
    assert client.bookmarks("C1") == []
    web.pins_list.side_effect = _error("channel_not_found")
    web.bookmarks_list.side_effect = _error("is_archived")
    with pytest.raises(ChannelGoneError):
        client.pins("C1")
    with pytest.raises(ChannelGoneError):
        client.bookmarks("C1")


def test_channel_info() -> None:
    client, web = _client(
        conversations_info=MagicMock(return_value={"channel": {"id": "C1"}})
    )
    assert client.channel_info("C1") == {"id": "C1"}
    web.conversations_info.side_effect = _error("channel_not_found")
    with pytest.raises(ChannelGoneError):
        client.channel_info("C1")
    web.conversations_info.side_effect = _error("fatal_error")
    with pytest.raises(SlackApiError):
        client.channel_info("C1")


def test_member_names_in_memory_only() -> None:
    client, web = _client(
        users_list=MagicMock(
            return_value={
                "members": [
                    {
                        "name": "alice",
                        "profile": {"real_name": "Alice Smith", "display_name": ""},
                    }
                ]
            }
        )
    )
    assert client.member_names() == {"alice", "Alice Smith"}
    web.users_list.side_effect = _error("missing_scope")
    assert client.member_names() == set()


def test_resolve_text() -> None:
    text = (
        "<@U1> <@U2|bob> see <#C1|laser> <#C2> <https://a.org|docs> <https://b.org>"
        " <https://c.org|https://c.org> <!here> <!subteam^S1|@hosts> <!subteam^S2>"
    )
    assert resolve_text(text) == (
        "@member @member see #laser #C2 docs (https://a.org) https://b.org"
        " https://c.org @here @hosts @group"
    )
