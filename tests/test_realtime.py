"""Live event handling tests (no Socket Mode connection)."""

from __future__ import annotations

import datetime
from typing import Any, cast
from unittest.mock import MagicMock, patch

from slack2rag.pg_store import PgStore, UnitRow
from slack2rag.realtime import LiveEvents

from .conftest import ME


class FakeStore:
    def __init__(self) -> None:
        self.dirty: list[tuple[str, str]] = []
        self.ensured: list[tuple[str, str, str]] = []
        self.rows: dict[str, list[UnitRow]] = {}

    def units_containing(self, channel: str, ts: str) -> list[UnitRow]:
        return self.rows.get(ts, [])

    def mark_dirty(self, channel: str, key: str) -> bool:
        self.dirty.append((channel, key))
        return True

    def ensure_dirty(self, channel: str, key: str, kind: str) -> None:
        self.ensured.append((channel, key, kind))


def _row(key: str) -> UnitRow:
    return UnitRow(
        "C1",
        key,
        "session",
        [key],
        "h",
        "v",
        True,
        False,
        False,
        datetime.datetime.now(tz=datetime.UTC),
    )


def _live() -> tuple[LiveEvents, FakeStore]:
    store = FakeStore()
    return LiveEvents(cast(PgStore, store), ME, frozenset({"CDENY"})), store


def test_new_reply_marks_its_thread() -> None:
    live, store = _live()
    live.handle({"type": "message", "channel": "C1", "ts": "2.0", "thread_ts": "1.0"})
    assert store.ensured == [("C1", "1.0", "thread")]
    assert live.wake.is_set()


def test_new_top_level_message_waits_for_catchup() -> None:
    live, store = _live()
    live.handle({"type": "message", "channel": "C1", "ts": "2.0"})
    assert store.ensured == []
    assert not live.wake.is_set()


def test_edit_and_delete_mark_units() -> None:
    live, store = _live()
    store.rows["5.0"] = [_row("4.0")]
    live.handle(
        {
            "type": "message",
            "subtype": "message_changed",
            "channel": "C1",
            "message": {"ts": "5.0", "thread_ts": "3.0"},
        }
    )
    live.handle(
        {
            "type": "message",
            "subtype": "message_deleted",
            "channel": "C1",
            "deleted_ts": "5.0",
            "previous_message": {},
        }
    )
    assert store.dirty == [("C1", "4.0"), ("C1", "4.0")]
    assert store.ensured == [("C1", "3.0", "thread")]
    live.handle({"type": "message", "subtype": "message_deleted", "channel": "C1"})


def test_denylisted_private_and_other_events_are_ignored() -> None:
    live, store = _live()
    live.handle({"type": "message", "channel": "CDENY", "ts": "2", "thread_ts": "1"})
    live.handle(
        {
            "type": "message",
            "channel": "G1",
            "channel_type": "group",
            "ts": "2",
            "thread_ts": "1",
        }
    )
    live.handle({"type": "reaction_added", "channel": "C1"})
    live.handle({"type": "message"})
    live.handle(
        {"type": "message", "channel": "C1", "subtype": "channel_join", "ts": "1"}
    )
    assert store.ensured == []
    assert store.dirty == []


def test_leaving_a_channel_queues_it_to_be_forgotten() -> None:
    live, _ = _live()
    live.handle({"type": "member_left_channel", "channel": "C1", "user": "UBOT"})
    live.handle({"type": "member_left_channel", "channel": "C2", "user": "USOMEONE"})
    live.handle({"type": "channel_left", "channel": "C3"})
    live.handle({"type": "channel_deleted", "channel": {"id": "C4"}})
    assert live.take_forgotten() == {"C1", "C3", "C4"}
    assert live.take_forgotten() == set()


def test_start_and_stop_wire_up_socket_mode() -> None:
    live, store = _live()
    client = MagicMock()
    client.socket_mode_request_listeners = []
    with patch("slack_sdk.socket_mode.SocketModeClient", return_value=client):
        live.start("xapp-1", web_client=None)
    [listener] = client.socket_mode_request_listeners
    request: Any = MagicMock(
        type="events_api",
        envelope_id="e1",
        payload={
            "event": {"type": "message", "channel": "C1", "ts": "2", "thread_ts": "1"}
        },
    )
    listener(client, request)
    client.send_socket_mode_response.assert_called_once()
    assert store.ensured == [("C1", "1", "thread")]
    bad: Any = MagicMock(type="events_api", envelope_id="e2", payload={"event": None})
    with patch.object(live, "handle", side_effect=RuntimeError("x")):
        listener(client, bad)  # logged, not raised
    live.stop()
    client.close.assert_called_once()
    live.stop()
