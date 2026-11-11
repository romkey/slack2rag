"""SyncState tests."""

from __future__ import annotations

import json
import logging

from src.state import SyncState


def test_missing_file_starts_empty(tmp_path) -> None:
    state = SyncState(str(tmp_path / "state.json"))
    assert state.get_cursor("C1") is None
    # Constructing must not create the file.
    assert not (tmp_path / "state.json").exists()


def test_set_cursor_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = SyncState(str(path))
    state.set_cursor("C1", "1700000000.000100")
    state.set_cursor("C2", "1700000001.000200")

    assert state.get_cursor("C1") == "1700000000.000100"
    assert json.loads(path.read_text()) == {
        "C1": "1700000000.000100",
        "C2": "1700000001.000200",
    }

    reloaded = SyncState(str(path))
    assert reloaded.get_cursor("C1") == "1700000000.000100"
    assert reloaded.get_cursor("C2") == "1700000001.000200"
    assert reloaded.get_cursor("C3") is None


def test_set_cursor_overwrites_existing_value(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = SyncState(str(path))
    state.set_cursor("C1", "1.0")
    state.set_cursor("C1", "2.0")
    assert state.get_cursor("C1") == "2.0"
    assert json.loads(path.read_text()) == {"C1": "2.0"}


def test_loads_existing_file(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"C9": "123.456"}))
    assert SyncState(str(path)).get_cursor("C9") == "123.456"


def test_corrupt_file_logs_warning_and_starts_empty(tmp_path, caplog) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    with caplog.at_level(logging.WARNING, logger="src.state"):
        state = SyncState(str(path))
    assert state.get_cursor("C1") is None
    assert "Could not read state file" in caplog.text

    # A subsequent save replaces the corrupt content with valid JSON.
    state.set_cursor("C1", "5.0")
    assert json.loads(path.read_text()) == {"C1": "5.0"}


def test_unreadable_path_logs_warning(tmp_path, caplog) -> None:
    # A directory exists at the path, so open() raises an OSError.
    path = tmp_path / "state.json"
    path.mkdir()
    with caplog.at_level(logging.WARNING, logger="src.state"):
        state = SyncState(str(path))
    assert state.get_cursor("C1") is None
    assert "Could not read state file" in caplog.text


def test_save_creates_parent_directories(tmp_path) -> None:
    path = tmp_path / "nested" / "deeper" / "state.json"
    state = SyncState(str(path))
    state.set_cursor("C1", "1.0")
    assert json.loads(path.read_text()) == {"C1": "1.0"}


def test_save_with_bare_filename_uses_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = SyncState("state.json")
    state.set_cursor("C1", "1.0")
    assert json.loads((tmp_path / "state.json").read_text()) == {"C1": "1.0"}


def test_save_failure_logs_warning_but_keeps_memory(tmp_path, caplog) -> None:
    path = tmp_path / "state.json"
    path.mkdir()  # writing to a directory raises IsADirectoryError (OSError)
    with caplog.at_level(logging.WARNING, logger="src.state"):
        state = SyncState(str(path))
        caplog.clear()
        state.set_cursor("C1", "1.0")
    assert "Could not save state" in caplog.text
    # The in-memory cursor still advances even though persistence failed.
    assert state.get_cursor("C1") == "1.0"
