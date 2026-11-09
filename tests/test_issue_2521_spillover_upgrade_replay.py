"""#2521: v0.29.2 bare spillover markers remain one-to-one replay identities (I1, I6)."""

import logging
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from copy import deepcopy

import pytest

from hermes_lcm import ingest_persisted
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_10_persisted_output_spillover import _messages, _write_output


SESSION = "spillover-upgrade"
CONVERSATION = "spillover-conversation"


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    directory = tmp_path / "tmp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))


def _engine(tmp_path):
    return LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "hermes" / "externalized"),
        ),
        hermes_home=str(tmp_path / "hermes"),
    )


def _run_phase(tmp_path, history):
    engine = _engine(tmp_path)
    try:
        engine.on_session_start(
            SESSION, platform="cli", context_length=128000, conversation_id=CONVERSATION,
        )
        engine.ingest(deepcopy(history))
        return dict(engine._last_ingest_reconciliation)
    finally:
        engine.shutdown()


def _read_store(tmp_path):
    uri = (tmp_path / "lcm.db").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM messages ORDER BY store_id")]


def _legacy_store(tmp_path, monkeypatch, history):
    # v0.29.2 recovers only hermes-results, never cache/spillover.
    with monkeypatch.context() as patch:
        patch.setattr(ingest_persisted, "_safe_hermes_spillover_file", lambda *_: None)
        _run_phase(tmp_path, history)
    rows = _read_store(tmp_path)
    assert len(rows) == len(history)
    for message, row in zip(history, rows):
        assert row["content"] == message["content"]
        if message["role"] == "tool":
            assert row["tool_call_id"] == message["tool_call_id"]
            assert "[LCM persisted-output" not in row["content"]
    assert not list((tmp_path / "hermes" / "externalized").glob("*.json"))
    return rows


def _replay(tmp_path, history, original, caplog):
    with caplog.at_level(logging.INFO):
        reconciliation = _run_phase(tmp_path, history)
    rows = _read_store(tmp_path)
    assert rows[:len(original)] == original  # No old row changed or disappeared.
    assert "skipped stale no-overlap snapshot" not in caplog.text
    assert reconciliation.get("reason") != "skipped stale no-overlap snapshot"
    # Raw host content survives, including the original bare marker.
    contents = {row["content"] for row in rows}
    assert all(message["content"] in contents for message in history)
    return rows, reconciliation


@pytest.mark.parametrize("shape", ["stamped", "unstamped-reply", "file-gone"])
def test_upgrade_replay_adds_only_new_turn(tmp_path, monkeypatch, caplog, shape):
    path = tmp_path / "hermes" / "cache" / "spillover" / "call_spill.txt"
    marker = _write_output(path)
    reply = {"role": "assistant", "content": "Finished reading the spillover."}
    if shape != "unstamped-reply":
        reply["timestamp"] = 104.0
    pair = _messages(marker)
    if shape == "stamped":
        # Match probe.py's fully stamped public-host history, including the empty call content.
        pair[0]["content"] = ""
        pair[0]["tool_calls"][0]["type"] = "function"
        pair[0]["timestamp"] = 102.0
        pair[1]["timestamp"] = 103.0
    history = [
        {"role": "user", "content": "Read the synthetic output.", "timestamp": 100.0},
        *pair,
        reply,
    ]
    original = _legacy_store(tmp_path, monkeypatch, history)
    if shape == "file-gone":
        moved = tmp_path / "moved-out-of-home" / path.name
        moved.parent.mkdir()
        path.rename(moved)
    new_turn = [
        {"role": "user", "content": "The next request.", "timestamp": 105.0},
        {"role": "assistant", "content": "The next reply.", "timestamp": 106.0},
    ]

    rows, reconciliation = _replay(tmp_path, [*history, *new_turn], original, caplog)

    tool_count = sum(row["role"] == "tool" and row["tool_call_id"] == "call_spill" for row in rows)
    assert [row["content"] for row in rows[len(original):]] == [row["content"] for row in new_turn], (
        f"stored rows={len(rows)}, tool rows={tool_count}, reconciliation={reconciliation}"
    )
    assert tool_count == 1
    if shape != "file-gone":
        assert reconciliation.get("reason") != "persisted ambiguous delta"


def test_identical_markers_keep_distinct_call_ids(tmp_path, monkeypatch, caplog):
    marker = _write_output(tmp_path / "hermes" / "cache" / "spillover" / "shared.txt")
    history = [*_messages(marker, "call_a"), *_messages(marker, "call_b")]
    original = _legacy_store(tmp_path, monkeypatch, history)

    rows, _ = _replay(tmp_path, history, original, caplog)
    assert rows == original
    rows, _ = _replay(tmp_path, [*history, *_messages(marker, "call_c")], original, caplog)
    assert len(rows) == len(original) + 2
    assert Counter(row["tool_call_id"] for row in rows if row["role"] == "tool") == {
        "call_a": 1, "call_b": 1, "call_c": 1,
    }


@pytest.mark.parametrize("stored_copies", [1, 2])
def test_repeated_call_id_consumes_each_stored_copy_once(tmp_path, monkeypatch, caplog, stored_copies):
    marker = _write_output(tmp_path / "hermes" / "cache" / "spillover" / "shared.txt")
    pair = _messages(marker)
    original = _legacy_store(tmp_path, monkeypatch, pair * stored_copies)

    rows, _ = _replay(tmp_path, pair * 2, original, caplog)

    assert len(rows) == 4
    assert sum(row["role"] == "tool" for row in rows) == 2
