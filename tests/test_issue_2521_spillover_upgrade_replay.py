"""#2521: v0.29.2 bare spillover markers remain one-to-one replay identities (I1, I6)."""

import logging
import tempfile
from collections import Counter

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
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "hermes" / "externalized"),
        ),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine._session_id = SESSION
    engine._conversation_id = CONVERSATION
    return engine


def _legacy_store(tmp_path, monkeypatch, history):
    engine = _engine(tmp_path)
    try:
        # v0.29.2 recovers only hermes-results, never cache/spillover.
        with monkeypatch.context() as patch:
            patch.setattr(ingest_persisted, "_safe_hermes_spillover_file", lambda *_: None)
            engine._ingest_messages(history)
        rows = engine._store.get_session_messages(SESSION)
        assert len(rows) == len(history)
        for message, row in zip(history, rows):
            assert row["content"] == message["content"]
            if message["role"] == "tool":
                assert row["tool_call_id"] == message["tool_call_id"]
                assert "[LCM persisted-output" not in row["content"]
        assert not list((tmp_path / "hermes" / "externalized").glob("*.json"))
        return rows
    finally:
        engine.shutdown()


def _replay(tmp_path, history, original, caplog):
    engine = _engine(tmp_path)
    try:
        engine._ingest_cursor_needs_reconcile = True
        with caplog.at_level(logging.INFO):
            engine._ingest_messages(history)
        rows = engine._store.get_session_messages(SESSION)
        assert rows[:len(original)] == original  # No old row changed or disappeared.
        assert "skipped stale no-overlap snapshot" not in caplog.text
        assert engine._last_ingest_reconciliation.get("reason") != "skipped stale no-overlap snapshot"
        # Raw host content survives, including the original bare marker.
        contents = {row["content"] for row in rows}
        assert all(message["content"] in contents for message in history)
        return rows, dict(engine._last_ingest_reconciliation)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["stamped", "unstamped-reply", "file-gone"])
def test_upgrade_replay_adds_only_new_user(tmp_path, monkeypatch, caplog, shape):
    path = tmp_path / "hermes" / "cache" / "spillover" / "call_spill.txt"
    marker = _write_output(path)
    reply = {"role": "assistant", "content": "Finished reading the spillover."}
    if shape != "unstamped-reply":
        reply["timestamp"] = 104.0
    history = [
        {"role": "user", "content": "Read the synthetic output.", "timestamp": 100.0},
        *_messages(marker),
        reply,
    ]
    original = _legacy_store(tmp_path, monkeypatch, history)
    if shape == "file-gone":
        path.unlink()
    new_user = {"role": "user", "content": "The next request.", "timestamp": 105.0}

    rows, reconciliation = _replay(tmp_path, [*history, new_user], original, caplog)

    if shape == "file-gone":
        # Pin main's existing no-live-file path, including its conservative duplicate pair.
        expected = [history[1]["content"], marker, new_user["content"]]
        tool_count = 2
        assert reconciliation.get("reason") == "persisted ambiguous delta"
    else:
        expected = [new_user["content"]]
        tool_count = 1
        assert reconciliation.get("reason") != "persisted ambiguous delta"
    assert [row["content"] for row in rows[len(original):]] == expected
    assert sum(row["role"] == "tool" and row["tool_call_id"] == "call_spill" for row in rows) == tool_count


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
