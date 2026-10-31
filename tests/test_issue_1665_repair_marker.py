"""Malformed repair markers preserve unknown coverage until operator repair (#1665)."""
import json
import logging
import sqlite3

import pytest

from hermes_lcm import command
from tests.test_issue_1586_local_pacing import local_store as local_store
from tests.test_issue_1650_bot_round import _run, _save, _saved
from tests.test_issue_1661_pending_repair import _backup_key, _backups, _seed_hole


@pytest.mark.parametrize("backlog", [True, False])
@pytest.mark.parametrize("marker", [["x"], {"a": 1}, 7, "", [], None, "ABCDEF0123456789", "abc"])
def test_malformed_marker_is_preserved_and_warned_once(monkeypatch, caplog, marker, backlog):
    identity = "marker-test"
    value = " " + json.dumps(dict(ids=[1], cursor=0, turn=0, repair_pending=marker)) + "\n"
    monkeypatch.setattr(command, "_CHUNK_PARTIAL_WARNED", set())
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO metadata VALUES (?, ?)", (f"chunk_partial_messages:{identity}", value))
        before = conn.total_changes
        for _ in range(2):
            assert command._chunk_partial_ids(conn, identity, backlog=backlog) is None
            assert conn.execute("SELECT value FROM metadata").fetchone()[0] == value
        assert conn.total_changes == before
    warnings = [r for r in caplog.records if "partial-message record" in r.message]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "/lcm embed backfill --corpus chunks --apply" in warnings[0].message
    assert command._CHUNK_PARTIAL_WARNED == {(identity, value)}


@pytest.mark.parametrize("backlog", [True, False])
@pytest.mark.parametrize("marked", [True, False])
def test_valid_or_absent_marker_keeps_existing_behavior(monkeypatch, caplog, backlog, marked):
    identity, marker = "marker-control", "abcdef0123456789"
    record = dict(ids=[1, 2], cursor=0, turn=0)
    if marked:
        record["repair_pending"] = marker
    value = json.dumps(record)
    monkeypatch.setattr(command, "_CHUNK_PARTIAL_WARNED", set())
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO metadata VALUES (?, ?)", (f"chunk_partial_messages:{identity}", value))
        for _ in range(2):
            assert command._chunk_partial_ids(conn, identity, backlog=backlog) == (
                None if marked and backlog else {1, 2}
            )
        assert conn.execute("SELECT value FROM metadata").fetchone()[0] == value
    assert len([r for r in caplog.records if "partial-message record" in r.message]) == int(marked and backlog)
    assert command._CHUNK_PARTIAL_WARNED == ({(identity, marker)} if marked and backlog else set())


def test_operator_repairs_malformed_marker_with_versioned_backup(local_store):
    engine, _ = local_store
    messages = _seed_hole(engine)
    value = ' {"ids": [], "cursor": 0, "turn": 0, "repair_pending": ["x"]} '
    _save(engine, value)
    result = _run(engine, limit=1, incremental=False)
    marker = json.loads(_saved(engine))["repair_pending"]
    assert marker == _backup_key(engine, value).rsplit(":", 1)[1]
    assert len(marker) == 16 and all(c in "0123456789abcdef" for c in marker)
    assert _backups(engine) == {_backup_key(engine, value): value}
    assert result["remaining"] >= 1 and result["status"] != "complete"
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall() == messages
