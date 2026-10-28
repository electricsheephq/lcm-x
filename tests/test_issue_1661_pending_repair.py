"""Malformed intent coverage stays unknown until an exhaustive repair completes."""
import hashlib
import json
import logging
import sqlite3
import threading

import pytest

from hermes_lcm import command
from tests.test_issue_1586_local_pacing import local_store as local_store
from tests.test_issue_1650_bot_round import _ids, _key, _published, _run, _save, _saved


def _backup_key(engine, value):
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    return _key(engine).replace("messages:", "messages_malformed:") + ":" + digest


def _backups(engine):
    with sqlite3.connect(engine._store.db_path) as conn:
        return dict(conn.execute("SELECT key, value FROM metadata WHERE key LIKE 'chunk_partial_messages_malformed:%'"))


def _seed_hole(engine):
    for i in range(3):
        engine._store.append("seed", {"role": "user", "content": f"message {i} sentence. " * 250})
    assert _run(engine, limit=10000, incremental=False)["status"] == "complete"
    with sqlite3.connect(engine._store.db_path) as conn:
        messages = conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall()
        middle = messages[1][0]
        conn.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id IN "
                     "(SELECT chunk_id FROM lcm_chunk_meta WHERE store_id=? AND chunk_index>0)", (middle,))
        conn.execute("DELETE FROM lcm_chunk_meta WHERE store_id=? AND chunk_index>0", (middle,))
    assert len(_published(engine, middle)) == 1
    return messages


def test_retry_uncertain_preserves_malformed_without_backup(local_store):
    engine, _ = local_store
    value = ' [1, "unknown coverage"] '
    _save(engine, value)
    command.handle_lcm_command("embed backfill --corpus chunks --apply --retry-uncertain", engine)
    assert _saved(engine) == value
    assert _backups(engine) == {}


def test_distinct_malformed_values_have_distinct_deduplicated_backups(local_store):
    engine, _ = local_store
    values = ("not json", ' [2, "different"] ')
    for value in (*values, values[0]):
        _save(engine, value)
        assert _run(engine, incremental=False)["status"] == "complete"
    assert _backups(engine) == {_backup_key(engine, value): value for value in values}


@pytest.mark.parametrize("interrupt", ["stop", "lease"])
def test_interrupted_repair_stays_unknown_through_automatic_passes(local_store, monkeypatch, caplog, interrupt):
    engine, _ = local_store
    messages = _seed_hole(engine)
    value = "malformed coverage"
    _save(engine, value)
    stop = threading.Event()
    repair = command._repair_chunk_partial_messages

    def interrupted(conn, identity, lease):
        repair(conn, identity, lease)
        if interrupt == "lease":
            successor = json.loads(conn.execute("SELECT value FROM metadata WHERE key=?",
                                               (command._EMBEDDING_BACKFILL_CLAIM_KEY,)).fetchone()[0])
            successor["generation"] += 1
            successor["heartbeat_at"] = 0
            conn.execute("UPDATE metadata SET value=? WHERE key=?",
                         (json.dumps(successor), command._EMBEDDING_BACKFILL_CLAIM_KEY))
            raise command._BackfillLeaseLost("scripted lease loss after repair")
        stop.set()

    monkeypatch.setattr(command, "_repair_chunk_partial_messages", interrupted)
    result = command._chunk_backfill_run(engine._config, engine._store.db_path, apply=True, limit=10000,
                                         retry_uncertain=False, policy="full", stop_event=stop)
    assert result["stop_reason"] == ("lease_lost" if interrupt == "lease" else "process_exit")
    marker = json.loads(_saved(engine))["repair_pending"]
    assert marker == _backup_key(engine, value).rsplit(":", 1)[1]
    monkeypatch.setattr(command, "_CHUNK_PARTIAL_WARNED", set())
    caplog.clear()
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    for _ in range(3):
        automatic = _run(engine)
        assert automatic["remaining"] >= 1 and automatic["status"] != "complete"
        assert json.loads(_saved(engine))["repair_pending"] == marker
    with sqlite3.connect(engine._store.db_path) as conn:
        assert command._chunk_backlog_estimate(conn, _key(engine).split(":", 1)[1], "full") == "1+"
        assert conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall() == messages
    assert _backups(engine) == {_backup_key(engine, value): value}
    warnings = [r for r in caplog.records if "partial-message record" in r.message]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "/lcm embed backfill --corpus chunks --apply" in warnings[0].message


def test_exhaustive_completion_clears_marker_and_embeds_every_chunk(local_store, monkeypatch):
    engine, _ = local_store
    messages = _seed_hole(engine)
    value = "malformed coverage"
    _save(engine, value)
    repaired = []
    repair = command._repair_chunk_partial_messages

    def observe(conn, identity, lease):
        repair(conn, identity, lease)
        repaired.append(json.loads(_saved(engine)))

    monkeypatch.setattr(command, "_repair_chunk_partial_messages", observe)
    result = _run(engine, limit=10000, incremental=False)
    assert isinstance(repaired[0], dict) and repaired[0]["repair_pending"]
    assert "repair_pending" not in json.loads(_saved(engine)) and _ids(engine) == []
    assert result["remaining"] == 0 and result["status"] == "complete"
    for row in messages:
        assert set(_published(engine, row[0])) == {c.chunk_id for c in command.chunk_message(*row, policy="full")}
    assert _backups(engine) == {_backup_key(engine, value): value}


def test_limit_bounded_repair_keeps_pending_marker(local_store, caplog, monkeypatch):
    engine, _ = local_store
    _seed_hole(engine)
    _save(engine, "malformed coverage")
    result = _run(engine, limit=1, incremental=False)
    assert json.loads(_saved(engine))["repair_pending"]
    assert result["remaining"] >= 1 and result["status"] != "complete"
    monkeypatch.setattr(command, "_CHUNK_PARTIAL_WARNED", set())
    caplog.clear()
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    for _ in range(3):
        assert _run(engine, limit=1)["status"] != "complete"
    assert len([r for r in caplog.records if "partial-message record" in r.message]) == 1
    assert _run(engine, limit=10000, incremental=False)["status"] == "complete"
    assert "repair_pending" not in json.loads(_saved(engine))
