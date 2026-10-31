"""Real backfill-path regressions for #1650's bounded bot fix round."""
import dataclasses
import hashlib
import json
import logging
import sqlite3
from collections import Counter

import pytest

from hermes_lcm import command, embedding_maintenance as maintenance
from tests.test_issue_1586_local_pacing import local_store as local_store


def _run(engine, *, limit=32, policy="full", incremental=True):
    result = command._chunk_backfill_run(
        engine._config, engine._store.db_path, apply=True, limit=limit,
        retry_uncertain=False, policy=policy, incremental=incremental,
    )
    assert isinstance(result, dict), result
    assert not result["error"], result
    return result


def _key(engine):
    with sqlite3.connect(engine._store.db_path) as conn:
        identity = conn.execute(
            "SELECT identity_hash FROM lcm_embedding_profile WHERE task='chunk' AND active=1"
        ).fetchone()[0]
    return f"chunk_partial_messages:{identity}"


def _saved(engine):
    with sqlite3.connect(engine._store.db_path) as conn:
        row = conn.execute("SELECT value FROM metadata WHERE key=?", (_key(engine),)).fetchone()
    return row[0] if row else "[]"


def _ids(engine):
    record = json.loads(_saved(engine))
    return record["ids"] if isinstance(record, dict) else record


def _save(engine, value):
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", (_key(engine), value))


def _published(engine, sid):
    with sqlite3.connect(engine._store.db_path) as conn:
        return dict(conn.execute("SELECT chunk_id, vec FROM lcm_chunk_vectors JOIN lcm_chunk_meta USING(chunk_id) "
                                 "WHERE store_id=?", (sid,)))


def test_recorded_tool_uses_active_conversational_policy(local_store):
    """A is false: a prior full-policy intent cannot bypass today's policy."""
    engine, _ = local_store
    sid = engine._store.append("tool-session", {"role": "tool", "content": "tool result sentence. " * 250})
    first = _run(engine, limit=1, incremental=False)
    assert first["embedded"] == 1 and sid in _ids(engine)
    before = _published(engine, sid)
    second = _run(engine, policy="conversational")
    assert second["embedded"] == 0
    assert _published(engine, sid) == before
    assert sid not in _ids(engine) and not second["remaining"]


def test_deleted_session_drops_full_policy_intent(local_store):
    engine, _ = local_store
    sid = engine._store.append("deleted-session", {"role": "user", "content": "partial sentence. " * 250})
    assert _run(engine, limit=1, incremental=False)["embedded"] == 1
    assert sid in _ids(engine)
    assert engine._store.delete_session_messages("deleted-session") == 1
    result = _run(engine)
    assert sid not in _ids(engine), result
    assert not result["remaining"] and result["status"] == "complete", result


def test_completed_record_window_is_bounded_and_converges(local_store, monkeypatch):
    engine, _ = local_store
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.executemany("INSERT INTO messages(session_id,role,content,timestamp) VALUES(?,?,?,?)",
                         [("seed", "user", f"complete message {i}", float(i)) for i in range(2000)])
    assert _run(engine, limit=10000, incremental=False)["embedded"] == 2000
    with sqlite3.connect(engine._store.db_path) as conn:
        ids = [row[0] for row in conn.execute("SELECT store_id FROM messages ORDER BY store_id")]
    _save(engine, json.dumps(ids))  # Exit after publication, before the final prune.
    real = command.chunk_message
    calls = Counter()

    def count(sid, *args, **kwargs):
        calls[sid] += 1
        return real(sid, *args, **kwargs)

    monkeypatch.setattr(command, "chunk_message", count)
    window = 4 * (32 // 2)
    for _ in range((len(ids) + window - 3) // (window - 2)):
        calls.clear()
        result = _run(engine)
        assert sum(calls.values()) <= window, (len(calls), result)
        assert not calls or max(calls.values()) == 1
        assert result["embedded"] == 0
        if not _ids(engine):
            break
        assert result["remaining"] and result["status"] != "complete"
    assert _ids(engine) == []
    assert not result["remaining"] and result["status"] == "complete", result


@pytest.mark.parametrize("policy", ["full", "conversational"])
@pytest.mark.parametrize("automatic", [False, True])
def test_limit_one_alternates_partial_and_new(local_store, policy, automatic):
    engine, _ = local_store
    partial = engine._store.append("seed", {"role": "user", "content": "large partial sentence. " * 500})
    assert _run(engine, limit=1, policy=policy, incremental=False)["embedded"] == 1
    before = _published(engine, partial)
    fresh = engine._store.append("seed", {"role": "user", "content": "fresh turn"})
    config = dataclasses.replace(engine._config, embedding_max_batch_items=1, embedding_content_policy=policy)
    for _ in range(2):
        if automatic:
            maintenance.run_incremental_embedding_pass(engine._store.db_path, config)
        else:
            _run(engine, limit=1, policy=policy)
    assert _published(engine, fresh), "new work starved behind the partial at limit=1"
    after = _published(engine, partial)
    assert len(after) > len(before), "partial work must also progress"
    assert all(after[cid] == vector for cid, vector in before.items())
    assert partial in _ids(engine)


@pytest.mark.parametrize("policy", ["full", "conversational"])
def test_manual_run_prunes_completed_intent(local_store, policy):
    engine, _ = local_store
    sid = engine._store.append("seed", {"role": "user", "content": "already complete sentence. " * 250})
    assert _run(engine, limit=10000, policy=policy, incremental=False)["embedded"] > 1
    before = _published(engine, sid)
    _save(engine, json.dumps([sid]))
    result = _run(engine, policy=policy, incremental=False)
    assert _ids(engine) == [], result
    assert not result["remaining"] and result["status"] == "complete", result
    assert result["embedded"] == 0 and _published(engine, sid) == before


@pytest.mark.parametrize("value", ["not json", "null", '[1, "bad"]'])
@pytest.mark.parametrize("incremental", [False, True])
def test_malformed_full_record_is_preserved_and_disclosed(local_store, caplog, monkeypatch, value, incremental):
    engine, _ = local_store
    caplog.set_level(logging.DEBUG, logger="hermes_lcm")
    monkeypatch.setattr(command, "_CHUNK_PARTIAL_WARNED", set())
    _save(engine, value)
    for _ in range(2):
        result = _run(engine, incremental=incremental)
        if incremental:
            assert _saved(engine) == value, result
            assert result["remaining"] and result["status"] != "complete", result
        else:
            assert _ids(engine) == [] and "repair_pending" not in _saved(engine), result
            assert not result["remaining"] and result["status"] == "complete", result
    assert any("partial-message record" in record.message and record.levelno == logging.WARNING
               for record in caplog.records)


@pytest.mark.parametrize("value", ["not json", "null", '[1, "bad"]'])
def test_operator_repairs_malformed_record_before_new_intents(local_store, value):
    engine, _ = local_store
    first = engine._store.append("seed", {"role": "user", "content": "unembedded first message"})
    _save(engine, value)
    backup_key = _key(engine).replace("chunk_partial_messages:", "chunk_partial_messages_malformed:")
    before = {}
    for original in (value, "another malformed value"):
        _save(engine, original)
        fresh = engine._store.append("seed", {"role": "user", "content": "unembedded new message"})
        with sqlite3.connect(engine._store.db_path) as conn:
            messages = conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
        result = _run(engine, incremental=False)
        assert _published(engine, first) and _published(engine, fresh)
        assert all(_published(engine, first)[cid] == vec for cid, vec in before.items())
        before = _published(engine, first)
        assert _ids(engine) == [] and not result["remaining"] and result["status"] == "complete"
        with sqlite3.connect(engine._store.db_path) as conn:
            for preserved in (value, original):
                digest = hashlib.sha256(preserved.encode()).hexdigest()[:16]
                assert conn.execute("SELECT value FROM metadata WHERE key=?",
                                    (f"{backup_key}:{digest}",)).fetchone()[0] == preserved
            assert conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall() == messages


@pytest.mark.parametrize("value", ["not json", "null", '[1, "bad"]'])
def test_incremental_malformed_record_refuses_new_intents_and_warns_once(local_store, caplog, monkeypatch, value):
    engine, _ = local_store
    caplog.set_level(logging.WARNING, logger="hermes_lcm")
    monkeypatch.setattr(command, "_CHUNK_PARTIAL_WARNED", set())
    fresh = engine._store.append("seed", {"role": "user", "content": "unembedded new message"})
    _save(engine, value)
    dispatches = []
    monkeypatch.setattr(command, "resolve_provider", lambda *_a, **_k: dispatches.append(True))
    for _ in range(2):
        result = command._chunk_backfill_run(
            engine._config, engine._store.db_path, apply=True, limit=32,
            retry_uncertain=False, policy="full", incremental=True,
        )
        assert result["error"] == "malformed partial-message record requires repair before dispatch"
        assert not result["embedded"] and result["remaining"] and result["status"] != "complete"
        assert _saved(engine) == value and not _published(engine, fresh)
    assert not dispatches
    warnings = [r for r in caplog.records if "partial-message record" in r.message]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "/lcm embed backfill --corpus chunks --apply" in warnings[0].message


@pytest.mark.parametrize("changed", ["owner", "generation"])
def test_operator_repair_obeys_lease_fence(local_store, monkeypatch, changed):
    engine, _ = local_store
    fresh = engine._store.append("seed", {"role": "user", "content": "unembedded message"})
    _save(engine, "not json")
    prepare = command._prepare_inflight_for_lease

    def lose_lease(conn, identity, lease):
        prepare(conn, identity, lease)
        row = conn.execute("SELECT value FROM metadata WHERE key=?",
                           (command._EMBEDDING_BACKFILL_CLAIM_KEY,)).fetchone()
        successor = json.loads(row[0])
        successor[changed] = "successor" if changed == "owner" else lease.generation + 1
        conn.execute("UPDATE metadata SET value=? WHERE key=?",
                     (json.dumps(successor), command._EMBEDDING_BACKFILL_CLAIM_KEY))

    monkeypatch.setattr(command, "_prepare_inflight_for_lease", lose_lease)
    result = command._chunk_backfill_run(
        engine._config, engine._store.db_path, apply=True, limit=32,
        retry_uncertain=False, policy="full", incremental=False,
    )
    assert result["error"] and result["remaining"] and not result["embedded"]
    assert _saved(engine) == "not json" and not _published(engine, fresh)
    with sqlite3.connect(engine._store.db_path) as conn:
        assert not conn.execute("SELECT 1 FROM metadata WHERE key=?",
                                (_key(engine).replace("messages:", "messages_malformed:"),)).fetchone()


@pytest.mark.parametrize("value", [' [1, 2] ', '{"ids": [1], "cursor": 0, "turn": 0}'])
def test_repair_never_touches_valid_record(local_store, value):
    engine, _ = local_store
    _save(engine, value)
    with sqlite3.connect(engine._store.db_path) as conn:
        lease = command._acquire_embedding_backfill_lease(conn, ttl_s=60, heartbeat_s=60)
        before = conn.total_changes
        command._repair_chunk_partial_messages(conn, _key(engine).split(":", 1)[1], lease)
        assert conn.total_changes == before
        assert _saved(engine) == value
        lease.release()
