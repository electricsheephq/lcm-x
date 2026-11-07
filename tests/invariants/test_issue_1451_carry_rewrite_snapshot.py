"""#1451 item 4: carry maintenance preserves occurrence evidence (I1, I2, I5)."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda *a, **k: (
        "Earlier turns retained. Expand for details about: turns", 1))
    engines = []

    def make(session):
        engine = engine_module.LCMEngine(config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"), fresh_tail_count=6, leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "ext")),
            hermes_home=str(tmp_path / "home"))
        engine.on_session_start(session, platform="acp", context_length=200000)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.shutdown()


def history():
    messages = [row for i in range(10) for row in (
        dict(role="user", content=f"user {i}: " + "alpha beta gamma delta " * 40, timestamp=100. + 2 * i),
        dict(role="assistant", content=f"reply {i}", timestamp=101. + 2 * i))]
    return messages + [dict(role="user", content="pending request", timestamp=200., message_uid="pending-old")]


def publish(engine, messages):
    engine.ingest(deepcopy(messages))
    output = engine.compress(deepcopy(messages), force=True)
    assert engine._last_compression_status == "compacted"
    assert engine._ingest_failure_count == 0
    return output


def pair(engine):
    return [engine._store.read_metadata_json(engine._replay_snapshot_metadata_key(prefix))
            for prefix in ("compaction_commit_proof", "host_occurrence_snapshot")]


@pytest.mark.parametrize("changed", ["timestamp", "message_uid"])
def test_sidecar_rewrite_equal_retry_is_stored(host, changed):
    engine = host("P")
    messages = history()
    output = publish(engine, messages)
    engine.on_session_end("P", deepcopy(messages))
    engine.on_session_start("C", boundary_reason="compression", old_session_id="P", platform="acp")
    before, _ = pair(engine)
    assert before["carry_ranges"]
    shortened = deepcopy(output[:-1])
    engine.ingest(shortened)  # Ordinary undo; no injected fault.
    after, _ = pair(engine)
    assert after["carry_ranges"] != before["carry_ranges"], "carry rewrite must be exercised"
    for field in ("session_id", "created_at", "effective_sha256_v4"):
        assert after[field] == before[field]
    engine.shutdown()
    engine = host("C")
    retry = deepcopy(messages[-1])
    retry[changed] = 1200. if changed == "timestamp" else "pending-new"
    engine.ingest(shortened + [retry])
    counts = Counter(engine._store._conn.execute("SELECT role, content FROM messages ORDER BY store_id"))
    assert engine._ingest_failure_count == 0
    assert counts[("user", "pending request")] == 2


@pytest.mark.parametrize("during_read", [False, True])
@pytest.mark.parametrize("same_output", [False, True])
def test_replaced_compaction_never_receives_old_snapshot(host, monkeypatch, during_read, same_output):
    engine = host("P")
    publish(engine, history())
    old_proof, old_snapshot = pair(engine)
    # A second real compaction publishes a distinct proof, even with equal tail bytes.
    messages = history() + [row for i in range(10, 20) for row in (
        dict(role="user", content=f"next {i}: " + "alpha beta gamma delta " * 40, timestamp=300. + 2 * i),
        dict(role="assistant", content=f"next reply {i}", timestamp=301. + 2 * i))]
    publish(engine, messages)
    new_proof, new_snapshot = pair(engine)
    assert new_proof["created_at"] != old_proof["created_at"]
    if same_output:
        # A later publication of the same output isolates the created_at guard:
        # shape validation alone cannot distinguish these two proofs.
        old_proof, old_snapshot = new_proof, new_snapshot
        engine._persist_compress_commit_proof(engine._compress_commit_proof)
        new_proof, new_snapshot = pair(engine)
        assert new_proof["created_at"] != old_proof["created_at"]
        assert new_proof["effective_sha256_v4"] == old_proof["effective_sha256_v4"]
        assert engine._occurrence_snapshot_shape_valid(new_snapshot, old_proof)
    proof_key = engine._replay_snapshot_metadata_key("compaction_commit_proof")
    snapshot_key = engine._replay_snapshot_metadata_key("host_occurrence_snapshot")
    read = engine._store.read_metadata_json
    reads = []

    def racing_read(key):
        if key == proof_key:
            reads.append(key)
            return deepcopy(old_proof if len(reads) == 1 else new_proof)
        if key == snapshot_key:
            return deepcopy(new_snapshot)
        return read(key)

    if during_read:
        monkeypatch.setattr(engine._store, "read_metadata_json", racing_read)
    else:
        engine._store.write_metadata_json([snapshot_key], json.dumps(old_snapshot))
        engine._rewrite_own_carry_ranges([], "stale snapshot must stay unbound", [])
        assert pair(engine)[1] == old_snapshot
    loaded = engine._durable_commit_proof_payload()
    assert loaded is not None
    assert "output_metadata" not in loaded
    if during_read:
        assert len(reads) == 2, "replacement must occur between proof reads"


def test_old_format_snapshot_still_attaches_when_unchanged(host):
    engine = host("P")
    publish(engine, history())
    proof, snapshot = pair(engine)
    legacy = {key: snapshot[key] for key in ("output_metadata", "output_length")}
    legacy["proof_sha256"] = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
    engine._store.write_metadata_json(
        [engine._replay_snapshot_metadata_key("host_occurrence_snapshot")], json.dumps(legacy))
    engine.shutdown()
    engine = host("P")
    assert engine._durable_commit_proof_payload()["output_metadata"] == legacy["output_metadata"]


def test_carry_rewrite_refreshes_existing_digest(host):
    engine = host("P")
    publish(engine, history())
    before, snapshot = pair(engine)
    engine._rewrite_own_carry_ranges([("P", 0, 1)], "binding regression", [])
    after, refreshed = pair(engine)
    assert before["carry_ranges"] != after["carry_ranges"]
    expected = deepcopy(snapshot)
    expected["proof_sha256"] = hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()
    assert refreshed == expected
    assert engine._durable_commit_proof_payload()["output_metadata"] == snapshot["output_metadata"]


@pytest.mark.parametrize("boundary", ["compaction_commit_proof", "host_occurrence_snapshot"])
def test_failed_carry_rewrite_rolls_back_both_records(host, monkeypatch, boundary):
    engine = host("P")
    publish(engine, history())
    before = pair(engine)
    target = engine._replay_snapshot_metadata_key(boundary)
    write = engine._store.write_metadata_json
    fired = []

    def fail(keys, *args, **kwargs):
        result = write(keys, *args, **kwargs)
        if target in keys:
            fired.append(target)
            raise sqlite3.OperationalError("injected carry rewrite failure")
        return result

    monkeypatch.setattr(engine._store, "write_metadata_json", fail)
    engine._rewrite_own_carry_ranges([("P", 0, 1)], "rollback regression", [])
    assert fired == [target]
    assert pair(engine) == before
    assert not engine._store._conn.in_transaction
