"""Occurrence snapshots publish with their proof and malformed evidence is unavailable."""

import copy
import hashlib
import json
import logging
import sqlite3

import pytest

from hermes_lcm import engine as engine_module
from tests.test_host_uid_engine_uids import _compress, _host
from tests.test_host_uid_shadow import _engine, _m, _state_db


class FaultConnection(sqlite3.Connection):
    fail_commit = False

    def commit(self):
        if self.fail_commit:
            self.fail_commit = False
            raise sqlite3.OperationalError("injected occurrence commit failure")
        return super().commit()


@pytest.fixture
def published(tmp_path, monkeypatch):
    original = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: original(*a, **kw, factory=FaultConnection))
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 4
    host = _host()
    output = _compress(engine, host)
    assert engine._compress_commit_proof is not None
    yield engine, host, output
    engine.shutdown()


def keys(engine):
    return [engine._replay_snapshot_metadata_key(prefix)
            for prefix in ("compaction_commit_proof", "host_occurrence_snapshot")]


def pair(engine):
    return [engine._store.read_metadata_json(key) for key in keys(engine)]


def test_one_atomic_publication_has_no_proof_only_commit(published, monkeypatch):
    engine, host, output = published
    original = engine._store.write_metadata_json
    observed = []
    trace = []
    engine._store._conn.set_trace_callback(trace.append)

    def write(*args, **kwargs):
        result = original(*args, **kwargs)
        with sqlite3.connect(engine._store.db_path) as reader:
            observed.append([reader.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0]
                             for key in keys(engine)])
        return result

    before = [json.dumps(value, sort_keys=True) for value in pair(engine)]
    monkeypatch.setattr(engine._store, "write_metadata_json", write)
    engine._record_compress_commit_proof(host, output)
    assert observed == [before, before]
    assert [sql for sql in trace if sql in ("BEGIN IMMEDIATE", "COMMIT")] == ["BEGIN IMMEDIATE", "COMMIT"]
    proof, snapshot = pair(engine)
    assert proof["version"] == 3 and proof["descriptor_version"] == 4
    assert snapshot["proof_sha256"] == hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("boundary", ["proof", "snapshot", "commit"])
def test_failed_publication_rolls_back_and_keeps_positional_proof(published, monkeypatch, caplog, boundary):
    engine, host, output = published
    before = pair(engine)
    original = engine._store.write_metadata_json
    target = keys(engine)[boundary == "snapshot"]

    def write(write_keys, *args, **kwargs):
        result = original(write_keys, *args, **kwargs)
        if boundary != "commit" and target in write_keys:
            raise sqlite3.OperationalError("injected occurrence write failure")
        return result

    monkeypatch.setattr(engine._store, "write_metadata_json", write)
    engine._store._conn.fail_commit = boundary == "commit"
    states_at_warning = []

    class WatchRollback(logging.Handler):
        def emit(self, record):
            if "occurrence" in record.getMessage().lower():
                states_at_warning.append(engine._store._conn.in_transaction)

    watcher = WatchRollback(level=logging.WARNING)
    logging.getLogger().addHandler(watcher)
    try:
        with caplog.at_level(logging.WARNING):
            engine._record_compress_commit_proof(host, output)
        assert pair(engine) == before
        assert engine._compress_commit_proof["durable"] is False
        assert "output_metadata" in engine._compress_commit_proof
        assert engine._durable_commit_proof_payload() is None
        assert states_at_warning == [False]
    finally:
        logging.getLogger().removeHandler(watcher)


MALFORMED = ("null", "scalar", "non-dict", "short", "long", "timestamp", "uid", "digest", "missing-field", "bool-timestamp", "bool-uid", "bool-digest")


def malformed(kind, valid):
    rows = copy.deepcopy(valid)
    if kind == "null":
        return None
    if kind == "scalar":
        return 42
    if kind == "non-dict":
        return [None] * len(rows)
    if kind == "short":
        return rows[:-1]
    if kind == "long":
        return rows + [copy.deepcopy(rows[-1])]
    if kind.startswith("bool-"):
        rows[-1][{"bool-timestamp": "timestamp", "bool-uid": "message_uid",
                  "bool-digest": "identity_sha256"}[kind]] = True
        return rows
    if kind == "missing-field":
        rows[-1].pop("timestamp")
    else:
        field = {"timestamp": "timestamp", "uid": "message_uid", "digest": "identity_sha256"}[kind]
        rows[-1][field] = {"unexpected": "value"}
    return rows


@pytest.mark.parametrize("kind", MALFORMED)
def test_malformed_snapshot_matches_missing_on_cold_ingest(tmp_path, kind):
    results = []
    for variant in ("missing", kind):
        home = tmp_path / variant
        home.mkdir()
        _state_db(home, [("S", None, None)])
        engine = _engine(home)
        engine._config.fresh_tail_count = 4
        output = _compress(engine, _host())
        snapshot_key = keys(engine)[1]
        snapshot = pair(engine)[1]
        if variant == "missing":
            engine._store.write_metadata_json([snapshot_key], "null")
        else:
            snapshot["output_metadata"] = malformed(kind, snapshot["output_metadata"])
            engine._store.write_metadata_json([snapshot_key], json.dumps(snapshot))
        engine.shutdown()
        engine = _engine(home)
        try:
            payload = engine._durable_commit_proof_payload()
            # Equality to missing evidence is stronger than merely swallowing an ingest exception.
            fresh = _m("user", "fresh synthetic row after malformed snapshot", 999.0, "fresh-uid")
            engine.ingest(copy.deepcopy(output) + [fresh])
            rows = engine._store.get_session_messages("S")
            assert any(row["content"] == fresh["content"] for row in rows)
            assert "output_metadata" not in payload
            results.append([(row["role"], row["content"], row["observed_at"]) for row in rows])
        finally:
            engine.shutdown()
    assert results[0] == results[1]


@pytest.mark.parametrize("kind", MALFORMED)
def test_invalid_writer_snapshot_never_gets_a_matching_digest(published, kind):
    engine, _host_input, _output = published
    before = pair(engine)
    proof = engine._compress_commit_proof
    invalid = malformed(kind, proof["output_metadata"])
    proof["output_metadata"] = invalid
    engine._persist_compress_commit_proof(proof)
    assert pair(engine) == before
    assert engine._compress_commit_proof["durable"] is False
    assert engine._compress_commit_proof["output_metadata"] == invalid
    assert engine._durable_commit_proof_payload() is None
    # Rejected metadata remains process-local but cannot crash positional ingest.
    engine.ingest(copy.deepcopy(_output))
    assert engine._ingest_failure_count == 0
    engine._persist_compress_commit_proof(proof)
    assert proof["durable"] is True
    assert pair(engine)[1] is None
    assert "output_metadata" not in engine._durable_commit_proof_payload()


@pytest.mark.parametrize("mutation", ["valid", "short", "long", "extra-scaffold", "bool-length"])
def test_legacy_snapshot_validates_effective_sequence_and_scaffolds(published, mutation):
    engine, _host_input, _output = published
    snapshot_key = keys(engine)[1]
    snapshot = pair(engine)[1]
    snapshot.pop("output_length", None)
    rows = snapshot["output_metadata"]
    if mutation in ("short", "long"):
        snapshot["output_metadata"] = malformed(mutation, rows)
    elif mutation == "extra-scaffold":
        rows.append({"timestamp": None, "message_uid": None, "identity_sha256": None})
    elif mutation == "bool-length":
        snapshot["output_length"] = True
    engine._store.write_metadata_json([snapshot_key], json.dumps(snapshot))
    loaded = engine._durable_commit_proof_payload()
    assert ("output_metadata" in loaded) == (mutation == "valid")


def test_legacy_snapshot_with_leading_system_anchor_attaches(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **kwargs: ("Synthetic summary preserving the task and tool result.", 1))
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 4
    try:
        output = _compress(engine, _host(system=True))
        assert engine._last_compression_status == "compacted"
        assert output[0]["role"] == "system"
        proof, snapshot = pair(engine)
        snapshot.pop("output_length")  # The only change needed to reproduce the old writer's snapshot.
        rows = snapshot["output_metadata"]
        assert {i for i, row in enumerate(rows) if row["identity_sha256"] is None} == {0, 1}
        assert {item["output_occurrence"]["index"] for item in proof["emissions"]
                if item.get("retained_source") is None} == {1}
        engine._store.write_metadata_json([keys(engine)[1]], json.dumps(snapshot))
        loaded = engine._durable_commit_proof_payload()
        assert loaded.get("output_metadata") == rows
    finally:
        engine.shutdown()


def test_boundary_can_republish_positional_proof_without_snapshot(published, monkeypatch, tmp_path):
    engine, host, output = published
    original = engine._store.write_metadata_json
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("injected failure")
    monkeypatch.setattr(engine._store, "write_metadata_json", fail)
    engine._record_compress_commit_proof(host, output)
    proof = engine._compress_commit_proof
    assert proof["durable"] is False and "output_metadata" in proof
    monkeypatch.setattr(engine._store, "write_metadata_json", original)
    engine._persist_compress_commit_proof(proof)
    assert engine._compress_commit_proof is proof and proof["durable"] is True
    assert "output_metadata" not in engine._durable_commit_proof_payload()
    assert pair(engine)[1] is None
    engine.shutdown()
    cold = _engine(tmp_path)
    try:
        assert "output_metadata" not in cold._durable_commit_proof_payload()
        before = len(cold._store.get_session_messages("S"))
        cold.ingest(copy.deepcopy(output) + [_m("user", "fresh row after retry", 999.0, "retry-fresh")])
        rows = cold._store.get_session_messages("S")
        assert len(rows) == before + 1
        assert rows[-1]["content"] == "fresh row after retry"
    finally:
        cold.shutdown()
