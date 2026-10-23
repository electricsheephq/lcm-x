# ruff: noqa: F811
"""Diagnostic records observe the existing end/proof decisions without authority."""
import copy
import logging
import sqlite3

import pytest

from hermes_lcm import engine as engine_module
from tests.test_issue_1451_atomic_snapshot import published, pair, keys  # noqa: F401, F811


def records(caplog):
    return [r for r in caplog.records if r.msg.startswith("LCM attribution phase=session_end")]


def fields(record):
    return dict(item.split("=", 1) for item in record.getMessage().split() if "=" in item)


@pytest.mark.parametrize("reason", ["fast", "no_proof", "session_mismatch", "conversation_mismatch",
                                    "already_consumed", "input_unavailable", "length_mismatch", "identity_mismatch"])
def test_first_rejected_gate(published, monkeypatch, caplog, reason):
    engine, host, _output = published
    proof = engine._compress_commit_proof
    before = engine._store.get_session_count("S")
    if reason == "no_proof":
        engine._compress_commit_proof = None
    elif reason == "session_mismatch":
        proof["session_id"] = "other"
    elif reason == "conversation_mismatch":
        proof["conversation_id"] = "other"
    elif reason == "already_consumed":
        proof["end_consumed"] = True
    elif reason == "input_unavailable":
        proof["input"] = None
    elif reason == "length_mismatch":
        proof["input"] = []
    elif reason == "identity_mismatch":
        proof["input"][0] = ("different",)
    original = engine._ingest_messages
    calls = []
    def ingest(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "_ingest_messages", ingest)
    with caplog.at_level(logging.INFO):
        engine.on_session_end("S", host)
    assert len(records(caplog)) == 1
    data = fields(records(caplog)[0])
    assert data["reason"] == reason
    assert bool(calls) is (reason != "fast")
    assert engine._store.get_session_count("S") == before
    assert data["engine_token"] == str(engine._diagnostic_engine_token)
    assert data["compress_sequence"] == str(engine._compress_calls)
    assert all(float(data[key]) >= 0 for key in ("outer_ms", "admission_ms", "classifier_ms", "ingest_ms",
                                               "finalize_ms", "snapshot_ms", "retire_ms"))
    assert not any(secret in records(caplog)[0].getMessage() for secret in ("question", "answer", "conversation_id=", "sha256"))


@pytest.mark.parametrize("boundary", ["proof", "snapshot", "commit"])
def test_publication_failure_keeps_existing_positional_authority(published, monkeypatch, caplog, boundary):
    engine, host, output = published
    before = pair(engine)
    write = engine._store.write_metadata_json
    target = keys(engine)[boundary == "snapshot"]
    def fail(write_keys, *args, **kwargs):
        result = write(write_keys, *args, **kwargs)
        if boundary != "commit" and target in write_keys:
            raise sqlite3.OperationalError("synthetic failure")
        return result
    monkeypatch.setattr(engine._store, "write_metadata_json", fail)
    engine._store._conn.fail_commit = boundary == "commit"
    engine._record_compress_commit_proof(host, output)
    assert engine._compress_commit_proof["durable"] is False
    assert pair(engine) == before
    assert engine._durable_commit_proof_payload() is None
    with caplog.at_level(logging.INFO):
        engine.on_session_end("S", host)
    data = fields(records(caplog)[0])
    assert data["writer_reason"] == "publication_failed"
    assert data["proof_state"] == "local"
    assert data["reason"] == "fast"  # #1882 preserves local proof; diagnostics cannot add a durability gate.
    assert engine._store.get_session_count("S") == len(host)


@pytest.mark.parametrize("reason", ["no_session", "non_list_output", "reconcile_pending", "cursor_mismatch",
                                    "identity_noop", "construction_error", "ready"])
def test_writer_reasons(published, monkeypatch, reason):
    engine, host, output = published
    if reason == "no_session":
        engine._session_id = ""
    elif reason == "non_list_output":
        output = None
    elif reason == "reconcile_pending":
        engine._ingest_cursor_needs_reconcile = True
    elif reason == "cursor_mismatch":
        engine._ingest_cursor = -1
    elif reason == "identity_noop":
        host = copy.deepcopy(output)
    elif reason == "construction_error":
        monkeypatch.setattr(engine, "_emission_binding", lambda: 1 / 0)
    engine._record_compress_commit_proof(host, output)
    assert engine._compress_commit_reason == reason
    assert bool(engine._compress_commit_proof) is (reason == "ready")


def test_clone_and_reset_ownership(published):
    engine, _host, _output = published
    clone = engine.clone_for_agent()
    try:
        assert clone._diagnostic_engine_token != engine._diagnostic_engine_token
        assert clone._compress_commit_reason == "no_proof"
        assert clone._compress_calls == 0
        engine.on_session_reset()
        assert engine._compress_commit_reason == "no_proof"
    finally:
        clone.shutdown()


def test_span_clock_and_exception_preservation(published, monkeypatch, caplog):
    engine, _host, _output = published
    clock = iter([1.0, 1.125])
    diagnostic = {}
    monkeypatch.setattr(engine, "_end_attribution", diagnostic)
    monkeypatch.setattr(engine_module.time, "perf_counter", lambda: next(clock))
    with pytest.raises(ValueError, match="original"):
        with engine._attribution_span("ingest"):
            raise ValueError("original")
    assert diagnostic == {"ingest_ms": 125.0}


@pytest.mark.parametrize("kind", ["runtime", "interrupt", "locked"])
def test_fallback_errors_keep_existing_end_semantics(published, monkeypatch, caplog, kind):
    engine, host, _output = published
    engine._compress_commit_proof = None
    error = {"runtime": RuntimeError, "interrupt": KeyboardInterrupt, "locked": sqlite3.OperationalError}[kind]
    def fail(*args, **kwargs):
        raise error("database is locked" if kind == "locked" else "private synthetic exception")
    monkeypatch.setattr(engine, "_ingest_messages", fail)
    with caplog.at_level(logging.INFO):
        if kind == "runtime":
            with pytest.raises(RuntimeError, match="private synthetic"):
                engine.on_session_end("S", host)
        else:
            engine.on_session_end("S", host)
    assert len(records(caplog)) == 1
    assert fields(records(caplog)[0])["reason"] == "no_proof"
    assert "private synthetic" not in records(caplog)[0].getMessage()
    assert not engine._session_end_flush_active


def test_classifier_short_circuits_without_identity_pass(published, monkeypatch, caplog):
    engine, host, _output = published
    engine._compress_commit_proof["session_id"] = "other"
    engine._compress_commit_proof["input"] = None
    monkeypatch.setattr(engine, "_proof_replay_identity", lambda *a, **k: pytest.fail("unevaluated identity gate"))
    monkeypatch.setattr(engine, "_ingest_messages", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_remember_session_end_replay_snapshot", lambda *a: None)
    with caplog.at_level(logging.INFO):
        engine.on_session_end("S", host)
    assert fields(records(caplog)[0])["reason"] == "session_mismatch"


def test_debug_gate_avoids_clocks_at_info(published, monkeypatch, caplog):
    engine, _host, _output = published
    monkeypatch.setattr(engine_module.time, "perf_counter", lambda: pytest.fail("disabled span read clock"))
    with caplog.at_level(logging.INFO):
        with engine._attribution_span("preflight", 3):
            pass


def test_outer_prelude_and_admission_clocks_are_separate(published, monkeypatch, caplog):
    engine, _host, _output = published
    monkeypatch.setattr(engine_module.time, "perf_counter", lambda: 5.0)
    with caplog.at_level(logging.INFO):
        with engine._session_end_diagnostic([], 1.0, 3.0, 2.0):
            pass
    data = fields(records(caplog)[0])
    assert float(data["outer_ms"]) == 4000.0
    assert float(data["prelude_ms"]) == 1000.0
    assert float(data["admission_ms"]) == 1000.0
