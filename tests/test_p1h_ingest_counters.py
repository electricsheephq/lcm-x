"""P1h PR1: per-attempt, fail-open telemetry; no ingest decisions change."""
import copy
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager

import pytest

from hermes_lcm import ingest_ledger
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_ingest_ledger_writes import engines as _engines

engines = _engines


def records(engine):
    with engine._ingest_lock_guard:
        return list(engine._ingest_lock_records)


def test_busy_attempt_and_commit_are_separate(engines, monkeypatch):
    engine = engines()
    original = engine._store._append_protected_batch
    calls = []

    def busy(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            time.sleep(.02)
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine._store, "_append_protected_batch", busy)
    view = [_m("user", "new row", 1.)]
    assert engine._ingest_messages(view) == view
    first, second = records(engine)
    assert [(r["attempt"], r["outcome"]) for r in (first, second)] == [(1, "busy"), (2, "commit")]
    assert first["busy"]["attempt"] and not second["busy"]["attempt"]
    assert first["hold_ms"] >= 20
    assert second["hold_ms"] < 20
    assert engine.get_status()["ingest_lock"]["busy_attempts"] == 1
    assert engine._store.get_session_count("S") == 1


def test_failed_begin_has_no_hold(engines, monkeypatch):
    engine = engines()
    original = engine._store.atomic
    calls = []

    @contextmanager
    def begin(*, immediate=False):
        calls.append(immediate)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        with original(immediate=immediate):
            yield

    monkeypatch.setattr(engine._store, "atomic", begin)
    view = [_m("user", "after busy BEGIN", 1.)]
    assert engine._ingest_messages(view) == view
    first, second = records(engine)
    assert (first["attempt"], first["outcome"], first["hold_ms"]) == (1, "busy", None)
    assert second["attempt"] == 2 and second["outcome"] == "commit" and second["hold_ms"] > 0
    status = engine.get_status()["ingest_lock"]
    assert status["samples"] == 1 and status["busy_attempts"] == 1
    assert sum(status["buckets"].values()) == 2
    assert calls == [True, True] and engine._store.get_session_count("S") == 1


def test_hold_excludes_post_commit_settlement(engines, monkeypatch):
    engine = engines()
    original = ingest_ledger.settle_pending_emissions

    def settle(*args, **kwargs):
        time.sleep(.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest_ledger, "settle_pending_emissions", settle)
    engine._ingest_messages([_m("user", "settled", 1.)])
    sample = records(engine)[-1]
    assert sample["hold_ms"] < 50
    assert engine._last_ingest_attribution["lock_hold_ms"] >= 50
    assert set(engine._last_ingest_attribution) == {
        "lock_hold_ms", "rows_written", "bytes_written", "busy_snapshot_retry", "ledger_digest_ms", "ledger_ms"}


def test_hold_includes_insert(engines, monkeypatch):
    engine = engines()
    original = engine._store._append_protected_batch

    def append(*args, **kwargs):
        time.sleep(.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine._store, "_append_protected_batch", append)
    engine._ingest_messages([_m("user", "insert", 1.)])
    assert records(engine)[-1]["hold_ms"] >= 50


def test_view_classification_and_respacing_delta(engines, monkeypatch):
    engine = engines()
    view = [_m("user", "question", 1.), _m("assistant", "answer", 2.)]
    original = engine._write_ingest_ledger

    def write(attempt, callbacks, attribution):
        original(attempt, callbacks, attribution)
        callbacks.append(lambda: engine._ledger_count("ledger.respaced", amount=2))
        callbacks.append(lambda: engine._ledger_count("ledger.emission_busy"))

    monkeypatch.setattr(engine, "_write_ingest_ledger", write)
    before = getattr(engine, "_ingest_ledger_counters", {}).get("ledger.respaced", 0)
    engine._ingest_messages(view)
    fresh = records(engine)[-1]
    assert (fresh["view_rows"], fresh["rows_classified"], fresh["rows_written"], fresh["bytes_written"]) == (2, 2, 2, 14)
    assert fresh["respaced"] == engine._ingest_ledger_counters["ledger.respaced"] - before == 2
    assert fresh["busy"] == dict(attempt=False, ledger=1)
    engine._ingest_messages(view)
    replay = records(engine)[-1]
    assert (replay["view_rows"], replay["rows_classified"], replay["rows_written"]) == (2, 0, 0)
    engine._ingest_cursor_needs_reconcile = True
    engine._ingest_messages(view)
    assert records(engine)[-1]["rows_classified"] == 2
    assert engine.get_status()["ingest_lock"]["respaced_total"] == 6


def test_compress_emission_write_is_recorded(engines):
    engine = engines()
    output = [_m("user", "summary", 1.)]
    engine._compress_exit_write(output, (output, [dict(kind="summary", output_occurrence=dict(index=0, of=1))], None))
    sample = records(engine)[-1]
    assert (sample["route"], sample["outcome"], sample["view_rows"]) == ("emission", "commit", 1)
    assert sample["hold_ms"] > 0
    status = engine.get_status()["ingest_lock"]
    assert status["samples"] == 0 and status["emission"]["samples"] == 1


def test_ring_and_cumulative_buckets(engines):
    engine = engines()
    for _ in range(520):
        engine._ingest_messages([_m("user", "bounded", 1.)])
    assert engine._ingest_lock_records.maxlen == len(records(engine)) == 512
    status = engine.get_status()["ingest_lock"]
    assert status["samples"] == 512
    assert sum(status["buckets"].values()) == sum(status["small_view_buckets"].values()) == 520


def test_quantiles_exclude_large_views_and_emissions(engines):
    engine = engines()
    for n, hold in [(1, 1.), (2000, 2.), (2001, 900.), (1, 1000.)]:
        sample = engine._new_ingest_lock_record(range(n), 1)
        sample["hold_ms"] = hold
        if hold == 1000:
            sample["route"] = "emission"
        engine._record_ingest_lock(sample, {})
    status = engine.get_status()["ingest_lock"]
    assert (status["samples"], status["p90_ms"], status["p99_ms"], status["max_ms"]) == (2, 2., 2., 2.)
    assert sum(status["buckets"].values()) == 4
    assert sum(status["small_view_buckets"].values()) == 3
    assert status["emission"]["max_ms"] == 1000.


def test_recording_failure_preserves_result_and_rows(engines, monkeypatch, caplog):
    engine, control = engines("broken"), engines("control")
    view = [_m("user", "durable", 1.)]
    expected = control._ingest_messages(copy.deepcopy(view))

    def fail(*args):
        raise ValueError("telemetry unavailable")

    monkeypatch.setattr(engine, "_record_ingest_lock", fail)
    with caplog.at_level(logging.DEBUG):
        assert engine._ingest_messages(copy.deepcopy(view)) == expected
        assert engine._ingest_ledger_counters["telemetry.failed"] == 1
        assert engine._store.get_session_messages("S")[0]["content"] == "durable"
        assert engine._ingest_messages(copy.deepcopy(view)) == expected
    assert engine._ingest_ledger_counters["telemetry.failed"] == 2
    assert sum("ingest lock telemetry failed" in r.getMessage() for r in caplog.records) == 1
    assert engine._store.get_session_count("S") == control._store.get_session_count("S") == 1


def test_status_failure_is_local_to_key(engines, monkeypatch):
    engine = engines()

    def fail():
        raise ValueError("broken summary")

    monkeypatch.setattr(engine, "_ingest_lock_status", fail)
    status = engine.get_status()
    assert status["ingest_lock"] == {"error": "ValueError"}
    assert status["ingest_failure_count"] == 0


def test_status_while_ingests_append(engines):
    engine = engines()
    errors, seen = [], []
    ready = threading.Event()

    def read():
        ready.set()
        try:
            for _ in range(20):
                seen.append(engine.get_status()["ingest_lock"])
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=read)
    thread.start()
    assert ready.wait(2)
    for i in range(50):
        engine._ingest_messages([_m("user", f"row {i}", float(i + 1))])
    thread.join(10)
    assert not thread.is_alive() and not errors
    assert len(seen) == 20 and all("error" not in s for s in seen)


@pytest.mark.parametrize("route", ["foreground", "warm_tail", "session_end", "adapter"])
def test_routes(engines, route):
    engine = engines()
    engine._session_end_flush_active = route == "session_end"
    if route == "adapter":
        engine._session_platform = "claude-code"
    engine._ingest_messages([_m("user", "routed", 1.)], no_view=route == "warm_tail")
    assert records(engine)[-1]["route"] == route


def test_shallow_clone_has_own_ring(engines):
    engine = engines()
    engine._ingest_messages([_m("user", "before clone", 1.)])
    clone = copy.copy(engine)
    assert clone._store is engine._store
    assert not records(clone)
    assert clone._ingest_lock_guard is not engine._ingest_lock_guard
    assert len(records(engine)) == 1


def test_rollback_record_preserves_exception_and_zero_rows(engines, monkeypatch):
    engine = engines()

    def fail(*args, **kwargs):
        raise ValueError("insert failure")

    monkeypatch.setattr(engine._store, "_append_protected_batch", fail)
    with pytest.raises(ValueError, match="insert failure"):
        engine._ingest_messages([_m("user", "not committed", 1.)])
    assert records(engine)[-1]["outcome"] == "rollback"
    assert engine._store.get_session_count("S") == 0
