"""P1h PR1: per-attempt, fail-open telemetry; no ingest decisions change."""
import copy
import hashlib
import json
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


@pytest.fixture
def clock(monkeypatch):
    """Advance perf_counter only at the transaction boundaries each test exercises."""
    class Clock:
        milliseconds = 0

        def advance(self, milliseconds):
            self.milliseconds += milliseconds

        def __call__(self):
            return self.milliseconds / 1000

    clock = Clock()
    monkeypatch.setattr(time, "perf_counter", clock)
    monkeypatch.setattr("hermes_lcm.engine._perf_counter", clock)
    return clock


def test_busy_attempt_and_commit_are_separate(engines, monkeypatch, clock):
    engine = engines()
    original = engine._store._append_protected_batch
    calls = []

    def busy(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            clock.advance(20)
            raise sqlite3.OperationalError("database is locked")
        clock.advance(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine._store, "_append_protected_batch", busy)
    view = [_m("user", "new row", 1.)]
    assert engine._ingest_messages(view) == view
    first, second = records(engine)
    assert [(r["attempt"], r["outcome"]) for r in (first, second)] == [(1, "busy"), (2, "commit")]
    assert first["busy"]["attempt"] and not second["busy"]["attempt"]
    assert first["hold_ms"] == pytest.approx(20)
    assert second["hold_ms"] == pytest.approx(5)
    assert engine.get_status()["ingest_lock"]["busy_attempts"] == 1
    assert engine._store.get_session_count("S") == 1


@pytest.mark.parametrize("prelock", [False, True], ids=["eager", "lazy"])
def test_failed_begin_has_no_hold(engines, monkeypatch, clock, prelock):
    engine = engines()
    monkeypatch.setattr(engine._store, "_ingest_prelock", prelock)
    calls = []
    if prelock:  # #2293: the lazy attempt's BEGIN IMMEDIATE runs at its first write, after this seam
        seam = type(engine._store)._ingest_prelock_before_begin

        def begin(store):
            calls.append(True)
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            return seam(store)

        monkeypatch.setattr(type(engine._store), "_ingest_prelock_before_begin", begin)
    else:
        original = engine._store.atomic

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
    print("RECORDS", records(engine), "CALLS", calls)
    first, second = records(engine)
    assert (first["attempt"], first["outcome"], first["hold_ms"]) == (1, "busy", None)
    assert second["attempt"] == 2 and second["outcome"] == "commit" and second["hold_ms"] == 0
    status = engine.get_status()["ingest_lock"]
    assert status["samples"] == 1 and status["busy_attempts"] == 1
    assert sum(status["buckets"].values()) == 2
    assert calls == [True, True] and engine._store.get_session_count("S") == 1


def test_hold_excludes_post_commit_settlement(engines, monkeypatch, clock):
    engine = engines()
    original = ingest_ledger.settle_pending_emissions
    original_append = engine._store._append_protected_batch

    def append(*args, **kwargs):
        clock.advance(7)
        return original_append(*args, **kwargs)

    monkeypatch.setattr(engine._store, "_append_protected_batch", append)

    def settle(*args, **kwargs):
        clock.advance(50)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest_ledger, "settle_pending_emissions", settle)
    engine._ingest_messages([_m("user", "settled", 1.)])
    sample = records(engine)[-1]
    assert sample["hold_ms"] == pytest.approx(7)
    assert engine._last_ingest_attribution["lock_hold_ms"] == pytest.approx(57)
    assert set(engine._last_ingest_attribution) == {
        "lock_hold_ms", "rows_written", "bytes_written", "busy_snapshot_retry", "ledger_digest_ms", "ledger_ms"}


def test_hold_includes_insert(engines, monkeypatch, clock):
    engine = engines()
    original = engine._store._append_protected_batch

    def append(*args, **kwargs):
        clock.advance(50)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine._store, "_append_protected_batch", append)
    engine._ingest_messages([_m("user", "insert", 1.)])
    assert records(engine)[-1]["hold_ms"] == pytest.approx(50)


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


def test_compress_emission_write_is_recorded(engines, clock):
    engine = engines()
    output = [_m("user", "summary", 1.)]
    engine._compress_exit_write(output, (output, [dict(kind="summary", output_occurrence=dict(index=0, of=1))], None))
    sample = records(engine)[-1]
    assert (sample["route"], sample["outcome"], sample["view_rows"]) == ("emission", "commit", 1)
    assert sample["hold_ms"] == 0
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


def test_shallow_copy_shares_telemetry(engines):
    engine = engines()
    engine._ingest_messages([_m("user", "before clone", 1.)])
    clone = copy.copy(engine)
    assert clone._store is engine._store
    assert clone._ingest_lock_records is engine._ingest_lock_records
    assert clone._ingest_lock_guard is engine._ingest_lock_guard
    assert clone._ingest_lock_totals is engine._ingest_lock_totals
    assert len(records(clone)) == len(records(engine)) == 1


def test_rollback_record_preserves_exception_and_zero_rows(engines, monkeypatch):
    engine = engines()

    def fail(*args, **kwargs):
        raise ValueError("insert failure")

    monkeypatch.setattr(engine._store, "_append_protected_batch", fail)
    with pytest.raises(ValueError, match="insert failure"):
        engine._ingest_messages([_m("user", "not committed", 1.)])
    assert records(engine)[-1]["outcome"] == "rollback"
    assert engine._store.get_session_count("S") == 0


@pytest.mark.parametrize("body_fails", [False, True])
def test_r4_proof_write_survives_missing_record(engines, monkeypatch, body_fails):
    engine = engines()
    proof = dict(output=[], output_metadata=[])
    payload = dict(effective_sha256=[])
    original = engine._store.write_metadata_json

    def fail_setup(*args):
        raise ValueError("telemetry setup failure")

    def write(keys, serialized, **kwargs):
        result = original(keys, serialized, **kwargs)
        if body_fails and "host_occurrence_snapshot" in keys[0]:
            raise ValueError("proof body failure")
        return result

    monkeypatch.setattr(engine, "_new_ingest_lock_record", fail_setup)
    monkeypatch.setattr(engine._store, "write_metadata_json", write)
    before = list(engine._store._conn.execute("SELECT key,value FROM metadata ORDER BY key"))
    if body_fails:
        with pytest.raises(ValueError, match="proof body failure"):
            engine._write_compress_commit_proof_payload(proof, payload)
        assert list(engine._store._conn.execute("SELECT key,value FROM metadata ORDER BY key")) == before
    else:
        engine._write_compress_commit_proof_payload(proof, payload)
        values = [row[0] for row in engine._store._conn.execute("SELECT value FROM metadata")]
        assert json.dumps(payload, sort_keys=True) in values
        snapshot = dict(output_metadata=[], output_length=0,
                        proof_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        assert json.dumps(snapshot, sort_keys=True) in values
    assert not engine._store._conn.in_transaction
    assert engine._ingest_ledger_counters["telemetry.failed"] == 1


@pytest.mark.parametrize("body_fails", [False, True])
def test_r4_timed_atomic_accepts_none(engines, body_fails):
    engine = engines()
    def write():
        with engine._store._timed_atomic(None):
            engine._store.write_metadata_json(["r4-none"], "true")
            if body_fails:
                raise ValueError("timer body failure")
    if body_fails:
        with pytest.raises(ValueError, match="timer body failure"):
            write()
    else:
        write()
    assert list(engine._store._conn.execute("SELECT value FROM metadata WHERE key='r4-none'")) == (
        [] if body_fails else [("true",)])


def test_r4_busy_emission_begin_has_no_hold(engines):
    engine = engines()
    conn = engine._store._conn
    with sqlite3.connect(engine._store.db_path, timeout=.01) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with engine._emission_lock_record(1) as timing, engine._emission_owner(
                    engine._store.ingest_ledger_salt(), timing):
                pytest.fail("busy BEGIN must not enter the body")
        assert not conn.in_transaction
    sample, = records(engine)
    assert (sample["route"], sample["outcome"], sample["hold_ms"]) == ("emission", "busy", None)
    assert engine.get_status()["ingest_lock"]["emission"] == dict(
        samples=0, p90_ms=None, p99_ms=None, max_ms=None)


def test_r4_rekey_inside_ingest_has_no_emission_record(engines, tmp_path, monkeypatch, clock):
    from tests.test_issue_1286_cleanup_rewrite_proof import Host, _commit_cleanup, _turns

    host = Host(tmp_path / "rekey")
    try:
        _turns(host, "reasoning")
        _commit_cleanup(host, "reasoning")
        engine = host.engine
        original = engine._rekey_durable_commit_proof
        calls = []

        def rekey(messages):
            assert engine._store._transaction_owner.owns(engine._store._conn)
            before = engine._durable_commit_proof_payload()
            original(messages)
            assert engine._durable_commit_proof_payload() != before
            calls.append(1)

        monkeypatch.setattr(engine, "_rekey_durable_commit_proof", rekey)
        host.live.insert(2, _m("user", "insert before cursor", host.stamp()))
        start, rows = len(records(engine)), host.stored()
        host.turn()
        assert calls == [1]
        sample, = records(engine)[start:]
        assert (sample["route"], sample["outcome"]) == ("foreground", "commit")
        assert sample["rows_written"] == host.stored() - rows > 0
        assert sample["hold_ms"] == 0
    finally:
        host.close()


def test_r4_holdless_attempt_has_own_bucket(engines):
    engine = engines()
    for size, hold in [(1, None), (2001, None), (1, 2.)]:
        sample = engine._new_ingest_lock_record(range(size), 1)
        sample["hold_ms"] = hold
        engine._record_ingest_lock(sample, {}, sqlite3.OperationalError("database is locked") if hold is None else None)
    status = engine.get_status()["ingest_lock"]
    assert status["buckets"]["no_hold"] == 2
    assert status["small_view_buckets"]["no_hold"] == 1
    assert status["buckets"]["le_50_ms"] == status["small_view_buckets"]["le_50_ms"] == 1
    assert sum(status["buckets"].values()) == 3
    assert sum(status["small_view_buckets"].values()) == 2
    assert status["samples"] == 1


@pytest.mark.parametrize("busy", [False, True])
def test_r4_late_rollback_reports_no_durable_writes(engines, monkeypatch, busy):
    engine = engines()
    error = sqlite3.OperationalError("database is locked") if busy else ValueError("late failure")
    def fail(*args):
        assert engine._last_ingest_attribution["rows_written"] == 1
        assert engine._last_ingest_attribution["bytes_written"] > 0
        raise error
    monkeypatch.setattr(engine, "_write_ingest_ledger", fail)
    with pytest.raises(type(error), match=str(error)):
        engine._ingest_messages([_m("user", "rolled back", 1.)])
    assert len(records(engine)) == (2 if busy else 1)
    for sample in records(engine):
        assert sample["outcome"] == ("busy" if busy else "rollback")
        assert sample["rows_written"] == sample["bytes_written"] == 0
    assert engine._store.get_session_count("S") == 0


def test_r4_hold_starts_after_begin_succeeds(engines, monkeypatch, clock):
    engine = engines()
    # #2293: pins the explicit BEGIN's timing, so it runs eager (the lazy attempt may begin through Python's
    # implicit BEGIN, which this wrapper cannot delay; its hold then starts when that BEGIN is issued).
    monkeypatch.setattr(engine._store, "_ingest_prelock", False)
    class DelayedBegin:
        def __init__(self, conn):
            self.conn, self.begins = conn, 0
        def __getattr__(self, name):
            return getattr(self.conn, name)
        def execute(self, sql, *args):
            if sql == "BEGIN IMMEDIATE":
                self.begins += 1
                clock.advance(150)
            return self.conn.execute(sql, *args)
    conn = DelayedBegin(engine._store._conn)
    monkeypatch.setattr(engine._store, "_conn", conn)
    original_append = engine._store._append_protected_batch

    def append(*args, **kwargs):
        clock.advance(25)
        return original_append(*args, **kwargs)

    monkeypatch.setattr(engine._store, "_append_protected_batch", append)
    started = time.perf_counter()
    engine._ingest_messages([_m("user", "after admission", 1.)])
    elapsed = (time.perf_counter() - started) * 1000
    assert conn.begins == 1 and elapsed == pytest.approx(175)
    assert records(engine)[-1]["hold_ms"] == pytest.approx(25)
    assert engine._store.get_session_count("S") == 1


@pytest.mark.parametrize("metadata", ["timestamp", "message_uid"])
@pytest.mark.parametrize("rewound", [0, 2])
def test_rows_classified_uses_rewound_prefix(engines, metadata, rewound):
    """#1179: a changed host occurrence exposes rows before the entry cursor."""
    engine = engines()
    view = [_m("user", "alpha", 1., "u1"), _m("assistant", "first", 2., "a1"),
            _m("user", "same text", 3., "u2"), _m("assistant", "old answer", 4., "a2")]
    engine._ingest_messages(copy.deepcopy(view))
    assert engine._ingest_cursor == len(view)
    retry = copy.deepcopy(view)
    retry[rewound][metadata] = 50. if metadata == "timestamp" else "retry-occurrence"
    engine._ingest_messages(retry)
    sample = records(engine)[-1]
    assert (sample["view_rows"], sample["rows_classified"]) == (len(retry), len(retry) - rewound)
    assert sample["outcome"] == "commit" and sample["rows_written"] == 1
    stored = engine._store.get_session_messages("S")
    assert len(stored) == len(view) + 1
    assert stored[-1]["content"] == view[rewound]["content"]
    engine._ingest_messages(copy.deepcopy(retry))
    assert records(engine)[-1]["rows_classified"] == 0
    assert engine._store.get_session_count("S") == len(stored)


@pytest.mark.parametrize("rollback", [False, True], ids=["commit", "rollback"])
def test_hold_includes_transaction_exit(engines, monkeypatch, clock, rollback):
    engine = engines()

    class ExitDelay:
        def __init__(self, conn):
            self.conn, self.exits = conn, []

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def commit(self):
            self.exits.append("commit")
            clock.advance(13)
            return self.conn.commit()

        def rollback(self):
            self.exits.append("rollback")
            clock.advance(17)
            return self.conn.rollback()

    conn = ExitDelay(engine._store._conn)
    monkeypatch.setattr(engine._store, "_conn", conn)
    original = engine._store._append_protected_batch

    def append(*args, **kwargs):
        clock.advance(7)
        if rollback:
            raise ValueError("timed rollback")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine._store, "_append_protected_batch", append)
    if rollback:
        with pytest.raises(ValueError, match="timed rollback"):
            engine._ingest_messages([_m("user", "transaction exit", 1.)])
    else:
        engine._ingest_messages([_m("user", "transaction exit", 1.)])
    assert conn.exits == ["rollback" if rollback else "commit"]
    assert records(engine)[-1]["hold_ms"] == pytest.approx(24 if rollback else 20)
    assert records(engine)[-1]["outcome"] == ("rollback" if rollback else "commit")


def test_emission_hold_times_its_write_only(engines, monkeypatch, clock):
    engine = engines()
    original = ingest_ledger.insert_emissions
    calls = []

    def write(*args, **kwargs):
        calls.append(1)
        clock.advance(11)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest_ledger, "insert_emissions", write)
    prepare = engine._ledger_view_entries

    def entries(*args, **kwargs):
        clock.advance(19)  # digest preparation is outside the write owner
        return prepare(*args, **kwargs)

    monkeypatch.setattr(engine, "_ledger_view_entries", entries)
    clock.advance(101)  # outside the emission owner
    output = [_m("user", "summary", 1.)]
    engine._compress_exit_write(output, (output, [dict(kind="summary", output_occurrence=dict(index=0, of=1))], None))
    clock.advance(103)  # outside the emission owner
    sample, = records(engine)
    assert calls == [1]
    assert (sample["route"], sample["outcome"]) == ("emission", "commit")
    assert sample["hold_ms"] == pytest.approx(11)
