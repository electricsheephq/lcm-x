"""T1-24 structural audit and repeatable reader/ingest deadlock probe.

Manual positive control only:
  CODEX_2293_UNSAFE_CALLBACK=1 CODEX_2293_D2_COUNT=30 pytest tests/test_2293_callbacks.py -k reader
Fixed run:
  CODEX_2293_D2_COUNT=300 pytest tests/test_2293_callbacks.py -k reader
Each cell has a process watchdog because the original mutex/GIL inversion can
prevent even a Python thread join timeout from resuming.
"""
import faulthandler
import multiprocessing
import os
import threading
from contextlib import contextmanager, nullcontext

import pytest


@pytest.mark.parametrize("method", ["set_authorizer", "set_trace_callback", "set_progress_handler", "create_function"])
def test_d1_audit_detects_product_callbacks(method, tmp_path):
    from hermes_lcm.store import MessageStore
    from tests.message_store_callback_audit import CALLS
    store = MessageStore(tmp_path / "callbacks.db")
    conn = store._conn
    args = {"set_authorizer": (lambda *_: 0,), "set_trace_callback": (lambda *_: None,),
            "set_progress_handler": (lambda: 0, 1), "create_function": ("probe", 0, lambda: 0)}
    start = len(CALLS)
    try:
        # Compile a product frame: a test ancestor must not exempt the call.
        from tests.message_store_callback_audit import ROOT
        code = compile("conn." + method + "(*args)", str(ROOT / "store.py"), "exec")
        exec(code, {"conn": conn, "args": args[method]})
        assert len(CALLS) == start + 1 and CALLS[-1][:2] == (method, True)
    finally:
        # This positive control is deliberately outside the suite's product verdict.
        del CALLS[start:]
        store.close()


def test_instrument_refuses_threads_and_never_denies(tmp_path):
    import sqlite3
    from types import SimpleNamespace
    from hermes_lcm.store import MessageStore
    from tests.prelock_instrument import PrelockRecorder, recording_prelock
    store = MessageStore(tmp_path / "record.db")
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait)
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="single-threaded"):
            with recording_prelock(store):
                pass
    finally:
        stop.set()
        worker.join(10)
    # The full suite may already have maintenance threads. Always check the
    # recording verdict directly, installing it only in a single-threaded test.
    context = recording_prelock(store) if threading.active_count() == 1 else nullcontext(PrelockRecorder(store))
    with context as recorder:
        store._ingest_prelock_state = SimpleNamespace(phase="prelock")
        for action in (sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT, sqlite3.SQLITE_CREATE_TABLE):
            assert recorder.authorize(action, "COMMIT", None, "main", None) == sqlite3.SQLITE_OK
        assert len(recorder.forbidden) == 3
        with pytest.raises(AssertionError):
            recorder.check()
        recorder.forbidden.clear()  # simulated forbidden events were detected; no real SQL was issued
        store._ingest_prelock_state = None
    store.close()


def test_oracle_checks_poison_at_the_failed_begin():
    import sqlite3
    from types import SimpleNamespace
    from tests.prelock_instrument import PrelockRecorder
    state = SimpleNamespace(phase="prelock", T_clean=0, poisoned=False)
    store = SimpleNamespace(_ingest_prelock_state=state, _ingest_is_prelock=lambda: True,
                            _conn=SimpleNamespace(total_changes=0, in_transaction=False))
    recorder = PrelockRecorder(store)
    assert recorder.authorize(sqlite3.SQLITE_TRANSACTION, "BEGIN", None, "main", None) == sqlite3.SQLITE_OK
    recorder.failed(sqlite3.OperationalError("database is locked"))
    state.poisoned = True  # a later state change cannot excuse a missing failure hook
    with pytest.raises(AssertionError):
        recorder.check()
    recorder = PrelockRecorder(store)
    recorder.authorize(sqlite3.SQLITE_TRANSACTION, "BEGIN", None, "main", None)
    recorder.failed(sqlite3.OperationalError("database is locked"))
    state.poisoned = False  # a successful eager rerun may clear it
    recorder.check()


def _reader_worker(root, positive, result):
    # The spawn child does not run conftest; the harness registers the package.
    from tests.behaviour import ingest_trace_harness  # noqa: F401
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    from tests.prelock_instrument import recording_prelock
    faulthandler.enable()
    faulthandler.dump_traceback_later(15)
    engine = LCMEngine(config=LCMConfig(database_path=str(root / "lcm.db"), embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
        large_output_externalization_enabled=False), hermes_home=str(root / "home"))
    engine.on_session_start("D2", platform="cli", context_length=1_000_000)
    if positive:
        original = engine._store.ingest_atomic

        @contextmanager
        def callback_in_production_position(**kwargs):
            with original(**kwargs) as timing:
                with recording_prelock(engine._store, unsafe_threaded_positive_control=True):
                    yield timing
        engine._store.ingest_atomic = callback_in_production_position
    done, ready = threading.Event(), threading.Event()
    errors, seen = [], []

    def reader():
        ready.set()
        try:
            while not done.is_set():
                seen.append(engine.get_status()["ingest_lock"])
        except BaseException as exc:
            errors.append(repr(exc))

    def writer():
        try:
            assert ready.wait(10)
            for i in range(100):
                engine._ingest_messages([dict(role="user", content=f"row {i}", timestamp=float(i + 1))])
        except BaseException as exc:
            errors.append(repr(exc))
        finally:
            done.set()

    threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=writer, daemon=True)]
    for thread in threads:
        thread.start()
    # The reader exits only after the writer sets done. Await that producer
    # first, then give the reader its join budget to observe the stop signal.
    for thread in reversed(threads):
        thread.join(20)  # per-repetition thread timeout; no elapsed-time assertions
    if any(thread.is_alive() for thread in threads):
        result.put(("hang", errors))
        return
    result.put(("ok", errors, len(seen), all("error" not in s for s in seen),
                engine._store.get_session_count("D2")))
    engine.shutdown()
    faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("repetition", range(int(os.environ.get("CODEX_2293_D2_COUNT", "1"))))
def test_d2_status_reader_until_all_ingests_finish(repetition, tmp_path):
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_reader_worker,
        args=(tmp_path, os.environ.get("CODEX_2293_UNSAFE_CALLBACK") == "1", result))
    process.start()
    process.join(50)
    try:
        assert not process.is_alive(), f"D2 hang in repetition {repetition} (faulthandler enabled)"
        assert process.exitcode == 0
        report = result.get(timeout=5)
        assert report[0] == "ok" and not report[1] and report[2] > 0 and report[3] and report[4] == 100, report
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        result.close()
