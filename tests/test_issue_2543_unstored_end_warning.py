"""#2543: report unstored ends without changing main's stored rows or retry decisions."""
import copy
import logging
import sqlite3
import time

import pytest

from hermes_lcm import engine as engine_module, maintenance_scheduler
from tests.test_issue_2055_ingest_ledger_writes import engines as engines
from tests.invariants.test_issue_2543_cold_inline_block import _call
from tests.invariants.test_issue_2539_offcurrent_marker_end import _setup, _changed_marker_view
from tests.invariants.test_p1e_cold_tails import fail_end_read, committed_rows
from tests.test_issue_2729_fallback_gates import cold as cold  # noqa: F401  (fixture)
from tests.test_issue_2729_fallback_gates import HISTORY, STALE, VIEW, busy as busy_atomic
from tests.test_issue_2729_offcurrent_end_after_restart import _start


CELLS = [(shape, route) for shape in ("wrapped-base64", "data-uri-key")
         for route in ("cold-pairwise", "cold-counted")]
CELLS += [(location, route) for location in ("spillover", "hermes-results")
          for route in ("direct", "busy")]


def _prepare(engines, monkeypatch, tmp_path, shape):
    if shape in ("spillover", "hermes-results"):
        engine, view, _, _ = _setup(engines, monkeypatch, tmp_path,
                                    "spillover" if shape == "spillover" else "tempdir")
        _changed_marker_view(engine, view, shape)
        return engine, view
    engine = engines(sessions=[("S1", None, None), ("foreground", None, None)])
    first = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
             {"role": "user", "content": "C"}, _call(shape),
             {"role": "tool", "tool_call_id": "m1", "content": "ok 1"}]
    first += [{"role": r, "content": t} for r, t in
              (("assistant", "D"), ("user", "E"), ("assistant", "F"), ("user", "G"), ("assistant", "H"))]
    for i, row in enumerate(first):
        row["timestamp"] = float(i + 1)
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    engine.ingest(copy.deepcopy(first))
    engine.on_session_start("foreground", platform="cli", context_length=200_000)
    return engine, [*copy.deepcopy(first), {"role": "assistant", "content": "FINAL", "timestamp": 11.0}]


def _end(engine, view, monkeypatch, route):
    if route == "direct":
        return engine.on_session_end("S1", view)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine))
        engine.on_session_end("S1", view)
    assert len(engine._pending_session_tails) == 1
    engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
    if route == "cold-counted":
        assert engine._cold_tail_absent_count(engine._pending_session_tails[0]) == 2
    engine._retry_pending_session_tails()


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "reason=" in r.getMessage() and "view_rows=" in r.getMessage()]


@pytest.mark.parametrize("shape,route", CELLS)
@pytest.mark.parametrize("logger_mode", ["normal", "silent", "raising"])
def test_pinned_loss_warns_once_without_changing_rows(engines, monkeypatch, tmp_path, caplog,
                                                       shape, route, logger_mode):
    engine, view = _prepare(engines, monkeypatch, tmp_path, shape)
    before = engine._store.get_session_messages("S1")
    assert len(before) == 10 and len(view) == 11
    warning = engine_module.logger.warning

    def diagnostic_logger(message, *args, **kwargs):
        if "reason=" in message or message.startswith("LCM session-end rows not stored:"):
            if logger_mode == "raising":
                raise RuntimeError("injected warning handler failure")
            if logger_mode == "silent":
                return
        return warning(message, *args, **kwargs)

    caplog.clear()
    with monkeypatch.context() as patch:
        patch.setattr(engine_module.logger, "warning", diagnostic_logger)
        assert _end(engine, view, monkeypatch, route) is None
    assert engine._store.get_session_messages("S1") == before
    assert not any(r["content"] == "FINAL" for r in before)
    assert len(engine._pending_session_tails) == (route != "direct")
    reason = "prefix_mismatch_bypass_lineage" if route == "direct" else "cold_tail_prefix_mismatch"
    warnings = _warnings(caplog)
    assert len(warnings) == (logger_mode == "normal")
    if warnings:
        assert f"reason={reason} view_rows=11" in warnings[0]
        assert warnings[0].startswith("LCM session-end rows not stored: session=S1" if route == "direct" else
                                     "LCM session-end cold tail prefix mismatch; retry pending")


@pytest.mark.parametrize("route", ["direct", "busy"])
def test_stored_suffix_has_no_unstored_warning(engines, monkeypatch, tmp_path, caplog, route):
    engine, view = _prepare(engines, monkeypatch, tmp_path, "plain")
    caplog.clear()
    _end(engine, view, monkeypatch, route)
    rows = engine._store.get_session_messages("S1")
    assert len(rows) == 11 and rows[-1]["content"] == "FINAL"
    assert not engine._pending_session_tails and not _warnings(caplog)


@pytest.mark.parametrize("terminal", ["shutdown", "exit", "exit-locked"])
def test_terminal_attempt_reports_each_pending_tail(engines, monkeypatch, tmp_path, caplog, terminal):
    engine, view = _prepare(engines, monkeypatch, tmp_path, "wrapped-base64")
    _end(engine, view, monkeypatch, "cold-pairwise")
    before = committed_rows(engine._db_path, "S1")
    caplog.clear()
    with monkeypatch.context() as patch:
        if terminal == "shutdown":
            engine.shutdown()
        else:
            patch.setattr(maintenance_scheduler, "_PENDING_TAIL_ENGINES", {engine})
            if terminal == "exit-locked":
                from contextlib import contextmanager

                @contextmanager
                def locked(*args, **kwargs):
                    yield False
                patch.setattr(engine, "_exclusive_lifecycle", locked)
            maintenance_scheduler._flush_pending_tails_at_exit()
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert f"reason=pending_tail_dropped_at_{'shutdown' if terminal == 'shutdown' else 'exit'} view_rows=11" in warnings[0]
    assert committed_rows(engine._db_path, "S1") == before


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("capped", [False, True])
def test_other_compare_exits_warn_once(engines, monkeypatch, tmp_path, caplog, current, capped):
    engine, view = _prepare(engines, monkeypatch, tmp_path, "wrapped-base64")
    _end(engine, view, monkeypatch, "cold-pairwise")
    tail = engine._pending_session_tails[0]
    tail.pop("unmatched_warned")  # Exercise the other existing once-per-tail branches independently.
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_cold_tail_one_conflict", lambda _: False if capped else (0, {}, 1.0))
        patch.setattr(engine, "_cold_tail_conflict_placed", lambda *a, **k: False)
        if current:
            patch.setattr(engine, "_session_id", "S1")
            patch.setattr(engine, "_conversation_id", "conv1")
            patch.setattr(engine, "_ingest_messages", lambda *a, **k: None)
        caplog.clear()
        engine._retry_pending_session_tails()
        engine._retry_pending_session_tails()
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    reason = "cold_tail_scan_capped" if capped else (
        "cold_tail_conflicting_stamp_unplaced" if current else "cold_tail_prefix_mismatch")
    assert f"reason={reason} view_rows=11" in warnings[0]


def test_unverified_stale_end_warns(engines, monkeypatch, caplog):
    engine = engines()
    engine.on_session_start("S", platform="cli", context_length=200_000)
    view = [{"role": "assistant", "content": "FINAL"}]
    caplog.clear()
    engine.on_session_end("unknown", view)
    assert _warnings(caplog) == [
        "LCM session-end rows not stored: session=unknown reason=unverified_stale_end view_rows=1"]
    assert not engine._store.get_session_messages("unknown")


@pytest.mark.parametrize("outer,error,reason", [
    (False, KeyboardInterrupt, "foreground_ingest_interrupted"),
    (False, sqlite3.OperationalError, "foreground_ingest_busy_unqueued"),
    (True, KeyboardInterrupt, "foreground_flush_interrupted"),
    (True, sqlite3.OperationalError, "foreground_flush_locked"),
])
def test_foreground_failed_flush_reports_unstored_view(engines, monkeypatch, caplog, outer, error, reason):
    engine = engines()
    engine.on_session_start("S", platform="cli", context_length=200_000)

    def fail(*args, **kwargs):
        raise error("database is locked")

    monkeypatch.setattr(engine_module if outer else engine,
                        "_temporary_sqlite_busy_timeout" if outer else "_ingest_messages", fail)
    caplog.clear()
    engine.on_session_end("S", [{"role": "assistant", "content": "FINAL"}])
    assert _warnings(caplog) == [f"LCM session-end rows not stored: session=S reason={reason} view_rows=1"]
    assert not engine._store.get_session_messages("S")


def test_same_id_bypass_suffix_warns(engines, monkeypatch, tmp_path, caplog):
    from hermes_lcm.session_patterns import compile_session_patterns
    engine, view, _, n = _setup(engines, monkeypatch, tmp_path, "plain")
    engine._compiled_ignore_session_patterns = compile_session_patterns(["cron"])
    engine.on_session_start("S1", platform="cron", context_length=200_000)
    engine.ingest(view[:-1])
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    caplog.clear()
    engine.on_session_end("S1", view)
    assert _warnings(caplog) == ["LCM session-end rows not stored: session=S1 reason=bypass_lineage_end view_rows=11"]
    assert len(engine._store.get_session_messages("S1")) == n


def test_ignored_session_off_current_end_has_no_unstored_warning(engines, caplog):
    # A cron-like ignored session is never stored by design: its off-current end is not a loss.
    from hermes_lcm.session_patterns import compile_session_patterns
    engine = engines(sessions=[("S1", None, None), ("foreground", None, None)])
    engine._compiled_ignore_session_patterns = compile_session_patterns(["cron"])
    engine.on_session_start("S1", platform="cron", context_length=200_000)
    view = [{"role": "user", "content": "A", "timestamp": 1.0}, {"role": "assistant", "content": "B", "timestamp": 2.0}]
    engine.ingest(copy.deepcopy(view))
    engine.on_session_start("foreground", platform="cli", context_length=200_000)
    assert engine._has_lcm_bypass_lineage_session("S1")
    caplog.clear()
    engine.on_session_end("S1", copy.deepcopy(view))
    assert not _warnings(caplog)
    assert not engine._store.get_session_messages("S1")


def test_bypass_lineage_busy_suffix_append_warns_before_reraise(engines, monkeypatch, caplog):
    # #2738's shape: the suffix append is re-raised unqueued; the loss stays (pinned there), the WARNING names it.
    from tests.invariants.test_issue_2738_bypass_lineage_busy_suffix import setup, contents
    engine, view = setup(engines)
    caplog.clear()

    def busy(*args, **kwargs):  # As #2738's busy_end: no real lock, so no busy-timeout wait.
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_off_current_session_end_suffix", busy)
        with pytest.raises(sqlite3.OperationalError):
            engine.on_session_end("S1", copy.deepcopy(view))
    assert _warnings(caplog) == [
        f"LCM session-end rows not stored: session=S1 reason=off_current_suffix_append_failed view_rows={len(view)}"]
    assert contents(engine) == ["A", "B"] and not engine._pending_session_tails


def test_repeated_busy_end_with_pending_tail_has_no_unstored_warning(engines, caplog):
    # The first end queues a warm tail; an identical repeated end is deduplicated, and the retry still stores it.
    engine = engines(sessions=[("S", None, None)])
    engine.on_session_start("S", platform="cli", context_length=200_000)
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"t-{i}", "timestamp": float(i + 1)}
               for i in range(6)]
    engine.ingest(copy.deepcopy(history[:4]))
    caplog.clear()
    locker = sqlite3.connect(str(engine._db_path), timeout=0, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", copy.deepcopy(history))
        assert len(engine._pending_session_tails) == 1 and not _warnings(caplog)
        engine.on_session_end("S", copy.deepcopy(history))
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    assert len(engine._pending_session_tails) == 1 and not _warnings(caplog)
    engine._retry_pending_session_tails()
    assert [r["content"] for r in engine._store.get_session_messages("S")] == [m["content"] for m in history]


STALE_END = "LCM session-end rows not stored: session=S2 reason=unverified_stale_end view_rows={}"


def _contents(engine):
    return [r["content"] for r in engine._store.get_session_messages("S2")]


def test_fallback_stored_stale_end_has_no_unstored_warning(cold, caplog):
    engine = cold()
    caplog.clear()
    engine.on_session_end("S2", copy.deepcopy(VIEW))
    assert _contents(engine) == [m["content"] for m in VIEW]
    assert not _warnings(caplog) and not engine._pending_session_tails


@pytest.mark.parametrize("refusal", ["foreign-profile", "two-conversations"])
def test_fallback_refused_direct_end_warns_once(cold, tmp_path, caplog, refusal):
    engine = cold()
    if refusal == "foreign-profile":
        other = tmp_path / "foreign-home"
        other.mkdir()
        _start(engine, "S1", hermes_home=str(other))
    else:
        engine._store._conn.execute(
            "UPDATE messages SET conversation_id='other-conversation' WHERE session_id='S2' AND content='reply'")
        engine._store._conn.commit()
    caplog.clear()
    engine.on_session_end("S2", copy.deepcopy(VIEW))
    assert STALE in caplog.text
    assert _warnings(caplog) == [STALE_END.format(len(VIEW))]
    assert _contents(engine) == [m["content"] for m in HISTORY] and not engine._pending_session_tails


@pytest.mark.parametrize("route", ["direct", "retry"])
def test_fallback_excluded_session_has_no_unstored_warning(cold, monkeypatch, caplog, route):
    from hermes_lcm.session_patterns import compile_session_patterns
    engine = cold()
    if route == "retry":
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "ingest_atomic", busy_atomic)
            engine.on_session_end("S2", copy.deepcopy(VIEW))
        assert len(engine._pending_session_tails) == 1
    engine._compiled_ignore_session_patterns = compile_session_patterns(["cli:*"])
    caplog.clear()
    if route == "direct":
        engine.on_session_end("S2", copy.deepcopy(VIEW))
    else:
        engine._retry_pending_session_tails()
    assert "gate 1: session excluded from storage" in caplog.text
    assert not _warnings(caplog) and not engine._pending_session_tails
    assert _contents(engine) == [m["content"] for m in HISTORY]


def test_busy_direct_end_queues_quietly_and_refused_retry_warns_once_with_tail_view(cold, monkeypatch, caplog):
    engine = cold()
    caplog.clear()
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "ingest_atomic", busy_atomic)
        engine.on_session_end("S2", copy.deepcopy(VIEW))
    assert len(engine._pending_session_tails) == 1 and not _warnings(caplog)
    tail = engine._pending_session_tails[0]
    tail["view"][0]["content"] = "different history"
    tail["view"].append(dict(role="user", content="later", timestamp=4., message_uid="u4"))
    engine._retry_pending_session_tails()
    assert "gate 3:" in caplog.text
    assert _warnings(caplog) == [STALE_END.format(len(VIEW) + 1)]
    assert not engine._pending_session_tails
    assert _contents(engine) == [m["content"] for m in HISTORY]
