"""#2543: report unstored ends without changing main's stored rows or retry decisions."""
import copy
import json
import logging
from pathlib import Path
import re
import sqlite3
import time
import types

import pytest

from hermes_lcm import engine as engine_module, maintenance_scheduler
from hermes_lcm.config import LCMConfig
from tests.test_issue_2055_ingest_ledger_writes import engines as engines
from tests.invariants.test_issue_2543_cold_inline_block import _call
from tests.invariants.test_issue_2539_offcurrent_marker_end import _setup, _changed_marker_view
from tests.invariants.test_p1e_cold_tails import fail_end_read, committed_rows
from tests.invariants.test_issue_2736_end_busy_rollback_fail import end_with_busy, tail_contents
from tests.test_issue_2729_fallback_gates import cold as cold  # noqa: F401  (fixture)
from tests.test_issue_2729_fallback_gates import HISTORY, STALE, VIEW, busy as busy_atomic
from tests.test_issue_2729_offcurrent_end_after_restart import _start


CELLS = [(shape, route) for shape in ("wrapped-base64", "data-uri-key")
         for route in ("cold-pairwise", "cold-counted")]
CELLS += [(location, route) for location in ("spillover", "hermes-results")
          for route in ("direct", "busy")]


@pytest.mark.parametrize("change", ["stamp", "uid", "unchanged"])
@pytest.mark.parametrize("failure", ["stores", "append-interrupt", "busy"])
def test_occurrence_change_below_cursor_warns(engines, monkeypatch, caplog, record_property, change, failure):
    engine = engines()
    history = _precision_history()[:4]
    for i, row in enumerate(history):
        row["message_uid"] = f"u-{i + 1}"
    engine.ingest(copy.deepcopy(history))
    before = engine._store.get_session_messages("S")
    assert engine._ingest_cursor == 4 and not engine._ingest_cursor_needs_reconcile
    end = copy.deepcopy(history)
    if change == "stamp":
        end[-1]["timestamp"] = 9.0
    elif change == "uid":
        end[-1]["message_uid"] = "u-9"
    assert [engine._message_replay_identity(m, strip_carrier=False) for m in end] == (
        engine._last_active_replay_source_identities)
    append_calls = []

    def interrupt(batch):
        append_calls.append(len(batch["messages"]))
        raise KeyboardInterrupt

    def busy(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    caplog.clear()
    with monkeypatch.context() as patch:
        if failure == "append-interrupt":
            patch.setattr(engine, "_append_ingest_batch", interrupt)
        elif failure == "busy":
            patch.setattr(engine, "_ingest_messages", busy)
        assert engine.on_session_end("S", copy.deepcopy(end)) is None
    changed = change != "unchanged"
    if failure == "stores":
        expected = [m["content"] for m in history] + ([end[-1]["content"]] if changed else [])
        assert _record_precision_rows(engine, record_property) == expected
        assert not _warnings(caplog)
    else:
        reason = "foreground_ingest_interrupted" if failure == "append-interrupt" else "foreground_ingest_busy_unqueued"
        expected = [f"LCM session-end rows not stored: session=S reason={reason} view_rows=4"] if changed else []
        assert _warnings(caplog) == expected, f"occurrence={change}: {reason} view_rows=4 expected={expected}"
        assert engine._store.get_session_messages("S") == before
        assert not engine._pending_session_tails
        if failure == "append-interrupt":
            assert append_calls == ([1] if changed else [])
        _record_precision_rows(engine, record_property)


@pytest.mark.parametrize("rollback_fails", [True, False])
def test_frozen_tail_requires_pending_warm_tail(engines, monkeypatch, caplog, record_property, rollback_fails):
    engine = engines()
    caplog.clear()
    # The #2736 helper uses the real RollbackProxy: rollback runs, then raises.
    end_with_busy(engine, monkeypatch, rollback_fails=rollback_fails)
    assert engine.__dict__.get("_frozen_tail_reached")
    assert _record_precision_rows(engine, record_property) == ["A", "B"]
    if rollback_fails:
        assert not engine._pending_session_tails and not tail_contents(engine)
        assert "SQLite rollback failed" in caplog.text
        expected = ["LCM session-end rows not stored: session=S "
                    "reason=foreground_ingest_busy_unqueued view_rows=4"]
        assert _warnings(caplog) == expected, "failed rollback: foreground_ingest_busy_unqueued view_rows=4 required"
    else:
        assert any(t["kind"] == "warm" and t["session_id"] == "S" and t["store"] is engine._store
                   for t in engine._pending_session_tails)
        assert tail_contents(engine) == ["C", "D"]
        assert not _warnings(caplog)



@pytest.mark.parametrize("other", ["other-session", "other-store"])
def test_stale_frozen_flag_with_unrelated_warm_tail_warns(engines, monkeypatch, caplog, other):
    """Only a warm tail of this session and store stands in for the end's own queued tail."""
    engine = engines()
    store = engine._store if other == "other-session" else types.SimpleNamespace(db_path="/nonexistent/other.db")
    engine._pending_session_tails.append(dict(
        kind="warm", store=store, session_id="X" if other == "other-session" else "S",
        conversation_id=engine._conversation_id, after_store_id=0, source="cli",
        messages=[{"role": "user", "content": "old-tail"}], estimates=[1],
        metadata_messages=[{"role": "user", "content": "old-tail"}]))
    monkeypatch.setattr(engine, "_retry_session_end_tails", lambda *a, **k: None)
    caplog.clear()
    end_with_busy(engine, monkeypatch, rollback_fails=True)
    assert engine.__dict__.get("_frozen_tail_reached")
    assert _warnings(caplog) == ["LCM session-end rows not stored: session=S "
                                 "reason=foreground_ingest_busy_unqueued view_rows=4"], other

def _precision_history():
    return [{"role": role, "content": content, "timestamp": float(i + 1)}
            for i, (role, content) in enumerate([
                ("user", "q1"), ("assistant", "a1"), ("user", "q2"),
                ("assistant", "partial answer"), ("user", "q3"), ("assistant", "a3")])]


def _record_precision_rows(engine, record_property):
    rows = engine._store.get_session_messages("S")
    # Compare durable identity, order, ownership and content across the two trees;
    # wall-clock bookkeeping is intentionally not a cross-run equality claim.
    keys = ("store_id", "session_id", "conversation_id", "role", "content", "observed_at",
            "tool_call_id", "tool_calls")
    record_property("precision_rows", json.dumps([{k: r.get(k) for k in keys} for r in rows], sort_keys=True))
    return [r["content"] for r in rows]


@pytest.mark.parametrize("failure", ["stores", "interrupt", "busy", "real-interrupt",
                                     "outer-interrupt", "outer-busy", "identity-mutated"])
def test_rewritten_final_row_with_zero_unflushed_warns(engines, monkeypatch, caplog, record_property, failure):
    engine = engines()
    history = _precision_history()[:4]
    engine.ingest(copy.deepcopy(history))
    before = engine._store.get_session_messages("S")
    assert engine._ingest_cursor == 4 and not engine._ingest_cursor_needs_reconcile
    end = copy.deepcopy(history)
    end[-1]["content"] = "partial answer, now the full final answer"

    def fail(*args, **kwargs):
        if failure == "identity-mutated":
            # Ingest can slice-assign this list; the failure guard needs its entry copy.
            engine._last_active_replay_source_identities[:] = [
                engine._message_replay_identity(m, strip_carrier=False) for m in end]
        if failure.endswith("busy"):
            raise sqlite3.OperationalError("database is locked")
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        if failure.startswith("outer-"):
            patch.setattr(engine_module, "_temporary_sqlite_busy_timeout", fail)
        elif failure == "real-interrupt":
            patch.setattr(engine._store, "ingest_atomic", fail)
        elif failure != "stores":
            patch.setattr(engine, "_ingest_messages", fail)
        caplog.clear()
        assert engine.on_session_end("S", copy.deepcopy(end)) is None
    if failure == "stores":
        assert _record_precision_rows(engine, record_property) == [m["content"] for m in history] + [end[-1]["content"]]
        assert not _warnings(caplog)
    else:
        reason = ("foreground_flush_" if failure.startswith("outer-") else "foreground_ingest_") + (
            "locked" if failure == "outer-busy" else "busy_unqueued" if failure == "busy" else "interrupted")
        assert _warnings(caplog) == [f"LCM session-end rows not stored: session=S reason={reason} view_rows=4"]
        assert engine._store.get_session_messages("S") == before
        _record_precision_rows(engine, record_property)
        assert not engine._pending_session_tails and "unflushed=0" in caplog.text


class _PrecisionIgnorePattern:
    # Optional regex's search(text, timeout=) API, without requiring that package.
    pattern = "IGNORE_THIS"

    def search(self, text, timeout=None):
        return re.search(self.pattern, text)


def _precision_engine(tmp_path, shape):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), sensitive_patterns_enabled=shape == "redacted")
    engine = engine_module.LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    if shape == "ignored":
        engine._compiled_ignore_message_patterns = [_PrecisionIgnorePattern()]
    engine.on_session_start("S", platform="cli", context_length=200_000)
    return engine


@pytest.mark.parametrize("shape", ["default", "ignored", "redacted"])
@pytest.mark.parametrize("repeat", [False, True])
def test_restart_busy_own_tail_and_identical_repeat_are_quiet(tmp_path, monkeypatch, caplog,
                                                            record_property, shape, repeat):
    _precision_busy_tail(tmp_path, monkeypatch, caplog, record_property, shape, repeat)


def test_new_tail_count_independently_suppresses_warning(tmp_path, monkeypatch, caplog, record_property):
    # Isolate the count guard from the independent warm-tail marker guard.
    _precision_busy_tail(tmp_path, monkeypatch, caplog, record_property, "default", False, clear_marker=True)


def test_known_limit_older_filtered_tail_failure_before_queue(tmp_path, monkeypatch, caplog, record_property):
    # Accepted log-only limit: the old filtered metadata cannot prove raw-view coverage.
    _precision_busy_tail(tmp_path, monkeypatch, caplog, record_property, "ignored", True, fail_before_queue=True)


def _precision_busy_tail(tmp_path, monkeypatch, caplog, record_property, shape, repeat,
                         clear_marker=False, fail_before_queue=False):
    for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
        monkeypatch.setattr(engine_module.LCMEngine, name, lambda *a, **k: None)
    history = _precision_history()
    if shape == "ignored":
        history[4]["content"] = "IGNORE_THIS heartbeat"
    if shape == "redacted":
        history[5]["content"] = "set password=synthetic-secret now"
    engine = _precision_engine(tmp_path, shape)
    engine.ingest(copy.deepcopy(history[:4]))
    if not fail_before_queue:
        engine.shutdown()
        engine = _precision_engine(tmp_path, shape)
    try:
        if fail_before_queue:
            assert engine._ingest_cursor == 4 and not engine._ingest_cursor_needs_reconcile
        else:
            assert engine._ingest_cursor == 0 and engine._ingest_cursor_needs_reconcile
        ingest = engine._ingest_messages

        def without_marker(*args, **kwargs):
            try:
                return ingest(*args, **kwargs)
            finally:
                engine.__dict__["_frozen_tail_reached"] = False

        def busy_before_queue(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        locker = sqlite3.connect(str(engine._db_path), timeout=5, isolation_level=None)
        try:
            locker.execute("BEGIN IMMEDIATE")
            with monkeypatch.context() as patch:
                if clear_marker:
                    patch.setattr(engine, "_ingest_messages", without_marker)
                caplog.clear()
                assert engine.on_session_end("S", copy.deepcopy(history)) is None
                assert len(engine._pending_session_tails) == 1
                assert not _warnings(caplog)
                tail = engine._pending_session_tails[0]
                assert tail["kind"] == "warm"
                if repeat:
                    if fail_before_queue:
                        patch.setattr(engine, "_ingest_messages", busy_before_queue)
                    caplog.clear()
                    assert engine.on_session_end("S", copy.deepcopy(history)) is None
                    assert len(engine._pending_session_tails) == 1 and engine._pending_session_tails[0] is tail
                    expected = ["LCM session-end rows not stored: session=S "
                                "reason=foreground_ingest_busy_unqueued view_rows=6"] if fail_before_queue else []
                    assert _warnings(caplog) == expected
                assert "raw-message ingest skipped due to SQLite lock" in caplog.text
        finally:
            locker.rollback()
            locker.close()
        engine._retry_pending_session_tails()
        assert not engine._pending_session_tails
        contents = _record_precision_rows(engine, record_property)
        if shape == "ignored":
            assert contents == [m["content"] for i, m in enumerate(history) if i != 4]
        elif shape == "redacted":
            assert contents == [m["content"] for m in history[:5]] + [
                "set password=[LCM sensitive redaction: name=password_assignment; chars=16; bytes=16] now"]
        else:
            assert contents == [m["content"] for m in history]
    finally:
        engine.shutdown()


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


@pytest.mark.parametrize("outer,error,main_line", [
    (False, KeyboardInterrupt, "raw-message ingest interrupted;"),
    (False, sqlite3.OperationalError, "raw-message ingest skipped due to SQLite lock after short wait;"),
    (True, KeyboardInterrupt, "ingest/finalize interrupted before bounded flush completed"),
    (True, sqlite3.OperationalError, "ingest/finalize skipped due to SQLite lock before bounded flush:"),
])
def test_fully_stored_foreground_failure_is_quiet(engines, monkeypatch, caplog, outer, error, main_line):
    engine = engines()
    engine.on_session_start("S", platform="cli", context_length=200_000)
    view = [{"role": "user", "content": "question", "timestamp": 1.0},
            {"role": "assistant", "content": "answer", "timestamp": 2.0}]
    engine.ingest(copy.deepcopy(view))
    before = engine._store.get_session_messages("S")
    assert engine._ingest_cursor == len(view) and not engine._ingest_cursor_needs_reconcile

    def fail(*args, **kwargs):
        raise error("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(engine_module if outer else engine,
                      "_temporary_sqlite_busy_timeout" if outer else "_ingest_messages", fail)
        caplog.clear()
        assert engine.on_session_end("S", copy.deepcopy(view)) is None
    assert not _warnings(caplog)
    assert f"LCM session-end {main_line}" in caplog.text and "unflushed=0" in caplog.text
    assert engine._store.get_session_messages("S") == before and not engine._pending_session_tails


def test_unknown_unflushed_foreground_failure_still_warns(engines, monkeypatch, caplog):
    engine = engines()
    engine.on_session_start("S", platform="cli", context_length=200_000)
    view = [{"role": "assistant", "content": "answer", "timestamp": 1.0}]
    engine.ingest(copy.deepcopy(view))
    engine._ingest_cursor_needs_reconcile = True

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ingest_messages", fail)
        caplog.clear()
        engine.on_session_end("S", copy.deepcopy(view))
    assert _warnings(caplog) == [
        "LCM session-end rows not stored: session=S reason=foreground_ingest_busy_unqueued view_rows=1"]
    assert "unflushed=unknown" in caplog.text


@pytest.mark.parametrize("case", ["stored", "restamped", "unstored_suffix"])
def test_off_current_bypass_end_warns_even_when_its_prefix_is_stored(engines, caplog, case):
    """The stored-prefix count compares content, not stamps, so it cannot prove a re-stamped row stored."""
    from tests.invariants.test_p1e_cold_tails import off_current_end
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    view = off_current_end(engine)[:2]
    engine._mark_lcm_bypass_lineage_session("S1", platform="cron")
    engine._remember_lcm_bypass_message_prefix("S1", copy.deepcopy(view))
    assert engine._matching_lcm_bypass_prefix_evidence("S1", view)[0] == len(view)
    assert engine._session_end_store_prefix_count("S1", view, conversation_id="conv1") == len(view)
    before = engine._store.get_session_messages("S1")
    if case == "restamped":
        view[-1] = dict(view[-1], timestamp=float(view[-1].get("timestamp") or 0) + 3600.0)
        assert engine._session_end_store_prefix_count("S1", view, conversation_id="conv1") == len(view)
    if case == "unstored_suffix":
        view.append({"role": "assistant", "content": "FINAL", "timestamp": 5.0})
    caplog.clear()
    assert engine.on_session_end("S1", copy.deepcopy(view)) is None
    assert _warnings(caplog) == ["LCM session-end rows not stored: session=S1 "
                                 f"reason=prefix_mismatch_bypass_lineage view_rows={len(view)}"]
    assert engine._store.get_session_messages("S1") == before and not engine._pending_session_tails


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


@pytest.mark.parametrize("tail_kind", ["warm", "cold"])
@pytest.mark.parametrize("new_rows", [False, True])
def test_repeated_busy_end_requires_covering_pending_tail(engines, monkeypatch, caplog, tail_kind, new_rows):
    engine = engines()
    engine.on_session_start("S", platform="cli", context_length=200_000)
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"t-{i}", "timestamp": float(i + 1)}
               for i in range(8)]
    engine.ingest(copy.deepcopy(history[:4]))
    before = engine._store.get_session_messages("S")

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    locker = sqlite3.connect(str(engine._db_path), timeout=0, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with monkeypatch.context() as patch:
            if tail_kind == "cold":
                patch.setattr(engine._store, "get_session_count", fail)
                engine._ingest_cursor_needs_reconcile = True
            engine.on_session_end("S", copy.deepcopy(history[:6]))
        assert len(engine._pending_session_tails) == 1
        tail = copy.deepcopy({k: v for k, v in engine._pending_session_tails[0].items() if k != "store"})
        assert tail["kind"] == tail_kind
        # Fail before another tail can be queued: only the older tail can cover this end.
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_ingest_messages", fail)
            caplog.clear()
            view = history if new_rows else history[:6]
            assert engine.on_session_end("S", copy.deepcopy(view)) is None
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    expected = ["LCM session-end rows not stored: session=S "
                "reason=foreground_ingest_busy_unqueued view_rows=8"] if new_rows else []
    assert _warnings(caplog) == expected
    assert "raw-message ingest skipped due to SQLite lock after short wait;" in caplog.text
    assert engine._store.get_session_messages("S") == before
    assert [{k: v for k, v in t.items() if k != "store"} for t in engine._pending_session_tails] == [tail]
    engine._retry_pending_session_tails()
    assert [r["content"] for r in engine._store.get_session_messages("S")] == [m["content"] for m in history[:6]]


def test_operator_guide_explains_warning_uncertainty_and_all_reasons():
    guide = (Path(__file__).resolve().parents[1] / "docs/operator-guide.md").read_text()
    paragraph = guide.split("The WARNING `LCM session-end rows not stored`", 1)[1].split(
        "The interrupted or SQLite-lock raw-flush warning", 1)[0]
    assert "rows may be missing, or their storage could not be verified" in paragraph
    assert "every row is already stored" in paragraph
    reasons = dict(re.findall(r"\| `([a-z_]+)` \| (.+) \|", paragraph))
    assert set(reasons) == {
        "prefix_mismatch_bypass_lineage", "bypass_lineage_end", "off_current_suffix_append_failed",
        "off_current_suffix_append_interrupted", "unverified_stale_end", "off_current_fallback_interrupted",
        "foreground_ingest_interrupted", "foreground_ingest_busy_unqueued", "foreground_flush_interrupted",
        "foreground_flush_locked", "cold_tail_prefix_mismatch", "cold_tail_scan_capped",
        "cold_tail_conflicting_stamp_unplaced", "pending_tail_dropped_at_shutdown", "pending_tail_dropped_at_exit",
    }
    for reason in ("off_current_suffix_append_failed", "off_current_suffix_append_interrupted"):
        assert "plain off-current" in reasons[reason] and "bypass-lineage" in reasons[reason]
    assert all(len(meaning.split()) >= 4 for meaning in reasons.values())


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


def test_fallback_direct_exception_warns_once(cold, monkeypatch, caplog):
    engine = cold()

    def fails(*args, **kwargs):
        raise RuntimeError("injected compare failure")

    monkeypatch.setattr(engine, "_off_current_fallback_prefix", fails)
    caplog.clear()
    engine.on_session_end("S2", copy.deepcopy(VIEW))
    assert "gate 3: compare failed" in caplog.text
    assert _warnings(caplog) == [STALE_END.format(len(VIEW))]
    assert _contents(engine) == [m["content"] for m in HISTORY] and not engine._pending_session_tails


def test_fallback_inline_gate4_refusal_warns_once(cold, monkeypatch, caplog):
    from contextlib import contextmanager
    engine = cold()
    store, original, entered = engine._store, engine._store.ingest_atomic, []

    @contextmanager
    def changed_between_reads(**kwargs):  # Outer gates passed; gate 4 rereads ownership under the lock.
        entered.append(1)
        store._conn.execute("UPDATE ledger_meta SET value='foreign:S2' WHERE key='session_lineage:S2'")
        store._conn.commit()
        with original(**kwargs) as timing:
            yield timing

    caplog.clear()
    with monkeypatch.context() as patch:
        patch.setattr(store, "ingest_atomic", changed_between_reads)
        engine.on_session_end("S2", copy.deepcopy(VIEW))
    assert entered == [1] and caplog.text.count(STALE) == 1
    assert _warnings(caplog) == [STALE_END.format(len(VIEW))]
    assert _contents(engine) == [m["content"] for m in HISTORY] and not engine._pending_session_tails


def _interrupt(*args, **kwargs):
    raise KeyboardInterrupt


@pytest.mark.parametrize("site", ["_off_current_fallback_prefix", "_append_off_current_session_end_suffix"])
def test_fallback_interrupt_warns_once_and_propagates(cold, monkeypatch, caplog, site):
    # Gate-3 compare and gate-4 append: the interrupt still propagates (#2543 is log-only).
    engine = cold()
    caplog.clear()
    with monkeypatch.context() as patch:
        patch.setattr(engine, site, _interrupt)
        with pytest.raises(KeyboardInterrupt):
            engine.on_session_end("S2", copy.deepcopy(VIEW))
    assert _warnings(caplog) == [
        f"LCM session-end rows not stored: session=S2 reason=off_current_fallback_interrupted view_rows={len(VIEW)}"]
    assert _contents(engine) == [m["content"] for m in HISTORY] and not engine._pending_session_tails


def test_bypass_lineage_suffix_append_interrupt_warns_once_and_propagates(engines, monkeypatch, caplog):
    from tests.invariants.test_issue_2738_bypass_lineage_busy_suffix import setup, contents
    engine, view = setup(engines)
    caplog.clear()
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_off_current_session_end_suffix", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            engine.on_session_end("S1", copy.deepcopy(view))
    assert _warnings(caplog) == [
        f"LCM session-end rows not stored: session=S1 reason=off_current_suffix_append_interrupted view_rows={len(view)}"]
    assert contents(engine) == ["A", "B"] and not engine._pending_session_tails
