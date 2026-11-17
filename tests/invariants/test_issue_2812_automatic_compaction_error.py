"""Refs #2812: an automatic compaction error preserves the turn and stored rows (I1/I2)."""

import copy
import hashlib
import json
import logging
import sqlite3
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
from tests.harness import make_engine, pin_char_counter


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: (
        "Earlier turns.\nExpand for details about: synthetic turns", 1))


@pytest.fixture
def engines(tmp_path):
    opened = []

    def build(name="store"):
        path = tmp_path / name
        path.mkdir()
        engine = make_engine(path, window=200_000, conversation_id="conv", fresh_tail_count=1,
                             leaf_chunk_tokens=1, temporal_rollups_enabled=False)
        engine._config.fresh_tail_pressure_yield_enabled = False
        engine._config.empty_lifecycle_gc_enabled = False
        opened.append(engine)
        return engine

    yield build
    for engine in opened:
        engine.shutdown()


def turns(start, stop):
    return [row for turn in range(start, stop) for row in (
        {"role": "user", "content": f"synthetic question {turn}", "timestamp": 1000.0 + turn},
        {"role": "assistant", "content": f"synthetic answer {turn}", "timestamp": 2000.0 + turn},
    )]


def digest(engine):
    rows = engine._store.get_session_messages(engine._session_id, limit=1000)
    fields = ("store_id", "session_id", "conversation_id", "role", "content", "timestamp", "observed_at",
              "tool_calls", "tool_call_id")
    content = [{key: row.get(key) for key in fields} for row in rows]
    return len(rows), hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("fit_fails", [False, True], ids=["already_fits", "fit_raises"])
def test_automatic_fallback_records_error_and_manual_still_raises(engines, monkeypatch, caplog, fit_fails):
    engine, view = engines(), turns(0, 3)
    engine.ingest(view)
    before, original = digest(engine), copy.deepcopy(view)
    rollback = Mock(wraps=engine._store.rollback_pending_write)
    monkeypatch.setattr(engine._store, "rollback_pending_write", rollback)

    def locked(*args, **kwargs):
        # An unfinished write must roll back before the host adopts the unchanged list.
        engine._store._conn.execute("UPDATE messages SET content = 'pending synthetic write'")
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "_compress_impl", locked)
    if fit_fails:
        monkeypatch.setattr(engine, "_survival_fit", Mock(side_effect=RuntimeError("synthetic fit failure")))
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        assert engine.compress(view, current_tokens=10_000) is view
    assert view == original and digest(engine) == before
    rollback.assert_called_once_with()
    assert engine._last_compression_status == "error"
    assert engine._survival_fit_reason == "exception:OperationalError"
    lines = [record.getMessage() for record in caplog.records
             if record.getMessage().startswith("LCM automatic compress failed")]
    assert lines == ["LCM automatic compress failed (OperationalError); returning the host messages"]
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        engine.compress(view, current_tokens=10_000, force=True)
    assert rollback.call_count == 2 and digest(engine) == before


@pytest.mark.parametrize("fit_fails", [True, False], ids=["fit_raises", "fit_unchanged"])
def test_over_window_automatic_failure_reraises_original_exception(engines, monkeypatch, caplog, fit_fails):
    engine = engines()
    engine.context_length = 100
    view = [{"role": "user", "content": "synthetic question " * 100, "timestamp": 1000.0}]
    engine.ingest(view)
    before, original = digest(engine), copy.deepcopy(view)
    measured = engine._survival_measure(view)
    assert measured > engine.context_length
    assert measured > engine._survival_fit_budget(view, measured)
    failure = sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(engine, "_compress_impl", Mock(side_effect=failure))
    fit = Mock(side_effect=RuntimeError("synthetic fit failure")) if fit_fails else Mock(return_value=view)
    monkeypatch.setattr(engine, "_survival_fit", fit)
    rollback = Mock(wraps=engine._store.rollback_pending_write)
    monkeypatch.setattr(engine._store, "rollback_pending_write", rollback)

    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        with pytest.raises(sqlite3.OperationalError, match="database is locked") as caught:
            engine.compress(view, current_tokens=measured)

    assert caught.value is failure
    fit.assert_called_once()
    assert fit.call_args.kwargs["after_exception"] is True
    rollback.assert_called_once_with()
    assert engine._last_compression_status == "error"
    assert engine._survival_fit_reason == "exception:OperationalError"
    assert view == original and digest(engine) == before
    assert not any(record.getMessage().startswith("LCM automatic compress failed") for record in caplog.records)


def test_next_automatic_pass_runs_normally(engines, monkeypatch):
    engine, view = engines(), turns(0, 3)
    engine.ingest(view)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_compress_impl", Mock(side_effect=sqlite3.OperationalError("database is locked")))
        assert engine.compress(view, current_tokens=10_000) is view
    view += turns(3, 4)
    engine.ingest(view)
    scan = Mock(wraps=engine._store_complete_owned_rows)
    monkeypatch.setattr(engine, "_store_complete_owned_rows", scan)
    engine.compress(view, current_tokens=10_001)
    assert scan.call_count > 0
    assert engine._last_compress_leaves[1] >= 1
    assert engine._last_compression_status != "error"
    assert not engine._retained_stop_hold_active()


def test_failed_and_control_turn_sequences_store_identical_rows(engines, monkeypatch):
    # Store timestamps are write times; give both independent runs the same clock.
    monkeypatch.setattr(engine_module.time, "time", lambda: 1_700_000_000.0)
    failed, control = engines("failed"), engines("control")
    views = [turns(0, 3), turns(0, 3)]
    for engine, view in zip((failed, control), views):
        engine.ingest(view)
    with monkeypatch.context() as patch:
        patch.setattr(failed, "_compress_impl", Mock(side_effect=sqlite3.OperationalError("database is locked")))
        views[0] = failed.compress(views[0], current_tokens=10_000)
    views[1] = control.compress(views[1], current_tokens=10_000)
    assert digest(failed) == digest(control)
    for index, engine in enumerate((failed, control)):
        engine.note_turn_complete()
        views[index] += turns(3, 4)
        engine.ingest(views[index])
        engine.compress(views[index], current_tokens=10_001)
    assert digest(failed)[0] == 8
    assert digest(failed) == digest(control)


def test_automatic_fallback_warning_has_traceback(engines, monkeypatch, caplog):
    engine, view = engines(), turns(0, 3)
    failure = sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(engine, "_compress_impl", Mock(side_effect=failure))
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        assert engine.compress(view, current_tokens=10_000) is view
    records = [record for record in caplog.records
               if record.getMessage().startswith("LCM automatic compress failed")]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is failure and records[0].exc_info[2] is not None


@pytest.mark.parametrize("early", [True, False], ids=["before_flag", "after_flag"])
def test_forced_overflow_error_reraises_original(engines, monkeypatch, early):
    engine, view = engines(), turns(0, 3)
    engine.ingest(view)
    assert engine._survival_measure(view) <= engine._survival_fit_budget(view, 10_000)
    monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **kwargs: True)
    failure = sqlite3.OperationalError("database is locked")
    target = "_rebind_after_unadopted_compaction_commit" if early else "_prepare_retained_user_anchor"
    failing_step = Mock(side_effect=failure)
    monkeypatch.setattr(engine, target, failing_step)
    with pytest.raises(sqlite3.OperationalError) as caught:
        engine.compress(view, current_tokens=10_000)
    assert caught.value is failure
    failing_step.assert_called_once()
    assert engine._last_compression_status == "error"


def test_rollback_failure_reraises_original(engines, monkeypatch, caplog):
    engine, view = engines(), turns(0, 3)
    engine.ingest(view)
    before = digest(engine)
    failure = sqlite3.OperationalError("database is locked")

    def locked(*args, **kwargs):
        engine._store._conn.execute("UPDATE messages SET content = 'pending synthetic write'")
        raise failure

    fit = Mock(wraps=engine._survival_fit)
    rollback = Mock(side_effect=sqlite3.OperationalError("synthetic rollback failure"))
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_compress_impl", locked)
        patch.setattr(engine, "_survival_fit", fit)
        patch.setattr(engine._store, "rollback_pending_write", rollback)
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
                with pytest.raises(sqlite3.OperationalError) as caught:
                    engine.compress(view, current_tokens=10_000)
            assert caught.value is failure
            assert engine._last_compression_status == "error"
            rollback.assert_called_once_with()
            fit.assert_not_called()
            assert engine._store._conn.in_transaction
            assert not any(record.getMessage().startswith("LCM automatic compress failed")
                           for record in caplog.records)
        finally:
            # The existing handler leaves this failed rollback to its caller; keep the fixture clean.
            engine._store._conn.rollback()
    assert digest(engine) == before
