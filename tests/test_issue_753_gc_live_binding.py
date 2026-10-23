"""#753: empty-lifecycle GC must keep the lifecycle row of a session another live engine in this
process has bound but not ingested into yet; the live session then keeps compacting.

Real engine/store/lifecycle/command code paths; only the summarizer model call is mocked.
"""
import time

import hermes_lcm.engine as lcm_engine
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engines(tmp_path, **overrides):
    cfg = LCMConfig(database_path=str(tmp_path / "lcm.db"), **overrides)
    home = str(tmp_path / "home")
    a = LCMEngine(config=cfg, hermes_home=home)
    b = LCMEngine(config=cfg, hermes_home=home)
    a.on_session_start("issue753-live", platform="cli", context_length=200_000)  # live, nothing stored yet
    return a, b


def _age_all_rows(engine, hours):
    stamp = time.time() - hours * 3600
    engine._lifecycle._conn.execute(
        "UPDATE lcm_lifecycle_state SET current_bound_at = ?, updated_at = ?", (stamp, stamp)
    )
    engine._lifecycle._conn.commit()


def _host_turns(n):
    host = [{"role": "system", "content": "sys"}]
    for i in range(n):
        host += [
            {"role": "user", "content": f"turn {i} " * 80},
            {"role": "assistant", "content": f"noted {i}"},
        ]
    return host


def test_automatic_gc_keeps_live_binding_of_other_engine_older_than_age_guard(tmp_path):
    # default threshold is 200, default age guard 24 h; threshold lowered so the GC fires
    a, b = _engines(tmp_path, empty_lifecycle_gc_threshold=1)
    try:
        _age_all_rows(a, 25)  # A was first bound 25 h ago and has not stored a message yet
        b.on_session_start("issue753-trigger", platform="cli", context_length=200_000)  # runs the automatic GC
        assert a._lifecycle.get_by_session("issue753-live") is not None, "the live binding was deleted"
    finally:
        a.shutdown()
        b.shutdown()


def test_doctor_clean_apply_keeps_live_binding_of_other_engine(tmp_path):
    a, b = _engines(tmp_path, doctor_clean_apply_enabled=True)  # A bound seconds ago
    try:
        b.on_session_start("issue753-trigger", platform="cli", context_length=200_000)
        result = handle_lcm_command("doctor clean lifecycle apply", b)
        assert "status: ok" in result, result
        assert a._lifecycle.get_by_session("issue753-live") is not None, result
    finally:
        a.shutdown()
        b.shutdown()


def test_live_session_still_compacts_after_other_engine_runs_doctor_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: ("summary", 1))
    a, b = _engines(
        tmp_path,
        doctor_clean_apply_enabled=True,
        fresh_tail_count=6,
        leaf_chunk_tokens=400,
        large_output_externalization_path=str(tmp_path / "ext"),
    )
    try:
        b.on_session_start("issue753-trigger", platform="cli", context_length=200_000)
        handle_lcm_command("doctor clean lifecycle apply", b)
        host = _host_turns(12)
        a.ingest(host)
        a.compress(list(host), force=True)  # raised RuntimeError at the issue sha
        assert a._last_compacted_store_id > 0
    finally:
        a.shutdown()
        b.shutdown()


def test_live_session_still_compacts_after_automatic_gc_prunes_aged_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: ("summary", 1))
    a, b = _engines(
        tmp_path,
        empty_lifecycle_gc_threshold=1,
        fresh_tail_count=6,
        leaf_chunk_tokens=400,
        large_output_externalization_path=str(tmp_path / "ext"),
    )
    try:
        _age_all_rows(a, 25)
        b.on_session_start("issue753-trigger", platform="cli", context_length=200_000)
        host = _host_turns(12)
        a.ingest(host)
        # repeated passes with more turns ingested between them: the wedge persists
        for extra in range(3):
            host = host + _host_turns(2)[1:]
            a.ingest(host)
            a.compress(list(host), force=True)
        assert a._last_compacted_store_id > 0
    finally:
        a.shutdown()
        b.shutdown()


def test_control_crash_orphan_without_live_engine_is_still_collected(tmp_path):
    # #256 intent: rows bound by no live engine (crash orphans) are still collected
    cfg = LCMConfig(database_path=str(tmp_path / "lcm.db"), empty_lifecycle_gc_threshold=1)
    home = str(tmp_path / "home")
    gone = LCMEngine(config=cfg, hermes_home=home)
    gone.on_session_start("issue753-crashed", platform="cli", context_length=200_000)
    gone.shutdown()  # unregisters; models a process that died
    b = LCMEngine(config=cfg, hermes_home=home)
    try:
        _age_all_rows(b, 25)
        b.on_session_start("issue753-trigger", platform="cli", context_length=200_000)
        assert b._lifecycle.get_by_session("issue753-crashed") is None
    finally:
        b.shutdown()


def test_doctor_clean_lifecycle_scan_reports_live_binding_of_other_engine_as_protected(tmp_path):
    a, b = _engines(tmp_path)
    try:
        b.on_session_start("issue753-trigger", platform="cli", context_length=200_000)
        result = handle_lcm_command("doctor clean lifecycle", b)
        # A (other live clone) and B (caller) are both protected, so neither is a candidate
        assert "status: ok" in result, result
        assert "empty_rows: 0" in result, result
    finally:
        a.shutdown()
        b.shutdown()


def test_control_doctor_apply_still_deletes_row_of_shut_down_engine(tmp_path):
    cfg = LCMConfig(database_path=str(tmp_path / "lcm.db"), doctor_clean_apply_enabled=True)
    home = str(tmp_path / "home")
    gone = LCMEngine(config=cfg, hermes_home=home)
    gone.on_session_start("issue753-gone", platform="cli", context_length=200_000)
    gone.shutdown()
    b = LCMEngine(config=cfg, hermes_home=home)
    try:
        b.on_session_start("issue753-trigger", platform="cli", context_length=200_000)
        result = handle_lcm_command("doctor clean lifecycle apply", b)
        assert "lifecycle_rows_deleted: 1" in result, result
        assert b._lifecycle.get_by_session("issue753-gone") is None
        assert b._lifecycle.get_by_session("issue753-trigger") is not None
    finally:
        b.shutdown()
