"""Refs #2815: pace identical retained stops without changing durable rows (I1/I2/I11)."""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.tools as lcm_tools
from hermes_lcm.engine import LCMEngine
from tests.harness import make_engine, pin_char_counter
from tests.test_issue_1162_retained_escape import _drop_r, _reorder, _rows


@pytest.fixture
def clock(monkeypatch):
    now = [engine_module.time.monotonic()]
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: now[0])
    return now


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: (
        "Earlier turns.\nExpand for details about: synthetic turns", 1))


@pytest.fixture
def stuck(tmp_path, monkeypatch):
    # Reuse the real legacy retained stop; the escape/hole unwedge is a separate lane.
    monkeypatch.setattr(LCMEngine, "_retained_escape_enabled", False)
    engines = []

    def build(name="store", sweep=False):
        (tmp_path / name).mkdir()
        engine, view = _reorder(tmp_path / name, sweep)
        engines.append(engine)
        engine._config.fresh_tail_pressure_yield_enabled = False
        engine._config.empty_lifecycle_gc_enabled = False
        view = engine.compress(view, current_tokens=10_000)  # publish the valid prefix before R
        assert engine._last_compress_leaves[1] >= 1
        clear_old_hold(engine)
        return engine, view

    yield build
    for engine in engines:
        engine.shutdown()


def clear_old_hold(engine):
    """Isolate the new pacing from the independently tested, unchanged 600 s hold."""
    engine._clear_persisted_no_progress_hold()
    engine._no_progress_hold = None


def retained_pass(engine, view, tokens=10_000, **kwargs):
    clear_old_hold(engine)
    result = engine.compress(view, current_tokens=tokens, **kwargs)
    assert engine._last_compress_leaves[1] == 0
    assert engine._last_compression_noop_reason.startswith(
        "leaf would end before an unresolved retained occurrence")
    return result


def arm(engine, view):
    view = retained_pass(engine, view)
    assert not engine._retained_stop_hold_active()
    view = retained_pass(engine, view)
    assert engine._retained_stop_hold_active()
    clear_old_hold(engine)
    return view


@pytest.mark.parametrize("sweep", [False, True], ids=["single", "sweep"])
def test_backoff_skip_expiry_and_cap(stuck, clock, monkeypatch, caplog, sweep):
    caplog.set_level(logging.INFO, logger="hermes_lcm")
    engine, view = stuck(sweep=sweep)
    scan = Mock(wraps=engine._store_complete_owned_rows)
    monkeypatch.setattr(engine, "_store_complete_owned_rows", scan)
    view = arm(engine, view)
    signature = (engine._store_complete_frontier(), engine._store_complete_blocked[0], None)
    before = scan.call_count
    assert before > 0
    for _ in range(3):
        assert not engine.should_compress(10_001)
        assert engine._automatic_compression_blocked()
        assert engine._compression_block_reason() == "cooldown:lcm_retained_stop"
        assert not engine.should_compress_preflight(view)
        assert engine.compress(view, current_tokens=10_001) == view
        assert engine._last_compression_status == "sanitized"
        clear_old_hold(engine)
    assert scan.call_count == before, "a held turn repeated the store walk"
    waits = [600, 1200, 2400, 3600, 3600]
    for wait in waits:
        start = clock[0]
        clock[0] = start + wait - 1
        assert engine._retained_stop_hold_active()
        clock[0] += 1
        assert not engine._retained_stop_hold_active()
        assert engine.should_compress(10_002)
        view = retained_pass(engine, view, 10_002)
        assert scan.call_count > before
        before = scan.call_count
        assert engine._retained_stop_hold_active()
        clear_old_hold(engine)
    starts = [record.getMessage() for record in caplog.records
              if record.getMessage().startswith("LCM retained-stop back-off:")]
    assert len(starts) == 1 + len(waits), "skipped turns must not emit back-off logs"
    assert [int(line.split("wait=", 1)[1][:-1]) for line in starts] == [600, 1200, 2400, 3600, 3600, 3600]
    assert all(f"frontier_store_id={signature[0]} blocked_store_id={signature[1]} hole_store_id=None" in line
               for line in starts)


@pytest.mark.parametrize("change", ["frontier", "blocked", "hole", "no_hole", "conversation"])
def test_signature_and_frontier_clear(stuck, clock, change):
    engine, _view = stuck()
    frontier = engine._store_complete_frontier()
    signature = (frontier, frontier + 1, frontier + 2)
    engine._record_retained_stop(signature)
    engine._record_retained_stop(signature)
    assert engine._retained_stop_hold_active()
    if change == "frontier":
        engine._last_compacted_store_id = frontier + 1
        signature = (frontier + 1, frontier + 2, None)
    elif change == "conversation":
        engine.on_session_start("other", conversation_id="other", context_length=200_000)
        signature = (engine._store_complete_frontier(), 10, None)
    else:
        signature = (frontier, frontier + (2 if change == "blocked" else 1),
                     None if change == "no_hole" else frontier + (3 if change == "hole" else 2))
    if change in {"frontier", "conversation"}:
        assert not engine._retained_stop_hold_active()
    engine._record_retained_stop(signature)
    assert not engine._retained_stop_hold_active(), "a changed signature must start at its first observation"
    engine._record_retained_stop(signature)
    assert engine._retained_stop_hold_active()
    clock[0] += 600
    assert not engine._retained_stop_hold_active(), "a changed signature must restart at 600 s"


def test_leaf_clears_backoff(stuck, clock):
    engine, view = stuck()
    view = arm(engine, view)
    clock[0] += 600
    engine.compress(_drop_r(view), current_tokens=10_000)
    assert engine._last_compress_leaves[1] >= 1
    assert not engine._retained_stop_hold_active()
    frontier = engine._store_complete_frontier()
    engine._record_retained_stop((frontier, frontier + 1, None))
    assert not engine._retained_stop_hold_active(), "a leaf must reset the consecutive-stop counter"


def test_home_rebind_drops_retained_stop_hold(stuck, clock, tmp_path):
    engine, view = stuck()
    view = arm(engine, view)
    frontier = engine._store_complete_frontier()
    other, _ = stuck("B")
    assert other._store_complete_frontier() == frontier
    other.shutdown()
    engine._config.database_path = ""
    engine.on_session_start("S", platform="cli", context_length=200_000,
                            conversation_id="conv", hermes_home=str(tmp_path / "B"))
    clear_old_hold(engine)
    assert Path(engine._store.db_path).parent == tmp_path / "B"
    assert engine._store_complete_frontier() == frontier
    assert not engine._retained_stop_hold_active(), "home B inherited home A's retained-stop hold"
    assert engine.should_compress(10_001)


@pytest.mark.parametrize("bypass", ["ceiling", "overflow", "recovery"])
def test_emergency_paths_bypass(stuck, clock, monkeypatch, bypass):
    engine, view = stuck()
    view = arm(engine, view)
    scan = Mock(wraps=engine._store_complete_owned_rows)
    monkeypatch.setattr(engine, "_store_complete_owned_rows", scan)
    tokens, kwargs = 10_000, {}
    if bypass == "ceiling":
        tokens = engine._survival_ceiling()
    elif bypass == "overflow":
        monkeypatch.setattr(engine, "_should_force_overflow_recovery", lambda **kwargs: True)
    else:
        kwargs["bypass_cooldown"] = True
    engine.last_prompt_tokens = engine._last_gate_tokens = tokens
    assert not engine._automatic_compression_blocked(ignore_cooldown=bypass == "recovery")
    if bypass != "recovery":
        assert engine.should_compress(tokens)
    engine.compress(view, current_tokens=tokens, **kwargs)
    assert scan.call_count > 0, "emergency/recovery compaction was held"


def test_error_neither_arms_nor_clears(stuck, clock, monkeypatch):
    engine, view = stuck()

    def error_pass():
        clear_old_hold(engine)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_compress_impl", Mock(side_effect=sqlite3.OperationalError("database is locked")))
            assert engine.compress(view, current_tokens=10_000) is view
        assert engine._last_compression_status == "error"

    error_pass()
    assert not engine._retained_stop_hold_active()
    view = retained_pass(engine, view)
    assert not engine._retained_stop_hold_active()
    error_pass()
    view = retained_pass(engine, view)
    assert engine._retained_stop_hold_active(), "an error cleared the first retained observation"
    error_pass()
    assert engine._retained_stop_hold_active()
    clock[0] += 600
    retained_pass(engine, view)
    clear_old_hold(engine)
    clock[0] += 1199
    assert engine._retained_stop_hold_active(), "an error reset the growing back-off"
    clock[0] += 1
    assert not engine._retained_stop_hold_active()


def test_manual_and_restart(stuck, clock, monkeypatch):
    engine, view = stuck()
    view = arm(engine, view)
    scan = Mock(wraps=engine._store_complete_owned_rows)
    monkeypatch.setattr(engine, "_store_complete_owned_rows", scan)
    retained_pass(engine, view, force=True)
    assert scan.call_count > 0, "manual compaction was held"
    clear_old_hold(engine)
    clock[0] += 600
    assert not engine._retained_stop_hold_active(), "manual compaction grew the automatic back-off"
    # A distinct engine has no process-local pacing state, even on the same fixture store.
    engine.shutdown()
    from tests.test_issue_1880_persisted_hold import rebuild
    rebuilt = rebuild(Path(engine._store.db_path).parent)
    try:
        assert not rebuilt._retained_stop_hold_active()
        assert rebuilt.should_compress(10_001)
    finally:
        rebuilt.shutdown()


def test_identical_rows_with_and_without_backoff(stuck, clock, monkeypatch):
    paced, paced_view = stuck("paced")
    control, control_view = stuck("control")
    monkeypatch.setattr(control, "_retained_stop_hold_active", lambda: False)
    engines, views = [paced, control], [paced_view, control_view]
    passes = [0, 0]
    for turn in range(7):
        clock[0] += 601  # let the old 600 s hold expire; growing waits still pace the stuck pass
        for index, engine in enumerate(engines):
            views[index] += [
                {"role": "user", "content": f"synthetic question {turn}", "timestamp": 1000.0 + turn},
                {"role": "assistant", "content": f"synthetic answer {turn}", "timestamp": 2000.0 + turn},
            ]
            engine.ingest(views[index])
            tokens = 10_000 + turn
            if engine.should_compress(tokens):
                views[index] = engine.compress(views[index], current_tokens=tokens)
                passes[index] += 1
            engine.note_turn_complete()
        # Compare every turn, so a temporary missing/modified row cannot hide at the final sample.
        def digest(engine):
            rows = [{key: row.get(key) for key in (
                "store_id", "session_id", "conversation_id", "role", "content", "tool_calls", "tool_call_id")}
                    for row in _rows(engine)]
            return len(rows), hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        assert digest(paced) == digest(control)
    assert 0 < passes[0] < passes[1], "back-off did not reduce automatic passes"


def test_nested_retained_stop_records_hole(tmp_path):
    engine = make_engine(tmp_path, window=200_000, conversation_id="conv",
                         fresh_tail_count=3, leaf_chunk_tokens=1, temporal_rollups_enabled=False)
    x, y, retained, m = ({"role": "user", "content": text, "timestamp": float(i)}
                         for i, text in enumerate(("first", "second", "retained", "chunk"), 1))
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "call", "content": "synthetic result"}
    composite = {"role": "user", "content": "first\n\nsecond", "timestamp": 1.0}
    try:
        engine.ingest([x, y, retained, m, call, result])
        ids = [int(row["store_id"]) for row in _rows(engine)]
        chunk = [m, call, composite]
        view = [*chunk, retained, result]
        mapping = {id(row): sid for row, sid in zip((retained, m, call, result), ids[2:])}
        out = engine._store_complete_input(chunk, {id(composite): ids[:2]}, mapping, view, chunk,
                                           0, [], 1, allow_retained_escape=True)
        assert out == [] and engine._store_complete_blocked[0] == ids[2]
        assert getattr(engine, "_store_complete_hole", None) == ids[0], (
            "the nested escape discarded the hole id needed by the stuck signature")
    finally:
        engine.shutdown()


def test_held_pass_delays_doctor_warning_until_expiry(stuck, clock):
    engine, view = stuck()
    view = retained_pass(engine, retained_pass(engine, view))
    clear_old_hold(engine)
    streak = engine.get_status()["zero_leaf_streak"]
    assert streak["passes"] == 2

    def progress():
        report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
        return next(check for check in report["checks"] if check["check"] == "compaction_progress")

    assert progress()["status"] == "pass"
    assert engine.compress(view, current_tokens=10_001) == view
    assert engine._last_compression_status == "sanitized"
    assert engine.get_status()["zero_leaf_streak"] == streak, "held calls must neither advance nor reset"
    assert progress()["status"] == "pass"
    clock[0] += 601
    clear_old_hold(engine)
    retained_pass(engine, view, 10_001)
    assert engine.get_status()["zero_leaf_streak"]["passes"] == 3
    assert progress() == {"check": "compaction_progress", "status": "warn",
                          "detail": {"passes": 3, "last_reason": streak["last_reason"]}}


def test_blocker_removed_while_held_waits_then_leaf_clears(stuck, clock):
    engine, view = stuck()
    view = retained_pass(engine, retained_pass(engine, view))
    clear_old_hold(engine)
    streak = engine.get_status()["zero_leaf_streak"]
    view = _drop_r(view)
    engine.compress(view, current_tokens=10_001)
    assert engine._last_compression_status == "sanitized"
    assert engine._last_compress_leaves[1] == 0, "a changed host view must still wait inside the back-off"
    assert engine.get_status()["zero_leaf_streak"] == streak
    assert engine._retained_stop_hold_active()
    clock[0] += 601
    clear_old_hold(engine)
    engine.compress(view, current_tokens=10_001)
    assert engine._last_compress_leaves[1] >= 1
    assert not engine._retained_stop_hold_active()
    assert not engine.__dict__.get("_retained_stop_backoffs", {})
    assert engine.get_status()["zero_leaf_streak"] is None


@pytest.mark.parametrize("failure", ["closing", "closed", "read_error"])
def test_host_gates_fail_open_with_backoff(stuck, clock, monkeypatch, failure):
    engine, view = stuck()
    view = arm(engine, view)
    state = dict(engine._retained_stop_backoffs)
    if failure == "closing":
        monkeypatch.setattr(engine, "_compaction_is_closing", lambda: True)
    elif failure == "closed":
        engine.shutdown()
    else:
        monkeypatch.setattr(engine, "_store_complete_frontier", Mock(
            side_effect=sqlite3.OperationalError("synthetic frontier read failure")))
    gates = {
        "should_compress": lambda: engine.should_compress(10_001),
        "preflight": lambda: engine.should_compress_preflight(view),
        "blocked": engine._automatic_compression_blocked,
        "reason": engine._compression_block_reason,
    }
    engine._retained_stop_backoffs = {}
    expected = {name: gate() for name, gate in gates.items()}
    engine._retained_stop_backoffs = state
    actual = {}
    for name, gate in gates.items():
        try:
            actual[name] = gate()
        except Exception as exc:
            actual[name] = type(exc).__name__
    assert actual == expected, "back-off must not add an exception or hold to a host gate"
    if failure == "closed":
        assert actual["preflight"] is False and actual["blocked"] is False and actual["reason"] is None


@pytest.mark.parametrize("state", ["first_stop", "expired"])
def test_inactive_backoff_does_not_read_frontier(stuck, clock, monkeypatch, state):
    engine, view = stuck()
    if state == "first_stop":
        retained_pass(engine, view)
    else:
        arm(engine, view)
        clock[0] += 600
    read = Mock(side_effect=sqlite3.OperationalError("synthetic frontier read failure"))
    monkeypatch.setattr(engine, "_store_complete_frontier", read)
    try:
        active = engine._retained_stop_hold_active()
    except Exception as exc:
        active = type(exc).__name__
    assert active is False
    read.assert_not_called()


def test_hold_deadline_crossing_is_not_an_error(stuck, clock, monkeypatch):
    engine, view = stuck()
    view = arm(engine, view)
    blocks = engine._no_progress_hold_blocks
    ingest = engine._ingest_messages
    ingested = []
    crossings = []

    def ingest_then_mark(*args, **kwargs):
        result = ingest(*args, **kwargs)
        ingested.append(True)
        return result

    def blocks_then_expire(tokens):
        result = blocks(tokens)
        if ingested and result:
            clock[0] += 601
            crossings.append(True)
        return result

    monkeypatch.setattr(engine, "_ingest_messages", ingest_then_mark)
    monkeypatch.setattr(engine, "_no_progress_hold_blocks", blocks_then_expire)
    assert engine._no_progress_hold is None
    assert engine.compress(view, current_tokens=10_001) == view
    assert crossings == [True]
    assert engine._last_compression_status == "sanitized"


def test_transient_frontier_read_is_not_an_error(stuck, clock, monkeypatch):
    engine, view = stuck()
    view = arm(engine, view)
    assert engine._no_progress_hold is None
    ingest = engine._ingest_messages
    frontier = engine._store_complete_frontier
    ingested = []
    reads = []

    def ingest_then_mark(*args, **kwargs):
        result = ingest(*args, **kwargs)
        ingested.append(True)
        return result

    def frontier_fails_once():
        if ingested:
            reads.append(True)
            if len(reads) == 1:
                # The captured hold fails open; the shared hold's later reads succeed.
                raise sqlite3.OperationalError("synthetic transient frontier read failure")
        return frontier()

    monkeypatch.setattr(engine, "_ingest_messages", ingest_then_mark)
    monkeypatch.setattr(engine, "_store_complete_frontier", frontier_fails_once)
    assert engine.compress(view, current_tokens=10_001) == view
    assert len(reads) >= 2, "the failed read must be followed by a successful hold recheck"
    assert engine._last_compression_status == "sanitized"
    assert engine._survival_fit_reason is None


def test_post_selection_error_preserves_armed_backoff(stuck, clock, monkeypatch):
    engine, view = stuck()
    view = arm(engine, view)
    clock[0] += 600
    before = dict(engine._retained_stop_backoffs)
    walk = Mock(side_effect=sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(engine, "_store_complete_owned_rows", walk)
    assert engine.compress(view, current_tokens=10_000) is view
    assert walk.call_count > 0, "the real pass must reach selection before the injected error"
    assert engine._last_compression_status == "error"
    assert engine._retained_stop_backoffs == before
