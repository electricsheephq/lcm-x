"""#618 / #625: the sweep hold follow-ups.

A session that LCM-X only holds for a while is deferred, never ended or reset, and a hold armed for one
conversation never holds another. Engine-level: a host list in; the returned list, the stored rows and nodes,
the public status, the host-shaped gate calls and the log lines out. Summaries are stubbed and counted; the
clock is time.monotonic() plus an offset that the test or a wrapped engine step advances."""

from __future__ import annotations

import inspect
import logging
import re
import time

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.survival_fit import SURVIVAL_FIT_COUNTER_KEY

PAD = " alpha beta gamma delta" * 30
SUMMARY = "Earlier turns.\nExpand for details about: turns"
HOLD = lcm_engine._SWEEP_BUDGET_HOLD_SECONDS


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """LCM's character estimate and no host estimator, as in the #608 tests (#615)."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


class _Clock:
    def __init__(self):
        self._real = time.monotonic
        self.offset = 0.0

    def __call__(self) -> float:
        return self._real() + self.offset


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


@pytest.fixture
def summaries(monkeypatch):
    calls: list[str] = []

    def summarize(**kwargs):
        calls.append(kwargs["text"])
        return SUMMARY, 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _engine(tmp_path, context_length: int = 200_000, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                # #1013: background rollup builds would add their own timed summariser calls.
                "temporal_rollups_enabled": False, "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float, pad: str = PAD) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{pad}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{pad}"}]


def _view(turns: int = 6, pad: str = PAD) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1), pad)]]


def _nodes(engine, depth=None) -> list:
    nodes = engine._dag.get_session_nodes(engine._session_id, limit=100_000)
    return [node for node in nodes if depth is None or node.depth == depth]


def _host_gate(compressor, *, bypass_cooldown: bool = False) -> bool:
    """The host's ``_automatic_compression_gate_blocks`` read (the same on 0.21.2, the 0.21.5 build and upstream
    main): the method on the TYPE; ``ignore_cooldown=True`` only when the call bypasses the cooldown and the
    signature accepts it; otherwise no argument. A missing method never blocks."""
    blocked = getattr(type(compressor), "_automatic_compression_blocked", None)
    if not callable(blocked):
        return False
    accepts = bypass_cooldown and "ignore_cooldown" in inspect.signature(blocked).parameters
    return bool(blocked(compressor, ignore_cooldown=True) if accepts else blocked(compressor))


def _host_transient(compressor) -> bool:
    """The host's ``_mark_compression_blocked_transient`` classification of the published reason."""
    reason = getattr(compressor, "_compression_block_reason", lambda: None)()
    return bool(reason) and str(reason).startswith(("cooldown", "structural_backoff"))


def _no_leaf_stop(engine, monkeypatch, clock) -> None:
    """A threshold sweep that spends its time budget in its first map: the #608 hold starts."""
    original = engine._get_store_id_map_for_messages

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.offset += 121.0
        return result

    view = _view()
    engine.ingest(view)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_get_store_id_map_for_messages", slow)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
    assert engine._last_compression_noop_reason == "threshold sweep time budget spent before the first leaf"
    assert engine._sweep_budget_hold_active()


# -- 1. #618 item 1: the hold belongs to the conversation it was armed for -------------------------------------

def test_item1_a_rebind_to_another_conversation_is_not_held(tmp_path, summaries, monkeypatch, clock):
    engine = _engine(tmp_path)
    try:
        _no_leaf_stop(engine, monkeypatch, clock)
        assert engine.should_compress(engine.threshold_tokens + 1) is False  # A is held
        engine.on_session_start("S2", platform="telegram", context_length=200_000, conversation_id="conv2")
        tokens_b = engine.threshold_tokens + 1
        assert tokens_b < int(200_000 * 0.85)  # over the threshold, under the survival ceiling
        assert engine.should_compress(tokens_b) is True
        assert engine._sweep_budget_hold_until == 0.0  # the reset cleared it
        assert _host_gate(engine) is False and engine._compression_block_reason() is None
    finally:
        engine.shutdown()


def test_item1_the_hold_is_keyed_even_without_a_reset(tmp_path, summaries, monkeypatch, clock):
    """The same session bound to another conversation resets only its counters: the key still ends the hold."""
    engine = _engine(tmp_path)
    try:
        _no_leaf_stop(engine, monkeypatch, clock)
        engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv2")
        assert engine._sweep_budget_hold_until > 0.0  # not reset on this path
        assert engine._sweep_budget_hold_active() is False
    finally:
        engine.shutdown()


def test_item1_session_and_profile_resets_clear_the_hold(tmp_path, summaries, monkeypatch, clock):
    engine = _engine(tmp_path)
    try:
        _no_leaf_stop(engine, monkeypatch, clock)
        engine._reset_session_scoped_runtime_state()
        assert engine._sweep_budget_hold_until == 0.0 and not engine._sweep_budget_hold_active()
        engine._start_sweep_budget_hold()
        engine._reset_profile_runtime_state()
        assert engine._sweep_budget_hold_until == 0.0 and not engine._sweep_budget_hold_active()
    finally:
        engine.shutdown()


# -- 2. #618 item 2: both holds and the boundary cooldown read the monotonic clock -----------------------------

@pytest.mark.parametrize("wall_shift", [-10_000.0, 10_000.0], ids=["clock-set-back", "clock-set-forward"])
def test_item2_moving_the_wall_clock_neither_extends_nor_ends_a_hold(tmp_path, monkeypatch, clock, wall_shift):
    engine = _engine(tmp_path)
    try:
        engine.threshold_tokens = 100
        engine._start_sweep_budget_hold()
        engine.record_rejected_compaction()  # the #651 no-progress hold
        engine.on_session_start("S9", old_session_id="S-unknown", boundary_reason="compression",
                                platform="telegram", context_length=200_000, conversation_id="conv")
        engine.threshold_tokens = 100
        engine._start_sweep_budget_hold()
        engine.record_rejected_compaction()
        assert engine._compression_boundary_cooldown_active()  # the 60 s boundary cooldown is running
        real_time = time.time
        monkeypatch.setattr(lcm_engine.time, "time", lambda: real_time() + wall_shift)
        assert engine._sweep_budget_hold_active() and engine._no_progress_hold_active()
        assert engine._compression_boundary_cooldown_active()
        clock.offset += 61.0  # the cooldown ends by the monotonic clock only
        assert not engine._compression_boundary_cooldown_active()
        assert engine._sweep_budget_hold_active() and engine._no_progress_hold_active()
        clock.offset += HOLD  # the holds end by the monotonic clock only
        assert not engine._sweep_budget_hold_active() and not engine._no_progress_hold_active()
    finally:
        engine.shutdown()


def test_item2_lcm_status_shows_a_wall_clock_until(tmp_path, clock):
    engine = _engine(tmp_path)
    try:
        engine.record_rejected_compaction()
        clock.offset += 100.0
        until = engine.get_status()["no_progress_hold"]["until"]
        assert abs(until - (time.time() + HOLD - 100.0)) < 5.0
    finally:
        engine.shutdown()


# -- 3. #618 item 3: only the sweep hold makes an automatic ceiling pass fit-only ------------------------------

def _ceiling_state(tmp_path, hold: str):
    """Window 6,000 (ceiling 5,100); a list of about 14,000 tokens; one hold of this conversation."""
    engine = _engine(tmp_path, context_length=6_000)
    large = _view(40)
    engine.ingest(large)
    if hold == "sweep":
        engine._start_sweep_budget_hold()
    else:
        engine.record_rejected_compaction()
    request = engine._survival_measure(large)
    assert request >= int(6_000 * 0.85)  # at or over the survival ceiling
    return engine, large, request


@pytest.mark.parametrize("hold", ["sweep", "no_progress"])
def test_item3_a_held_automatic_pass_at_the_ceiling_makes_no_call_and_fits(tmp_path, summaries, hold):
    engine, large, request = _ceiling_state(tmp_path, hold)
    try:
        assert engine.should_compress(request) is True and _host_gate(engine) is False  # the ceiling is exempt
        result = engine.compress(large, current_tokens=request)
        if hold == "sweep":
            assert summaries == [] and _nodes(engine) == []
            assert (engine._last_compression_status, engine._last_compression_noop_reason) == ("noop", "held")
            assert result is not large and engine._survival_measure(result) <= int(6_000 * 0.85)
            assert engine._last_survival_fit["reason"] == "noop"
        else:  # rc4: the #651 hold remains exempt at the ceiling, as in v0.24.8
            assert summaries and _nodes(engine, 0)
            assert (engine._last_compression_status, engine._last_compression_noop_reason) != ("noop", "held")
    finally:
        engine.shutdown()


def test_item3_manual_compress_during_a_hold_at_the_ceiling_still_summarises(tmp_path, summaries):
    engine, large, request = _ceiling_state(tmp_path, "sweep")
    try:
        engine.compress(large, current_tokens=request, force=True)
        assert summaries and _nodes(engine)
    finally:
        engine.shutdown()


def test_item3_a_recovery_attempt_is_not_fit_only(tmp_path, summaries):
    """The host's recovery attempt (``bypass_cooldown``) keeps the #651 rule: never held."""
    engine, large, request = _ceiling_state(tmp_path, "sweep")
    try:
        engine.compress(large, current_tokens=request, bypass_cooldown=True)
        assert summaries and engine._last_compression_noop_reason != "held"
    finally:
        engine.shutdown()


def test_item3_without_a_survival_fit_the_pass_is_not_fit_only(tmp_path, summaries):
    engine = _engine(tmp_path, context_length=6_000, survival_fit=False)
    large = _view(40)
    try:
        engine.ingest(large)
        engine._start_sweep_budget_hold()
        engine.compress(large, current_tokens=engine._survival_measure(large))
        assert summaries and engine._last_compression_noop_reason != "held"
    finally:
        engine.shutdown()


# -- 4. #625: the #608 hold is published as a transient block ---------------------------------------------------

def test_item4_the_sweep_hold_blocks_the_host_gate_with_a_cooldown_reason(tmp_path, summaries, monkeypatch, clock):
    engine = _engine(tmp_path)
    try:
        assert _host_gate(engine) is False and engine._compression_block_reason() is None
        _no_leaf_stop(engine, monkeypatch, clock)
        engine._no_progress_hold = None  # the #651 hold the same stop armed: only the #608 hold is left
        engine.should_compress(engine.threshold_tokens + 1)  # the gate reads the latest observation
        assert _host_gate(engine) is True  # no ignore_cooldown: blocked
        assert engine._compression_block_reason() == "cooldown:lcm_sweep_budget" and _host_transient(engine)
        assert _host_gate(engine, bypass_cooldown=True) is False  # a recovery attempt still reaches compress()
        clock.offset += HOLD + 1  # the hold lapses
        assert _host_gate(engine) is False and engine._compression_block_reason() is None
    finally:
        engine.shutdown()


def test_item4_the_exemptions_are_those_of_the_651_hold(tmp_path, monkeypatch):
    engine = _engine(tmp_path, context_length=100_000, max_assembly_tokens=90_000)
    try:
        engine.threshold_tokens = 1_000
        engine._start_sweep_budget_hold()
        engine.should_compress(5_000)
        assert _host_gate(engine) is True
        engine.should_compress(85_000)  # the survival ceiling
        assert _host_gate(engine) is False
        engine.should_compress(95_000)  # forced overflow (over the assembly cap)
        assert _host_gate(engine) is False
        engine.should_compress(5_000)
        engine._preflight_below_threshold_cleanup_only = True  # a pending cleanup-only pass
        assert _host_gate(engine) is False
        engine._preflight_below_threshold_cleanup_only = False
        assert _host_gate(engine) is True
        monkeypatch.setattr(engine, "_bypasses_lcm_context_management", lambda: True)  # a bypassed session
        assert _host_gate(engine) is False
    finally:
        engine.shutdown()


def test_item4_the_no_progress_reason_keeps_its_name(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine._start_sweep_budget_hold()
        engine.record_rejected_compaction()
        assert engine._compression_block_reason() == "cooldown:lcm_host_rejected"
    finally:
        engine.shutdown()


# -- 5. #618 item 14: a damaged survival-fit count is repaired, not a failure on every later fit ---------------

@pytest.mark.parametrize("damaged", ['"twelve"', "[1]", "1e400"], ids=["text", "list", "infinite"])
def test_item5_a_damaged_count_restarts_at_one_and_marks_the_loss(tmp_path, caplog, damaged):
    engine = _engine(tmp_path, context_length=6_000)
    try:
        engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], f'{{"count": {damaged}, "last_reason": "x"}}')
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            engine._survival_record("noop", 2, [1, 2], 9_000, 4_000, 5_000, False, "")
            engine._survival_record("noop", 2, [3, 4], 9_000, 4_000, 5_000, False, "")
        record = engine._store.read_metadata_json(SURVIVAL_FIT_COUNTER_KEY)
        assert record["count"] == 2 and record["count_lost"] is True
        assert "counter write failed" not in caplog.text
        text = handle_lcm_command("doctor", engine)
        observation = next(line for line in text.splitlines() if "survival_fit: applied" in line)
        assert "applied 2 time(s)" in observation and "restore the database backup" in observation
    finally:
        engine.shutdown()


# -- 6. #627 ask 4: the compaction line carries the host's figure when the host passed one ---------------------

def test_item6_the_compaction_line_prints_host_tokens_when_known(tmp_path, summaries, caplog):
    engine = _engine(tmp_path)
    view = _view()
    try:
        engine.ingest(view)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(view, current_tokens=12_345)
        line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("LCM compaction #"))
        assert re.search(r"\d+→\d+ tokens, host_tokens=12345, \d+ DAG nodes", line), line
    finally:
        engine.shutdown()


def test_item6_the_compaction_line_has_no_host_tokens_when_unknown(tmp_path, summaries, caplog):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=False)
    view = _view()
    try:
        engine.ingest(view)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(view, force=True)
        line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("LCM compaction #"))
        assert "host_tokens" not in line and re.search(r"\d+→\d+ tokens, \d+ DAG nodes", line), line
    finally:
        engine.shutdown()


# -- #718 review: the stop line names a time stop with the sweep off ------------------------------------------

class _Provider:
    """``seconds`` per call on the clock, then a summary."""

    def __init__(self, clock: _Clock, seconds: float):
        self.clock, self.seconds, self.calls = clock, seconds, []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        self.calls.append(timeout)
        self.clock.offset += self.seconds
        return SUMMARY


def _sweep_off_engine(tmp_path, **config) -> LCMEngine:
    return _engine(tmp_path, threshold_full_sweep_enabled=False, l3_truncate_tokens=2, condensation_fanin=2,
                   **config)


def test_stop_line_names_the_soft_target_stop_of_a_sweep_off_condensation(tmp_path, monkeypatch, clock, caplog):
    """The state of test_issue_605_sweep_off_forced_f2 (U7, the condensation refused after a slow leaf)."""
    engine = _sweep_off_engine(tmp_path)
    provider = _Provider(clock, 40.0)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    for _ in range(8):
        engine._foreground_estimates.record_call("", 40.0)
    for index in range(3):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=f"group {index}{PAD}", token_count=1000,
                                         source_token_count=2000, source_ids=[], source_type="messages",
                                         created_at=float(index + 1)))
    view = _view(8)
    try:
        engine.ingest(view)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert len(provider.calls) == 1 and engine._last_condensation_suppressed_reason == "soft_target_reached"
        assert engine._last_compression_status == "compacted"
        line = next(r.getMessage() for r in caplog.records if "LCM compaction stop:" in r.getMessage())
        assert "reason=soft_target_reached" in line, line
    finally:
        engine.shutdown()


# -- #718 bot round: pre-compaction extraction under the hard bound --------------------------------------------

def test_extraction_with_a_slow_route_on_a_sweep_off_compaction_ends_within_the_hard_bound(
        tmp_path, monkeypatch, clock):
    engine = _sweep_off_engine(tmp_path, extraction_enabled=True, summary_timeout_ms=300_000)
    extraction_timeouts = []

    def slow_extraction(**kwargs):  # a route that hangs to its timeout
        extraction_timeouts.append(kwargs["timeout"])
        clock.offset += kwargs["timeout"]

    monkeypatch.setattr(lcm_engine, "extract_before_compaction", slow_extraction)
    provider = _Provider(clock, 10.0)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    view = _view(8)
    try:
        engine.ingest(view)
        started = time.monotonic()
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        hard = engine._foreground_budget_seconds()[1]
        assert extraction_timeouts and extraction_timeouts[0] <= hard
        assert time.monotonic() - started <= hard + 1.0
        assert engine._last_compression_status != "error"
    finally:
        engine.shutdown()


# -- #718 bot round (thread n0bW0): a no-call source is not route-gated ----------------------------------------

def _open_circuit(engine) -> None:
    breaker = engine._summary_circuit_breaker
    for _ in range(breaker.failure_threshold):
        breaker.record_failure(engine._config.summary_model)
    assert not breaker.allows(engine._config.summary_model)


class _Calls:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        self.calls.append(model)
        return SUMMARY


@pytest.mark.parametrize("sweep", [True, False], ids=["sweep-on", "sweep-off"])
def test_a_leaf_source_within_the_level_3_bound_is_stored_while_every_route_is_refused(
        tmp_path, monkeypatch, sweep):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=sweep, leaf_chunk_tokens=100)
    provider = _Calls()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    view = _view(12, pad=" alpha")  # short rows: a backlog over one chunk, every leaf source within 512 tokens
    try:
        _open_circuit(engine)
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        leaves = _nodes(engine, 0)
        assert leaves and provider.calls == []
        assert all("user turn alpha" in leaf.summary for leaf in leaves)  # the source text, stored whole
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep", [True, False], ids=["sweep-on", "sweep-off"])
def test_a_leaf_source_over_the_level_3_bound_is_still_stopped(tmp_path, monkeypatch, sweep):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=sweep, leaf_chunk_tokens=100, l3_truncate_tokens=2)
    provider = _Calls()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    view = _view(12, pad=" alpha")
    try:
        _open_circuit(engine)
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert _nodes(engine) == [] and provider.calls == []
        assert engine._last_compression_noop_reason == "summary route unavailable"
        assert engine._sweep_budget_hold_active()
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("pad", "written"), [("", True), (PAD * 4, False)], ids=["within-512", "over-512"])
def test_condensation_with_every_route_refused_passes_only_a_no_call_group(tmp_path, monkeypatch, pad, written):
    engine = _engine(tmp_path, condensation_fanin=2)
    provider = _Calls()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    for index in range(2):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=f"group {index}{pad}", token_count=1000,
                                         source_token_count=2000, source_ids=[], source_type="messages",
                                         created_at=float(index)))
    try:
        _open_circuit(engine)
        passes = engine._maybe_condense(force_overflow=False)
        assert provider.calls == []
        if written:
            assert passes == 1 and len(_nodes(engine, 1)) == 1
            assert sorted(_nodes(engine, 1)[0].summary.split("\n\n---\n\n")) == ["group 0", "group 1"]  # whole
        else:
            assert passes == 0 and _nodes(engine, 1) == []
            assert engine._last_condensation_suppressed_reason == "summary_route_unavailable"
    finally:
        engine.shutdown()


# -- review of #723: a refused-route pass that stores nothing returns the list it started with ------------------

def _task_list_view(turns: int = 12) -> list[dict]:
    """A leading preserved-task-list row (scaffold, not ingested) ahead of raw backlog, as in the reviewer's case."""
    view = _view(turns, pad=" alpha")
    task_list = {"role": "user", "content": "[Your active task list was preserved across context compression]\n"
                                             "- [>] 1. keep the task list (in_progress)"}
    return [view[0], task_list, *view[1:]]


@pytest.mark.parametrize("sweep", [True, False], ids=["sweep-on", "sweep-off"])
def test_a_refused_pass_that_stores_nothing_returns_its_input(tmp_path, monkeypatch, sweep):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=sweep, leaf_chunk_tokens=100, l3_truncate_tokens=2)
    provider = _Calls()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    view = _task_list_view()
    original = [dict(message) for message in view]
    try:
        _open_circuit(engine)
        engine.ingest(view)
        # force: a manual call has no exit fit, so the pass's own output is what returns
        out = engine.compress(view, current_tokens=engine.threshold_tokens + 1, force=True)
        assert out == original  # the same rows in the same order: the task-list row is kept
        assert provider.calls == [] and _nodes(engine) == []
        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == "summary route unavailable"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep", [True, False], ids=["sweep-on", "sweep-off"])
def test_a_refused_pass_still_stores_a_small_oldest_source_whole(tmp_path, monkeypatch, sweep):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=sweep, leaf_chunk_tokens=100)
    provider = _Calls()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    view = _task_list_view()
    try:
        _open_circuit(engine)
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        leaves = _nodes(engine, 0)
        assert leaves and provider.calls == []
        assert all("user turn alpha" in leaf.summary for leaf in leaves)  # #605 F2: the source text, stored whole
    finally:
        engine.shutdown()


def test_a_full_tail_with_nothing_adopted_stops_before_the_scan(tmp_path, monkeypatch):
    engine = _engine(tmp_path, leaf_chunk_tokens=100, l3_truncate_tokens=2, fresh_tail_count=100,
                     fresh_tail_pressure_yield_enabled=True, fresh_tail_pressure_yield_min_observations=1)
    provider = _Calls()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    view = _view(12, pad=" alpha")
    scans = []
    real_scan = engine._store_complete_backlog
    monkeypatch.setattr(engine, "_store_complete_backlog", lambda *args: scans.append(args) or real_scan(*args))
    try:
        _open_circuit(engine)
        engine.ingest(view)
        assert engine._fresh_tail_start(view) <= engine._leading_anchor_count(view)
        streak = engine._pressure_yield_blocked_streak
        out = engine.compress(view, current_tokens=engine.threshold_tokens + 1, force=True)
        assert out == view
        assert scans == [] and engine._pressure_yield_blocked_streak == streak
        assert provider.calls == [] and _nodes(engine) == []
        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == "summary route unavailable"
    finally:
        engine.shutdown()
