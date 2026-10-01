"""#605 lane A: one foreground budget per compress(): a 60 s soft target and a 120 s hard admission bound.

The clock is time.monotonic() plus an offset that the stubbed provider call (or a wrapped engine step)
advances; no test waits. The provider stands in at ``escalation._invoke_summary_llm`` so the real escalation
chain, its admission, its timeout classes and its spend slot run."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import types

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
SUMMARY = "Earlier turns.\nExpand for details about: turns"
STOP_LINE = re.compile(r"LCM compaction stop: reason=(\S+) leaves=(\d+) elapsed=([\d.]+)s pre=([\d.]+)s "
                       r"calls=([\d.]+)s finalize=([\d.]+)s backlog_tokens=(\d+)")


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """Same counter for every run (#615): LCM's character estimate, no host estimator."""
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
    monkeypatch.setattr(time, "monotonic", fake)  # every module reads time.monotonic through the module
    return fake


class _Provider:
    """One provider call: ``seconds[model]`` on the clock, then a summary; ``None`` hangs to its timeout."""

    def __init__(self, clock: _Clock, seconds: dict, default: float | None = 30.0):
        self.clock, self.seconds, self.default, self.calls = clock, seconds, default, []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        started = self.clock.offset
        self.calls.append((model, timeout, started))
        took = self.seconds.get(model, self.default)
        if took is None or (timeout is not None and took > timeout):
            self.clock.offset += timeout
            raise TimeoutError("provider timed out")
        self.clock.offset += took
        return SUMMARY


def _provider(monkeypatch, clock, default=30.0, **seconds) -> _Provider:
    provider = _Provider(clock, seconds, default)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    return provider


def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                "condensation_fanin": 2, "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _view(turns: int = 12) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1))]]


def _compress(engine, view, caplog):
    engine.ingest(view)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        return engine.compress(view, current_tokens=engine.threshold_tokens + 1)


def _advance_on(monkeypatch, engine, name: str, clock: _Clock, seconds: float, calls=(1,)):
    """Wrap engine step ``name``: the listed invocations (all when None) move the clock after the step."""
    original, count = getattr(engine, name), 0

    def wrapped(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if calls is None or count in calls:
            clock.offset += seconds
        return result

    monkeypatch.setattr(engine, name, wrapped)


def _stop_lines(caplog) -> list[tuple]:
    found = [STOP_LINE.search(record.getMessage()) for record in caplog.records]
    return [match.groups() for match in found if match]


def _leaves(engine) -> list:
    return [node for node in engine._dag.get_session_nodes("S", limit=100_000) if node.depth == 0]


def _depth_0_nodes(engine, count: int) -> None:
    for index in range(count):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=f"group {index}", token_count=1000,
                                         source_token_count=2000, source_ids=[], source_type="messages",
                                         created_at=float(index + 1)))


# -- the soft target ---------------------------------------------------------------------------------------------

def test_soft_target_stops_after_the_stored_leaf_whose_successor_would_end_past_60s(
        tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, clock, default=29.0)
    try:
        _compress(engine, _view(), caplog)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert [round(started) for _model, _timeout, started in provider.calls] == [0, 29]  # a third ends at 87
        assert len(_leaves(engine)) == 2 and clock.offset == pytest.approx(58.0)
        assert telemetry["stop_reason"] == "soft_target_reached" and telemetry["status"] == "partial"
        assert telemetry["budget_exhausted"] is False
        assert not any("deterministic truncation" in node.summary for node in _leaves(engine))
    finally:
        engine.shutdown()


def test_first_leaf_is_admitted_past_the_soft_target(tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, clock, default=30.0)
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 40.0)  # pre-work: the leaf ends at +70
    try:
        _compress(engine, _view(), caplog)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert len(provider.calls) == 1 and provider.calls[0][1] == pytest.approx(60.0, abs=0.5)
        assert len(_leaves(engine)) == 1 and telemetry["stop_reason"] == "soft_target_reached"
    finally:
        engine.shutdown()


def test_soft_zero_keeps_the_hard_admission_only(tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path, foreground_soft_seconds=0)
    provider = _provider(monkeypatch, clock, default=30.0)
    try:
        _compress(engine, _view(), caplog)
        telemetry = engine.get_status()["threshold_full_sweep"]
        # +90 + 30 + the 5 s reserve is past 120: the fourth leaf does not start.
        assert [round(started) for _model, _timeout, started in provider.calls] == [0, 30, 60]
        assert telemetry["stop_reason"] == "time_budget_exhausted" and telemetry["budget_exhausted"] is True
        assert all(timeout <= 120 - 5 - started + 0.5 for _model, timeout, started in provider.calls)
    finally:
        engine.shutdown()


# -- Astra counterexample (a): pre-leaf condensation leaves the first leaf inside the soft target -----------------

def test_a_oversized_frontier_condenses_at_0_only_and_the_first_leaf_ends_by_60(tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path)
    for _ in range(8):  # 29 s walls (30 s would sit on the 60 s boundary with the real clock's slack)
        engine._foreground_estimates.record_call("", 29.0)
    provider = _provider(monkeypatch, clock, default=29.0)
    _depth_0_nodes(engine, 6)
    try:
        _compress(engine, _view(), caplog)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["pre_leaf_condensation_passes"] == 1
        assert telemetry["pre_leaf_condensation_stop_reason"] == "soft_target_reached"
        # At +29 a second condensation would leave the first leaf ending at +87: refused. Without the #605 rule
        # the half-budget split admitted condensations at +0 and +29, and the first leaf ended at +87.
        assert [round(started) for _model, _timeout, started in provider.calls] == [0, 29]  # condensation, leaf
        assert telemetry["leaf_passes"] == 1 and clock.offset == pytest.approx(58.0)
        assert telemetry["stop_reason"] == "soft_target_reached"
    finally:
        engine.shutdown()


# -- Astra counterexample (b): a stale high estimate never starves the first leaf ---------------------------------

def test_b_stale_115s_estimate_and_56s_of_pre_work_still_attempt_the_first_leaf_across_a_hold_expiry(
        tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path, summary_timeout_ms=300_000)
    for _ in range(8):  # a full ring of slow walls: the estimate is its ceiling, hard - reserve = 115 s
        engine._foreground_estimates.record_call("", 200.0)
    assert engine._foreground_estimates.call_estimate("", 115.0) == 115.0
    provider = _provider(monkeypatch, clock, default=None)  # compaction 1: the call hangs to its timeout
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 56.0, calls=None)
    view = _view()
    try:
        _compress(engine, view, caplog)
        assert len(provider.calls) == 1 and provider.calls[0][1] == pytest.approx(59.0, abs=0.5)
        assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "time_budget_exhausted"
        assert _leaves(engine) == [] and engine._sweep_budget_hold_until > time.time()
        assert engine._summary_circuit_breaker._failures.get("<task-default>", 0) == 0  # a budget cut
        real_time = time.time
        monkeypatch.setattr(time, "time", lambda: real_time() + 601.0)  # both holds have ended
        provider.default = 20.0
        clock.offset = 0.0
        _compress(engine, view, caplog)
        assert len(provider.calls) == 2 and len(_leaves(engine)) == 1
    finally:
        engine.shutdown()


# -- Astra counterexample (c): an ordinary route timeout keeps its accounting and its fallback ---------------------

def test_c_configured_timeout_is_an_ordinary_failure_and_the_fallback_stores_by_80(
        tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path, summary_model="primary", summary_fallback_models=["fallback"])
    provider = _provider(monkeypatch, clock, primary=None, fallback=20.0)
    try:
        _compress(engine, _view(), caplog)
        assert [(model, round(timeout), round(started)) for model, timeout, started in provider.calls] == [
            ("primary", 60, 0), ("fallback", 55, 60)]
        assert engine._summary_circuit_breaker._failures.get("primary") == 1
        assert len(_leaves(engine)) == 1 and clock.offset == pytest.approx(80.0)
        # The primary's 60 s wall is its estimate now: +80 + 60 + 5 is past the hard bound.
        assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "time_budget_exhausted"
    finally:
        engine.shutdown()


def test_c_a_call_cut_by_the_budget_counts_no_failure_ends_the_chain_and_takes_one_slot(monkeypatch, clock):
    provider = _provider(monkeypatch, clock, default=None)
    breaker, guard = escalation.SummaryCircuitBreaker(), escalation.SummarySpendGuard()
    budget = escalation.ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0,
                                         estimates=escalation.ForegroundEstimates())
    clock.offset += 75.0  # 40 s of usable time left: the budget, not the configured 60 s, bounds the call
    with pytest.raises(escalation.SweepBudgetExhausted) as raised:
        escalation.summarize_with_escalation(
            "source text " * 400, source_tokens=4000, token_budget=50, model="m1", fallback_models=["m2"],
            timeout=60.0, circuit_breaker=breaker, spend_guard=guard, budget=budget)
    assert raised.value.reason == "time_budget_exhausted"
    assert [(model, round(timeout)) for model, timeout, _started in provider.calls] == [("m1", 40)]
    assert breaker._failures == {} and len(guard._calls) == 1


# -- Astra counterexample (d): the classifier reads the error channel; the host deadline seam ----------------------

def _host_module(monkeypatch, clock, seen: list):
    module = types.ModuleType("agent.auxiliary_client")
    module._deadline = None

    def _current_aux_stream_deadline():
        return module._deadline

    class _Scope:
        def __init__(self, deadline):
            self.deadline, self.previous = deadline, None

        def __enter__(self):
            self.previous, module._deadline = module._deadline, self.deadline

        def __exit__(self, *exc):
            module._deadline = self.previous

    def call_llm(**kwargs):
        seen.append((kwargs.get("timeout"), module._deadline))
        clock.offset += kwargs["timeout"]  # the host stops the stream at the deadline it was given
        raise TimeoutError("auxiliary stream timed out at the host compression deadline")

    module._current_aux_stream_deadline = _current_aux_stream_deadline
    module.aux_stream_deadline = _Scope
    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    return module


def test_d_the_error_reaches_the_classifier_through_the_summary_call_channel(monkeypatch, clock):
    seen: list = []
    host = _host_module(monkeypatch, clock, seen)
    host._deadline = original = clock() + 500.0  # the host waits longer: the budget's deadline is installed
    breaker, guard = escalation.SummaryCircuitBreaker(), escalation.SummarySpendGuard()
    budget = escalation.ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0,
                                         estimates=escalation.ForegroundEstimates())
    clock.offset += 80.0
    with pytest.raises(escalation.SweepBudgetExhausted):
        escalation.summarize_with_escalation(
            "source text " * 400, source_tokens=4000, token_budget=50, timeout=60.0,
            circuit_breaker=breaker, spend_guard=guard, budget=budget)
    assert isinstance(escalation._summary_call.error, TimeoutError)  # caught inside _call_llm_for_summary
    assert len(seen) == 1 and seen[0][0] == pytest.approx(35.0, abs=0.5)
    assert seen[0][1] == pytest.approx(budget.t0 + 115.0, abs=0.01)  # min(host deadline, t0 + hard - reserve)
    assert host._deadline == original  # restored on exit
    assert breaker._failures == {}


def test_d_a_tighter_host_deadline_is_kept(monkeypatch, clock):
    seen: list = []
    host = _host_module(monkeypatch, clock, seen)
    host._deadline = clock() + 30.0
    budget = escalation.ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0,
                                         estimates=escalation.ForegroundEstimates())
    tight = host._deadline
    with pytest.raises(escalation.SweepBudgetExhausted):
        escalation.summarize_with_escalation(
            "source text " * 400, source_tokens=4000, token_budget=50, timeout=60.0, budget=budget)
    assert seen[0][1] == tight and host._deadline == tight


# -- Astra counterexample (e): a slow finalize step shows in the stop line and grows the reserve -------------------

def test_e_a_121s_finalize_step_shows_in_the_stop_line_and_the_next_reserve_is_20s(
        tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path)
    _provider(monkeypatch, clock, default=10.0)
    _advance_on(monkeypatch, engine, "_refresh_raw_backlog_debt", clock, 121.0)
    try:
        assert engine._foreground_estimates.finalize_reserve() == 5.0
        _compress(engine, _view(), caplog)
        (reason, leaves, elapsed, pre, calls, finalize, _backlog), = _stop_lines(caplog)
        assert reason == "soft_target_reached" and int(leaves) == 5
        assert float(finalize) == pytest.approx(121.0, abs=0.2) and float(calls) == pytest.approx(50.0, abs=0.2)
        assert float(pre) + float(calls) + float(finalize) == pytest.approx(float(elapsed), abs=0.15)
        assert engine._foreground_estimates.finalize_reserve() == 20.0
    finally:
        engine.shutdown()


# -- the INFO stop line ------------------------------------------------------------------------------------------

def test_one_info_stop_line_per_compaction_whose_seconds_add_up(tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path)
    _provider(monkeypatch, clock, default=25.0)
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 3.0)
    try:
        _compress(engine, _view(), caplog)
        (reason, leaves, elapsed, pre, calls, finalize, backlog), = _stop_lines(caplog)
        assert reason == "soft_target_reached" and int(leaves) == 2 and int(backlog) > 0
        assert float(pre) == pytest.approx(3.0, abs=0.2) and float(calls) == pytest.approx(50.0, abs=0.2)
        assert float(pre) + float(calls) + float(finalize) == pytest.approx(float(elapsed), abs=0.15)
        stop = next(r for r in caplog.records if STOP_LINE.search(r.getMessage()))
        assert stop.levelno == logging.INFO and "alpha" not in stop.getMessage()
    finally:
        engine.shutdown()


# -- the spend guard: one slot per compaction; rollups on their own guard ------------------------------------------

def test_one_spend_slot_per_foreground_compaction_and_rollups_have_their_own_guard(
        tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, clock, default=5.0)
    try:
        result = _compress(engine, _view(), caplog)
        assert len(provider.calls) > 2 and len(engine._summary_spend_guard._calls) == 1
        calls = len(provider.calls)
        _compress(engine, [*result, *[row for i in range(12, 20) for row in _turn(f"T{i}", 10.0 * (i + 1))]], caplog)
        assert len(provider.calls) > calls
        assert len(engine._summary_spend_guard._calls) == 2
        assert engine._rollup_spend_guard is not engine._summary_spend_guard
        guards = []
        monkeypatch.setattr(lcm_engine, "run_rollup_maintenance",
                            lambda dag, config, scope, **kwargs: guards.append(kwargs["spend_guard"]))
        engine._schedule_rollup_maintenance("profile")
        engine.drain_rollup_maintenance(timeout=10.0)
        assert guards == [engine._rollup_spend_guard]
    finally:
        engine.shutdown()


# -- the stop reason in the status surfaces ------------------------------------------------------------------------

def test_soft_target_reached_is_a_partial_stop_in_get_status_and_lcm_status(tmp_path, monkeypatch, clock, caplog):
    engine = _engine(tmp_path)
    _provider(monkeypatch, clock, default=30.0)
    try:
        _compress(engine, _view(), caplog)
        status = json.loads(lcm_tools.lcm_status({}, engine=engine))
        assert status["threshold_full_sweep"]["stop_reason"] == "soft_target_reached"
        assert status["threshold_full_sweep"]["status"] == "partial"
        assert status["config"]["threshold_full_sweep_max_seconds"] == 120
        assert status["config"]["foreground_soft_seconds"] == 60
    finally:
        engine.shutdown()


# -- the settings ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("env", "expected"), [
    ({}, (60.0, 120.0)),
    ({"LCM_FOREGROUND_SOFT_SECONDS": "0"}, (0.0, 120.0)),
    ({"LCM_FOREGROUND_SOFT_SECONDS": "45", "LCM_FOREGROUND_HARD_SECONDS": "90"}, (45.0, 90.0)),
    ({"LCM_FOREGROUND_SOFT_SECONDS": "200", "LCM_FOREGROUND_HARD_SECONDS": "90"}, (90.0, 90.0)),
    ({"LCM_FOREGROUND_SOFT_SECONDS": "soon", "LCM_FOREGROUND_HARD_SECONDS": "-3"}, (60.0, 120.0)),
])
def test_settings_parse_clamp_and_fall_back(tmp_path, monkeypatch, env, expected):
    for key in ("LCM_FOREGROUND_SOFT_SECONDS", "LCM_FOREGROUND_HARD_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = LCMConfig.from_env()
    config.database_path = str(tmp_path / "lcm.db")
    engine = LCMEngine(config=config)
    try:
        assert engine._foreground_budget_seconds() == expected
    finally:
        engine.shutdown()
