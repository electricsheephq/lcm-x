"""#666 round 3: the threshold sweep's absolute deadline governs every summariser attempt, a condensation rejection
logs the stop line once, and a fail-open keeps the completed pre-leaf condensation passes.

The clock is time.monotonic() plus an offset that the stubbed provider call advances; no test waits."""

from __future__ import annotations

import logging
import sqlite3
import time

import pytest

import hermes_lcm.compaction as lcm_compaction
import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
LEAF = "Earlier turns.\nExpand for details about: turns"
MERGED = "Merged arc.\nExpand for details about: arc"
LEVEL_3 = "head text\n\n[...deterministic truncation — details available via lcm_expand...]\n\ntail text"
REJECTED_LINE = "LCM compaction stopped: summary result rejected at level 3"
SOURCE = "source text " * 400


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


class _SlowFailingRoute:
    """Stands in for one provider call: it takes min(60 s, its timeout) on the clock, then fails."""

    def __init__(self, clock: _Clock):
        self.clock, self.calls = clock, []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        self.calls.append((model, timeout, self.clock()))
        self.clock.offset += 60.0 if timeout is None else min(60.0, timeout)
        raise TimeoutError("provider timed out")


# -- D1, D2: summarize_with_escalation ------------------------------------------------------------------------

def test_d1_deadline_bounds_every_route_attempt_and_never_falls_to_level_3(monkeypatch, clock):
    route = _SlowFailingRoute(clock)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", route)
    deadline = clock() + 120.0
    with pytest.raises(escalation.SweepBudgetExhausted):
        escalation.summarize_with_escalation(
            SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=50, model="m1",
            fallback_models=["m2"], timeout=60.0, deadline=deadline)
    assert len(route.calls) == 2  # the third would start with 0 s left
    assert all(timeout <= deadline - started + 0.01 for _model, timeout, started in route.calls)  # real clock slack
    assert lcm_engine.SweepBudgetExhausted is escalation.SweepBudgetExhausted


def test_d2_no_deadline_keeps_the_timeouts_and_level_3(monkeypatch, clock):
    route = _SlowFailingRoute(clock)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", route)
    summary, level = escalation.summarize_with_escalation(
        SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=50, model="m1",
        fallback_models=["m2"], timeout=60.0, deadline=None)
    assert [(model, timeout) for model, timeout, _ in route.calls] == [("m1", 60.0), ("m2", 60.0)] * 2
    assert level == 3 and "deterministic truncation" in summary


# -- D3: the reviewer's case, end to end ----------------------------------------------------------------------

def _engine(tmp_path, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                "condensation_fanin": 2, "database_path": str(tmp_path / "lcm.db"),
                # #1013: background rollup builds would add their own timed summariser calls.
                "temporal_rollups_enabled": False, **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _view(turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1))]]


def _depth_0_nodes(engine, count: int) -> None:
    for index in range(count):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=f"group {index}", token_count=1000,
                                         source_token_count=2000, source_ids=[], source_type="messages",
                                         created_at=float(index + 1)))


def _compress(engine, view, caplog):
    engine.ingest(view)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        return engine.compress(view, current_tokens=engine.threshold_tokens + 1)


def test_d3_slow_failing_routes_stay_inside_the_sweep_deadline(tmp_path, monkeypatch, clock, caplog):
    # #605 F2: the condensation groups are over a lowered level 3 bound, so each one calls the slow routes.
    engine = _engine(tmp_path, summary_fallback_models=["fallback-a"], l3_truncate_tokens=2)
    route = _SlowFailingRoute(clock)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", route)
    _depth_0_nodes(engine, 6)
    try:
        _compress(engine, _view(), caplog)
        spent = clock.offset  # the provider time; the real steps between calls take milliseconds
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert route.calls and spent <= lcm_compaction._THRESHOLD_FULL_SWEEP_MAX_SECONDS, (spent, route.calls)
        assert telemetry["pre_leaf_condensation_stop_reason"] == "time_budget_exhausted"
        assert telemetry["stop_reason"] in {"time_budget_exhausted", "summary_result_rejected"}
        nodes = engine._dag.get_session_nodes("S", limit=100_000)
        assert len(nodes) == 6 and not any("deterministic truncation" in node.summary for node in nodes)
    finally:
        engine.shutdown()


# -- D4: the stop line, once per compaction -------------------------------------------------------------------

def _stub(monkeypatch, leaf_levels=(1,), condensed_level=3):
    """Leaves take the next scripted level (the last repeats); every condensation returns ``condensed_level``."""
    leaves: list[int] = []

    def summarize(**kwargs):
        if int(kwargs.get("depth", 0)) > 0:
            return (MERGED, 1) if condensed_level == 1 else (LEVEL_3, 3)
        level = leaf_levels[min(len(leaves), len(leaf_levels) - 1)]
        leaves.append(level)
        return (LEAF, level) if level != 3 else (LEVEL_3, 3)

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)


@pytest.mark.parametrize("case", ["sweep-condensation-only", "non-sweep-condensation-only", "leaf-and-condensation"])
def test_d4_a_rejection_logs_the_stop_line_once(tmp_path, monkeypatch, caplog, case):
    if case == "non-sweep-condensation-only":
        engine = _engine(tmp_path, threshold_full_sweep_enabled=False)
        _stub(monkeypatch)
        _depth_0_nodes(engine, 3)
    else:
        engine = _engine(tmp_path)
        _stub(monkeypatch, leaf_levels=(3,) if case == "leaf-and-condensation" else (1,))
        _depth_0_nodes(engine, 6)
    try:
        _compress(engine, _view(), caplog)
        assert sum(REJECTED_LINE in record.getMessage() for record in caplog.records) == 1
    finally:
        engine.shutdown()


# -- D5: a later fail-open keeps the completed pre-leaf passes ------------------------------------------------

def test_d5_leaf_publication_lock_keeps_the_pre_leaf_passes(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path)
    _stub(monkeypatch, condensed_level=1)
    _depth_0_nodes(engine, 6)
    real_add = engine._dag.add_node

    def add_node(node, **kwargs):
        if node.source_type == "messages":
            raise sqlite3.OperationalError("database is locked")
        return real_add(node, **kwargs)

    monkeypatch.setattr(engine._dag, "add_node", add_node)
    try:
        _compress(engine, _view(), caplog)
        condensed = [node for node in engine._dag.get_session_nodes("S", limit=100_000) if node.depth > 0]
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["status"] == "error" and len(condensed) >= 1
        assert telemetry["condensation_passes"] >= len(condensed)
        assert telemetry["total_passes"] >= len(condensed)
    finally:
        engine.shutdown()


# -- Round 4: the terminal fallback and the pre-leaf rejection without a leaf ---------------------------------

def test_r4_deadline_spent_by_the_last_attempt_raises_instead_of_level_3(monkeypatch, clock):
    route = _SlowFailingRoute(clock)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", route)
    breaker, guard = escalation.SummaryCircuitBreaker(), escalation.SummarySpendGuard()
    with pytest.raises(escalation.SweepBudgetExhausted):
        escalation.summarize_with_escalation(
            SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=50, model="m1", timeout=60.0,
            circuit_breaker=breaker, spend_guard=guard, deadline=clock() + 120.0)
    assert [model for model, _timeout, _started in route.calls] == ["m1", "m1"]  # level 1, level 2


def test_r4_content_rejections_with_time_left_still_give_level_3(monkeypatch, clock):
    calls = []

    def not_shorter(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append(model)
        return prompt  # never shorter than its source

    monkeypatch.setattr(escalation, "_invoke_summary_llm", not_shorter)
    summary, level = escalation.summarize_with_escalation(
        SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=50, model="m1", timeout=60.0,
        circuit_breaker=escalation.SummaryCircuitBreaker(), spend_guard=escalation.SummarySpendGuard(),
        deadline=clock() + 120.0)
    assert calls == ["m1", "m1"] and level == 3 and "deterministic truncation" in summary


def test_r4_pre_leaf_rejection_without_a_leaf_logs_the_stop_line_once(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path)
    _stub(monkeypatch)  # every condensation comes back as a truncating level 3
    _depth_0_nodes(engine, 6)
    try:
        _compress(engine, _view(1), caplog)  # one fresh turn: no raw backlog for a leaf
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["pre_leaf_condensation_stop_reason"] == "summary_result_rejected"
        assert telemetry["leaf_passes"] == 0
        assert sum(REJECTED_LINE in record.getMessage() for record in caplog.records) == 1
    finally:
        engine.shutdown()
