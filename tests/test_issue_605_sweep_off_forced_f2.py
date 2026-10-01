"""#605 lane A PR 2: the sweep-off, manual, forced-overflow and recovery paths under the one foreground budget;
F2 (a source within the level 3 bound is written verbatim with no call); the #597 sweep exit.

Same harness as test_issue_605_foreground_budget.py: the clock is time.monotonic() plus an offset that the
stubbed provider (at ``escalation._invoke_summary_llm``) or a wrapped engine step advances; no test waits."""

from __future__ import annotations

import time

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.rollup_builder as rollup_builder
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
SUMMARY = "Earlier turns.\nExpand for details about: turns"


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
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


class _Provider:
    """``seconds`` per call on the clock, then a summary; ``None`` hangs to its timeout; ``fail`` raises."""

    def __init__(self, clock: _Clock, seconds: float | None = 30.0, fail: bool = False):
        self.clock, self.seconds, self.fail, self.calls = clock, seconds, fail, []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        self.calls.append({"model": model, "timeout": timeout, "started": self.clock.offset, "prompt": prompt})
        if self.fail:
            raise RuntimeError("route refused")
        if self.seconds is None or (timeout is not None and self.seconds > timeout):
            self.clock.offset += timeout
            raise TimeoutError("provider timed out")
        self.clock.offset += self.seconds
        return SUMMARY


def _provider(monkeypatch, clock, seconds: float | None = 30.0, fail: bool = False) -> _Provider:
    provider = _Provider(clock, seconds, fail)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    return provider


def _engine(tmp_path, context_length: int = 200_000, **config) -> LCMEngine:
    # l3_truncate_tokens under every source unless a test sets it: those cells time model calls, so no source
    # takes the no-call verbatim path (F2) of one within the level 3 bound.
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": False, "max_assembly_tokens": 100_000, "l3_truncate_tokens": 2,
                "condensation_fanin": 2, "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float, pad: str = PAD) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{pad}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{pad}"}]


def _view(turns: int = 12, pad: str = PAD) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1), pad)]]


def _advance_on(monkeypatch, engine, name: str, clock: _Clock, seconds: float):
    original = getattr(engine, name)

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.offset += seconds
        return result

    monkeypatch.setattr(engine, name, wrapped)


def _nodes(engine, depth=None) -> list:
    nodes = engine._dag.get_session_nodes("S", limit=100_000)
    return [node for node in nodes if depth is None or node.depth == depth]


def _no_fragment(engine) -> bool:
    return not any(escalation._L3_TRUNCATION_MARKER in node.summary for node in _nodes(engine))


def _depth_0_nodes(engine, count: int, pad: str = PAD) -> None:
    for index in range(count):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=f"group {index}{pad}", token_count=1000,
                                         source_token_count=2000, source_ids=[], source_type="messages",
                                         created_at=float(index + 1)))


def _hard_bound_held(provider, budget_t0_offset=0.0, hard=120.0, reserve=5.0) -> bool:
    """No call started past t0 + hard - reserve, and none was given a timeout past it."""
    return all(call["started"] + call["timeout"] <= budget_t0_offset + hard - reserve + 0.5 for call in provider.calls)


# -- U5 / F2: a source within the level 3 bound is written verbatim, with no call ---------------------------------

def test_u5_a_22_token_leaf_is_written_verbatim_with_no_call(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, leaf_chunk_tokens=20, l3_truncate_tokens=512)
    provider = _provider(monkeypatch, clock)
    view = _view(6, pad="")  # each row is a few tokens: the leaf's input is far under 512
    try:
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        leaves = _nodes(engine, 0)
        assert provider.calls == [] and leaves
        for leaf in leaves:
            assert tokens.count_tokens(leaf.summary) <= 512 and "user turn" in leaf.summary
        assert engine._last_compression_status == "compacted" and _no_fragment(engine)
    finally:
        engine.shutdown()


def test_u5_a_small_condensation_group_is_written_verbatim_with_no_call(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, l3_truncate_tokens=512)
    provider = _provider(monkeypatch, clock)
    _depth_0_nodes(engine, 2, pad=" small")
    try:
        nodes = _nodes(engine, 0)
        source, summary_tokens, level = engine._condense_summary_nodes(nodes)
        condensed, = _nodes(engine, 1)
        assert provider.calls == [] and level == 3
        assert condensed.summary == "\n\n---\n\n".join(node.summary for node in nodes)
    finally:
        engine.shutdown()


def test_u5_rollups_and_callers_without_the_flag_still_call(monkeypatch, clock):
    provider = _provider(monkeypatch, clock, seconds=1.0)
    text = "a short source of a few tokens " * 10  # ~80 tokens: within the 512 level 3 bound
    assert escalation.summarize_with_escalation(text, source_tokens=400, token_budget=50) == (SUMMARY, 1)
    provenance: dict = {}
    assert escalation.summarize_with_escalation(text, source_tokens=400, token_budget=50, provenance=provenance,
                                                verbatim_small_source=True) == (text, 3)
    assert provenance == {"model": "deterministic"} and len(provider.calls) == 1
    # The rollup builder (rollup_builder.py) does not set the flag: a rollup of the same size still calls.
    summary, _tokens = rollup_builder._summarize_capped(
        text, target_tokens=50, max_tokens=512, config=LCMConfig(),
        summarizer=escalation.summarize_with_escalation, circuit_breaker=None, spend_guard=None)
    assert summary == SUMMARY and len(provider.calls) == 2


# -- U7: sweep off: one bounded leaf; _maybe_condense under the budget ---------------------------------------------

def test_u7_maybe_condense_is_refused_by_the_soft_target_after_a_slow_leaf(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, clock, seconds=40.0)  # the leaf ends at +40; a 40 s condensation at +80
    for _ in range(8):
        engine._foreground_estimates.record_call("", 40.0)
    _depth_0_nodes(engine, 3)
    view = _view(8)
    try:
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert len(provider.calls) == 1 and len(_nodes(engine, 0)) == 4  # the leaf; no condensation started
        assert _nodes(engine, 1) == [] and engine._last_condensation_suppressed_reason == "soft_target_reached"
        assert engine._last_compression_status == "compacted" and _no_fragment(engine)
    finally:
        engine.shutdown()


def test_u7_a_time_stop_before_the_sweep_off_leaf_stores_nothing_and_holds(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, clock)
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 110.0)  # 5 s of usable time left
    view = _view(8)
    try:
        engine.ingest(view)
        result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert provider.calls == [] and _nodes(engine) == [] and isinstance(result, list)
        assert engine._last_compression_status != "error"  # a stop, not an exception
        assert engine._last_compression_noop_reason.endswith("time budget spent before the first leaf")
        assert engine._sweep_budget_hold_until > time.time()  # the #608 hold ends at its time
    finally:
        engine.shutdown()


# -- manual /compress (force=True): the hard bound only --------------------------------------------------------

def test_manual_compress_has_no_soft_target_stop_and_keeps_the_hard_bound(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=True)
    provider = _provider(monkeypatch, clock, seconds=29.0)
    view = _view(12)
    try:
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1, force=True)
        telemetry = engine.get_status()["threshold_full_sweep"]
        # +87 + 29 + the 5 s reserve is past 120: the fourth leaf does not start; no soft stop at +58.
        assert [round(call["started"]) for call in provider.calls] == [0, 29, 58]
        assert telemetry["stop_reason"] == "time_budget_exhausted" and _hard_bound_held(provider)
    finally:
        engine.shutdown()


# -- U8: forced overflow --------------------------------------------------------------------------------------

def _forced(tmp_path, **config) -> tuple[LCMEngine, list[dict]]:
    engine = _engine(tmp_path, max_assembly_tokens=2_000, **config)
    view = _view(12)
    engine.ingest(view)
    assert engine._should_force_overflow_recovery(observed_tokens=engine._survival_measure(view), messages=view)
    return engine, view


def test_u8_forced_overflow_with_no_time_left_makes_no_call_and_fits_its_cap(tmp_path, monkeypatch, clock):
    engine, view = _forced(tmp_path)
    provider = _provider(monkeypatch, clock)
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 110.0)
    try:
        result = engine.compress(view, current_tokens=engine._survival_measure(view))
        assert provider.calls == [] and _nodes(engine) == []
        assert tokens.count_tokens(str(result)) and engine._last_compression_status == "overflow_recovery"
        assert sum(map(lambda m: tokens.count_tokens(m.get("content") or ""), result)) <= 2_000
        assert engine._sweep_budget_hold_until == 0.0  # forced overflow is never held
    finally:
        engine.shutdown()


def test_u8_with_the_fit_off_a_refused_route_still_writes_the_level_3_leaf_as_today(tmp_path, monkeypatch, clock):
    engine, view = _forced(tmp_path, survival_fit=False)
    provider = _provider(monkeypatch, clock, fail=True)
    try:
        engine.compress(view, current_tokens=engine._survival_measure(view))
        leaf, = _nodes(engine, 0)  # #652: no fit can run, so the level 3 truncation converges as before
        assert provider.calls and tokens.count_tokens(leaf.summary) <= 2
    finally:
        engine.shutdown()


def test_u8_with_the_fit_off_time_never_yields_level_3(tmp_path, monkeypatch, clock):
    engine, view = _forced(tmp_path, survival_fit=False)
    provider = _provider(monkeypatch, clock, fail=True)
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 110.0)
    try:
        engine.compress(view, current_tokens=engine._survival_measure(view))
        assert provider.calls == [] and _nodes(engine) == []
    finally:
        engine.shutdown()


# -- U9: a recovery attempt with a slow route ------------------------------------------------------------------

def test_u9_a_recovery_attempt_with_a_slow_route_fits_under_95_percent_and_respects_the_hard_bound(
        tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, context_length=6_000)
    engine.threshold_tokens = 3_000
    provider = _provider(monkeypatch, clock, seconds=None)  # every call hangs to its timeout
    # 70 s of pre-work: 45 s of usable time, under the configured 60 s, so the budget bounds the call.
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 70.0)
    view = _view(8)
    try:
        engine.ingest(view)
        request = engine._survival_measure(view) + 1_000
        result = engine.compress(view, current_tokens=request, bypass_cooldown=True)
        assert engine._survival_measure(result) + 1_000 <= int(3_000 * 0.95)
        assert provider.calls and _hard_bound_held(provider) and _no_fragment(engine)
        assert engine._summary_circuit_breaker._failures.get("<task-default>", 0) == 0  # a budget cut
    finally:
        engine.shutdown()
