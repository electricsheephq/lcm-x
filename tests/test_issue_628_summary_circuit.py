"""#628: a rejected summary result does not switch the summary route off, and an open circuit does not turn
a compaction into level 3 truncations.

Escalation-level tests drive the real chain with a stubbed provider helper; engine-level tests drive
compress() with that same stub, so the breaker, the leaf loop and the log lines are the real ones. Levels are
read from a spy around the real summarize_with_escalation; no test pins a token count."""

from __future__ import annotations

import logging
import re
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import SummaryCircuitBreaker

PAD = " alpha beta gamma delta" * 30
ACCEPTED = "Earlier turns.\nExpand for details about: turns"
STOP_LINE = "LCM compaction stopped: summary route unavailable"
REJECTED_LINE = "LCM summary result rejected"
COMPACTION_LINE = re.compile(
    r"^LCM compaction #\d+: \d+ messages → \d+ \(\d+ leaf pass(?:es)?, \d+→\d+ tokens, \d+ DAG nodes"
    r"(?P<rest>.*)\)$")


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """Same counter for every run (#615): LCM's character estimate, no host estimator."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


class _Provider:
    """Stands in for _call_llm_for_summary: each call takes the next scripted answer (the last one repeats).
    "reject" returns the prompt itself, which is never shorter than its source; None is a provider failure."""

    def __init__(self, *script):
        self.script = list(script) or [ACCEPTED]
        self.calls: list[str] = []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        answer = self.script[min(len(self.calls), len(self.script) - 1)]
        self.calls.append(model)
        return prompt if answer == "reject" else answer


@pytest.fixture
def levels(monkeypatch):
    """Levels of every summarize_with_escalation call the engine makes, in order."""
    seen: list[int] = []
    real = escalation.summarize_with_escalation

    def spy(*args, **kwargs):
        summary, level = real(*args, **kwargs)
        seen.append(level)
        return summary, level

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", spy)
    return seen


def _provider(monkeypatch, *script) -> _Provider:
    provider = _Provider(*script)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    return provider


def _engine(tmp_path, context_length: int = 200_000, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _view(turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            *[row for i in range(turns) for row in _turn(f"T{i}", 10.0 * (i + 1))]]


def _open_circuit(engine) -> None:
    breaker = engine._summary_circuit_breaker
    for _ in range(breaker.failure_threshold):
        breaker.record_failure(engine._config.summary_model)
    assert not breaker.allows(engine._config.summary_model)


def _frontier(engine) -> int:
    return int(getattr(engine._lifecycle.get_by_conversation("conv"), "current_frontier_store_id", 0))


def _nodes(engine) -> list:
    return engine._dag.get_session_nodes("S")


def _count(caplog, text: str) -> int:
    return sum(text in record.getMessage() for record in caplog.records)


def _compress(engine, view, caplog, **kwargs):
    engine.ingest(view)
    kwargs.setdefault("current_tokens", engine.threshold_tokens + 1)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        return engine.compress(view, **kwargs)


# -- T1: a leaf whose results are not shorter than its source ------------------------------------------------

def test_t1_not_shorter_results_give_level_3_without_opening_the_circuit(tmp_path, monkeypatch, levels, caplog):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, "reject", "reject", ACCEPTED)
    try:
        _compress(engine, _view(), caplog)
        assert levels[0] == 3 and provider.calls[:2] == ["", ""]
        assert engine._summary_circuit_breaker.allows("")
        assert len(provider.calls) >= 3 and levels[1] == 1  # the next leaf calls the model
    finally:
        engine.shutdown()


# -- T2: provider failures still open the circuit at two -----------------------------------------------------

def test_t2_two_provider_failures_open_the_circuit(monkeypatch):
    provider = _provider(monkeypatch, None)
    breaker = SummaryCircuitBreaker()
    summary, level = escalation.summarize_with_escalation(
        "source text " * 80, source_tokens=200, token_budget=50, circuit_breaker=breaker)
    assert level == 3 and len(provider.calls) == 2
    assert not breaker.allows("")


# -- T3: content rejections have their own threshold ---------------------------------------------------------

def test_t3_six_consecutive_rejections_open_the_circuit_and_a_success_resets_the_count():
    breaker = SummaryCircuitBreaker(cooldown_seconds=300)
    assert breaker.rejection_threshold == 6
    for _ in range(5):
        breaker.record_rejection("m", now=0.0)
    assert breaker.allows("m", now=0.0)
    breaker.record_success("m")
    for _ in range(5):
        breaker.record_rejection("m", now=0.0)
    assert breaker.allows("m", now=0.0)
    breaker.record_rejection("m", now=0.0)
    assert not breaker.allows("m", now=1.0)
    assert breaker.allows("m", now=301.0)  # after the cooldown the count stays
    breaker.record_rejection("m", now=301.0)
    assert not breaker.allows("m", now=302.0)


def test_t3_failures_and_rejections_do_not_clear_each_other():
    breaker = SummaryCircuitBreaker(failure_threshold=2, rejection_threshold=2)
    breaker.record_failure("m", now=0.0)
    breaker.record_rejection("m", now=0.0)
    assert breaker.allows("m", now=0.0)
    breaker.record_failure("m", now=0.0)
    assert not breaker.allows("m", now=1.0)
    other = SummaryCircuitBreaker(failure_threshold=2, rejection_threshold=2)
    other.record_rejection("m", now=0.0)
    other.record_failure("m", now=0.0)
    other.record_rejection("m", now=0.0)
    assert not other.allows("m", now=1.0)


def test_t3_rejection_threshold_is_configured(monkeypatch, tmp_path):
    assert LCMConfig().summary_circuit_breaker_rejection_threshold == 6
    monkeypatch.setenv("LCM_SUMMARY_CIRCUIT_BREAKER_REJECTION_THRESHOLD", "3")
    assert LCMConfig.from_env().summary_circuit_breaker_rejection_threshold == 3
    engine = _engine(tmp_path, summary_circuit_breaker_rejection_threshold=4)
    try:
        assert engine._summary_circuit_breaker.rejection_threshold == 4
    finally:
        engine.shutdown()


# -- T4: a threshold sweep with every route refused ----------------------------------------------------------

def test_t4_sweep_with_every_route_refused_writes_nothing_and_holds(tmp_path, monkeypatch, levels, caplog):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, ACCEPTED)
    view = _view()
    try:
        _open_circuit(engine)
        frontier = _frontier(engine)
        result = _compress(engine, view, caplog)
        assert len(result) < len(view) and engine._last_survival_fit["reason"] == "exit_fit:noop"
        assert _nodes(engine) == [] and _frontier(engine) == frontier
        assert provider.calls == [] and levels == []
        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == "summary route unavailable"
        assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "summary_route_unavailable"
        cooldown = engine._summary_circuit_breaker.cooldown_seconds
        assert time.time() < engine._sweep_budget_hold_until <= time.time() + cooldown
        assert engine.should_compress(engine.threshold_tokens + 1) is False
        assert _count(caplog, STOP_LINE) == 1
        line = next(r.getMessage() for r in caplog.records if STOP_LINE in r.getMessage())
        assert re.search(r"\(circuit open, \d+s left\); 0 leaves written, backlog kept$", line)
    finally:
        engine.shutdown()


# -- T5: a forced overflow recovery still converges through level 3 ------------------------------------------

def test_t5_forced_overflow_recovery_converges_through_level_3(tmp_path, monkeypatch, levels, caplog):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, ACCEPTED)
    try:
        _open_circuit(engine)
        _compress(engine, _view(), caplog, current_tokens=150_000)  # over the 100k assembly cap
        assert _nodes(engine) and levels and set(levels) == {3}
        assert provider.calls == [] and _count(caplog, STOP_LINE) == 0
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


# -- T6: every rejected result names its reason ---------------------------------------------------------------

def _install_fake_auxiliary_client(monkeypatch, content):
    module = ModuleType("agent.auxiliary_client")
    module.call_llm = lambda **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)


@pytest.mark.parametrize(("content", "helper_line"), [
    ("", "LCM summary discarded empty output"),
    ("<think>only reasoning</think>", "LCM summary discarded reasoning-only output"),
    ("a plain answer outside the envelope", "LCM summary discarded output that violated the integrity contract"),
], ids=["empty", "reasoning-only", "contract"])
def test_t6_no_content_results_log_the_helper_line_and_the_chain_line(monkeypatch, caplog, content, helper_line):
    _install_fake_auxiliary_client(monkeypatch, content)
    source = "source text " * 80
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        _summary, level = escalation.summarize_with_escalation(
            source, source_tokens=escalation.count_tokens(source), token_budget=50)
    assert level == 3
    assert _count(caplog, helper_line) == 2  # level 1 and level 2
    chain = [r.getMessage() for r in caplog.records if REJECTED_LINE in r.getMessage()]
    assert len(chain) == 2 and all("reason=no_content" in line and "result_tokens=0" in line for line in chain)


def test_t6_not_shorter_result_logs_both_token_counts(monkeypatch, caplog):
    long_answer = "a long answer " * 400
    _provider(monkeypatch, long_answer)
    source = "source text " * 80
    source_tokens = escalation.count_tokens(source)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        escalation.summarize_with_escalation(source, source_tokens=source_tokens, token_budget=50, model="m1")
    chain = [r.getMessage() for r in caplog.records if REJECTED_LINE in r.getMessage()]
    expected = (f"(reason=not_shorter, source_tokens={source_tokens}, "
                f"result_tokens={escalation.count_tokens(long_answer)}, model=m1)")
    assert chain == [f"{REJECTED_LINE} {expected}"] * 2


def test_t6_chain_without_source_tokens_prints_unknown(monkeypatch, caplog):
    _provider(monkeypatch, "")
    with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
        assert escalation._invoke_summary_llm_chain("prompt", 10) is None
    assert _count(caplog, "source_tokens=unknown") == 1


# -- T7: the compaction line counts level 3 leaves ------------------------------------------------------------

def _compaction_line(caplog) -> re.Match:
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LCM compaction #")]
    assert len(lines) == 1
    match = COMPACTION_LINE.match(lines[0])
    assert match, lines[0]
    return match


def test_t7_compaction_line_without_level_3_leaves_is_unchanged(tmp_path, monkeypatch, levels, caplog):
    engine = _engine(tmp_path)
    _provider(monkeypatch, ACCEPTED)
    try:
        _compress(engine, _view(), caplog)
        assert levels and 3 not in levels
        assert _compaction_line(caplog).group("rest") == ""
    finally:
        engine.shutdown()


def test_t7_compaction_line_counts_two_level_3_leaves(tmp_path, monkeypatch, levels, caplog):
    engine = _engine(tmp_path)
    _provider(monkeypatch, "reject", "reject", "reject", "reject", ACCEPTED)
    try:
        _compress(engine, _view(), caplog)
        assert levels.count(3) == 2
        assert _compaction_line(caplog).group("rest") == ", 2 level 3 leaves"
    finally:
        engine.shutdown()


# -- T8: the same stop with the full sweep off ----------------------------------------------------------------

def test_t8_full_sweep_off_with_every_route_refused_writes_no_leaf(tmp_path, monkeypatch, levels, caplog):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=False)
    provider = _provider(monkeypatch, ACCEPTED)
    view = _view()
    try:
        _open_circuit(engine)
        result = _compress(engine, view, caplog)
        assert len(result) < len(view) and engine._last_survival_fit["reason"] == "exit_fit:noop"
        assert _nodes(engine) == [] and provider.calls == [] and levels == []
        assert engine._last_compression_noop_reason == "summary route unavailable"
        assert engine._sweep_budget_hold_active()
        assert _count(caplog, STOP_LINE) == 1
    finally:
        engine.shutdown()


# -- T9: the circuit opens on leaf 3 --------------------------------------------------------------------------

def test_t9_circuit_opening_on_leaf_3_keeps_leaves_1_to_3_and_stops_before_leaf_4(
        tmp_path, monkeypatch, levels, caplog):
    engine = _engine(tmp_path)
    provider = _provider(monkeypatch, ACCEPTED, ACCEPTED, None, None, ACCEPTED)
    try:
        result = _compress(engine, _view(8), caplog)
        assert levels == [1, 1, 3] and len(provider.calls) == 4
        assert len(_nodes(engine)) == 3 and result is not None
        assert engine._last_compression_status == "compacted"
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["leaf_passes"] == 3 and telemetry["stop_reason"] == "summary_route_unavailable"
        assert telemetry["status"] == "partial"
        assert _count(caplog, STOP_LINE) == 1 and _count(caplog, "3 leaves written, backlog kept") == 1
        assert engine._sweep_budget_hold_until == 0.0  # a stored leaf: no hold
        assert _compaction_line(caplog).group("rest") == ", 1 level 3 leaves"
    finally:
        engine.shutdown()


# -- T10: a recovery attempt with every route refused ---------------------------------------------------------

THRESHOLD = 3_000
HOST_TOKENS = 1_000


def test_t10_recovery_attempt_with_every_route_refused_meets_the_recovery_budget(
        tmp_path, monkeypatch, levels, caplog):
    """The #617 recovery assertions of test_issue_608_sweep_budget_stop.py (C1), with the circuit open."""
    engine = _engine(tmp_path, context_length=6_000)
    engine.threshold_tokens = THRESHOLD
    provider = _provider(monkeypatch, ACCEPTED)
    view = _view(8)
    try:
        _open_circuit(engine)
        engine.ingest(view)
        request = engine._survival_measure(view) + HOST_TOKENS
        assert THRESHOLD < request < int(6_000 * 0.85)
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=request, bypass_cooldown=True)
        assert _nodes(engine) == [] and provider.calls == [] and levels == []
        assert engine._last_compression_status == "noop"
        assert len(result) < len(view)
        assert engine._survival_measure(result) + HOST_TOKENS <= int(THRESHOLD * 0.95)
        stored = {(r["role"], r["content"]) for r in engine._store.get_session_messages("S", limit=100_000)}
        kept = {(m["role"], m["content"]) for m in result}
        assert all((m["role"], m["content"]) in stored for m in view[1:] if (m["role"], m["content"]) not in kept)
        assert engine._last_survival_fit["reason"] == "recovery_attempt:noop"
    finally:
        engine.shutdown()


# -- T11: condensation with every route refused ---------------------------------------------------------------

def _depth_0_nodes(engine, count: int) -> None:
    for index in range(count):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=f"group {index}", token_count=1000,
                                         source_token_count=2000, source_ids=[], source_type="messages",
                                         created_at=index))


def test_t11_sweep_condensation_with_every_route_refused_writes_no_node(tmp_path, monkeypatch, levels):
    engine = _engine(tmp_path, condensation_fanin=2, summary_prefix_target_tokens=100)
    provider = _provider(monkeypatch, ACCEPTED)
    _depth_0_nodes(engine, 2)
    try:
        _open_circuit(engine)
        passes, reason = engine._run_threshold_sweep_condensation(
            target_tokens=100, pass_budget=5, deadline=time.monotonic() + 100.0)
        assert (passes, reason) == (0, "summary_route_unavailable")
        assert len(_nodes(engine)) == 2 and provider.calls == [] and levels == []
    finally:
        engine.shutdown()


def test_t11_condensation_not_forced_writes_no_node_and_forced_converges(tmp_path, monkeypatch, levels):
    engine = _engine(tmp_path, condensation_fanin=2)
    provider = _provider(monkeypatch, ACCEPTED)
    _depth_0_nodes(engine, 2)
    try:
        _open_circuit(engine)
        assert engine._maybe_condense(force_overflow=False) == 0
        assert len(_nodes(engine)) == 2 and levels == []
        assert engine._maybe_condense(force_overflow=True) == 1
        assert len(_nodes(engine)) == 3 and levels == [3] and provider.calls == []
    finally:
        engine.shutdown()


# -- T12, T13: without a survival fit the stop does not apply; level 3 converges as at the base -----------------

@pytest.mark.parametrize("no_fit", ["survival_fit_off", "window_unknown"])
def test_t12_t13_without_a_survival_fit_the_leaf_is_written_at_level_3(tmp_path, monkeypatch, levels, caplog, no_fit):
    config = {"survival_fit": False} if no_fit == "survival_fit_off" else {}
    engine = _engine(tmp_path, **config)
    if no_fit == "window_unknown":
        engine.context_length = 0
    provider = _provider(monkeypatch, ACCEPTED)
    view = _view()
    try:
        _open_circuit(engine)
        assert engine.threshold_tokens > 0
        result = _compress(engine, view, caplog)
        assert result is not view and _nodes(engine) and levels and set(levels) == {3}
        assert provider.calls == [] and engine._last_compression_status == "compacted"
        assert engine._sweep_budget_hold_until == 0.0 and _count(caplog, STOP_LINE) == 0
    finally:
        engine.shutdown()


# -- T14: _maybe_condense checks the route before every depth ------------------------------------------------

def _condensation_state(engine) -> None:
    """Two depth-0 nodes and one depth-1 node, uncondensed; the route has 4 rejections already."""
    for index, depth in enumerate((0, 0, 1)):
        text = f"group {index} at depth {depth}"
        engine._dag.add_node(SummaryNode(session_id="S", depth=depth, summary=text,
                                         token_count=escalation.count_tokens(text), source_token_count=0,
                                         source_ids=[], source_type="messages", created_at=index))
    for _ in range(4):
        engine._summary_circuit_breaker.record_rejection(engine._config.summary_model)
    assert engine._summary_circuit_breaker.allows(engine._config.summary_model)


@pytest.mark.parametrize("force_overflow", [False, True], ids=["not-forced", "forced"])
def test_condensation_rechecks_route_before_every_depth(tmp_path, monkeypatch, levels, force_overflow):
    engine = _engine(tmp_path, condensation_fanin=2, threshold_full_sweep_enabled=False)
    provider = _provider(monkeypatch, "reject")  # never shorter than its source
    _condensation_state(engine)
    try:
        passes = engine._maybe_condense(force_overflow=force_overflow)
        new = sorted(node.depth for node in _nodes(engine)[3:])
        assert len(provider.calls) == 2  # depth 0: level 1 and level 2, rejected; the counts 5 and 6 open the route
        if force_overflow:
            assert passes == 2 and new == [1, 2] and levels == [3, 3]
        else:
            assert passes == 1 and new == [1] and levels == [3]
            assert engine._last_condensation_suppressed_reason == "summary_route_unavailable"
    finally:
        engine.shutdown()
