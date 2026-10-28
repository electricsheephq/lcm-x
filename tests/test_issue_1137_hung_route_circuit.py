"""#1137: a full-window hang opens its route for the next compaction."""

from __future__ import annotations

import json
import logging
import time

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

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

    def __call__(self):
        return self._real() + self.offset


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


def _provider(monkeypatch, clock, *, l2=False, masked=False):
    calls = []

    def hung(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append((model, timeout))
        if model == "m2":
            return SUMMARY
        if l2 and len(calls) == 1:
            return ""  # reject L1 so the hang is reached through L2
        clock.offset += escalation._summary_call.stream_deadline - time.monotonic()
        if masked:
            raise RuntimeError("invalid argument: OpenAI Responses request is missing input")
        raise TimeoutError("auxiliary stream timed out at the host compression deadline")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", hung)
    return calls


def _budget(*, configured_timeout=120.0, leaf_reserve=0.0):
    budget = escalation.ForegroundBudget(
        soft=0, hard=120.0, configured_timeout=configured_timeout,
        estimates=escalation.ForegroundEstimates())
    budget.leaf_reserve = leaf_reserve
    return budget


def _summarize(breaker, budget, *, fallback_models=None, route_key_prefix=""):
    return escalation.summarize_with_escalation(
        "source text " * 400, source_tokens=4000, token_budget=50, model="m1",
        fallback_models=fallback_models, timeout=budget.configured_timeout,
        circuit_breaker=breaker, budget=budget, route_key_prefix=route_key_prefix)


@pytest.mark.parametrize("l2", [False, True], ids=["L1", "L2"])
@pytest.mark.parametrize("leaf_reserve", [0.0, 30.0], ids=["leaf", "condensation"])
def test_full_window_hang_uses_fallback_on_second_compaction(monkeypatch, clock, l2, leaf_reserve):
    calls = _provider(monkeypatch, clock, l2=l2)
    breaker = escalation.SummaryCircuitBreaker()
    first = _budget(leaf_reserve=leaf_reserve)
    with pytest.raises(escalation.SweepBudgetExhausted) as raised:
        _summarize(breaker, first, fallback_models=None if l2 else ["m2"])
    assert raised.value.reason == first.stop_reason == "time_budget_exhausted"
    assert [model for model, _timeout in calls] == (["m1", "m1"] if l2 else ["m1"])
    assert calls[-1][1] == pytest.approx(115.0 - leaf_reserve, abs=0.5)
    second = _budget(leaf_reserve=leaf_reserve)
    assert _summarize(breaker, second, fallback_models=["m2"]) == (SUMMARY, 1)
    assert [model for model, _timeout in calls] == (["m1", "m1", "m2"] if l2 else ["m1", "m2"])
    assert breaker.route_status(["m1"])["last_error_class"] == "hung"
    assert not breaker.allows("m1") and breaker.allows("m2")


@pytest.mark.parametrize("configured_timeout", [60.0, 120.0])
def test_partial_window_masked_error_remains_a_budget_cut(monkeypatch, clock, configured_timeout):
    calls = _provider(monkeypatch, clock, masked=True)
    breaker = escalation.SummaryCircuitBreaker()
    budget = _budget(configured_timeout=configured_timeout)
    clock.offset += 75.0  # only 40 s of the original window remain
    with pytest.raises(escalation.SweepBudgetExhausted) as raised:
        _summarize(breaker, budget, fallback_models=["m2"])
    assert raised.value.reason == "time_budget_exhausted"
    assert calls == [("m1", pytest.approx(40.0, abs=0.5))]
    assert breaker._failures == {} and breaker._open_until == {}
    assert breaker.route_status(["m1"])["last_error_class"] is None


def test_early_host_fence_cut_is_a_budget_cut_not_a_hang(monkeypatch, clock):
    """The host's own compression deadline can fall before LCM's usable deadline (host timeout below
    foreground_hard_seconds): a cut after 30 s of a 115 s grant must not open the route as hung."""
    calls = []

    def early_fence(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append((model, timeout))
        clock.offset += 30.0
        raise TimeoutError("auxiliary stream timed out at the host compression deadline")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", early_fence)
    breaker = escalation.SummaryCircuitBreaker()
    budget = _budget()
    with pytest.raises(escalation.SweepBudgetExhausted) as raised:
        _summarize(breaker, budget, fallback_models=["m2"])
    assert raised.value.reason == "time_budget_exhausted"
    assert calls == [("m1", pytest.approx(115.0, abs=0.5))]
    assert breaker._failures == {} and breaker._open_until == {}
    assert breaker.route_status(["m1"])["last_error_class"] is None
    assert breaker.allows("m1")


def test_full_window_hang_opens_only_the_rollup_prefixed_key(monkeypatch, clock):
    calls = _provider(monkeypatch, clock)
    breaker = escalation.SummaryCircuitBreaker()
    with pytest.raises(escalation.SweepBudgetExhausted):
        _summarize(breaker, _budget(), route_key_prefix="<rollup>:")
    assert not breaker.allows("<rollup>:m1")
    assert breaker.route_status(["m1"], route_key_prefix="<rollup>:")["last_error_class"] == "hung"
    assert breaker.allows("m1") and breaker.allows("<rollup>:m2")
    assert breaker.route_status(["m1"])["last_error_class"] is None
    assert [model for model, _timeout in calls] == ["m1"]


def test_hang_status_cooldown_success_and_one_warning_per_opening(tmp_path, clock, caplog):
    engine = LCMEngine(config=LCMConfig(summary_model="m1", database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    breaker = engine._summary_circuit_breaker
    try:
        with caplog.at_level(logging.WARNING, logger=escalation.logger.name):
            breaker.record_hang("m1")
            breaker.record_hang("m1")
        opened = breaker.route_status(["m1"])
        assert opened["state"] == "open" and opened["last_error_class"] == "hung"
        assert 0 < opened["seconds_left"] <= breaker.cooldown_seconds
        assert json.loads(lcm_tools.lcm_status({}, engine=engine))["summary_route"] == opened
        warnings = [r for r in caplog.records if "whole window" in r.getMessage()]
        assert len(warnings) == 1 and "m1" in warnings[0].getMessage()
        clock.offset += breaker.cooldown_seconds + 1
        assert breaker.allows("m1")
        with caplog.at_level(logging.WARNING, logger=escalation.logger.name):
            breaker.record_hang("m1", now=clock())
        assert len([r for r in caplog.records if "whole window" in r.getMessage()]) == 2
        clock.offset += breaker.cooldown_seconds + 1
        assert breaker.allows("m1")
        breaker.record_success("m1")
        assert breaker._failures == breaker._rejections == breaker._open_until == {}
        closed = breaker.route_status(["m1"])
        assert closed["state"] == "closed" and closed["last_error_class"] is None
        assert json.loads(lcm_tools.lcm_status({}, engine=engine))["summary_route"] == closed
    finally:
        engine.shutdown()


@pytest.mark.parametrize("use_full_window", [True, False], ids=["deadline", "early-fence"])
@pytest.mark.parametrize("leaf_reserve", [0.0, 30.0], ids=["leaf", "condensation"])
def test_short_configured_timeout_with_installed_deadline_records_only_full_window_hangs(
        monkeypatch, clock, use_full_window, leaf_reserve):
    monkeypatch.setattr(escalation, "_host_stream_deadline_seam", lambda: True)

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        assert timeout == 60.0  # configured timeout, not bound by the foreground budget
        clock.offset += (escalation._summary_call.stream_deadline - time.monotonic()
                         if use_full_window else 30.0)
        raise TimeoutError("auxiliary stream timed out at the host compression deadline")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    breaker = escalation.SummaryCircuitBreaker()
    budget = _budget(configured_timeout=60.0, leaf_reserve=leaf_reserve)
    with pytest.raises(escalation.SweepBudgetExhausted):
        _summarize(breaker, budget)
    assert breaker.allows("m1") is not use_full_window
    assert breaker.route_status(["m1"])["last_error_class"] == ("hung" if use_full_window else None)


@pytest.mark.parametrize("failure,fallbacks,blocked", [
    ("hang", ["m2"], False),
    ("hang", [], False),
    ("hang", ["m2"], True),
    ("prework", ["m2"], False),
], ids=["hang-fallback", "hang-no-fallback", "hang-blocked-fallback", "non-hang-budget"])
def test_automatic_compaction_holds_only_without_a_hung_route_alternative(
        tmp_path, monkeypatch, clock, caplog, failure, fallbacks, blocked):
    from tests.test_issue_605_foreground_budget import _engine, _view

    engine = _engine(tmp_path, summary_model="m1", summary_fallback_models=fallbacks,
                     summary_timeout_ms=120_000)
    calls = _provider(monkeypatch, clock)
    view = _view()
    engine.ingest(view)
    if blocked:
        engine._summary_circuit_breaker.record_hang("m2")
    if failure == "prework":
        # A previous pass's hang must not make this non-hang budget stop eligible.
        engine._summary_circuit_breaker.record_hang("m1")
        original = engine._get_store_id_map_for_messages

        def spend_budget(*args, **kwargs):
            result = original(*args, **kwargs)
            clock.offset += 120.0
            return result

        monkeypatch.setattr(engine, "_get_store_id_map_for_messages", spend_budget)
    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "time_budget_exhausted"
        assert engine._dag.get_session_nodes("S") == []
        retry = failure == "hang" and bool(fallbacks) and not blocked
        assert engine._sweep_budget_hold_active() is not retry
        assert engine._no_progress_hold_active() is (failure != "hang")
        assert engine.should_compress(engine.threshold_tokens + 1) is retry
        skip_lines = [r for r in caplog.records if "skipping sweep budget hold" in r.getMessage()]
        assert len(skip_lines) == int(retry)
        if retry:
            assert skip_lines[0].levelno == logging.INFO
            assert "m1" in skip_lines[0].getMessage()
            assert [model for model, _timeout in calls] == ["m1"]
            assert not engine._summary_circuit_breaker.allows("m1")
            engine.compress(result, current_tokens=engine.threshold_tokens + 1)
            assert calls[1:] and {model for model, _timeout in calls[1:]} == {"m2"}
            assert engine._dag.get_session_nodes("S")
            assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


def test_fallback_hang_after_a_quick_primary_failure_retries_the_primary(tmp_path, monkeypatch, clock, caplog):
    """A primary that fails fast (circuit still closed) stays usable when the fallback hangs: no holds."""
    from tests.test_issue_605_foreground_budget import _engine, _view

    engine = _engine(tmp_path, summary_model="m1", summary_fallback_models=["m2"], summary_timeout_ms=120_000)
    calls = []

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append(model)
        if model == "m1":
            if calls.count("m1") == 1:
                raise RuntimeError("upstream returned 502")
            return SUMMARY
        clock.offset += escalation._summary_call.stream_deadline - time.monotonic()
        raise TimeoutError("auxiliary stream timed out at the host compression deadline")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    view = _view()
    engine.ingest(view)
    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert calls == ["m1", "m2"]
        assert engine._summary_circuit_breaker.allows("m1")
        assert not engine._summary_circuit_breaker.allows("m2")
        assert not engine._sweep_budget_hold_active()
        assert not engine._no_progress_hold_active()
        assert engine.should_compress(engine.threshold_tokens + 1)
        assert len([r for r in caplog.records if "skipping sweep budget hold" in r.getMessage()]) == 1
        engine.compress(result, current_tokens=engine.threshold_tokens + 1)
        assert calls[2] == "m1" and "m2" not in calls[2:]
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()
