"""#1120: a host fallback can mask a foreground deadline timeout with a different error."""

from __future__ import annotations

import sys
import time
import types

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens

SUMMARY = "Earlier turns.\nExpand for details about: turns"


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


def _provider(monkeypatch, clock, advance):
    calls = []

    def masked_error(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append((model, timeout, clock.offset))
        if model == "m2":
            return SUMMARY
        if advance == "deadline":
            clock.offset += escalation._summary_call.stream_deadline - time.monotonic()
        else:
            clock.offset += timeout if advance == "timeout" else advance
        raise RuntimeError("invalid argument: OpenAI Responses request is missing input")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", masked_error)
    return calls


def test_budget_bound_masked_error_at_the_deadline_is_a_cut(monkeypatch, clock):
    calls = _provider(monkeypatch, clock, "timeout")
    breaker, guard = escalation.SummaryCircuitBreaker(), escalation.SummarySpendGuard()
    budget = escalation.ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0,
                                         estimates=escalation.ForegroundEstimates())
    clock.offset += 75.0  # 40 s usable: the budget binds the call
    with pytest.raises(escalation.SweepBudgetExhausted) as raised:
        escalation.summarize_with_escalation(
            "source text " * 400, source_tokens=4000, token_budget=50, model="m1", fallback_models=["m2"],
            timeout=60.0, circuit_breaker=breaker, spend_guard=guard, budget=budget)
    assert raised.value.reason == "time_budget_exhausted"
    assert [(model, round(timeout)) for model, timeout, _started in calls] == [("m1", 40)]
    assert breaker._failures == {} and len(guard._calls) == 1


def test_installed_host_deadline_without_budget_binding_makes_a_masked_error_at_the_deadline_a_cut(
        monkeypatch, clock):
    _host_module(monkeypatch, clock, [])
    calls = _provider(monkeypatch, clock, "deadline")
    breaker, guard = escalation.SummaryCircuitBreaker(), escalation.SummarySpendGuard()
    budget = escalation.ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0,
                                         estimates=escalation.ForegroundEstimates())
    clock.offset += 0.4  # about 114.6 s usable: the configured 60 s timeout still binds
    with pytest.raises(escalation.SweepBudgetExhausted) as raised:
        escalation.summarize_with_escalation(
            "source text " * 400, source_tokens=4000, token_budget=50, model="m1", fallback_models=["m2"],
            timeout=60.0, circuit_breaker=breaker, spend_guard=guard, budget=budget)
    assert raised.value.reason == budget.stop_reason == "time_budget_exhausted"
    assert [(model, timeout) for model, timeout, _started in calls] == [("m1", 60.0)]
    assert clock.offset - calls[0][2] == pytest.approx(114.6, abs=0.5)
    assert breaker._failures == {} and len(guard._calls) == 1


def test_masked_error_well_before_the_deadline_is_a_route_failure_and_uses_the_fallback(monkeypatch, clock):
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", types.ModuleType("agent.auxiliary_client"))
    calls = _provider(monkeypatch, clock, 5.0)
    breaker, guard = escalation.SummaryCircuitBreaker(), escalation.SummarySpendGuard()
    budget = escalation.ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0,
                                         estimates=escalation.ForegroundEstimates())
    result = escalation.summarize_with_escalation(
        "source text " * 400, source_tokens=4000, token_budget=50, model="m1", fallback_models=["m2"],
        timeout=60.0, circuit_breaker=breaker, spend_guard=guard, budget=budget)
    assert result == (SUMMARY, 1)
    assert [(model, timeout) for model, timeout, _started in calls] == [("m1", 60.0), ("m2", 60.0)]
    assert breaker._failures == {"m1": 1} and len(guard._calls) == 1


def test_budget_bound_masked_error_early_is_a_route_failure_and_admits_the_fallback(monkeypatch, clock):
    calls = _provider(monkeypatch, clock, 5.0)
    breaker, guard = escalation.SummaryCircuitBreaker(), escalation.SummarySpendGuard()
    budget = escalation.ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0,
                                         estimates=escalation.ForegroundEstimates())
    clock.offset += 75.0
    result = escalation.summarize_with_escalation(
        "source text " * 400, source_tokens=4000, token_budget=50, model="m1", fallback_models=["m2"],
        timeout=60.0, circuit_breaker=breaker, spend_guard=guard, budget=budget)
    assert result == (SUMMARY, 1)
    assert [(model, round(timeout), round(started)) for model, timeout, started in calls] == [
        ("m1", 40, 75), ("m2", 35, 80)]
    assert breaker._failures == {"m1": 1} and len(guard._calls) == 1


def test_no_seam_or_budget_binding_keeps_a_masked_error_at_the_deadline_a_route_failure(monkeypatch, clock):
    """#1099: ignoring the configured timeout without an installed deadline remains a route failure."""
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", types.ModuleType("agent.auxiliary_client"))
    calls = _provider(monkeypatch, clock, "deadline")
    breaker, guard = escalation.SummaryCircuitBreaker(), escalation.SummarySpendGuard()
    budget = escalation.ForegroundBudget(soft=0, hard=120.0, configured_timeout=60.0,
                                         estimates=escalation.ForegroundEstimates())
    clock.offset += 0.4
    with pytest.raises(escalation.SweepBudgetExhausted) as raised:  # no time left to admit the fallback
        escalation.summarize_with_escalation(
            "source text " * 400, source_tokens=4000, token_budget=50, model="m1", fallback_models=["m2"],
            timeout=60.0, circuit_breaker=breaker, spend_guard=guard, budget=budget)
    assert raised.value.reason == "time_budget_exhausted"
    assert [(model, timeout) for model, timeout, _started in calls] == [("m1", 60.0)]
    assert clock.offset - calls[0][2] == pytest.approx(114.6, abs=0.5)
    assert breaker._failures == {"m1": 1} and len(guard._calls) == 1
