"""Outcome decisions from frozen facts, without provider, logger or breaker objects."""
from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

values = importlib.import_module("hermes_lcm.summary_attempt")
outcomes = importlib.import_module("hermes_lcm.summary_outcome")


def facts(**changes: object) -> Any:
    return replace(outcomes.SummaryOutcomeFacts(
        budget_present=False, budget_bound=False, call_timeout=100.0, configured_timeout=120.0,
        hard=115.0, reserve=5.0, leaf_reserve=10.0, usable_deadline=1100.0, now=1100.0,
        breaker_present=True, accepts_result=False, verbatim_small_source=False,
        source_tokens=10, result_tokens=11, context_length_rescue=False,
    ), **changes)


def attempt(text: str | None = None, error: BaseException | None = None, *,
            ended: float | None = 1100.0, deadline: bool = False) -> Any:
    return values.SummaryAttempt(text, error, {}, "", values.SummaryTiming(1000.0, ended, deadline))


@pytest.mark.parametrize("text,error,changes,kind,reason", [
    ("short", None, {"accepts_result": True}, "ACCEPTED", ""),
    ("long source", None, {"verbatim_small_source": True}, "VERBATIM_SMALL_SOURCE", ""),
    ("long source", None, {"verbatim_small_source": True, "accepts_result": True}, "ACCEPTED", ""),
    ("short", None, {}, "REJECTION", "not_shorter"),
    ("", None, {}, "REJECTION", "no_content"),
    ("short", None, {"breaker_present": False}, "REJECTED", "not_shorter"),
    ("", None, {"breaker_present": False}, "REJECTED", "no_content"),
    (None, RuntimeError("unknown model"), {}, "CONFIG_ERROR", ""),
    (None, RuntimeError("Error code: 400 - unknown model content_filter"), {}, "CONFIG_ERROR", ""),
    (None, RuntimeError("Error code: 400 - content_filter"), {}, "CONTENT_FILTER", ""),
    (None, RuntimeError("context length exceeded"), {"context_length_rescue": True}, "CONTEXT_LENGTH", ""),
    (None, RuntimeError("context length exceeded"), {}, "FAILURE", ""),
    (None, RuntimeError("provider failed"), {}, "FAILURE", ""),
    (None, None, {}, "FAILURE", ""),
    (None, None, {"budget_present": True}, "FAILURE", ""),
    (None, TimeoutError("timeout"), {}, "FAILURE", ""),
    ("long source", None, {"verbatim_small_source": True, "source_tokens": None}, "REJECTION", "not_shorter"),
    ("short", None, {"verbatim_small_source": True, "result_tokens": 9}, "REJECTION", "not_shorter"),
])
def test_outcome_kinds(text: str | None, error: BaseException | None, changes: dict[str, object],
                       kind: str, reason: str) -> None:
    observed = outcomes.classify_summary_outcome(attempt(text, error, ended=1100.0 if changes.get("budget_present") else None), facts(**changes))
    assert (observed.kind.name, observed.reason, observed.hang, observed.record_hang) == (kind, reason, False, False)


@pytest.mark.parametrize("elapsed,grant,host,bound,now,error,kind,hang", [
    (100, 100, False, True, 1100, "timeout", "BUDGET_CUT", True),
    (90, 100, True, True, 1090, "host compression deadline timeout", "BUDGET_CUT", True),
    (89.9, 100, True, True, 1089.9, "host compression deadline timeout", "BUDGET_CUT", False),
    (100, 89.9, False, True, 1100, "timeout", "BUDGET_CUT", False),
    (30, 100, True, True, 1030, "host compression deadline timeout", "BUDGET_CUT", False),
    (100, 30, True, False, 1100, "masked failure", "BUDGET_CUT", True),
    (100, 30, False, False, 1100, "masked failure", "FAILURE", False),
    (30, 30, True, False, 1100, "timeout", "FAILURE", False),
    (99, 100, False, True, 1098, "timeout", "BUDGET_CUT", True),
    (98.9, 100, False, True, 1098, "timeout", "FAILURE", False),
    (20, 100, False, True, 1099, "masked failure", "BUDGET_CUT", False),
    (20, 100, False, True, 1098.9, "masked failure", "FAILURE", False),
])
@pytest.mark.parametrize("breaker", [True, False])
def test_budget_arithmetic(elapsed: float, grant: float, host: bool, bound: bool, now: float,
                           error: str, kind: str, hang: bool, breaker: bool) -> None:
    observed = outcomes.classify_summary_outcome(
        attempt(error=RuntimeError(error), ended=1000 + elapsed, deadline=host),
        facts(budget_present=True, budget_bound=bound, call_timeout=grant, now=now,
              configured_timeout=120.0 if bound else 30.0, breaker_present=breaker))
    assert (observed.kind.name, observed.hang, observed.record_hang) == (kind, hang, hang and breaker)


def test_values_are_frozen_and_classification_does_not_mutate_inputs() -> None:
    captured = facts(budget_present=True, budget_bound=True)
    value = attempt(error=TimeoutError("timeout"))
    before = (repr(captured), repr(value))
    result = outcomes.classify_summary_outcome(value, captured)
    assert (repr(captured), repr(value)) == before
    with pytest.raises(FrozenInstanceError):
        captured.now = 0.0
    with pytest.raises(FrozenInstanceError):
        result.hang = False
