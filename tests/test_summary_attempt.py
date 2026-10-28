"""Characterize the summary adapter seams before extracting attempt records (I7)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

escalation = importlib.import_module("hermes_lcm.escalation")
SUMMARY = "Earlier turns.\nExpand for details about: turns"


@pytest.mark.parametrize("supports_timeout", [False, True])
def test_provider_seam_keeps_legacy_signatures(
    monkeypatch: pytest.MonkeyPatch, supports_timeout: bool,
) -> None:
    calls: list[tuple[str, int, str, float | None]] = []

    def legacy(prompt: str, max_tokens: int, model: str = "") -> str:
        calls.append((prompt, max_tokens, model, None))
        return SUMMARY

    def timed(prompt: str, max_tokens: int, model: str = "", timeout: float | None = None) -> str:
        calls.append((prompt, max_tokens, model, timeout))
        return SUMMARY

    monkeypatch.setattr(escalation, "_call_llm_for_summary", timed if supports_timeout else legacy)
    assert escalation._invoke_summary_llm("source", 50, model="primary", timeout=30.0) == SUMMARY
    assert calls == [("source", 50, "primary", 30.0 if supports_timeout else None)]


@pytest.mark.parametrize("raises", [False, True])
def test_invoke_seam_resets_metadata_and_clears_deadline(
    monkeypatch: pytest.MonkeyPatch, raises: bool,
) -> None:
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(escalation, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(escalation, "_host_stream_deadline_seam", lambda: True)
    budget = escalation.ForegroundBudget(
        soft=0, hard=100, configured_timeout=30, estimates=escalation.ForegroundEstimates(),
    )
    breaker = escalation.SummaryCircuitBreaker()
    escalation._summary_call.route = {"provider": "stale"}
    escalation._summary_call.error = RuntimeError("stale")
    escalation._summary_call.contract = "stale"
    seen: list[float] = []

    def invoke(
        prompt: str, max_tokens: int, model: str = "", timeout: float | None = None,
        reasoning_effort: str = "",
    ) -> str:
        assert (prompt, max_tokens, model, timeout, reasoning_effort) == ("source", 50, "primary", 30, "low")
        assert escalation._summary_call.route == {}
        assert escalation._summary_call.error is None
        assert escalation._summary_call.contract == ""
        seen.append(escalation._summary_call.stream_deadline)
        escalation._summary_call.route = {"provider": "scripted", "model": model}
        clock.now += 2
        if raises:
            raise RuntimeError("provider unavailable")
        return SUMMARY

    monkeypatch.setattr(escalation, "_invoke_summary_llm", invoke)
    result = escalation._invoke_summary_llm_chain(
        "source", 50, model="primary", timeout=30, reasoning_effort="low",
        circuit_breaker=breaker, budget=budget,
    )
    assert result == (None if raises else SUMMARY)
    assert seen == [budget.usable_deadline]
    assert escalation._summary_call.stream_deadline is None
    assert budget.estimates.calls == {"primary": [2.0]}
    assert breaker.route_status(["primary"]) == {
        "state": "closed", "seconds_left": 0,
        "last_error_class": "provider_failure" if raises else None,
        "provider": "scripted", "model": "primary",
    }


@pytest.mark.parametrize(
    ("message", "status", "expected"),
    [
        ("unknown model", 404, (True, False, False)),
        ("maximum context length exceeded", 400, (False, True, False)),
        ("content_filter", 400, (False, False, True)),
        ("too many tokens", 429, (False, False, False)),
        ("model not found timeout", 404, (False, False, False)),
        ("provider unavailable", 503, (False, False, False)),
    ],
)
def test_error_predicate_table(message: str, status: int, expected: tuple[bool, bool, bool]) -> None:
    error = RuntimeError(message)
    error.status_code = status  # type: ignore[attr-defined]
    assert (
        escalation.is_summary_route_config_error(error),
        escalation.is_summary_context_length_error(error),
        escalation.is_summary_content_filter_error(error),
    ) == expected
