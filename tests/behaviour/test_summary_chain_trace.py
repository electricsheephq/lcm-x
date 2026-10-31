"""Ordered summary-chain effects captured before the outcome extraction.

Set LCM_REGENERATE_SUMMARY_CHAIN=1 to regenerate the reviewed golden fixture.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

escalation = importlib.import_module("hermes_lcm.escalation")
GOLDEN = Path(__file__).parents[1] / "fixtures" / "summary_chain_trace.json"
Trace = list[list[object]]


@dataclass(frozen=True)
class Case:
    name: str
    text: str | None = None
    error: str = ""
    contract: str = ""
    breaker: bool = True
    budget: bool = False
    elapsed: float = 1.0
    usable: float = 100.0
    timeout: float = 120.0
    host: bool = False
    accept: bool | None = False
    small: bool = False
    rescue: bool = False
    open: bool = False
    config_episode: bool = False
    spend: bool = True
    hang_opens: bool = True
    round_two: bool = False


CASES = [
    Case("accepted", text="short", accept=True, budget=True),
    Case("accepted-no-predicate", text="short", accept=None),
    Case("verbatim-small", text="long source", small=True),
    *[Case(f"{reason}-{contract or 'no-contract'}-{breaker}", text=text,
           contract=contract, breaker=breaker)
      for reason, text in [("not-shorter", "long source"), ("no-content", "")]
      for contract in ["", "closing_hint"] for breaker in [False, True]],
    Case("config", error="Error code: 404 - unknown model"),
    Case("config-no-breaker", error="unknown model", breaker=False),
    Case("config-before-filter", error="Error code: 400 - unknown model content_filter"),
    Case("content-filter", error="Error code: 400 - content_filter"),
    Case("context-rescue", error="maximum context length exceeded", rescue=True),
    Case("context-no-rescue", error="maximum context length exceeded"),
    Case("failure", error="provider unavailable"),
    Case("failure-no-error"),
    Case("failure-no-breaker", error="provider unavailable", breaker=False),
    Case("configured-timeout", error="request timed out", budget=True, timeout=30, elapsed=30),
    Case("budget-hang", error="request timed out", budget=True, elapsed=100),
    Case("budget-no-breaker", error="request timed out", budget=True, elapsed=100, breaker=False),
    Case("budget-hang-still-allowed", error="request timed out", budget=True, elapsed=100, hang_opens=False),
    Case("budget-partial", error="masked failure", budget=True, elapsed=40, usable=40),
    Case("early-host-fence", error="host compression deadline timeout", budget=True, elapsed=30, host=True),
    Case("below-hang-window", error="host compression deadline timeout", budget=True, elapsed=85, host=True),
    Case("below-granted-window", error="request timed out", budget=True, elapsed=85, usable=85),
    Case("host-no-progress", error="masked failure", budget=True, elapsed=100, timeout=30, host=True),
    Case("open-circuit", open=True),
    Case("open-config-circuit", open=True, config_episode=True),
    Case("spend-exhausted", spend=False, budget=True),
    Case("fallback-hang-round-two", error="host compression deadline timeout", budget=True,
         elapsed=99, round_two=True),
]


class Breaker:
    def __init__(self, trace: Trace, case: Case) -> None:
        self.trace, self.case = trace, case
        self.blocked: set[str] = {"scope:primary"} if case.open else set()

    def allows(self, key: str) -> bool:
        verdict = key not in self.blocked
        self.trace.append(["allows", key, verdict])
        return verdict

    def in_config_episode(self, key: str) -> bool:
        self.trace.append(["in_config_episode", key, self.case.config_episode])
        return self.case.config_episode

    def note_route(self, key: str, route: dict[str, object]) -> None:
        self.trace.append(["note_route", key, route])

    def record_success(self, key: str) -> None:
        self.trace.append(["record_success", key])

    def record_hang(self, key: str) -> None:
        self.trace.append(["record_hang", key])
        if self.case.hang_opens:
            self.blocked.add(key)

    def record_config_error(self, key: str, *, route: dict[str, object], error: BaseException,
                            sent_model: str) -> None:
        self.trace.append(["record_config_error", key, route, str(error), sent_model])

    def record_failure(self, key: str) -> None:
        self.trace.append(["record_failure", key])

    def record_rejection(self, key: str) -> None:
        self.trace.append(["record_rejection", key])


class Budget:
    def __init__(self, trace: Trace, case: Case, clock: SimpleNamespace) -> None:
        self.trace, self.clock = trace, clock
        self.hard, self.reserve, self.leaf_reserve = 115.0, 5.0, 10.0
        self.configured_timeout = case.timeout
        self.usable_deadline = clock.now + case.usable
        self.slot_taken = False
        self._hung_route: str | None = None

    @property
    def hung_route(self) -> str | None:
        return self._hung_route

    @hung_route.setter
    def hung_route(self, key: str) -> None:
        self.trace.append(["hung_route", key])
        self._hung_route = key

    def admit(self, key: str) -> float:
        left: float = self.usable_deadline - self.clock.now
        self.trace.append(["admit", key, left])
        return left

    def record_call(self, key: str, started: float, ended: float) -> None:
        self.trace.append(["record_call", key, started, ended])

    def stop(self, message: str) -> BaseException:
        self.trace.append(["stop", message])
        return escalation.SweepBudgetExhausted(message)  # type: ignore[no-any-return]


class SpendGuard:
    def __init__(self, trace: Trace, allowed: bool) -> None:
        self.trace, self.allowed = trace, allowed

    def try_record_call(self) -> bool:
        self.trace.append(["spend", self.allowed])
        return self.allowed


class Provenance(dict[str, object]):
    def __init__(self, trace: Trace) -> None:
        super().__init__()
        self.trace = trace

    def __setitem__(self, key: str, value: object) -> None:
        self.trace.append(["provenance", key, value])
        super().__setitem__(key, value)


class TraceHandler(logging.Handler):
    def __init__(self, trace: Trace) -> None:
        super().__init__()
        self.trace = trace

    def emit(self, record: logging.LogRecord) -> None:
        self.trace.append(["log", record.levelname, record.msg,
                           [str(x) if isinstance(x, BaseException) else x for x in (record.args or ())],
                           record.getMessage()])


def observe(case: Case, monkeypatch: pytest.MonkeyPatch) -> Trace:
    trace: Trace = []
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(escalation, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(escalation, "_host_stream_deadline_seam", lambda: case.host)
    monkeypatch.setattr(escalation, "count_tokens", len)
    breaker = Breaker(trace, case)
    guard = SpendGuard(trace, case.spend)
    provenance = Provenance(trace)
    calls = 0

    def provider(prompt: str, max_tokens: int, model: str = "", timeout: float | None = None,
                 reasoning_effort: str = "") -> str | None:
        nonlocal calls
        calls += 1
        trace.append(["provider", model, max_tokens, timeout, reasoning_effort])
        escalation._summary_call.route = {"provider": "scripted", "model": model}
        escalation._summary_call.contract = case.contract
        if case.round_two and calls == 1:
            clock.now += 1.0
            raise RuntimeError("quick primary failure")
        if case.round_two and calls == 3:
            clock.now += 1.0
            return "short"
        clock.now += case.elapsed
        if case.error:
            raise RuntimeError(case.error)
        return case.text

    def accepts(text: str) -> bool:
        trace.append(["accepts", text])
        return bool(case.accept or (case.round_two and calls == 3))

    monkeypatch.setattr(escalation, "_call_llm_for_summary", provider)
    handler = TraceHandler(trace)
    escalation.logger.addHandler(handler)
    # setLevel (not a raw attribute patch) clears the logging cache, so earlier tests cannot leave DEBUG disabled.
    previous_level = escalation.logger.level
    escalation.logger.setLevel(logging.DEBUG)
    try:
        for _ in range(2 if case.round_two else 1):
            budget = Budget(trace, case, clock) if case.budget else None
            try:
                result = escalation._invoke_summary_llm_chain(
                    "prompt", 10, model="primary", fallback_models=["fallback"] if case.round_two else [],
                    timeout=case.timeout, reasoning_effort="low", circuit_breaker=breaker if case.breaker else None,
                    spend_guard=guard, accepts_result=accepts if case.accept is not None else None,
                    source_tokens=10, provenance=provenance, route_key_prefix="scope:", budget=budget,
                    verbatim_small_source=case.small, context_length_rescue=case.rescue)
                trace.append(["return", result])
            except escalation.SweepBudgetExhausted as exc:
                trace.append(["raise", type(exc).__name__, str(exc), exc.reason])
            trace.append(["state", dict(provenance), budget.hung_route if budget else None,
                          budget.slot_taken if budget else None])
    finally:
        escalation.logger.removeHandler(handler)
        escalation.logger.setLevel(previous_level)
    return trace


def test_summary_chain_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    for case in CASES:
        with monkeypatch.context() as patch:
            observed[case.name] = observe(case, patch)
    encoded = json.dumps(observed, indent=2, sort_keys=True) + "\n"
    if os.environ.get("LCM_REGENERATE_SUMMARY_CHAIN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(encoded)
    assert encoded == GOLDEN.read_text()
