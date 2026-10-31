"""Pin the provider outcome/route table before replacing the summary side channel.

Four sweeps expose the shipped two-failure and six-rejection thresholds, then
prove that an open circuit skips calls. Output-token overruns, broken envelopes,
and foreground time cuts are intentionally separate outcomes. #1120 is already
fixed on this base, so its correct-behavior rows pass rather than falsely xfail.
"""

import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.escalation as escalation
from tests.harness import ScriptedSummarizer, pin_char_counter

SOURCE = "source text " * 400
SUMMARY = ScriptedSummarizer.healthy_reply
OVER_BUDGET = "detail " * 80 + "\nExpand for details about: turns"
L3 = "source text source text source text source text source text"
MASKED_ERROR = "invalid argument: OpenAI Responses request is missing input"


@dataclass(frozen=True)
class Outcome:
    name: str
    # Calls per route on each sweep (hangs use only the first available route).
    attempts: tuple[int, ...]
    error: str = ""
    classification: str | None = None
    contract: str = ""
    result: str = L3
    level: int | None = 3


OUTCOMES = [
    Outcome("valid", (1, 1, 1, 1), result=SUMMARY, level=1),
    Outcome("empty", (2, 2, 2, 0), classification="rejected"),
    Outcome("contract-rejected", (2, 2, 2, 0), classification="rejected", contract="closing_hint"),
    # Current behavior accepts a reply above max_tokens if it is shorter than the source.
    Outcome("over-budget-text", (1, 1, 1, 1), result=OVER_BUDGET, level=1),
    Outcome("truncated-envelope", (2, 2, 2, 0), classification="rejected", contract="envelope"),
    Outcome("provider-exception", (2, 0, 0, 0), "provider unavailable", "provider_failure"),
    Outcome("configured-timeout", (2, 0, 0, 0), "configured timeout expired", "provider_failure"),
    Outcome("config-error", (1, 0, 0, 0), "Error code: 404 - unknown model", "config_error"),
    Outcome("content-filter", (2, 2, 2, 2), "Error code: 400 - content_filter"),
    Outcome("context-length-rescue", (2, 2, 2, 2), "maximum context length exceeded"),
    # #1209: each full-window cut opens its route; later sweeps try the next route, then L3.
    Outcome("foreground-budget-cut", (1, 1, 0, 0), "request timed out", "hung", level=None),
    # #1209: a masked error at the budget-bound full-window deadline is also a hang.
    Outcome("masked-budget-cut-1120", (1, 1, 0, 0), MASKED_ERROR, "hung", level=None),
    # #1209: an installed deadline reached after the full host window opens the route.
    Outcome("masked-host-deadline-1120", (1, 1, 0, 0), MASKED_ERROR, "hung", level=None),
]


def _normalise(text):
    # Include unit-suffixed values such as 300s, but preserve L1/L2/L3 labels.
    return re.sub(r"\b\d+(?:\.\d+)?", "<n>", text)


def _attempt_logs(outcome, model):
    """Literal observable log lines, independent of production logger templates."""
    route = model or "<task-default>"
    display = model or "<default>"
    if outcome.error:
        return [("DEBUG" if outcome.classification == "config_error" else "WARNING",
                 f"LLM summarization failed: {outcome.error}")]
    if outcome.level == 1:
        return []
    if outcome.contract:
        lines = [
            ("WARNING", "LCM summary discarded output that violated the integrity contract "
             f"(model={display}, check={outcome.contract}); escalating"),
            ("INFO", f"LCM summary contract shape: check={outcome.contract} last_line=missing"),
        ]
    else:
        lines = [("WARNING", f"LCM summary discarded empty output (model={display}); escalating")]
    reason = "no_content" + (f" contract={outcome.contract}" if outcome.contract else "")
    return lines + [("WARNING", f"LCM summary result rejected (reason={reason}, "
                     f"source_tokens=<n>, result_tokens=<n>, model={route})")]


def _skip_logs(models, config=False):
    severity = "DEBUG" if config else "WARNING"
    return [(severity, f"LCM summary route skipped by open circuit: {model or '<task-default>'}")
            for model in models] + [(severity, "LCM summary fallback chain exhausted: "
                                    "all routes are temporarily open")]


def _expected_logs(outcome, models, sweep):
    attempts = outcome.attempts[sweep]
    if outcome.level is None:
        # #1209: skip prior hung routes, log the new hang once, or exhaust both L1/L2 chains.
        if sweep < len(models):
            skipped = [("WARNING", f"LCM summary route skipped by open circuit: {model or '<task-default>'}")
                       for model in models[:sweep]]
            route = models[sweep] or "<task-default>"
            return skipped + _attempt_logs(outcome, models[sweep]) + [
                ("WARNING", f"LCM summary route circuit opened for {route}: "
                 "route used the whole window; cooldown=<n>s")]
        return _skip_logs(models) * 2 + [("DEBUG", "L3 deterministic truncation (<n> tokens)")]
    if outcome.level == 1:
        return [("DEBUG", "L1 summarization succeeded (<n> tokens)")]
    lines = []
    for attempt in range(attempts):
        for model in models:
            route = model or "<task-default>"
            lines += _attempt_logs(outcome, model)
            if outcome.classification == "config_error":
                sent = f"model {model!r}" if model else "no model (summary_model unset)"
                lines.append(("WARNING", "LCM summary route cannot serve the summary model: "
                              f"provider=scripted model={model or 'default'}; LCM-X sent {sent}; "
                              f"circuit {route} open for <n>s per failure until a summary succeeds "
                              f"({outcome.error}). Fix: set auxiliary.compression.provider and "
                              "auxiliary.compression.model together in the profile's config.yaml, "
                              "or a consistent model.provider / model.default pair"))
            elif outcome.classification == "provider_failure" and attempt == 1:
                lines.append(("WARNING", f"LCM summary route circuit opened for {route} "
                              "after <n> failure(s); cooldown=<n>s"))
            elif outcome.classification == "rejected" and sweep == 2 and attempt == 1:
                lines.append(("WARNING", f"LCM summary route circuit opened for {route} "
                              "after <n> rejected result(s); cooldown=<n>s"))
    # Config errors open on L1; later sweeps skip both L1 and L2.
    for _ in range(2 - attempts):
        lines += _skip_logs(models, config=outcome.classification == "config_error")
    return lines + [("DEBUG", "L3 deterministic truncation (<n> tokens)")]


@pytest.mark.parametrize("models", [("",), ("primary", "fallback")],
                         ids=["default-route", "primary-plus-fallback"])
@pytest.mark.parametrize("outcome", OUTCOMES, ids=lambda row: row.name)
def test_summary_outcome_table(monkeypatch, caplog, outcome, models):
    pin_char_counter(monkeypatch)
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(escalation, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(escalation, "secrets", SimpleNamespace(token_hex=lambda size: "a" * (2 * size)))
    monkeypatch.setattr(escalation, "_route_info_support", {})
    # Retain the real adapter/contract checks: the harness observes its requested
    # seam, while this offline host supplies raw replies rather than faking the
    # very outcome classification that the table is intended to preserve.
    provider = ScriptedSummarizer(mode="scripted", fn=escalation._call_llm_for_summary).install(monkeypatch)
    host = ModuleType("agent.auxiliary_client")
    host.deadline = None

    @contextmanager
    def deadline_scope(deadline):
        previous, host.deadline = host.deadline, deadline
        try:
            yield
        finally:
            host.deadline = previous

    host.aux_stream_deadline = deadline_scope
    host._current_aux_stream_deadline = lambda: host.deadline
    seen_timeouts = []

    def call_llm(*, messages, max_tokens, timeout, route_info, model=None, **kwargs):
        route_info.update(provider="scripted", model=model or "default")
        seen_timeouts.append(timeout)
        if outcome.name == "configured-timeout":
            clock.now += timeout
        elif outcome.level is None:
            assert host.deadline is not None
            clock.now = host.deadline
        if outcome.error:
            error_type = TimeoutError if outcome.name in {"configured-timeout", "foreground-budget-cut"} else RuntimeError
            raise error_type(outcome.error)
        if outcome.name == "empty":
            content = ""
        else:
            nonce = messages[0]["content"].split('<lcm-summary nonce="', 1)[1].split('"', 1)[0]
            body = outcome.result if outcome.level == 1 else "Earlier turns describe a completed task."
            content = f'<lcm-summary nonce="{nonce}">\n{body}'
            if outcome.name != "truncated-envelope":
                content += "\n</lcm-summary>"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    host.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", host)
    import agent
    monkeypatch.setattr(agent, "auxiliary_client", host, raising=False)

    breaker = escalation.SummaryCircuitBreaker()
    guard = escalation.SummarySpendGuard()
    estimates = escalation.ForegroundEstimates()
    caplog.set_level(logging.DEBUG, logger="hermes_lcm.escalation")
    expected_spend = 0
    keys = [model or "<task-default>" for model in models]
    for sweep, attempts in enumerate(outcome.attempts):
        # #1209: a full-window hang opens one route per sweep; no calls once all are open.
        if outcome.classification == "hung":
            attempts = int(sweep < len(models))
        caplog.clear()
        start = len(provider.calls)
        hard = 120.0 if outcome.name == "masked-host-deadline-1120" else 45.0 if outcome.level is None else 1000.0
        budget = escalation.ForegroundBudget(soft=0, hard=hard, configured_timeout=60.0, estimates=estimates)
        provenance = {}
        kwargs = dict(source_tokens=1201, token_budget=50, l3_truncate_tokens=16,
                      model=models[0], fallback_models=list(models[1:]), timeout=60.0,
                      circuit_breaker=breaker, spend_guard=guard, budget=budget,
                      provenance=provenance, context_length_rescue=True)
        # #1209: only a newly hung route cuts the sweep; all-open chains reach L3 instead.
        if outcome.level is None and attempts:
            with pytest.raises(escalation.SweepBudgetExhausted) as raised:
                escalation.summarize_with_escalation(SOURCE, **kwargs)
            assert str(raised.value) == "summariser call cut by the foreground budget"
            assert raised.value.reason == budget.stop_reason == "time_budget_exhausted"
            assert provenance == {}
        else:
            # #1209: after every route hangs, the next sweeps return deterministic L3.
            level = 3 if outcome.level is None else outcome.level
            assert escalation.summarize_with_escalation(SOURCE, **kwargs) == (outcome.result, level)
            expected_provenance = {"model": models[0] if outcome.level == 1 else "deterministic"}
            if outcome.name in {"content-filter", "context-length-rescue"}:
                expected_provenance["content_filter_error" if outcome.name == "content-filter"
                                    else "context_length_error"] = True
            assert provenance == expected_provenance
            assert budget.stop_reason == ""

        # #1209: subsequent sweeps call the first route that has not hung, not the primary again.
        called_models = (models[sweep:sweep + 1] if outcome.classification == "hung"
                         else models[:1] if outcome.level == 1 else models)
        assert [(call.model, call.max_tokens) for call in provider.calls[start:]] == [
            (model, ceiling) for ceiling in (100, 50)[:attempts] for model in called_models]
        assert seen_timeouts[start:] == [40.0 if hard == 45.0 else 60.0] * (len(provider.calls) - start)
        assert host.deadline is None
        expected_spend += bool(attempts)
        assert len(guard._calls) == expected_spend
        assert guard._backoff_until == 0.0
        assert budget.slot_taken is bool(attempts)
        assert budget.leaves == 0 and budget.progress == ""

        failures = 2 if outcome.classification == "provider_failure" else 0
        rejections = min(2 * (sweep + 1), 6) if outcome.classification == "rejected" else 0
        assert breaker._failures == (dict.fromkeys(keys, failures) if failures else {})
        assert breaker._rejections == (dict.fromkeys(keys, rejections) if rejections else {})
        opened = bool(failures or rejections == 6 or outcome.classification == "config_error")
        # #1209: only the routes that have consumed a full window are open.
        opened_keys = set(keys[:sweep + 1]) if outcome.classification == "hung" else set(keys) if opened else set()
        assert set(breaker._open_until) == opened_keys
        assert breaker._config_episodes == (set(keys) if outcome.classification == "config_error" else set())
        # #1209: a hung route is refused while an untried fallback remains usable.
        assert [breaker.allows(model) for model in models] == [key not in opened_keys for key in keys]
        # Time spent on later routes reduces the primary's cooldown in the
        # configured-timeout cell; route_status reports the earliest reopening.
        seconds = 240 if opened and len(models) == 2 and outcome.name == "configured-timeout" else 300
        # #1209: status stays closed until every route hangs; the primary keeps its hung class.
        all_open = len(opened_keys) == len(keys)
        # #1209: time spent hanging the fallback reduces the primary's remaining cooldown.
        if outcome.classification == "hung" and len(models) == 2:
            seconds = 300 - (hard - 5)
        assert breaker.route_status(models) == {
            "state": "open" if all_open else "closed", "seconds_left": seconds if all_open else 0,
            "last_error_class": outcome.classification,
            "provider": None if outcome.level is None else "scripted",
            "model": None if outcome.level is None else models[0] or "default",
        }
        observed_logs = [(record.levelname, _normalise(record.getMessage())) for record in caplog.records
                         if record.name == "hermes_lcm.escalation"]
        assert observed_logs == [(level, _normalise(line)) for level, line in _expected_logs(outcome, models, sweep)]
