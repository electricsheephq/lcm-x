"""Synthetic threshold sweeps at the admission boundary."""

from copy import deepcopy
import hashlib
import json
import time

import pytest

import hermes_lcm.escalation as escalation
from hermes_lcm.dag import SummaryNode
from tests import test_issue_605_foreground_budget as budget_fixtures
from tests.test_issue_605_foreground_budget import (
    SUMMARY,
    _advance_on,
    _compress,
    _engine,
    _pin_lcm_char_counter,  # noqa: F401 - shared autouse fixture
    _view,
)

clock = budget_fixtures.clock


def _sweep_engine(tmp_path, window=1_000_000, **settings):
    engine = _engine(tmp_path, **{
        "context_threshold": 0.5,
        "max_assembly_tokens": window,
        "summary_timeout_ms": 120_000,
        **settings,
    })
    engine.on_session_start("S", platform="telegram", context_length=window, conversation_id="conv")
    return engine


def _seed_prefix(engine, tokens):
    for i in range(2):
        engine._dag.add_node(SummaryNode(
            session_id="S", depth=0, summary=f"Earlier arc {i}. " + "synthetic history " * 40,
            token_count=tokens // 2, source_token_count=tokens, source_ids=[], source_type="messages",
            created_at=float(i + 1),
        ))


def _script_provider(monkeypatch, clock, *, seconds=0, leaf=SUMMARY, completed=None):
    calls = []

    def summarize(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        kind = "leaf" if "user turn" in prompt else "condensation"
        calls.append(kind)
        if timeout is not None and seconds > timeout:
            clock.offset += timeout
            raise TimeoutError("synthetic call timed out")
        clock.offset += seconds
        if completed is not None:
            completed.append(kind)
        return leaf if kind == "leaf" else SUMMARY

    monkeypatch.setattr(escalation, "_invoke_summary_llm", summarize)
    return calls


def _status(engine):
    return engine.get_status()["threshold_full_sweep"]


def _store_rows(engine):
    return engine._store.get_session_messages("S", limit=100_000)


def _snapshot(result):
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize(("estimate", "reserve", "pre_work", "admitted"), [
    (30, 30, 59, False), (115, 0, 55, True), (115, 0, 61, False),
])
def test_first_call_cold_fit_and_stale_escape(monkeypatch, clock, estimate, reserve, pre_work, admitted):
    monkeypatch.setattr(clock, "_real", lambda: 0.0)
    estimates = escalation.ForegroundEstimates()
    if estimate != 30:
        estimates.record_call("", estimate)
    budget = escalation.ForegroundBudget(soft=60, hard=120, configured_timeout=120, estimates=estimates)
    budget.leaf_reserve = reserve
    clock.offset += pre_work
    if admitted:
        assert budget.admit("") == 60
    else:
        with pytest.raises(escalation.SweepBudgetExhausted, match="time budget"):
            budget.admit("")


def test_late_entry_refuses_without_preparing_leaf_or_changing_input(tmp_path, monkeypatch, clock):
    engine = _sweep_engine(tmp_path)
    view = _view()
    before = deepcopy(view)
    calls = _script_provider(monkeypatch, clock, seconds=44)
    engine._foreground_estimates.record_call(engine._primary_summary_route(), 44)
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 80)
    remaining_work = []
    prepare = engine._prepare_leaf_input

    def prepare_leaf(*args, **kwargs):
        remaining_work.append("prepare_leaf_input")
        return prepare(*args, **kwargs)

    monkeypatch.setattr(engine, "_prepare_leaf_input", prepare_leaf)
    try:
        engine.ingest(view)
        rows = _store_rows(engine)
        result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert (calls, remaining_work, result, view) == ([], [], before, before)
        assert _store_rows(engine) == rows
        assert _status(engine)["leaf_passes"] == 0
        assert _status(engine)["stop_reason"] == "time_budget_exhausted"
    finally:
        engine.shutdown()


def _route_chain(budget):
    return escalation._invoke_summary_llm_routes(
        "synthetic source", 50, note_error=lambda error: None, model="primary",
        fallback_models=["fallback"], budget=budget, timeout=60,
    )


@pytest.mark.parametrize(("warm", "cut", "expected"), [
    (None, 22, 33), (55, 35, 55), (30, 40, 60), (45, 50, 60),
])
def test_cut_duration_is_a_lower_bound(monkeypatch, clock, warm, cut, expected):
    monkeypatch.setattr(clock, "_real", lambda: 0.0)
    estimates = escalation.ForegroundEstimates()
    if warm is not None:
        estimates.record_call("primary", warm)
        for _ in range(7):
            estimates.record_call("primary", 15)
    budget = escalation.ForegroundBudget(soft=60, hard=120, configured_timeout=60, estimates=estimates)

    def cutoff(prompt, max_tokens, *, model, timeout, reasoning_effort):
        assert model == "primary"
        clock.offset += timeout
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(escalation, "_invoke_summary_llm", cutoff)
    # Run the attempt at entry with a smaller usable window. The host can try
    # its own fallback inside auxiliary_client.call_llm, outside this route chain.
    budget.leaf_reserve = 115 - cut
    if cut < 30 or (warm is not None and cut < warm):
        # The timing sample models a host deadline reached after admission:
        # route setup consumed part of the admitted window.
        def late_cutoff(prompt, max_tokens, *, model, timeout, reasoning_effort):
            assert model == "primary"
            clock.offset += cut
            raise TimeoutError("host compression deadline timeout")

        monkeypatch.setattr(escalation, "_invoke_summary_llm", late_cutoff)
        budget.leaf_reserve = 0
        clock.offset = 115 - cut
        monkeypatch.setattr(budget, "admit", lambda route: cut)
    with pytest.raises(escalation.SweepBudgetExhausted):
        _route_chain(budget)
    assert estimates.call_estimate("primary", 60) == expected
    estimates.record_call("fallback", 20)
    assert estimates.call_estimate("fallback", 60) == 20
    assert estimates.call_estimate("primary", 60) == expected
    estimates.record_call("primary", 20, completed=False)
    assert estimates.call_estimate("primary", 60) == expected
    estimates.record_call("primary", 20)
    assert estimates.call_estimate("primary", 60) == cut


def test_plugin_fallback_refused_when_its_estimate_does_not_fit(monkeypatch, clock):
    monkeypatch.setattr(clock, "_real", lambda: 0.0)
    estimates = escalation.ForegroundEstimates()
    budget = escalation.ForegroundBudget(soft=60, hard=120, configured_timeout=60, estimates=estimates)
    clock.offset += 35
    calls = []

    def provider(prompt, max_tokens, *, model, timeout, reasoning_effort):
        calls.append((model, timeout))
        if model == "primary":
            clock.offset += 60
            raise TimeoutError("provider timed out")
        return SUMMARY

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    # This is the plugin route chain. The host's recovery ladder runs inside
    # auxiliary_client.call_llm and sees its own deadlines.
    with pytest.raises(escalation.SweepBudgetExhausted):
        _route_chain(budget)
    assert calls == [("primary", 60)] and clock.offset == 95


@pytest.mark.parametrize("tokens", [60_000, 99_000])
def test_over_target_prefix_does_not_take_first_leaf_call(tmp_path, monkeypatch, clock, caplog, tokens):
    engine = _sweep_engine(tmp_path, leaf_chunk_tokens=24_000)
    try:
        _seed_prefix(engine, tokens)
        calls = _script_provider(monkeypatch, clock, seconds=44)
        _compress(engine, _view(240), caplog)
        assert calls == ["leaf"] and _status(engine)["leaf_passes"] == 1
        assert _status(engine)["summary_prefix_target_tokens"] == 50_000
    finally:
        engine.shutdown()


def test_repeated_partial_sweeps_bound_the_prefix(tmp_path, monkeypatch, clock, caplog):
    monkeypatch.setattr(clock, "_real", lambda: 0.0)
    engine = _sweep_engine(tmp_path, leaf_chunk_tokens=24_000)
    try:
        _seed_prefix(engine, 99_000)
        leaf = "Earlier synthetic facts. " * 150
        calls = _script_provider(monkeypatch, clock, seconds=44, leaf=leaf)
        view = _view(1720)
        prefixes, first_calls = [], []
        for _ in range(4):
            prior_count = len(calls)
            view = _compress(engine, view, caplog)
            first_calls.append(calls[prior_count])
            prefixes.append(engine._summary_frontier_tokens())
            engine.note_turn_complete()
        leaf_tokens = prefixes[0] - 99_000
        print(f"partial sweeps: calls={first_calls}, prefixes={prefixes}, first leaf tokens={leaf_tokens}")
        assert first_calls[:3] == ["leaf", "leaf", "condensation"]
        assert 0 < leaf_tokens and max(prefixes) <= 100_000 + leaf_tokens
        assert prefixes[2] < prefixes[1]
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("window", "target", "turns"), [
    (64_000, 8_000, 110), (200_000, 24_000, 240), (272_000, 24_000, 240), (480_000, 24_000, 240),
])
def test_legacy_window_target_order_and_return(tmp_path, monkeypatch, clock, caplog, window, target, turns):
    monkeypatch.setattr(clock, "_real", lambda: 0.0)
    results = []
    for explicit in (False, True):
        root = tmp_path / str(explicit)
        root.mkdir()
        settings = {"summary_prefix_target_tokens": target} if explicit else {}
        engine = _sweep_engine(root, window=window, leaf_chunk_tokens=24_000, **settings)
        try:
            _seed_prefix(engine, target + 2000)
            calls = _script_provider(monkeypatch, clock, seconds=44)
            result = _compress(engine, _view(turns), caplog)
            status = _status(engine)
            assert status["status"] in {"complete", "partial"}
            results.append((status["summary_prefix_target_tokens"], calls, status["stop_reason"], result))
        finally:
            engine.shutdown()
    print(f"window {window}: {results[0][:3]}, returned rows={len(results[0][3])}, sha={_snapshot(results[0][3])}")
    assert results[0] == results[1]
    assert results[0][:2] == (target, ["condensation"])


def test_measured_59_second_setup_does_not_spend_the_first_call_on_condensation(
    tmp_path, monkeypatch, clock, caplog,
):
    engine = _sweep_engine(tmp_path, leaf_chunk_tokens=24_000)
    try:
        _seed_prefix(engine, 60_000)
        calls = _script_provider(monkeypatch, clock, seconds=44)
        _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 59)
        view = _view(240)
        result = _compress(engine, view, caplog)
        if calls:
            assert calls == ["leaf"] and _status(engine)["leaf_passes"] == 1
        else:
            assert result == view and _status(engine)["stop_reason"] == "time_budget_exhausted"
    finally:
        engine.shutdown()


def test_three_late_turns_use_the_existing_budget_hold(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(clock, "_real", lambda: 0.0)
    engine = _sweep_engine(tmp_path)
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 93, calls=None)
    calls = _script_provider(monkeypatch, clock, seconds=44)
    view = _view()
    durations, holds = [], []
    try:
        for _ in range(3):
            engine.ingest(view)
            start = time.monotonic()
            assert engine.compress(view, current_tokens=engine.threshold_tokens + 1) == view
            durations.append(time.monotonic() - start)
            holds.append((engine._sweep_budget_hold_until - time.monotonic(), engine._sweep_budget_hold_turn_paced,
                          engine._hung_summary_strike))
            engine.note_turn_complete()
        print(f"late turns: durations={durations}, calls={calls}, holds={holds}, total={sum(durations)}")
        assert sum(durations) <= 115
        assert durations == [93, 0, 0] and calls == []
        assert holds == [(600, False, None)] * 3
    finally:
        engine.shutdown()


def test_cold_cut_raises_the_next_estimate_until_a_call_completes(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(clock, "_real", lambda: 0.0)
    engine = _sweep_engine(tmp_path, window=200_000, summary_model="primary")
    for turn, seconds in enumerate((80, 79, 62), start=1):
        _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, seconds, calls=(turn,))
    completed = []
    calls = _script_provider(monkeypatch, clock, seconds=40, completed=completed)
    view = _view()
    passes = []
    try:
        engine.ingest(view)
        rows = _store_rows(engine)
        assert engine._foreground_estimates.call_estimate("primary", 120) == 30
        for turn in range(3):
            start, started_count, completed_count = time.monotonic(), len(calls), len(completed)
            result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
            elapsed = time.monotonic() - start
            estimate = engine._foreground_estimates.call_estimate("primary", 120)
            passes.append((elapsed, len(calls) - started_count, len(completed) - completed_count, estimate))
            if turn < 2:
                assert result == view and _store_rows(engine) == rows
                assert _status(engine)["stop_reason"] == "time_budget_exhausted"
                assert engine._sweep_budget_hold_until - time.monotonic() == 600
                assert engine._summary_circuit_breaker.allows("primary") and engine._hung_summary_strike is None
            engine.note_turn_complete()
            clock.offset += 601  # Let the existing holds expire without adding to compress time.
        print(f"cold route passes (elapsed, started, completed, next estimate): {passes}")
        assert passes == [(115, 1, 0, 52.5), (79, 0, 0, 52.5), (102, 1, 1, 40)]
        assert _status(engine)["leaf_passes"] == 1
    finally:
        engine.shutdown()
