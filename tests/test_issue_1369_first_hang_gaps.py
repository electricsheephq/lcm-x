"""#1369: intervening retries and stored progress respect the first-hang episode."""

import pytest

from tests import test_issue_1096_turn_end_probe as turn_probe
from tests.test_issue_605_foreground_budget import _engine, _view
from tests.test_issue_909_condensation_starves_leaf import _frontier

SUMMARY = turn_probe.SUMMARY
_pass = turn_probe._pass
_pin_lcm_char_counter = turn_probe._pin_lcm_char_counter
_provider = turn_probe._provider
clock = turn_probe.clock
episode = turn_probe.episode


def test_forced_retry_hang_disarms_turn_end_release_before_hold_expires(episode, clock, monkeypatch):
    engine, calls, view = episode
    result = _pass(engine, view)
    strike, until = engine._hung_summary_strike, engine._sweep_budget_hold_until
    assert strike == ("conv", "m1") and engine._sweep_budget_hold_turn_paced
    clock.offset += 301
    assert engine._sweep_budget_hold_active() and engine._summary_circuit_breaker.allows("m1")

    engine.compress(result, current_tokens=200_001)
    assert len(calls) == 2
    assert engine._hung_summary_strike == strike
    assert engine._sweep_budget_hold_until == until  # forced passes do not start another hold
    assert not engine._sweep_budget_hold_turn_paced
    assert engine._sweep_budget_hold_active()
    breaker = engine._summary_circuit_breaker
    releases = []
    monkeypatch.setattr(breaker, "release_hang", releases.append)
    engine.note_turn_complete()
    assert not releases and not breaker.allows("m1")


@pytest.mark.parametrize("progress", ["leaf", "condensation"])
def test_hang_after_stored_progress_gets_exactly_one_turn_end_retry(tmp_path, monkeypatch, clock, progress):
    engine = _engine(tmp_path, summary_model="m1", summary_timeout_ms=120_000,
                     max_assembly_tokens=200_000, foreground_soft_seconds=0)
    calls = _provider(monkeypatch, clock)
    from hermes_lcm.escalation import _invoke_summary_llm as hung

    def progress_then_hang(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        if not calls:
            calls.append((model, timeout))
            clock.offset += 1
            return SUMMARY
        return hung(prompt, max_tokens, model=model, timeout=timeout, reasoning_effort=reasoning_effort)

    monkeypatch.setattr("hermes_lcm.escalation._invoke_summary_llm", progress_then_hang)
    monkeypatch.setattr(engine._foreground_estimates, "call_estimate", lambda route, ceiling: 30.0)
    view = _view(12 if progress == "leaf" else 2)
    engine.ingest(view)
    if progress == "condensation":
        _frontier(engine)
    stored = engine._store.get_session_messages("S")
    try:
        result = _pass(engine, view)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["leaf_passes"] == int(progress == "leaf")
        assert telemetry["pre_leaf_condensation_passes"] == int(progress == "condensation")
        assert telemetry["stop_reason"] == "time_budget_exhausted" and len(calls) == 2
        assert engine._dag.get_session_nodes("S")
        assert engine._store.get_session_messages("S") == stored
        assert engine._hung_summary_strike == ("conv", "m1")
        assert engine._sweep_budget_hold_turn_paced and engine._sweep_budget_hold_active()
        breaker = engine._summary_circuit_breaker
        assert not breaker.allows("m1")
        releases = []
        release = breaker.release_hang

        def record_release(route):
            releases.append(route)
            release(route)

        monkeypatch.setattr(breaker, "release_hang", record_release)
        engine.note_turn_complete()
        engine.note_turn_complete()
        assert releases == ["m1"] and breaker.allows("m1")
        nodes = engine._dag.get_session_nodes("S")
        _pass(engine, result)
        assert len(calls) == 3 and engine._dag.get_session_nodes("S") == nodes
        assert engine._hung_summary_strike == ("conv", "m1")
        assert not engine._sweep_budget_hold_turn_paced and not breaker.allows("m1")
        engine.note_turn_complete()
        assert releases == ["m1"] and not breaker.allows("m1")
    finally:
        engine.shutdown()
