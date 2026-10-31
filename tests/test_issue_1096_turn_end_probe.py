"""#1096 Option A: one turn-end retry, then the existing timed backstop.

The provider/clock seam exercises real escalation, hang classification and leaf publication.
"""

import logging
import time

import pytest

from hermes_lcm.escalation import SummaryCircuitBreaker
from tests.test_issue_605_foreground_budget import _engine, _view
from tests import test_issue_1137_hung_route_circuit as hung_route
from tests.test_issue_909_condensation_starves_leaf import _frontier

SUMMARY = hung_route.SUMMARY
_provider = hung_route._provider
_pin_lcm_char_counter = hung_route._pin_lcm_char_counter
clock = hung_route.clock


@pytest.fixture
def episode(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, summary_model="m1", summary_timeout_ms=120_000, max_assembly_tokens=200_000)
    calls = _provider(monkeypatch, clock)
    view = _view(2)  # exactly one non-tail turn, so a healthy probe makes exactly one call
    engine.ingest(view)
    try:
        yield engine, calls, view
    finally:
        engine.shutdown()


def _pass(engine, view):
    return engine.compress(view, current_tokens=engine.threshold_tokens + 1)


def test_first_hang_holds_only_this_turn_and_is_fit_only_at_ceiling(episode, caplog):
    engine, calls, view = episode
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = _pass(engine, view)
        assert engine._sweep_budget_hold_active()
        assert engine._sweep_budget_hold_turn_paced
        assert not engine._no_progress_hold_active()
        assert engine._hung_summary_strike == ("conv", "m1")
        assert not engine._summary_circuit_breaker.allows("m1")
        assert engine._compression_block_reason() == "cooldown:lcm_sweep_budget"
        assert not engine.should_compress(engine.threshold_tokens + 1)
        ceiling = engine._survival_ceiling()
        assert engine.should_compress(ceiling)
        assert engine._hold_fit_only_applies(ceiling)
        engine.compress(result, current_tokens=ceiling)
    assert len(calls) == 1 and not engine._dag.get_session_nodes("S")
    arms = [r for r in caplog.records if "hung summary hold armed" in r.getMessage()]
    assert len(arms) == 1 and arms[0].levelno == logging.INFO and "m1" in arms[0].getMessage()


def test_turn_end_allows_one_healthy_call_and_leaf_clears_episode(episode, monkeypatch, caplog):
    engine, calls, view = episode
    result = _pass(engine, view)

    def healthy(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append((model, timeout))
        return SUMMARY

    monkeypatch.setattr("hermes_lcm.escalation._invoke_summary_llm", healthy)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        engine.note_turn_complete()
        engine.note_turn_complete()  # duplicate host notification cannot release twice
    assert engine._summary_circuit_breaker.allows("m1")
    assert not engine._sweep_budget_hold_active() and not engine._no_progress_hold_active()
    assert engine.should_compress(engine.threshold_tokens + 1)
    _pass(engine, result)
    assert [model for model, _ in calls] == ["m1", "m1"]
    assert engine._dag.get_session_nodes("S")
    assert engine._hung_summary_strike is None
    assert not engine._sweep_budget_hold_turn_paced
    assert not engine._sweep_budget_hold_active() and not engine._no_progress_hold_active()
    assert engine._summary_route_status()["last_error_class"] is None
    releases = [r for r in caplog.records if "hung summary hold released" in r.getMessage()]
    assert len(releases) == 1 and releases[0].levelno == logging.INFO and "m1" in releases[0].getMessage()


def test_second_hang_keeps_both_timed_holds_and_strike_after_backstop(episode, clock):
    engine, calls, view = episode
    result = _pass(engine, view)
    engine.note_turn_complete()
    _pass(engine, result)
    assert len(calls) == 2
    assert not engine._sweep_budget_hold_turn_paced
    assert engine._sweep_budget_hold_active() and engine._no_progress_hold_active()
    until, no_progress = engine._sweep_budget_hold_until, engine._no_progress_hold
    engine.note_turn_complete()
    assert engine._sweep_budget_hold_until == until and engine._no_progress_hold == no_progress
    assert not engine._summary_circuit_breaker.allows("m1")
    assert until - time.monotonic() == pytest.approx(600, abs=1)
    clock.offset += 601
    assert not engine._sweep_budget_hold_active() and not engine._no_progress_hold_active()
    assert engine._hung_summary_strike == ("conv", "m1")
    _pass(engine, result)
    assert len(calls) == 3 and not engine._sweep_budget_hold_turn_paced
    assert engine._sweep_budget_hold_active() and engine._no_progress_hold_active()


def test_bypass_recovery_preserves_turn_end_probe(episode, monkeypatch, caplog):
    engine, calls, view = episode
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = _pass(engine, view)
        assert engine._sweep_budget_hold_active() and engine._sweep_budget_hold_turn_paced
        result = engine.compress(result, current_tokens=engine.threshold_tokens + 1, bypass_cooldown=True)
        assert len(calls) == 1
        assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "summary_route_unavailable"
        assert not engine._summary_circuit_breaker.allows("m1")
        assert engine._sweep_budget_hold_active() and engine._sweep_budget_hold_turn_paced
        assert engine._hung_summary_strike == ("conv", "m1")
        engine.note_turn_complete()
    arms = [r for r in caplog.records if "hung summary hold armed" in r.getMessage()]
    assert len(arms) == 1 and arms[0].levelno == logging.INFO
    assert engine._summary_circuit_breaker.allows("m1")
    assert not engine._sweep_budget_hold_active()

    def healthy(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append((model, timeout))
        return SUMMARY

    monkeypatch.setattr("hermes_lcm.escalation._invoke_summary_llm", healthy)
    assert engine.should_compress(engine.threshold_tokens + 1)
    _pass(engine, result)
    assert [model for model, _ in calls] == ["m1", "m1"]
    assert engine._dag.get_session_nodes("S")
    assert engine._hung_summary_strike is None


def test_route_unavailable_without_turn_paced_hold_stays_timed(episode, caplog):
    engine, calls, view = episode
    breaker = engine._summary_circuit_breaker
    for _ in range(breaker.failure_threshold):
        breaker.record_failure("m1")
    assert not engine._sweep_budget_hold_turn_paced
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        engine.compress(view, current_tokens=engine.threshold_tokens + 1, bypass_cooldown=True)
        assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "summary_route_unavailable"
        assert engine._sweep_budget_hold_active() and not engine._sweep_budget_hold_turn_paced
        until = engine._sweep_budget_hold_until
        engine.note_turn_complete()
    assert not calls and not breaker.allows("m1")
    assert engine._sweep_budget_hold_active() and engine._sweep_budget_hold_until == until
    assert engine._hung_summary_strike is None
    assert not any("hung summary hold armed" in r.getMessage() for r in caplog.records)


def test_no_turn_end_keeps_the_600_second_backstop(episode, clock):
    engine, calls, view = episode
    result = _pass(engine, view)
    armed = time.monotonic()
    assert engine._sweep_budget_hold_until - armed == pytest.approx(600, abs=1)
    assert engine._summary_route_seconds_left() == pytest.approx(300, abs=1)
    clock.offset += 301
    assert engine._summary_circuit_breaker.allows("m1")
    assert not engine.should_compress(engine.threshold_tokens + 1)
    assert len(calls) == 1  # circuit expiry alone does not end the host threshold hold
    clock.offset += 300
    assert engine.should_compress(engine.threshold_tokens + 1)
    _pass(engine, result)
    assert len(calls) == 2 and engine._no_progress_hold_active()
    assert not engine._sweep_budget_hold_turn_paced  # no turn end did not erase strike 1


@pytest.mark.parametrize("late_pass", ["forced_overflow", "pre_leaf_condensation"])
def test_expired_turn_paced_hold_cannot_release_a_later_hang(episode, monkeypatch, clock, late_pass):
    engine, calls, view = episode
    result = _pass(engine, view)
    assert len(calls) == 1 and engine._sweep_budget_hold_turn_paced
    clock.offset += 601  # no turn end before the original hold and circuit expire
    assert not engine._sweep_budget_hold_active()
    assert engine._summary_circuit_breaker.allows("m1")

    if late_pass == "forced_overflow":
        engine.compress(result, current_tokens=200_001)
        assert len(calls) == 2
    else:
        source_ids = _frontier(engine)
        # Keep the timing estimate cold so pre-leaf condensation has its reserved leaf window.
        monkeypatch.setattr(engine._foreground_estimates, "call_estimate", lambda route, ceiling: 30.0)
        hung = hung_route.escalation._invoke_summary_llm

        def condense_then_hang(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
            if len(calls) == 1:
                calls.append((model, timeout))
                return SUMMARY
            return hung(prompt, max_tokens, model=model, timeout=timeout, reasoning_effort=reasoning_effort)

        monkeypatch.setattr("hermes_lcm.escalation._invoke_summary_llm", condense_then_hang)
        _pass(engine, result)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["pre_leaf_condensation_passes"] == 1
        assert telemetry["leaf_passes"] == 0
        assert len(calls) == 3
        assert all(engine._dag.get_node(node_id) is not None for node_id in source_ids)

    assert not engine._sweep_budget_hold_turn_paced  # #1369: every repeat hang disarms turn-end release
    breaker = engine._summary_circuit_breaker
    assert breaker.route_status(["m1"])["last_error_class"] == "hung"
    assert not breaker.allows("m1")
    calls_before_turn_end = len(calls)
    engine.note_turn_complete()
    circuit_still_open = not breaker.allows("m1")
    _pass(engine, result)
    assert len(calls) == calls_before_turn_end  # no summary call inside the new 300 s cooldown
    assert circuit_still_open


@pytest.mark.parametrize("reset", ["conversation", "session", "profile"])
def test_foreign_or_bypassed_turn_end_cannot_release_and_resets_drop_strike(episode, monkeypatch, reset):
    engine, _, view = episode
    _pass(engine, view)
    until = engine._sweep_budget_hold_until
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_conversation_id", "other")
        engine.note_turn_complete()
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_bypasses_lcm_context_management", lambda: True)
        engine.note_turn_complete()
    assert engine._sweep_budget_hold_until == until and engine._sweep_budget_hold_turn_paced
    assert not engine._summary_circuit_breaker.allows("m1")
    if reset == "profile":
        engine._reset_profile_runtime_state()
    else:
        engine.on_session_start("S" if reset == "conversation" else "S2", platform="telegram",
                                context_length=200_000, conversation_id="conv2")
        engine.note_turn_complete()
        assert not engine._summary_circuit_breaker.allows("m1")
    assert engine._hung_summary_strike is None
    assert not engine._sweep_budget_hold_active()


def test_fallback_waiver_counts_as_first_strike(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, summary_model="m1", summary_fallback_models=["m2"], summary_timeout_ms=120_000)
    calls = _provider(monkeypatch, clock)
    view = _view(2)
    engine.ingest(view)
    try:
        result = _pass(engine, view)
        assert engine._hung_summary_strike == ("conv", "m1")
        assert not engine._sweep_budget_hold_active() and not engine._no_progress_hold_active()
        engine.note_turn_complete()
        assert not engine._summary_circuit_breaker.allows("m1")  # waiver must not half-open the primary
        _pass(engine, result)
        assert [model for model, _ in calls] == ["m1", "m2"]
        assert engine._dag.get_session_nodes("S") and engine._hung_summary_strike is None
    finally:
        engine.shutdown()


def test_prework_budget_stop_retains_both_timed_holds(episode, monkeypatch, clock):
    engine, calls, view = episode
    original = engine._get_store_id_map_for_messages

    def spend_budget(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.offset += 120
        return result

    monkeypatch.setattr(engine, "_get_store_id_map_for_messages", spend_budget)
    _pass(engine, view)
    assert not calls
    assert engine._hung_summary_strike is None and not engine._sweep_budget_hold_turn_paced
    assert engine._sweep_budget_hold_active() and engine._no_progress_hold_active()
    until, no_progress = engine._sweep_budget_hold_until, engine._no_progress_hold
    engine.note_turn_complete()
    assert engine._sweep_budget_hold_until == until and engine._no_progress_hold == no_progress


@pytest.mark.parametrize("outcome", ["hung", "provider_failure", "config_error", "rejected"])
def test_release_hang_never_releases_a_different_error(episode, outcome):
    engine, _, view = episode
    _pass(engine, view)
    breaker = engine._summary_circuit_breaker
    if outcome == "provider_failure":
        for _ in range(breaker.failure_threshold):
            breaker.record_failure("m1")
    elif outcome == "config_error":
        breaker.record_config_error("m1")
    elif outcome == "rejected":
        for _ in range(breaker.rejection_threshold):
            breaker.record_rejection("m1")
    opened = dict(breaker._open_until)
    counters = (dict(breaker._failures), dict(breaker._rejections), set(breaker._config_episodes))
    engine.note_turn_complete()
    assert breaker.allows("m1") is (outcome == "hung")
    if outcome != "hung":
        assert breaker._open_until == opened
    assert (breaker._failures, breaker._rejections, breaker._config_episodes) == counters
    assert breaker.route_status(["m1"])["last_error_class"] == outcome
    # Normalization and route isolation also apply to the default/rollup keys.
    isolated = SummaryCircuitBreaker()
    isolated.record_hang(None)
    isolated.record_hang("<rollup>:m1")
    isolated.release_hang("")
    assert isolated.allows(None) and not isolated.allows("<rollup>:m1")
