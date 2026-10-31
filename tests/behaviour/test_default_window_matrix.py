"""Small-window host loops with shipped defaults and one control changed at a time."""

import logging
from collections import Counter

import pytest

from hermes_lcm.dag import SummaryNode
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import (
    ScriptedSummarizer, assistant, call, engine_context, host_call,
    pin_char_counter, prose, tool, user,
)


WINDOWS = (16, 24, 32, 48, 64, 96, 127)
CONTROLS = (
    "defaults", "survival-fit-off", "full-sweep-off", "externalization-off", "assembly-cap",
)
OBJECTIVE = "⟦objective⟧"
OVERFLOW_WARNING = "LCM overflow recovery could not get under cap="


class ImmediateRecompaction(AssertionError):
    """The #1123 expectation cannot hide a fit, objective, or accounting failure."""


def _cells():
    for mode in ("healthy", "outage"):
        for control in CONTROLS:
            for window_k in WINDOWS:
                repeats = (mode, control, window_k) == ("healthy", "full-sweep-off", 127)
                marks = [pytest.mark.xfail(
                    strict=True, raises=ImmediateRecompaction,
                    reason="#1123: emitted view remains over the threshold; next host call compacts again",
                )] if repeats else []
                yield pytest.param(window_k, control, mode, marks=marks,
                                   id=f"{mode}-{control}-{window_k}K")


def _check_sent(engine, result, caplog, objective):
    assert result.sent_tokens == count_messages_tokens(result.view)
    if result.sent_tokens > engine.context_length:
        assert result.overflow_failed, "over-window view without overflow failure flag"
        assert any(record.levelno >= logging.WARNING and OVERFLOW_WARNING in record.message
                   for record in caplog.records), "over-window view without overflow warning"
    # This small objective fits even in the smallest window. The provider may
    # project long prose; the contract here is preservation of the sentinel.
    assert any(objective in row.get("content", "") for row in result.view)


def _check_stub_reservations(engine):
    """An implicit result and an explicit identical stub must cost exactly once.

    At the exact boundary, double reservation evicts a call or prefix.
    Compare both paths including summaries; the isolated node cell below checks
    under-reservation with a summary that must be excluded.
    """
    system = {"role": "system", "content": "Inspect the records."}
    tail = [assistant("Checking the final records.", calls=[call("missing-0"), call("missing-1")])]
    stubs = [engine._missing_tool_result_stub(f"missing-{i}") for i in range(2)]
    complete = [*tail, *stubs]
    expected = engine._assemble_context(system, complete, include_lcm_note=False, persist=False)
    cap = count_messages_tokens(expected)
    for source in (tail, complete):
        actual = engine._assemble_context(
            system, source, assembly_cap_override=cap, include_lcm_note=False, persist=False,
        )
        assert actual == expected, "stub reservation changed the exact-boundary assembly"
        assert count_messages_tokens(actual) == cap
        results = Counter(row["tool_call_id"] for row in actual if row.get("role") == "tool")
        assert results == {"missing-0": 1, "missing-1": 1}


def test_stub_reservation_excludes_summary_at_exact_cap(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=16 * 1024) as engine:
        _check_stub_reservations(engine)
        system = {"role": "system", "content": "Inspect the records."}
        tail = [assistant("Checking the final records.", calls=[call("missing-0"), call("missing-1")])]
        complete = [*tail, *(engine._missing_tool_result_stub(f"missing-{i}") for i in range(2))]
        expected = engine._assemble_context(system, complete, include_lcm_note=False, persist=False)
        cap = count_messages_tokens(expected)
        # This small node fits only if the selector forgets the implicit stubs.
        node = SummaryNode(session_id=engine.current_session_id, summary="STUB_BUDGET_SENTINEL",
                           token_count=5, created_at=1.0)
        engine._dag.add_node(node)
        for source in (tail, complete):
            actual = engine._assemble_context(
                system, source, assembly_cap_override=cap, include_lcm_note=False, persist=False,
            )
            assert actual == expected, "a summary consumed the missing-result stub reservation"
        # Raising the cap by the measured summary cost admits exactly this node.
        with_node = engine._assemble_context(system, complete, include_lcm_note=False, persist=False)
        assert any(node.summary in row.get("content", "") for row in with_node)
        admitted = engine._assemble_context(
            system, tail, assembly_cap_override=count_messages_tokens(with_node),
            include_lcm_note=False, persist=False,
        )
        assert admitted == with_node


@pytest.mark.parametrize("window_k,control,mode", _cells())
def test_default_window_matrix(tmp_path, monkeypatch, caplog, window_k, control, mode):
    pin_char_counter(monkeypatch)
    provider = ScriptedSummarizer(mode).install(monkeypatch)
    window = window_k * 1024
    overrides = {
        "defaults": {},
        "survival-fit-off": {"survival_fit": False},
        "full-sweep-off": {"threshold_full_sweep_enabled": False},
        "externalization-off": {"large_output_externalization_enabled": False},
        "assembly-cap": {"max_assembly_tokens": window * 3 // 4},
    }[control]
    view = [{"role": "system", "content": "Keep the objective and inspect the records."}]
    known_recompaction = (mode, control, window_k) == ("healthy", "full-sweep-off", 127)
    # Extra input pressure leaves a substantial margin after the single sweep.
    pressure_target = window * (23 if known_recompaction else 21) // 20
    compressed_rounds = 0
    repeated_rounds = []
    with engine_context(tmp_path, window=window, **overrides) as engine:
        for turn in range(3):
            objective = f"{OBJECTIVE} round {turn}: inspect the records and preserve the evidence."
            view.append(user(objective))
            # Short assistant rows exercise rc7's #1227 outage rescue. Include
            # one larger tool payload to exercise default externalization too.
            index = 0
            current_tokens = count_messages_tokens(view)
            while current_tokens <= pressure_target:
                call_id = f"round-{turn}-call-{index}"
                pair = [
                    assistant(prose(1100, seed=index), calls=[call(call_id)]),
                    tool(call_id, prose(12_100 if index == 0 else 1100, seed=index + 1)),
                ]
                view.extend(pair)
                current_tokens += count_messages_tokens(pair)
                index += 1
            assert count_messages_tokens(view) > engine.threshold_tokens
            caplog.clear()
            result = host_call(engine, view)
            assert result.compressed, "pressured host call did not attempt compaction"
            compressed_rounds += 1
            _check_sent(engine, result, caplog, objective)
            calls_before_followup = len(provider.calls)
            if known_recompaction and turn == 0:
                assert result.sent_tokens >= engine.threshold_tokens + 1000
            # Feed the emitted view back just as the host does. This is a new
            # call without new pressure, rather than a second compress() call.
            caplog.clear()
            following = host_call(engine, result.view)
            _check_sent(engine, following, caplog, objective)
            if mode == "outage" and turn == 0:
                assert len(provider.calls) == calls_before_followup
            if known_recompaction and turn == 0:
                assert len(provider.calls) > calls_before_followup, "#1123 must call the summary provider again"
            if following.compressed:
                repeated_rounds.append(turn)
            view = following.view
        assert compressed_rounds == 3
        assert provider.calls, "session never exercised the summary provider"
        _check_stub_reservations(engine)
        if mode == "outage" and control in {"defaults", "full-sweep-off", "externalization-off"}:
            # The rejected first summary arms no hold. The round-0 follow-up
            # sees the open circuit without calling the provider and arms #608.
            assert repeated_rounds == [0]
            # Check the held gate itself too: a fitted view can fall below the
            # threshold and hide a missing hold in an individual window cell.
            assert not engine.should_compress(engine.threshold_tokens)
            return
        # Defer this assertion until every round and the accounting check have
        # run, including in the exact cells carrying the #1123 expectation.
        if repeated_rounds:
            raise ImmediateRecompaction(f"immediate re-compaction in rounds {repeated_rounds}")
