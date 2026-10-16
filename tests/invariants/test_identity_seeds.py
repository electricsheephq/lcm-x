"""Occurrence seeds for compress, #1179 retries and #1286 stale commit proofs."""
from copy import deepcopy
import re
import sys
import types

import pytest

pytest.importorskip("hypothesis")
from hypothesis import settings, strategies as st
from hypothesis.stateful import invariant, rule, run_state_machine_as_test

from tests.invariants.test_exactly_once_machine import (
    KNOWN_OPEN, PROFILE, ExactlyOnceMachine, assert_occurrences, identity,
)


class RetryCollapse(AssertionError):
    """Only the precise #1179 missing-prompt result, not an arbitrary failure."""


class RetryDiscoveryMachine(ExactlyOnceMachine):
    include_known_open = True

    @rule(text=st.sampled_from(("alpha", "same text")), structured=st.booleans(), retry=st.booleans())
    def new_turn(self, text, structured, retry):
        # The open issue's seed is a plain retry. Tool retries have a separate
        # deterministic xfail below so a broader failure cannot hide in this one.
        super().new_turn(text, structured, 0, retry, "lookup")

    @invariant()
    def exactly_once(self):
        try:
            super().exactly_once()
        except AssertionError as error:
            index = self.retry_occurrence_index
            wanted = list(map(identity, self.occurrences))
            actual = list(map(identity, self.rows()))
            if index is not None and actual == wanted[:index] + wanted[index + 1:]:
                raise RetryCollapse(str(error)) from error
            raise


@pytest.mark.xfail(strict=True, reason="#1179: retry loses a new equal-text user occurrence", raises=RetryCollapse)
def test_machine_discovers_retry_1179():
    run_state_machine_as_test(RetryDiscoveryMachine, settings=PROFILE)


@pytest.mark.xfail(strict=True, reason="#1179: retry loses a new equal-text user occurrence", raises=RetryCollapse)
def test_retry_1179():
    machine = RetryDiscoveryMachine()
    try:
        machine.new_turn("alpha", False, False)
        machine.new_turn("same text", False, False)
        machine.exactly_once()
        machine.new_turn("same text", False, True)
        machine.exactly_once()
    finally:
        machine.teardown()


class RetryToolCollapse(AssertionError):
    """The #1179 retry shape also loses the assistant call before its result."""


@pytest.mark.xfail(strict=True, reason="#1179-shaped retry also skips its first assistant tool call",
                   raises=RetryToolCollapse)
def test_retry_1179_with_tool_call():
    machine = RetryDiscoveryMachine()
    try:
        machine.new_turn("alpha", False, False)
        machine.new_turn("same text", False, False)
        machine.exactly_once()
        ExactlyOnceMachine.new_turn(machine, "same text", False, 1, True, "lookup")
        expected, actual = list(map(identity, machine.occurrences)), list(map(identity, machine.rows()))
        try:
            assert_occurrences(machine.occurrences, machine.rows())
        except AssertionError as error:
            index = machine.retry_occurrence_index
            if actual == expected[:index] + expected[index + 2:]:
                raise RetryToolCollapse(str(error)) from error
            raise
    finally:
        machine.teardown()


def test_oracle_reports_duplicates_and_collapses_separately():
    row = {"role": "user", "content": "same text"}
    with pytest.raises(AssertionError, match=r"duplicates=\{.+collapses=\{\}"):
        assert_occurrences([row], [row, row])
    with pytest.raises(AssertionError, match=r"duplicates=\{\}; collapses=\{.+"):
        assert_occurrences([row, row], [row])


def test_oracle_checks_order_and_assistant_tool_metadata():
    user = {"role": "user", "content": "same text"}
    assistant = {"role": "assistant", "content": "same text"}
    with pytest.raises(AssertionError, match="order_mismatch=True"):
        assert_occurrences([user, assistant], [assistant, user])
    for changed in ({"id": "different", "function": {"name": "lookup"}},
                    {"id": "call", "function": {"name": "different"}}):
        expected = dict(assistant, tool_calls=[{"id": "call", "function": {"name": "lookup"}}])
        with pytest.raises(AssertionError, match="collapses="):
            assert_occurrences([expected], [dict(assistant, tool_calls=[changed])])


def test_compress_replay_resume_append():
    machine = ExactlyOnceMachine()
    try:
        for text in ("alpha", "same text"):
            machine.new_turn(text, False, 0, False, "lookup")
        before = machine.engine._dag.get_session_node_count("identity-session")
        machine.compress()
        assert machine.engine._dag.get_session_node_count("identity-session") > before
        assert machine.host[0]["content"].startswith("[Recent Summary (")
        machine.exactly_once()
        machine.replay()
        machine.exactly_once()
        machine.cold_resume()
        machine.exactly_once()
        machine.new_turn("same text", True, 0, False, "inspect")
        machine.exactly_once()
    finally:
        machine.teardown()


class StaleProofCarrierStored(AssertionError):
    """Only an extra carrier after cold resume with a stale durable proof."""


@pytest.mark.xfail(strict=True, reason="#1286: changed sanitized output leaves the durable commit proof stale",
                   raises=StaleProofCarrierStored)
@pytest.mark.parametrize(("counts", "compress_after"), [
    ((1, 1, 0), (0, 0, 2)),
    ((0, 1, 1), (0, 1, 2)),
    ((0, 1, 0), (0, 2, 2)),
], ids=["arc", "objective", "durable"])
def test_cold_stale_proof_1286_does_not_store_carrier(counts, compress_after):
    machine = ExactlyOnceMachine()
    try:
        for tool_count, compressions in zip(counts, compress_after):
            machine.new_turn("alpha", False, tool_count, False, "lookup")
            machine.exactly_once()
            for _ in range(compressions):
                machine.compress()
                machine.exactly_once()
        assert machine.engine._last_compression_status == "sanitized"
        assert KNOWN_OPEN[1286](machine.stale_proof, "cold_resume")
        carrier = deepcopy(machine.host[0])
        # Exercise the excluded transition without going through its rule guard.
        machine.engine.shutdown()
        machine.engine = machine.start_engine()
        machine.host = deepcopy(machine.host)
        machine.engine.ingest(machine.host)
        try:
            machine.exactly_once()
        except AssertionError as error:
            if list(map(identity, machine.rows())) == list(map(identity, [*machine.occurrences, carrier])):
                raise StaleProofCarrierStored(str(error)) from error
            raise
    finally:
        machine.teardown()


def test_cold_compacted_arc_keeps_exact_occurrences():
    """T(1) T(1) T(0) C publishes an arc with a current durable proof."""
    machine = ExactlyOnceMachine()
    try:
        for tool_count in (1, 1, 0):
            machine.new_turn("alpha", False, tool_count, False, "lookup")
        machine.compress()
        assert machine.engine._last_compression_status == "compacted"
        assert "[Session Arc Summary (" in machine.host[0]["content"]
        assert not KNOWN_OPEN[1286](machine.stale_proof, "cold_resume")
        machine.exactly_once()
        machine.cold_resume()
        machine.exactly_once()
    finally:
        machine.teardown()


def test_survival_carrier_1000_is_already_fixed(tmp_path, monkeypatch):
    """#1000 remains open for GA, but #1001 already fixed this rc's carrier.

    Survival fit is outside the append-only machine. Reuse the established
    held-fit construction and apply the same complete-row oracle to its replay.
    """
    from hermes_lcm import escalation, tokens
    from tests.test_issue_650_survival_fit_keeps_summary import _held_fit_after_inplace_boundary, _rough

    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = _rough
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *args, **kwargs:
                        "Earlier turns." + " summary" * 150 + "\nExpand for details about: turns")
    engine, fitted, _remainder = _held_fit_after_inplace_boundary(tmp_path, _rough, monkeypatch)
    try:
        expected = deepcopy(engine._store.get_session_messages("S"))
        reply = {"role": "assistant", "content": "reply CUR", "timestamp": 30_001.0}
        fitted[0]["timestamp"] = 40_000.0
        engine.ingest([*fitted, reply])
        assert_occurrences([*expected, reply], engine._store.get_session_messages("S"))
    finally:
        engine.shutdown()
        tokens._count_tokens_cached.cache_clear()


class HistoricalSteerMachine(ExactlyOnceMachine):
    """#633 sensitivity probe, outside the main append-only input domain.

    A known host prefix insertion supplies the historical seed. The same ordered
    oracle checks one transition; ordinary machine operations stay unchanged.
    """

    def __init__(self):
        super().__init__()
        self.steer = {"role": "user", "content": "repeat this instruction", "display_kind": "steer"}
        self.host = [self.stamp({"role": "user", "content": "plan"}), deepcopy(self.steer),
                     self.stamp({"role": "assistant", "content": "planned"}),
                     self.stamp({"role": "user", "content": "inspect"}),
                     self.stamp({"role": "assistant", "content": "", "tool_calls": [
                         {"id": "historical-call", "type": "function",
                          "function": {"name": "lookup", "arguments": "{}"}}]}),
                     self.stamp({"role": "tool", "content": "done", "tool_call_id": "historical-call"}),
                     self.stamp({"role": "assistant", "content": "inspected"})]
        self.occurrences = deepcopy(self.host)
        self.engine.ingest(self.host)
        self.exactly_once()

    @rule(text=st.sampled_from(("continue", "verify")))
    def new_turn(self, text):
        steer = deepcopy(self.steer)
        batch = [self.stamp({"role": "user", "content": text}),
                 self.stamp({"role": "assistant", "content": "completed " + text})]
        self.host.insert(6, steer)
        self.host.extend(batch)
        self.occurrences.extend(deepcopy([steer, *batch]))
        self.engine.ingest(self.host)


def test_historical_633_machine():
    run_state_machine_as_test(HistoricalSteerMachine, settings=settings(PROFILE, stateful_step_count=1))


def test_known_open_exclusions_track_their_strict_xfails():
    """The main machine may skip an open issue's input shape only while that issue's strict xfail is still here.
    When the fix lands and its xfail is removed, this fails until the KNOWN_OPEN entry goes too."""
    reasons = {str(mark.kwargs.get("reason", ""))
               for value in list(globals().values()) if callable(value)
               for mark in getattr(value, "pytestmark", [])
               if mark.name == "xfail" and mark.kwargs.get("strict")}
    xfailed = {int(found.group(1)) for reason in reasons if (found := re.match(r"#(\d+)", reason))}
    assert set(KNOWN_OPEN) <= xfailed, (set(KNOWN_OPEN) - xfailed)
