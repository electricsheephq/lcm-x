"""Occurrence seeds for compress, recovery, rotation, retries and stale proofs."""
from copy import deepcopy
import json
import re
import subprocess
import sys
import types

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st
from hypothesis.stateful import invariant, rule, run_state_machine_as_test

from tests.invariants.test_exactly_once_machine import (
    KNOWN_OPEN, PROFILE, REPROJECTED_USER,
    ExactlyOnceMachine, RecoveryMachine, RotationMachine, assert_occurrences, identity,
)


class RetryCollapse(AssertionError):
    """Only the precise #1179 missing-prompt result, not an arbitrary failure."""


class RetryDiscoveryMachine(ExactlyOnceMachine):
    include_known_open = True

    @rule(text=st.sampled_from(("alpha", "same text")), structured=st.booleans(), retry=st.booleans())
    def new_turn(self, text, structured, retry):
        # Plain retries and tool retries both retain every occurrence.
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


def test_machine_discovers_retry_1179():
    run_state_machine_as_test(RetryDiscoveryMachine, settings=PROFILE)


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


def test_retry_1179_with_tool_call():
    machine = RetryDiscoveryMachine()
    try:
        machine.new_turn("alpha", False, False)
        machine.new_turn("same text", False, False)
        machine.exactly_once()
        ExactlyOnceMachine.new_turn(machine, "same text", False, 1, True, "lookup")
        expected, actual = list(map(identity, machine.occurrences)), list(map(identity, machine.rows()))
        try:
            assert_occurrences(machine.occurrences, machine.rows(), engine=machine.engine)
        except AssertionError as error:
            index = machine.retry_occurrence_index
            if actual == expected[:index] + expected[index + 2:]:
                raise RetryToolCollapse(str(error)) from error
            raise
    finally:
        machine.teardown()


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("entry", ["ingest", "compress"])
@pytest.mark.parametrize("metadata,fresh", [
    ("new-both", True), ("new-stamp", True), ("new-uid", True),
    ("same", False), ("missing-old", False), ("missing-new", False),
    ("unstamped", False),
])
def test_retry_prefix_occurrence_metadata_1179(anchor, tool, entry, metadata, fresh):
    """The prefix proof respects known conflicts and keeps legacy unknowns."""
    machine = ExactlyOnceMachine()
    machine.patch.setenv("LCM_IDENTITY_ANCHOR", str(int(anchor)))
    try:
        original = [machine.stamp(dict(role=role, content=text)) for role, text in
                    [("user", "alpha"), ("assistant", "first"),
                     ("user", "same text"), ("assistant", "old answer")]]
        retry = deepcopy(original[2])
        if metadata in ("new-both", "new-stamp"):
            retry["timestamp"] = 50.0
        if metadata in ("new-both", "new-uid"):
            retry["message_uid"] = "retry-occurrence"
        for row in ([original[2]] if metadata == "missing-old" else
                    [retry] if metadata == "missing-new" else
                    [original[2], retry] if metadata == "unstamped" else []):
            row.pop("timestamp", None)
            row.pop("message_uid", None)
        machine.engine.ingest(original)
        host = deepcopy(original[:2]) + [retry]
        if entry == "compress":
            machine.engine.compress(host, current_tokens=1)
        else:
            machine.engine.ingest(host)
        assert_occurrences(original + ([retry] if fresh else []), machine.rows(), engine=machine.engine)
        if not fresh:
            machine.engine.ingest(deepcopy(host))
            assert_occurrences(original, machine.rows(), engine=machine.engine)
            return
        suffix = []
        if tool:
            suffix = [machine.stamp(dict(role="assistant", content="calling", tool_calls=[
                dict(id="retry-call", type="function", function=dict(name="lookup", arguments="{}"))])),
                machine.stamp(dict(role="tool", content="result", tool_call_id="retry-call"))]
        suffix.append(machine.stamp(dict(role="assistant", content="new answer")))
        host.extend(suffix)
        expected = original + ([retry] if fresh else []) + suffix
        machine.engine.ingest(host)
        assert_occurrences(expected, machine.rows(), engine=machine.engine)
        machine.engine.ingest(deepcopy(host))
        assert_occurrences(expected, machine.rows(), engine=machine.engine)
        # A fully stamped retry also replays after a cold start without repair.
        if metadata == "new-both" and anchor:
            machine.engine.shutdown()
            machine.engine = machine.start_engine()
            machine.engine.ingest(deepcopy(host))
            assert_occurrences(expected, machine.rows(), engine=machine.engine)
    finally:
        machine.teardown()


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("metadata,fresh", [
    ("new-both", True), ("new-stamp", True), ("new-uid", True),
    ("same", False), ("missing-old", False), ("missing-new", False),
])
def test_compacted_prefix_occurrence_metadata_1179(anchor, metadata, fresh):
    """A real compaction's output prefix must not hide replacement occurrences."""
    machine = ExactlyOnceMachine()
    machine.patch.setenv("LCM_IDENTITY_ANCHOR", str(int(anchor)))
    try:
        if metadata == "missing-old":
            machine.patch.setattr(machine, "stamp", lambda row: dict(row))
        for text in ("alpha", "same text"):
            machine.new_turn(text, False, 0, False, "lookup")
        machine.compress()
        assert machine.engine._last_compression_status == "compacted"
        assert machine.engine._compress_commit_proof
        assert [identity(row) for row in machine.host[-2:]] == [
            identity(row) for row in machine.occurrences[-2:]]
        if metadata == "missing-old":
            assert all(row.get("timestamp") is None and row.get("message_uid") is None
                       for row in machine.host[-2:])
        host = deepcopy(machine.host)
        for index, row in enumerate(host[-2:]):
            if metadata in ("new-both", "new-stamp", "missing-old"):
                row["timestamp"] = 50.0 + index
            if metadata in ("new-both", "new-uid", "missing-old"):
                row["message_uid"] = f"compacted-retry-{index}"
            if metadata == "missing-new":
                row.pop("timestamp", None)
                row.pop("message_uid", None)
        expected = machine.occurrences + (host[-2:] if fresh else [])
        machine.engine.ingest(host)
        assert_occurrences(expected, machine.rows(), engine=machine.engine)
        machine.engine.ingest(deepcopy(host))
        assert_occurrences(expected, machine.rows(), engine=machine.engine)
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


@pytest.mark.parametrize("field", ["arguments", "type"])
def test_oracle_keeps_tool_payloads_distinct(field):
    call = {"id": "call", "type": "function", "function": {"name": "lookup", "arguments": '{"query":1}'}}
    other = deepcopy(call)
    if field == "arguments":
        other["function"]["arguments"] = '{"query":2}'
    else:
        other["type"] = "changed"
    expected = {"role": "assistant", "content": "calling", "tool_calls": [call]}
    changed = dict(expected, tool_calls=[other])
    assert identity(expected) != identity(changed)
    with pytest.raises(AssertionError, match="collapses="):
        assert_occurrences([expected, changed], [expected, expected])


def test_oracle_normalizes_json_arguments_only():
    def row(arguments):
        return {"role": "assistant", "tool_calls": [{"id": "call", "type": "function",
                "function": {"name": "lookup", "arguments": arguments}}]}

    assert_occurrences([row('{"b": [2, 3], "a":1}')], [row('{"a": 1, "b":[2,3]}')])
    for raw in ("invalid JSON", "", None, 7, {"raw": "object"},
                ' {"x":1,"x":2} ', '{"nested":[{"x":1,"x":2}]}'):
        assert identity(row(raw))[-1][0][-1] == raw
    with pytest.raises(AssertionError, match="collapses="):
        assert_occurrences([row('{"x":1,"x":2}')], [row('{"x":2}')])


def test_rotation_tool_arguments_and_active_ownership():
    machine = RotationMachine()
    try:
        for _ in range(2):
            machine.new_turn("alpha", False, 0, False, "lookup")
        machine.rotate()
        assert machine.session_id != "identity-session"
        machine.new_turn("alpha", False, 3, False, "lookup",
                         ('{"query":2}', '{"query":0}', '{"query":1}'))
        calls = machine.occurrences[-5]["tool_calls"]
        assert len({json.loads(call["function"]["arguments"])["query"] for call in calls}) == 3
        assert [row["session_id"] for row in machine.rows()] == [
            "identity-session"] * 4 + [machine.session_id] * 6
        for operation in (machine.replay, machine.cold_resume):
            operation()
            machine.exactly_once()
    finally:
        machine.teardown()


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


@pytest.mark.parametrize("text", ["alpha", "same text"], ids=["merged-carrier", "user-projection"])
def test_recover_replay_resume_append(text):
    """Both recovery output shapes use the complete, unchanged occurrence oracle."""
    machine = RecoveryMachine()
    try:
        for _ in range(3):
            machine.new_turn(text, False, 0, False, "lookup")
        ledger = deepcopy(machine.occurrences)
        machine.recover()
        assert machine.occurrences == ledger
        if text == "alpha":
            assert machine.host[0]["content"].startswith("[Recent Summary (")
            assert machine.engine._generated_context_carrier_remainder(machine.host[0]) == "alpha"
            assert machine.host[-2]["content"] == "alpha"
        else:
            assert "[LCM survival fit: this user message " in machine.host[0]["content"]
        machine.exactly_once()
        machine.replay()
        machine.exactly_once()
        machine.cold_resume()
        machine.exactly_once()
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.exactly_once()
    finally:
        machine.teardown()


class ReprojectedUserStored(AssertionError):
    """Only the extra user projection after a repeat fit misbinds its source."""


@pytest.mark.parametrize("marker_only", [False, True], ids=["equal-copy", "persisted-marker"])
def test_rotation_no_progress_fires_no_hooks(marker_only):
    machine = RotationMachine()
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        previous = deepcopy(machine.host)

        def unchanged(messages, **kwargs):
            machine.engine._last_compression_status = "sanitized"
            return [dict(row, _db_persisted=True) for row in messages] if marker_only else deepcopy(messages)

        def unexpected_hook(*args, **kwargs):
            pytest.fail("unchanged compression must not fire session hooks")

        with machine.patch.context() as patch:
            patch.setattr(machine.engine, "compress", unchanged)
            patch.setattr(machine.engine, "on_session_end", unexpected_hook)
            patch.setattr(machine.engine, "on_session_start", unexpected_hook)
            machine.rotate()
        assert machine.host == previous
        assert machine.session_ids == ["identity-session"]
        cold_restarts = 0
        start = machine.start_engine

        def counted_start():
            nonlocal cold_restarts
            cold_restarts += 1
            return start()

        machine.patch.setattr(machine, "start_engine", counted_start)
        old_engine = machine.engine
        machine.cold_resume()
        print(f"marker_only={marker_only}; cold_restarts={cold_restarts}")
        assert cold_restarts == 1
        assert machine.engine is not old_engine
        assert not machine.stale_proof
        machine.exactly_once()
    finally:
        machine.teardown()


@pytest.mark.parametrize("turns", [1, 2], ids=["no-rewrite", "published-control"])
def test_rotation_without_rewrite_preserves_occurrences(turns):
    machine = RotationMachine()
    try:
        for _ in range(turns):
            machine.new_turn("alpha", False, 0, False, "lookup")
        machine.rotate()
        assert len(machine.session_ids) == turns
        machine.exactly_once()
        machine.cold_resume()
        machine.exactly_once()
    finally:
        machine.teardown()


@pytest.mark.parametrize("resume", [True, False], ids=["cold-control", "warm-audit"])
def test_rotation_warm_audit_preserves_tail(resume):
    machine = RotationMachine()
    try:
        for _ in range(2):
            machine.new_turn("alpha", False, 0, False, "lookup")
        machine.rotate()
        assert machine.engine._last_compression_status == "compacted"
        assert machine.engine._identity_anchor_chain() == ["identity-session"]
        machine.exactly_once()
        if resume:
            machine.cold_resume()
            machine.exactly_once()
        machine.replay()
        machine.exactly_once()
    finally:
        machine.teardown()


@pytest.mark.parametrize("condense", [False, True], ids=["published-control", "sanitized"])
def test_rotation_stale_proof_preserves_tail(condense):
    """Host ancestry keeps rotated sanitized output exact; #1286 is in-place."""
    machine = RotationMachine()
    try:
        for count in (1, 1, 0):
            machine.new_turn("alpha", False, count, False, "lookup")
        if condense:
            machine.compress()
        machine.rotate()
        assert machine.stale_proof == condense
        machine.exactly_once()
        # Bypass the inherited #1286 cold guard to prove this rotation control.
        machine.engine.shutdown()
        machine.engine = machine.start_engine()
        machine.engine.ingest(machine.host)
        machine.exactly_once()
    finally:
        machine.teardown()


def test_rotation_cold_resume_then_repeated_compress():
    """TIME-1: T(0) T(1) O K C C, with committed host ancestry."""
    machine = RotationMachine()
    try:
        for count in (0, 1):
            machine.new_turn("alpha", False, count, False, "lookup")
            machine.exactly_once()
        for operation in (machine.rotate, machine.cold_resume, machine.compress, machine.compress):
            operation()
            machine.exactly_once()
    finally:
        machine.teardown()


def test_rotation_resume_replay_append_compress():
    machine = RotationMachine()
    try:
        for _ in range(2):
            for text in ("alpha", "same text"):
                machine.new_turn(text, False, 0, False, "lookup")
                machine.exactly_once()
            ledger = deepcopy(machine.occurrences)
            machine.rotate()
            assert machine.engine._last_compression_status == "compacted"
            assert machine.occurrences == ledger
            machine.exactly_once()
            machine.cold_resume()
            assert machine.engine._session_id == machine.session_ids[-1]
            machine.exactly_once()
            machine.replay()
            machine.exactly_once()
        machine.new_turn("βeta", True, 1, False, "inspect")
        machine.exactly_once()
        machine.compress()
        machine.exactly_once()
    finally:
        machine.teardown()


@pytest.mark.parametrize("recoveries", [
    pytest.param(1, id="single-fit-control"),
    pytest.param(2, id="repeated-fit", marks=pytest.mark.xfail(
        strict=True, reason=REPROJECTED_USER, raises=ReprojectedUserStored)),
])
def test_repeated_recovery_preserves_user_source(recoveries):
    machine = RecoveryMachine()
    try:
        for _ in range(2):
            machine.new_turn("same text", True, 0, False, "lookup")
        for _ in range(recoveries):
            machine.recover()
            machine.exactly_once()
        assert bool(machine.recovery_aliases) == (recoveries == 2)
        projected_user = deepcopy(machine.host[0])
        # Bypass the input exclusion; keep exactly_once as the sole judge.
        machine.engine.shutdown()
        machine.engine = machine.start_engine()
        machine.host = deepcopy(machine.host)
        machine.engine.ingest(machine.host)
        try:
            machine.exactly_once()
        except AssertionError as error:
            if list(map(identity, machine.rows())) == list(map(identity, [*machine.occurrences, projected_user])):
                raise ReprojectedUserStored(str(error)) from error
            raise
    finally:
        machine.teardown()


@pytest.mark.parametrize("first_text", [
    pytest.param("alpha", id="distinct-source-control"),
    # #1403: the accepted compaction prefix keeps the warm ingest from re-storing the projected user.
    pytest.param("βeta", id="equal-source-warm"),
])
def test_recovery_reprojection_warm_ingest(first_text):
    machine = RecoveryMachine()
    try:
        for text in (first_text, "βeta"):
            machine.new_turn(text, True, 0, False, "lookup")
        machine.compress()
        machine.exactly_once()
        machine.recover()
        machine.exactly_once()
        assert bool(machine.recovery_aliases) == (first_text == "βeta")
        projected_user = deepcopy(machine.host[0])
        # Exercise the excluded warm ingest through the original compress rule.
        ExactlyOnceMachine.compress(machine)
        try:
            machine.exactly_once()
        except AssertionError as error:
            if list(map(identity, machine.rows())) == list(map(identity, [*machine.occurrences, projected_user])):
                raise ReprojectedUserStored(str(error)) from error
            raise
    finally:
        machine.teardown()


def test_survival_carrier_1000_is_already_fixed(tmp_path, monkeypatch):
    """#1000 remains open for GA, but #1001 already fixed this rc's carrier.

    Reuse the established
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
        assert_occurrences([*expected, reply], engine._store.get_session_messages("S"), engine=engine)
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
    from tests.invariants.test_retry_r5 import test_retry_host_merged_scaffold_after_compaction

    seeds = [*globals().values(), test_retry_host_merged_scaffold_after_compaction]
    reasons = {str(mark.kwargs.get("reason", ""))
               for value in seeds if callable(value)
               for mark in getattr(value, "pytestmark", [])
               if mark.name == "xfail" and mark.kwargs.get("strict")}
    reasons.update(str(mark.kwargs.get("reason", ""))
                   for value in list(globals().values()) if callable(value)
                   for mark in getattr(value, "pytestmark", []) if mark.name == "parametrize"
                   for parameter in mark.args[1] if hasattr(parameter, "marks")
                   for mark in parameter.marks if mark.name == "xfail" and mark.kwargs.get("strict"))
    xfailed = {int(found.group(1)) for reason in reasons if (found := re.match(r"#(\d+)", reason))}
    xfailed.update(reason for reason in reasons if re.match(r"(?:#\d+:|UNFILED: )", reason))
    assert set(KNOWN_OPEN) <= xfailed, (set(KNOWN_OPEN) - xfailed)


@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotation"])
def test_host_merge_replay_resume_append(in_place):
    from tests.invariants.test_exactly_once_machine import HostMergeMachine

    machine = HostMergeMachine()
    try:
        for _ in range(3):
            machine.new_turn("alpha", False, 0, False, "lookup")
        ledger = deepcopy(machine.occurrences)
        machine.host_merge(in_place)
        assert machine.occurrences == ledger
        assert machine.host[0].get("_merged_turn_prefix")
        machine.exactly_once()
        machine.new_turn("same text", False, 0, False, "lookup")
        machine.exactly_once()
        machine.replay()
        machine.exactly_once()
        machine.cold_resume()
        machine.exactly_once()
        machine.new_turn("βeta", True, 1, False, "inspect")
        machine.exactly_once()
    finally:
        machine.teardown()


def test_gateway_reset_retains_deliveries_across_fresh_engines():
    from tests.invariants.test_exactly_once_machine import GatewayResetMachine

    machine = GatewayResetMachine()
    try:
        for _ in range(3):
            machine.new_turn("alpha", False, 0, False, "lookup")
        machine.compress()
        machine.exactly_once()
        for _ in range(2):
            ledger = deepcopy(machine.occurrences)
            old_engine = machine.engine
            machine.reset()
            assert machine.engine is not old_engine
            assert machine.host == []
            assert machine.occurrences == ledger
            assert machine.engine.current_conversation_id == "identity-conversation"
            machine.exactly_once()
            for _ in range(2):
                machine.new_turn("alpha", False, 0, False, "lookup")
                machine.exactly_once()
            machine.compress()
            machine.exactly_once()
            machine.replay()
            machine.exactly_once()
            machine.cold_resume()
            machine.exactly_once()
    finally:
        machine.teardown()


class CliResetReplayStored(AssertionError):
    """Only the exact #1306 stale-owner or skipped-delivery row shapes."""


def assert_cli_reset_occurrences(ledgers, stored, old, new, start):
    errors = []
    for conversation, ledger in ledgers.items():
        rows = stored[conversation]
        try:
            assert_occurrences(ledger, rows)
            assert all(row["session_id"] == conversation for row in rows), "session ownership"
        except AssertionError as error:
            errors.append(f"{conversation}: {error}")
    if not errors:
        return
    # #1306 leaves the ended conversation intact. Alpha/beta can be ingested
    # there by the failed compress; gamma is skipped. The hooks can also skip
    # every post-/new delivery. No partial loss, duplicate or reorder qualifies.
    actual_old = list(map(identity, stored[old]))
    expected_old = list(map(identity, ledgers[old]))
    before_compress = list(map(identity, ledgers[new][:-2]))
    if (not start and not stored[new]
            and all(row["session_id"] == old for row in stored[old])
            and actual_old in (expected_old, expected_old + before_compress)):
        raise CliResetReplayStored("\n".join(errors))
    raise AssertionError("\n".join(errors))


@pytest.mark.parametrize("post_new_count", [0, 4])
def test_cli_reset_classifier_accepts_only_named_shapes(post_new_count):
    ledgers = {"old": [{"role": "user", "content": "before"}],
               "new": [{"role": "user", "content": str(i)} for i in range(6)]}
    stored = {"old": [dict(row, session_id="old") for row in
                      ledgers["old"] + ledgers["new"][:post_new_count]], "new": []}
    with pytest.raises(CliResetReplayStored):
        assert_cli_reset_occurrences(ledgers, stored, "old", "new", False)


@pytest.mark.parametrize("corruption", ["old-loss", "old-duplicate", "old-order", "partial-new-loss",
                                        "new-row", "wrong-owner", "explicit-start"])
def test_cli_reset_classifier_rejects_other_mismatches(corruption, monkeypatch):
    ledgers = {"old": [{"role": "user", "content": "before"}],
               "new": [{"role": "user", "content": str(i)} for i in range(6)]}
    stored = {"old": [dict(row, session_id="old") for row in
                      ledgers["old"] + ledgers["new"][:4]], "new": []}
    if corruption == "old-loss":
        stored["old"].pop(0)
    elif corruption == "old-duplicate":
        stored["old"].insert(0, stored["old"][0])
    elif corruption == "old-order":
        stored["old"].reverse()
    elif corruption == "partial-new-loss":
        stored["old"].pop()
    elif corruption == "new-row":
        stored["new"].append(dict(ledgers["new"][-1], session_id="new"))
    elif corruption == "wrong-owner":
        stored["old"][-1]["session_id"] = "new"
    checked = []
    oracle = assert_occurrences

    def check(ledger, rows):
        checked.append(ledger)
        oracle(ledger, rows)

    monkeypatch.setattr(sys.modules[__name__], "assert_occurrences", check)
    with pytest.raises(AssertionError) as caught:
        assert_cli_reset_occurrences(ledgers, stored, "old", "new", corruption == "explicit-start")
    assert type(caught.value) is AssertionError
    assert checked == list(ledgers.values())


@pytest.mark.parametrize("start", [
    pytest.param(False, id="bare-new"),
    pytest.param(True, id="explicit-start-control"),
])
def test_cli_new_post_llm_compress_replay(start, caplog, monkeypatch):
    """Real CLI: two turns, /new, two turns, unchanged /compress, final turn.

    Deliveries belong to the host's persisted session, even when LCM writes
    them elsewhere. The control supplies only the missing start after /new.
    """
    import importlib.util
    import hermes_lcm

    # conftest exposes submodules without executing the plugin entry point.
    spec = importlib.util.spec_from_file_location("hermes_lcm", hermes_lcm.__file__)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    from hermes_lcm import plugin_identity
    import sqlite3

    def start_cli(machine):
        RotationMachine._publish_host_session(machine, "identity-session")
        engine = LCMEngine(config=LCMConfig(database_path=str(machine.home / "lcm.db"),
                                           fresh_tail_count=2, leaf_chunk_tokens=64),
                           hermes_home=str(machine.home / "home"))
        engine.on_session_start("identity-session", platform="cli", context_length=8192)
        return engine

    monkeypatch.setattr(ExactlyOnceMachine, "start_engine", start_cli)
    machine = ExactlyOnceMachine()
    prototype = None
    try:
        config = machine.engine._config
        old, new = "identity-session", "identity-reset-1"
        hooks, engines = {}, []
        machine.patch.setattr(LCMConfig, "from_env", classmethod(lambda cls: config))
        machine.patch.setenv("HERMES_HOME", str(machine.home / "home"))
        machine.patch.setattr(plugin_identity, "load_hermes_config", lambda: {})
        machine.patch.setattr(plugin, "_warn_if_embedding_provider_unavailable", lambda config: None)
        plugin.register(types.SimpleNamespace(register_context_engine=engines.append,
                                              register_hook=lambda name, hook: hooks.update({name: hook})))
        prototype = engines[0]
        ledgers = {old: [], new: []}

        def turn(session, text):
            batch = [machine.stamp({"role": "user", "content": text}),
                     machine.stamp({"role": "assistant", "content": f"answer {text}"})]
            machine.host.extend(batch)
            ledgers[session].extend(deepcopy(batch))
            # CLI passes the session ID, no gateway key or context_compressor.
            hooks["post_llm_call"](session_id=session, conversation_history=machine.host, platform="cli")

        for text in ("before-one", "before-two"):
            turn(old, text)
        machine.engine.on_session_end(old, machine.host)
        with sqlite3.connect(machine.home / "home" / "state.db") as connection:
            connection.execute("UPDATE sessions SET end_reason = 'new_session' WHERE id = ?", (old,))
            connection.execute("INSERT INTO sessions VALUES (?, NULL, NULL)", (new,))
        machine.host = []
        machine.engine.on_session_reset()
        # Hermes reset_session_state also offers the new id through this
        # optional callback, even when its bare reset omits on_session_start.
        bind = getattr(machine.engine, "bind_session_state", None)
        if not start and callable(bind):
            bind(None, new)
        if start:
            machine.engine.on_session_start(new, platform="cli", context_length=8192)
        caplog.clear()
        for text in ("alpha", "beta"):
            turn(new, text)
        assert "resident_engine_conflict" not in caplog.text
        previous = deepcopy(machine.host)
        compressed = machine.engine.compress(machine.host, current_tokens=machine.engine.threshold_tokens + 1,
                                             force=True)
        assert machine.engine._last_compression_status == "compacted"
        assert compressed != previous
        machine.engine.on_session_end(new, previous)
        machine.engine.on_session_start(new, platform="cli", boundary_reason="compression", old_session_id=new)
        machine.host = compressed
        caplog.clear()
        turn(new, "gamma")
        assert "resident_engine_conflict" not in caplog.text
        stored = {}
        for conversation in ledgers:
            cursor = machine.engine._store.connection.execute(
                "SELECT role, content, tool_call_id, tool_calls, session_id FROM messages "
                "WHERE conversation_id = ? ORDER BY store_id", (conversation,))
            stored[conversation] = [dict(zip((column[0] for column in cursor.description), row))
                                    for row in cursor.fetchall()]
        # A row under any other conversation is a new failure, never the #1306 shape the xfail names.
        outside = machine.engine._store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id IS NULL OR conversation_id NOT IN "
            f"({', '.join('?' * len(ledgers))})", tuple(ledgers)).fetchone()[0]
        assert outside == 0, f"{outside} rows stored outside the host's conversations"
        assert_cli_reset_occurrences(ledgers, stored, old, new, start)
    finally:
        if prototype is not None:
            prototype.shutdown()
        machine.teardown()


@pytest.mark.parametrize("point", ["in-place", "rotation-ended", "user-persisted"])
def test_restart_replay_resume_append(point):
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    try:
        for text in ("alpha", "same text"):
            machine.new_turn(text, False, 0, False, "lookup")
        engine = machine.engine
        if point == "user-persisted":
            # A structured next prompt is not merged with the pending plain
            # user. Both are delivered at the next completed turn, with no
            # cold-start ingest. The plain/plain #849 shape is seeded below.
            user = machine.stamp({"role": "user", "content": "interrupted user"})
            machine.host.append(user)
            machine.occurrences.append(dict(deepcopy(user), session_id=machine.session_id))
            machine._persist_host_view(machine.session_id, machine.host)
            machine._restart_from_host()
            assert all("message_uid" not in row for row in machine.host)
            machine.new_turn("βeta", True, 0, False, "lookup")
        else:
            machine.restart(point, "interrupted user")
        assert machine.engine is not engine
        assert machine.engine._session_id == machine.session_id
        assert len(machine.session_ids) == 1
        machine.exactly_once()
        for operation in (machine.replay, machine.cold_resume):
            operation()
            machine.exactly_once()
        machine.new_turn("βeta", True, 1, False, "inspect")
        machine.exactly_once()
        machine.rotate()
        machine.exactly_once()
    finally:
        machine.teardown()


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
@pytest.mark.parametrize(("compressions", "text"), [
    (0, "alpha"), (1, "alpha"), (2, "alpha 2"),
], ids=["no-compress", "one-compress", "two-compress-distinct"])
def test_restart_repeated_prompt_controls(cold, compressions, text):
    """Separate restart, repeated text and no-progress compression factors."""
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.exactly_once()
        if cold:
            machine.cold_resume()
        else:
            machine.new_turn("next prompt", False, 0, False, "lookup")
        machine.exactly_once()
        for _ in range(compressions):
            machine.compress()
            machine.exactly_once()
        machine.new_turn(text, False, 0, False, "lookup")
        assert len(machine.occurrences) == 6
        assert machine.occurrences[0]["timestamp"] != machine.occurrences[-2]["timestamp"]
        machine.exactly_once()
    finally:
        machine.teardown()


class RecompressionUserCollapsed(AssertionError):
    """Only #1328's second repeated user missing, with every other row ordered."""


def test_restart_repeated_user_1328():
    """Engine API seed; the CLI refuses manual compression under four rows."""
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    try:
        for text in ("alpha", "next prompt"):
            machine.new_turn(text, False, 0, False, "lookup")
            machine.exactly_once()
        machine.compress()
        machine.exactly_once()
        assert machine.engine._last_compression_status == "compacted"
        previous = deepcopy(machine.host)
        machine.compress()
        machine.exactly_once()
        assert machine.engine._last_compression_status == "sanitized"
        assert machine.host == previous
        machine.new_turn("alpha", False, 0, False, "lookup")
        expected = list(map(identity, machine.occurrences))
        assert len(expected) == 6
        assert expected[0] == expected[-2]
        assert machine.occurrences[0]["timestamp"] != machine.occurrences[-2]["timestamp"]
        actual = list(map(identity, machine.rows()))
        try:
            machine.exactly_once()
        except AssertionError as error:
            if actual == expected[:-2] + expected[-1:]:
                raise RecompressionUserCollapsed(str(error)) from error
            raise
    finally:
        machine.teardown()


class ReplayUserCollapsed(AssertionError):
    """Only #1328's final repeated user missing after an unchanged replay."""


def test_restart_rotation_repeated_user_1328():
    """A rejected no-progress rotation loses the proof before a fresh repeat."""
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.cold_resume()
        machine.exactly_once()
        machine.compress()
        assert machine.engine._compress_commit_proof
        previous = deepcopy(machine.host)
        machine.rotate()
        assert machine.host == previous
        assert machine.engine._compress_commit_proof is None
        machine.exactly_once()
        machine.new_turn("alpha", False, 0, False, "lookup")
        assert len(machine.occurrences) == 6
        assert machine.occurrences[0]["timestamp"] != machine.occurrences[-2]["timestamp"]
        machine.exactly_once()
    finally:
        machine.teardown()


def test_restart_replay_user_1328():
    """Engine API seed; the CLI refuses manual compression under four rows."""
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.exactly_once()
        machine.cold_resume()
        machine.exactly_once()
        machine.compress()
        machine.exactly_once()
        assert machine.engine._last_compression_status == "compacted"
        proof = machine.engine._compress_commit_proof
        assert proof and proof.get("emissions")
        previous = deepcopy(machine.host)
        machine.replay()
        machine.exactly_once()
        assert machine.host == previous
        assert machine.engine._compress_commit_proof is None
        machine.new_turn("alpha", False, 0, False, "lookup")
        expected = list(map(identity, machine.occurrences))
        assert len(expected) == 6
        assert expected[0] == expected[-2]
        assert machine.occurrences[0]["timestamp"] != machine.occurrences[-2]["timestamp"]
        actual = list(map(identity, machine.rows()))
        try:
            machine.exactly_once()
        except AssertionError as error:
            if actual == expected[:-2] + expected[-1:]:
                raise ReplayUserCollapsed(str(error)) from error
            raise
    finally:
        machine.teardown()


class ReplayToolTailUserCollapsed(AssertionError):
    """Only the final repeated user missing from the nine-row tool-tail trace."""


def test_restart_replay_tool_tail_user_1328():
    """T(alpha), T(next prompt with two tools), compress, replay, T(alpha)."""
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.new_turn("next prompt", False, 2, False, "lookup")
        machine.exactly_once()
        machine.compress()
        machine.exactly_once()
        assert machine.engine._last_compression_status == "compacted"
        proof = machine.engine._compress_commit_proof
        assert proof and proof.get("emissions")
        previous = deepcopy(machine.host)
        machine.replay()
        machine.exactly_once()
        assert machine.host == previous
        assert machine.engine._compress_commit_proof is None
        machine.new_turn("alpha", False, 0, False, "lookup")
        expected = list(map(identity, machine.occurrences))
        assert len(expected) == 9
        assert expected[0] == expected[-2]
        assert machine.occurrences[0]["timestamp"] != machine.occurrences[-2]["timestamp"]
        actual = list(map(identity, machine.rows()))
        try:
            machine.exactly_once()
        except AssertionError as error:
            if actual == expected[:-2] + expected[-1:]:
                raise ReplayToolTailUserCollapsed(str(error)) from error
            raise
    finally:
        machine.teardown()


@settings(PROFILE)
@given(case=st.sampled_from(("known-open", "distinct-text", "no-compress", "one-compress", "four-row", "tool-tail")),
       cold=st.booleans())
def test_restart_recompression_generated_controls(case, cold):
    """The former #1328 triggers and their neighbors all deliver the fresh turn."""
    from tests.invariants.test_exactly_once_machine import RestartMachine, repeated_user_after_proof_loss

    machine = RestartMachine()
    known_open = case == "known-open"
    machine.include_known_open = True
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        if cold:
            machine.cold_resume()
        else:
            machine.new_turn("next prompt", False, 2 if case == "tool-tail" else 0, False, "lookup")
        if case == "four-row":
            machine.engine._config.fresh_tail_count = 4
            machine.engine.protect_last_n = 4
            machine.new_turn("third prompt", False, 0, False, "lookup")
        elif case == "tool-tail" and cold:
            machine.new_turn("tool prompt", False, 2, False, "lookup")
        compressions = {"no-compress": 0, "one-compress": 1}.get(case, 2)
        for _ in range(compressions):
            if case == "four-row":
                assert len(machine.host) >= 4
            machine.compress()
            machine.exactly_once()
        delivered = len(machine.occurrences)
        text = "alpha 2" if case == "distinct-text" else "alpha"
        user = dict(role="user", content=text, timestamp=float(machine.sequence + 1))
        assert bool(repeated_user_after_proof_loss(machine.proof_lost_view == machine.host,
                    machine.engine._compress_commit_proof, machine.host, machine.occurrences, user)) == known_open
        machine.new_turn(text, False, 0, False, "lookup")
        assert len(machine.occurrences) == delivered + 2
        machine.exactly_once()
    finally:
        machine.teardown()


@settings(PROFILE)
@given(case=st.sampled_from(("known-open", "no-compress", "new-turn", "distinct-text", "tool-tail", "four-row")),
       cold=st.booleans())
def test_restart_replay_generated_controls(case, cold):
    """Replay neighbors and equal-identity triggers all deliver the fresh turn."""
    from tests.invariants.test_exactly_once_machine import RestartMachine, repeated_user_after_proof_loss

    machine = RestartMachine()
    known_open = case in ("known-open", "tool-tail")
    machine.include_known_open = True
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        if cold:
            machine.cold_resume()
        else:
            machine.new_turn("next prompt", False, 2 if case == "tool-tail" else 0, False, "lookup")
        if case == "four-row":
            machine.engine._config.fresh_tail_count = 4
            machine.engine.protect_last_n = 4
            machine.new_turn("third prompt", False, 0, False, "lookup")
        elif case == "tool-tail" and cold:
            machine.new_turn("tool prompt", False, 2, False, "lookup")
        machine.exactly_once()
        if case != "no-compress":
            machine.compress()
            machine.exactly_once()
            assert machine.engine._last_compression_status == "compacted"
        if case == "new-turn":
            machine.new_turn("intervening prompt", False, 0, False, "lookup")
            machine.exactly_once()
        else:
            previous = deepcopy(machine.host)
            machine.replay()
            assert machine.host == previous
            machine.exactly_once()
        delivered = len(machine.occurrences)
        text = "alpha 2" if case == "distinct-text" else "alpha"
        user = dict(role="user", content=text, timestamp=float(machine.sequence + 1))
        assert bool(repeated_user_after_proof_loss(machine.proof_lost_view == machine.host,
                    machine.engine._compress_commit_proof, machine.host, machine.occurrences, user)) == known_open
        machine.new_turn(text, False, 0, False, "lookup")
        assert len(machine.occurrences) == delivered + 2
        machine.exactly_once()
    finally:
        machine.teardown()


class RestartScaffoldStored(AssertionError):
    """Only the unchanged ledger plus one restored carrier stored as a source."""


def check_restart_scaffold(machine):
    expected, actual = list(map(identity, machine.occurrences)), list(map(identity, machine.rows()))
    try:
        machine.exactly_once()
    except AssertionError as error:
        if actual == expected[:-2] + [identity(machine.host[0])] + expected[-2:]:
            raise RestartScaffoldStored(str(error)) from error
        raise


@pytest.mark.xfail(strict=True, reason="#1286: cold published child lacks its parent emission proof",
                   raises=RestartScaffoldStored)
def test_restart_published_child_without_start_proof():
    """Non-default compression.in_place=False; check the next completed turn."""
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    try:
        for _ in range(2):
            machine.new_turn("alpha", False, 0, False, "lookup")
        machine.exactly_once()
        machine.restart("rotation-published", "alpha")
        check_restart_scaffold(machine)
    finally:
        machine.teardown()


def test_restart_published_child_after_start_control(monkeypatch):
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    cold = machine._restart_from_host

    def after_start():
        child = machine.session_ids[-1]
        machine.engine.on_session_start(child, platform="cli", conversation_id="identity-conversation",
                                         boundary_reason="compression", old_session_id=machine.session_id)
        cold()

    monkeypatch.setattr(machine, "_restart_from_host", after_start)
    try:
        for _ in range(2):
            machine.new_turn("alpha", False, 0, False, "lookup")
        machine.restart("rotation-published", "alpha")
        machine.exactly_once()
    finally:
        machine.teardown()


@pytest.mark.parametrize("point", ["in-place", "rotation-ended"])
@pytest.mark.xfail(strict=True, reason="#1286: unadopted compaction supersedes the durable input emission",
                   raises=RestartScaffoldStored)
def test_restart_before_recompression_adoption(point):
    """Default in-place window, plus rotation-ended with in_place=False."""
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.new_turn("alpha", False, 1, False, "lookup")
        machine.rotate()
        machine.exactly_once()
        machine.restart(point, "alpha")
        check_restart_scaffold(machine)
    finally:
        machine.teardown()


@pytest.mark.parametrize("point", ["in-place", "rotation-ended"])
def test_restart_before_recompression_adoption_control(point):
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.rotate()
        machine.exactly_once()
        machine.restart(point, "alpha")
        machine.exactly_once()
    finally:
        machine.teardown()


class RestartMergedStored(AssertionError):
    """Only the new prompt replaced by the host's held-plus-new composite."""


@pytest.mark.xfail(strict=True, reason="#849: unanswered persisted user merges with the next completed turn",
                   raises=RestartMergedStored)
def test_restart_pending_prompt_849():
    """Minimal no-duplicate variant: T(alpha), pending X, restart, completed Y."""
    from tests.invariants.test_exactly_once_machine import RestartMachine

    machine = RestartMachine()
    machine.include_known_open = True
    try:
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.restart("user-persisted", "X pending")
        expected, actual = list(map(identity, machine.occurrences)), list(map(identity, machine.rows()))
        try:
            machine.exactly_once()
        except AssertionError as error:
            if actual == expected[:-3] + [identity(machine.host[-2]), expected[-1]]:
                raise RestartMergedStored(str(error)) from error
            raise
    finally:
        machine.teardown()


_LEAK_PROBE = """
import sys
from hypothesis import Phase, given, settings, strategies as st

@settings(phases=[Phase[name] for name in sys.argv[1:]], database=None, derandomize=True, deadline=None, max_examples=5)
@given(st.integers())
def fails(value):
    raise AssertionError(value)

try:
    fails()
except AssertionError:
    pass
print([sys.monitoring.get_events(tool) for tool in range(6)])
"""


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="sys.monitoring exists on Python 3.12+")
def test_failing_example_leaves_no_monitoring_events():
    """A failing example under PROFILE's phases leaves no sys.monitoring events set. Hypothesis's explain
    phase did not: its LINE events stayed set and slowed every later test. A fresh interpreter gives a clean
    baseline, so a leak from an earlier test in this session cannot hide one here."""
    probe = subprocess.run([sys.executable, "-c", _LEAK_PROBE, *(phase.name for phase in PROFILE.phases)],
                           capture_output=True, text=True, timeout=120, check=True)
    assert probe.stdout.strip() == str([0] * 6), probe.stdout + probe.stderr
