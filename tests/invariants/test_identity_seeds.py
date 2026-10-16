"""Occurrence seeds for compress, recovery, rotation, retries and stale proofs."""
from copy import deepcopy
import re
import sys
import types

import pytest

pytest.importorskip("hypothesis")
from hypothesis import settings, strategies as st
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

        machine.patch.setattr(machine.engine, "compress", unchanged)
        machine.patch.setattr(machine.engine, "on_session_end", unexpected_hook)
        machine.patch.setattr(machine.engine, "on_session_start", unexpected_hook)
        machine.rotate()
        assert machine.host == previous
        assert machine.session_ids == ["identity-session"]
        assert machine.stale_proof == marker_only
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
    pytest.param("βeta", id="equal-source-warm", marks=pytest.mark.xfail(
        strict=True, reason=REPROJECTED_USER, raises=ReprojectedUserStored)),
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
    reasons.update(str(mark.kwargs.get("reason", ""))
                   for value in list(globals().values()) if callable(value)
                   for mark in getattr(value, "pytestmark", []) if mark.name == "parametrize"
                   for parameter in mark.args[1] if hasattr(parameter, "marks")
                   for mark in parameter.marks if mark.name == "xfail" and mark.kwargs.get("strict"))
    xfailed = {int(found.group(1)) for reason in reasons if (found := re.match(r"#(\d+)", reason))}
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
    pytest.param(False, id="bare-new", marks=pytest.mark.xfail(
        strict=True, reason="#1306: CLI /new keeps LCM bound to the ended session",
        raises=CliResetReplayStored)),
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
        if start:
            machine.engine.on_session_start(new, platform="cli", context_length=8192)
        caplog.clear()
        for text in ("alpha", "beta"):
            turn(new, text)
        if not start:
            assert "resident_engine_conflict" in caplog.text
            assert machine.engine._store.connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4
        previous = deepcopy(machine.host)
        compressed = machine.engine.compress(machine.host, current_tokens=machine.engine.threshold_tokens + 1,
                                             force=True)
        if start:
            assert machine.engine._last_compression_status == "compacted"
            assert compressed != previous
            machine.engine.on_session_end(new, previous)
            machine.engine.on_session_start(new, platform="cli", boundary_reason="compression", old_session_id=new)
            machine.host = compressed
        else:
            assert machine.engine._last_compression_status == "error"
            assert machine.engine._survival_fit_reason == "lifecycle_binding_changed"
            assert compressed == previous
            assert machine.host == previous
            # Real CLI sees unchanged output and skips both boundary hooks.
        caplog.clear()
        turn(new, "gamma")
        if not start:
            assert "resident_engine_conflict" in caplog.text
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
