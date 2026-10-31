"""Cover the four C4 frontier writes, including the known replay bug (#1317).

Failed reconciliation probes must defer cursor trust until the next ingest.

The inspected host starts engines in agent/agent_init.py:2106 and notifies
compression children in agent/conversation_compression.py:2555, 3445. That
rotation does not end the engine session; run_agent.py:217 ends real sessions.
Consequently an end-consumed proof with a changed durable reset epoch needs
an engine-level setup, rather than a claim that normal host rotation does it.
"""
from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# conftest registers the plugin's virtual package; it is not an installed package.
engine_module = importlib.import_module("hermes_lcm.engine")
config_module = importlib.import_module("hermes_lcm.config")


class ReplayStoredTwice(Exception):
    """The failed reconciliation probe caused durable replay rows to repeat."""


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Any]:
    instance = engine_module.LCMEngine(
        config=config_module.LCMConfig(
            database_path=str(tmp_path / "frontier.db"),
            large_output_externalization_path=str(tmp_path / "externalized"),
            fresh_tail_count=3,
            leaf_chunk_tokens=400,
            temporal_rollups_enabled=False,
            stateless_session_patterns=["bypassed-source"],
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        yield instance
    finally:
        instance.shutdown()


def _compact(engine: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def summarize(*args: Any, **kwargs: Any) -> tuple[str, int]:
        return "Earlier turns summarized. Expand for the original turns.", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    messages: list[dict[str, Any]] = []
    for index in range(8):
        messages.extend([
            {"role": "user", "content": f"Turn {index}: " + "alpha beta gamma delta " * 40},
            {"role": "assistant", "content": f"Recorded turn {index}."},
        ])
    messages.append({"role": "user", "content": "Current question."})
    engine.ingest(messages)
    output: list[dict[str, Any]] = engine.compress(list(messages), force=True)
    assert engine._last_compression_status == "compacted"
    assert engine._compress_commit_proof is not None
    assert engine._last_emission_descriptors["emissions"]
    return messages, output


def test_rotation_drops_descriptors_from_a_previous_reset_epoch(
    engine: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3860: a durable reset may invalidate an otherwise transferable proof.

    No inspected host sequence interleaves a reset with an end-consumed proof.
    Exercise the boundary helper with real writer-produced proofs and a real
    lifecycle reset (the durable operation used by on_session_reset), leaving
    the source engine's in-memory snapshot intact as another engine would.
    The public start's final binding filter also clears this stale cache; the
    helper test isolates the earlier write instead of hiding that redundancy.
    """
    engine.on_session_start("parent", platform="acp", context_length=200_000)
    messages, output = _compact(engine, monkeypatch)
    engine.on_session_end("parent", messages)
    state = engine._lifecycle.record_reset(engine._conversation_id)
    assert state is not None and state.last_reset_at is not None
    assert engine._compress_commit_proof["end_consumed"]
    assert engine._last_emission_descriptors["reset_epoch"] is None

    engine._continue_compression_boundary("child", "parent", {"platform": "acp"})

    # A stale cache must not authorize generated-span removal on the helper's
    # output. The durable reader rejects the old reset epoch as well.
    occurrences, _ = engine._replay_occurrences(output)
    assert occurrences
    assert len(occurrences) == len(output)
    assert occurrences[-1][1] == ("user", "Current question.", "", "", "")
    assert all(entry.generated_span is None for entry, _identity in occurrences)
    assert engine._last_emission_descriptors is None
    assert engine._store.get_session_count("parent") == len(messages)
    assert engine._store.get_session_count("child") == 0


@pytest.mark.parametrize("already_bound", [False, True], ids=["fresh-child", "bound-child"])
def test_bypassed_source_boundary_restarts_the_cursor(
    engine: Any, already_bound: bool,
) -> None:
    """4121-4122: child adoption with no different previous engine session.

    Host recovery calls start(child, old=parent) at
    agent/conversation_compression.py:1703; its normal initialization may
    already have bound that child (agent/agent_init.py:2106). The fresh case
    also exercises a zero cursor without inventing nonzero initial state.
    """
    messages = [{"role": "user", "content": "Existing child turn."}]
    if already_bound:
        engine.on_session_start("child", platform="acp")
        engine.ingest(messages)
        assert engine._ingest_cursor == 1
    else:
        assert engine._session_id == ""
    engine.on_session_start(
        "child", platform="acp", boundary_reason="compression", old_session_id="bypassed-source",
    )
    assert engine._ingest_cursor == 0
    assert engine._last_compacted_store_id == 0
    continuation = {"role": "assistant", "content": "Bypassed continuation."}
    complete_history = [*messages, continuation]
    # The host forwards the end history unchanged (run_agent.py:182-187 at
    # NousResearch/hermes-agent@d923a5eb487a17c0a68801bba9ef0f56c81040a8).
    # This child inherits stateless bypass: neither ingest nor end may store
    # its continuation; any already-bound normal row must remain intact.
    engine.ingest(complete_history)
    engine.on_session_end("child", complete_history)
    rows = engine._store.get_session_messages("child")
    assert [row["content"] for row in rows] == [
        message["content"] for message in messages if already_bound
    ]
    assert sum(row["content"] == continuation["content"] for row in rows) == 0
    assert engine._store.get_session_count("child") == int(already_bound)
    assert engine._store.get_session_count("bypassed-source") == 0
    state = engine._lifecycle.get_by_session("child")
    assert state is not None and state.current_frontier_store_id == 0
    # 4122 is overwritten by _bind_lifecycle_state (engine.py:2442).
    # There is no intervening reader: skipping it has no durable consequence.


def test_reconcile_probe_failure_trusts_zero_cursor(
    engine: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5278: inject only the store probe failure, then let ingest use the store.

    A normal rebind is host-shaped; the store failure is a defensive fault.
    The probe must be reached, and the replay must retain exactly one copy.
    """
    engine.on_session_start("session", platform="acp")
    messages = [
        {"role": "user", "content": "Durable question."},
        {"role": "assistant", "content": "Durable answer."},
    ]
    engine.ingest(messages)

    probed: list[str] = []

    def fail_probe(session_id: str) -> int:
        probed.append(session_id)
        raise RuntimeError("injected reconciliation probe failure")

    with monkeypatch.context() as patch:
        # The probe was get_session_count() > 0 until #1174 made it has_session_messages(); fault whichever exists.
        for name in ("get_session_count", "has_session_messages"):
            if hasattr(engine._store, name):
                patch.setattr(engine._store, name, fail_probe)
        engine.on_session_start("session", platform="acp")
    assert probed, "the reconciliation probe was not reached; the injected fault must hit the probe the engine calls"
    assert engine._store.get_session_count("session") == len(messages)
    engine.ingest([*messages, {"role": "user", "content": "New question."}])
    rows = engine._store.get_session_messages("session")
    contents = [row["content"] for row in rows]
    if contents == ["Durable question.", "Durable answer."] * 2 + ["New question."]:
        raise ReplayStoredTwice(f"Expected reconciled replay, stored {contents!r}")
    # Any other shape (a lost or reordered row) is a new failure, not #1317.
    assert contents == ["Durable question.", "Durable answer.", "New question."]


def _start_with_failed_probe(engine: Any, monkeypatch: pytest.MonkeyPatch, probe: str, session: str) -> None:
    """Fault only the scheduled probe, not unrelated lifecycle reads."""
    schedule = engine._schedule_ingest_cursor_reconciliation
    reached = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        reached.append(probe)
        raise RuntimeError("injected session-start probe failure")

    def schedule_with_fault() -> None:
        target = engine._store if probe == "has_session_messages" else engine
        with monkeypatch.context() as patch:
            patch.setattr(target, probe, fail)
            schedule()

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_schedule_ingest_cursor_reconciliation", schedule_with_fault)
        engine.on_session_start(session, platform="acp")
    assert reached == [probe], "the selected scheduled probe must actually raise"


def test_failed_probe_then_failed_ingest_recovers_without_loss_or_duplicates(
    engine: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine.on_session_start("session", platform="acp")
    messages = [
        {"role": "user", "content": "Durable question."},
        {"role": "assistant", "content": "Durable answer."},
    ]
    engine.ingest(messages)
    _start_with_failed_probe(engine, monkeypatch, "has_session_messages", "session")
    history = [*messages, {"role": "user", "content": "Keep the pending question."}]
    reads = []

    def fail_tail(*args: Any, **kwargs: Any) -> Any:
        reads.append(True)
        raise RuntimeError("store is still unavailable at next ingest")

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_session_tail", fail_tail)
        engine.ingest(history)  # Public ingest records the error rather than raising.
    assert [row["content"] for row in engine._store.get_session_messages("session")] == [
        message["content"] for message in messages
    ]
    assert reads, "the unavailable reconciliation read must be attempted"
    assert engine._ingest_cursor_needs_reconcile
    engine.ingest(history)
    engine.ingest(history)  # A successful retry must also leave the cursor usable.
    assert [row["content"] for row in engine._store.get_session_messages("session")] == [
        message["content"] for message in history
    ]
    assert not engine._ingest_cursor_needs_reconcile
    assert engine._ingest_cursor == len(history)


@pytest.mark.parametrize("append_unavailable", [True, False], ids=["count-and-append-outage", "count-only-outage"])
def test_failed_probe_then_failed_count_preserves_reconciliation(
    engine: Any, monkeypatch: pytest.MonkeyPatch, append_unavailable: bool,
) -> None:
    """An unavailable count is not evidence that replay starts at cursor zero."""
    engine.on_session_start("session", platform="acp")
    messages = [
        {"role": "user", "content": "Durable question."},
        {"role": "assistant", "content": "Durable answer."},
    ]
    engine.ingest(messages)
    _start_with_failed_probe(engine, monkeypatch, "has_session_messages", "session")
    assert engine._ingest_cursor_needs_reconcile
    history = [*messages, {"role": "user", "content": "Keep the pending question."}]
    counts = []
    appends = []
    append_batch = engine._store._append_protected_batch
    failures_before = engine._ingest_failure_count

    def fail_count(*args: Any, **kwargs: Any) -> Any:
        counts.append(True)
        raise RuntimeError("synthetic count-read outage")

    def observe_append(*args: Any, **kwargs: Any) -> Any:
        appends.append(True)
        if append_unavailable:
            raise RuntimeError("synthetic append outage")
        return append_batch(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_session_count", fail_count)
        patch.setattr(engine._store, "_append_protected_batch", observe_append)
        engine.ingest(history)  # Public ingest records an unavailable-store error.
    outage_contents = [row["content"] for row in engine._store.get_session_messages("session")]
    pending_after_outage = engine._ingest_cursor_needs_reconcile
    failures_after_outage = engine._ingest_failure_count
    engine.ingest(history)
    recovered_contents = [row["content"] for row in engine._store.get_session_messages("session")]
    engine.ingest(history)
    repeated_contents = [row["content"] for row in engine._store.get_session_messages("session")]

    assert counts, "the unavailable reconciliation count must actually be attempted"
    expected = [message["content"] for message in history]
    # Check recovery first so the unfixed persistent-outage case exposes five
    # durable rows, while retaining the outage flag and no-write invariants.
    assert recovered_contents == expected
    assert repeated_contents == expected
    assert outage_contents == [message["content"] for message in messages]
    assert pending_after_outage
    assert failures_after_outage == failures_before + 1
    assert not appends, "a failed count must stop ingestion before any append"
    assert not engine._ingest_cursor_needs_reconcile
    assert engine._ingest_cursor == len(history)


@pytest.mark.parametrize("probe", [
    "has_session_messages", "_durable_commit_proof_payload", "_load_native_recovery_replay_snapshot_digests",
])
def test_new_session_failed_probe_stores_once(
    engine: Any, monkeypatch: pytest.MonkeyPatch, probe: str,
) -> None:
    _start_with_failed_probe(engine, monkeypatch, probe, "new-session")
    pending_reconciliation = engine._ingest_cursor_needs_reconcile
    messages = [{"role": "user", "content": "First question."}]
    engine.ingest(messages)
    engine.ingest(messages)
    assert [row["content"] for row in engine._store.get_session_messages("new-session")] == ["First question."]
    # Empty-store row parity alone passes the base: the safety invariant is
    # still to defer trust when any of the three reads cannot prove emptiness.
    assert pending_reconciliation
    assert not engine._ingest_cursor_needs_reconcile


@pytest.mark.parametrize("fail_probe", [False, True], ids=["working-probe", "failed-probe"])
@pytest.mark.parametrize("probe", ["has_session_messages", "_durable_commit_proof_payload"])
def test_empty_rotation_child_probe_matches_committed_outcome(
    engine: Any, monkeypatch: pytest.MonkeyPatch, fail_probe: bool, probe: str,
) -> None:
    engine.on_session_start("parent", platform="acp")
    messages, output = _compact(engine, monkeypatch)
    engine.on_session_end("parent", messages)
    engine.on_session_start("child", boundary_reason="compression", old_session_id="parent", platform="acp")
    assert engine._store.get_session_count("child") == 0
    assert engine._durable_commit_proof_payload() is not None
    if fail_probe:
        _start_with_failed_probe(engine, monkeypatch, probe, "child")
    else:
        engine.on_session_start("child", platform="acp")
    before = engine._store._conn.execute("SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()
    new_turn = {"role": "user", "content": "Question after rotation."}
    engine.ingest([*output, new_turn])
    engine.ingest([*output, new_turn])
    after = engine._store._conn.execute("SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()
    assert after == [*before, ("child", "user", new_turn["content"])]
