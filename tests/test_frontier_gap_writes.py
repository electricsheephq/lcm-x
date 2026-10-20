"""Cover the four C4 frontier writes, including the known replay bug (#1317).

The failed reconciliation probe uses a strict xfail for duplicate replay rows;
its plain preconditions must still fail normally when the setup changes.

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


@pytest.mark.xfail(
    strict=True,
    raises=ReplayStoredTwice,
    reason="#1317: a failed reconcile probe trusts cursor 0 and re-stores the replay",
)
def test_reconcile_probe_failure_trusts_zero_cursor(
    engine: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5278: inject only the store probe failure, then let ingest use the store.

    A normal rebind is host-shaped; the store failure is a defensive fault.
    Only duplicate replay rows are an expected failure under #1317. A corrected
    write must reconcile the replay and produce a strict XPASS; setup failures
    remain ordinary assertion failures.
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
