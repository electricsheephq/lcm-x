"""Regression tests for issue #255 — the fork/side-channel ingest guard
silently drops legitimate post-compaction batches.

Shape under test (issue #255, the case the guard's own tests never covered):
a session is compacted, the process dies before the turns that followed the
compaction were ingested, and the restarted host rebinds with its active
context — its system prompt, LCM's compaction summary, and the turns that
followed.  Those turns have no identity overlap with the durable tail *by
construction*: they are new.  That satisfies all three clauses of the
fork guard (shorter than the durable session, more than five identities,
zero last-five tail overlap), so the batch is skipped and reported as
consumed (``cursor = len(messages)``).  The turns are lost permanently and
nothing downstream can observe the loss.

Interception boundary, verified here as well: when the LCM system note
survives verbatim on the leading system message the scaffold-only-prefix
proof fires first and no loss occurs.  The loss needs only that the host
re-renders its own system prompt on restart — the note is LCM's, not the
host's.

All tests use synthetic messages.  No real session data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

# engine.py imports agent.context_engine at module level; provide a stub
# before the conftest partial-import can poison sys.modules.
if "agent.context_engine" not in sys.modules:
    _agent_mod = ModuleType("agent")
    _agent_mod.__path__ = []
    _ce_mod = ModuleType("agent.context_engine")

    class _StubContextEngine:
        def __init__(self, **kwargs):
            self.compression_count = 0
            self.last_prompt_tokens = 0

        def get_status(self):
            return {}

    _ce_mod.ContextEngine = _StubContextEngine
    sys.modules["agent"] = _agent_mod
    sys.modules["agent.context_engine"] = _ce_mod

# Clear any broken partial import left by conftest
_existing = sys.modules.get("hermes_lcm.engine")
if _existing is not None and not hasattr(_existing, "LCMEngine"):
    sys.modules.pop("hermes_lcm.engine", None)

import hermes_lcm.engine as lcm_engine  # noqa: E402
from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.engine import LCMEngine  # noqa: E402

SESSION_ID = "issue-255-post-compaction"
HOST_SYSTEM_PROMPT = "System policy."


def _deterministic_summary(*_args, **_kwargs):
    return (
        "Earlier turns established four durable facts and one pending blocker.\n"
        "Expand for details about: durable facts and pending blocker",
        1,
    )


def _open_engine(database_path: str) -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=database_path,
            fresh_tail_count=4,
            leaf_chunk_tokens=1,
        )
    )
    engine.on_session_start(SESSION_ID, context_length=200000)
    return engine


def _pre_compaction_history(turns: int = 8) -> list[dict]:
    messages = [{"role": "system", "content": HOST_SYSTEM_PROMPT}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"Question {index}: " + "x" * 200})
        messages.append({"role": "assistant", "content": f"Answer {index}: " + "y" * 200})
    return messages


def _post_compaction_turns(turns: int) -> list[dict]:
    """Turns that happened after compaction and were never ingested."""
    messages = []
    for index in range(turns):
        messages.append({"role": "user", "content": f"Post-compaction question {index}"})
        messages.append({"role": "assistant", "content": f"Post-compaction answer {index}"})
    return messages


def _seed_compacted_session(tmp_path: Path, monkeypatch) -> tuple[str, dict, int]:
    """Run one real compaction; return (db path, LCM summary message, row count)."""
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _deterministic_summary)
    database_path = str(tmp_path / "issue-255.db")

    engine = _open_engine(database_path)
    try:
        active_context = engine.compress(_pre_compaction_history(), current_tokens=100000)
        durable_rows = engine._store.get_session_count(SESSION_ID)
        summary_message = next(
            message
            for message in active_context
            if message.get("role") != "system"
            and engine._is_replayed_context_scaffold_message(message)
        )
    finally:
        engine.shutdown()

    assert durable_rows == len(_pre_compaction_history())
    return database_path, dict(summary_message), durable_rows


class TestPostCompactionBatchIsNotSilentlyDropped:
    """The batch must persist, or at minimum must not be reported consumed
    while its rows are absent."""

    def test_post_compaction_continuation_persists_after_crash_restart(
        self, tmp_path, monkeypatch
    ):
        """Six turns that followed a compaction must survive a crash restart.

        The restarted host supplies its own system prompt, LCM's compaction
        summary, and the six turns.  Nothing here is a fork: it is the same
        conversation continuing after the process that owned it died before
        ingesting them.
        """
        database_path, summary_message, durable_rows = _seed_compacted_session(
            tmp_path, monkeypatch
        )
        new_turns = _post_compaction_turns(6)
        snapshot = [
            {"role": "system", "content": HOST_SYSTEM_PROMPT},
            summary_message,
            *new_turns,
        ]

        restarted = _open_engine(database_path)
        try:
            restarted.compress(snapshot, current_tokens=0)
            reconciliation = dict(restarted._last_ingest_reconciliation)
            stored = [
                str(row["content"])
                for row in restarted._store.get_session_messages(SESSION_ID)
            ]
            row_count = restarted._store.get_session_count(SESSION_ID)
        finally:
            restarted.shutdown()

        missing = [
            str(message["content"])
            for message in new_turns
            if str(message["content"]) not in stored
        ]
        assert not missing, (
            f"{len(missing)} of {len(new_turns)} post-compaction messages were dropped "
            f"while reconciliation reported action={reconciliation.get('action')!r} "
            f"cursor={reconciliation.get('cursor')} incoming={reconciliation.get('incoming')}: "
            f"{missing[:3]}"
        )
        # Each new turn lands exactly once, and the durable history it
        # continues is untouched.  The batch may also re-persist replayed
        # scaffolding — that is the pre-existing ambiguous-delta cost the
        # fallback has always paid (see fork-guard t4/t5), not message loss.
        for message in new_turns:
            assert stored.count(str(message["content"])) == 1, (
                f"Post-compaction message duplicated: {message['content']!r}"
            )
        assert row_count >= durable_rows + len(new_turns)
        for original in _pre_compaction_history():
            if original["role"] == "system":
                continue
            assert stored.count(str(original["content"])) == 1, (
                f"Pre-compaction message duplicated: {str(original['content'])[:40]!r}"
            )

    def test_post_compaction_batch_never_reported_consumed_while_absent(
        self, tmp_path, monkeypatch
    ):
        """The silent-drop contract violation, stated directly.

        Reporting ``cursor == len(messages)`` means "this batch is fully
        accounted for".  A reconciliation may only claim that when the rows
        are in the store.  Claiming it while the rows are absent makes the
        loss unobservable to every caller.
        """
        database_path, summary_message, _ = _seed_compacted_session(tmp_path, monkeypatch)
        new_turns = _post_compaction_turns(6)
        snapshot = [
            {"role": "system", "content": HOST_SYSTEM_PROMPT},
            summary_message,
            *new_turns,
        ]

        restarted = _open_engine(database_path)
        try:
            restarted._ingest_messages(snapshot)
            reconciliation = dict(restarted._last_ingest_reconciliation)
            stored = [
                str(row["content"])
                for row in restarted._store.get_session_messages(SESSION_ID)
            ]
        finally:
            restarted.shutdown()

        reported_consumed = reconciliation.get("cursor") == len(snapshot)
        rows_present = all(
            str(message["content"]) in stored for message in new_turns
        )
        assert not (reported_consumed and not rows_present), (
            "Batch reported fully consumed while its rows are absent: "
            f"{reconciliation}"
        )

    def test_issue_255_arithmetic_summary_plus_six_identities(
        self, tmp_path, monkeypatch
    ):
        """Issue #255's literal arithmetic: one summary + six new messages.

        Seven identities — one over the guard's ``> 5`` floor, shorter than
        the durable session, zero last-five overlap.  The smallest batch the
        guard can eat.
        """
        database_path, summary_message, _ = _seed_compacted_session(tmp_path, monkeypatch)
        new_turns = _post_compaction_turns(3)
        snapshot = [
            {"role": "system", "content": HOST_SYSTEM_PROMPT},
            summary_message,
            *new_turns,
        ]

        restarted = _open_engine(database_path)
        try:
            restarted._ingest_messages(snapshot)
            reconciliation = dict(restarted._last_ingest_reconciliation)
            stored = [
                str(row["content"])
                for row in restarted._store.get_session_messages(SESSION_ID)
            ]
        finally:
            restarted.shutdown()

        for message in new_turns:
            assert stored.count(str(message["content"])) == 1, (
                f"Post-compaction message {message['content']!r} not persisted exactly once "
                f"(reconciliation: {reconciliation})."
            )


class TestPostCompactionInterceptionBoundary:
    """Where the durable-snapshot replay proofs do catch the batch first.

    These pin the boundary #255 asked to establish.  They pass with or
    without the guard; they exist so a future change that moves the boundary
    is visible.
    """

    def test_surviving_lcm_system_note_is_intercepted_by_scaffold_proof(
        self, tmp_path, monkeypatch
    ):
        """LCM's own note on the leading system message → scaffold-only
        prefix proof consumes the scaffold and persists the new turns."""
        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _deterministic_summary)
        database_path = str(tmp_path / "issue-255-boundary.db")

        engine = _open_engine(database_path)
        try:
            active_context = engine.compress(_pre_compaction_history(), current_tokens=100000)
            durable_rows = engine._store.get_session_count(SESSION_ID)
        finally:
            engine.shutdown()

        assert engine._is_replayed_context_scaffold_message(active_context[0])
        new_turns = _post_compaction_turns(6)
        snapshot = [*active_context[:2], *new_turns]

        restarted = _open_engine(database_path)
        try:
            restarted._ingest_messages(snapshot)
            reconciliation = dict(restarted._last_ingest_reconciliation)
            row_count = restarted._store.get_session_count(SESSION_ID)
        finally:
            restarted.shutdown()

        assert reconciliation.get("action") == "advanced cursor"
        assert row_count == durable_rows + len(new_turns)

    def test_full_active_snapshot_with_fresh_tail_is_intercepted(
        self, tmp_path, monkeypatch
    ):
        """The whole post-compaction active context (scaffold + fresh tail +
        new turns) replays through the durable-tail proof."""
        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _deterministic_summary)
        database_path = str(tmp_path / "issue-255-fulltail.db")

        engine = _open_engine(database_path)
        try:
            active_context = engine.compress(_pre_compaction_history(), current_tokens=100000)
            durable_rows = engine._store.get_session_count(SESSION_ID)
        finally:
            engine.shutdown()

        new_turns = _post_compaction_turns(6)
        snapshot = [*active_context, *new_turns]

        restarted = _open_engine(database_path)
        try:
            restarted._ingest_messages(snapshot)
            reconciliation = dict(restarted._last_ingest_reconciliation)
            row_count = restarted._store.get_session_count(SESSION_ID)
        finally:
            restarted.shutdown()

        assert reconciliation.get("reason") == "replayed durable tail"
        assert row_count == durable_rows + len(new_turns)
