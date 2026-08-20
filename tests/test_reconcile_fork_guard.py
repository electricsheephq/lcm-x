"""Tests for the fork/side-channel ingest guard in reconciliation.

A forked agent (background review, cron side-channel) shares the parent's
session_id but carries a shorter, divergent message list.  Without the
guard, its ingest corrupts the stored tail and causes the parent's next
reconcile to fail → cursor=0 → full re-ingest → duplication.

All tests use synthetic messages.  No real session data.
"""
# HISTORY (2026-08-20, architect ruling on eval-queue #252): the fork/side-channel
# guard these tests originally fenced was VETOED and surgically removed -- silent
# consumption with data loss on the ingest path (reported consumed, rows absent;
# proven in tests/test_issue_255_post_compaction_ingest.py) outranks the fork
# duplication it prevented. Tests t1/t2/t9/t10, which asserted the guard's skip
# behavior, were removed with the guard. Fork batches now fall through to the
# ambiguous-delta path and PERSIST (visible duplication, recoverable) rather than
# vanish (invisible, permanent). The successor mechanism is positive fork
# provenance -- see the follow-up issue referenced in the revert PR.


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

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _make_engine(tmp_path: Path, *, session_id: str = "fork-guard") -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine._session_id = session_id
    return engine


def _make_messages(n: int, *, prefix: str = "msg") -> list[dict]:
    """Build a synthetic message list: system + n user/assistant turns."""
    msgs = [{"role": "system", "content": "System prompt."}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"{prefix}-{i}"})
    return msgs


class TestForkSideChannelGuard:
    """T1-T10 from the risk analysis."""



    def test_t3_legitimate_restart_full_replay(self, tmp_path):
        """Normal restart with full replay + new messages → cursor advances."""
        engine = _make_engine(tmp_path)
        try:
            original = _make_messages(19, prefix="orig")
            engine._ingest_messages(original)
            assert engine._store.get_session_count("fork-guard") == 20

            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                replay_msgs = original + [
                    {"role": "user", "content": "new-after-restart"},
                ]
                replay_engine._ingest_messages(replay_msgs)

                count = replay_engine._store.get_session_count("fork-guard")
                assert count == 21, (
                    f"Expected 21 (20 replay + 1 new), got {count}."
                )
                rows = replay_engine._store.get_session_messages("fork-guard")
                assert rows[-1]["content"] == "new-after-restart"
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_t4_legitimate_restart_short_delta(self, tmp_path):
        """Short delta with partial tail match → existing logic persists all.

        The reconcile requires full-replay evidence (candidate covers the
        full durable session) to advance the cursor.  A 5-message delta
        against a 50-message session cannot provide that evidence, so the
        existing fallback persists the entire delta.  This is pre-existing
        behavior, not affected by the fork guard.  The guard is skipped
        here because the delta's tail overlaps with the stored tail.
        """
        engine = _make_engine(tmp_path)
        try:
            original = _make_messages(49, prefix="orig")
            engine._ingest_messages(original)
            assert engine._store.get_session_count("fork-guard") == 50

            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                # Short delta: last 3 messages + 2 new
                delta = original[-3:] + [
                    {"role": "user", "content": "delta-new-1"},
                    {"role": "assistant", "content": "delta-new-2"},
                ]
                replay_engine._ingest_messages(delta)

                count = replay_engine._store.get_session_count("fork-guard")
                # Guard skipped (tail overlap exists).  Existing reconcile
                # cannot prove full replay from 5 messages against 50, so
                # it persists the ambiguous delta (cursor=0).  The 3
                # overlapping messages are duplicated — this is the
                # pre-existing limitation the fork guard does NOT address.
                assert count == 55, (
                    f"Expected 55 (50 + 5 ambiguous delta), got {count}."
                )
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_t5_incoming_equals_session_count_guard_skipped(self, tmp_path):
        """incoming == session_count → guard not triggered."""
        engine = _make_engine(tmp_path)
        try:
            original = _make_messages(9, prefix="orig")
            engine._ingest_messages(original)
            assert engine._store.get_session_count("fork-guard") == 10

            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                # Same count, different content — guard must NOT skip
                divergent = _make_messages(9, prefix="divergent")
                replay_engine._ingest_messages(divergent)

                count = replay_engine._store.get_session_count("fork-guard")
                # Guard skipped (10 < 10 is false), falls through to
                # existing logic which persists the ambiguous delta
                assert count == 20, (
                    f"Expected 20 (guard skipped, delta persisted), got {count}."
                )
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_t6_very_short_session_guard_skipped(self, tmp_path):
        """Very short session (3 msgs) + restart → guard not triggered."""
        engine = _make_engine(tmp_path)
        try:
            original = _make_messages(2, prefix="orig")
            engine._ingest_messages(original)
            assert engine._store.get_session_count("fork-guard") == 3

            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                divergent = _make_messages(2, prefix="divergent")
                replay_engine._ingest_messages(divergent)

                count = replay_engine._store.get_session_count("fork-guard")
                # 3 < 3 is false → guard skipped → existing logic
                assert count == 6, (
                    f"Expected 6 (guard skipped), got {count}."
                )
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_t6b_small_delta_not_skipped(self, tmp_path):
        """1-5 message delta against large session → guard not triggered.

        Legitimate restarts often deliver a small delta of new messages.
        The guard requires incoming > 5 to avoid skipping these.
        """
        engine = _make_engine(tmp_path)
        try:
            original = _make_messages(49, prefix="orig")
            engine._ingest_messages(original)
            assert engine._store.get_session_count("fork-guard") == 50

            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                # Single new message — must NOT be skipped
                delta = [{"role": "user", "content": "brand-new-message"}]
                replay_engine._ingest_messages(delta)

                count = replay_engine._store.get_session_count("fork-guard")
                assert count == 51, (
                    f"Expected 51 (50 + 1 new), got {count}. "
                    "Guard incorrectly skipped a small delta."
                )
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_t7_empty_session_early_return(self, tmp_path):
        """Empty session → early return, guard never reached."""
        engine = _make_engine(tmp_path)
        try:
            assert engine._store.get_session_count("fork-guard") == 0

            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                msgs = _make_messages(4, prefix="new")
                replay_engine._ingest_messages(msgs)

                count = replay_engine._store.get_session_count("fork-guard")
                assert count == 5, (
                    f"Expected 5 (fresh ingest), got {count}."
                )
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_t8_fork_with_tail_overlap_guard_skipped(self, tmp_path):
        """Fork carries parent's tail → guard skipped (accepted false neg)."""
        engine = _make_engine(tmp_path)
        try:
            original = _make_messages(49, prefix="orig")
            engine._ingest_messages(original)
            assert engine._store.get_session_count("fork-guard") == 50

            fork_engine = _make_engine(tmp_path)
            try:
                fork_engine._ingest_cursor_needs_reconcile = True
                # Fork carries parent's last 2 messages (tail overlap)
                fork_msgs = [
                    {"role": "system", "content": "Fork system prompt."},
                    {"role": "user", "content": "fork-unique-1"},
                    {"role": "assistant", "content": "fork-unique-2"},
                ] + original[-2:]
                fork_engine._ingest_messages(fork_msgs)

                count = fork_engine._store.get_session_count("fork-guard")
                # Guard skipped (tail overlap exists), existing logic handles
                assert count > 50, (
                    f"Expected >50 (guard skipped, fork persisted), got {count}."
                )
            finally:
                fork_engine.shutdown()
        finally:
            engine.shutdown()


