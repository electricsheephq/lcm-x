"""Regression tests for compaction-scaffold-aware replay identity.

After context compression the host appends a volatile task-list annotation
(``_PRESERVED_TODO_CONTEXT_PREFIX``) to the last user message.  The annotation
changes on every compression cycle (task statuses update), so including it in
the replay identity causes ``_find_reconciled_cursor_for_store_tail`` to fail
suffix matching, fall through to ``cursor=0``, and re-persist every message —
creating duplicates with distinct store_ids and timestamps.

These tests verify that:
1. ``_message_replay_identity`` strips the todo annotation before hashing.
2. Reconciliation after a compression boundary correctly advances the cursor
   even when the todo annotation changed between compaction cycles.
3. ``_is_replayed_context_scaffold_message`` recognises standalone todo
   annotation messages as scaffolding (not durable user content).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

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
from hermes_lcm.reconcile import (
    _PRESERVED_TODO_CONTEXT_PREFIX,
)

_TODO_ANNOTATION_V1 = (
    f"\n\n{_PRESERVED_TODO_CONTEXT_PREFIX}\n"
    "- [>] 1. First task (in_progress)\n"
    "- [ ] 2. Second task (pending)\n"
)

_TODO_ANNOTATION_V2 = (
    f"\n\n{_PRESERVED_TODO_CONTEXT_PREFIX}\n"
    "- [x] 1. First task (completed)\n"
    "- [>] 2. Second task (in_progress)\n"
    "- [ ] 3. Third task (pending)\n"
)


def _make_engine(tmp_path: Path, *, session_id: str = "reconcile-todo") -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine._session_id = session_id
    return engine


class TestTodoAnnotationIdentity:
    """_message_replay_identity must be stable across todo annotation changes."""

    def test_identity_strips_todo_annotation(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg_plain = {"role": "user", "content": "Do the thing"}
            msg_annotated = {
                "role": "user",
                "content": "Do the thing" + _TODO_ANNOTATION_V1,
            }
            msg_annotated_v2 = {
                "role": "user",
                "content": "Do the thing" + _TODO_ANNOTATION_V2,
            }

            id_plain = engine._message_replay_identity(msg_plain)
            id_v1 = engine._message_replay_identity(msg_annotated)
            id_v2 = engine._message_replay_identity(msg_annotated_v2)

            assert id_plain == id_v1, (
                "Identity with todo annotation v1 must match plain content"
            )
            assert id_plain == id_v2, (
                "Identity with todo annotation v2 must match plain content"
            )
            assert id_v1 == id_v2, (
                "Identity must be stable across different todo annotation versions"
            )
        finally:
            engine.shutdown()

    def test_identity_preserves_content_without_annotation(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg_a = {"role": "user", "content": "Alpha message"}
            msg_b = {"role": "user", "content": "Beta message"}
            assert (
                engine._message_replay_identity(msg_a)
                != engine._message_replay_identity(msg_b)
            )
        finally:
            engine.shutdown()

    def test_identity_strips_annotation_from_stored_row(self, tmp_path):
        """Stored rows carry the annotation verbatim; identity must still strip."""
        engine = _make_engine(tmp_path)
        try:
            stored = {
                "role": "user",
                "content": "Persisted content" + _TODO_ANNOTATION_V1,
            }
            incoming = {
                "role": "user",
                "content": "Persisted content" + _TODO_ANNOTATION_V2,
            }
            id_stored = engine._message_replay_identity(stored, stored_row=True)
            id_incoming = engine._message_replay_identity(incoming)
            assert id_stored == id_incoming
        finally:
            engine.shutdown()


class TestTodoAnnotationScaffoldDetection:
    """Standalone todo annotation messages must be recognised as scaffolding."""

    def test_standalone_todo_is_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg = {
                "role": "user",
                "content": _PRESERVED_TODO_CONTEXT_PREFIX + "\n- [ ] task",
            }
            assert engine._is_replayed_context_scaffold_message(msg) is True
        finally:
            engine.shutdown()

    def test_objective_prefix_still_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg = {
                "role": "user",
                "content": "[Current user objective preserved from compacted history]\nDo X",
            }
            assert engine._is_replayed_context_scaffold_message(msg) is True
        finally:
            engine.shutdown()

    def test_regular_user_message_not_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg = {"role": "user", "content": "Just a normal question"}
            assert engine._is_replayed_context_scaffold_message(msg) is False
        finally:
            engine.shutdown()


class TestReconcileAfterCompactionWithTodoAnnotation:
    """End-to-end: ingest → compress → re-ingest with changed annotation."""

    def test_no_duplication_when_todo_annotation_changes(self, tmp_path):
        """Simulates the production bug: compaction changes the todo annotation
        on the last user message, then reconciliation must still recognise the
        stored messages and advance the cursor past them."""
        engine = _make_engine(tmp_path)
        try:
            # Phase 1: initial ingest (pre-compaction)
            original_messages = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "First user request"},
                {"role": "assistant", "content": "First assistant reply"},
                {"role": "user", "content": "Second user request" + _TODO_ANNOTATION_V1},
                {"role": "assistant", "content": "Second assistant reply"},
            ]
            engine._ingest_messages(original_messages)
            count_after_first = engine._store.get_session_count("reconcile-todo")
            assert count_after_first == 5

            # Phase 2: simulate compression boundary + restart.
            # The host rebuilds the message list with a DIFFERENT todo
            # annotation (task statuses changed during compaction).
            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                replay_messages = [
                    {"role": "system", "content": "System prompt."},
                    {"role": "user", "content": "First user request"},
                    {"role": "assistant", "content": "First assistant reply"},
                    {"role": "user", "content": "Second user request" + _TODO_ANNOTATION_V2},
                    {"role": "assistant", "content": "Second assistant reply"},
                    {"role": "user", "content": "Brand new question after restart"},
                ]
                replay_engine._ingest_messages(replay_messages)

                count_after_replay = replay_engine._store.get_session_count(
                    "reconcile-todo"
                )
                # Must be 6 (5 original + 1 new), NOT 11 (5 dup + 5 + 1).
                assert count_after_replay == 6, (
                    f"Expected 6 messages (5 original + 1 new), got {count_after_replay}. "
                    "Duplication bug: reconciliation failed to match stored tail "
                    "because the todo annotation changed."
                )

                # Verify no content duplication
                rows = replay_engine._store.get_session_messages("reconcile-todo")
                contents = [r["content"] for r in rows]
                assert contents.count("First user request") == 1
                assert contents.count("First assistant reply") == 1
                assert contents.count("Second assistant reply") == 1
                assert contents.count("Brand new question after restart") == 1
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_no_duplication_with_identical_annotation(self, tmp_path):
        """Control: identical annotation should also produce no duplication."""
        engine = _make_engine(tmp_path)
        try:
            original_messages = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "User request" + _TODO_ANNOTATION_V1},
                {"role": "assistant", "content": "Assistant reply"},
            ]
            engine._ingest_messages(original_messages)
            assert engine._store.get_session_count("reconcile-todo") == 3

            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                replay_messages = [
                    {"role": "system", "content": "System prompt."},
                    {"role": "user", "content": "User request" + _TODO_ANNOTATION_V1},
                    {"role": "assistant", "content": "Assistant reply"},
                    {"role": "user", "content": "Follow-up"},
                ]
                replay_engine._ingest_messages(replay_messages)
                assert replay_engine._store.get_session_count("reconcile-todo") == 4
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()


# #516: the host folds its todo snapshot into the trailing real user row, and its
# consecutive-user merge (hermes-agent agent/agent_runtime_helpers.py
# _merge_consecutive_users: ``prev + "\n\n" + next``) can glue the NEXT user row
# behind that annotation.  The identity cut must drop only the annotation span.
_NEW = "NEW instruction typed before any reply"
_RELOAD_NOTICE = (
    "[Skills pruned during compression — reload before acting on these tasks]\n"
    "The task list above crossed the compression boundary verbatim, but the skill "
    "instructions that governed it were pruned. Before executing any preserved task that "
    "depends on these skills, reload them first: skill_view(name='deploy'). After reloading, "
    "re-check that each pending task is still justified — findings recorded before the "
    "boundary may have invalidated it."
)
_TODO_ANNOTATION_NESTED = (
    f"\n\n{_PRESERVED_TODO_CONTEXT_PREFIX}\n"
    "- [x] ship. Ship the release (completed)\n"
    "  - [>] ship.1. Tag the build (in_progress)\n"
    "    - [ ] ship.1.a. Push the tag (pending)\n"
    "- [ ] 4. Write notes (pending)"
    f"\n\n{_RELOAD_NOTICE}"
)


def _identity(engine, content, **kwargs):
    return engine._message_replay_identity({"role": "user", "content": content}, **kwargs)


class TestTodoAnnotationSpanKeepsMergedRow:
    """#516: only the annotation span is volatile; a row merged behind it is content."""

    def test_merged_row_keeps_its_identity_distinct(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            stored = _identity(engine, "Do the thing" + _TODO_ANNOTATION_V1, stored_row=True)
            merged = _identity(engine, "Do the thing" + _TODO_ANNOTATION_V2.rstrip("\n") + "\n\n" + _NEW)
            assert merged != stored, "the merged NEW row must survive the annotation cut"
            assert merged[1] == "Do the thing\n\n" + _NEW
            assert _identity(engine, "Do the thing" + _TODO_ANNOTATION_V2) == stored
        finally:
            engine.shutdown()

    def test_nested_items_and_reload_notice_are_one_span(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            stored = _identity(engine, "Do the thing" + _TODO_ANNOTATION_V1, stored_row=True)
            assert _identity(engine, "Do the thing" + _TODO_ANNOTATION_NESTED) == stored
            merged = _identity(engine, "Do the thing" + _TODO_ANNOTATION_NESTED + "\n\n" + _NEW)
            assert merged[1] == "Do the thing\n\n" + _NEW
        finally:
            engine.shutdown()

    def test_item_text_holding_a_blank_line_runs_on_to_its_status(self, tmp_path):
        """The host keeps an item's text verbatim, blank lines included: the span ends at its status."""
        engine = _make_engine(tmp_path)
        try:
            annotation = f"\n\n{_PRESERVED_TODO_CONTEXT_PREFIX}\n- [>] 5. Line one\n\nline two (in_progress)"
            stored = _identity(engine, "Do the thing" + _TODO_ANNOTATION_V1, stored_row=True)
            assert _identity(engine, "Do the thing" + annotation) == stored
            assert _identity(engine, "Do the thing" + annotation + "\n\n" + _NEW)[1] == "Do the thing\n\n" + _NEW
        finally:
            engine.shutdown()

    def test_header_without_items_is_the_whole_span(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            content = f"Do the thing\n\n{_PRESERVED_TODO_CONTEXT_PREFIX}\n\nfree text"
            identity = _identity(engine, content)
            assert identity[1] == "Do the thing\n\nfree text"
            assert _PRESERVED_TODO_CONTEXT_PREFIX not in identity[1]
        finally:
            engine.shutdown()


class TestTodoAnnotationSpanFixRound1:
    """#516 fix round 1: exactly one host delimiter is removed after the span, and a model-switch
    note at the head of the merged row is stripped by the same rule as at the head of any row."""

    _HOST_ANNOTATION = _TODO_ANNOTATION_V2.rstrip("\n")  # the host renders no trailing newline

    def test_merged_rows_differing_in_leading_newlines_keep_distinct_identities(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            head = "Do the thing" + self._HOST_ANNOTATION + "\n\n"
            assert _identity(engine, head + _NEW) != _identity(engine, head + "\n" + _NEW)
            assert _identity(engine, head + _NEW)[1] == "Do the thing\n\n" + _NEW
        finally:
            engine.shutdown()

    def test_merged_row_keeps_its_own_leading_newline(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            identity = _identity(engine, "Do the thing" + self._HOST_ANNOTATION + "\n\n\n" + _NEW)
            assert identity[1] == "Do the thing\n\n\n" + _NEW
        finally:
            engine.shutdown()

    def test_model_switch_note_heading_the_merged_row_is_stripped(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            head = "Do the thing" + self._HOST_ANNOTATION + "\n\n"
            note = "[Note: model was just switched from qwen3.8 to kimi-k3 via router.]\n\n"
            assert _identity(engine, head + note + _NEW) == _identity(engine, head + _NEW)
        finally:
            engine.shutdown()


def _compacted_tail_one(tmp_path, monkeypatch, mode):
    """A real tail-1 compaction; the host folds the todo snapshot into the retained trailing user row."""
    from tests.test_issue_488_emission_proof import _phase1_compacted_engine

    engine, pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=1)
    child = "S0" if mode == "inplace" else "S1"
    engine.on_session_end("S0", pre)
    engine.on_session_start(child, boundary_reason="compression", old_session_id="S0", platform="acp")
    tail_row = compressed[-1]["content"]
    folded = [dict(m) for m in compressed[:-1]] + [{"role": "user", "content": tail_row + _TODO_ANNOTATION_V1.rstrip("\n")}]
    engine.ingest(folded)
    return engine, compressed, child, tail_row


def _stored_user_identities(engine):
    from tests.test_issue_488_emission_proof import _stored_user_rows

    return [_identity(engine, content, stored_row=True)[1] for _store_id, content in _stored_user_rows(engine)]


def _host_merged(compressed, tail_row, annotation, *, glue_carrier):
    """``[.., U + todo]`` then NEW with no reply between, merged as hermes-agent v2026.9.24
    agent/agent_runtime_helpers.py:553 ``_merge_consecutive_users`` does (``prev + "\\n\\n" + next``;
    a summary carrier is never merged into) or, ``glue_carrier``, as the pinned 37aad38c copy does."""
    rows = [dict(m) for m in compressed[:-1]] + [{"role": "user", "content": tail_row + annotation + "\n\n" + _NEW}]
    if glue_carrier:
        rows = [{"role": "user", "content": "\n\n".join(m["content"] for m in rows)}]
    return rows


@pytest.mark.parametrize("glue_carrier", [False, True], ids=["host-v2026.9.24", "pinned-37aad38c"])
@pytest.mark.parametrize("mode", ["inplace", "rotation"])
def test_merged_row_behind_the_annotation_is_stored_once_after_restart(tmp_path, monkeypatch, mode, glue_carrier):
    """#516: the host merges ``U + todo + NEW``. After a restart NEW is stored exactly once, whole
    (multiset over identities); a refreshed snapshot alone stores nothing."""
    from tests.test_issue_488_emission_proof import _restart, _stored_user_rows

    engine, compressed, child, tail_row = _compacted_tail_one(tmp_path, monkeypatch, mode)
    annotation = _TODO_ANNOTATION_V2.rstrip("\n")
    refreshed = [dict(m) for m in compressed[:-1]] + [{"role": "user", "content": tail_row + annotation}]
    merged = _host_merged(compressed, tail_row, annotation, glue_carrier=glue_carrier)
    try:
        before = _stored_user_identities(engine)
        engine = _restart(engine, tmp_path, child, tail=1)
        engine.ingest([dict(m) for m in refreshed])
        assert _stored_user_identities(engine) == before  # control: the refresh is a replay
        engine = _restart(engine, tmp_path, child, tail=1)
        engine.ingest([dict(m) for m in merged])
        stored_new = [content for _store_id, content in _stored_user_rows(engine) if _NEW in (content or "")]
        assert stored_new == [merged[-1]["content"]], "NEW must be stored exactly once, whole"
        after = _stored_user_identities(engine)
        gained = Counter(after) - Counter(before)
        expected = _identity(engine, merged[-1]["content"], stored_row=True)[1]
        assert expected.endswith(tail_row.rstrip() + "\n\n" + _NEW)  # the cut rstrips the head (rc4)
        assert [i for i in gained.elements() if _NEW in i] == [expected]
        assert after.count(tail_row) == before.count(tail_row)
        # A rotation child also stores the carrier once: the plain merge does the same on the base.
        assert sum(gained.values()) == (1 if mode == "inplace" or glue_carrier else 2)
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, reason="pre-existing #499 family: a row the host merged into the retained "
                   "tail row is re-ingested on each LATER restart; the plain merge does the same on the base")
@pytest.mark.parametrize("annotation", ["", _TODO_ANNOTATION_V2.rstrip("\n")], ids=["plain", "annotated"])
def test_merged_row_is_not_stored_again_on_a_later_restart(tmp_path, monkeypatch, annotation):
    from tests.test_issue_488_emission_proof import _restart

    engine, compressed, child, tail_row = _compacted_tail_one(tmp_path, monkeypatch, "inplace")
    merged = _host_merged(compressed, tail_row, annotation, glue_carrier=False)
    try:
        for _ in range(2):
            engine = _restart(engine, tmp_path, child, tail=1)
            engine.ingest([dict(m) for m in merged])
        assert sum(_NEW in identity for identity in _stored_user_identities(engine)) == 1
    finally:
        engine.shutdown()


def test_legacy_objective_head_cuts_only_the_annotation_span(tmp_path, monkeypatch):
    """#516 x F3': an objective head + annotation + a merged row returns the head, and the merged
    row is stored whole once; the annotation alone is still the emitted head (rc4 skip)."""
    from tests.test_compression_boundary import _config
    from tests.test_issue_488_consumer_switch import _legacy_v3_restart, _rows
    from tests.test_issue_488_emission_proof import _phase1_compacted_engine

    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=0)
    head = compressed[0]["content"]
    annotated = {"role": "user", "content": head + _TODO_ANNOTATION_V1}
    merged = {"role": "user", "content": head + _TODO_ANNOTATION_V2 + "\n\n" + _NEW}
    assert engine._legacy_objective_head(annotated) is None
    assert engine._legacy_objective_head(merged) == head
    engine = _legacy_v3_restart(engine, tmp_path, [dict(m) for m in compressed], tail=0)
    try:
        before = _rows(engine)
        engine.ingest([dict(annotated)])
        assert _rows(engine) == before  # control: the annotation alone is the emitted head
    finally:
        engine.shutdown()
    for _ in range(2):  # stored whole once; a further restart re-stores nothing
        engine = LCMEngine(config=_config(tmp_path, fresh_tail_count=0), hermes_home=str(tmp_path / "home"))
        try:
            engine.on_session_start("S0", platform="acp", context_length=200_000)
            engine.ingest([dict(merged)])
            assert [content for _role, content in _rows(engine) if _NEW in (content or "")] == [merged["content"]]
        finally:
            engine.shutdown()
