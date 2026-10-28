"""Shared-engine auxiliary compaction must preserve a pending foreground commit."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

# conftest registers the plugin under this runtime-only package name.
import hermes_lcm.engine as lcm_engine  # type: ignore[import-not-found]
import hermes_lcm.survival_fit as survival_fit  # type: ignore[import-not-found]
import hermes_lcm.tokens as tokens  # type: ignore[import-not-found]
from hermes_lcm.config import LCMConfig  # type: ignore[import-not-found]

Message = dict[str, Any]
SID = "foreground"
CONVERSATION = "foreground-conversation"
PAD = " alpha beta gamma delta" * 30


class ForegroundProofCleared(Exception):
    """The auxiliary compress discarded the pending foreground commit proof."""


@contextmanager
def _compacted_foreground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Any, list[Message], list[Message]]]:
    calls: list[str] = []

    def summarize(**kwargs: Any) -> tuple[str, int]:
        calls.append(str(kwargs["text"]))
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: (len(text) // 4 + 1) if text else 0)
    engine = lcm_engine.LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "externalized"),
            fresh_tail_count=2,
            leaf_chunk_tokens=100,
            fresh_tail_pressure_yield_enabled=False,
            temporal_rollups_enabled=False,
            embeddings_enabled=False,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        engine.on_session_start(
            SID, platform="acp", conversation_id=CONVERSATION, context_length=200_000,
        )
        original: list[Message] = []
        for turn in range(6):
            original.extend([
                {"role": "user", "content": f"turn {turn}{PAD}", "timestamp": 10.0 * (turn + 1)},
                {"role": "assistant", "content": f"reply {turn}{PAD}"},
            ])
            engine.ingest(original)
        adopted = engine.compress(list(original), force=True)
        assert calls and engine._last_compression_status == "compacted"
        assert len(adopted) < len(original)
        assert engine._compress_commit_proof is not None
        assert engine._compress_commit_proof["session_id"] == SID
        assert engine._ingest_cursor == len(adopted)
        yield engine, original, adopted
    finally:
        engine.shutdown()


def _compress_auxiliary(engine: Any) -> None:
    """Use the real thread-local bypass, as in the auxiliary no-progress tests."""
    def run_child() -> None:
        engine._mark_thread_context_stateless("aux-review")
        try:
            assert engine._bypasses_lcm_context_management()
            assert engine._thread_context_session_id() == "aux-review"
            messages = [{"role": "user", "content": "auxiliary review input"}]
            assert engine.compress(messages) == messages
        finally:
            engine._clear_thread_context_stateless()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(run_child).result()
    assert not engine._bypasses_lcm_context_management()
    assert engine.current_session_id == SID
    assert engine.current_conversation_id == CONVERSATION
    assert engine._store.get_session_count("aux-review") == 0


def test_auxiliary_compress_preserves_foreground_commit_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _compacted_foreground(tmp_path, monkeypatch) as (engine, _original, _adopted):
        proof = engine._compress_commit_proof
        _compress_auxiliary(engine)
        if engine._compress_commit_proof is not proof:
            raise ForegroundProofCleared("auxiliary compress cleared the foreground commit proof")


@pytest.mark.parametrize("with_child", [
    pytest.param(False, id="foreground-control"),
    pytest.param(True, id="shared-child"),
])
def test_host_commit_stores_each_foreground_row_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, with_child: bool,
) -> None:
    """Measure the end/start/adopt path even if the pending proof disappears."""
    caplog.set_level(logging.INFO)
    with _compacted_foreground(tmp_path, monkeypatch) as (engine, original, adopted):
        before = engine._store.get_session_messages(SID)
        expected = [(row["role"], row["content"]) for row in original]
        assert [(row["role"], row["content"]) for row in before] == expected
        proof = engine._compress_commit_proof
        if with_child:
            _compress_auxiliary(engine)
        proof_intact = engine._compress_commit_proof is proof
        engine.on_session_end(SID, original)
        rows_at_end = engine._store.get_session_count(SID)
        if proof_intact:
            assert engine._compress_commit_proof["end_consumed"] is True
        engine.on_session_start(
            SID, boundary_reason="compression", old_session_id=SID,
            platform="acp", conversation_id=CONVERSATION,
        )
        boundary = [record.getMessage() for record in caplog.records
                    if "LCM in-place compression boundary kept" in record.getMessage()]
        assert len(boundary) == 1
        cursor = str(len(adopted)) if proof_intact else "reconcile"
        assert f"cursor={cursor})" in boundary[0]
        assert engine._ingest_cursor_needs_reconcile is (not proof_intact)
        engine.ingest(list(adopted))
        rows_after_adoption = engine._store.get_session_count(SID)
        reply = {"role": "assistant", "content": "reply after adoption"}
        engine.ingest([*adopted, reply])
        expected.append((reply["role"], reply["content"]))
        stored = engine._store.get_session_messages(SID)
        actual = [(row["role"], row["content"]) for row in stored]
        duplicates = list((Counter(actual) - Counter(expected)).elements())
        lost = list((Counter(expected) - Counter(actual)).elements())
        print(f"with_child={with_child} proof_intact={proof_intact} "
              f"rows: before={len(before)} end={rows_at_end} adopted={rows_after_adoption} "
              f"after_reply={len(stored)} roles={[row['role'] for row in stored]} "
              f"duplicates={len(duplicates)} lost={len(lost)}")
        print(f"duplicate_rows={[(role, content.removesuffix(PAD)) for role, content in duplicates]}")
        print(boundary[0])
        # #1291's symptom is re-stored rows with none lost; a lost row is a new failure, not this one.
        if not proof_intact and duplicates and not lost:
            raise ForegroundProofCleared("foreground rows were stored again after the commit proof was cleared")
        assert actual == expected
        assert not duplicates and not lost
        assert rows_at_end == rows_after_adoption == len(original)
