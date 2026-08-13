"""Atomic occurrence provenance for LCM-X issue #3."""

from __future__ import annotations

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _literal_scaffold() -> dict:
    return {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 1)]\n"
            "Literal user-authored text\n"
            "[Expand for details: literal request]"
        ),
    }


def test_user_scaffold_occurrence_and_provenance_commit_together(tmp_path) -> None:
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "issue-3-atomic.db"))
    )
    engine.on_session_start(
        "issue-3-atomic",
        platform="cli",
        context_length=200_000,
    )

    try:
        engine._ingest_messages([_literal_scaffold()])
        rows = engine._store.get_session_messages(engine._session_id)
        store_id = rows[0]["store_id"]
        assert engine._has_real_user_scaffold_provenance(store_id)
    finally:
        engine.shutdown()


def test_provenance_failure_rolls_back_message_batch(
    tmp_path,
    monkeypatch,
) -> None:
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "issue-3-rollback.db"))
    )
    engine.on_session_start(
        "issue-3-rollback",
        platform="cli",
        context_length=200_000,
    )

    def fail_metadata(_message, _store_id):
        raise OSError("simulated provenance failure")

    monkeypatch.setattr(
        engine,
        "_real_user_scaffold_metadata_rows",
        fail_metadata,
    )
    try:
        with pytest.raises(OSError, match="simulated provenance failure"):
            engine._ingest_messages([_literal_scaffold()])
        assert engine._store.get_session_count(engine._session_id) == 0
    finally:
        engine.shutdown()


def test_late_session_end_suffix_commits_scaffold_provenance(tmp_path) -> None:
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "issue-3-late-end.db"))
    )
    engine.on_session_start(
        "session-a",
        platform="cli",
        context_length=200_000,
    )
    engine.on_session_start(
        "session-b",
        platform="cli",
        context_length=200_000,
    )

    try:
        engine.on_session_end("session-a", [_literal_scaffold()])
        rows = engine._store.get_session_messages("session-a")
        assert len(rows) == 1
        assert engine._has_real_user_scaffold_provenance(rows[0]["store_id"])
    finally:
        engine.shutdown()
