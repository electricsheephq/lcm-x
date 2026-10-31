"""Keep the deferred #1525 undo/compress cursor loss visible as a strict xfail."""
from __future__ import annotations

import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.harness import engine_context


class StoredRowsMismatch(AssertionError):
    """Only the #1525 stored-row loss may satisfy the strict xfail."""


def message(content: str, stamp: float, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": content, "timestamp": stamp}


@pytest.mark.xfail(
    strict=True, raises=StoredRowsMismatch,
    reason="#1525: after an undo and an unchanged compress, the next prompt's rows sit below the ingest cursor and are not stored; deferred to host occurrence keys",
)
def test_undo_then_prompt(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")
    full = [message("earlier request", 10.), message("earlier reply", 11., "assistant"),
            message("rebuild the index", 20.), message("reply", 21., "assistant"),
            message("later request", 40.), message("later reply", 41., "assistant"),
            message("fourth request", 60.), message("fourth reply", 61., "assistant")]
    context: Any = engine_context
    with context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        view = deepcopy(full[:-2])
        output = engine.compress(view, current_tokens=100, force=True)
        assert output == view
        assert engine._last_compression_status != "compacted"
        assert engine.current_session_id == "parent"
        fresh = [message("unique prompt after undo", 70.), message("unique reply after undo", 71., "assistant")]
        engine.ingest(deepcopy(view + fresh))
        rows = engine._store.get_session_messages("parent")
        stored = [(row["role"], row["content"], row["observed_at"]) for row in rows]
        expected = [(row["role"], row["content"], row["timestamp"]) for row in full + fresh]
        if stored != expected:
            raise StoredRowsMismatch(f"stored {len(stored)} rows; expected {len(expected)}")
