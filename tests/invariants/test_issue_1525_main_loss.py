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
        if output != view or engine._last_compression_status == "compacted":
            pytest.fail(f"setup: the compress changed the view ({engine._last_compression_status!r})")
        if engine.current_session_id != "parent":
            pytest.fail(f"setup: the compress rotated the session to {engine.current_session_id!r}")
        fresh = [message("unique prompt after undo", 70.), message("unique reply after undo", 71., "assistant")]
        engine.ingest(deepcopy(view + fresh))
        rows = engine._store.get_session_messages("parent")
        stored = [(row["role"], row["content"], row["observed_at"]) for row in rows]
        expected = [(row["role"], row["content"], row["timestamp"]) for row in full + fresh]
        _expect_stored_shape(stored, expected, expected[:len(full)],
                             f"#1525 known shape: {stored}", StoredRowsMismatch)


def _expect_stored_shape(observed, fixed, known, message, error_type=AssertionError):
    """Accept only the complete fixed list or the complete named-limit list, in store order."""
    if observed == fixed:
        return
    if observed == known:
        raise error_type(message)
    differences = []
    for label, expected in (("fixed", fixed), ("known", known)):
        index = next((i for i, (actual, wanted) in enumerate(zip(observed, expected)) if actual != wanted),
                     min(len(observed), len(expected)))
        differences.append(f"{label}: first difference={index}, lengths={len(observed)}/{len(expected)}")
    pytest.fail("unexpected stored shape; " + "; ".join(differences))


@pytest.mark.parametrize("case", ["known", "fixed", "doubled", "swapped"])
def test_stored_shape_comparison(case):
    fixed = [("user", "first"), ("assistant", "second")]
    known = [fixed[0]]
    observed = {"known": known, "fixed": fixed, "doubled": [fixed[0], *fixed], "swapped": fixed[::-1]}[case]
    if case == "known":
        with pytest.raises(StoredRowsMismatch, match="synthetic known shape"):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", StoredRowsMismatch)
    elif case == "fixed":
        _expect_stored_shape(observed, fixed, known, "synthetic known shape", StoredRowsMismatch)
    else:
        with pytest.raises(pytest.fail.Exception, match="first difference=.*lengths="):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", StoredRowsMismatch)
