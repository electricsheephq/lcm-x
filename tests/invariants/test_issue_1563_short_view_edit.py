"""#1563 (a v0.30.0 named loss limit): an in-place edit of an earlier row inside a shorter host view is stored."""
from __future__ import annotations

import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.harness import engine_context


def message(content: str, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": content}


def _counts(tmp_path: Path, views: list) -> tuple[list[int], list[str]]:
    home = tmp_path / "home"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")
    counts = []
    context: Any = engine_context
    with context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        for view in views:
            engine.ingest(deepcopy(view))
            counts.append(len(engine._store.get_session_messages("parent")))
        texts = [row["content"] for row in engine._store.get_session_messages("parent")]
    return counts, texts


FULL = [message(f"request {n}" if n % 2 == 0 else f"reply {n}", "user" if n % 2 == 0 else "assistant")
        for n in range(8)]


def test_an_unchanged_short_view_stores_nothing_new(tmp_path: Path) -> None:
    """Control: the same 8 -> 5 -> 8 views with no edit store nothing new (no copy of an unchanged row)."""
    counts, texts = _counts(tmp_path, [FULL, FULL[:5], FULL])
    assert counts == [8, 8, 8]
    assert sorted(texts) == sorted(row["content"] for row in FULL)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1563: an edit inside a shorter host view sits below the ingest cursor, so the "
                                                              "edited text is never stored (no host-edit signal; v0.27.0 loses it too)")
@pytest.mark.parametrize("edited", [1, 4])
def test_an_edit_inside_a_short_prefix_is_stored_once(tmp_path: Path, edited: int) -> None:
    """Shape 1: 8 unstamped rows, then a 5-row view whose row 1 (or 4) has new text, then the original 8 rows.
    Today every release stores 8 -> 8 -> 8; the goal is 8 -> 9 -> 9."""
    short = deepcopy(FULL[:5])
    short[edited]["content"] = f"edited row {edited} text"
    counts, texts = _counts(tmp_path, [FULL, short, FULL])
    assert counts[0] == 8
    assert texts.count(short[edited]["content"]) == 1, counts
    assert counts == [8, 9, 9]
