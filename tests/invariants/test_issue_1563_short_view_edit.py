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


def _counts(tmp_path: Path, views: list, stored_views=None) -> tuple[list[int], list[str]]:
    home = tmp_path / "home"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")
    counts = []
    context: Any = engine_context
    with context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        for index, view in enumerate(views):
            engine.ingest(deepcopy(view))
            rows = engine._store.get_session_messages("parent")
            counts.append(len(rows))
            if stored_views is not None:
                if engine._ingest_failure_count:
                    pytest.fail("setup: a short-view ingest failed")
                stored_views.extend((index, row["role"], row["content"]) for row in rows)
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
    observed = []
    counts, _ = _counts(tmp_path, [FULL, short, FULL], observed)
    fixed = [(index, row["role"], row["content"]) for index, view in enumerate(
        [FULL, [*FULL, short[edited]], [*FULL, short[edited]]]) for row in view]
    known = [(index, row["role"], row["content"]) for index in range(3) for row in FULL]
    _expect_stored_shape(observed, fixed, known,
                         f"#1563 known shape: counts={counts}, edited text not stored")


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
        with pytest.raises(AssertionError, match="synthetic known shape"):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
    elif case == "fixed":
        _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
    else:
        with pytest.raises(pytest.fail.Exception, match="first difference=.*lengths="):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
