"""I1/I2: an accounted-for shorter host view must not hide the next occurrence."""
from __future__ import annotations

import sqlite3
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.harness import engine_context


def message(content: str, stamp: float, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": content, "timestamp": stamp}


def lineage(root: Path) -> None:
    home = root / "home"
    home.mkdir(exist_ok=True)
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.execute("INSERT INTO sessions VALUES ('parent', NULL, NULL)")


def original() -> list[dict[str, Any]]:
    return [message("earlier request", 10.), message("earlier reply", 11., "assistant"),
            message("rebuild the index", 20.), message("reply", 21., "assistant"),
            message("later request", 40.), message("later reply", 41., "assistant")]


def assert_occurrences(engine: Any, expected: list[dict[str, Any]]) -> None:
    rows = engine._store.get_session_messages("parent")
    assert [(r["role"], r["content"], r["observed_at"]) for r in rows] == [
        (r["role"], r["content"], r["timestamp"]) for r in expected]


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
@pytest.mark.parametrize("keep", [0, 3, 2, 4], ids=["no-trim", "keep3", "keep2", "keep4"])
@pytest.mark.parametrize("separate", [True, False], ids=["shrink-alone", "shrink-append"])
@pytest.mark.parametrize("repeated", [True, False], ids=["repeated", "distinct"])
@pytest.mark.parametrize("side", ["front", "tail"])
def test_shrink_then_append(tmp_path: Path, cold: bool, keep: int, separate: bool,
                            repeated: bool, side: str) -> None:
    """Port the trim/repeat family, plus CLI undo's retained-prefix siblings."""
    lineage(tmp_path)
    context: Any = engine_context
    full = original()
    with ExitStack() as stack:
        engine = stack.enter_context(context(tmp_path, window=128000, session_id="parent", conversation_id="conv"))
        engine.ingest(deepcopy(full))
        if cold:
            stack.close()
            engine = stack.enter_context(context(tmp_path, window=128000, session_id="parent", conversation_id="conv"))
            engine.ingest(deepcopy(full))
        view = (full[-keep:] if side == "front" else full[:keep]) if keep else full
        if keep and separate:
            engine.ingest(deepcopy(view))
            assert_occurrences(engine, full)
        fresh = [message("rebuild the index" if repeated else "a brand new request", 50.),
                 message("its reply", 51., "assistant")]
        engine.ingest(deepcopy(view + fresh))
        assert_occurrences(engine, full + fresh)
        engine.ingest(deepcopy(view + fresh))
        assert_occurrences(engine, full + fresh)


@pytest.mark.parametrize("compress", [True, False], ids=["noop-compress", "direct-control"])
def test_undo_then_prompt(tmp_path: Path, compress: bool) -> None:
    """Engine emulation of four exchanges, undo, full no-leaf compress, prompt."""
    lineage(tmp_path)
    context: Any = engine_context
    full = original() + [message("fourth request", 60.), message("fourth reply", 61., "assistant")]
    with context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        view = deepcopy(full[:-2])
        if compress:
            output = engine.compress(view, current_tokens=100, force=True)
            assert output == view
            assert engine._last_compression_status != "compacted"
            assert engine.current_session_id == "parent"
        fresh = [message("unique prompt after undo", 70.), message("unique reply after undo", 71., "assistant")]
        engine.ingest(deepcopy(view + fresh))
        assert_occurrences(engine, full + fresh)


@pytest.mark.parametrize("side", ["front", "tail"])
def test_shrink_then_reexpand(tmp_path: Path, side: str) -> None:
    lineage(tmp_path)
    context: Any = engine_context
    full = original()
    with context(tmp_path, window=128000, session_id="parent", conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        engine.ingest(deepcopy(full[-3:] if side == "front" else full[:3]))
        fresh = message("new after restored rows", 50.)
        engine.ingest(deepcopy(full + [fresh]))
        assert_occurrences(engine, full + [fresh])
