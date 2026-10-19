"""I1/I2: preserve reset/restart resends while pinning the #1570 limitation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests import harness

Row = dict[str, Any]
make_engine: Any = harness.make_engine
MIXES = {
    "none": "nnnnnnnn", "stamp": "ssssssss", "stamphead": "ssssnnnn",
    "alt": "sunsunsu", "s0only": "snnnnnnn", "stamptail": "nnnnssss",
    "uidhead": "uuuunnnn",
}
SHAPES: dict[str, list[list[int] | str]] = {
    "LC9a": ["RESET", list(range(9))],
    "LC9b": ["RESET", [0, 1, 2, 3, 8, 4, 5, 6, 7]],
    "LC10a": ["RESTART", list(range(9))],
    "LC10b": ["RESTART", [0, 1, 2, 3, 8, 4, 5, 6, 7]],
    "LC10c": [[0, 1, 2, 3, 8, 4, 5, 6, 7], "RESTART", [0, 1, 2, 3, 8, 4, 5, 6, 7, 9]],
}


def run_shape(
    tmp_path: Path, shape: str, metadata: str,
) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    rows: list[Row] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"row-{i}"}
        for i in range(8)
    ]
    for i, (row, kind) in enumerate(zip(rows, MIXES[metadata])):
        if kind == "s":
            row["timestamp"] = float(10 + i)
        elif kind == "u":
            row["message_uid"] = f"old-{i}"
    rows.extend({"role": "user", "content": f"fresh-{name}"} for name in ("F", "G"))
    steps: list[list[int] | str] = [list(range(8)), list(range(4)), [0, 1, 2, 3, 8]]
    sent: set[tuple[str, str]] = set()
    engine = make_engine(tmp_path, window=128000, conversation_id="conv")
    try:
        for step in steps + SHAPES[shape]:
            if step == "RESET":
                engine.on_session_reset()
            elif step == "RESTART":
                engine.shutdown()
                engine = make_engine(tmp_path, window=128000, conversation_id="conv")
            else:
                assert isinstance(step, list)
                view = deepcopy([rows[i] for i in step])
                sent.update((row["role"], row["content"]) for row in view)
                engine.ingest(view)
        stored = [(row["role"], row["content"]) for row in engine._store.get_session_messages("S")]
        return stored, sent
    finally:
        engine.shutdown()


@pytest.mark.parametrize("metadata", MIXES)
@pytest.mark.parametrize("shape", ["LC9a", "LC9b", "LC10b", "LC10c"])
def test_reset_restart_resend_is_bounded(tmp_path: Path, shape: str, metadata: str) -> None:
    stored, sent = run_shape(tmp_path, shape, metadata)
    assert sent <= set(stored)
    assert 1 <= stored.count(("user", "fresh-F")) <= 2


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#1570: a reset or restart ends the in-memory replay episode, so the resent unkeyed row is stored again; host occurrence keys fix this",
)
@pytest.mark.parametrize("metadata", MIXES)
@pytest.mark.parametrize("shape", ["LC9a", "LC9b", "LC10b", "LC10c"])
def test_reset_restart_resend_stores_once(tmp_path: Path, shape: str, metadata: str) -> None:
    stored, _ = run_shape(tmp_path, shape, metadata)
    assert stored.count(("user", "fresh-F")) == 1


@pytest.mark.parametrize("metadata", MIXES)
def test_restart_then_full_regrow_stores_once(tmp_path: Path, metadata: str) -> None:
    stored, _ = run_shape(tmp_path, "LC10a", metadata)
    assert stored.count(("user", "fresh-F")) == 1
