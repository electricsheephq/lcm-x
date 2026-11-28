"""Identical, deterministic host views for the two busy-bar trees."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

CELLS = ("ingest", "compress", "session-end", "restart")
TURNS = 12
A_OPERATIONS = {"ingest": 3, "compress": TURNS + 3, "session-end": 3, "restart": 2}


def turn(number: int, owner: str = "A") -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": f"{owner} turn {number}: " + "alpha beta gamma delta " * 40,
         "timestamp": 1000.0 + number * 2},
        {"role": "assistant", "content": f"{owner} reply {number}: noted.",
         "timestamp": 1001.0 + number * 2},
    ]


def writer_view(number: int, rows: int) -> list[dict[str, Any]]:
    """A bounded, all-new B window; no row repeats across B operations."""
    offset = number * rows
    return [{"role": "user" if index % 2 == 0 else "assistant",
             "content": f"B row {offset + index}: " + "alpha beta gamma delta " * 40,
             "timestamp": 100_000.0 + offset + index}
            for index in range(rows)]


def start(engine: Any, owner: str = "A") -> None:
    engine.on_session_start(
        owner, platform="acp", context_length=200_000, conversation_id=f"busy-bar-{owner}"
    )


def sequence(cell: str, engine: Any, restart: Callable[[], Any],
             operation: Callable[..., Any]) -> Any:
    """Only the shared public host API; restart closes and creates an engine."""
    view = turn(1)

    def ingest() -> None:
        operation(engine, "A", "ingest", lambda: engine.ingest(deepcopy(view)), len(view))

    if cell == "ingest":
        ingest()  # Seed prefix, then bind as the host does on a resumed session.
        start(engine)
        view.append(turn(2)[0])
        ingest()
        ingest()
    elif cell == "compress":
        view = []
        for number in range(1, TURNS + 1):
            view.extend(turn(number))
            ingest()
        view.append(turn(TURNS + 1)[0])
        ingest()
        compressed = operation(engine, "A", "compress",
                         lambda: engine.compress(deepcopy(view), force=True), len(view))
        if compressed is not None:
            view = compressed
        engine = restart()
        ingest()
    elif cell == "session-end":
        ingest()
        view.extend(turn(2))
        operation(engine, "A", "session-end",
                  lambda: engine.on_session_end("A", deepcopy(view)), len(view))
        start(engine)
        ingest()
    elif cell == "restart":
        ingest()
        engine = restart()
        view.extend(turn(2))
        ingest()
    else:
        raise ValueError(f"unknown cell: {cell}")
    return engine
