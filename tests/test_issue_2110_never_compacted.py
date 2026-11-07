"""#2110 with no proof at all: a session that was never compacted has no v4 proof, so cursor reconciliation runs the
no-proof branch, which gives identity None to a user row that passes the verified-scaffold test (one that starts
with the preserved-objective or preserved-task-list prefix). Does a real user turn of that shape get stored? Cells:
warm and after a restart, at the tail and before the last stored row. Controls: a plain user turn in every cell. On
main the scaffold-shaped turn is lost in every cell except warm at the tail (strict xfails)."""
from __future__ import annotations

import pytest

from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX, _PRESERVED_TODO_CONTEXT_PREFIX
from tests.test_compression_boundary import _config, _turn

_SHAPES = {"objective-prefix": f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\nKeep this objective exactly as I wrote it.",
           "todo-prefix": f"{_PRESERVED_TODO_CONTEXT_PREFIX}\n- [ ] write the report",
           "plain": "A plain user turn with no scaffold shape."}
_LOST = pytest.mark.xfail(strict=True, raises=AssertionError,
                          reason="#2110: with no proof in force (never compacted) a scaffold-shaped user turn gets "
                                 "identity None and is skipped")
_CELLS = [pytest.param(shape, place, restart, id=f"{shape}-{place}-{'restart' if restart else 'warm'}",
                       marks=_LOST if shape != "plain" and (restart or place == "before-last-row") else ())
          for shape in _SHAPES for place in ("after-tail", "before-last-row") for restart in (False, True)]


def _engine(tmp_path):
    engine = LCMEngine(config=_config(tmp_path, fresh_tail_count=0), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    return engine


@pytest.mark.parametrize("shape,place,restart", _CELLS)
def test_a_scaffold_shaped_user_turn_is_stored_in_a_never_compacted_session(tmp_path, shape, place, restart):
    engine = _engine(tmp_path)
    host = [m for i in range(1, 4) for m in _turn(i)]
    engine.ingest([dict(m) for m in host])
    if restart:
        engine.shutdown()
        engine = _engine(tmp_path)
    try:
        if engine._active_emission_proof():
            pytest.fail("setup: a never-compacted session has a proof")
        turn = {"role": "user", "content": _SHAPES[shape]}
        if engine._is_verified_replay_scaffold_message(turn) != (shape != "plain"):
            pytest.fail("setup: the turn is not the shape under test")
        reply = {"role": "assistant", "content": "reply to the quoting turn"}
        view = [*host, turn, reply] if place == "after-tail" else [*host[:-1], turn, host[-1], reply]
        engine.ingest([dict(m) for m in view])
        rows = engine._store._conn.execute("SELECT role, content FROM messages").fetchall()
        if sum(c == reply["content"] for r, c in rows if r == "assistant") != 1:
            pytest.fail("setup: the batch was not ingested")
        assert sum(c == turn["content"] for r, c in rows if r == "user") == 1, "the user turn was skipped"
    finally:
        engine.shutdown()
