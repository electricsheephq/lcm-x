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
_REASON = ("with no proof in force (never compacted) a scaffold-shaped user turn gets identity None and is "
           "skipped when it is reconciled against stored rows")
_LOST = {"objective-prefix": pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"#2110: {_REASON}"),
         "todo-prefix": pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"#2418 and #2110: {_REASON}")}
_CELLS = [pytest.param(shape, place, restart, id=f"{shape}-{place}-{'restart' if restart else 'warm'}",
                       marks=_LOST[shape] if shape != "plain" and (restart or place == "before-last-row") else ())
          for shape in _SHAPES for place in ("after-tail", "before-last-row") for restart in (False, True)]


def _engine(tmp_path):
    engine = LCMEngine(config=_config(tmp_path, fresh_tail_count=0), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    return engine


def _expect_stored_once(stored, fixed, known=None):
    """Only the complete known loss may satisfy the named-limit xfail."""
    if stored == fixed:
        return
    if known is not None and stored == known:
        raise AssertionError("#2110 known shape: the scaffold-shaped user turn is skipped (stored 0 times)")
    first_difference = next((i for i, (a, b) in enumerate(zip(stored, fixed)) if a != b),
                            min(len(stored), len(fixed)))
    pytest.fail(f"not the #2110 shape: stored rows={len(stored)}, fixed rows={len(fixed)}, "
                f"known rows={len(known) if known is not None else 0}, first difference={first_difference}")


@pytest.mark.parametrize("order,outcome", [
    pytest.param(["head", "last", "reply"], AssertionError, id="known"),
    pytest.param(["head", "turn", "last", "reply"], None, id="fixed"),
    pytest.param(["head", "turn", "last", "last", "reply"], pytest.fail.Exception, id="last-twice"),
    pytest.param(["head", "last", "last", "reply"], pytest.fail.Exception, id="lost-and-last-twice"),
    pytest.param(["head", "turn", "reply", "last"], pytest.fail.Exception, id="reply-before-last"),
    pytest.param(["head", "reply", "last"], pytest.fail.Exception, id="lost-and-reply-before-last"),
])
def test_stored_once_accepts_only_the_known_or_fixed_list(order, outcome):
    rows = {"head": ("user", "head"), "turn": ("user", "turn"),
            "last": ("assistant", "last"), "reply": ("assistant", "reply")}
    stored = [rows[name] for name in order]
    fixed = [rows[name] for name in ("head", "turn", "last", "reply")]
    known = [rows[name] for name in ("head", "last", "reply")]
    if outcome is None:
        _expect_stored_once(stored, fixed, known)
    else:
        with pytest.raises(outcome):
            _expect_stored_once(stored, fixed, known)


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
        additions = [turn, reply]
        # Before-last: warm reconciliation replays the last row; the cold plain control replays the whole view.
        if place == "before-last-row":
            if restart and shape == "plain":
                additions = view
            elif not restart:
                additions = view[-3:]
        prefix = [(row["role"], row["content"]) for row in host]
        fixed = prefix + [(row["role"], row["content"]) for row in additions]
        known = (prefix + [(row["role"], row["content"]) for row in additions if row is not turn]
                 if shape != "plain" and (restart or place == "before-last-row") else None)
        engine.ingest([dict(m) for m in view])
        rows = engine._store._conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? AND role != 'system' ORDER BY store_id",
            (engine._session_id,)).fetchall()
        _expect_stored_once(rows, fixed, known)
    finally:
        engine.shutdown()
