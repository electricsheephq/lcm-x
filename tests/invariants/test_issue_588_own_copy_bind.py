"""#588 (P1 half): a fresh bind maps each replayed row to its own stored copy, never to a neighbour's later copy set.

The issue's shape: a session whose store holds every row once, followed by a later, incomplete set of byte-exact
copies of some of them (left by an older bind). A fresh bind that replays the host's history must store only the
genuinely new row, and every later bind must store nothing again (no growth per restart).
"""

import contextlib
from copy import deepcopy

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _m(role, content, stamp):
    return {"role": role, "content": content, "timestamp": stamp}


def _engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), temporal_rollups_enabled=False, embeddings_enabled=False,
        deferred_maintenance_enabled=False))
    engine.on_session_start("S", platform="cli", conversation_id="conv", context_length=200_000)
    return engine


def _history(n=8):
    return [_m("user" if i % 2 == 0 else "assistant", f"row-{i}", float(i + 1)) for i in range(n)]


@pytest.mark.parametrize("copies", [(1, 2), (2, 3, 4), (5,)], ids=["pair", "triple", "single"])
@pytest.mark.parametrize("ledger", ["shadow", "off"])
def test_fresh_bind_maps_rows_to_their_own_copies(tmp_path, monkeypatch, copies, ledger):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    engine = _engine(tmp_path)
    view = _history()
    engine.ingest(deepcopy(view))
    # The later incomplete copy set: byte-exact copies of a few rows, appended after the session's own rows.
    # ``ingest_atomic`` owns the write on the ledger-era store; older stores append directly.
    with getattr(engine._store, "ingest_atomic", contextlib.nullcontext)():
        for i in copies:
            engine._store.append("S", deepcopy(view[i]), conversation_id="conv")
    stored = len(engine._store.get_session_messages("S"))
    assert stored == len(view) + len(copies)
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    for bind, new in enumerate((_m("user", "new-1", 100.), _m("assistant", "new-2", 101.)), 1):
        engine.shutdown()
        engine = _engine(tmp_path)  # a fresh process binds the existing session
        view = [*view, new]
        engine.ingest(deepcopy(view))
        assert engine._ingest_failure_count == 0, engine._last_ingest_error
        rows = engine._store.get_session_messages("S")
        assert len(rows) == stored + bind, [(r["store_id"], r["content"]) for r in rows]
        assert rows[-1]["content"] == new["content"]
        if ledger == "shadow" and bind == 1:
            # The rebind (design §3.6) binds each pre-ledger row only to its own exact copy: never to a row of the
            # later copy set, and not at all where byte-exact twins make the copy ambiguous.
            own = {i + 1 for i in range(len(view) - 1) if i not in copies}
            migrated = {store_id for (store_id,) in engine._store._conn.execute(
                "SELECT store_id FROM occurrence_bindings WHERE disposition='migrated'")}
            assert migrated == own, (sorted(migrated), sorted(own))
    engine.shutdown()
