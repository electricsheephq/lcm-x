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


# Main's shape: each fresh bind stores the replayed history again (growth, stored contents), and with the ledger on
# every copy carries a possible_replay_of link (stored row positions: copy -> original).
_ROWS = [f"row-{i}" for i in range(8)]
_STORED_588 = [*_ROWS, "row-1", "row-2", *_ROWS, "new-1", *_ROWS, "new-1", "new-2"]
KNOWN_588 = {
    "shadow": dict(growth=[9, 10], stored=_STORED_588, links=[
        *((10 + i, i) for i in range(8)), (19, 10), (20, 8), (21, 9), *((22 + i, 13 + i) for i in range(6))]),
    "off": dict(growth=[9, 10], stored=_STORED_588, links=[]),
}


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "#588: with unstamped rows a later copy set defeats the replay match, so every fresh bind stores the replayed "
    "history again (linked possible_replay_of when the ledger is on); the v0.30.0 named limit for unstamped copy "
    "sets (linked duplicates), fixed in v0.30.x"))
@pytest.mark.parametrize("ledger", ["shadow", "off"])
def test_unstamped_fresh_bind_over_a_copy_set_stores_only_new_rows(tmp_path, monkeypatch, ledger):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    engine = _engine(tmp_path)
    view = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"row-{i}"} for i in range(8)]
    engine.ingest(deepcopy(view))
    if engine._ingest_failure_count:  # a swallowed setup failure must not pass as the pin
        pytest.fail(f"setup ingest failed: {engine._last_ingest_error}")
    with getattr(engine._store, "ingest_atomic", contextlib.nullcontext)():
        for i in (1, 2):
            engine._store.append("S", deepcopy(view[i]), conversation_id="conv")
    stored = len(engine._store.get_session_messages("S"))
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    growth = []
    for bind in (1, 2):
        engine.shutdown()
        engine = _engine(tmp_path)
        view = [*view, {"role": "user", "content": f"new-{bind}"}]
        engine.ingest(deepcopy(view))
        if engine._ingest_failure_count:  # a swallowed ingest failure must not pass as the pin
            pytest.fail(f"ingest failed: {engine._last_ingest_error}")
        rows = len(engine._store.get_session_messages("S"))
        if rows <= stored:  # a bind that stored nothing (bypass path) must not pass as the pin
            pytest.fail(f"bind {bind} stored no row")
        growth.append(rows - stored)
        stored = rows
    rows = engine._store.get_session_messages("S")
    at = {row["store_id"]: index for index, row in enumerate(rows)}
    links = [(at.get(source), at.get(target)) for source, target in engine._store._conn.execute(
        "SELECT store_id, related_store_id FROM message_relations WHERE kind = 'possible_replay_of' "
        "ORDER BY relation_id")] if engine._store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'message_relations'").fetchone() else []
    engine.shutdown()
    if growth == [1, 1]:
        return  # fixed: the strict marker turns this into a loud XPASS
    observed = dict(growth=growth, stored=[row["content"] for row in rows], links=links)
    if observed == KNOWN_588[ledger]:
        raise AssertionError(f"#588 known shape: {observed}")
    pytest.fail(f"not the #588 shape: {observed}")
