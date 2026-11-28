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
    source = [*view, view[1], view[2]]
    initial = engine._store.get_session_messages("S")
    _expect_stored_shape([row["content"] for row in initial], [row["content"] for row in source],
                         [row["content"] for row in source], "setup")
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    snapshots, sent, growth = [], [], []
    previous = len(initial)
    for bind in (1, 2):
        engine.shutdown()
        engine = _engine(tmp_path)
        view = [*view, {"role": "user", "content": f"new-{bind}"}]
        sent.append(deepcopy(view))
        engine.ingest(deepcopy(view))
        if engine._ingest_failure_count:
            pytest.fail(f"ingest failed: {engine._last_ingest_error}")
        rows = engine._store.get_session_messages("S")
        snapshots.extend((bind, row["role"], row["content"]) for row in rows)
        growth.append(len(rows) - previous)
        previous = len(rows)
    at = {row["store_id"]: index for index, row in enumerate(rows)}
    links = [(at.get(source), at.get(target)) for source, target in engine._store._conn.execute(
        "SELECT store_id, related_store_id FROM message_relations WHERE kind = 'possible_replay_of' "
        "ORDER BY relation_id")] if engine._store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'message_relations'").fetchone() else []
    engine.shutdown()
    fixed = [(bind, row["role"], row["content"]) for bind in (1, 2) for row in [*source, *sent[bind - 1][8:]]]
    known = [(bind, row["role"], row["content"]) for bind in (1, 2)
             for row in [*source, *(row for view in sent[:bind] for row in view)]]
    # Main links each re-stored view to unused earlier copies, consuming the initial copy set on bind two.
    n, start = len(sent[0]) - 1, len(source)
    known_links = [*((start + i, i) for i in range(n)), (start + n + 1, start),
                   (start + n + 2, n), (start + n + 3, n + 1),
                   *((start + n + 4 + i, start + 3 + i) for i in range(n - 2))] if ledger == "shadow" else []
    diagnostic = dict(growth=growth, stored=[row["content"] for row in rows], links=links)
    _expect_stored_shape([*snapshots, *(("link", *link) for link in links)], fixed,
                         [*known, *(("link", *link) for link in known_links)], f"#588 known shape: {diagnostic}")


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
