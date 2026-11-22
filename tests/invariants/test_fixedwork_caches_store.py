"""#2856 PR2 C4: the store-id map's candidate rows, remembered per store write generation, always equal a fresh read.

Random sessions with own-connection writes, a SECOND connection's write between two builds (the next build must see
it), a transaction that writes then rolls back (no fill inside it; the next build outside equals a fresh one), metadata
writes on either connection, frontier moves, carry-range changes and rebinds. The oracle is a build with the memo
disabled; after every step a held slot the next build could use is also compared with an independent fresh read.
Synthetic data only.
"""
import gc
import json
import random
import weakref
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest

import hermes_lcm.reconcile as reconcile
from hermes_lcm.store import MessageStore
from tests.harness import engine_context

QUIET = dict(embeddings_enabled=False, temporal_rollups_enabled=False)
WORD = " lorem ipsum dolor"
C_SLOT = "_store_id_map_candidate_rows_memo"


@contextmanager
def _no_memo():
    def fresh(owner, slot, key, load, copy):
        return load()
    with mock.patch.object(reconcile, "_store_generation_memo", fresh):
        yield


def _page(store, session, frontier, ranges):
    """An independent fresh read of what the map pages in (the oracle for a held C slot)."""
    rows, after = [], frontier
    while True:
        page = store.get_session_messages_after(session, after_store_id=after)
        if not page:
            break
        rows += page
        after = page[-1]["store_id"]
    for source, start, end in ranges:
        nxt = max(int(frontier or 0), start) + 1
        while nxt <= end:
            page = store.get_range(source, start_id=nxt, end_id=end)
            if not page:
                break
            rows += page
            nxt = int(page[-1]["store_id"]) + 1
    return rows


def _usable(engine, slot):
    """The held slot and its key when the next read would use it (same store, connection and generation)."""
    held, store = engine.__dict__.get(slot), engine._store
    if held is None or held[0] is not store or held[1] is not store._conn or store._conn.in_transaction:
        return None
    generation, key = held[2]
    return (key, held[3]) if generation == store.write_generation() else None


def _check_slots(engine):
    store = engine._store
    if (c := _usable(engine, C_SLOT)) is not None:
        session, _conversation, frontier, ranges = c[0]
        assert c[1] == _page(store, session, frontier, ranges)


def _row(rng, i, role=None):
    role = role or rng.choice(["user", "assistant", "assistant", "tool"])
    row = {"role": role, "content": f"[w{i}] {role}" + WORD * rng.randrange(1, 40)}
    if role == "tool":
        row.update(tool_call_id=f"call_{i}", name="read_file")
    if role == "assistant" and rng.random() < 0.3:
        row["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                              "function": {"name": "read_file", "arguments": json.dumps({"n": i, "x": [1, {"y": 2}]})}}]
    return row


def _compare_all(engine, view):
    built = engine._get_store_id_map_for_messages(view)
    again = engine._get_store_id_map_for_messages(view)
    with _no_memo():
        assert built == again == engine._get_store_id_map_for_messages(view)
    _check_slots(engine)
    return built


@pytest.mark.parametrize("seed", range(12))
def test_c4_remembered_candidate_rows_equal_fresh_reads(tmp_path, seed):
    rng = random.Random(seed)
    hits, original = {C_SLOT: [0, 0]}, reconcile._store_generation_memo

    def counted(owner, slot, key, load, copy):  # [calls, loads] per slot: a call that did not load was a hit
        ran = []
        hits[slot][0] += 1
        out = original(owner, slot, key, lambda: ran.append(1) or load(), copy)
        hits[slot][1] += bool(ran)
        return out
    with engine_context(tmp_path, window=200_000, conversation_id="conv", **QUIET) as engine, \
            mock.patch.object(reconcile, "_store_generation_memo", counted):
        store, i = engine._store, 0
        other = MessageStore(store.db_path, hermes_home=str(tmp_path))
        view = []
        for _ in range(12):
            view.append(_row(rng, i))
            i += 1
        engine.ingest(deepcopy(view))
        for _ in range(8):
            store.append("P", _row(rng, i), conversation_id="conv")
            i += 1
        p_ids = [r["store_id"] for r in store.get_session_messages_after("P", after_store_id=0)]
        try:
            for step in range(36):
                op = rng.choice(["ingest", "own", "second", "rollback", "frontier", "carry", "rebind", "meta_own",
                                 "meta_second", "none"])
                if op == "ingest":
                    view.append(_row(rng, i))
                    i += 1
                    engine.ingest(deepcopy(view))
                elif op == "own":
                    row = _row(rng, i)
                    i += 1
                    store.append(engine._session_id, row, conversation_id="conv")
                    view.append(row)
                elif op == "second":  # another connection's commit between two reads must be seen by the next
                    row = _row(rng, i)
                    i += 1
                    engine._get_store_id_map_for_messages(view)
                    new_id = other.append(engine._session_id, row, conversation_id="conv")
                    view.append(row)
                    engine._get_store_id_map_for_messages(view)
                    held = _usable(engine, C_SLOT)
                    if held is not None and held[0][0] == engine._session_id and new_id > int(held[0][2] or 0):
                        assert new_id in [r["store_id"] for r in held[1]]
                elif op == "rollback":  # a write inside a transaction that rolls back is never remembered
                    conn = store._conn
                    _compare_all(engine, view)
                    before = engine.__dict__.get(C_SLOT)
                    conn.execute("BEGIN")
                    rid = conn.execute("INSERT INTO messages(session_id, role, content, conversation_id, timestamp) "
                                       "VALUES (?, 'user', 'rolled back', 'conv', 1.0)", (engine._session_id,)).lastrowid
                    _compare_all(engine, view)
                    assert engine.__dict__.get(C_SLOT) is before  # no fill inside
                    conn.execute("ROLLBACK")
                    _compare_all(engine, view)
                    held = _usable(engine, C_SLOT)
                    assert held is None or rid not in [r["store_id"] for r in held[1]]
                elif op == "frontier":
                    ids = [r["store_id"] for r in store.get_session_messages_after(engine._session_id, after_store_id=0)]
                    engine._last_compacted_store_id = rng.choice([0, *ids]) if ids else 0
                elif op == "carry":
                    if rng.random() < 0.3:
                        engine.__dict__.pop("_load_compression_carry_ranges", None)
                    else:
                        start = rng.choice(p_ids[:-1]) - 1
                        ranges = [("P", start, rng.choice([x for x in p_ids if x > start]))]
                        engine._load_compression_carry_ranges = lambda *a, ranges=ranges, **k: list(ranges)
                elif op == "rebind":
                    engine.on_session_start(rng.choice(["S", "S", "S2"]), platform="cli", context_length=200_000,
                                            conversation_id="conv")
                elif op == "meta_own":  # moves the generation, not the rows: the next build reloads equal rows
                    store.write_metadata_json([f"pr2_probe:{rng.randrange(4)}"], json.dumps(rng.randrange(99)))
                elif op == "meta_second":
                    other.write_metadata_json([f"pr2_probe:{rng.randrange(4)}"], json.dumps(rng.randrange(99)))
                _compare_all(engine, view)
        finally:
            other.close()
    # Not vacuous: the slot served some builds and filled on others.
    assert all(calls > loads > 0 for calls, loads in hits.values()), hits


def test_c4_reopen_releases_old_store_before_paging(tmp_path):
    db_path = tmp_path / "lcm.db"
    owner = SimpleNamespace(_store=MessageStore(db_path))
    try:
        owner._store.append("S", {"role": "user", "content": "row before reopen"})
        expected = reconcile._store_generation_memo(
            owner, C_SLOT, "S", lambda: _page(owner._store, "S", 0, ()), reconcile._copy_store_rows,
        )
        assert len(expected) == 1
        old_store = weakref.ref(owner._store)
        owner._store.close()
        owner._store = MessageStore(db_path)
        assert old_store() is not None  # the old slot is still held at the start of the miss
        released_at_page = []

        def observe_page():
            gc.collect()
            released_at_page.append(old_store() is None)
            return _page(owner._store, "S", 0, ())

        assert reconcile._store_generation_memo(
            owner, C_SLOT, "S", observe_page, reconcile._copy_store_rows,
        ) == expected
        assert released_at_page == [True]
        assert old_store() is None
    finally:
        owner._store.close()


@pytest.mark.parametrize(("limit", "cap"), [
    ("_STORE_GENERATION_MEMO_MAX_ROWS", 2),
    ("_STORE_GENERATION_MEMO_MAX_BYTES", 4096),
], ids=["row-count", "accounted-bytes"])
def test_c4_oversize_candidate_rows_page_again_with_equal_results(tmp_path, monkeypatch, limit, cap):
    monkeypatch.setattr(reconcile, limit, cap, raising=False)
    view = [
        {"role": "user", "content": "row one"},
        {"role": "assistant", "content": "row two", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {
                "name": "test", "arguments": json.dumps({"input": "x" * 8192}),
            }},
        ]},
        {"role": "tool", "content": "row three", "tool_call_id": "call_1", "name": "test"},
    ]
    with engine_context(tmp_path, window=200_000, conversation_id="conv", **QUIET) as engine:
        store = engine._store
        for row in view:
            store.append(engine._session_id, row, conversation_id="conv")
        expected_rows = _page(store, engine._session_id, engine._last_compacted_store_id, ())
        assert len(expected_rows) == len(view)
        with _no_memo():
            expected_map = engine._get_store_id_map_for_messages(view)
        returned_rows = []
        memo = reconcile._store_generation_memo

        def capture_rows(owner, slot, key, load, copy):
            rows = memo(owner, slot, key, load, copy)
            returned_rows.append(rows)
            return rows

        with mock.patch.object(reconcile, "_store_generation_memo", capture_rows), mock.patch.object(
            engine, "_page_store_id_map_candidates", wraps=engine._page_store_id_map_candidates,
        ) as page:
            assert engine._get_store_id_map_for_messages(view) == expected_map
            assert engine._get_store_id_map_for_messages(view) == expected_map

        assert returned_rows == [expected_rows, expected_rows]
        assert page.call_count == 2
        assert C_SLOT not in engine.__dict__
