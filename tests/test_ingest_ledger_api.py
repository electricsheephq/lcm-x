"""#2055 P1b-2a: pure API and transaction-owned ledger fixtures."""

import random
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from hermes_lcm import ingest_ledger as ledger
from hermes_lcm.engine import LCMEngine
from hermes_lcm.host_uid import HostUidShadowMixin
from hermes_lcm.reconcile import ReconcileMixin, _MODEL_SWITCH_NOTIFICATION_PREFIX, _PRESERVED_TODO_CONTEXT_PREFIX
from hermes_lcm.store import MessageStore, ledger_digest
from hermes_lcm.sqlite_session import TransactionOwner
from hermes_lcm.survival_fit import SurvivalFitMixin

SALT = b"s" * 32
DIGEST = b"d" * 16


@pytest.fixture
def store(tmp_path):
    instance = MessageStore(tmp_path / "ledger.db")
    instance.ensure_ingest_ledger()
    try:
        yield instance
    finally:
        instance.close()


def binding(ordinal, **kwargs):
    return dict(ordinal=ordinal, identity_digest=DIGEST, disposition="stored", view_seq=0, **kwargs)


def test_digests_and_aliases(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("ledger identity evaluated a replay alias")
    for owner, name in ((ReconcileMixin, "_message_replay_identity"),
                        (SurvivalFitMixin, "_survival_projection_source"),
                        (ReconcileMixin, "_pruned_tool_output_identity"),
                        (ReconcileMixin, "_host_rewrite_override_content"),
                        (LCMEngine, "_generated_context_carrier_remainder")):
        monkeypatch.setattr(owner, name, forbidden)
    message = dict(role="tool", content="raw", tool_call_id="id", tool_calls=[{"id": "a"}], tool_name="read")
    expected = ledger_digest(SALT, "tool", "raw", "id", '[{"id":"a"}]', "read")
    assert ledger.identity_digest(SALT, message) == expected
    for field, value in (("role", "user"), ("content", "changed"), ("tool_call_id", "other"),
                         ("tool_calls", [{"id": "b"}]), ("tool_name", "write")):
        assert ledger.identity_digest(SALT, {**message, field: value}) != expected
    assert ledger.identity_digest(SALT, {**message, "timestamp": 123, "store_id": 9}) == expected
    assert ledger.identity_digest(SALT, {**message, "content": "carrier\nraw"}) != expected
    assert ledger.identity_digest(SALT, {**message, "content": "[Old tool output cleared to save context space]"}) != expected
    def calls(arguments):
        return {**message, "tool_calls": [{"function": {"arguments": arguments}}]}
    assert ledger.identity_digest(SALT, calls('{"b":2,"a":1}')) == ledger.identity_digest(SALT, calls('{"a":1, "b":2}'))
    assert ledger.identity_digest(SALT, {"role": "user", "content": {"b": 2, "a": 1}}) == ledger_digest(
        SALT, "user", '{"a": 1, "b": 2}', "", "", "")
    assert ledger_digest(SALT, "ab", "c") != ledger_digest(SALT, "a", "bc")
    assert len({ledger_digest(SALT, value) for value in (None, b"1", "1", 1)}) == 4
    assert ledger.stamp_digest(SALT, None) is None
    assert ledger.stamp_digest(SALT, "") != ledger.stamp_digest(SALT, None)
    assert ledger.uidset_digest(SALT, None, ["a"]) is None
    assert ledger.uidset_digest(SALT, "u", ["b", "a", "a"]) == ledger.uidset_digest(SALT, "u", ["a", "b"])
    assert ledger.uidset_digest(SALT, "u", ["a"]) != ledger.uidset_digest(SALT, "u", ["b"])
    for uid, absorbed in ((1, []), (False, []), ("u", [1])):
        with pytest.raises(ledger.LedgerError):
            ledger.uidset_digest(SALT, uid, absorbed)


def test_identity_canonical_scaffolding():
    def digest(content):
        return ledger.identity_digest(SALT, {"role": "user", "content": content})
    note = _MODEL_SWITCH_NOTIFICATION_PREFIX + "A to B]\n\n"
    todo = "\n\n" + _PRESERVED_TODO_CONTEXT_PREFIX + "\n- [>] 1. Task (in_progress)"
    assert digest(note + "real") == digest("real" + todo) == digest("real")
    assert digest("real" + todo + "\n\nNEW") == digest("real\n\nNEW") != digest("real")
    assert digest("different") != digest("real")


def test_vectors_and_prefixes():
    entries = [ledger.ViewEntry(DIGEST, None, bytes(16), -1),
               ledger.ViewEntry(b"e" * 16, b"s" * 16, None, 7, True)]
    blob = ledger.encode_entries(entries)
    assert len(blob) == 1 + 57 * 2
    assert blob[0] == 1 and blob[17] == 2 and blob[58 + 16] == 5
    assert blob[50:58] == (-1).to_bytes(8, "big", signed=True)
    assert ledger.decode_entries(blob) == entries
    assert ledger.decode_entries(ledger.encode_entries([])) == []
    for bad in (b"", b"\x02", blob[:-1], blob[:17] + b"\x80" + blob[18:]):
        with pytest.raises(ledger.LedgerFormatError):
            ledger.decode_entries(bad)
    chain = ledger.prefix_chain(SALT, entries)
    assert chain[:16] == ledger_digest(SALT, "chain", DIGEST, b"", bytes(16))
    changed = ledger.prefix_chain(SALT, [entries[0], entries[1]._replace(identity_digest=DIGEST)])
    assert chain[:16] == changed[:16] and chain[16:] != changed[16:]
    assert ledger.common_prefix_len(chain, changed) == 1
    assert ledger.common_prefix_len(chain, chain) == 2
    assert ledger.common_prefix_len(chain, ledger.prefix_chain(SALT, entries[::-1])) == 0
    assert ledger.common_prefix_len(b"", chain) == 0
    assert ledger.decode_entries(ledger.encode_entries([(DIGEST, None, None, None)])) == [
        ledger.ViewEntry(DIGEST, None, None, None)]
    assert ledger.decode_entries(ledger.encode_entries([(DIGEST, None, None, 4, "yes")]))[0].is_descriptor is True
    unbound = ledger.encode_entries([(DIGEST, None, None, None)])
    assert unbound[17] == 8 and unbound[-8:] == bytes(8)
    for bad in (unbound[:17] + b"\x0c" + unbound[18:], unbound[:-1] + b"\x01",
                unbound[:17] + b"\x10" + unbound[18:]):
        with pytest.raises(ledger.LedgerFormatError):
            ledger.decode_entries(bad)
    with pytest.raises(ledger.LedgerFormatError):
        ledger.encode_entries([(DIGEST, None, None, None, True)])


def test_ordinal_examples():
    step, place = ledger.ORDINAL_STEP, ledger.Placement
    assert ledger.plan_ordinals([], [place(None, None)] * 2) == ([step, 2 * step], {})
    assert ledger.plan_ordinals([step], [place(step, None)] * 2) == ([2 * step, 3 * step], {})
    assert ledger.plan_ordinals([100, 200, 300], [place(100, 300)] * 2) == ([233, 266], {})
    assert ledger.plan_ordinals([100, 110], [place(100, 110)] * 3) == ([102, 105, 107], {})
    assert ledger.plan_ordinals([100, 200, 300], [place(None, None, 100)]) == ([150], {})
    values, mapping = ledger.plan_ordinals([1024, 1025, 8192], [place(1024, 1025)] * 2)
    assert len(mapping) == 1 and 1025 in mapping
    assert 1024 < values[0] < values[1] < mapping[1025] < 8192
    values, mapping = ledger.plan_ordinals([1, 2, 3], [place(1, 2), place(2, 3)])
    assert mapping.get(1, 1) < values[0] < mapping.get(2, 2) < values[1] < mapping.get(3, 3)
    assert ledger.plan_ordinals([100, 200, 250, 300], [place(100, 300), place(100, 300, 200)]) == ([266, 283], {})
    assert ledger.plan_ordinals([100, 200, 300], [place(100, None), place(100, None, 200)]) == (
        [300 + step, 300 + 2 * step], {})
    for placements in ([place(200, 300), place(100, 300)], [place(100, None), place(100, 300)]):
        with pytest.raises(ledger.LedgerError):
            ledger.plan_ordinals([100, 200, 300], placements)


@pytest.mark.parametrize("existing, placements, expected", [
    ([100, 200, 300, 400], [ledger.Placement(100, 200, 400), ledger.Placement(200, 300)], [150, 250]),
    ([100, 200, 300], [ledger.Placement(200, 300, 100)], [250]),
])
def test_supersedes_outside_view_bounds(existing, placements, expected):
    assert ledger.plan_ordinals(existing, placements) == (expected, {})


def test_seeded_ordinal_sequences_and_minimal_windows():
    rng = random.Random(2055)
    for _ in range(300):
        existing = sorted(rng.sample(range(1, 20), 5)) + [1 << 20]
        for _ in range(8):
            placements, expected, cursor, last_left = [], [("old", value) for value in existing], 0, 0
            for gap in sorted(rng.sample(range(len(existing) + 1), rng.randint(1, 3))):
                left_index = rng.randint(last_left, max(last_left, gap - 1))
                left, right = existing[left_index - 1] if left_index else None, existing[gap] if gap < len(existing) else None
                last_left = left_index
                for _ in range(rng.randint(1, 3)):
                    target = rng.choice(existing) if rng.randrange(2) else None
                    placements.append(ledger.Placement(left, right, target))
                    target_in_bounds = target is not None and (left is None or left <= target) and (right is None or target < right)
                    predecessor = target if target_in_bounds else existing[gap - 1] if gap else None
                    position = expected.index(("old", predecessor)) + 1 if predecessor is not None else 0
                    position = max(position, cursor)
                    expected.insert(position, ("new", len(placements) - 1))
                    cursor = position + 1
            values, mapping = ledger.plan_ordinals(existing, placements)
            assert all(a < b for a, b in zip(values, values[1:]))
            for placement, value in zip(placements, values):
                assert (mapping.get(placement.left, placement.left) if placement.left is not None else 0) < value
                assert placement.right is None or value < mapping.get(placement.right, placement.right)
            moved = [mapping.get(value, value) for value in existing]
            actual = sorted([(("old", old), new) for old, new in zip(existing, moved)]
                            + [(("new", index), value) for index, value in enumerate(values)], key=lambda item: item[1])
            assert [token for token, value in actual] == expected
            merged = [value for token, value in actual]
            assert merged == sorted(set(merged))
            counts, gap = {}, 0
            for kind, value in expected:
                if kind == "old":
                    gap += 1
                else:
                    counts[gap] = counts.get(gap, 0) + 1
            if not mapping:
                assert all(g == len(existing) or existing[g] - (existing[g - 1] if g else 0) >= k + 1
                           for g, k in counts.items())
            for index, old in enumerate(existing):
                if old in mapping:
                    low = moved[index - 1] if index else 0
                    high = moved[index + 1] if index + 1 < len(moved) else None
                    span = [low] + [v for v in merged if v > low and (high is None or v <= high)]
                    assert all(b - a >= ledger.MIN_GAP for a, b in zip(span, span[1:]))
            if mapping and len(counts) == 1:
                gap, count = next(iter(counts.items()))
                # Exhaust all contiguous windows; the first feasible WIDTH is
                # the independent capacity lower bound on moved neighbours.
                candidates = []
                for start in range(gap + 1):
                    for end in range(gap, len(existing) + 1):
                        low = existing[start - 1] if start else 0
                        high = existing[end] if end < len(existing) else None
                        slots = end - start + count + 1
                        if high is None or high - low >= slots * ledger.MIN_GAP:
                            candidates.append((end - start, start, end))
                width = min(item[0] for item in candidates)
                windows = [(start, end) for size, start, end in candidates if size == width
                           and set(mapping) <= set(existing[start:end])]
                assert windows, "a smaller feasible re-spacing window exists"
                assert any(all(b - a >= ledger.MIN_GAP for a, b in zip(
                    [existing[start - 1] if start else 0] + merged[start:end + count],
                    merged[start:end + count] + ([existing[end]] if end < len(existing) else [])))
                           for start, end in windows)
            existing = merged


def test_lineage_keys(tmp_path):
    host = SimpleNamespace(_session_id="session", _state_db_path=lambda: tmp_path / "state.db",
                           _host_uid_read_lineage=lambda path, session: ("root", None))
    resolved, error = HostUidShadowMixin._host_uid_lineage_key(host)
    assert error is None and re.fullmatch(r"[0-9a-f]{16}:root", resolved)
    assert ledger.lineage_key(resolved, "session") == resolved
    assert ledger.lineage_key(None, "session") == "provisional:session"
    assert ledger.lineage_key(None, resolved).split(":")[0] != resolved.split(":")[0]
    with pytest.raises(ledger.LedgerError):
        ledger.lineage_key("provisional:root", "session")


def test_owned_writers_and_readers(store):
    conn = store.connection
    with store.atomic():  # Ownership works even before SQLite's first write.
        ids = ledger.insert_bindings(conn, "L", [binding(1), binding(2), binding(3)])
        assert ledger.apply_respace(conn, "L", {1: 2, 2: 3, 3: 4}) == 3
        assert ledger.lineage_ordinals(conn, "L") == [2, 3, 4]
        assert [row["occ_id"] for row in ledger.bindings_by_digest(conn, "L", DIGEST)] == ids
        for kind in ("ingest", "emission"):
            for _ in range(3):
                seq = ledger.write_view(conn, SALT, "L", kind, "proposed", [ledger.ViewEntry(DIGEST, None, None, ids[0])])
            assert ledger.newest_view(conn, "L", kind)["view_seq"] == seq
            assert conn.execute("SELECT count(*) FROM ingest_views WHERE lineage='L' AND kind=?", (kind,)).fetchone()[0] == 2
        assert ledger.write_view(conn, SALT, "other", "ingest", None, []) == 7
        for occ_id in ids:
            for number in range(4):
                assert ledger.add_stamp_observation(conn, "L", occ_id, str(number), "alt_stamp", 7) == ledger.STAMP_WRITTEN
            assert ledger.add_stamp_observation(conn, "L", occ_id, "4", "alt_stamp", 7) == ledger.STAMP_CAPPED
            assert ledger.add_stamp_observation(conn, "L", occ_id, "0", "alt_stamp", 7) == ledger.STAMP_EXISTS
            if occ_id != ids[-1]:
                assert ledger.marks(conn, "L") is None
        assert ledger.marks(conn, "L")["stamp_policy"] == "stamp_unreliable"
        assert ledger.add_stamp_observation(conn, "L", ids[0], "carried", "carried_stamp", 8, source="legacy") == ledger.STAMP_WRITTEN
        for writer, value in ((ledger.update_store_mark, 9), (ledger.update_leaf_mark, 10),
                              (ledger.update_legacy_proof_digest, DIGEST)):
            assert writer(conn, "L", value)
            before = conn.total_changes
            assert not writer(conn, "L", value)
            assert conn.total_changes == before
        conn.execute("DELETE FROM occurrence_bindings WHERE occ_id=?", (ids[-1],))
        assert ledger.insert_bindings(conn, "L", [binding(5)])[0] > ids[-1]
    assert not conn.in_transaction


def test_copied_context_cannot_write_from_worker_and_missing_stamp_reads_nothing(store):
    conn, statements = store.connection, []
    with store.atomic():
        with ThreadPoolExecutor(max_workers=1) as worker, pytest.raises(ledger.LedgerTransactionError):
            worker.submit(copy_context().run, ledger.next_view_seq, conn).result()
        conn.set_trace_callback(statements.append)
        with pytest.raises(ledger.LedgerError):
            ledger.add_stamp_observation(conn, "L", 1, None, "alt_stamp", 0)
        conn.set_trace_callback(None)
        assert not statements


def test_owner_cleanup_when_context_token_cannot_reset(store):
    token = TransactionOwner._active_owners.set(())
    try:
        scope = store.atomic()
        scope.__enter__()
        copy_context().run(scope.__exit__, None, None, None)
        assert store._transaction_owner._active_connection is None
        assert store._transaction_owner._owner_thread is None
        with pytest.raises(ledger.LedgerTransactionError):
            ledger.next_view_seq(store.connection)
    finally:
        TransactionOwner._active_owners.reset(token)


@pytest.mark.parametrize("raw_transaction", [False, True])
def test_all_writers_require_owner(store, raw_transaction):
    conn = store.connection
    if raw_transaction:
        conn.execute("BEGIN IMMEDIATE")
    writers = [lambda: ledger.next_view_seq(conn), lambda: ledger.insert_bindings(conn, "L", []),
               lambda: ledger.apply_respace(conn, "L", {}), lambda: ledger.write_view(conn, SALT, "L", "ingest", None, []),
               lambda: ledger.add_stamp_observation(conn, "L", 1, "s", "alt_stamp", 0),
               lambda: ledger.update_store_mark(conn, "L", 1), lambda: ledger.update_leaf_mark(conn, "L", 1),
               lambda: ledger.update_legacy_proof_digest(conn, "L", DIGEST)]
    try:
        for writer in writers:
            with pytest.raises(ledger.LedgerTransactionError):
                writer()
    finally:
        conn.rollback()


def test_writer_failure_rolls_back_with_owner(store):
    conn = store.connection
    with pytest.raises(sqlite3.IntegrityError), store.atomic():
        ledger.next_view_seq(conn)
        ledger.insert_bindings(conn, "L", [binding(1), binding(1)])
    assert not ledger.bindings_by_digest(conn, "L", DIGEST)
    assert conn.execute("SELECT value FROM ledger_meta WHERE key='view_seq'").fetchone()[0] == 0
    with pytest.raises(RuntimeError), store.atomic():
        ledger.write_view(conn, SALT, "L", "ingest", None, [])
        ledger.update_store_mark(conn, "L", 2)
        raise RuntimeError("outer failure")
    assert ledger.newest_view(conn, "L", "ingest") is None and ledger.marks(conn, "L") is None
