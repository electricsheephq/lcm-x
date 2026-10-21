"""#1748: filled history must not amplify discovery, or weaken gap repair."""
import json
import sqlite3
import time

import pytest

from hermes_lcm import command
from hermes_lcm.db_bootstrap import ensure_chunk_tables
from tests._issue_1748_base_gapfill import _chunk_gapfill_rows as base_gapfill
from tests.test_issue_1586_gap_discovery import get, publish, seed
from tests.test_issue_1586_local_pacing import local_store as local_store


def record_walk(conn, identity, state):
    # This is the same production post-selection seam, without provider work.
    writer = getattr(command, "_record_chunk_gapfill_walk", None)
    if writer:
        writer(conn, identity, state)


def discover(conn, identity, high, limit=2000):
    state = {}
    rows = list(command._chunk_gapfill_rows(
        conn, identity, (1, high), "conversational", limit,
        record_progress=True, gap_page_size=32, gap_scan_state=state,
    ))
    record_walk(conn, identity, state)
    return rows


class VisitedConnection(sqlite3.Connection):
    """Count the message predicates the actual discovery SQL evaluates."""
    def execute(self, sql, parameters=()):
        if getattr(self, "count_visits", False) and "FROM messages AS m " in sql:
            sql = sql.replace("AND NOT EXISTS", "AND visit(m.store_id) AND NOT EXISTS")
        return super().execute(sql, parameters)


def cost(conn, identity, high):
    steps = 0
    statements = []
    visits = 0

    def visit(_store_id):
        nonlocal visits
        visits += 1
        return 1

    def step():
        nonlocal steps
        steps += 1
        return 0

    conn.set_progress_handler(step, 1)
    conn.set_trace_callback(statements.append)
    conn.create_function("visit", 1, visit)
    conn.count_visits = True
    start = time.perf_counter()
    rows = discover(conn, identity, high)
    elapsed = time.perf_counter() - start
    conn.set_progress_handler(None, 0)
    conn.set_trace_callback(None)
    conn.count_visits = False
    return dict(sqlite_steps=steps, statements=len(statements),
                message_rows_visited=visits, selected_rows=len(rows), elapsed_s=elapsed)


def test_discovery_cost_independent_of_filled_prefix(tmp_path):
    """250k generated rows, real discovery/post-selection state, synthetic publication."""
    high = 250_000
    conn = sqlite3.connect(tmp_path / "cost.db", factory=VisitedConnection)
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT)")
    ensure_chunk_tables(conn)
    conn.executemany("INSERT INTO messages VALUES(?, 'user', 'A synthetic fact.')",
                     ((i,) for i in range(1, high + 1)))
    measurements = {}
    filled = 0
    for target in (0, 50_000, 240_000):
        while filled < target:
            rows = discover(conn, "synthetic", high, min(2000, target - filled))
            assert rows
            conn.executemany(
                "INSERT INTO lcm_chunk_meta(chunk_id, identity_hash, store_id) VALUES(?, 'synthetic', ?)",
                ((f"{r[0]}:0", r[0]) for r in rows),
            )
            filled += len(rows)
        measurements[str(target)] = cost(conn, "synthetic", high)
    conn.close()
    print("GAPWALK_COST " + json.dumps(measurements, sort_keys=True))
    assert measurements["240000"]["sqlite_steps"] <= 2 * measurements["0"]["sqlite_steps"], measurements
    assert measurements["240000"]["message_rows_visited"] <= 2 * measurements["0"]["message_rows_visited"]


def run(engine, limit=8, policy="conversational"):
    result = command._chunk_backfill_run(
        engine._config, engine._store.db_path, apply=True, limit=limit,
        retry_uncertain=False, policy=policy, incremental=True,
    )
    assert not result["error"] and not result["failed"], result
    return result


def missing(engine, policy="conversational"):
    with sqlite3.connect(engine._store.db_path) as conn:
        return {r[0] for r in conn.execute(
            "SELECT store_id FROM messages m WHERE "
            "(role IN ('user','assistant') OR (?='heads' AND role='tool')) AND NOT EXISTS "
            "(SELECT 1 FROM lcm_chunk_meta c WHERE c.store_id=m.store_id)", (policy,))}


def drain(engine, expected, policy="conversational"):
    """Check exact remaining via the exhaustive manual counter, and eventual fill."""
    for _ in range(40):
        actual = missing(engine, policy)
        assert actual <= expected
        with sqlite3.connect(engine._store.db_path) as conn:
            identity = conn.execute("SELECT identity_hash FROM lcm_chunk_meta LIMIT 1").fetchone()[0]
        exact = command._chunk_backfill_remaining(
            engine._store.db_path, identity, policy, 0, 0, incremental=False)[0]
        assert exact == len(actual)
        probe = command._chunk_backfill_remaining(
            engine._store.db_path, identity, policy, 0, 0, incremental=True)[0]
        assert bool(probe) == bool(actual), (probe, actual)
        if not actual:
            return
        result = run(engine, policy=policy)
        assert bool(result["remaining"]) == bool(missing(engine, policy))
    pytest.fail(f"unfilled: {missing(engine)}")


@pytest.mark.parametrize("event", ["remove", "revise", "chunk_revision", "policy_revision", "arrival",
                                        "abandon", "interrupt"])
def test_cursor_invalidation_and_interruptions(local_store, monkeypatch, event):
    engine, _ = local_store
    identity = seed(engine, 120)
    publish(engine, identity, [10])
    policy = "conversational"
    if event == "policy_revision":
        with sqlite3.connect(engine._store.db_path) as conn:
            conn.execute("UPDATE messages SET role='tool' WHERE store_id=117")
    run(engine)
    run(engine)
    with sqlite3.connect(engine._store.db_path) as conn:
        saved = get(conn, identity, "chunk_gapfill_scan")
        assert saved and "walk" in saved
    if event == "remove":
        # Passed coverage and an older row below the walk both need repair.
        with sqlite3.connect(engine._store.db_path) as conn:
            conn.execute("DELETE FROM lcm_chunk_meta WHERE store_id IN (115, 10)")
            assert get(conn, identity, "chunk_gapfill_scan") is None
    elif event == "revise":
        with sqlite3.connect(engine._store.db_path) as conn:
            conn.execute("UPDATE lcm_chunk_meta SET store_id=1000, chunk_id='1000:0' WHERE store_id=115")
            assert get(conn, identity, "chunk_gapfill_scan") is None
            # Clear the synthetic stale vector, leaving source context untouched.
            conn.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id='115:0'")
    elif event == "chunk_revision":
        monkeypatch.setattr(command, "CHUNK_POLICY_REVISION", command.CHUNK_POLICY_REVISION + 1)
    elif event == "policy_revision":
        policy = "heads"
    elif event == "arrival":
        with sqlite3.connect(engine._store.db_path) as conn:
            conn.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                         "VALUES(121,'seed','user','New synthetic fact.',0)")
        publish(engine, identity, [121])
    elif event == "abandon":
        with sqlite3.connect(engine._store.db_path) as conn:
            before = missing(engine)
            rows = command._chunk_gapfill_rows(conn, identity, (1, 120), "conversational", 2000,
                                               record_progress=True, gap_page_size=32)
            assert next(rows)[0] in before
            rows.close()
        assert missing(engine) == before
    elif event == "interrupt":
        writer = command._record_chunk_gapfill_walk

        def interrupted(conn, *args):
            writer(conn, *args)
            conn.commit()  # Crash after durable proof, before publication.
            raise RuntimeError("synthetic_interrupt_after_cursor_write")

        before = missing(engine)
        with monkeypatch.context() as patch:
            patch.setattr(command, "_record_chunk_gapfill_walk", interrupted)
            result = command._chunk_backfill_run(
                engine._config, engine._store.db_path, apply=True, limit=8,
                retry_uncertain=False, policy="conversational", incremental=True)
            assert result["error"] == "synthetic_interrupt_after_cursor_write"
        assert missing(engine) == before
    drain(engine, missing(engine, policy), policy)


def test_fifty_slice_selection_matches_pinned_base_with_arrivals(local_store, monkeypatch):
    engine, _ = local_store
    identity = seed(engine, 2502)
    candidate = command._chunk_gapfill_rows
    selections = []
    for turn in range(55):
        if turn % 7 == 0:
            with sqlite3.connect(engine._store.db_path) as conn:
                conn.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                             "VALUES(?,'seed','user','Arriving synthetic fact.',0)", (2503 + turn,))
        with sqlite3.connect(engine._store.db_path) as conn:
            conn.execute("SAVEPOINT compare")
            # Strip only the candidate proof for a genuinely base-shaped scan.
            record = get(conn, identity, "chunk_gapfill_scan")
            if record and "walk" in record:
                record.pop("walk")
                conn.execute("UPDATE metadata SET value=? WHERE key=?",
                             (json.dumps(record), f"chunk_gapfill_scan:{identity}"))
            monkeypatch.setattr(command, "_chunk_gapfill_rows", base_gapfill)
            expected = command._chunk_pending_rows(conn, identity, "conversational", 8, incremental=True)
            conn.execute("ROLLBACK TO compare")
            conn.execute("RELEASE compare")
            monkeypatch.setattr(command, "_chunk_gapfill_rows", candidate)
            actual = command._chunk_pending_rows(conn, identity, "conversational", 8, incremental=True)
            assert actual == expected, turn
            selections.append([r[0] for r in actual[1]])
        result = run(engine)
        assert result["embedded"] == len(actual[1])
    assert len(selections) >= 50


def test_trigger_installation_distrusts_legacy_skip_proof():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany("INSERT INTO metadata VALUES(?, 'legacy')", [
        ("chunk_gapfill_scan:synthetic",), ("chunk_policy_revision_through:synthetic",),
        ("source_context",),
    ])
    ensure_chunk_tables(conn)
    assert list(conn.execute("SELECT key FROM metadata")) == [("source_context",)]
    conn.execute("INSERT INTO metadata VALUES('chunk_gapfill_scan:synthetic', 'current')")
    ensure_chunk_tables(conn)
    assert conn.execute("SELECT value FROM metadata WHERE key='chunk_gapfill_scan:synthetic'").fetchone()[0] == "current"
    conn.close()

