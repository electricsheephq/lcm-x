"""#1748: filled history must not amplify discovery, or weaken gap repair."""
import json
import random
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


def active_identity(engine):
    return command.EmbeddingIdentity.canonical(
        engine._config.embedding_provider, engine._config.embedding_model, "", 2, "float32", "little", "chunk",
    ).identity_hash


def missing(engine, policy="conversational"):
    with sqlite3.connect(engine._store.db_path) as conn:
        return {r[0] for r in conn.execute(
            "SELECT store_id FROM messages m WHERE "
            "(role IN ('user','assistant') OR (?='heads' AND role='tool')) "
            "AND trim(COALESCE(content,''))!='' AND NOT EXISTS "
            "(SELECT 1 FROM lcm_chunk_meta c WHERE c.store_id=m.store_id AND c.identity_hash=?)",
            (policy, active_identity(engine)))}


def drain(engine, expected, policy="conversational"):
    """Check exact remaining via the exhaustive manual counter, and eventual fill."""
    for _ in range(40):
        actual = missing(engine, policy)
        assert actual <= expected
        identity = active_identity(engine)
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


@pytest.mark.parametrize("mutation", ["delete", "identity_move"])
@pytest.mark.parametrize("established", [False, True])
def test_invalidation_between_selection_and_walk_write(local_store, monkeypatch, mutation, established):
    engine, _ = local_store
    identity = seed(engine, 200)
    if established:
        for _ in range(3):
            run(engine)
    else:
        publish(engine, identity, list(range(150, 190)))
    victim = 190 if established else 170
    assert victim not in missing(engine)
    writer = command._record_chunk_gapfill_walk

    def raced(conn, ident, state):
        assert state["walk_progress"]["walk"]["below"] < victim
        with sqlite3.connect(engine._store.db_path) as other:
            if mutation == "delete":
                other.execute("DELETE FROM lcm_chunk_meta WHERE store_id=?", (victim,))
            else:
                other.execute("UPDATE lcm_chunk_meta SET identity_hash='moved' WHERE store_id=?", (victim,))
            other.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id=?", (f"{victim}:0",))
        writer(conn, ident, state)
        assert get(conn, ident, "chunk_gapfill_scan") is None

    with monkeypatch.context() as patch:
        patch.setattr(command, "_record_chunk_gapfill_walk", raced)
        run(engine)
    drain(engine, missing(engine))


@pytest.mark.parametrize("initial", ["empty", "tool"])
def test_row_becomes_eligible_after_walk(local_store, initial):
    engine, _ = local_store
    identity = seed(engine, 200)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET role=?, content=? WHERE store_id=150",
                     ("tool" if initial == "tool" else "user", "fact" if initial == "tool" else ""))
    for _ in range(8):
        run(engine)
    with sqlite3.connect(engine._store.db_path) as conn:
        assert get(conn, identity, "chunk_gapfill_scan")["walk"]["below"] < 150
        conn.execute("UPDATE messages SET role='user', content='Edited synthetic fact.' WHERE store_id=150")
        assert get(conn, identity, "chunk_gapfill_scan") is None
    drain(engine, missing(engine))


def test_tool_content_rewrite_keeps_proof(local_store):
    engine, _ = local_store
    identity = seed(engine, 120)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET role='tool' WHERE store_id=90")
    run(engine)
    with sqlite3.connect(engine._store.db_path) as conn:
        before = get(conn, identity, "chunk_gapfill_scan")
        assert before
        conn.execute("UPDATE messages SET content='Externalized tool result.' WHERE store_id=90")
        assert get(conn, identity, "chunk_gapfill_scan") == before


def test_raw_chunk_tables_create_metadata():
    with sqlite3.connect(":memory:") as conn:
        ensure_chunk_tables(conn)
        conn.execute("INSERT INTO lcm_chunk_meta(chunk_id,identity_hash,store_id) VALUES('1:0','x',1)")
        conn.execute("DELETE FROM lcm_chunk_meta")


def test_exhaustion_certificate_cas(tmp_path):
    db = tmp_path / "certificate.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT)")
        ensure_chunk_tables(conn)
        conn.executemany("INSERT INTO messages VALUES(?,'user','Synthetic fact.')", [(1,), (2,)])
        conn.execute("INSERT INTO lcm_chunk_meta(chunk_id,identity_hash,store_id) VALUES('2:0','x',2)")
        conn.execute("INSERT INTO metadata VALUES('chunk_gapfill_scan:x', '{}')")
    with sqlite3.connect(db) as conn:
        rows = command._chunk_gapfill_rows(conn, "x", (1, 2), "conversational", 8, record_progress=True)
        assert next(rows)[0] == 1
        with sqlite3.connect(db) as other:
            other.execute("DELETE FROM lcm_chunk_meta WHERE store_id=2")
        assert list(rows) == []
        assert get(conn, "x", "chunk_gapfill_scan") is None
        assert get(conn, "x", "chunk_policy_revision_through") is None


def test_hole_above_old_high_eventually_filled(local_store):
    engine, _ = local_store
    identity = seed(engine, 120)
    publish(engine, identity, [10])
    run(engine)
    run(engine)
    with sqlite3.connect(engine._store.db_path) as conn:
        assert "walk" in get(conn, identity, "chunk_gapfill_scan")
        conn.executemany("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                         "VALUES(?,'seed','user','New synthetic fact.',0)", [(121,), (122,), (123,)])
    publish(engine, identity, [123])
    drain(engine, missing(engine))


def test_live_arrival_cost(tmp_path):
    with sqlite3.connect(tmp_path / "arrival.db", factory=VisitedConnection) as conn:
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT)")
        ensure_chunk_tables(conn)
        conn.executemany("INSERT INTO messages VALUES(?,'user','Synthetic fact.')",
                         ((i,) for i in range(1, 100_001)))

        def publish_rows(rows):
            conn.executemany("INSERT INTO lcm_chunk_meta(chunk_id,identity_hash,store_id) VALUES(?,'synthetic',?)",
                             ((f"{r[0]}:0", r[0]) for r in rows))

        for _ in range(30):
            publish_rows(discover(conn, "synthetic", 100_000))
        steady = cost(conn, "synthetic", 100_000)
        measurements = []
        for high in range(100_001, 100_021):
            conn.execute("INSERT INTO messages VALUES(?,'user','New synthetic fact.')", (high,))
            publish_rows([(high,)])
            measured = cost(conn, "synthetic", high)
            measurements.append(measured)
            publish_rows(discover(conn, "synthetic", high))
        print("GAPWALK_ARRIVAL_COST " + json.dumps(dict(steady=steady, arrivals=measurements), sort_keys=True))
        assert all(m["sqlite_steps"] <= 2 * steady["sqlite_steps"] for m in measurements), measurements


@pytest.mark.parametrize("random_seed", range(200))
def test_randomized_arrivals_converge(local_store, random_seed):
    engine, _ = local_store
    rng = random.Random(random_seed)
    identity = seed(engine, 150)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.executemany("UPDATE messages SET content='   ' WHERE store_id=?",
                         [(i,) for i in rng.sample(range(2, 150), 30)])
        conn.executemany("UPDATE messages SET role='tool' WHERE store_id=?",
                         [(i,) for i in rng.sample(range(2, 150), 15)])
    publish(engine, identity, rng.sample(sorted(missing(engine)), 40))
    top = 150
    for _ in range(60):
        action = rng.random()
        if action < 0.7:
            run(engine, limit=rng.randint(1, 12))
        elif action < 0.88:
            ids = list(range(top + 1, top + rng.randint(1, 4) + 1))
            top = ids[-1]
            with sqlite3.connect(engine._store.db_path) as conn:
                conn.executemany("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                                 "VALUES(?,'seed','user','New synthetic fact.',0)", [(i,) for i in ids])
            if rng.random() < 0.6:
                publish(engine, identity, [top])
        else:
            with sqlite3.connect(engine._store.db_path) as conn:
                have = [r[0] for r in conn.execute("SELECT DISTINCT store_id FROM lcm_chunk_meta")]
                if have:
                    conn.executemany("DELETE FROM lcm_chunk_meta WHERE store_id=?",
                                     [(v,) for v in rng.sample(have, min(len(have), rng.randint(1, 3)))])
    drain(engine, missing(engine))
