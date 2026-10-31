"""#1748: filled history must not amplify discovery, or weaken gap repair."""
import json
import gc
from contextlib import closing
import os
import itertools
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
    assert isinstance(result, dict), result
    assert not result["error"] and not result["failed"], result
    return result


def active_identity(engine):
    return command.EmbeddingIdentity.canonical(
        engine._config.embedding_provider, engine._config.embedding_model, "", 2, "float32", "little", "chunk",
    ).identity_hash


def missing(engine, policy="conversational"):
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        return {r[0] for r in conn.execute(
            "SELECT store_id, role, content FROM messages m WHERE "
            "(role IN ('user','assistant') OR (?='heads' AND role='tool')) "
            "AND trim(COALESCE(content,''))!='' AND NOT EXISTS "
            "(SELECT 1 FROM lcm_chunk_meta c WHERE c.store_id=m.store_id AND c.identity_hash=?)",
            (policy, active_identity(engine)))
                if any(c.char_end > c.char_start for c in command.chunk_message(*r, policy=policy))}


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
        with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
            conn.execute("UPDATE messages SET role='tool' WHERE store_id=117")
    run(engine)
    run(engine)
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        saved = get(conn, identity, "chunk_gapfill_scan")
        assert saved and "walk" in saved
    if event == "remove":
        # Passed coverage and an older row below the walk both need repair.
        with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
            conn.execute("DELETE FROM lcm_chunk_meta WHERE store_id IN (115, 10)")
            assert get(conn, identity, "chunk_gapfill_scan") is None
    elif event == "revise":
        with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
            conn.execute("UPDATE lcm_chunk_meta SET store_id=1000, chunk_id='1000:0' WHERE store_id=115")
            assert get(conn, identity, "chunk_gapfill_scan") is None
            # Clear the synthetic stale vector, leaving source context untouched.
            conn.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id='115:0'")
    elif event == "chunk_revision":
        monkeypatch.setattr(command, "CHUNK_POLICY_REVISION", command.CHUNK_POLICY_REVISION + 1)
    elif event == "policy_revision":
        policy = "heads"
    elif event == "arrival":
        with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
            conn.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                         "VALUES(121,'seed','user','New synthetic fact.',0)")
        publish(engine, identity, [121])
    elif event == "abandon":
        with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
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
            with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
                conn.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                             "VALUES(?,'seed','user','Arriving synthetic fact.',0)", (2503 + turn,))
        with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
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
        with closing(sqlite3.connect(engine._store.db_path)) as other, other:
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
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        conn.execute("UPDATE messages SET role=?, content=? WHERE store_id=150",
                     ("tool" if initial == "tool" else "user", "fact" if initial == "tool" else ""))
    for _ in range(8):
        run(engine)
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        assert get(conn, identity, "chunk_gapfill_scan")["walk"]["below"] < 150
        conn.execute("UPDATE messages SET role='user', content='Edited synthetic fact.' WHERE store_id=150")
        assert get(conn, identity, "chunk_gapfill_scan") is None
    drain(engine, missing(engine))


def test_tool_content_rewrite_keeps_proof(local_store):
    engine, _ = local_store
    identity = seed(engine, 120)
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        conn.execute("UPDATE messages SET role='tool' WHERE store_id=90")
    run(engine)
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        before = conn.execute("SELECT value FROM metadata WHERE key=?",
                              (f"chunk_gapfill_scan:{identity}",)).fetchone()[0]
        assert before
    assert engine._store.gc_externalized_tool_result(90, "[Externalized tool output: ref=synthetic]")
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        assert conn.execute("SELECT value FROM metadata WHERE key=?",
                            (f"chunk_gapfill_scan:{identity}",)).fetchone()[0] == before


def test_raw_chunk_tables_create_metadata():
    with closing(sqlite3.connect(":memory:")) as conn, conn:
        ensure_chunk_tables(conn)
        conn.execute("INSERT INTO lcm_chunk_meta(chunk_id,identity_hash,store_id) VALUES('1:0','x',1)")
        conn.execute("DELETE FROM lcm_chunk_meta")


def test_exhaustion_certificate_cas(tmp_path):
    db = tmp_path / "certificate.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT)")
        ensure_chunk_tables(conn)
        conn.executemany("INSERT INTO messages VALUES(?,'user','Synthetic fact.')", [(1,), (2,)])
        conn.execute("INSERT INTO lcm_chunk_meta(chunk_id,identity_hash,store_id) VALUES('2:0','x',2)")
        conn.execute("INSERT INTO metadata VALUES('chunk_gapfill_scan:x', '{}')")
    with closing(sqlite3.connect(db)) as conn, conn:
        rows = command._chunk_gapfill_rows(conn, "x", (1, 2), "conversational", 8, record_progress=True)
        assert next(rows)[0] == 1
        with closing(sqlite3.connect(db)) as other, other:
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
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        assert "walk" in get(conn, identity, "chunk_gapfill_scan")
        conn.executemany("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                         "VALUES(?,'seed','user','New synthetic fact.',0)", [(121,), (122,), (123,)])
    publish(engine, identity, [123])
    drain(engine, missing(engine))


def test_live_arrival_cost(tmp_path):
    with closing(sqlite3.connect(tmp_path / "arrival.db", factory=VisitedConnection)) as conn, conn:
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


@pytest.mark.parametrize("blank,dense", [("\n", False), ("\n\n", False), ("\t", False), ("\n", True)])
def test_nonchunkable_prefix_keeps_live_walk(tmp_path, blank, dense):
    """Catch-up through a SQL-eligible, non-chunkable prefix stays bounded."""
    with closing(sqlite3.connect(tmp_path / "blank.db", factory=VisitedConnection)) as conn, conn:
        conn.execute("CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT)")
        ensure_chunk_tables(conn)
        conn.executemany("INSERT INTO messages VALUES(?,'user',?)",
                         ((i, blank if (i % 50 == 0 if dense else i == 90_000) else "Synthetic fact.")
                          for i in range(1, 100_001)))

        def publish_rows(rows):
            conn.executemany("INSERT OR IGNORE INTO lcm_chunk_meta(chunk_id,identity_hash,store_id) "
                             "VALUES(?,'synthetic',?)", ((f"{r[0]}:0", r[0]) for r in rows))

        if dense:
            filled = 0
            while filled < 60_000:
                rows = discover(conn, "synthetic", 100_000)[:60_000 - filled]
                assert rows
                publish_rows(rows)
                filled += len(rows)
        steady = cost(conn, "synthetic", 100_000)
        measurements, walks = [], []
        for high in range(100_001, 100_021 if dense else 100_031):
            conn.execute("INSERT INTO messages VALUES(?,'user','Arriving synthetic fact.')", (high,))
            publish_rows([(high,)])
            measured = cost(conn, "synthetic", high)
            measurements.append(measured)
            saved = get(conn, "synthetic", "chunk_gapfill_scan")
            walks.append(bool(saved and "walk" in saved))
            publish_rows(discover(conn, "synthetic", high))
        print("GAPWALK_BLANK_COST " + json.dumps(dict(blank=blank, dense=dense, steady=steady,
                                                      arrivals=measurements, walks=walks), sort_keys=True))
        assert all(m["sqlite_steps"] <= 2 * steady["sqlite_steps"] for m in measurements), measurements
        # Dense history can naturally finish on the final passes.
        assert all(walks if not dense else walks[:-2]), walks


def test_appended_newline_keeps_product_walk(local_store, monkeypatch):
    engine, _ = local_store
    identity = seed(engine, 27_999)
    engine._store.append("seed", {"role": "assistant", "content": "\n"})
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        assert conn.execute("SELECT role,content FROM messages WHERE store_id=28000").fetchone() == ("assistant", "\n")
        conn.executemany("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                         "VALUES(?,'seed','user','Synthetic fact.',0)", [(i,) for i in range(28_001, 30_001)])
    publish(engine, identity, [30_000])
    real = command._chunk_pending_rows
    counts = []

    def counted(conn, *args, **kwargs):
        steps = 0

        def step():
            nonlocal steps
            steps += 1
            return 0

        conn.set_progress_handler(step, 1)
        try:
            return real(conn, *args, **kwargs)
        finally:
            conn.set_progress_handler(None, 0)
            counts.append(steps)

    monkeypatch.setattr(command, "_chunk_pending_rows", counted)
    walks = []
    for turn in range(14):
        engine._store.append("live", {"role": "user", "content": f"Arriving synthetic fact {turn}."})
        run(engine, limit=2000)
        with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
            saved = get(conn, identity, "chunk_gapfill_scan")
            walks.append(bool(saved and "walk" in saved))
    print("GAPWALK_BLANK_PRODUCT " + json.dumps(dict(steps=counts, walks=walks)))
    assert max(counts) <= 2 * max(counts[:3]), counts
    assert all(walks), walks


@pytest.mark.parametrize("mutation", ["delete", "message_update"])
@pytest.mark.parametrize("second_publishes", [False, True])
def test_initial_selection_cas_rejects_aba(local_store, mutation, second_publishes):
    """A new first-selection epoch cannot revive a stale writer's walk."""
    engine, _ = local_store
    identity = seed(engine, 200)
    publish(engine, identity, list(range(150, 191)))
    db = engine._store.db_path
    if mutation == "message_update":
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute("UPDATE messages SET role='tool' WHERE store_id=175")
            conn.execute("DELETE FROM lcm_chunk_meta WHERE store_id=175")
            conn.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id='175:0'")
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("DELETE FROM metadata WHERE key LIKE 'chunk_gapfill_scan:%' "
                     "OR key LIKE 'chunk_policy_revision_through:%'")
    victim = 170 if mutation == "delete" else 175
    with closing(sqlite3.connect(db)) as a, a, closing(sqlite3.connect(db)) as b, b:
        state_a, state_b = {}, {}
        first = command._chunk_gapfill_rows(a, identity, (1, 200), "conversational", 32,
                                           record_progress=True, gap_page_size=32, gap_scan_state=state_a)
        list(itertools.islice(first, 32))
        a.commit()
        assert state_a["walk_progress"]["walk"]["below"] < victim
        assert victim not in state_a["walk_progress"]["walk"]["pending"]
        with closing(sqlite3.connect(db)) as other, other:
            if mutation == "delete":
                other.execute("DELETE FROM lcm_chunk_meta WHERE store_id=170")
                other.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id='170:0'")
            else:
                other.execute("UPDATE messages SET role='user' WHERE store_id=175")
        second = command._chunk_gapfill_rows(b, identity, (1, 200), "conversational", 8,
                                            record_progress=True, gap_page_size=32, gap_scan_state=state_b)
        rows_b = list(itertools.islice(second, 8 if second_publishes else 1))
        b.commit()
        token_b = state_b["scan_value"]
        record_walk(a, identity, state_a)
        a.commit()
        after_stale = a.execute("SELECT value FROM metadata WHERE key=?",
                                (f"chunk_gapfill_scan:{identity}",)).fetchone()[0]
        if second_publishes:
            record_walk(b, identity, state_b)
            b.commit()
        first.close()
        second.close()
    if second_publishes:
        publish(engine, identity, [r[0] for r in rows_b])
    for _ in range(40):
        if not missing(engine):
            break
        run(engine)
    remaining = command._chunk_backfill_remaining(db, identity, "conversational", 0, 0, incremental=True)[0]
    assert not missing(engine), (sorted(missing(engine)), remaining, state_a["walk_progress"]["walk"]["below"])
    assert after_stale == token_b
    assert token_b != state_a["scan_value"]


@pytest.mark.parametrize("empty", [None, "", "   ", "\t\n\r\v\f"])
def test_heads_tool_empty_to_text_is_filled(local_store, empty):
    engine, _ = local_store
    identity = seed(engine, 200)
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        conn.execute("UPDATE messages SET role='tool',content=? WHERE store_id=150", (empty,))
    for _ in range(8):
        run(engine, policy="heads")
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        assert get(conn, identity, "chunk_gapfill_scan")["walk"]["below"] < 150
        conn.execute("UPDATE messages SET content='Tool output fact.' WHERE store_id=150")
    drain(engine, missing(engine, "heads"), policy="heads")
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        assert conn.execute("SELECT 1 FROM lcm_chunk_meta WHERE store_id=150 AND identity_hash=?",
                            (identity,)).fetchone()
        # The reverse empty/nonempty transition must invalidate too.
        conn.execute("INSERT OR REPLACE INTO metadata VALUES(?, 'proof')", (f"chunk_gapfill_scan:{identity}",))
        conn.execute("UPDATE messages SET content=? WHERE store_id=150", (empty,))
        assert get(conn, identity, "chunk_gapfill_scan") is None


@pytest.fixture
def seed_fd_guard(request):
    # Disabling cyclic GC proves deterministic close, rather than a lucky GC run.
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    before = len(os.listdir("/dev/fd"))
    try:
        yield
    finally:
        after = len(os.listdir("/dev/fd"))
        if enabled:
            gc.enable()
        print(f"GAPWALK_FDS seed={request.node.callspec.params['random_seed']} "
              f"worker={os.environ.get('PYTEST_XDIST_WORKER', 'main')} "
              f"pid={os.getpid()} before={before} after={after} delta={after - before}")
        assert after == before, f"per-seed fd delta: {after - before}"


@pytest.mark.parametrize("random_seed", range(200))
def test_randomized_arrivals_converge(seed_fd_guard, local_store, random_seed):
    engine, _ = local_store
    rng = random.Random(random_seed)
    identity = seed(engine, 150)
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        conn.executemany("UPDATE messages SET content=? WHERE store_id=?",
                         [(rng.choice(["   ", "\n", "\n\n", "\t", " \t\r\n "]), i)
                          for i in rng.sample(range(2, 150), 30)])
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
            with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
                conn.executemany("INSERT INTO messages(store_id,session_id,role,content,timestamp) "
                                 "VALUES(?,'seed','user',?,0)",
                                 [(i, rng.choice(["New synthetic fact.", "\n", "\t"])) for i in ids])
            if rng.random() < 0.6 and top in missing(engine):
                publish(engine, identity, [top])
        else:
            with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
                have = [r[0] for r in conn.execute("SELECT DISTINCT store_id FROM lcm_chunk_meta")]
                if have:
                    conn.executemany("DELETE FROM lcm_chunk_meta WHERE store_id=?",
                                     [(v,) for v in rng.sample(have, min(len(have), rng.randint(1, 3)))])
    drain(engine, missing(engine))


def test_identity_move_invalidates_destination_proof(local_store):
    engine, _ = local_store
    identity = seed(engine, 200, low=100)
    publish(engine, identity, range(100, 201))
    publish(engine, identity, [10])
    with closing(sqlite3.connect(engine._store.db_path)) as conn, conn:
        conn.execute("UPDATE lcm_chunk_meta SET identity_hash='source' WHERE store_id=10")
        for ident in ("source", identity):
            conn.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)",
                         (f"chunk_gapfill_scan:{ident}", json.dumps(dict(
                             revision=command.CHUNK_POLICY_REVISION, policy="conversational",
                             high=200, pending=[]))))
            conn.execute("INSERT OR REPLACE INTO metadata VALUES(?, '200')",
                         (f"chunk_policy_revision_through:{ident}",))
            conn.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)",
                         (f"chunk_policy_revision:{ident}", str(command.CHUNK_POLICY_REVISION)))
        conn.execute("UPDATE lcm_chunk_meta SET identity_hash=? WHERE identity_hash='source'", (identity,))
        proofs = {ident: [get(conn, ident, key) for key in
                         ("chunk_gapfill_scan", "chunk_policy_revision_through")]
                  for ident in ("source", identity)}
    for _ in range(40):
        run(engine)
        if not missing(engine):
            break
    assert not missing(engine), sorted(missing(engine))
    assert proofs == {"source": [None, None], identity: [None, None]}
