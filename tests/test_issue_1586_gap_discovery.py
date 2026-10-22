"""#1586: lazy gap work must preserve stock selection and lossless coverage."""
import sqlite3
from contextlib import closing
import json
import random
from types import SimpleNamespace

import pytest

from hermes_lcm.chunking import CHUNK_POLICY_REVISION
import sys

from hermes_lcm import command, embedding_maintenance as maintenance, tools
from tests.test_issue_1014_incremental_embedding import FakeProvider
from tests.test_issue_1586_local_pacing import local_store as local_store


def seed(engine, count, *, text="My bicycle is blue.", low=1, first_text=None):
    db = engine._store.db_path
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.executemany(
            "INSERT INTO messages(store_id, session_id, role, content, timestamp) VALUES(?, 'seed', 'user', ?, 0)",
            [(i, first_text if i == 1 and first_text is not None else text) for i in range(1, count + 1)],
        )
    vectors = command.VectorStore(db, config=engine._config)
    try:
        identity = command.EmbeddingIdentity.canonical(
            engine._config.embedding_provider, engine._config.embedding_model, "", 2, "float32", "little", "chunk",
        )
        for i in (low, count):
            row = vectors.connection.execute("SELECT store_id, role, content FROM messages WHERE store_id=?", (i,)).fetchone()
            chunk = command.chunk_message(*row, policy="conversational")[0]
            vectors.record_chunk_embedding(
                chunk.chunk_id, engine._config.embedding_model, [1.0, 0.0], store_id=i,
                chunk_index=0, char_start=chunk.char_start, char_end=chunk.char_end,
                token_estimate=chunk.token_estimate, identity=identity,
            )
    finally:
        vectors.close()
    return identity.identity_hash


def test_gap_discovery_work_bound_at_production_defaults(local_store, monkeypatch):
    engine, _ = local_store
    seed(engine, 2102)  # 2,100 uncovered interior rows; production 2,000 budget.
    assert command._CHUNK_HISTORY_SCAN_MESSAGES == 2000
    chunk_message = command.chunk_message
    calls = []

    def counted(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_chunk_gapfill_rows":
            calls.append(args[0])
        return chunk_message(*args, **kwargs)

    monkeypatch.setattr(command, "chunk_message", counted)
    state = {}
    passes = []
    for stage in range(4):
        calls.clear()
        result = maintenance.run_incremental_embedding_pass(
            engine._store.db_path, engine._config, slice_state=state,
        )
        assert result["outcome"] != "error", result
        if stage in (1, 3):
            passes.append((result, len(calls)))
    assert passes[0][0]["vectors_written"] == 32
    assert passes[0][1] <= 64, passes
    assert passes[1][1] <= 32, passes


# Frozen from base e1dc2a164c66d2419d8d580852d69eb851f95eed.
def stock_gapfill_rows(conn, identity_hash, bounds, policy, limit, *, record_progress=False):
    """Scan new territory once; revisit unpublished candidates by id until covered."""
    low, high = bounds
    scan_key = f"chunk_gapfill_scan:{identity_hash}"
    saved = conn.execute("SELECT value FROM metadata WHERE key = ?", (scan_key,)).fetchone()
    try:
        scan = json.loads(saved[0]) if saved else {}
        valid = (
            isinstance(scan, dict)
            and type(scan.get("revision")) is int and type(scan.get("high")) is int
            and isinstance(scan.get("policy"), str)
            and ("pending" not in scan or (isinstance(scan["pending"], list)
                 and all(type(i) is int for i in scan["pending"])))
            and ("ceiling" not in scan or type(scan["ceiling"]) is int)
        )
        if not valid:
            scan = {}
    except Exception:
        scan = {}  # Unusable progress metadata requires the full gap revalidation.
    if scan.get("revision") != CHUNK_POLICY_REVISION or scan.get("policy") != policy:
        scan = {}
    pending = []
    ceiling = high + 1
    if "pending" in scan and scan["high"] <= high:
        ids = [i for i in scan["pending"] if i >= low]
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            pending.extend(conn.execute(
                "SELECT store_id, role, content FROM messages AS p "
                f"WHERE store_id IN ({','.join('?' for _ in batch)}) "
                "AND NOT EXISTS (SELECT 1 FROM lcm_chunk_meta AS c "
                "WHERE c.store_id = p.store_id AND c.identity_hash = ?)",
                (*batch, identity_hash),
            ).fetchall())
        low = max(low, scan["high"] + 1)
    elif scan.get("high") == high:
        # Only our own scan can certify a zero-chunk prefix. Older writers can
        # move the legacy skip key without updating any of our progress keys.
        ceiling = scan.get("ceiling", ceiling)
    roles = "'assistant', 'user', 'tool'" if policy == "heads" else "'assistant', 'user'"
    rows = conn.execute(
        f"SELECT store_id, role, content FROM messages AS m "
        f"WHERE store_id >= ? AND store_id <= ? AND store_id < ? "
        f"AND lower(trim(role)) IN ({roles}) AND trim(COALESCE(content, '')) != '' "
        "AND NOT EXISTS (SELECT 1 FROM lcm_chunk_meta AS c "
        "WHERE c.store_id = m.store_id AND c.identity_hash = ?) "
        "ORDER BY store_id DESC LIMIT ?",
        (low, high, ceiling, identity_hash, limit),
    ).fetchall() if low <= high else []
    candidates = []
    for row in sorted([*rows, *pending], key=lambda r: r[0], reverse=True):
        if any(c.char_end > c.char_start for c in command.chunk_message(*row, policy=policy)):
            candidates.append(row)
        elif not candidates:
            ceiling = min(ceiling, row[0])
    if record_progress:
        progress = {"revision": CHUNK_POLICY_REVISION, "policy": policy, "high": high}
        values = []
        if len(rows) < limit and len(candidates) < limit:
            # Persist every candidate before yielding: selection is not publication,
            # and interruption must leave all unpublished rows recoverable.
            progress["pending"] = [r[0] for r in candidates]
            through = min(progress["pending"], default=high + 1) - 1
            values.append((f"chunk_policy_revision_through:{identity_hash}", str(through)))
            if not candidates:
                values.append((f"chunk_policy_revision:{identity_hash}", str(CHUNK_POLICY_REVISION)))
        elif ceiling < high + 1:
            # A LIMIT-sized result does not prove the whole range.
            progress["ceiling"] = ceiling
        else:
            progress = None
        if progress is not None:
            values.append((scan_key, json.dumps(progress)))
            conn.execute(
                "INSERT INTO metadata(key, value) VALUES " + ",".join("(?, ?)" for _ in values)
                + " ON CONFLICT(key) DO UPDATE SET value = excluded.value WHERE value != excluded.value",
                tuple(item for pair in values for item in pair),
            )
    yield from candidates



def put(conn, identity, key, value):
    conn.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)",
                 (f"{key}:{identity}", json.dumps(value)))


def get(conn, identity, key):
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (f"{key}:{identity}",)).fetchone()
    value = json.loads(row[0]) if row else None
    if key == "chunk_policy_revision_through" and isinstance(value, dict):
        return value["through"]
    return value


def scan_record(conn, identity, high, policy="conversational", **kwargs):
    put(conn, identity, "chunk_gapfill_scan", dict(
        revision=CHUNK_POLICY_REVISION, policy=policy, high=high, **kwargs))


def hold(conn, identity, ids, state="uncertain"):
    command._ensure_inflight_table(conn)
    conn.executemany(
        "INSERT INTO lcm_embedding_backfill_inflight "
        "(embedded_id, identity_hash, lease_id, generation, claimed_at, state, updated_at) "
        "VALUES (?, ?, 'fixture', 1, 0, ?, 0)",
        [(f"{i}:0", identity, state) for i in ids],
    )


def select(conn, identity, limit=32, **kwargs):
    return command._chunk_pending_rows(conn, identity, kwargs.pop("policy", "conversational"), limit,
                                       incremental=True, **kwargs)


def frozen_gap(*args, **kwargs):
    kwargs.pop("gap_page_size", None)
    kwargs.pop("gap_scan_state", None)
    return stock_gapfill_rows(*args, **kwargs)


def publish(engine, identity, ids):
    vectors = command.VectorStore(engine._store.db_path, config=engine._config)
    try:
        profile = command.EmbeddingIdentity.canonical(
            engine._config.embedding_provider, engine._config.embedding_model, "", 2, "float32", "little", "chunk")
        assert profile.identity_hash == identity
        for i in ids:
            row = vectors.connection.execute(
                "SELECT store_id, role, content FROM messages WHERE store_id=?", (i,)).fetchone()
            for c in command.chunk_message(*row, policy="conversational"):
                vectors.record_chunk_embedding(
                    c.chunk_id, engine._config.embedding_model, [1.0, 0.0], store_id=i,
                    chunk_index=c.chunk_index, char_start=c.char_start, char_end=c.char_end,
                    token_estimate=c.token_estimate, identity=profile)
    finally:
        vectors.close()


@pytest.mark.parametrize("policy", ["conversational", "heads", "full"])
def test_bounded_selection_equals_stock_selection(local_store, monkeypatch, policy):
    engine, _ = local_store
    identity = seed(engine, 112)
    lazy = command._chunk_gapfill_rows
    rng = random.Random(1586)
    with sqlite3.connect(engine._store.db_path) as conn:
        # Explicit metadata shapes, with randomized roles, sizes, held states and
        # limits. Each comparison starts from the identical unmodified store.
        for shape in ("legacy", "ceiling", "pending", "grown_pending", "through"):
            for attempt in range(8):
                conn.execute("SAVEPOINT scenario")
                for i in range(2, 112):
                    role = rng.choice(["user", "assistant", "tool", "system"])
                    text = rng.choice(["\t\n", f"Synthetic fact {i}.", "Several facts. " * 700])
                    conn.execute("UPDATE messages SET role=?, content=? WHERE store_id=?", (role, text, i))
                hold(conn, identity, range(90, 100), rng.choice(["claimed", "dispatched", "uncertain"]))
                # Real partial row and out-of-window partial bookkeeping.
                conn.execute("UPDATE messages SET role='user', content=? WHERE store_id=1",
                             ("A factual sentence with several details. " * 2100,))
                put(conn, identity, "chunk_partial_messages", [1])
                if shape == "ceiling":
                    conn.execute("UPDATE messages SET content='\t\n' WHERE store_id>=100 AND store_id<112")
                    scan_record(conn, identity, 112, policy=policy, ceiling=100)
                elif shape in ("pending", "grown_pending"):
                    high = 112 if shape == "pending" else 70
                    scan_record(conn, identity, high, policy=policy, pending=list(range(2, high)))
                elif shape == "through":
                    put(conn, identity, "chunk_policy_revision", CHUNK_POLICY_REVISION)
                    put(conn, identity, "chunk_policy_revision_through", 40)
                limit = rng.choice([1, 8, 32, 96])
                kwargs = dict(policy=policy, include_partials=bool(attempt % 2),
                              partial_window={1} if attempt % 3 else set())
                monkeypatch.setattr(command, "_chunk_gapfill_rows", frozen_gap)
                expected = select(conn, identity, limit, **kwargs)
                monkeypatch.setattr(command, "_chunk_gapfill_rows", lazy)
                actual = select(conn, identity, limit, **kwargs)
                assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), (shape, attempt)
                conn.execute("ROLLBACK TO scenario")
                conn.execute("RELEASE scenario")


@pytest.mark.parametrize("provider", ["ollama", "fastembed"])
@pytest.mark.parametrize("state", ["claimed", "dispatched", "uncertain"])
def test_held_rows_beyond_slice_bound_do_not_starve_gap(local_store, provider, state):
    engine, _ = local_store
    engine._config.embedding_provider = provider
    vectors = command.VectorStore(engine._store.db_path, config=engine._config)
    vectors.register_profile(engine._config.embedding_model, provider, 2, task="chunk")
    vectors.close()
    identity = seed(engine, 2102)
    with sqlite3.connect(engine._store.db_path) as conn:
        # Exercise the full stock step-over budget: 1,999 pinned rows, then one
        # selectable row, even though the production page size is only 32.
        hold(conn, identity, range(103, 2102), state)
        _, docs, _ = select(conn, identity)
        assert docs, "held rows must not hide the deeper selectable gap"
        assert docs[0][0] == "102:0"
    engine._config.embedding_provider = provider
    # Selection is shared by sliced FastEmbed and unsliced Ollama callers.
    result = command._chunk_backfill_run(engine._config, engine._store.db_path,
        apply=True, limit=32, retry_uncertain=False, policy="conversational", incremental=True,
        **({"maintenance_slice": 1.0} if provider == "fastembed" else {}))
    assert result["embedded"] > 0, result


def test_pinned_candidate_then_zero_chunk_run_does_not_starve(local_store, monkeypatch):
    engine, _ = local_store
    identity = seed(engine, 140)
    with sqlite3.connect(engine._store.db_path) as conn:
        hold(conn, identity, [139])
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 40 AND 138")
        _, docs, _ = select(conn, identity)
        assert len(docs) == 32 and docs[0][0] == "39:0"
    # A provider-rejected pinned row is still part of stock selection.
    lazy = command._chunk_gapfill_rows
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("DELETE FROM lcm_embedding_backfill_inflight")
        monkeypatch.setattr(command, "_chunk_gapfill_rows", frozen_gap)
        expected = select(conn, identity)
        monkeypatch.setattr(command, "_chunk_gapfill_rows", lazy)
        assert select(conn, identity) == expected
        assert expected[1][0][0] == "139:0" and expected[1][1][0] == "39:0"


def test_saved_pending_held_row_does_not_freeze_gap_discovery(local_store):
    engine, _ = local_store
    identity = seed(engine, 2110)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 10 AND 2109")
        conn.execute("UPDATE messages SET content='Deeper uncovered fact.' WHERE store_id=50")
        put(conn, identity, "chunk_policy_revision", CHUNK_POLICY_REVISION)
        put(conn, identity, "chunk_policy_revision_through", 4)
        scan_record(conn, identity, 9, pending=[5])
        hold(conn, identity, [5])
        _, docs, _ = select(conn, identity, record_progress=True)
        first_scan = get(conn, identity, "chunk_gapfill_scan")
        selected = {doc[0] for doc in docs}
        for _ in range(4):
            if "50:0" in selected:
                break
            _, docs, _ = select(conn, identity, record_progress=True)
            selected.update(doc[0] for doc in docs)
        assert "50:0" in selected, "held saved pending row must not freeze discovery"
        assert "5:0" not in selected
        assert first_scan == dict(revision=CHUNK_POLICY_REVISION, policy="conversational",
                                  high=2110, ceiling=110)


def test_empty_pending_held_row_under_zero_prefix_does_not_freeze_gap_discovery(local_store):
    engine, _ = local_store
    identity = seed(engine, 3501)
    publish(engine, identity, range(2, 1001))
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 1501 AND 2999 "
                     "OR store_id BETWEEN 3001 AND 3500")
        put(conn, identity, "chunk_policy_revision", CHUNK_POLICY_REVISION)
        put(conn, identity, "chunk_policy_revision_through", 1000)
        scan_record(conn, identity, 1000, pending=[])
        hold(conn, identity, [3000])
    expected = set(range(1001, 1501))
    for pass_index in range(40):
        result = command._chunk_backfill_run(engine._config, engine._store.db_path,
            apply=True, limit=32, retry_uncertain=False, policy="conversational", incremental=True)
        assert not result["error"] and not result["failed"], result
        with sqlite3.connect(engine._store.db_path) as conn:
            if pass_index == 0:
                first_scan = get(conn, identity, "chunk_gapfill_scan")
            covered = {r[0] for r in conn.execute("SELECT store_id FROM lcm_chunk_meta")}
        if expected <= covered:
            break
    assert expected <= covered, "held candidate must not hide deeper chunkable rows"
    assert 3000 not in covered
    assert first_scan == dict(revision=CHUNK_POLICY_REVISION, policy="conversational",
                              high=3501, ceiling=3001)


@pytest.mark.parametrize("legacy_pending", [False, True])
def test_zero_chunk_run_longer_than_window_advances_and_cycle_continues(local_store, legacy_pending):
    engine, _ = local_store
    identity = seed(engine, 2110)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 10 AND 2109")
        conn.execute("UPDATE messages SET content='Deeper uncovered fact.' WHERE store_id=50")
        if legacy_pending:
            scan_record(conn, identity, 9, pending=list(range(2, 10)))
    before = maintenance._scan_frontier(engine._store.db_path, command)
    state = {}
    results = [maintenance.run_incremental_embedding_pass(
        engine._store.db_path, engine._config, slice_state=state) for _ in range(4)]
    assert sum(r["vectors_written"] for r in results) or before != maintenance._scan_frontier(engine._store.db_path, command)
    assert all("no_progress" not in r["outcome"] for r in results), results
    assert state["continue"]
    for _ in range(16):
        result = maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config, slice_state=state)
        assert "no_progress" not in result["outcome"], result
        if not state["continue"]:
            break
    assert not state["continue"], result
    with sqlite3.connect(engine._store.db_path) as conn:
        assert {r[0] for r in conn.execute("SELECT store_id FROM lcm_chunk_meta")} == {1, *range(2, 10), 50, 2110}
        assert get(conn, identity, "chunk_policy_revision_through") == 2110


def test_remaining_check_never_false_zero(local_store):
    engine, _ = local_store
    identity = seed(engine, 2110)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 10 AND 2109")
    assert command._chunk_backfill_remaining(engine._store.db_path, identity,
        "conversational", 0, 0, incremental=True)[0] >= 1
    result = command._chunk_backfill_run(engine._config, engine._store.db_path,
        apply=True, limit=1, retry_uncertain=True, policy="conversational", incremental=True)
    assert result["remaining"] >= 1
    with sqlite3.connect(engine._store.db_path) as conn:
        record = get(conn, identity, "embedding_catchup:chunk")
        assert record["remaining"] is True
    provider = SimpleNamespace(provider_id="fastembed", model_id=engine._config.embedding_model)
    assert tools._lcm_recall_catchup_reason(engine, provider, "chunk") is not None


def test_certification_requires_complete_range_and_pending_read(local_store):
    engine, _ = local_store
    identity = seed(engine, 2001)
    pending = list(range(2000, 1, -1))
    with sqlite3.connect(engine._store.db_path) as conn:
        scan_record(conn, identity, 2001, pending=pending)
        put(conn, identity, "chunk_policy_revision", CHUNK_POLICY_REVISION)
        put(conn, identity, "chunk_policy_revision_through", 1)
        # First consumption abandons replay early: persisted pending is intact.
        selected = select(conn, identity, record_progress=True)
        assert selected[1][0][0] == "2000:0"
        assert get(conn, identity, "chunk_gapfill_scan")["pending"] == pending
        assert get(conn, identity, "chunk_policy_revision_through") == 1
    for _ in range(66):
        result = command._chunk_backfill_run(engine._config, engine._store.db_path,
            apply=True, limit=32, retry_uncertain=False, policy="conversational", incremental=True)
        assert not result["error"] and not result["failed"], result
        with sqlite3.connect(engine._store.db_path) as conn:
            covered = {r[0] for r in conn.execute("SELECT store_id FROM lcm_chunk_meta")}
            unread = set(pending) - covered
            saved = get(conn, identity, "chunk_gapfill_scan")["pending"]
            assert unread <= set(saved)
            through = get(conn, identity, "chunk_policy_revision_through")
            assert through <= min(unread, default=2002) - 1
        if not result["remaining"] and through == 2001:
            break
    assert covered == set(range(1, 2002))
    assert through == 2001


def test_ceiling_is_leading_and_written_before_yield(local_store):
    engine, _ = local_store
    identity = seed(engine, 120)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 70 AND 119")
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 40 AND 68")
        rows = command._chunk_gapfill_rows(conn, identity, (1, 120), "conversational", 2000,
                                           gap_page_size=32, record_progress=True)
        assert next(rows)[0] == 69
        saved = get(conn, identity, "chunk_gapfill_scan")
        assert saved is not None, "leading zero-chunk ceiling must be durable before yield"
        assert saved["ceiling"] == 70 and "pending" not in saved
        assert next(rows)[0] == 39
        assert get(conn, identity, "chunk_gapfill_scan") == saved
        rows.close()
        assert get(conn, identity, "chunk_gapfill_scan") == saved
        assert get(conn, identity, "chunk_policy_revision_through") is None


@pytest.mark.parametrize("pending", [[], [1, 16]], ids=["empty", "filtered"])
def test_empty_pending_saves_grown_ceiling_before_slice_stops(local_store, monkeypatch, pending):
    engine, _ = local_store
    identity = seed(engine, 16)
    publish(engine, identity, range(2, 16))
    with sqlite3.connect(engine._store.db_path) as conn:
        scan_record(conn, identity, 16, pending=pending)
        put(conn, identity, "chunk_policy_revision", CHUNK_POLICY_REVISION)
        put(conn, identity, "chunk_policy_revision_through", 16)
        conn.executemany(
            "INSERT INTO messages(store_id, session_id, role, content, timestamp) "
            "VALUES (?, 'seed', 'user', ?, 0)",
            [(i, "\t\n" if 82 <= i <= 145 else f"New fact {i}.") for i in range(17, 147)],
        )
    publish(engine, identity, [146])
    original = command.chunk_message
    gap_calls = []

    def counted(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_chunk_gapfill_rows":
            gap_calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(command, "chunk_message", counted)
    with sqlite3.connect(engine._store.db_path) as conn:
        _, docs, _ = select(conn, identity, record_progress=True)
        assert [row[0] for row in docs] == [f"{i}:0" for i in range(81, 49, -1)]
        assert set(range(82, 146)) <= set(gap_calls)
        assert get(conn, identity, "chunk_gapfill_scan") == {
            "revision": CHUNK_POLICY_REVISION, "policy": "conversational", "high": 146, "ceiling": 82,
        }
        assert get(conn, identity, "chunk_policy_revision_through") == 16
    publish(engine, identity, range(50, 82))
    gap_calls.clear()
    with sqlite3.connect(engine._store.db_path) as conn:
        _, docs, _ = select(conn, identity, record_progress=True)
        assert [row[0] for row in docs] == [f"{i}:0" for i in range(49, 17, -1)]
        assert gap_calls and not set(range(82, 146)).intersection(gap_calls)


def test_zero_chunk_prefix_rewalk_after_high_change(local_store, monkeypatch):
    engine, _ = local_store
    identity = seed(engine, 140)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 40 AND 139")
        scan_record(conn, identity, 140, ceiling=40)
    original = command.chunk_message
    lazy = command._chunk_gapfill_rows
    for turn in range(3):
        high = 141 + turn
        with sqlite3.connect(engine._store.db_path) as conn:
            conn.execute("INSERT INTO messages(store_id, session_id, role, content, timestamp) "
                         "VALUES (?, 'seed', 'user', 'New covered fact.', 0)", (high,))
        publish(engine, identity, [high])
        counts = []
        for discovery in (frozen_gap, lazy):
            calls = []
            def counted(*args, **kwargs):
                calls.append(args[0])
                return original(*args, **kwargs)
            monkeypatch.setattr(command, "chunk_message", counted)
            monkeypatch.setattr(command, "_chunk_gapfill_rows", discovery)
            with sqlite3.connect(engine._store.db_path) as conn:
                conn.execute("SAVEPOINT rewalk")
                _, docs, _ = select(conn, identity, record_progress=True)
                assert docs[0][0] == "39:0"
                assert get(conn, identity, "chunk_gapfill_scan")["high"] == high
                if discovery is frozen_gap:
                    assert 39 in get(conn, identity, "chunk_gapfill_scan")["pending"]
                if discovery is lazy:
                    assert get(conn, identity, "chunk_gapfill_scan")["ceiling"] == 40
                counts.append(len(calls))
                conn.execute("ROLLBACK TO rewalk")
                conn.execute("RELEASE rewalk")
        assert counts[1] <= counts[0]


def test_status_backlog_and_manual_paths_unchanged(local_store, monkeypatch):
    engine, _ = local_store
    identity = seed(engine, 2102, low=2)
    with sqlite3.connect(engine._store.db_path) as conn:
        before = list(conn.execute("SELECT * FROM metadata ORDER BY key"))
        assert command._chunk_backlog_estimate(conn, identity, "conversational") == "2000+ (gap-fill)"
        assert list(conn.execute("SELECT * FROM metadata ORDER BY key")) == before
        lazy = command._chunk_gapfill_rows
        sqls, results = [], []
        for discovery in (frozen_gap, lazy):
            sql = []
            conn.set_trace_callback(sql.append)
            monkeypatch.setattr(command, "_chunk_gapfill_rows", discovery)
            results.append(command._chunk_pending_rows(conn, identity, "conversational", 32))
            sqls.append(sql)
        assert sqls[0] == sqls[1] and results[0] == results[1]
        assert results[0][1][0][0] == "2101:0"
    assert "embedding_backlog_chunks: 2000+ (gap-fill)" in command._embedding_maintenance_status_lines(engine)
    # Actual dry-run call remains positional (test_config_reference uses this seam).
    with monkeypatch.context() as patch:
        patch.setattr(command, "_chunk_pending_rows", lambda *args: (7, [], {}))
        dry = command._chunk_backfill_run(engine._config, engine._store.db_path,
            apply=False, limit=32, retry_uncertain=False, policy="conversational")
        assert dry["pending"] == 7
    manual = command._chunk_backfill_run(engine._config, engine._store.db_path,
        apply=True, limit=32, retry_uncertain=False, policy="conversational", incremental=False)
    assert manual["embedded"] == 32 and manual["remaining"] == 2068


def test_mixed_upgrade_store_reaches_exact_coverage(local_store, monkeypatch):
    engine, _ = local_store
    identity = seed(engine, 160, first_text="A factual sentence with several details. " * 2300)
    with sqlite3.connect(engine._store.db_path) as conn:
        for i in range(2, 160):
            conn.execute("UPDATE messages SET role=?, content=? WHERE store_id=?",
                         ("user" if i % 2 else "assistant", f"Fact {i}. " * (1, 30, 250)[i % 3], i))
        expected = {c.chunk_id for row in conn.execute("SELECT store_id, role, content FROM messages")
                    for c in command.chunk_message(*row, policy="conversational") if c.char_end > c.char_start}
        original = list(conn.execute("SELECT * FROM messages ORDER BY store_id"))
    assert len([c for c in expected if c.startswith("1:")]) > 40
    publish(engine, identity, range(4, 161, 4))
    dispatched = []
    class Tracking(FakeProvider):
        def embed_document_batches(self, texts, **kwargs):
            dispatched.extend(texts)
            yield from super().embed_document_batches(texts, **kwargs)
    monkeypatch.setattr(command, "resolve_provider", lambda c, **kwargs: Tracking(c.embedding_provider, c.embedding_model))
    with sqlite3.connect(engine._store.db_path) as conn:
        initial = {r[0] for r in conn.execute("SELECT chunk_id FROM lcm_chunk_meta")}
    state = {}
    for _ in range(160):
        result = maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config, slice_state=state)
        assert result["outcome"] != "error" and "no_progress" not in result["outcome"], result
        if not state["continue"]:
            break
    assert not state["continue"], result
    with sqlite3.connect(engine._store.db_path) as conn:
        actual = {r[0] for r in conn.execute("SELECT chunk_id FROM lcm_chunk_meta")}
        assert actual == expected
        assert len(dispatched) == len(actual - initial)  # No duplicate dispatch.
        assert get(conn, identity, "chunk_policy_revision") == CHUNK_POLICY_REVISION
        assert get(conn, identity, "chunk_policy_revision_through") == 160
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert list(conn.execute("SELECT * FROM messages ORDER BY store_id")) == original


def test_pending_zero_rows_do_not_extend_ceiling_across_unread_range(local_store):
    engine, _ = local_store
    identity = seed(engine, 2110)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET content='\t\n' WHERE store_id BETWEEN 2 AND 2109")
        conn.execute("UPDATE messages SET content='Unclassified deeper fact.' WHERE store_id=50")
        scan_record(conn, identity, 9, pending=list(range(2, 10)))
        rows = list(command._chunk_gapfill_rows(conn, identity, (1, 2110), "conversational", 2000,
                                               gap_page_size=32, record_progress=True))
        assert not rows
        # Only 2109..110 were classified in the new range; old zero-chunk
        # pending rows cannot certify the unread 109..10 interval.
        assert get(conn, identity, "chunk_gapfill_scan")["ceiling"] == 110
        assert get(conn, identity, "chunk_policy_revision_through") is None
        assert select(conn, identity)[1][0][0] == "50:0"
