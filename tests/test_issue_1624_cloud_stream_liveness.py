"""Cloud metadata proof, interruptible discovery and arrival-every-call liveness."""
import importlib
import threading
from types import SimpleNamespace

import pytest

from tests.test_issue_1536_chunk_revision import (
    LONG, add_rows, command, counts, metadata, store as store,
)
from tests.test_issue_1543_coverage_marker import ROWS, certify, marker, older_walk, run


@pytest.mark.parametrize("gap", [None, 3])
def test_cloud_maintenance_certifies_only_contiguous_metadata(store, monkeypatch, gap):
    config, vectors, _, texts = store
    config.embedding_provider = "voyage"
    vectors.register_profile("model-a", "voyage", 2, task="chunk")
    vectors.register_profile("model-a", "voyage", 2)
    identity = command.EmbeddingIdentity.canonical("voyage", "model-a", "", 2, "float32", "little", "chunk")
    cloud = config, vectors, identity, texts
    add_rows(cloud, [(i, "user", LONG) for i in range(1, 6)], set(range(1, 6)) - {gap})
    metadata(cloud, "chunk_policy_revision", 2)
    maintenance = importlib.import_module("hermes_lcm.embedding_maintenance")
    providers = importlib.import_module("hermes_lcm.embedding_provider")
    monkeypatch.setattr(providers, "probe_provider_availability", lambda _: {"available": True})
    monkeypatch.setattr(command, "_embedding_backfill_summary_run", lambda *a, **k: {
        "status": "complete", "embedded": 0, "selected": 0, "privacy_withheld": 0,
        "skipped": [], "failed": [], "stop_reason": None, "error": None,
    })
    before = counts(cloud)
    with monkeypatch.context() as guarded:
        guarded.setattr(command, "chunk_message", lambda *a, **k: pytest.fail("cloud validation chunked text"))
        guarded.setattr(command, "resolve_provider", lambda *a, **k: pytest.fail("cloud validation dispatched"))
        assert maintenance.run_incremental_embedding_pass(config.database_path, config) != "error"
    assert marker(cloud) == {"from": 1, "through": 2 if gap else 5, "policy": "conversational"}
    assert counts(cloud) == before and not texts
    monkeypatch.setattr(command, "_embedding_current_profile", lambda _: None)
    snapshot = list(vectors.connection.iterdump())
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
    assert f"embedding_backlog_chunks: {1 if gap else 0}" in "\n".join(command._embedding_maintenance_status_lines(engine))
    assert list(vectors.connection.iterdump()) == snapshot


def test_stream_discovery_first_yield_and_cancellation(store, monkeypatch):
    add_rows(store, [(i, "user", "synthetic archive " * 240) for i in range(1, 2003)], {1, 2002})
    metadata(store, "chunk_policy_revision", 2)
    original = command.chunk_message
    calls = []
    stop = threading.Event()

    def counted(*args, **kwargs):
        calls.append(args[0])
        stop.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(command, "chunk_message", counted)
    rows = command._chunk_gapfill_rows(store[1].connection, store[2].identity_hash,
                                      (1, 2002), "conversational", 2000)
    assert next(rows)[0] == 2001
    assert calls == [2001], "first yield must validate only its own row"
    rows.close()
    calls.clear()
    stop.clear()
    result = command._chunk_backfill_run(store[0], store[0].database_path, apply=True, limit=32,
                                         retry_uncertain=False, policy="conversational",
                                         incremental=True, stop_event=stop)
    assert result["stop_reason"] == "process_exit", result
    assert result["embedded"] == 0
    assert len(calls) <= 8, "cancellation must be checked during discovery"


def test_low_band_liveness_with_arrivals_before_every_writable_call(store):
    add_rows(store, ROWS, set())
    run(store, limit=2)
    run(store, limit=1)
    older_walk(store, {1, 3})
    for call in range(12):
        nxt = 8 + call * 2
        add_rows(store, [(nxt, "user", LONG), (nxt + 1, "assistant", LONG)], set())
        run(store)
        if set(counts(store)) == set(range(1, nxt + 2)) and marker(store) == {
                "from": 1, "through": nxt + 1, "policy": "conversational"}:
            break
    else:
        pytest.fail(f"low band did not converge after {call + 1} calls: {marker(store)}")


def test_low_walk_cannot_survive_parent_metadata_invalidation(store):
    add_rows(store, ROWS, {6, 7})
    metadata(store, "chunk_policy_revision", 2)
    command._record_chunk_policy_revision(store[1].connection, store[2].identity_hash, (6, 7), "conversational")
    older_walk(store, {1, 3})
    conn, identity = store[1].connection, store[2].identity_hash
    state = {"coverage": (6, 7, "conversational")}
    rows = command._chunk_gapfill_rows(conn, identity, (1, 5), "conversational", 1,
                                      record_progress=True, gap_scan_state=state)
    assert next(rows)[0] == 5
    rows.close()
    command._record_chunk_gapfill_walk(conn, identity, state)
    low_key = f"chunk_gapfill_scan:{identity}:low"
    assert conn.execute("SELECT 1 FROM metadata WHERE key=?", (low_key,)).fetchone()
    conn.execute("DELETE FROM lcm_chunk_meta WHERE store_id=1")
    assert metadata(store, "chunk_gapfill_scan") is None
    command._record_chunk_gapfill_progress(conn, identity, state, {"high": 5},
                                          [(f"chunk_policy_revision_through:{identity}", "999")])
    assert metadata(store, "chunk_policy_revision_through") is None
    resumed = list(command._chunk_gapfill_rows(conn, identity, (1, 5), "conversational", 32,
                    record_progress=True, gap_scan_state={"coverage": (6, 7, "conversational")}))
    assert 1 in [row[0] for row in resumed]


def test_first_disjoint_band_call_keeps_upper_selection(store):
    add_rows(store, ROWS, {6, 7})
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (6, 7))
    older_walk(store, {1, 3})
    add_rows(store, [(i, "user", LONG) for i in (8, 9, 10)], {10})
    _, selected, _ = command._chunk_pending_rows(store[1].connection, store[2].identity_hash,
        "conversational", 1, incremental=True, record_progress=True, gap_scan_state={})
    assert selected[0][0] == "9:0", "only an actual deferred low band authorizes alternation"


def test_parent_invalidation_before_first_low_insert_terminates(store):
    add_rows(store, ROWS, {6, 7})
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (6, 7))
    older_walk(store, {1, 3})
    conn, identity = store[1].connection, store[2].identity_hash
    attempts = []

    class InvalidateBeforeCAS:
        def execute(self, sql, parameters=()):
            if sql.startswith("INSERT INTO metadata(key,value) SELECT") and parameters[0].endswith(":low"):
                attempts.append(1)
                assert len(attempts) == 1, "a rejected parent fence must not spin"
                conn.execute("DELETE FROM lcm_chunk_meta WHERE store_id=1")
            return conn.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(conn, name)

    state = {"coverage": (6, 7, "conversational")}
    assert list(command._chunk_gapfill_rows(InvalidateBeforeCAS(), identity, (1, 5),
                "conversational", 32, record_progress=True, gap_scan_state=state)) == []
    assert attempts == [1] and state["unproven"]
    assert metadata(store, "chunk_policy_revision_through") is None
