"""Shipped rollup defaults survive an older reader's additional compaction."""

from collections.abc import Callable
from pathlib import Path
import sqlite3

import pytest

from tests.compat.helpers import PREVIOUS_GA, run_old
from tests.compat.storage_fixtures import (
    SESSION, VECTOR_TABLES, assert_complete_vectors, assert_preserved, bind, embed_all,
    install_fakes, payload_hashes, plugin, rebuild_rollup, seed_history, storage_config,
    table_hashes, warmup,
)

OLD_INGEST_AND_COMPRESS = r'''
from tests.compat.storage_fixtures import (
    SESSION, bind, install_fakes, seed_history, storage_config,
)
install_fakes()
config = storage_config(Path(db).parent, fleet=data == "fleet", embeddings=data == "fleet")
engine = LCMEngine(config=config, hermes_home=home)
try:
    bind(engine)
    before = engine._store.get_session_messages(SESSION)
    nodes = len(engine._dag.get_session_nodes(SESSION))
    assert before and nodes
    seed_history(engine, "old-reader")
    assert len(engine._store.get_session_messages(SESSION)) > len(before)
    assert len(engine._dag.get_session_nodes(SESSION)) > nodes
    assert engine._store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(json.dumps({"messages": len(engine._store.get_session_messages(SESSION))}))
finally:
    engine.shutdown()
'''


# Columns an invalidation (rollup_store: status='stale', generation+1, lease cleared, error) may change.
INVALIDATION_FIELDS = {"status", "generation", "lease_expires_at", "lease_nonce", "error"}


def rollup_row_columns(database):
    with sqlite3.connect(database) as conn:  # its own read connection, never the engine's
        names = [row[1] for row in conn.execute('PRAGMA table_info("lcm_rollups")')]
        return {row[0]: dict(zip(names, row)) for row in conn.execute(f'SELECT {", ".join(names)} FROM lcm_rollups')}


def profile_rows(database):
    """Embedding profiles by rowid, with data_version split from the identity fields."""
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        rows = {row["profile_rowid"]: dict(row) for row in conn.execute(
            "SELECT rowid AS profile_rowid, * FROM lcm_embedding_profile ORDER BY rowid")}
    return ({key: {name: value for name, value in row.items() if name != "data_version"}
             for key, row in rows.items()},
            {key: row["data_version"] for key, row in rows.items()})


@pytest.mark.parametrize("source_reader", ("v0.26.2", PREVIOUS_GA))
@pytest.mark.parametrize("policy", ["defaults", "fleet"])
def test_shipped_rollups_rollback(
    tmp_path: Path, old_tree: Callable[[str], Path], host_root: Path,
    monkeypatch: pytest.MonkeyPatch, policy: str, source_reader: str,
) -> None:
    install_fakes(monkeypatch)
    config = storage_config(tmp_path, fleet=policy == "fleet", embeddings=policy == "fleet")
    assert config.temporal_rollups_enabled
    # Leave defaults untouched in the first arm; only isolate filesystem destinations.
    engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        bind(engine)
        seed_history(engine, "candidate")
        rebuild_rollup(engine)
        if policy == "fleet":
            assert engine.drain_embedding_maintenance(timeout=10)
            warmup(engine)
            embed_all(engine)
            assert_complete_vectors(engine)
        before = table_hashes(tmp_path, ("messages", "summary_nodes", "summary_node_provenance"))
        payloads = payload_hashes(tmp_path)
        assert payloads
        rollups = table_hashes(tmp_path, ("lcm_rollup_sources",))
        rollup_rows = table_hashes(tmp_path, ("lcm_rollups",))["lcm_rollups"]
        with sqlite3.connect(tmp_path / "lcm.db") as conn:
            ids = {row[0] for row in conn.execute("SELECT rollup_id FROM lcm_rollups")}
            rollup_columns = rollup_row_columns(tmp_path / "lcm.db")
        vectors = table_hashes(tmp_path, VECTOR_TABLES) if policy == "fleet" else {}
        profiles, versions = profile_rows(tmp_path / "lcm.db") if policy == "fleet" else ({}, {})
    finally:
        engine.shutdown()
    report = run_old(old_tree(source_reader), tmp_path, host_root, OLD_INGEST_AND_COMPRESS, data=policy)
    assert_preserved(tmp_path, before)
    assert_preserved(tmp_path, rollups)
    # Newer writers bump the profile's data_version while embedding their new rows;
    # every other profile field must survive unchanged.
    assert_preserved(tmp_path, {table: rows for table, rows in vectors.items()
                                if source_reader == "v0.26.2" or table != "lcm_embedding_profile"})
    if source_reader != "v0.26.2" and profiles:
        profiles_after, versions_after = profile_rows(tmp_path / "lcm.db")
        assert profiles_after == profiles
        assert all(versions_after[key] >= versions[key] for key in versions)
        assert any(versions_after[key] > versions[key] for key in versions)
    assert payloads.items() <= payload_hashes(tmp_path).items()
    # Both versions understand these rows: publication invalidates them, rather
    # than deleting them. Freeze automatic rebuilding to inspect that durable debt.
    monkeypatch.setattr(plugin("engine").LCMEngine, "_schedule_rollup_maintenance", lambda *args: None)
    engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        bind(engine)
        assert len(engine._store.get_session_messages(SESSION)) == report["messages"]
        assert_preserved(tmp_path, before)
        assert_preserved(tmp_path, rollups)
        assert payloads.items() <= payload_hashes(tmp_path).items()
        rollup_store = plugin("rollup_store").RollupStore(tmp_path / "lcm.db")
        try:
            rollup_store.drain_invalidations()
        finally:
            rollup_store.close()
        conn = engine._store.connection
        rows = dict(conn.execute("SELECT rollup_id, status FROM lcm_rollups"))
        assert ids <= rows.keys()
        assert any(rows[rollup_id] == "stale" for rollup_id in ids)
        assert all(rows[rollup_id] in {"ready", "stale"} for rollup_id in ids)
        after = table_hashes(tmp_path, ("lcm_rollups",))["lcm_rollups"]
        assert all(after.get(row_id) == digest or rows.get(row_id) == "stale"
                   for row_id, digest in rollup_rows.items())
        # Invalidation may only touch its own fields: the durable summary, token count, fingerprint and
        # build time of a row that went stale must survive unchanged.
        # A row that was not yet ready may legitimately be built by the old release in between.
        changed = rollup_row_columns(tmp_path / "lcm.db")
        invalidated = [row_id for row_id, columns in rollup_columns.items()
                       if columns["status"] == "ready" and rows.get(row_id) == "stale"]
        for row_id in invalidated:
            kept = {key: value for key, value in rollup_columns[row_id].items() if key not in INVALIDATION_FIELDS}
            assert {key: changed[row_id][key] for key in kept} == kept, f"rollup {row_id} lost durable fields"
        assert invalidated, "no ready rollup was invalidated, so the durable-field check did not run"
        assert conn.execute("SELECT count(*) FROM lcm_rollup_sources s LEFT JOIN summary_nodes n "
                            "ON s.node_id=n.node_id LEFT JOIN lcm_rollups r ON s.rollup_id=r.rollup_id "
                            "WHERE n.node_id IS NULL OR r.rollup_id IS NULL").fetchone()[0] == 0
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        if policy == "fleet":
            assert engine.drain_embedding_maintenance(timeout=10)
            embed_all(engine)
            assert_complete_vectors(engine)
    finally:
        engine.shutdown()
