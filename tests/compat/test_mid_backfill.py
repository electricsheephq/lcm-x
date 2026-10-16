"""An old writer can interleave with incomplete derived-index maintenance."""

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.compat.helpers import config_for, run_old
from tests.compat.storage_fixtures import (
    SESSION, VECTOR_TABLES, assert_calls_once, assert_preserved, bind, embed_all,
    finish_embeddings, install_fakes, plugin, seed_history, storage_config,
    table_hashes, vector_ids, warmup,
)

OLD_STEM_INTERLUDE = r'''
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    engine.on_session_start("compat-storage", platform="cli", context_length=200_000,
                            conversation_id="compat-conversation")
    rows = engine._store.get_session_messages("compat-storage")
    assert len(rows) == 1501
    engine.ingest([{"role": "user", "content": "Old writer visited deployments", "timestamp": 9999.0}])
    after = engine._store.get_session_messages("compat-storage")
    inserted = [row["store_id"] for row in after if row["store_id"] not in {r["store_id"] for r in rows}]
    assert len(inserted) == 1
    print(json.dumps({"inserted": inserted[0]}))
finally:
    engine.shutdown()
'''

OLD_EMBEDDING_INTERLUDE = r'''
from tests.compat.storage_fixtures import bind, install_fakes, seed_history, storage_config
provider = install_fakes()
engine = LCMEngine(config=storage_config(Path(db).parent, embeddings=True, small=True), hermes_home=home)
try:
    bind(engine)
    conn = engine._store.connection
    assert conn.execute("SELECT count(*) FROM lcm_embedding_vectors").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM lcm_chunk_vectors").fetchone()[0] == 2
    seed_history(engine, "old-during-backfill", turns=8)
    assert provider.documents == []
    print(json.dumps({"messages": len(engine._store.get_session_messages("compat-storage"))}))
finally:
    engine.shutdown()
'''


def test_stem_backfill_survives_old_writer(
    tmp_path: Path, old_tree: Callable[[str], Path], host_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    boot = plugin("db_bootstrap")
    store_mod = plugin("store")
    monkeypatch.setattr(boot, "dispatch_background_stem_backfill", lambda *args: None)
    config = config_for(tmp_path)
    # The existing pre-stem seam leaves original messages outside the new index.
    with monkeypatch.context() as setup:
        setup.setattr(store_mod, "ensure_message_stem_fts", lambda *args: None)
        engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
        bind(engine)
        engine.ingest([{"role": "user", "content": f"Visited deployment {index}", "timestamp": float(index)}
                       for index in range(1501)])
    try:
        conn = engine._store.connection
        spec = store_mod.build_message_stem_fts_spec()
        boot.ensure_message_stem_fts(conn, spec)

        def interrupt_after_batch(_: float) -> None:
            raise RuntimeError("synthetic interruption after one committed batch")

        with monkeypatch.context() as interrupted:
            interrupted.setattr(boot.time, "sleep", interrupt_after_batch)
            boot._run_stem_backfill(str(tmp_path / "lcm.db"), spec, "first-pass")
        assert boot._stem_metadata(conn, "fts_stem_state") == "backfilling"
        assert conn.execute("SELECT count(*) FROM messages_fts_stem_docsize").fetchone()[0] == 1000
        before = table_hashes(tmp_path, ("messages",))
    finally:
        engine.shutdown()
    report = run_old(old_tree("v0.26.2"), tmp_path, host_root, OLD_STEM_INTERLUDE)
    engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        bind(engine)
        conn = engine._store.connection
        assert boot._stem_metadata(conn, "fts_stem_state") == "backfilling"
        monkeypatch.setattr(boot.time, "sleep", lambda _: None)
        boot._run_stem_backfill(str(tmp_path / "lcm.db"), spec, "resumed-pass")
        boot.join_background_stem_backfills(timeout=10)
        expected = [row[0] for row in conn.execute("SELECT store_id FROM messages ORDER BY store_id")]
        indexed = [row[0] for row in conn.execute("SELECT id FROM messages_fts_stem_docsize ORDER BY id")]
        assert indexed == expected and len(indexed) == 1502
        assert report["inserted"] > 1501 and report["inserted"] in indexed
        conn.execute("INSERT INTO messages_fts_stem(messages_fts_stem, rank) VALUES('integrity-check', 1)")
        conn.commit()
        assert boot._stem_metadata(conn, "fts_stem_state") == "ready"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert_preserved(tmp_path, before)
    finally:
        engine.shutdown()


def test_partial_embeddings_survive_old_writer(
    tmp_path: Path, old_tree: Callable[[str], Path], host_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch)
    config = storage_config(tmp_path, embeddings=True, small=True)
    with monkeypatch.context() as setup:
        setup.setattr(plugin("engine").LCMEngine, "_schedule_embedding_maintenance", lambda *args: None)
        engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
        try:
            bind(engine)
            seed_history(engine, "partial-candidate", turns=8)
            warmup(engine)
            embed_all(engine, limit=2)
            previous = vector_ids(engine)
            assert all(len(ids) == 2 for ids in previous)
            conn = engine._store.connection
            assert conn.execute("SELECT count(*) FROM summary_nodes WHERE depth=0").fetchone()[0] > 2
            assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] > 2
            before = table_hashes(tmp_path, VECTOR_TABLES)
        finally:
            engine.shutdown()
    report = run_old(old_tree("v0.26.2"), tmp_path, host_root, OLD_EMBEDDING_INTERLUDE)
    assert table_hashes(tmp_path, VECTOR_TABLES) == before
    provider = install_fakes(monkeypatch)
    engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        bind(engine)
        finish_embeddings(engine)
        assert len(engine._store.get_session_messages(SESSION)) == report["messages"]
        assert_calls_once(provider, previous, engine)
        assert_preserved(tmp_path, {table: rows for table, rows in before.items()
                                    if table != "lcm_embedding_profile"})
        sent = list(provider.documents)
        finish_embeddings(engine)
        assert provider.documents == sent
    finally:
        engine.shutdown()
