"""Gap proofs upgrade once; the pinned pre-walk reader keeps writing safely (I5)."""
import sqlite3

import pytest

from hermes_lcm.db_bootstrap import ensure_chunk_tables
from tests.compat.helpers import PREVIOUS_GA, READERS, run_old

BASE = "f3cc8de918ba14c21ae246099652762a26255554"
OLD_CHUNKS = r'''
sys.path.insert(0, old)
from hermes_lcm import command
from hermes_lcm.vector_store import VectorStore
from tests.test_issue_1014_incremental_embedding import FakeProvider
assert Path(command.__file__).resolve().parent == Path(old).resolve()
config = LCMConfig(database_path=db, embeddings_enabled=True, embedding_provider="fastembed",
                   embedding_model="fixture-local", temporal_rollups_enabled=False)
command.resolve_provider = lambda c, **kw: FakeProvider(c.embedding_provider, c.embedding_model)
engine = LCMEngine(config=config, hermes_home=home)
def chunk_identity():
    return command.EmbeddingIdentity.canonical(config.embedding_provider, config.embedding_model, "", 2,
                                               "float32", "little", "chunk")
def publish(engine, identity, ids):
    vectors = VectorStore(db, config=config)
    try:
        for i in ids:
            row = vectors.connection.execute("SELECT store_id,role,content FROM messages WHERE store_id=?", (i,)).fetchone()
            for chunk in command.chunk_message(*row, policy="conversational"):
                vectors.record_chunk_embedding(chunk.chunk_id, config.embedding_model, [1.0, 0.0], store_id=i,
                    chunk_index=chunk.chunk_index, char_start=chunk.char_start, char_end=chunk.char_end,
                    token_estimate=chunk.token_estimate, identity=chunk_identity())
    finally:
        vectors.close()
try:
    vectors = VectorStore(db, config=config)
    vectors.register_profile(config.embedding_model, "fastembed", 2, task="chunk")
    vectors.close()
    if mode == "write":
        for i in range(80):
            engine._store.append("seed", {"role": "user", "content": f"Synthetic rollback fact {i}."})
        identity = chunk_identity().identity_hash
        publish(engine, identity, [1, 80])
        c = engine._store.connection
        c.execute("INSERT INTO metadata VALUES(?, 'legacy')", (f"chunk_gapfill_scan:{identity}",))
        c.execute("INSERT INTO metadata VALUES(?, '40')", (f"chunk_policy_revision_through:{identity}",))
        c.commit()
    else:
        identity = engine._store.connection.execute("SELECT identity_hash FROM lcm_chunk_meta LIMIT 1").fetchone()[0]
        for incremental in (True, False):
            result = command._chunk_backfill_run(config, db, apply=True, limit=100,
                retry_uncertain=False, policy="conversational", incremental=incremental)
            assert not result["error"] and not result["failed"], result
        publish(engine, identity, [40])  # replacement invalidates the newer trigger proof
        # Exercise session deletion after a fresh old-reader open, independently
        # of its long-lived FTS connection during backfill schema setup.
        engine.shutdown()
        engine = LCMEngine(config=config, hermes_home=home)
        engine._store.append("disposable", {"role": "user", "content": "Disposable synthetic fact."})
        result = command._chunk_backfill_run(config, db, apply=True, limit=100,
            retry_uncertain=False, policy="conversational", incremental=True)
        assert not result["error"] and not result["failed"], result
        assert engine._store.delete_session_messages("disposable") == 1
        for _ in range(5):
            result = command._chunk_backfill_run(config, db, apply=True, limit=100,
                retry_uncertain=False, policy="conversational", incremental=True)
            assert not result["error"] and not result["failed"], result
        assert command._chunk_backfill_remaining(db, identity, "conversational", 0, 0, incremental=False)[0] == 0
    print(json.dumps({"identity": identity, "rows": len(engine._store.get_session_messages("seed"))}))
finally:
    engine.shutdown()
'''


def upgrade(path, identity):
    with sqlite3.connect(path) as conn:
        before = list(conn.execute("SELECT * FROM messages ORDER BY store_id"))
        ensure_chunk_tables(conn)
        triggers = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert {"lcm_chunk_gapfill_delete", "lcm_chunk_gapfill_update", "lcm_chunk_gapfill_message"} <= triggers
        assert not conn.execute("SELECT 1 FROM metadata WHERE key IN (?,?)",
                                (f"chunk_gapfill_scan:{identity}", f"chunk_policy_revision_through:{identity}")).fetchall()
        conn.execute("INSERT INTO metadata VALUES(?, 'current')", (f"chunk_gapfill_scan:{identity}",))
        conn.commit()
        ensure_chunk_tables(conn)
        assert conn.execute("SELECT value FROM metadata WHERE key=?",
                            (f"chunk_gapfill_scan:{identity}",)).fetchone()[0] == "current"
        assert list(conn.execute("SELECT * FROM messages ORDER BY store_id")) == before


def test_gapwalk_forward(tmp_path, old_tree, host_root):
    old = old_tree("gapwalk-base", {"gapwalk-base": BASE})
    report = run_old(old, tmp_path, host_root, OLD_CHUNKS, mode="write")
    upgrade(tmp_path / "lcm.db", report["identity"])


@pytest.mark.parametrize("reader", ["gapwalk-base", "v0.28.1", PREVIOUS_GA])
def test_gapwalk_rollback(tmp_path, old_tree, host_root, reader):
    old = old_tree(reader, {**READERS, "gapwalk-base": BASE,
                            "v0.28.1": "77b2b4f795fb6a4f84c55abf6d0441da4469971e"})
    original = run_old(old, tmp_path, host_root, OLD_CHUNKS, mode="write")
    upgrade(tmp_path / "lcm.db", original["identity"])
    assert run_old(old, tmp_path, host_root, OLD_CHUNKS) == original
    with sqlite3.connect(tmp_path / "lcm.db") as conn:
        before = list(conn.execute("SELECT * FROM messages ORDER BY store_id"))
        ensure_chunk_tables(conn)
        ensure_chunk_tables(conn)
        assert list(conn.execute("SELECT * FROM messages ORDER BY store_id")) == before
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
