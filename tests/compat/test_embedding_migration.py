"""Old vectors remain byte-identical and are never sent to the provider twice."""

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.compat.helpers import PREVIOUS_GA, run_old
from tests.harness import make_engine
from tests.compat.storage_fixtures import (
    VECTOR_TABLES, assert_calls_once, assert_complete_vectors, assert_preserved, bind,
    embed_all, install_fakes, plugin, seed_history, storage_config, table_hashes, vector_ids, warmup,
)

OLD_EMBEDDED_STORE = r'''
from tests.compat.storage_fixtures import (
    assert_complete_vectors, bind, embed_all, install_fakes, seed_history, storage_config, warmup,
)
provider = install_fakes()
engine = LCMEngine(config=storage_config(Path(db).parent, embeddings=True, dtype=data, small=True),
                   hermes_home=home)
try:
    bind(engine)
    seed_history(engine, "previous-ga", turns=8)
    warmup(engine)
    embed_all(engine)
    assert_complete_vectors(engine)
    assert provider.documents
    print(json.dumps({"documents": provider.documents}))
finally:
    engine.shutdown()
'''


@pytest.mark.parametrize("source_reader", ("v0.26.2", PREVIOUS_GA))
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_previous_ga_vectors_survive_upgrade(
    tmp_path: Path, old_tree: Callable[[str], Path], host_root: Path,
    monkeypatch: pytest.MonkeyPatch, dtype: str, source_reader: str,
) -> None:
    from collections import Counter
    import sqlite3

    old = run_old(old_tree(source_reader), tmp_path, host_root, OLD_EMBEDDED_STORE, data=dtype)
    before = table_hashes(tmp_path, VECTOR_TABLES)
    assert all(before[table] for table in VECTOR_TABLES if dtype == "int8" or not table.endswith("binary"))
    with sqlite3.connect(tmp_path / "lcm.db") as conn:
        (identity_hash,) = [row[0] for row in conn.execute(
            "SELECT DISTINCT identity_hash FROM lcm_chunk_meta")]
        revision_gaps = [row[0] for row in conn.execute(
            "SELECT content FROM messages AS m WHERE lower(trim(role)) IN ('user', 'assistant') "
            "AND NOT EXISTS (SELECT 1 FROM lcm_chunk_meta AS c "
            "WHERE c.store_id = m.store_id AND c.identity_hash = ?)", (identity_hash,))
            if row[0] and row[0].strip()]
    provider = install_fakes(monkeypatch)
    config = storage_config(tmp_path, embeddings=True, dtype=dtype, small=True)
    engine = make_engine(tmp_path, embeddings_enabled=config.embeddings_enabled,
                         embedding_provider=config.embedding_provider, embedding_model=config.embedding_model,
                         embedding_storage_dtype=config.embedding_storage_dtype,
                         fresh_tail_count=config.fresh_tail_count, leaf_chunk_tokens=config.leaf_chunk_tokens,
                         large_output_externalization_path=config.large_output_externalization_path)
    try:
        bind(engine)
        assert engine.drain_embedding_maintenance(timeout=10)
        assert Counter(provider.documents) == Counter(revision_gaps), (
            "an upgrade embeds only revision gaps, never old content")
        assert not set(provider.documents).intersection(old["documents"])
        assert_preserved(tmp_path, {table: rows for table, rows in before.items()
                                    if table != "lcm_embedding_profile"})
        assert_complete_vectors(engine)
        revision = engine._store.connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (f"chunk_policy_revision:{identity_hash}",),
        ).fetchone()
        assert revision is not None and int(revision[0]) == plugin("chunking").CHUNK_POLICY_REVISION
        # assert_calls_once below measures only dispatch after this vector snapshot.
        provider.documents.clear()
        previous = vector_ids(engine)
        seed_history(engine, "new-candidate", turns=8)
        assert engine.drain_embedding_maintenance(timeout=10)
        embed_all(engine)
        assert_complete_vectors(engine)
        current = vector_ids(engine)
        assert all(now > was for now, was in zip(current, previous))
        assert not set(provider.documents).intersection(old["documents"])
        assert_calls_once(provider, previous, engine)
        assert_preserved(tmp_path, {table: rows for table, rows in before.items()
                                    if table != "lcm_embedding_profile"})
        sent = list(provider.documents)
        embed_all(engine)
        assert provider.documents == sent, "catch-up must be idempotent"
    finally:
        engine.shutdown()


def test_distinct_sources_with_identical_text_embed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = install_fakes(monkeypatch)
    # Keep all dispatch in the explicit backfill so its source delta is exact.
    monkeypatch.setattr(plugin("engine").LCMEngine, "_schedule_embedding_maintenance", lambda *args: None)
    engine = plugin("engine").LCMEngine(
        config=storage_config(tmp_path, embeddings=True), hermes_home=str(tmp_path / "home"))
    try:
        bind(engine)
        warmup(engine)
        embed_all(engine)  # Materialize both empty vector tables.
        previous = vector_ids(engine)
        assert previous == (set(), set())
        text = "The same text belongs to two distinct source messages. " * 20
        engine.ingest([{"role": role, "content": text, "timestamp": float(index)}
                       for index, role in enumerate(("user", "assistant"))])
        embed_all(engine)
        current = vector_ids(engine)
        assert not current[0] and len(current[1]) == 2
        assert_calls_once(provider, previous, engine)
        assert provider.documents == [text, text]
        embed_all(engine)
        assert provider.documents == [text, text], "repeat backfill must not re-dispatch either source"
    finally:
        engine.shutdown()
