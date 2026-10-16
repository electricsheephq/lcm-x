"""Old vectors remain byte-identical and are never sent to the provider twice."""

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.compat.helpers import run_old
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


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_previous_ga_vectors_survive_upgrade(
    tmp_path: Path, old_tree: Callable[[str], Path], host_root: Path,
    monkeypatch: pytest.MonkeyPatch, dtype: str,
) -> None:
    old = run_old(old_tree("v0.26.2"), tmp_path, host_root, OLD_EMBEDDED_STORE, data=dtype)
    before = table_hashes(tmp_path, VECTOR_TABLES)
    assert all(before[table] for table in VECTOR_TABLES if dtype == "int8" or not table.endswith("binary"))
    provider = install_fakes(monkeypatch)
    config = storage_config(tmp_path, embeddings=True, dtype=dtype, small=True)
    engine = plugin("engine").LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        bind(engine)
        assert engine.drain_embedding_maintenance(timeout=10)
        assert provider.documents == [], "opening a fully embedded store must not re-embed old content"
        assert table_hashes(tmp_path, VECTOR_TABLES) == before
        assert_complete_vectors(engine)
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
