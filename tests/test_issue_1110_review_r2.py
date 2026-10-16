"""#1110 round 2: reviewer probes as asserted, offline regressions."""

from __future__ import annotations

import logging
import threading
import time

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.embedding_maintenance as maintenance_mod
import hermes_lcm.embedding_provider as provider_mod
from hermes_lcm.dag import SummaryNode
from hermes_lcm.embedding_provider import FastembedProvider, ProviderNotWarmedUp
from hermes_lcm.vector_store import VectorStore
from tests.test_issue_1014_incremental_embedding import (
    FILLER,
    SESSION,
    UnavailableProvider,
    _make_engine,
    home as home,
    providers as providers,
)


@pytest.mark.parametrize("boundary", ["high", "low"])
def test_revision_one_inflight_boundary_embeds_each_chunk_once(tmp_path, home, providers, boundary):
    """REGRESSION F1: gap-fill must not duplicate either re-read boundary."""
    engine = _make_engine(tmp_path, home)
    long_text = ". ".join(f"Sentence{index} {FILLER}" for index in range(12)) + "."
    try:
        contents = [FILLER, long_text] if boundary == "high" else [long_text, FILLER]
        for content in contents:
            engine._store.append(SESSION, {"role": "user", "content": content})
        store = VectorStore(engine._config.database_path, config=engine._config)
        try:
            conn = store.connection
            messages = conn.execute("SELECT store_id, content FROM messages ORDER BY store_id").fetchall()
            seed = next(row[0] for row in messages if row[1] == FILLER)
            long_id = next(row[0] for row in messages if row[1] == long_text)
            identity = store._identity_from_profile(command_mod._chunk_current_profile(conn))
            chunk = command_mod.chunk_message(seed, "user", FILLER)[0]
            store.record_chunk_embedding(
                chunk.chunk_id, "model-a", [1.0, 0.0], store_id=seed,
                chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=identity,
            )
            command_mod._ensure_inflight_table(conn)
            conn.execute(
                "INSERT INTO lcm_embedding_backfill_inflight "
                "(embedded_id, identity_hash, state, updated_at) VALUES (?, ?, 'uncertain', 1)",
                (f"{long_id}:0", identity.identity_hash),
            )
            conn.commit()
            assert command_mod._chunk_policy_revision(conn, identity.identity_hash) == 1
            pending = command_mod.chunk_message(long_id, "user", long_text)[1:]
            assert len(pending) == 3
            total, docs, _meta = command_mod._chunk_pending_rows(
                conn, identity.identity_hash, "conversational", 50, incremental=True,
            )
            backlog = command_mod._chunk_backlog_estimate(conn, identity.identity_hash, "conversational")
            assert (total, [doc[0] for doc in docs], backlog) == (
                3, [chunk.chunk_id for chunk in pending], "3 (gap-fill)",
            )
        finally:
            store.close()
        result = command_mod._chunk_backfill_run(
            engine._config, engine._config.database_path, apply=True, limit=50,
            retry_uncertain=False, policy="", incremental=True,
        )
        # The intentionally excluded uncertain chunk keeps the run partial.
        assert result["status"] == "partial", result
        assert not result["error"] and not result["failed"], result
        assert result["embedded"] == 3
        assert providers[-1].documents == [chunk.text for chunk in pending]
        print(f"F1 {boundary}: selected={total} backlog={backlog}; embedded={result['embedded']} once")
        store = VectorStore(engine._config.database_path, config=engine._config)
        try:
            ids = store.connection.execute(
                "SELECT chunk_id FROM lcm_chunk_meta WHERE store_id = ? ORDER BY chunk_index", (long_id,),
            ).fetchall()
            assert [row[0] for row in ids] == [chunk.chunk_id for chunk in pending]
            assert command_mod._chunk_backlog_estimate(
                store.connection, identity.identity_hash, "conversational",
            ).split()[0] == "0"
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_local_predispatch_failure_preserves_pass_outcome(tmp_path, home, monkeypatch, caplog):
    """REGRESSION F2: an empty retry cannot turn a failed ordinary stage green."""
    monkeypatch.setattr(provider_mod, "probe_provider_availability",
                        lambda _config: {"available": True, "detail": "fake"})
    monkeypatch.setattr(command_mod, "resolve_provider",
                        lambda config, **_kwargs: UnavailableProvider(
                            config.embedding_provider, config.embedding_model,
                        ))
    engine = _make_engine(tmp_path, home, provider="fastembed")
    try:
        engine._store.append(SESSION, {"role": "user", "content": FILLER})
        engine._dag.add_node(SummaryNode(
            session_id=SESSION, depth=0, summary="leaf text", created_at=1.0, latest_at=1.0,
        ))
        _, before = maintenance_mod.incremental_embedding_state(engine._config.database_path)
        with caplog.at_level(logging.DEBUG, logger=maintenance_mod.__name__):
            outcome = maintenance_mod.run_incremental_embedding_pass(
                engine._config.database_path, engine._config,
            )
        last, after = maintenance_mod.incremental_embedding_state(engine._config.database_path)
        assert outcome == "summaries=partial chunks=partial"
        assert after - before == 1
        assert last[1] == outcome
        assert f"outcome=ran detail={outcome}" in caplog.text
        assert "provider embedded no selected document" in caplog.text
        print(f"F2: outcome={outcome}; recorded_failures={after - before}")
    finally:
        engine.shutdown()


def test_queries_fail_fast_during_three_second_download(monkeypatch, tmp_path):
    """REGRESSION F3: a download cannot occupy query workers via the cache lock."""
    downloading = threading.Event()
    made, errors, queries = [], [], []

    class Model:
        def __init__(self, **kwargs):
            if kwargs["local_files_only"]:
                raise RuntimeError("not cached")
            downloading.set()
            time.sleep(3.0)
            made.append(self)

        def query_embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        embed = query_embed

    monkeypatch.setattr(provider_mod, "_load_fastembed", lambda: Model)
    warm = FastembedProvider("local", cache_dir=tmp_path)

    def prepare():
        try:
            warm.warmup()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=prepare)
    thread.start()
    try:
        assert downloading.wait(2)
        for _ in range(6):
            query = FastembedProvider("local", cache_dir=tmp_path, timeout=0.5)
            start = time.monotonic()
            try:
                result = query.embed_query("q")
            except Exception as exc:
                result = exc
            queries.append((time.monotonic() - start, result))
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert all(isinstance(result, ProviderNotWarmedUp) for _, result in queries), queries
    assert max(wall for wall, _ in queries) < 0.25, queries
    assert len(made) == 1
    query = FastembedProvider("local", cache_dir=tmp_path, timeout=0.5)
    assert query.embed_query("ready") == [1.0, 0.0]
    assert query._model is warm._model
    assert len(made) == 1
    print(f"F3: 3s download; query_walls={[round(wall, 4) for wall, _ in queries]}; "
          f"all ProviderNotWarmedUp; constructions={len(made)}")
