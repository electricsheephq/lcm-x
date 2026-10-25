"""Regressions for the first LOCAL pacing bot fix round."""
import dataclasses
import threading

import pytest

from hermes_lcm import command, embedding_maintenance as maintenance
from tests.test_issue_1014_incremental_embedding import FakeProvider
from tests.test_issue_1586_local_pacing import Clock, REAL_PREPARE
from tests.test_issue_1586_local_pacing import local_store as local_store


@pytest.mark.parametrize("slow_step", ["_chunk_pending_rows", "_prepare_embedding_provider_documents"])
def test_slow_discovery_still_drains_across_slices(local_store, monkeypatch, slow_step):
    engine, _ = local_store
    config = dataclasses.replace(engine._config, embedding_max_batch_items=2)
    for i in range(10):
        engine._store.append("seed", {"role": "user", "content": f"slow discovery {i} " + "detail " * 60})
    clock = Clock()
    original = getattr(command, slow_step)
    chunk_run = command._chunk_backfill_run
    chunks, batches = [], []

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now += 2.0
        return result

    def measured(*args, **kwargs):
        result = chunk_run(*args, **kwargs)
        chunks.append(result)
        return result

    class Provider(FakeProvider):
        def embed_document_batches(self, texts, **kwargs):
            batches.append(len(texts))
            yield from super().embed_document_batches(texts, **kwargs)

    monkeypatch.setattr(command, "time", clock)
    monkeypatch.setattr(maintenance, "time", clock)
    monkeypatch.setattr(command, slow_step, slow)
    monkeypatch.setattr(command, "_chunk_backfill_run", measured)
    monkeypatch.setattr(command, "resolve_provider", lambda c, **kw: Provider(c.embedding_provider, c.embedding_model))
    state, results = {}, []
    for _ in range(40):
        last = maintenance.run_incremental_embedding_pass(engine._store.db_path, config, slice_state=state)
        results.append(last)
        if not last["continue"]:
            break
    assert not any("no_progress" in r["outcome"] and r["cycle_remaining"] for r in results)
    assert last["outcome"].endswith(" complete")
    selected = [r for r in chunks if r["selected"]]
    assert selected and all(r["embedded"] > 0 and not r["failed"] for r in selected)
    assert len(selected) > 1 and any(r["remaining"] for r in selected)
    assert batches and max(batches) <= 2
    assert any(r["rows_written"] and r["overrun_s"] >= 1.0 for r in results)


@pytest.mark.parametrize("other_attempts", [5, 6])
def test_other_model_attempts_do_not_block_preparation(local_store, monkeypatch, other_attempts):
    engine, _ = local_store
    other = dataclasses.replace(engine._config, embedding_model="other-model")
    maintenance._PREPARE_RETRY[maintenance._local_model_key(other)] = (other_attempts, 0.0)
    monkeypatch.setattr(FakeProvider, "warmup", lambda self: [1.0, 0.0], raising=False)
    assert REAL_PREPARE(engine._store.db_path, engine._config)
    maintenance._PREPARE_THREAD.join(3)
    assert maintenance.local_model_state(engine._config) == ("present", "")


def test_success_clears_only_its_model_retry_counter(local_store, monkeypatch):
    engine, _ = local_store
    key = maintenance._local_model_key(engine._config)
    other = maintenance._local_model_key(dataclasses.replace(engine._config, embedding_model="other-model"))
    maintenance._PREPARE_RETRY.update({key: (5, 0.0), other: (0, 600.0)})
    monkeypatch.setattr(FakeProvider, "warmup", lambda self: [1.0, 0.0], raising=False)
    assert REAL_PREPARE(engine._store.db_path, engine._config)
    maintenance._PREPARE_THREAD.join(3)
    assert maintenance.local_model_state(engine._config) == ("present", "")
    assert key not in maintenance._PREPARE_RETRY
    assert maintenance._PREPARE_RETRY[other] == (0, 600.0)


def test_exhaustion_status_is_per_model(local_store):
    engine, _ = local_store
    other = dataclasses.replace(engine._config, embedding_model="other-model")
    key = maintenance._local_model_key(other)
    maintenance._PREPARE[key] = ("unavailable", "offline fixture")
    maintenance._PREPARE_RETRY[key] = (6, 0.0)
    maintenance._PREPARE[maintenance._local_model_key(engine._config)] = ("unavailable", "offline fixture")
    db = engine._store.db_path
    assert maintenance.run_incremental_embedding_pass(db, engine._config) == "model_unavailable"
    assert maintenance.run_incremental_embedding_pass(db, other) == "model_prepare_exhausted"


def test_sixth_attempt_still_accepts_callbacks_while_preparing(local_store, monkeypatch):
    engine, _ = local_store
    key = maintenance._local_model_key(engine._config)
    maintenance._PREPARE_RETRY[key] = (5, 0.0)
    entered, release = threading.Event(), threading.Event()
    callbacks = []

    class Preparing(FakeProvider):
        def warmup(self):
            entered.set()
            assert release.wait(3)

    monkeypatch.setattr(command, "resolve_provider", lambda c, **kw: Preparing(c.embedding_provider, c.embedding_model))
    try:
        assert REAL_PREPARE(engine._store.db_path, engine._config, on_ready=lambda: callbacks.append(1))
        assert entered.wait(3)
        assert not REAL_PREPARE(engine._store.db_path, engine._config, on_ready=lambda: callbacks.append(2))
        assert len(maintenance._PREPARE_READY[key]) == 2
    finally:
        release.set()
        maintenance._PREPARE_THREAD.join(3)
    assert callbacks == [1, 2]
    assert key not in maintenance._PREPARE_RETRY


@pytest.mark.parametrize("other_attempts", [0, 6])
def test_six_failures_back_off_only_this_key_and_keep_live_callbacks(local_store, monkeypatch, other_attempts):
    engine, _ = local_store
    key = maintenance._local_model_key(engine._config)
    other = maintenance._local_model_key(dataclasses.replace(engine._config, embedding_model="other-model"))
    maintenance._PREPARE_RETRY[other] = (other_attempts, 0.0)
    clock, calls, callbacks = Clock(), [], []

    class Unavailable(FakeProvider):
        def warmup(self):
            calls.append(clock.now)
            raise RuntimeError("offline fixture")

    monkeypatch.setattr(maintenance, "time", clock)
    monkeypatch.setattr(command, "resolve_provider", lambda c, **kw: Unavailable(c.embedding_provider, c.embedding_model))
    for attempt in range(6):
        assert REAL_PREPARE(engine._store.db_path, engine._config, on_ready=lambda: callbacks.append(1), owner="fixture-owner")
        maintenance._PREPARE_THREAD.join(3)
        assert not maintenance._PREPARE_THREAD.is_alive()
        assert maintenance._PREPARE_RETRY[key][0] == attempt + 1
        if attempt < 5:
            assert maintenance._PREPARE_READY[key]
            clock.now += 600.0
    assert maintenance._PREPARE_RETRY[other] == (other_attempts, 0.0)
    assert len(maintenance._PREPARE_READY[key]) == 1
    assert not REAL_PREPARE(engine._store.db_path, engine._config, on_ready=lambda: callbacks.append(2), owner="fixture-owner")
    assert len(maintenance._PREPARE_READY[key]) == 1
    assert len(calls) == 6 and not callbacks
    assert maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config) == "model_prepare_exhausted"
