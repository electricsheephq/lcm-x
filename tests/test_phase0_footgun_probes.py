"""Confirmed Phase 0 footguns and regression probes."""

from collections import OrderedDict
import threading
from types import SimpleNamespace

import pytest

from hermes_lcm import retrieval_core
from hermes_lcm.config import LCMConfig
from hermes_lcm.presets import shipped_presets
from hermes_lcm.vector_store import VectorStore


@pytest.mark.parametrize("preset", shipped_presets(), ids=lambda preset: preset.name)
def test_preset_leaf_chunk_matches_product_default(preset):
    # #1129 raised the leaf default to 24k. #1139 caps the summary PREFIX
    # at a quarter of the threshold, not the leaf-chunk recommendation.
    assert preset.runtime_env["leaf_chunk_tokens"] == LCMConfig().leaf_chunk_tokens


class _ClosedBeforeQuery(AssertionError):
    pass


@pytest.mark.parametrize("action", ["third-key-eviction", "reset"])
def test_handed_out_vector_store_survives_until_query(tmp_path, monkeypatch, action):
    # Isolate this pool so cleanup never closes stores owned by another test.
    monkeypatch.setattr(retrieval_core, "_vector_store_pool", OrderedDict())
    monkeypatch.setattr(retrieval_core, "_pool_lock", threading.Lock())
    engines = [
        SimpleNamespace(_store=SimpleNamespace(db_path=tmp_path / f"{key}.db"), _config=LCMConfig())
        for key in ("first", "second", "third")
    ]
    acquired, proceed = threading.Event(), threading.Event()
    original = retrieval_core._acquire_vector_store
    outcomes = []

    def acquire(engine, **kwargs):
        result = original(engine, **kwargs)
        if engine is engines[0]:
            acquired.set()
            if not proceed.wait(1):
                raise RuntimeError("probe did not release the query")
        return result

    def query(store):
        if store._conn is None:
            raise _ClosedBeforeQuery("handed-out VectorStore was closed before its query")
        return store._conn.execute("SELECT 1").fetchone()[0]

    def requester():
        try:
            outcomes.append(retrieval_core._run_pooled_knn(
                engines[0], vector_store_cls=VectorStore, scan_rows=None,
                deadline=float("inf"), query=query,
            ))
        except BaseException as exc:
            outcomes.append(exc)

    def evict():
        try:
            for engine in (engines[1:] if action == "third-key-eviction" else engines[1:2]):
                _, store_lock, _ = original(engine, vector_store_cls=VectorStore, scan_rows=None)
                with store_lock:
                    pass
            if action == "reset":
                retrieval_core._reset_vector_store_pool()
        except BaseException as exc:
            outcomes.append(exc)
        finally:
            proceed.set()

    monkeypatch.setattr(retrieval_core, "_acquire_vector_store", acquire)
    reader = threading.Thread(target=requester, daemon=True)
    evictor = threading.Thread(target=evict, daemon=True)
    reader.start()
    try:
        assert acquired.wait(1), "requester did not acquire its store"
        evictor.start()
        evictor.join(1)
        assert not evictor.is_alive(), "eviction did not complete"
    finally:
        proceed.set()
        reader.join(1)
        if evictor.ident is not None:
            evictor.join(0.5)
        assert not reader.is_alive() and not evictor.is_alive(), "probe leaked a worker"
        retrieval_core._reset_vector_store_pool()
    assert len(outcomes) == 1, outcomes
    if isinstance(outcomes[0], BaseException):
        raise outcomes[0]
    assert outcomes == [1]
