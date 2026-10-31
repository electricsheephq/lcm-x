"""Deadline readers borrow caches; only the owning store may clear them."""

import gc
import time

import pytest

from hermes_lcm.dag import SummaryDAG, SummaryNode
import hermes_lcm.vector_store as vector_store_module
from hermes_lcm.vector_store import VectorStore, _PrescreenDeadlineExpired


_CACHE_NAMES = (
    "_matrix_cache", "_chunk_matrix_cache",
    "_binary_matrix_cache", "_chunk_binary_matrix_cache",
)


@pytest.fixture
def store(tmp_path):
    store = VectorStore(tmp_path / "vectors.db")
    yield store
    store.close()


def _seed_caches(store):
    for name in _CACHE_NAMES:
        getattr(store, name)[("seed", 1)] = object()
    return {name: dict(getattr(store, name)) for name in _CACHE_NAMES}


def _assert_caches(store, expected):
    for name, entries in expected.items():
        assert dict(getattr(store, name)) == entries, name


def test_regression_deadline_reader_preserves_all_shared_caches(store):
    expected = _seed_caches(store)
    assert store._read_with_deadline(lambda reader: 42, time.monotonic() + 5) == 42
    gc.collect()
    _assert_caches(store, expected)


def test_regression_loader_fills_remain_shared(store):
    def load(reader):
        for name in _CACHE_NAMES:
            assert getattr(reader, name) is getattr(store, name)
        return _seed_caches(reader)

    expected = store._read_with_deadline(load, time.monotonic() + 5)
    gc.collect()
    _assert_caches(store, expected)


def test_regression_repeated_deadline_knn_reuses_float_matrix(store, monkeypatch):
    dag = SummaryDAG(store.db_path)
    try:
        node_id = dag.add_node(SummaryNode(session_id="fixture", summary="memory"))
    finally:
        dag.close()
    store.register_profile("fixture", "deterministic", 2)
    store.record_embedding(
        str(node_id), "summary", "fixture", [1.0, 0.0],
        identity=store.capture_identity("fixture"),
    )
    # The float32 identity has no binary prescreen: exercise the real exact path.
    monkeypatch.setattr(vector_store_module, "_load_numpy", lambda: __import__("numpy"))
    original = store._load_matrix
    loads = []

    def counted(*args, **kwargs):
        loads.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_load_matrix", counted)
    first = store.knn([1.0, 0.0], k=1, deadline=time.monotonic() + 5)
    assert first and first.coverage == "full"
    assert len(loads) == 1
    keys = tuple(store._matrix_cache)
    second = store.knn([1.0, 0.0], k=1, deadline=time.monotonic() + 5)
    assert second == first
    assert len(loads) == 1
    assert tuple(store._matrix_cache) == keys


def test_control_normal_close_clears_caches_and_checkpoints(store):
    _seed_caches(store)
    statements = []
    store.connection.set_trace_callback(statements.append)
    store.close()
    assert "PRAGMA wal_checkpoint(PASSIVE)" in statements
    assert store._conn is None
    assert all(not getattr(store, name) for name in _CACHE_NAMES)


def test_control_expired_deadline_preserves_caches(store):
    expected = _seed_caches(store)
    with pytest.raises(_PrescreenDeadlineExpired):
        store._read_with_deadline(lambda reader: pytest.fail("loader ran"), 0.0)
    gc.collect()
    _assert_caches(store, expected)


def test_regression_expiry_after_reader_creation_preserves_caches(store, monkeypatch):
    expected = _seed_caches(store)
    clock = [1.0]
    monkeypatch.setattr(vector_store_module, "_monotonic", lambda: clock[0])

    def load(reader):
        clock[0] = 3.0
        return 42

    with pytest.raises(_PrescreenDeadlineExpired):
        store._read_with_deadline(load, 2.0)
    gc.collect()
    _assert_caches(store, expected)
