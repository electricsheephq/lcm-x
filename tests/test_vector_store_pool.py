"""Pool lifetime and unchanged retrieval resource contracts."""

from collections import OrderedDict
import threading
from types import SimpleNamespace

import pytest

from hermes_lcm import retrieval_core
from hermes_lcm.config import LCMConfig
from hermes_lcm.vector_store import VectorStore


@pytest.fixture
def engines(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval_core, "_vector_store_pool", OrderedDict())
    monkeypatch.setattr(retrieval_core, "_pool_lock", threading.Lock())
    yield [
        SimpleNamespace(
            _store=SimpleNamespace(db_path=tmp_path / f"{key}.db"),
            _config=LCMConfig(),
        )
        for key in ("first", "second", "third")
    ]
    retrieval_core._reset_vector_store_pool()


def _run(engine, query=lambda store: store, *, store_cls=VectorStore, scan_rows=None):
    return retrieval_core._run_pooled_knn(
        engine, vector_store_cls=store_cls, scan_rows=scan_rows,
        deadline=float("inf"), query=query,
    )


def test_pool_preserves_keys_lru_and_idle_close(engines):
    first = _run(engines[0])
    second = _run(engines[0], scan_rows=10)
    assert _run(engines[0]) is first  # A hit makes this entry most recent.
    third = _run(engines[1])

    assert list(retrieval_core._vector_store_pool) == [
        (str(engines[0]._store.db_path), -1),
        (str(engines[1]._store.db_path), -1),
    ]
    assert second._conn is None
    assert first._conn is not None and third._conn is not None
    retrieval_core._reset_vector_store_pool()
    assert not retrieval_core._vector_store_pool
    assert first._conn is None and third._conn is None


@pytest.mark.parametrize("fail_query", [False, True])
def test_transient_store_closes_after_query(engines, fail_query):
    events = []

    class TransientStore:
        def __init__(self, *args, **kwargs):
            events.append("open")

        def close(self):
            events.append("close")

    def query(store):
        events.append("query")
        if fail_query:
            raise ValueError("query failed")
        return 42

    if fail_query:
        with pytest.raises(ValueError, match="query failed"):
            _run(engines[0], query, store_cls=TransientStore)
    else:
        assert _run(engines[0], query, store_cls=TransientStore) == 42
    assert events == ["open", "query", "close"]
    assert not retrieval_core._vector_store_pool


def test_pooled_store_is_reused_after_query_error(engines):
    first = _run(engines[0])

    def query(store):
        assert store is first
        raise ValueError("query failed")

    with pytest.raises(ValueError, match="query failed"):
        _run(engines[0], query)
    assert _run(engines[0]) is first
    assert first._conn.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.parametrize("action", ["third-key-eviction", "reset"])
@pytest.mark.parametrize("fail_query", [False, True])
def test_retired_store_closes_after_last_release(engines, monkeypatch, action, fail_query):
    first = _run(engines[0])
    closed = []
    original_close = first.close

    def close():
        closed.append(first)
        original_close()

    monkeypatch.setattr(first, "close", close)

    def inner_query(store):
        assert store is first
        if action == "reset":
            retrieval_core._reset_vector_store_pool()
        else:
            _run(engines[1])
            _run(engines[2])
        assert first not in [entry["store"] for entry in retrieval_core._vector_store_pool.values()]
        assert not closed
        assert store._conn.execute("SELECT 1").fetchone()[0] == 1

    def outer_query(store):
        _run(engines[0], inner_query)  # A second reservation uses the same entry.
        assert not closed  # Releasing one borrower must not close the other.
        assert store._conn.execute("SELECT 1").fetchone()[0] == 1
        if fail_query:
            raise ValueError("query failed")
        return 42

    if fail_query:
        with pytest.raises(ValueError, match="query failed"):
            _run(engines[0], outer_query)
    else:
        assert _run(engines[0], outer_query) == 42
    assert closed == [first]
    assert first._conn is None
    retrieval_core._reset_vector_store_pool()
    assert closed == [first]
