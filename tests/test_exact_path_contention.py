"""Exact-path transfer and leased cancellation regressions for #1586."""
import json
import sqlite3
import time
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from hermes_lcm import retrieval_core
from hermes_lcm.vector_store import VectorStore
from tests.test_chunk_vector_store import DIM, MODEL, PROVIDER, _seed_messages, _write
from tests.test_vector_store import _add_summary
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.config import LCMConfig


@pytest.fixture
def corpus(tmp_path):
    db = tmp_path / 'exact.db'
    _seed_messages(db, [(i, 'keep' if i < 4 else 'other', 'history',
                        'user', 'synthetic', float(i)) for i in range(1, 6)])
    dag = SummaryDAG(db)
    store = VectorStore(db, bounded_scan_rows=2)
    chunk = store.register_profile(MODEL, PROVIDER, DIM, task='chunk')
    for i in range(1, 6):
        _write(store, f'{i}:0', i, 0, [i, 1, 0, 0])
    summary = store.register_profile(MODEL, PROVIDER, DIM)
    ids = []
    for i in range(1, 6):
        ident = _add_summary(dag, session_id='keep', created_at=float(i))
        ids.append(str(ident))
        store.record_embedding(ident, 'summary', MODEL, [i, 1, 0, 0],
                               identity=store.capture_identity(MODEL))
    yield store, chunk, summary, ids
    store.close()
    dag.close()


@pytest.mark.parametrize('chunk', [False, True])
def test_zero_binary_skips_vector_count(corpus, chunk):
    store, chunk_id, summary_id, _ = corpus
    traced = []
    store.connection.set_trace_callback(traced.append)
    assert not store._binary_fully_synced(chunk_id if chunk else summary_id, chunk=chunk)
    table = 'lcm_chunk_vectors' if chunk else 'lcm_embedding_vectors'
    assert not any('COUNT(*)' in sql.upper() and table in sql for sql in traced)


@pytest.mark.parametrize('chunk', [False, True])
def test_coalesced_load_equals_row_reference(corpus, monkeypatch, chunk):
    np = pytest.importorskip('numpy')
    store, chunk_id, summary_id, ids = corpus
    identity = chunk_id if chunk else summary_id
    table = 'lcm_chunk_vectors' if chunk else 'lcm_embedding_vectors'
    meta = 'lcm_chunk_meta' if chunk else 'lcm_embedding_meta'
    key = 'chunk_id' if chunk else 'embedded_id'
    selected = [f'{i}:0' for i in range(1, 6)] if chunk else ids
    store.connection.execute(f'UPDATE {table} SET vec=? WHERE {key}=?', (b'short', selected[1]))
    store.connection.execute(f'UPDATE {meta} SET archived=1 WHERE {key}=?', (selected[2],))
    store.connection.commit()
    if chunk:
        filtered = store._bounded_chunk_candidate_ids(identity, since=1, until=4,
            conversation_ids=['keep'], source='history', limit=-1)
        assert filtered == ['2:0', '1:0']
        def loader():
            return store._load_chunk_matrix(np, identity, DIM, filtered, 'float32')
    else:
        filtered = store._bounded_candidate_ids(identity, since=1, until=3,
            conversation_ids=['keep'], source=None, limit=-1)
        def loader():
            return store._load_matrix(np, identity, DIM, filtered, 'float32')
    traced = []
    store.connection.set_trace_callback(traced.append)
    actual = loader()
    assert any('group_concat' in sql.lower() for sql in traced)
    # Existing row-wise SQL, executed by the fallback, is an independent oracle.
    original_execute = store.connection
    class RowsOnly:
        def execute(self, sql, args=()):
            if 'group_concat' in sql.lower():
                exc = sqlite3.DataError('aggregate too big')
                exc.sqlite_errorcode = sqlite3.SQLITE_TOOBIG
                raise exc
            return original_execute.execute(sql, args)
    # Use the explicit row-wise loader for chunks and a forced aggregate fallback for summaries.
    if chunk:
        expected = store._load_chunk_vectors_for_ids(identity, DIM, filtered)
        assert actual[:3] == expected[:3]
        np.testing.assert_array_equal(actual[3], np.asarray(expected[3], dtype=np.float32))
    else:
        monkeypatch.setattr(store, '_conn', RowsOnly())
        expected = loader()
        assert actual[:3] == expected[:3]
        np.testing.assert_array_equal(actual[3], expected[3])
        monkeypatch.setattr(store, '_conn', original_execute)
    assert actual[1] == [selected[0]]


def test_chunk_candidate_transfer_is_coalesced(corpus):
    store, chunk, _, _ = corpus
    traced = []
    store.connection.set_trace_callback(traced.append)
    assert store._bounded_chunk_candidate_ids(chunk, since=None, until=None,
        conversation_ids=None, source=None, limit=-1) == [f'{i}:0' for i in range(5, 0, -1)]
    assert any('json_group_array' in sql for sql in traced)


@pytest.mark.parametrize('encoding', ['UTF-8', 'UTF-16le', 'UTF-16be'])
def test_exact_transfer_preserves_arbitrary_bytes_and_storage(tmp_path, encoding):
    conn = sqlite3.connect(tmp_path / 'bytes.db')
    conn.execute(f"PRAGMA encoding='{encoding}'")
    conn.execute('CREATE TABLE fixture(vector_id, vector_kind, vec)')
    values = [bytes(range(256)), b'\x00\xff\x00', b'', None, 17, 'text', b'short']
    conn.executemany("INSERT INTO fixture VALUES (?, 'chunk', ?)", enumerate(values))
    store = object.__new__(VectorStore)
    store._conn = conn
    sql = 'SELECT rowid, vector_id, vector_kind, vec FROM fixture ORDER BY rowid DESC'
    try:
        actual = store._fetch_exact_rows(sql, ())
        expected = conn.execute(sql).fetchall()
        assert [tuple(bytes(v) if isinstance(v, memoryview) else v for v in row)
                for row in actual] == expected
    finally:
        conn.close()


def test_summary_cache_survives_chunk_multibatch(corpus):
    np = pytest.importorskip('numpy')
    store, chunk, summary, ids = corpus
    warm = store._numpy_rows(np, summary, DIM, ids[:1])
    store.knn_chunks([1, 0, 0, 0], model=MODEL, provider=PROVIDER, full_scan=True)
    assert len(store._matrix_cache) == 1
    assert next(iter(store._matrix_cache.values())) is warm
    assert not store._chunk_matrix_cache


def test_leased_coalesced_sql_deadline_and_next_borrower(corpus, monkeypatch):
    store, _, _, _ = corpus
    engine = SimpleNamespace(_store=SimpleNamespace(db_path=store.db_path), _config=LCMConfig())
    monkeypatch.setattr(retrieval_core, '_vector_store_pool', OrderedDict())
    with pytest.raises(sqlite3.OperationalError, match='interrupted'):
        retrieval_core._run_pooled_knn(engine, vector_store_cls=VectorStore,
            scan_rows=None, deadline=time.monotonic() + .05,
            query=lambda s: s.connection.execute('WITH RECURSIVE x(n) AS '
                '(VALUES(0) UNION ALL SELECT n+1 FROM x WHERE n<100000000) '
                "SELECT group_concat(n, '') FROM x").fetchone())
    start = time.monotonic()
    # A progress-handler-free aggregate must still be interrupted; stale cancellation must drain.
    callbacks = []
    def quick(s):
        s.connection.set_trace_callback(callbacks.append)
        return json.loads(s.connection.execute('SELECT json_group_array(1)').fetchone()[0])
    assert retrieval_core._run_pooled_knn(engine, vector_store_cls=VectorStore,
        scan_rows=None, deadline=time.monotonic()+1, query=quick) == [1]
    assert time.monotonic()-start < 1
    # Real SQL work, with no per-1000-op Python callback on the leased connection.
    entry = next(iter(retrieval_core._vector_store_pool.values()))
    class ConnProxy:
        def __getattr__(self, key):
            return getattr(entry_conn, key)
        def set_progress_handler(self, *args):
            pytest.fail('hot pooled Python progress handler')
    entry_conn = entry['store']._conn
    entry['store']._conn = ConnProxy()
    try:
        assert retrieval_core._run_pooled_knn(engine, vector_store_cls=VectorStore,
            scan_rows=None, deadline=time.monotonic()+1, query=lambda s: s.connection.execute('SELECT 1').fetchone()[0]) == 1
    finally:
        entry['store']._conn = entry_conn
        retrieval_core._reset_vector_store_pool()
