"""#1749: deterministic SQLite handoffs and frozen-base recall identity."""
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hermes_lcm import db_bootstrap, tools, vector_store as vm
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore
from hermes_lcm.retrieval_core import _reset_vector_store_pool


class Cursor(sqlite3.Cursor):
    def execute(self, sql, *args, **kwargs):
        self.target = any(s in sql for s in (
            'FROM distinct_sessions', 'FROM lcm_chunk_binary b',
            'FROM lcm_embedding_binary b',
        ))
        return super().execute(sql, *args, **kwargs)

    def fetchall(self):
        rows = super().fetchall()
        if self.target:
            self.connection.rows += len(rows)
        return rows

    def fetchone(self):
        row = super().fetchone()
        if self.target:
            self.connection.rows += int(row is not None)
        return row


class Connection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = 0
        self.handlers = []
        self.fail_code = None

    def execute(self, sql, *args, **kwargs):
        if self.fail_code is not None and 'json_group_array' in sql:
            exc = sqlite3.OperationalError('synthetic aggregate failure')
            exc.sqlite_errorcode = self.fail_code
            raise exc
        return self.cursor(Cursor).execute(sql, *args, **kwargs)

    def set_progress_handler(self, callback, n):
        self.handlers.append((callback, n))
        return super().set_progress_handler(callback, n)


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(db_bootstrap, 'dispatch_background_stem_backfill', lambda *a: None)
    config = LCMConfig(database_path=str(tmp_path/'rows.db'), embeddings_enabled=True,
                       embedding_provider='mock', embedding_model='row-model',
                       embedding_storage_dtype='int8', recall_query_timeout_s=30)
    store = MessageStore(config.database_path, hermes_home=str(tmp_path))
    dag = SummaryDAG(store.db_path)
    vs = vm.VectorStore(store.db_path, config=config)
    identities = {}
    for task in ('summary', 'chunk'):
        vs.register_profile('row-model', 'mock', 16, task=task, dtype='int8')
        identities[task] = vm.EmbeddingIdentity.canonical(
            'mock', 'row-model', '', 16, 'int8', 'little', task)
    for i in range(90):
        session = f'session-{i % 7}'
        timestamp = 1700000000.0 + i
        ident = store.append(session, {'role': 'user', 'content': f'topicword{i % 30} detail {i}',
                                      'timestamp': timestamp}, source='telegram')
        node = dag.add_node(SummaryNode(session_id=session, summary=f'topicword{i % 30} summary',
            source_ids=[ident], source_type='messages', token_count=3,
            created_at=timestamp, earliest_at=timestamp, latest_at=timestamp))
        vector = [(-1.0 if (i >> bit) & 1 else 1.0) for bit in range(16)]
        vs.record_embedding(str(node), 'summary', 'row-model', vector,
                            identity=identities['summary'])
        vs.record_chunk_embedding(f'{ident}:0', 'row-model', vector, store_id=ident,
            chunk_index=0, char_start=0, char_end=12, token_estimate=3,
            identity=identities['chunk'])
    store.connection.execute('UPDATE messages SET timestamp=observed_at')
    store.connection.commit()
    engine = SimpleNamespace(_store=store, _dag=dag, _config=config,
                             _hermes_home=str(tmp_path), current_session_id='session-0')
    try:
        yield engine, vs, identities
    finally:
        _reset_vector_store_pool()
        vs.close()
        dag.close()
        store.close()


@pytest.fixture
def counted(monkeypatch):
    original = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        conn = original(*args, **kwargs, factory=Connection)
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, 'connect', connect)
    return connections


def load(vs, identities, chunk, **kwargs):
    common = dict(since=None, until=None, conversation_ids=None)
    common.update(kwargs)
    if chunk:
        return vs._load_chunk_binary_matrix(np, identities['chunk'].identity_hash,
                                            source=None, **common)
    return vs._load_embedding_binary_matrix(np, identities['summary'].identity_hash, **common)


def stock_functions():
    data = json.loads((Path(__file__).parent/'fixtures/issue_1749_stock.json').read_text())
    functions = {}
    for name, module in [('tools.py', tools), ('vector_store.py', vm)]:
        namespace = vars(module).copy()
        for key, source in data['sources'][name].items():
            exec(compile(source, '<stock-1749>', 'exec'), namespace)
            functions[key] = namespace[key]
    return functions


def test_session_scope_crosses_python_in_two_rows(seeded, counted):
    engine, _, _ = seeded
    actual, expired = tools._retrieval_candidate_session_ids(
        engine, {'session-0'}, deadline=time.monotonic()+30)
    assert not expired
    assert actual == [f'session-{i}' for i in range(1, 7)]
    assert sum(c.rows for c in counted) == 2
    assert not any(cb is not None for c in counted for cb, _ in c.handlers)


@pytest.mark.parametrize('chunk', [True, False])
def test_binary_matrix_crosses_python_in_one_row(seeded, counted, chunk):
    _, vs, identities = seeded
    ids, matrix = vs._load_binary_with_deadline(
        lambda reader: load(reader, identities, chunk), time.monotonic()+30)
    assert len(ids) == len(matrix) == 90
    assert sum(c.rows for c in counted) == 1
    assert not any(cb is not None for c in counted for cb, _ in c.handlers)
    assert not any(t.name == 'sqlite-read-deadline' for t in threading.enumerate())


@pytest.mark.parametrize('chunk', [True, False])
@pytest.mark.parametrize('reverse', [False, True])
def test_binary_order_and_bytes_match_stock_with_filters_and_bad_widths(seeded, chunk, reverse):
    _, vs, identities = seeded
    name = '_load_chunk_binary_matrix' if chunk else '_load_embedding_binary_matrix'
    original = stock_functions()[name]
    table, key = ('lcm_chunk_binary', 'chunk_id') if chunk else ('lcm_embedding_binary', 'embedded_id')
    # Keep the old skip-empty/first-nonempty-width semantics, including arbitrary bytes.
    for value, blob in [('1:0' if chunk else '1', b''),
                        ('2:0' if chunk else '2', b'\x00\xff'),
                        ('3:0' if chunk else '3', b'\x80')]:
        vs.connection.execute(f'UPDATE {table} SET bits=? WHERE {key}=?', (blob, value))
    vs.connection.commit()
    vs.connection.execute(f'PRAGMA reverse_unordered_selects={int(reverse)}')
    for filters in ({}, {'conversation_ids': ['session-2', 'session-4']},
                    {'since': 1700000010.0, 'until': 1700000060.0},
                    {'conversation_ids': []}):
        kwargs = dict(since=None, until=None, conversation_ids=None, **({'source': None} if chunk else {}))
        kwargs.update(filters)
        expected = original(vs, np, identities['chunk' if chunk else 'summary'].identity_hash, **kwargs)
        actual = getattr(vs, name)(np, identities['chunk' if chunk else 'summary'].identity_hash, **kwargs)
        assert actual[0] == expected[0]
        assert actual[1].shape == expected[1].shape
        assert actual[1].tobytes() == expected[1].tobytes()


@pytest.mark.parametrize('code', [sqlite3.SQLITE_TOOBIG, sqlite3.SQLITE_NOMEM, sqlite3.SQLITE_INTERRUPT])
@pytest.mark.parametrize('site', ['sessions', 'chunk', 'summary'])
def test_aggregate_errors_keep_resource_fallback_and_terminal_interrupt(seeded, counted, monkeypatch, code, site):
    engine, vs, identities = seeded
    connect = sqlite3.connect

    def failing(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.fail_code = code
        return conn

    monkeypatch.setattr(sqlite3, 'connect', failing)
    def invoke():
        if site == 'sessions':
            return tools._retrieval_candidate_session_ids(engine, {'excluded'})[0]
        return vs._load_binary_with_deadline(
            lambda reader: load(reader, identities, site == 'chunk'), time.monotonic()+30)[0]

    if code == sqlite3.SQLITE_INTERRUPT:
        with pytest.raises((sqlite3.OperationalError, vm._PrescreenDeadlineExpired)):
            invoke()
    else:
        assert len(invoke()) == (7 if site == 'sessions' else 90)
        assert sum(c.rows for c in counted) == (14 if site == 'sessions' else 90)


def test_private_vector_deadline_drains_and_restores_worker_slots(seeded):
    _, vs, _ = seeded
    slots = threading.BoundedSemaphore(1)
    def expensive(reader):
        return reader.connection.execute('''WITH RECURSIVE n(x) AS (
            VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000000
        ) SELECT sum(x) FROM n''').fetchone()
    with pytest.raises(vm._PrescreenDeadlineExpired):
        tools._run_within_deadline(
            lambda: vs._load_binary_with_deadline(expensive, time.monotonic()+0.05),
            remaining_s=5, name='1749-vector', worker_slots=slots)
    assert slots.acquire(blocking=False)
    slots.release()
    assert not any(t.name in {'sqlite-read-deadline', '1749-vector'} for t in threading.enumerate())
    assert vs.connection.execute('SELECT 1').fetchone()[0] == 1


def test_fixed_30_query_recall_ids_order_and_payload_identity(seeded, monkeypatch):
    engine, _, _ = seeded
    class Provider:
        provider_id, model_id, dim = 'mock', 'row-model', 16
        def embed_query(self, text):
            return [1.0]*16
    monkeypatch.setattr(tools, 'resolve_provider', lambda config: Provider())
    monkeypatch.setattr(tools.time, 'time', lambda: 1800000000.0)
    stock = stock_functions()
    for i in range(30):
        for filters in ({}, {'exclude_session_ids': ['session-0']},
                        {'include': 'summaries'},
                        {'exclude_session_ids': ['session-0'], 'include': 'verbatim'}):
            args = dict(query=f'topicword{i}', scope_bias=0.0, **filters)
            _reset_vector_store_pool()
            with monkeypatch.context() as patch:
                patch.setattr(tools, '_retrieval_candidate_session_ids', stock['_retrieval_candidate_session_ids'])
                for name in ('_load_chunk_binary_matrix', '_load_embedding_binary_matrix', '_load_binary_with_deadline'):
                    patch.setattr(vm.VectorStore, name, stock[name])
                expected = json.loads(tools.lcm_recall(args, engine=engine))
            _reset_vector_store_pool()
            actual = json.loads(tools.lcm_recall(args, engine=engine))
            assert actual.get('hits'), actual
            assert actual['hits'] == expected['hits']
            assert hashlib.sha256(json.dumps(actual['hits'], sort_keys=True).encode()).digest() == hashlib.sha256(
                json.dumps(expected['hits'], sort_keys=True).encode()).digest()
            assert actual['provenance'] == expected['provenance']
            for arm in ('summary', 'chunk'):
                if arm in actual['provenance']['arms_run']:
                    assert actual['provenance']['coverage'][arm] == 'full_approx'

            # lcm_recall has no time/source filter contract. Exercise those
            # supported filters on lcm_grep's semantic path with the same queries.
            grep_args = dict(query=f'topicword{i}', mode='semantic', limit=8, session_scope='all',
                             time_from=1700000010.0, time_to=1700000060.0,
                             source='telegram', **({'session_id': 'session-2'} if i % 2 else {}))
            if i % 2:
                grep_args['session_scope'] = 'session'
            _reset_vector_store_pool()
            with monkeypatch.context() as patch:
                for name in ('_load_chunk_binary_matrix', '_load_embedding_binary_matrix', '_load_binary_with_deadline'):
                    patch.setattr(vm.VectorStore, name, stock[name])
                expected_grep = json.loads(tools.lcm_grep(grep_args, engine=engine))
            _reset_vector_store_pool()
            actual_grep = json.loads(tools.lcm_grep(grep_args, engine=engine))
            if i % 2 == 0:
                assert actual_grep.get('results'), actual_grep
            assert actual_grep['results'] == expected_grep['results']
