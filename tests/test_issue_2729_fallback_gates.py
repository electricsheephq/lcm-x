"""#2729: durable ownership, whole-prefix admission and atomic late-end retries."""

import copy
import logging
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from hermes_lcm import ingest_ledger
from tests.test_host_uid_shadow import _engine
from tests.test_issue_2055_ingest_ledger_writes import engines as engines
from tests.test_issue_2729_offcurrent_end_after_restart import _start
from tests.invariants.test_issue_2539_offcurrent_marker_end import _setup as marker_setup

STALE = 'LCM ignored unverified stale session-end callback for S2 while bound to S1'
HISTORY = [dict(role='user', content='question', timestamp=1., message_uid='u1'),
           dict(role='assistant', content='reply', timestamp=2., message_uid='u2')]
VIEW = HISTORY + [dict(role='assistant', content='FINAL', timestamp=3., message_uid='u3')]


def lifecycle(engine, conversations=('conv-S1', 'conv-S2')):
    return ([asdict(s) if (s := engine._lifecycle.get_by_conversation(cid)) else None for cid in conversations],
            engine._session_id, engine._conversation_id, engine._embedding_maintenance_retired)


def assert_rows(engine, expected, before_lifecycle, *, session='S2', conversation='conv-S2',
                conversations=('conv-S1', 'conv-S2')):
    rows = engine._store.get_session_messages(session)
    assert [r['content'] for r in rows] == [r['content'] for r in expected]
    assert Counter(r['content'] for r in rows) == Counter(r['content'] for r in expected)
    assert [r['conversation_id'] for r in rows] == [conversation] * len(expected)
    assert not engine._store.get_session_messages('S1' if session == 'S2' else 'S2')
    assert lifecycle(engine, conversations) == before_lifecycle


@pytest.fixture
def cold(engines):
    made = []

    def create(*, history=None, name='case'):
        engine = engines(name, sessions=[('S1', None, None), ('S2', None, None)])
        _start(engine, 'S2')
        engine.ingest(copy.deepcopy(HISTORY if history is None else history))
        _start(engine, 'S1')
        home = Path(engine._hermes_home)
        engine.shutdown()
        replacement = _engine(home, session='S1', conversation='conv-S1')
        made.append(replacement)
        assert not replacement._lcm_session_last_normal_conversation_id.get('S2')
        return replacement

    yield create
    for engine in reversed(made):
        engine.shutdown()


def test_foreign_profile_shared_store(cold, tmp_path, caplog):
    engine = cold()
    other = tmp_path / 'foreign-home'
    other.mkdir()
    store = engine._store
    _start(engine, 'S1', hermes_home=str(other))
    assert engine._store is store
    before = lifecycle(engine)
    with caplog.at_level(logging.WARNING):
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert_rows(engine, HISTORY, before)
    assert STALE in caplog.text and 'gate 1: foreign home' in caplog.text
    assert not engine._pending_session_tails


@pytest.mark.parametrize('shape', ['advanced', 'diverged', 'after-eight', 'stamp', 'uid'])
def test_replaced_or_advanced_sequence_refuses(cold, shape, caplog):
    history = HISTORY if shape != 'after-eight' else [dict(role='user', content=f'row {i}', timestamp=float(i + 1))
                                                    for i in range(10)]
    engine = cold(history=history)
    view = copy.deepcopy(history) + [dict(role='assistant', content='FINAL', timestamp=30.)]
    expected = copy.deepcopy(history)
    if shape == 'advanced':
        engine._append_off_current_session_end_suffix('S2', [dict(role='assistant', content='newer', timestamp=4.)],
            source='cli', conversation_id='conv-S2')
        expected.append(dict(content='newer'))
        view = copy.deepcopy(history)
    elif shape in ('diverged', 'after-eight'):
        view[0 if shape == 'diverged' else 9]['content'] = 'different history'
    elif shape == 'stamp':
        view[0]['timestamp'] = 99.
    else:
        view[0]['message_uid'] = 'foreign-uid'
    before = lifecycle(engine)
    engine.on_session_end('S2', view)
    assert_rows(engine, expected, before)
    assert STALE in caplog.text and 'gate 3:' in caplog.text
    assert not engine._pending_session_tails


@pytest.mark.parametrize('state', ['off', 'not-ready', 'missing', 'provisional', 'foreign-root'])
def test_unproven_lineage_refuses(cold, monkeypatch, caplog, state):
    engine = cold()
    conn = engine._store._conn
    if state == 'off':
        monkeypatch.setenv('LCM_INGEST_LEDGER', 'off')
    elif state == 'not-ready':
        monkeypatch.setattr(engine._store, 'ingest_ledger_state', lambda: 'absent')
    elif state == 'missing':
        conn.execute("DELETE FROM ledger_meta WHERE key='session_lineage:S2'")
        conn.commit()
    else:
        conn.execute("UPDATE ledger_meta SET value=? WHERE key='session_lineage:S2'",
                     ('provisional:S2' if state == 'provisional' else 'foreign:S2',))
        conn.commit()
    before = lifecycle(engine)
    engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert_rows(engine, HISTORY, before)
    assert STALE in caplog.text and 'gate 1:' in caplog.text
    assert not engine._pending_session_tails


@pytest.mark.parametrize('conversation', ['other-conversation', '', None])
def test_conversation_ambiguity_refuses(cold, conversation, caplog):
    engine = cold()
    conn = engine._store._conn
    conn.execute("UPDATE messages SET conversation_id=? WHERE session_id='S2' AND content='reply'", (conversation,))
    conn.commit()
    rows_before = engine._store.get_session_messages('S2')
    before = lifecycle(engine)
    engine.on_session_end('S2', copy.deepcopy(VIEW))
    rows = engine._store.get_session_messages('S2')
    assert rows == rows_before
    assert Counter(r['content'] for r in rows) == Counter(r['content'] for r in HISTORY)
    assert [r['conversation_id'] for r in rows] == ['conv-S2', conversation or '']
    assert not engine._store.get_session_messages('S1')
    assert lifecycle(engine) == before
    assert STALE in caplog.text and 'gate 2:' in caplog.text


def test_repeated_end_and_restart_store_suffix_once(cold, caplog):
    engine = cold()
    home = Path(engine._hermes_home)
    before = lifecycle(engine)
    with caplog.at_level(logging.INFO):
        engine.on_session_end('S2', copy.deepcopy(VIEW))
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert_rows(engine, VIEW, before)
    infos = [r.message for r in caplog.records if 'durable off-current session-end fallback stored' in r.message]
    assert infos == ['LCM durable off-current session-end fallback stored session=S2 rows=1']
    engine.shutdown()
    replacement = _engine(home, session='S1', conversation='conv-S1')
    try:
        before = lifecycle(replacement)
        replacement.on_session_end('S2', copy.deepcopy(VIEW))
        assert_rows(replacement, VIEW, before)
    finally:
        replacement.shutdown()


def test_two_connections_race_after_prefix_read(cold, monkeypatch):
    first = cold()
    second = _engine(Path(first._hermes_home), session='S1', conversation='conv-S1')
    barrier = threading.Barrier(2)
    seen = Counter()
    try:
        for engine in (first, second):
            original = engine._off_current_fallback_prefix

            def after_read(tail, rows, lineage, original=original, engine=engine):
                result = original(tail, rows, lineage)
                seen[id(engine)] += 1
                if seen[id(engine)] == 1:
                    assert len(rows) == len(HISTORY)
                    barrier.wait(timeout=10)
                return result

            monkeypatch.setattr(engine, '_off_current_fallback_prefix', after_read)
        before = lifecycle(first)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(e.on_session_end, 'S2', copy.deepcopy(VIEW)) for e in (first, second)]
            for future in futures:
                future.result(timeout=20)
        assert all(n == 2 for n in seen.values())
        assert_rows(first, VIEW, before)
        assert_rows(second, VIEW, before)
    finally:
        second.shutdown()


@contextmanager
def busy(**kwargs):
    raise sqlite3.OperationalError('database is locked')
    yield  # pragma: no cover


@pytest.mark.parametrize('retry_change', ['none', 'foreign-home', 'lineage', 'divergence', 'conversation'])
def test_busy_append_revalidates_owned_pending_tail(cold, monkeypatch, retry_change, tmp_path, caplog):
    engine = cold()
    store, home = engine._store, engine._hermes_home
    before = lifecycle(engine)
    with monkeypatch.context() as patch:
        patch.setattr(store, 'ingest_atomic', busy)
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert_rows(engine, HISTORY, before)
    assert len(engine._pending_session_tails) == 1
    tail = engine._pending_session_tails[0]
    assert tail['store'] is store
    assert (tail['hermes_home'], tail['session_id'], tail['conversation_id']) == (home, 'S2', 'conv-S2')
    assert tail['fallback_lineage'] == ingest_ledger.persisted_lineage(store._conn, 'S2', None)
    if retry_change == 'foreign-home':
        other = tmp_path / 'retry-home'
        other.mkdir()
        _start(engine, 'S1', hermes_home=str(other))
        assert engine._store is store
    elif retry_change == 'lineage':
        store._conn.execute("UPDATE ledger_meta SET value='foreign:S2' WHERE key='session_lineage:S2'")
        store._conn.commit()
    elif retry_change == 'divergence':
        store._conn.execute("UPDATE messages SET content='advanced rewrite' WHERE session_id='S2' AND content='reply'")
        store._conn.commit()
    elif retry_change == 'conversation':
        store._conn.execute("UPDATE messages SET conversation_id='replaced' WHERE session_id='S2'")
        store._conn.commit()
    before = lifecycle(engine)
    engine._retry_pending_session_tails()
    engine._retry_pending_session_tails()
    assert not engine._pending_session_tails
    if retry_change in ('none', 'foreign-home'):
        # The queue retains its original proven home across a later profile switch.
        assert_rows(engine, VIEW, before)
    elif retry_change == 'lineage':
        assert_rows(engine, HISTORY, before)
        assert 'gate 1:' in caplog.text
    elif retry_change == 'divergence':
        assert_rows(engine, [HISTORY[0], dict(content='advanced rewrite')], before)
        assert 'gate 3:' in caplog.text
    else:
        assert_rows(engine, HISTORY, before, conversation='replaced')
        assert 'gate 2:' in caplog.text


def test_busy_ownership_lookup_never_queues(cold, monkeypatch, caplog):
    engine = cold()
    before = lifecycle(engine)
    with monkeypatch.context() as patch:
        def locked(*args):
            raise sqlite3.OperationalError('database is locked')
        patch.setattr(ingest_ledger, 'persisted_lineage', locked)
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert_rows(engine, HISTORY, before)
    assert not engine._pending_session_tails
    assert STALE in caplog.text and 'gate 1:' in caplog.text


@pytest.mark.parametrize('kind', ['spillover', 'tempdir', 'nofile', 'nofile-tempdir', 'plain', 'pruned'])
def test_marker_forms_after_restart(engines, monkeypatch, tmp_path, kind):
    engine, view, _, n = marker_setup(engines, monkeypatch, tmp_path, kind)
    home = Path(engine._hermes_home)
    engine.shutdown()
    replacement = _engine(home, session='S2', conversation='conv2')
    try:
        before = lifecycle(replacement, ('conv1', 'conv2'))
        replacement.on_session_end('S1', copy.deepcopy(view))
        replacement.on_session_end('S1', copy.deepcopy(view))
        rows = replacement._store.get_session_messages('S1')
        assert len(rows) == n + 1
        assert Counter(r['content'] for r in rows)['FINAL'] == 1
        assert all(r['conversation_id'] == 'conv1' for r in rows)
        assert all(replacement._session_end_rows_equal(r, m, session_id='S1') for r, m in zip(rows, view))
        assert not replacement._store.get_session_messages('S2')
        assert lifecycle(replacement, ('conv1', 'conv2')) == before
    finally:
        replacement.shutdown()


@pytest.mark.parametrize('change', ['lineage', 'conversation', 'prefix', 'advanced'])
def test_atomic_recheck_after_initial_proof(cold, monkeypatch, change, caplog):
    engine = cold()
    store = engine._store
    original = store.ingest_atomic
    expected = copy.deepcopy(HISTORY)

    @contextmanager
    def changed_between_reads(**kwargs):
        if change == 'lineage':
            store._conn.execute("UPDATE ledger_meta SET value='foreign:S2' WHERE key='session_lineage:S2'")
        elif change == 'conversation':
            store._conn.execute("UPDATE messages SET conversation_id='replaced' WHERE session_id='S2'")
        elif change == 'prefix':
            store._conn.execute("UPDATE messages SET content='changed' WHERE session_id='S2' AND content='reply'")
            expected[1]['content'] = 'changed'
        else:
            engine._append_off_current_session_end_suffix('S2', [dict(role='assistant', content='advanced', timestamp=4.)],
                source='cli', conversation_id='conv-S2')
            expected.append(dict(content='advanced'))
        store._conn.commit()
        with original(**kwargs) as timing:
            yield timing

    before = lifecycle(engine)
    with monkeypatch.context() as patch:
        patch.setattr(store, 'ingest_atomic', changed_between_reads)
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert_rows(engine, expected, before, conversation='replaced' if change == 'conversation' else 'conv-S2')
    assert STALE in caplog.text


def test_session_reads_are_bounded_and_atomic(cold, monkeypatch):
    engine = cold()
    store, reads = engine._store, []
    original = store.get_range

    def bounded(session, *args, **kwargs):
        if session == 'S2':
            reads.append((kwargs, store._conn.in_transaction))
        return original(session, *args, **kwargs)

    monkeypatch.setattr(store, 'get_range', bounded)
    before = lifecycle(engine)
    engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert_rows(engine, VIEW, before)
    assert reads == [({'limit': len(VIEW) + 1}, False), ({'limit': len(VIEW) + 1}, True)]
