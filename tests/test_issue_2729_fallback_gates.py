"""#2729: durable ownership, whole-prefix admission and atomic late-end retries."""

import base64
import copy
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from hermes_lcm import ingest_ledger, maintenance_scheduler
from hermes_lcm.externalize import get_large_output_storage_dir
from hermes_lcm.store import MessageStore
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


@pytest.mark.parametrize('n', [10, 100])
def test_matching_uid_gate3_computes_no_identity_digests(cold, monkeypatch, n):
    history = [dict(role='user' if i % 2 == 0 else 'assistant', content=f'row {i}',
                    timestamp=float(i + 1), message_uid=f'uid-{i}') for i in range(n)]
    engine = cold(history=history)
    prefix, digest = engine._off_current_fallback_prefix, ingest_ledger.identity_digest
    calls, passes = [], []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return digest(*args, **kwargs)

    def gate3(*args):
        passes.append(True)
        with monkeypatch.context() as patch:
            patch.setattr(ingest_ledger, 'identity_digest', counted)
            return prefix(*args)

    monkeypatch.setattr(engine, '_off_current_fallback_prefix', gate3)
    view = history + [dict(role='assistant', content='FINAL', timestamp=float(n + 1), message_uid='final-uid')]
    before = lifecycle(engine)
    engine.on_session_end('S2', copy.deepcopy(view))
    assert passes
    assert calls == []
    assert_rows(engine, view, before)
    assert fallback_counts(engine) == {'stored': 1}


@pytest.mark.parametrize('ingests', [(4,), (1, 2, 3, 4)], ids=['single', 'growing'])
@pytest.mark.parametrize('matching', [False, True], ids=['foreign-uid', 'matching-uid'])
@pytest.mark.parametrize('stamped', [True, False], ids=['same-stamp', 'absent-stamp'])
def test_retained_view_uid_veto_after_restart(engines, caplog, ingests, matching, stamped):
    history = HISTORY + [dict(role='user', content='more', timestamp=3., message_uid='u3'),
                         dict(role='assistant', content='answer', timestamp=4., message_uid='u4')]
    engine = engines(sessions=[('S1', None, None), ('S2', None, None)])
    _start(engine, 'S2')
    for count in ingests:
        engine.ingest(copy.deepcopy(history[:count]))
    _start(engine, 'S1')
    home = Path(engine._hermes_home)
    engine.shutdown()
    replacement = _engine(home, session='S1', conversation='conv-S1')
    try:
        conn = replacement._store._conn
        lineage = ingest_ledger.persisted_lineage(conn, 'S2', None)
        retained = list(conn.execute("SELECT view_seq,entries FROM ingest_views WHERE lineage=? AND kind='ingest'",
                                     (lineage,)))
        assert len(retained) == min(2, len(ingests))
        if len(ingests) > 1:
            first = replacement._store.get_session_messages('S2')[0]['store_id']
            seq, occ = conn.execute('SELECT view_seq,occ_id FROM occurrence_bindings WHERE lineage=? AND store_id=?',
                                    (lineage, first)).fetchone()
            assert seq not in {seq for seq, _ in retained}
            assert all(ingest_ledger.decode_entries(blob)[0].ref != occ for _, blob in retained)
        view = copy.deepcopy(history) + [dict(role='assistant', content='FINAL', timestamp=5., message_uid='u5')]
        if not matching:
            view[0]['message_uid'] = 'foreign-uid'
        if not stamped:
            del view[0]['timestamp']
        before = lifecycle(replacement)
        rows_before = replacement._store.get_session_messages('S2')
        for _ in range(2):
            replacement.on_session_end('S2', copy.deepcopy(view))
        assert_rows(replacement, view if matching else history, before)
        assert replacement._store.get_session_messages('S2')[:len(history)] == rows_before
        assert not replacement._pending_session_tails
        assert fallback_counts(replacement) == ({'stored': 1, 'noop': 1} if matching else {'refused.gate3': 2})
        if not matching:
            assert 'gate 3: occurrence UID mismatch' in caplog.text
    finally:
        replacement.shutdown()


def test_closed_fallback_store_counts_unbound_then_drains_once(cold, monkeypatch):
    engine = cold()
    before = lifecycle(engine)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, 'ingest_atomic', busy)
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert len(engine._pending_session_tails) == 1
    tail = engine._pending_session_tails[0]
    assert 'fallback_lineage' in tail
    closed = MessageStore(engine._store.db_path)
    closed.close()
    tail['store'] = closed
    counters = engine._ingest_ledger_counters
    unbound, failed = (counters.get(key, 0) for key in ('ledger.tail.cold_store_unbound', 'ledger.tail.retry_failed'))
    for attempt in range(1, 3):
        engine._retry_pending_session_tails()
        assert counters.get('ledger.tail.cold_store_unbound', 0) == unbound + attempt
        assert counters.get('ledger.tail.retry_failed', 0) == failed
        assert engine._pending_session_tails == [tail]
        assert_rows(engine, HISTORY, before)
    tail['store'] = engine._store
    for _ in range(2):
        engine._retry_pending_session_tails()
        assert not engine._pending_session_tails
        assert_rows(engine, VIEW, before)


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
        assert sorted(seen.values()) == [1, 2]  # Only the losing racer repeats the full comparison.
        assert_rows(first, VIEW, before)
        assert_rows(second, VIEW, before)
    finally:
        second.shutdown()


@contextmanager
def busy(**kwargs):
    raise sqlite3.OperationalError('database is locked')
    yield  # pragma: no cover


@pytest.mark.parametrize('error', [UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid payload'), KeyboardInterrupt()])
@pytest.mark.parametrize('matching', [False, True])
def test_failed_fallback_proof_cannot_authorize_retry(cold, monkeypatch, caplog, error, matching):
    engine = cold()
    before = lifecycle(engine)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, 'ingest_atomic', busy)
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert len(engine._pending_session_tails) == 1
    tail = engine._pending_session_tails[0]
    if not matching:
        tail['view'][0]['content'] = 'different history'
    compare, calls = engine._session_end_rows_equal, []
    def raises_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise error
        return compare(*args, **kwargs)
    monkeypatch.setattr(engine, '_session_end_rows_equal', raises_once)
    engine._retry_pending_session_tails()
    assert engine._pending_session_tails == [tail]
    assert_rows(engine, HISTORY, before)
    engine._retry_pending_session_tails()
    assert_rows(engine, VIEW if matching else HISTORY, before)
    assert not engine._pending_session_tails
    assert len(calls) > 1
    if not matching:
        assert 'gate 3: stored sequence mismatch' in caplog.text
    engine._retry_pending_session_tails()
    assert_rows(engine, VIEW if matching else HISTORY, before)


def test_direct_fallback_begin_uses_session_end_busy_timeout(cold):
    engine = cold()
    conn, observed = engine._store._conn, []
    previous = conn.execute('PRAGMA busy_timeout').fetchone()[0]
    def trace(sql):
        if sql.upper() == 'BEGIN IMMEDIATE':
            observed.append(conn.execute('PRAGMA busy_timeout').fetchone()[0])
    before = lifecycle(engine)
    conn.set_trace_callback(trace)
    try:
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    finally:
        conn.set_trace_callback(None)
    assert observed == [50]
    assert conn.execute('PRAGMA busy_timeout').fetchone()[0] == previous
    assert_rows(engine, VIEW, before)


def test_direct_fallback_busy_queues_lineage_then_drains_once(cold, latency_check):
    engine = cold()
    store = engine._store
    lineage = ingest_ledger.persisted_lineage(store._conn, 'S2', None)
    previous = store._conn.execute('PRAGMA busy_timeout').fetchone()[0]
    before = lifecycle(engine)
    locker = sqlite3.connect(store.db_path)
    try:
        locker.execute('BEGIN IMMEDIATE')
        started = time.monotonic()
        engine.on_session_end('S2', copy.deepcopy(VIEW))
        elapsed = time.monotonic() - started
        if latency_check:
            assert elapsed < 1.0
        assert_rows(engine, HISTORY, before)
        assert len(engine._pending_session_tails) == 1
        tail = engine._pending_session_tails[0]
        assert tail['kind'] == 'cold'
        assert tail['store'] is store
        assert (tail['session_id'], tail['conversation_id'], tail['fallback_lineage']) == ('S2', 'conv-S2', lineage)
        assert tail['view'] == VIEW
        assert store._conn.execute('PRAGMA busy_timeout').fetchone()[0] == previous
    finally:
        locker.rollback()
        locker.close()
    engine._retry_pending_session_tails()
    assert not engine._pending_session_tails
    assert_rows(engine, VIEW, before)
    engine._retry_pending_session_tails()
    assert_rows(engine, VIEW, before)


@pytest.mark.parametrize('foreground', [False, True])
def test_cross_instance_fallback_covers_busy_pending_tail(engines, monkeypatch, foreground):
    first = engines(sessions=[('S1', None, None), ('S2', None, None)])
    _start(first, 'S2')
    first.ingest(copy.deepcopy(HISTORY))
    if not foreground:
        _start(first, 'S1')
    assert first._lcm_session_last_normal_conversation_id['S2'] == 'conv-S2'
    # Start B before queueing so its session-start sibling retry cannot drain A.
    second = _engine(Path(first._hermes_home), session='S1', conversation='conv-S1')
    try:
        with monkeypatch.context() as patch:
            def locked(*args, **kwargs):
                raise sqlite3.OperationalError('database is locked')
            patch.setattr(first._store, 'ingest_atomic' if foreground else 'get_range', busy if foreground else locked)
            first.on_session_end('S2', copy.deepcopy(VIEW))
        assert len(first._pending_session_tails) == 1
        assert 'fallback_lineage' not in first._pending_session_tails[0]
        assert first._pending_session_tails[0].get('kind', 'warm') == ('warm' if foreground else 'cold')
        before = lifecycle(first)
        assert_rows(first, HISTORY, before)
        assert not second._lcm_session_last_normal_conversation_id.get('S2')
        before = lifecycle(second)
        with monkeypatch.context() as patch:
            patch.setattr(maintenance_scheduler, '_PENDING_TAIL_ENGINES',
                          maintenance_scheduler._PENDING_TAIL_ENGINES - {first})
            second.on_session_end('S2', copy.deepcopy(VIEW))
        assert fallback_counts(second) == {'stored': 1}
        assert len(first._pending_session_tails) == 1
        assert_rows(second, VIEW, before)
        before = lifecycle(first)
        def no_append(*args, **kwargs):
            pytest.fail('A must not append after B covered its tail')
        monkeypatch.setattr(first, '_append_ingest_batch', no_append)
        first._retry_pending_session_tails()
        assert not first._pending_session_tails
        assert_rows(first, VIEW, before)
    finally:
        second.shutdown()


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


@pytest.mark.parametrize('sources,expected', [(['A', 'A'], 'A'), (['A', ''], 'A'), (['', ''], 'unknown')])
@pytest.mark.parametrize('queued', [False, True])
def test_fallback_source_belongs_to_ended_session(cold, monkeypatch, sources, expected, queued):
    engine = cold()
    store = engine._store
    for row, source in zip(store.get_session_messages('S2'), sources):
        store._conn.execute('UPDATE messages SET source=? WHERE store_id=?', (source, row['store_id']))
    store._conn.commit()
    engine._session_platform = 'B'
    before = lifecycle(engine)
    with monkeypatch.context() as patch:
        if queued:
            patch.setattr(store, 'ingest_atomic', busy)
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    if queued:
        assert engine._pending_session_tails[0]['source'] == (expected if any(sources) else '')
        engine._session_platform = 'C'
        engine._retry_pending_session_tails()
        assert not engine._pending_session_tails
    assert_rows(engine, VIEW, before)
    assert [r[0] for r in store._conn.execute(
        "SELECT source FROM messages WHERE session_id='S2' ORDER BY store_id")] == sources + [expected]


@pytest.mark.parametrize('fresh', [False, True])
def test_resume_after_fallback_stores_full_history_once(cold, fresh):
    engine = cold()
    engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert fallback_counts(engine) == {'stored': 1}
    home = Path(engine._hermes_home)
    resumed = engine
    if fresh:
        engine.shutdown()
        resumed = _engine(home, session='S2', conversation='conv-S2')
    try:
        resumed.on_session_start('S2', platform='cli', context_length=200_000, conversation_id='conv-S2')
        history = VIEW + [dict(role='user', content='next question', timestamp=4., message_uid='u4')]
        before = lifecycle(resumed)
        resumed.ingest(copy.deepcopy(history))
        assert_rows(resumed, history, before)
    finally:
        if fresh:
            resumed.shutdown()


def fallback_counts(engine):
    return {k.removeprefix('ledger.offcurrent_fallback.'): v
            for k, v in getattr(engine, '_ingest_ledger_counters', {}).items()
            if k.startswith('ledger.offcurrent_fallback.')}


@pytest.mark.parametrize('path', ['stored', 'noop', 'gate1', 'gate2', 'gate3', 'gate4', 'gate5'])
def test_fallback_outcome_counted_once(cold, monkeypatch, caplog, path):
    engine = cold()
    store, view = engine._store, copy.deepcopy(VIEW)
    expected = HISTORY
    if path == 'gate1':
        monkeypatch.setenv('LCM_INGEST_LEDGER', 'off')
    elif path == 'gate2':
        store._conn.execute("UPDATE messages SET conversation_id='' WHERE session_id='S2'")
        store._conn.commit()
    elif path == 'gate3':
        def broken_compare(*args, **kwargs):
            raise ValueError('comparison unavailable')
        monkeypatch.setattr(engine, '_session_end_rows_equal', broken_compare)
    elif path == 'gate4':
        @contextmanager
        def broken_atomic(**kwargs):
            raise ValueError('transaction unavailable')
            yield  # pragma: no cover
        monkeypatch.setattr(store, 'ingest_atomic', broken_atomic)
    elif path == 'gate5':
        monkeypatch.setattr(store, 'ingest_atomic', busy)
        def broken_queue(*args, **kwargs):
            raise ValueError('queue unavailable')
        monkeypatch.setattr(engine, '_queue_cold_session_tail', broken_queue)
    elif path == 'noop':
        view = copy.deepcopy(HISTORY)
    else:
        expected = VIEW
    before = lifecycle(engine)
    with caplog.at_level(logging.DEBUG):
        engine.on_session_end('S2', view)
    assert_rows(engine, expected, before, conversation='' if path == 'gate2' else 'conv-S2')
    assert fallback_counts(engine) == {('refused.' + path if path.startswith('gate') else path): 1}
    if path == 'gate3':
        assert 'gate 3: compare failed' in caplog.text
        assert 'gate 4:' not in caplog.text
    elif path == 'noop':
        assert [r.message for r in caplog.records if 'fallback covered' in r.message] == [
            'LCM durable off-current session-end fallback covered session=S2']


def test_ownership_precedes_rows_and_busy_rows_never_queue(cold, monkeypatch, caplog):
    engine = cold()
    before = lifecycle(engine)
    owner, events = engine._off_current_fallback_owner, []
    def owned(tail):
        events.append('owner')
        return owner(tail)
    def locked(*args, **kwargs):
        events.append('rows')
        raise sqlite3.OperationalError('database is locked')
    with monkeypatch.context() as patch:
        patch.setattr(engine, '_off_current_fallback_owner', owned)
        patch.setattr(engine._store, 'get_range', locked)
        engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert events == ['owner', 'rows']
    assert not engine._pending_session_tails
    assert_rows(engine, HISTORY, before)
    assert fallback_counts(engine) == {'refused.gate1': 1}
    assert 'gate 1: ownership proof unavailable' in caplog.text


@pytest.mark.parametrize('change', ['none', 'binding-stamp', 'view-uid', 'salt'])
def test_snapshot_checks_binding_and_view_contents(cold, monkeypatch, change):
    engine = cold()
    store, calls = engine._store, []
    atomic, compare = store.ingest_atomic, engine._off_current_fallback_prefix
    def counted(*args):
        calls.append(store._conn.in_transaction)
        return compare(*args)
    @contextmanager
    def changed(**kwargs):
        conn = store._conn
        if change == 'binding-stamp':
            conn.execute("UPDATE occurrence_bindings SET stamp='99.0' WHERE store_id="
                         "(SELECT min(store_id) FROM messages WHERE session_id='S2')")
        elif change == 'view-uid':
            lineage = ingest_ledger.persisted_lineage(conn, 'S2', None)
            seq, blob = conn.execute("SELECT view_seq,entries FROM ingest_views WHERE lineage=? AND kind='ingest'",
                                    (lineage,)).fetchone()
            entries = ingest_ledger.decode_entries(blob)
            entries[0] = entries[0]._replace(uidset_digest=b'x' * 16)
            conn.execute('UPDATE ingest_views SET entries=? WHERE view_seq=?',
                         (ingest_ledger.encode_entries(entries), seq))
        elif change == 'salt':
            conn.execute("UPDATE ledger_meta SET value=? WHERE key='digest_salt'", (b'x' * 32,))
        conn.commit()
        with atomic(**kwargs) as timing:
            yield timing
    monkeypatch.setattr(store, 'ingest_atomic', changed)
    monkeypatch.setattr(engine, '_off_current_fallback_prefix', counted)
    before = lifecycle(engine)
    engine.on_session_end('S2', copy.deepcopy(VIEW))
    assert_rows(engine, VIEW if change == 'none' else HISTORY, before)
    assert calls == ([False] if change in ('none', 'view-uid') else [False, True])
    assert fallback_counts(engine) == ({'stored': 1} if change == 'none' else {'refused.gate3': 1})


@pytest.mark.parametrize('change', ['config', 'view', 'payload'])
def test_snapshot_checks_mutable_comparison_dependencies(cold, monkeypatch, change):
    engine = cold()
    store, view, calls = engine._store, copy.deepcopy(VIEW), []
    payload = None
    if change == 'payload':
        directory = get_large_output_storage_dir(engine._config, hermes_home=engine._hermes_home, create=True)
        payload = directory / 'snapshot-probe.dat'
        payload.write_text(json.dumps(dict(content='question', session_id='S2', kind='tool_result')))
        store._conn.execute("UPDATE messages SET content=? WHERE session_id='S2' AND content='question'",
                            ('[Externalized tool output: chars=8; ref=snapshot-probe.dat]',))
        store._conn.commit()
    atomic, compare = store.ingest_atomic, engine._off_current_fallback_prefix
    def counted(*args):
        calls.append(store._conn.in_transaction)
        return compare(*args)
    @contextmanager
    def changed(**kwargs):
        if change == 'config':
            engine._config.sensitive_patterns_enabled = not engine._config.sensitive_patterns_enabled
        elif change == 'view':
            view[0]['content'] = 'different host row'
        else:
            old = payload.stat()
            payload.write_text(json.dumps(dict(content='tampered', session_id='S2', kind='tool_result')))
            os.utime(payload, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
            assert payload.stat().st_size == old.st_size
            assert payload.stat().st_mtime_ns != old.st_mtime_ns
        with atomic(**kwargs) as timing:
            yield timing
    monkeypatch.setattr(store, 'ingest_atomic', changed)
    monkeypatch.setattr(engine, '_off_current_fallback_prefix', counted)
    before = lifecycle(engine)
    rows_before = store.get_session_messages('S2')
    engine.on_session_end('S2', view)
    assert calls == [False, True]
    if change == 'config':
        assert_rows(engine, VIEW, before)
    else:
        assert store.get_session_messages('S2') == rows_before
        assert lifecycle(engine) == before
        assert not store.get_session_messages('S1')
    assert fallback_counts(engine) == ({'stored': 1} if change == 'config' else {'refused.gate3': 1})


@pytest.mark.parametrize('missing', [False, True])
def test_snapshot_discovers_nested_marker_and_revalidates_without_file_reads(cold, monkeypatch, missing):
    from tests.test_issue_10_persisted_output_spillover import _write_output
    import hermes_lcm.engine as engine_module

    engine = cold()
    home = Path(engine._hermes_home)
    directory = get_large_output_storage_dir(engine._config, hermes_home=str(home), create=True)
    output = home / 'cache' / 'spillover' / 'nested.txt'
    marker = _write_output(output)
    inner, outer = directory / 'inner.json', directory / 'outer.json'
    def placeholder(name):
        return f'[Externalized tool output: chars=8; ref={name}]'
    if not missing:
        inner.write_text(json.dumps(dict(content=marker)))
    outer.write_text(json.dumps(dict(content=placeholder(inner.name))))
    tail = dict(store=engine._store, hermes_home=str(home), session_id='S2',
                view=[dict(content=placeholder(outer.name))])
    lineage = ingest_ledger.persisted_lineage(engine._store._conn, 'S2', None)
    reads, read_bytes = [], Path.read_bytes
    def counted(path):
        reads.append(str(path))
        return read_bytes(path)
    def unexpected(*args, **kwargs):
        pytest.fail('unchanged discovery must not load or compare payload content')
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', counted)
        patch.setattr(engine, '_session_end_prefix_compare_value', unexpected)
        first = engine._off_current_fallback_inputs(tail, [], lineage)
    expected = {str(outer), str(inner)} | (set() if missing else {str(output)})
    assert set(first[6]) == expected
    assert reads == [str(outer)] + ([] if missing else [str(inner)])
    for path, generation in first[6].items():
        stat = Path(path).stat() if Path(path).exists() else None
        assert generation == ((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) if stat else None)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', unexpected)
        patch.setattr(Path, 'read_text', unexpected)
        patch.setattr(engine_module, 'recover_hermes_persisted_output_with_file_stat', unexpected)
        patch.setattr(engine, '_session_end_prefix_compare_value', unexpected)
        with engine._store.ingest_atomic():
            second = engine._off_current_fallback_inputs(tail, [], lineage)
    assert second == first
    if missing:
        inner.write_text(json.dumps(dict(content=marker)))
    else:
        old = output.stat()
        os.utime(output, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
    assert engine._off_current_fallback_inputs(tail, [], lineage) != first


def test_snapshot_reuses_repeated_embedded_reference_discovery(cold, monkeypatch):
    engine = cold()
    directory = get_large_output_storage_dir(engine._config, hermes_home=engine._hermes_home, create=True)
    for name in ('embedded.json', 'unrelated.json'):
        (directory / name).write_text(json.dumps(dict(content='payload', kind='ingest_payload', session_id='S2')))
    nested = directory / 'nested.json'
    nested.write_text(json.dumps(dict(content='nested payload')))
    (directory / 'embedded.json').write_text(json.dumps(dict(kind='ingest_payload', session_id='S2',
        content='[Externalized tool output: chars=14; ref=nested.json]')))
    embedded = 'before [Externalized LCM ingest payload: chars=7; ref=embedded.json] after'
    tail = dict(store=engine._store, hermes_home=engine._hermes_home, session_id='S2', view=[
        dict(content=embedded),
        dict(content='[Externalized tool output: chars=7; ref=unrelated.json]'),
        dict(content=embedded)])
    lineage = ingest_ledger.persisted_lineage(engine._store._conn, 'S2', None)
    calls, compare = [], engine._session_end_prefix_compare_value
    def counted(*args, **kwargs):
        calls.append(args[0])
        return compare(*args, **kwargs)
    monkeypatch.setattr(engine, '_session_end_prefix_compare_value', counted)
    first = engine._off_current_fallback_inputs(tail, [], lineage)
    assert calls == [embedded]
    def unexpected(*args, **kwargs):
        pytest.fail('unchanged embedded references must not read payload bytes in the lock')
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', unexpected)
        patch.setattr(Path, 'read_text', unexpected)
        patch.setattr(engine, '_session_end_prefix_compare_value', unexpected)
        with engine._store.ingest_atomic():
            assert engine._off_current_fallback_inputs(tail, [], lineage) == first
    old = nested.stat()
    nested.write_text(json.dumps(dict(content='changed nested payload')))
    os.utime(nested, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
    assert engine._off_current_fallback_inputs(tail, [], lineage) != first
    assert calls == [embedded]


def test_embedded_reference_in_lock_stats_scale_linearly(cold, monkeypatch, record_property):
    counts = {}
    for n in (100, 1000):
        history = []
        for i in range(n):
            blob = base64.b64encode(hashlib.sha256(str(i).encode()).digest() * 40).decode()
            history.append(dict(role='user' if i % 2 == 0 else 'assistant',
                content=f'shot {i} data:image/png;base64,{blob} end', timestamp=float(i + 1)))
        engine = cold(history=history, name=f'embedded-{n}')
        store = engine._store
        before = store.get_session_messages('S2')
        assert len(before) == n
        assert all(' [Externalized LCM ingest payload:' in row['content'] for row in before)
        snapshot, stat = engine._off_current_fallback_inputs, Path.stat
        counts[n] = 0
        def counted_stat(path, *args, **kwargs):
            counts[n] += 1
            return stat(path, *args, **kwargs)
        def counted_snapshot(*args):
            if not store._conn.in_transaction:
                return snapshot(*args)
            with monkeypatch.context() as patch:
                patch.setattr(Path, 'stat', counted_stat)
                return snapshot(*args)
        with monkeypatch.context() as patch:
            patch.setattr(engine, '_off_current_fallback_inputs', counted_snapshot)
            engine.on_session_end('S2', copy.deepcopy(history) + [
                dict(role='assistant', content='FINAL', timestamp=float(n + 1))])
        after = store.get_session_messages('S2')
        assert after[:-1] == before
        assert after[-1]['content'] == 'FINAL'
        assert fallback_counts(engine) == {'stored': 1}
        assert counts[n] > 0
        record_property(f'in_lock_stat_calls_{n}', counts[n])
    assert counts[1000] <= 3 * 1000
    assert counts[1000] <= 12 * counts[100]


@pytest.mark.parametrize('agreeing', [False, True], ids=['summary-diverges', 'identical-history'])
def test_retained_compression_child_uid_veto_requires_agreeing_history(engines, agreeing, caplog):
    engine = engines(sessions=[('S1', None, None), ('S2', None, 'compression'), ('S3', 'S2', None)])
    _start(engine, 'S2')
    engine.ingest(copy.deepcopy(HISTORY))
    lineage = ingest_ledger.persisted_lineage(engine._store._conn, 'S2', None)
    _start(engine, 'S3')
    child = copy.deepcopy(HISTORY)
    if not agreeing:
        child[0]['content'] = 'compression summary of the parent'
    child[1]['message_uid'] = 'child-occurrence'
    engine.ingest(child)
    conn = engine._store._conn
    assert ingest_ledger.persisted_lineage(conn, 'S3', None) == lineage
    entries = ingest_ledger.decode_entries(ingest_ledger.newest_view(conn, lineage, 'ingest')['entries'])
    salt = engine._store.ingest_ledger_salt()
    assert entries[1].identity_digest == ingest_ledger.identity_digest(salt, HISTORY[1])
    assert entries[1].uidset_digest != ingest_ledger.uidset_digest(salt, HISTORY[1]['message_uid'], [])
    assert (entries[0].identity_digest == ingest_ledger.identity_digest(salt, HISTORY[0])) == agreeing
    parent_before = engine._store.get_session_messages('S2')
    child_before = engine._store.get_session_messages('S3')
    home = Path(engine._hermes_home)
    engine.shutdown()
    replacement = _engine(home, session='S1', conversation='conv-S1')
    try:
        before = lifecycle(replacement)
        assert not replacement._lcm_session_last_normal_conversation_id.get('S2')
        with caplog.at_level(logging.WARNING):
            replacement.on_session_end('S2', copy.deepcopy(VIEW))
            replacement.on_session_end('S2', copy.deepcopy(VIEW))
        rows = replacement._store.get_session_messages('S2')
        if agreeing:
            assert rows == parent_before
            assert 'gate 3: occurrence UID mismatch' in caplog.text
            assert fallback_counts(replacement) == {'refused.gate3': 2}
        else:
            assert 'occurrence UID mismatch' not in caplog.text
            assert rows[:-1] == parent_before
            assert [row['content'] for row in rows] == [row['content'] for row in VIEW]
            assert Counter(row['content'] for row in rows)['FINAL'] == 1
            assert fallback_counts(replacement) == {'stored': 1, 'noop': 1}
        assert replacement._store.get_session_messages('S3') == child_before
        assert not replacement._store.get_session_messages('S1')
        assert lifecycle(replacement) == before
        assert not replacement._pending_session_tails
    finally:
        replacement.shutdown()


@pytest.fixture
def ingest_during_parent_end(engines, monkeypatch):
    """T8: commit a second connection's ingest after the outer proof, before retry."""
    monkeypatch.setattr(time, 'time', lambda: 1_700_000_000.)

    def run(shape, *, force_full=False):
        engine = engines(f'{shape}-{force_full}', sessions=[
            ('S1', None, None), ('S2', None, 'compression'), ('S3', 'S2', None)])
        _start(engine, 'S2')
        engine.ingest(copy.deepcopy(HISTORY))
        home = Path(engine._hermes_home)
        engine.shutdown()
        parent = _engine(home, session='S1', conversation='conv-S1')
        child = _engine(home, session='S1' if shape == 'unrelated' else 'S3',
                        conversation='conv-S1' if shape == 'unrelated' else 'conv-S3')
        before = {sid: parent._store.get_session_messages(sid) for sid in ('S1', 'S2', 'S3')}
        calls, reasons, comparing, retrying = [], [], [], False
        prefix, retry = parent._off_current_fallback_prefix, parent._retry_off_current_fallback
        equal, warn = parent._session_end_rows_equal, parent._warn_off_current_fallback

        def counted_prefix(*args):
            calls.append(parent._store._conn.in_transaction)
            return prefix(*args)

        def counted_equal(*args, **kwargs):
            if retrying:
                comparing.append(parent._store._conn.in_transaction)
            return equal(*args, **kwargs)

        def warned(session, reason):
            reasons.append(reason)
            return warn(session, reason)

        def retry_after_ingest(tail):
            nonlocal retrying
            view = copy.deepcopy(HISTORY)
            if shape == 'divergent':
                view[0]['content'] = 'compression summary of the parent'
            view[1]['message_uid'] = 'child-occurrence'
            child.ingest(view)
            assert not child._store._conn.in_transaction
            lineage = ingest_ledger.persisted_lineage(parent._store._conn, 'S2', None)
            inputs = parent._off_current_fallback_inputs(tail, before['S2'], lineage)
            snapshot = tail['fallback_snapshot']
            assert inputs[:3] + inputs[4:] == snapshot[:3] + snapshot[4:]
            assert (inputs[3] == snapshot[3]) == (shape == 'unrelated')
            if force_full:
                tail['fallback_snapshot'] = snapshot[:4] + (b'force-full-rerun',) + snapshot[5:]
            retrying = True
            try:
                return retry(tail)
            finally:
                retrying = False

        try:
            with monkeypatch.context() as patch:
                patch.setattr(parent, '_off_current_fallback_prefix', counted_prefix)
                patch.setattr(parent, '_session_end_rows_equal', counted_equal)
                patch.setattr(parent, '_warn_off_current_fallback', warned)
                patch.setattr(parent, '_retry_off_current_fallback', retry_after_ingest)
                parent.on_session_end('S2', copy.deepcopy(VIEW))
            rows = {sid: parent._store.get_session_messages(sid) for sid in before}
            counts = fallback_counts(parent)
            if shape == 'agreeing':
                assert rows['S2'] == before['S2']
                assert reasons == ['gate 3: occurrence UID mismatch']
                assert counts == {'refused.gate3': 1}
            else:
                assert rows['S2'][:-1] == before['S2']
                assert Counter(row['content'] for row in rows['S2'])['FINAL'] == 1
                assert reasons == []
                assert counts == {'stored': 1}
            target = 'S1' if shape == 'unrelated' else 'S3'
            assert rows[target] == child._store.get_session_messages(target)
            assert rows['S3' if target == 'S1' else 'S1'] == []
            if force_full:
                assert comparing and all(comparing)
            else:
                assert comparing == [], 'UID-only retry must make zero in-lock row comparisons'
            assert calls == ([False, True] if force_full else [False])
            assert not parent._pending_session_tails
            return reasons, dict(parent._ingest_ledger_counters), rows
        finally:
            child.shutdown()
            parent.shutdown()

    return run


@pytest.mark.parametrize('shape', ['divergent', 'agreeing', 'unrelated'])
def test_ingest_between_outer_proof_and_retry_checks_only_uids(ingest_during_parent_end, shape):
    """T8/T8b: the new views retain their veto without a second whole-prefix pass."""
    ingest_during_parent_end(shape)


@pytest.mark.parametrize('shape', ['divergent', 'agreeing', 'unrelated'])
def test_views_only_retry_equals_forced_full_rerun(ingest_during_parent_end, shape):
    """T8c: reasons, every ledger counter, and complete stored rows are identical."""
    assert ingest_during_parent_end(shape) == ingest_during_parent_end(shape, force_full=True)
