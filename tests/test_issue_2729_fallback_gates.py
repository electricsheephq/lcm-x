"""#2729: durable ownership, whole-prefix admission and atomic late-end retries."""

import copy
import json
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
from hermes_lcm.externalize import get_large_output_storage_dir
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
        assert sorted(seen.values()) == [1, 2]  # Only the losing racer repeats the full comparison.
        assert_rows(first, VIEW, before)
        assert_rows(second, VIEW, before)
    finally:
        second.shutdown()


@contextmanager
def busy(**kwargs):
    raise sqlite3.OperationalError('database is locked')
    yield  # pragma: no cover


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
        second.on_session_end('S2', copy.deepcopy(VIEW))
        assert_rows(second, VIEW, before)
        before = lifecycle(first)
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


@pytest.mark.parametrize('sources,expected', [(['A', 'A'], 'A'), (['A', ''], 'A'), (['', ''], '')])
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
        assert engine._pending_session_tails[0]['source'] == expected
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
    assert calls == ([False] if change == 'none' else [False, True])
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
            payload.write_text(json.dumps(dict(content='different payload', session_id='S2', kind='tool_result')))
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
