"""#2707 capture, isolated writer and typed recall regressions (I1/I5/I6)."""
from itertools import product
import json
import random
import sqlite3

import pytest

from hermes_lcm import queued_user_rows as q
from hermes_lcm import store as sm, tools
from tests.harness import engine_context


def user(text, stamp=10., uid=None):
    return dict(role='user', content=text, timestamp=stamp, message_uid=uid)


def records(engine):
    with sm.records_read_cut(engine._store) as cut:
        return sm.read_records(cut, 'queued_user_rows')[0]


def captured(engine, texts=('lost', 'newest'), turn='T'):
    run = [user(t, 10. + i, f'{turn}-{i}') for i, t in enumerate(texts)]
    q.capture(engine, dict(turn_id=turn, session_id='S', conversation_id='conv',
                          conversation_history=run, user_message=texts[-1]))
    return run


def folded(engine, texts=('lost', 'newest'), turn='T'):
    run = captured(engine, texts, turn)
    run[0]['content'] = texts[-1]
    view = [run[0], dict(role='assistant', content='answer', timestamp=50.)]
    engine.ingest(view)
    return run, view


def test_two_phase_freeze_and_canonical_control(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        run = captured(e)
        assert not sm.records_path(e._store).exists(), 'hook wrote a records file'
        run[0].update(content='newest', timestamp=99., message_uid='changed')
        view = [run[0], dict(role='assistant', content='answer', timestamp=100.)]
        e.ingest(view)
        row, = records(e)
        assert (row['content'], row['observed_at'], row['captured_uid']) == ('lost', 10., 'T-0')
        assert [r['content'] for r in e._store.get_session_messages('S')] == ['newest', 'answer']
        e.ingest(view)
        assert len(records(e)) == 1
        for name in ('lcm_load_session', 'lcm_grep', 'lcm_recall'):
            args = dict(session_id='S', session_scope='session', limit=100, query='lost')
            reply = json.loads(getattr(tools, name)(args, engine=e))
            assert 'error' not in reply
            assert any(r.get('owner_store_id') == row['owner_store_id'] for r in reply.get('queued_user_records', reply.get('queued_user_record_hits', [])))
        expanded = json.loads(tools.lcm_expand(dict(store_id=row['owner_store_id']), engine=e))
        assert expanded['queued_user_records'][0]['content'] == 'lost'


@pytest.mark.parametrize('shape,reason', [
    ('steer', 'capture_boundary'), ('carrier', 'capture_boundary'), ('list', 'capture_boundary'),
    ('prefix', 'prefix_host'), ('survived', 'run_object_survived'), ('not-clean', 'head_not_clean'),
    ('unchanged', 'join_unchanged'), ('earlier-ingest', 'earlier_ingest'),
])
def test_false_positive_refusals(tmp_path, shape, reason):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        run = [user('P'), user('R', 11.)]
        clean = 'R'
        if shape == 'steer':
            run[0]['display_kind'] = 'steer'
        if shape == 'carrier':
            run[0]['content'] = 'CONTEXT SUMMARY\nEarlier turns.'
        if shape == 'list':
            run[0]['content'] = []
        if shape == 'unchanged':
            run[0]['content'] = ''
        q.capture(e, dict(turn_id='T', session_id='S', conversation_history=run, user_message=clean))
        if shape == 'earlier-ingest':
            e._ingest_messages(run)
        if shape == 'prefix':
            run[0]['_merged_turn_prefix'] = 'P\n\n'
        run[0]['content'] = 'wrong' if shape == 'not-clean' else 'R'
        view = run if shape == 'survived' else [run[0]]
        e.ingest(view)
        assert records(e) == []
        assert e._queued_counts[reason] == 1


@pytest.mark.parametrize('text,old,reason', [
    ('P', 'P', 'backward_overlap'), ('P1\n\nP2', 'P1', 'joined_overlap'),
    (' \t ', ' \t ', 'backward_overlap'), (' \t ', 'X\n\n \t ', 'backward_overlap'),
    (' \n\n\t ', 'X\n\n \n\n\t ', 'backward_overlap'),
])
def test_backward_overlap_retry_reload_and_blank(tmp_path, text, old, reason):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        e._store.append('S', user(old, 1.), conversation_id='other')
        folded(e, (text, 'R'))
        assert records(e) == [], 'duplicate extra record'
        assert e._queued_counts[reason] == 1


def test_separate_row_joined_reload(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        for i, text in enumerate(('P1', 'P2')):
            e._store.append('S', user(text, i + 1.), conversation_id='conv')
        folded(e, ('P1\n\nP2', 'R'))
        assert not records(e), 'duplicate extra record'
        assert e._queued_counts['joined_overlap'] == 1


def test_fresh_blank_is_eligible(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e, (' \t ', 'R'))
        assert [r['content'] for r in records(e)] == [' \t ']


def member_oracle(p, row):
    for start in range(len(row) + 1):
        for end in range(start, len(row) + 1):
            if row[start:end] == p and (start == 0 or row[max(0, start-2):start] == '\n\n') \
                    and (end == len(row) or row[end:end+2] == '\n\n'):
                return True
    return False


def blank_cases():
    neighbours = ['', 'x', 'x\n', '\nx', ' ', '\n']
    for size in range(5):
        for chars in product(' \t\n', repeat=size):
            p = ''.join(chars)
            for members in range(1, 4):
                for parts in product([p, *neighbours], repeat=members):
                    yield p, '\n\n'.join(parts)
                for parts in product(neighbours, repeat=members):
                    yield p, '\n\n'.join(parts)
    rng = random.Random(2707)
    for _ in range(5000):
        p = ''.join(rng.choices(' \t\n', k=rng.choice([5, 6])))
        yield p, '\n\n'.join(rng.choices([p, *neighbours], k=rng.randint(1, 4)))


def test_generated_blank_members(record_property):
    total = agree = 0
    for p, row in blank_cases():
        expected = member_oracle(p, row)
        actual = q.blank_member(p, row)
        assert actual == expected, (repr(p), repr(row), expected)
        total += 1
        agree += actual == expected
    record_property('generated_cases', total)
    record_property('oracle_agreement', agree)
    assert total == agree == 84497


@pytest.mark.parametrize('count,size,refused', [(63, 1, False), (64, 1, False), (65, 1, True),
    (1, 1048575, False), (1, 1048576, False), (1, 1048577, True)])
def test_backward_limits(tmp_path, count, size, refused):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        for i in range(count):
            sid = e._store.append('S', user('x', i + 1.))
            e._store.connection.execute('UPDATE messages SET content=? WHERE store_id=?', ('x' * size, sid))
            e._store.connection.commit()
        folded(e)
        assert bool(records(e)) is not refused
        assert e._queued_counts['backward_unscanned'] == int(refused)


@pytest.mark.parametrize('text,reason', [('data:image/png;base64,' + 'A' * 300, 'inline_payload'),
                                       ('word ' * 25000, 'whole_row_payload')], ids=['inline', 'whole-row'])
def test_protection_refusals(tmp_path, text, reason):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e, (text, 'R'))
        assert not records(e)
        assert e._queued_counts[reason] == 1


def test_ignore_pattern_drop(tmp_path, monkeypatch):
    from hermes_lcm import message_patterns
    from tests.engine_helpers import _FakeTimeoutRegexEngine
    # Patterns need the optional regex engine, which CI does not install; main's tests use this stand-in.
    monkeypatch.setattr(message_patterns, '_regex_engine', _FakeTimeoutRegexEngine)
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False,
                        ignore_message_patterns=['lost']) as e:
        folded(e)
        assert not records(e)
        assert e._queued_counts['ignore_pattern'] == 1


def test_drain_lock_isolation_and_purge_revalidation(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        original = e._store_queued_user_rows
        calls = []
        def writer(entries):
            assert not e._store._write_lock._is_owned(), 'drain inside canonical write lock'
            calls.append(True)
            raise sqlite3.OperationalError('busy')
        monkeypatch.setattr(e, '_store_queued_user_rows', writer)
        folded(e)
        assert calls, 'drain inside canonical write lock'
        assert e._queued_entries and not sm.records_path(e._store).exists()
        e._store.delete_session_messages('S')
        monkeypatch.setattr(e, '_store_queued_user_rows', original)
        q.drain(e)
        with sm.records_read_cut(e._store) as cut:
            attached, unattached = sm.read_records(cut, 'queued_user_rows')
            assert not attached and unattached == 0, 'purge resurrection'
        assert not e._queued_entries
        assert e._queued_counts['dropped_permanent'] == 1
        q.drain(e, exiting=True)
        assert e._queued_counts['exit_kept'] == 0


def test_withheld_two_conversations_and_footer_pages(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        e._store.append('S', user('lost\n\nnewest', 55.), conversation_id='other')
        cursor, pages = 0, []
        while True:
            page = json.loads(tools.lcm_load_session(dict(session_id='S', limit=1, after_store_id=cursor), engine=e))
            pages.append(page)
            if page['next_cursor'] is None:
                break
            cursor = page['next_cursor']
        footers = [p['queued_user_records_withheld'] for p in pages if 'queued_user_records_withheld' in p]
        assert len(footers) == 1 and footers[0]['counts'] == dict(overlap=1, unscanned=0)
        assert all(not p.get('queued_user_records') for p in pages)
        assert records(e)[0]['superseded_by'] is None
        locator, = footers[0]['locators']
        expanded = json.loads(tools.lcm_expand(locator['reference']['args'], engine=e))
        assert expanded['queued_user_records'][0]['content'] == 'lost'
        assert expanded['queued_user_records'][0]['state'] == 'overlap'
        for name in ('lcm_grep', 'lcm_recall'):
            reply = json.loads(getattr(tools, name)(dict(query='lost', session_id='S', session_scope='session', limit=100), engine=e))
            assert not reply.get('queued_user_record_hits')


def test_zero_footer_and_mcp_surface(tmp_path):
    from hermes_lcm.mcp_reader import StoreReader, READER_TOOLS
    from hermes_lcm import schemas
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        empty = json.loads(tools.lcm_load_session(dict(session_id='S'), engine=e))
        assert not any(k.startswith('queued_user_records') for k in empty)
        folded(e)
        reader = StoreReader(e._store.db_path, tmp_path / 'home', config=e._config, environ={})
        assert READER_TOOLS['lcm_expand'] is schemas.LCM_EXPAND
        assert 'queued_user_ref' not in READER_TOOLS['lcm_expand']['parameters']['properties']
        for name, args in [('lcm_load_session', dict(session_id='S', limit=1)),
                           ('lcm_grep', dict(query='lost', session_scope='all')),
                           ('lcm_recall', dict(query='lost')),
                           ('lcm_expand', dict(store_id=records(e)[0]['owner_store_id']))]:
            text, error = reader.call(name, args)
            assert not error, text
            payload = json.loads(text)
            assert 'lost' in text, (name, payload)
            if name == 'lcm_load_session':
                assert payload['queued_user_records'][0]['content'] == 'lost'


def test_long_session_and_owner_floors(tmp_path, monkeypatch, record_property):
    import time
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        # This is a read-surface fixture; SQL avoids 100k separate ingest turns.
        e._store.connection.executemany('INSERT INTO messages(session_id, role, content, timestamp, ingested_at) '
                                        'VALUES (?,?,?,?,?)',
            (('S', 'user', 'later-' + str(i), float(i + 100), float(i + 100)) for i in range(100000)))
        e._store.connection.commit()
        measures = []
        for label, rows, size in [('budget', 200000, 64 * 1048576), ('near-floor', 65, 1048577)]:
            monkeypatch.setattr(q, 'SCAN_ROWS', rows)
            monkeypatch.setattr(q, 'SCAN_BYTES', size)
            start = time.perf_counter()
            reply = json.loads(tools.lcm_load_session(dict(session_id='S', limit=1), engine=e))
            elapsed = (time.perf_counter() - start) * 1000
            counts = reply.get('queued_user_records_withheld', {}).get('counts', dict(overlap=0, unscanned=0))
            assert counts == dict(overlap=0, unscanned=int(label == 'near-floor'))
            measures.append(dict(label=label, counts=counts, page_ms=elapsed))
        record_property('long_session', measures)


@pytest.mark.parametrize('mode', ['off', 'shadow'])
def test_equal_unrelated_reload_has_no_supersession(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('LCM_HOST_MESSAGE_UID', mode)
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        e._store.append('S', user('lost\n\nnewest', 100., 'unrelated-uid'), conversation_id='conv')
        record, = records(e)
        assert record['superseded_by'] is None
        with sm.records_read_cut(e._store) as cut:
            _, states = q.decide(cut, 'S')
        assert states == {(record['owner_store_id'], 0): 'overlap'}


def test_retired_turn_and_reused_head(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        run = captured(e)
        e.on_turn_complete([], turn_id='T', interrupted=True)
        run[0]['content'] = 'newest'
        e.ingest([run[0]])
        assert not records(e)
        q.capture(e, dict(turn_id='next', session_id='S', conversation_history=[run[0], user('R')], user_message='R'))
        assert e._queued_counts['reused_head'] == 1


@pytest.mark.parametrize('roles', [None, ['user'], ['assistant']])
def test_limit_one_listing_matches_unpaged(tmp_path, roles):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        e._store.append('S', user('before', 1.))
        folded(e, ('first', 'second', 'newest'))
        e._store.append('S', user('after', 60.))
        args = dict(session_id='S', include_exact_ref=True)
        if roles:
            args['roles'] = roles
        whole = json.loads(tools.lcm_load_session(dict(args, limit=100), engine=e))
        expected = whole['messages']
        expected_records = whole.get('queued_user_records', [])
        actual, actual_records, cursor, seen = [], [], 0, set()
        while True:
            page = json.loads(tools.lcm_load_session(dict(args, limit=1, after_store_id=cursor), engine=e))
            actual.extend(page['messages'])
            actual_records.extend(page.get('queued_user_records', []))
            assert not page['has_more'] or page['next_cursor'] is not None, 'more rows without cursor'
            if page['next_cursor'] is None:
                break
            cursor = page['next_cursor']
            key = cursor
            assert key not in seen, 'pagination stalled'
            seen.add(key)
        assert actual == expected
        assert actual_records == expected_records



def test_refused_head_does_not_block_later_entry(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        original = e._store_queued_user_rows
        monkeypatch.setattr(e, '_store_queued_user_rows', lambda _: (_ for _ in ()).throw(ValueError('head refused')))
        folded(e, ('bad', 'good', 'newest'))
        def write(entries):
            if dict(entries[0].values)['ordinal'] == 0:
                raise ValueError('head refused')
            original(entries)
        monkeypatch.setattr(e, '_store_queued_user_rows', write)
        q.drain(e)
        assert [r['content'] for r in records(e)] == ['good']
        assert [dict(r.values)['content'] for r in e._queued_entries] == ['bad']


def test_oversize_refused_with_externalization_off(tmp_path, monkeypatch):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        e._config.large_output_externalization_enabled = False
        monkeypatch.setattr(q, 'DRAIN_BYTES', 8)
        folded(e, ('x' * 9, 'R'))
        assert not e._queued_entries and not records(e)
        assert e._queued_counts['refused_oversize'] == 1


def test_exit_reports_residual_and_uses_storage_lock(tmp_path, monkeypatch, caplog):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        monkeypatch.setattr(e, '_store_queued_user_rows', lambda _: (_ for _ in ()).throw(ValueError('fault')))
        folded(e)
        calls = []
        def write(_):
            calls.append(e._storage_lock._is_owned())
            raise ValueError('fault')
        monkeypatch.setattr(e, '_store_queued_user_rows', write)
        q.flush_at_exit()
        assert calls == [True]
        assert 'discarded_at_exit=1' in caplog.text and 'dropped=0' not in caplog.text


def test_writer_sweeps_rollback_purge_orphans(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        owner = records(e)[0]['owner_store_id']
        e._session_id = 'live'
        q.capture(e, dict(turn_id='live', session_id='live', conversation_history=[user('keep'), user('R')], user_message='R'))
        head = e._queued_captures['live'].objects[0]
        head['content'] = 'R'
        e.ingest([head])
        e._store.connection.execute('DELETE FROM messages WHERE store_id=?', (owner,))
        e._store.connection.commit()
        # A fresh process would have no first-writer-open bookkeeping.
        sm._RECORDS_SWEPT.discard(str(sm.records_path(e._store)))
        writer = sm.open_records_writer(e._store)
        writer.close()
        with sqlite3.connect(sm.records_path(e._store)) as side:
            assert side.execute('SELECT content FROM queued_user_rows').fetchall() == [('keep',)]
        assert 'orphan_swept=1' in sm.records_doctor(e._store)


@pytest.mark.parametrize('mode', ['stored', 'permanent', 'engine_failure'])
def test_exit_flush_continues_after_concurrent_removal(tmp_path, monkeypatch, caplog, mode):
    from collections import Counter, deque
    from threading import RLock

    class Engine:
        def __init__(self):
            self._storage_lock = RLock()
            self._queued_counts = Counter()
            self._queued_entries = deque([q.Entry('identity', tmp_path, (('content', 'lost'),))])

    class Engines(list):
        def discard(self, engine):
            if engine in self:
                self.remove(engine)

    first, second = Engine(), Engine()
    def already_drained(entries):
        first._queued_entries.remove(entries[0])
        if mode == 'permanent':
            raise ValueError('pending purge intent')
    def retry_later(entries):
        raise ValueError('retry later')
    first._store_queued_user_rows = already_drained
    second._store_queued_user_rows = retry_later
    if mode == 'engine_failure':
        class BrokenLock:
            def __enter__(self):
                raise RuntimeError('private exit failure')
            def __exit__(self, *args):
                pass
        first._storage_lock = BrokenLock()
    monkeypatch.setattr(q, '_EXIT_ENGINES', Engines([first, second]))
    q.flush_at_exit()
    if mode != 'engine_failure':
        assert not first._queued_entries and first._queued_counts['exit_kept'] == 0
        assert 'discarded_at_exit=0' in caplog.text
    else:
        assert 'queued records exit failed: RuntimeError' in caplog.text
        assert 'private exit failure' not in caplog.text
    assert second._queued_counts['exit_kept'] == 1
    assert 'discarded_at_exit=1' in caplog.text


def test_permanent_refusal_counted_once_when_another_drain_removed_it(tmp_path):
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        orig = e._store_queued_user_rows
        e._store_queued_user_rows = lambda _: (_ for _ in ()).throw(sqlite3.OperationalError('busy'))
        folded(e)
        e._store.delete_session_messages('S')
        e._store_queued_user_rows = orig
        entry = e._queued_entries[0]

        def write(entries):
            if entry in e._queued_entries:  # another drain refuses and removes it meanwhile
                e._queued_entries.remove(entry)
                e._queued_counts['dropped_permanent'] += 1
            return orig(entries)
        e._store_queued_user_rows = write
        q.drain(e)
        assert e._queued_counts['dropped_permanent'] == 1


def test_doctor_reports_withheld_states(tmp_path):
    # lcm_doctor passes decide to store.records_doctor; without it these states would vanish silently.
    with engine_context(tmp_path, window=128000, conversation_id='conv', embeddings_enabled=False) as e:
        folded(e)
        e._store.append('S', user('lost\n\nnewest', 55.), conversation_id='other')
        report = json.loads(tools.lcm_doctor({}, engine=e))
        detail, = [check['detail'] for check in report['checks'] if check['check'] == 'records']
        assert 'superseded=0 withheld.overlap=1 withheld.unscanned=0' in detail
