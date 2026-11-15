"""A's original-ingest proof and W's per-occurrence pending-tail plan (I1/I2/I8/I9)."""
import copy
from types import SimpleNamespace

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm import ingest_ledger as ledger
from hermes_lcm import maintenance_scheduler
from tests.invariants.test_issue_2742_warm_tail_refinal import _bind, _busy_end, _open, _row, _turn
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _args, _view
from tests.invariants.test_p1e_decided_tails import contents, end_view, history, queue, replay
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_ingest_ledger_writes import engines as engines, sql_rows


@pytest.mark.parametrize('shape', ['wrapped-base64', 'data-uri-key'])
def test_original_stored_receipt_at_cursor_zero(tmp_path, monkeypatch, shape):
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        target = engine._store.get_session_messages('parent')[2]
        assert 'Externalized LCM ingest payload:' in str(target['tool_calls'])
        binding = next(b for b in sql_rows(engine, 'occurrence_bindings') if b['store_id'] == target['store_id'])
        assert binding['disposition'] == 'stored'
        assert binding['identity_digest'] == ledger.identity_digest(engine._store.ingest_ledger_salt(), view[2])
        assert engine._message_replay_identity(view[2]) not in engine._stored_row_forms(target)
        view[0]['timestamp'] += 1000
        plan = engine._identity_anchor_prematch(view, view, 0)
        assert 2 in plan['replayed'], plan
        assert plan['matched'][2][0]['store_id'] == target['store_id']
        assert 3 in plan['replayed']
    finally:
        engine.shutdown()


@pytest.mark.parametrize('disposition', ['migrated', 'ambiguous', 'composite_part'])
def test_nonstored_binding_is_not_an_original_receipt(tmp_path, monkeypatch, disposition):
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _view('wrapped-base64')
        engine.ingest(copy.deepcopy(view))
        conn = engine._store._conn
        conn.execute('UPDATE occurrence_bindings SET disposition=? WHERE store_id=3', (disposition,))
        conn.commit()
        view[0]['timestamp'] += 1000
        plan = engine._identity_anchor_prematch(view, view, 0)
        assert 2 not in plan['replayed'], plan
        assert 3 not in plan['replayed'], plan  # whole-segment rule remains intact
    finally:
        engine.shutdown()


@pytest.mark.parametrize('change', ['payload', 'stamp', 'uid'])
def test_receipt_vetoes_distinct_occurrence(tmp_path, monkeypatch, change):
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _view('wrapped-base64')
        engine.ingest(copy.deepcopy(view))
        view[0]['timestamp'] += 1000
        if change == 'payload':
            import json
            args = _args('wrapped-base64')
            args['path'] = 'distinct.b64'
            view[2]['tool_calls'][0]['function']['arguments'] = json.dumps(args)
        elif change == 'stamp':
            view[2]['timestamp'] += 1000
        else:
            view[2]['message_uid'] = 'different-call-occurrence'
        plan = engine._identity_anchor_prematch(view, view, 0)
        assert 2 not in plan['replayed'], plan
    finally:
        engine.shutdown()


def test_second_changed_argument_cold_replay_adds_nothing(tmp_path, monkeypatch):
    import json
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _view('wrapped-base64')
        engine.ingest(copy.deepcopy(view))
        args = _args('wrapped-base64')
        args['content'] = args['content'].replace('hdr', 'new', 1)
        view[2]['tool_calls'][0]['function']['arguments'] = json.dumps(args)
        counts = []
        for _ in range(3):
            engine.shutdown()
            engine = engine.__class__(config=engine._config, hermes_home=engine._hermes_home)
            _bind(engine)
            engine.ingest(copy.deepcopy(view))
            counts.append(len(engine._store.get_session_messages('parent')))
        assert counts[0] > 4
        assert counts == [counts[0]] * 3, counts
    finally:
        engine.shutdown()


def test_warm_match_beyond_old_window(engines):
    engine = engines()
    tail = queue(engine)
    boundary = tail['after_store_id']
    replay(engine, [_m('user', f'foreign-{i}', 50. + i) for i in range(5)])
    [target] = replay(engine, history()[3:])
    proof = engine._match_pending_tail(tail)
    assert proof[0] == 0 and proof[1] == boundary
    assert 1 in proof[6], proof
    assert proof[6][1]['store_id'] == target, proof
    engine._retry_pending_session_tails()
    assert contents(engine).count('C') == contents(engine).count('D') == 1
    assert not engine._pending_session_tails


def test_hole_retains_boundary_and_places_bindings_in_original_order(engines, monkeypatch):
    engine = engines()
    view = history() + [_m('user', 'E', 5., 'E')]
    tail = queue(engine, view)
    boundary = tail['after_store_id']
    replay(engine, history()[3:])
    assert engine._match_pending_tail(tail)[1] == boundary
    atomic = engine._store.ingest_atomic

    def busy(**_kwargs):
        import sqlite3
        raise sqlite3.OperationalError('database is locked')

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, 'ingest_atomic', busy)
        engine._retry_pending_session_tails()
    assert engine._pending_session_tails == [tail]
    assert tail['after_store_id'] == boundary
    assert [m['content'] for m in tail['messages']] == ['C', 'D', 'E']
    selected = engine._select_tail({**tail, 'senders': ['C', 'D', 'E']}, [0, 2])
    assert selected['senders'] == ['C', 'E']
    for key in ('messages', 'estimates', 'metadata_messages'):
        assert selected[key] == [tail[key][0], tail[key][2]]
    for key in ('entries', 'stamps', 'remainder'):
        assert selected['ledger'][key] == [tail['ledger'][key][0], tail['ledger'][key][2]]
    observed, append = [], engine._append_ingest_batch

    def inspect_batch(batch):
        assert engine._store._conn.in_transaction
        observed.append(batch)
        return append(batch)

    monkeypatch.setattr(engine, '_append_ingest_batch', inspect_batch)
    assert engine._store.ingest_atomic == atomic
    engine._retry_pending_session_tails()
    assert observed[0]['messages'] == selected['messages']
    assert observed[0]['metadata_messages'] == selected['metadata_messages']
    assert observed[0]['estimates'] == selected['estimates']
    assert {text: contents(engine).count(text) for text in 'ABCDE'} == dict.fromkeys('ABCDE', 1)
    bindings = {b['store_id']: b['ordinal'] for b in sql_rows(engine, 'occurrence_bindings')}
    rows = {r['content']: r['store_id'] for r in engine._store.get_session_messages('S')}
    assert bindings[rows['C']] < bindings[rows['D']] < bindings[rows['E']]


def test_repeated_identical_replies_consume_each_target_once(engines):
    engine = engines()
    echo = _m('assistant', 'echo', 4., 'echo')
    queue(engine, history()[:3] + [copy.deepcopy(echo) for _ in range(3)])
    replay(engine, [echo])
    engine._retry_pending_session_tails()
    assert contents(engine).count('C') == 1
    assert contents(engine).count('echo') == 3


@pytest.mark.parametrize('field', ['timestamp', 'message_uid'])
def test_warm_reply_known_occurrence_conflict_stays_new(engines, field):
    engine = engines()
    queue(engine)
    row = dict(history()[3], **{field: 104. if field == 'timestamp' else 'foreign-D'})
    replay(engine, [row])
    engine._retry_pending_session_tails()
    assert contents(engine).count('C') == 1
    assert contents(engine).count('D') == 2


@pytest.mark.parametrize('complete', [False, True])
def test_warm_plan_keeps_tool_groups_whole(tmp_path, monkeypatch, complete):
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _turn(0) + _turn(1, 'plain')
        engine.ingest(copy.deepcopy(view[:2]))
        _busy_end(engine, monkeypatch, view)
        tail = engine._pending_session_tails[0]
        # A committed non-prefix call, optionally with its result.
        engine._store.append_batch('parent', copy.deepcopy(view[4:6] if complete else view[4:5]),
                                   conversation_id='conv')
        plan = engine._match_pending_tail(tail)[6]
        assert (2 in plan) == complete
        assert (3 in plan) == complete
        engine._retry_pending_session_tails()
        rows = engine._store.get_session_messages('parent')
        assert sum(r['content'] == 'question-1' for r in rows) == 1
        assert sum(r['content'] == 'reply-1' for r in rows) == 1
        assert sum(r['content'] == 'result-1' for r in rows) == 1
        assert sum(bool(r.get('tool_calls')) for r in rows) == (1 if complete else 2)
    finally:
        engine.shutdown()


class VirtualClock:
    def __init__(self, monkeypatch):
        self.now = 100000.
        self.advances = 0
        clock = SimpleNamespace(**{k: getattr(engine_module.time, k) for k in dir(engine_module.time)
                                  if not k.startswith('__')})
        clock.monotonic = lambda: self.now
        monkeypatch.setattr(engine_module, 'time', clock)
        monkeypatch.setattr(maintenance_scheduler, 'time', clock)

    def expire(self):
        self.now += .060
        self.advances += 1


@pytest.mark.parametrize('route', ['end', 'sibling', 'shutdown', 'new-engine'])
def test_budget_after_first_commit_keeps_later_tail_for_resume(tmp_path, monkeypatch, route):
    engine = _open(tmp_path, monkeypatch, None)
    made = [engine]
    try:
        view = _turn(0)
        engine.ingest(copy.deepcopy(view))
        first = view + [_row('FIRST', 1001., 'assistant')]
        _busy_end(engine, monkeypatch, first)
        second = first + [_row('SECOND', 1002., 'assistant')]
        _busy_end(engine, monkeypatch, second)
        assert len(engine._pending_session_tails) == 2
        clock = VirtualClock(monkeypatch)
        append = engine._append_ingest_batch

        def slow_commit(batch):
            result = append(batch)
            clock.expire()
            return result

        with monkeypatch.context() as patch:
            patch.setattr(engine, '_append_ingest_batch', slow_commit)
            engine._retry_pending_session_tails()
        assert len(engine._pending_session_tails) == 1
        assert clock.advances == 1
        if route == 'end':
            engine.on_session_end('parent', copy.deepcopy(second))
        elif route == 'shutdown':
            engine.shutdown()
        else:
            if route == 'new-engine':
                engine.shutdown()
            sibling = engine.__class__(config=engine._config, hermes_home=engine._hermes_home)
            made.append(sibling)
            _bind(sibling)
        reader = made[-1]
        if reader._stable_use_closed:
            reader = reader.__class__(config=reader._config, hermes_home=reader._hermes_home)
            made.append(reader)
            _bind(reader)
        rows = reader._store.get_session_messages('parent')
        assert [(r['content'], r['observed_at']) for r in rows] == [
            ('question-0', 1.), ('reply-0', 2.), ('FIRST', 1001.), ('SECOND', 1002.)]
        assert all(not e._pending_session_tails for e in made)
    finally:
        for e in reversed(made):
            e.shutdown()


def test_budget_before_admission_never_skips_unscanned_rows(engines, monkeypatch):
    engine = engines()
    tail = queue(engine)
    boundary = tail['after_store_id']
    foreign = [_m('user', f'foreign-{i}', 50. + i) for i in range(5)]
    replay(engine, foreign)
    [target] = replay(engine, history()[3:])
    assert target > boundary + len(tail['messages'])
    clock = VirtualClock(monkeypatch)
    match = engine._match_pending_tail
    seen, admissions = [], []

    def expired(t):
        proof = match(t)
        seen.append(proof)
        clock.expire()
        return proof

    def failed_admission(**_kwargs):
        import sqlite3
        admissions.append(engine._store._conn.execute('PRAGMA busy_timeout').fetchone()[0])
        raise sqlite3.OperationalError('database is locked')

    with monkeypatch.context() as patch:
        patch.setattr(engine, '_match_pending_tail', expired)
        patch.setattr(engine._store, 'ingest_atomic', failed_admission)
        engine._retry_pending_session_tails()
    assert admissions == [0]
    assert seen and seen[0][6][1]['content'] == 'D'
    assert engine._pending_session_tails == [tail] and tail['after_store_id'] == boundary
    engine._retry_pending_session_tails()
    assert contents(engine) == ['A', 'B', *[m['content'] for m in foreign], 'D', 'C']
    assert not engine._pending_session_tails


def test_commit_retires_gap_tail_even_if_callback_raises(engines, monkeypatch):
    engine = engines()
    queue(engine)
    replay(engine, history()[3:])
    count = engine._ledger_count

    def failed_callback(key, *args, **kwargs):
        if key == 'ledger.tail.retry.stored_rows':
            raise ValueError('post-commit callback')
        return count(key, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, '_ledger_count', failed_callback)
        engine._retry_pending_session_tails()
    assert not engine._pending_session_tails
    engine._retry_pending_session_tails()
    assert contents(engine).count('C') == contents(engine).count('D') == 1


def test_offcurrent_fallback_advances_store_before_older_warm_tail(engines, monkeypatch):
    engine = engines(sessions=[("parent", None, None), ("other", None, None)])
    _bind(engine)
    sibling = None
    try:
        view = _turn(0)
        engine.ingest(copy.deepcopy(view))
        first = view + [_row('FIRST', 1001., 'assistant')]
        _busy_end(engine, monkeypatch, first)
        tail = engine._pending_session_tails[0]
        fallback_calls = []
        fallback = engine.__class__._retry_off_current_fallback

        def busy_match(_tail):
            import sqlite3
            raise sqlite3.OperationalError('database is locked')

        def observe_fallback(instance, queued):
            result = fallback(instance, queued)
            fallback_calls.append(result)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(engine, '_match_pending_tail', busy_match)
            patch.setattr(engine.__class__, '_retry_off_current_fallback', observe_fallback)
            sibling = engine.__class__(config=engine._config, hermes_home=engine._hermes_home)
            sibling.on_session_start('other', platform='cli', context_length=128000, conversation_id='other-conv')
            sibling.on_session_end('parent', copy.deepcopy(first))
            assert engine._pending_session_tails == [tail]
            assert fallback_calls == [True]
            assert sum(r['content'] == 'FIRST' for r in engine._store.get_session_messages('parent')) == 1
        engine._retry_pending_session_tails()
        rows = engine._store.get_session_messages('parent')
        assert [(r['content'], r['observed_at']) for r in rows] == [
            ('question-0', 1.), ('reply-0', 2.), ('FIRST', 1001.)]
        assert not engine._pending_session_tails
    finally:
        if sibling is not None:
            sibling.shutdown()
        engine.shutdown()


def test_inserted_tail_known_uid_conflict_vetoes_gap_match(engines):
    engine = engines()
    view = history()
    engine.ingest(copy.deepcopy([view[0], view[1], view[3]]))
    tail = end_view(engine, view)
    assert 'ledger' not in tail
    replay(engine, [{**view[2], 'message_uid': 'different-C'}])
    assert engine._match_pending_tail(tail)[6] == {}
    engine._retry_pending_session_tails()
    assert contents(engine).count('C') == 2


def test_nonpositional_head_match_does_not_hide_a_later_hole(engines):
    engine = engines()
    view = history() + [_m('user', 'E', 5., 'E')]
    tail = queue(engine, view)
    boundary = tail['after_store_id']
    head = max(b['ordinal'] for b in sql_rows(engine, 'occurrence_bindings'))
    # Another view committed E before C physically, but bound them in their host order C < E.
    replay(engine, [view[4], view[2]], ordinals=[head + 3 * 1024, head + 1024])
    proof = engine._match_pending_tail(tail)
    assert set(proof[6]) == {0, 2}
    assert proof[0] == 0 and proof[1] == boundary, proof
    engine._retry_pending_session_tails()
    assert {text: contents(engine).count(text) for text in 'ABCDE'} == dict.fromkeys('ABCDE', 1)
    bindings = {b['store_id']: b['ordinal'] for b in sql_rows(engine, 'occurrence_bindings')}
    rows = {r['content']: r['store_id'] for r in engine._store.get_session_messages('S')}
    assert bindings[rows['C']] < bindings[rows['D']] < bindings[rows['E']]


def test_receipt_known_uid_veto_survives_ingest_view_retention(engines, monkeypatch):
    engine = engines()
    monkeypatch.setenv('LCM_HOST_MESSAGE_UID', 'shadow')
    view = _view('wrapped-base64')
    engine.ingest(copy.deepcopy(view))
    conn = engine._store._conn
    assert conn.execute("SELECT uid FROM host_uid_bindings WHERE store_id=3 AND kind='canonical'").fetchall()
    conn.execute('DELETE FROM ingest_views')  # Retained canonical UID proof outlives its ingest vector.
    conn.commit()
    view[0]['timestamp'] += 1000
    view[2]['message_uid'] = 'a-different-call-occurrence'
    plan = engine._identity_anchor_prematch(view, view, 0)
    assert 2 not in plan['replayed'], plan


@pytest.mark.parametrize('shape', ['wrapped-base64', 'data-uri-key'])
def test_distinct_inline_payload_at_same_stamp_stays_new(tmp_path, monkeypatch, shape):
    import json
    engine = _open(tmp_path, monkeypatch, None)
    try:
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        salt = engine._store.ingest_ledger_salt()
        original = ledger.identity_digest(salt, view[2])
        args = _args(shape)
        if shape == 'wrapped-base64':
            body = args['content']
            args['content'] = body[:4] + ('A' if body[4] != 'A' else 'B') + body[5:]
        else:
            key = next(k for k in args if k.startswith('data:'))
            offset = key.index(',') + 1
            changed = key[:offset] + ('A' if key[offset] != 'A' else 'B') + key[offset + 1:]
            args[changed] = args.pop(key)
        view[2]['tool_calls'][0]['function']['arguments'] = json.dumps(args)
        assert view[2]['timestamp'] == 3.
        assert ledger.identity_digest(salt, view[2]) != original
        view[0]['timestamp'] += 1000
        assert 2 not in engine._identity_anchor_prematch(view, view, 0)['replayed']
    finally:
        engine.shutdown()
