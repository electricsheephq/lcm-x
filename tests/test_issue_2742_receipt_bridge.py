"""Original-ingest receipt proof and positional pending-tail controls (I1/I2/I8/I9)."""
import copy
from types import SimpleNamespace

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm import ingest_ledger as ledger
from hermes_lcm import maintenance_scheduler
from tests.invariants.test_issue_2742_warm_tail_refinal import _bind, _busy_end, _open, _row, _turn
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _args, _view
from tests.invariants.test_p1e_decided_tails import contents, history, queue, replay
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


@pytest.mark.parametrize('field', ['timestamp', 'message_uid'])
def test_warm_reply_known_occurrence_conflict_stays_new(engines, field):
    engine = engines()
    queue(engine)
    row = dict(history()[3], **{field: 104. if field == 'timestamp' else 'foreign-D'})
    replay(engine, [row])
    engine._retry_pending_session_tails()
    assert contents(engine).count('C') == 1
    assert contents(engine).count('D') == 2


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
