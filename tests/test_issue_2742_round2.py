"""Round-two receipt multiplicity and proven-prefix regressions (I1/I2/I8/I9)."""
import copy

import pytest

from tests.invariants.test_issue_2742_warm_tail_refinal import _busy_end, _open, _turn
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _view
from hermes_lcm import ingest_ledger
from tests.invariants.test_p1e_decided_tails import contents, end_view, history, replay
from tests.test_issue_2055_ingest_ledger_writes import engines as engines


@pytest.mark.parametrize('shape', ['plain', 'wrapped-base64', 'data-uri-key'])
@pytest.mark.parametrize('uid', ['none', 'distinct'])
@pytest.mark.parametrize('uid_mode', [None, 'off'])
def test_warm_repeat_reserves_prefix_receipt(tmp_path, monkeypatch, shape, uid, uid_mode):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        call = copy.deepcopy(view[2])
        call.pop('message_uid')
        if uid == 'distinct':
            call['message_uid'] = 'second-call'
        engine.ingest(copy.deepcopy(view + [call]))
        rows = engine._store.get_session_messages('parent')
        assert len(rows) == 5, rows
        assert sum(bool(r.get('tool_calls')) for r in rows) == 2
        assert [r['observed_at'] for r in rows] == [1., 2., 3., 4., 3.]
    finally:
        engine.shutdown()


@pytest.mark.parametrize('shape', ['wrapped-base64', 'data-uri-key'])
@pytest.mark.parametrize('uid_mode', [None, 'off'])
def test_repeat_call_then_new_result_keeps_its_call(tmp_path, monkeypatch, shape, uid_mode):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        call = copy.deepcopy(view[2])
        call.pop('message_uid')
        engine.ingest(copy.deepcopy(view + [call]))
        rows = engine._store.get_session_messages('parent')
        assert len(rows) == 5 and rows[-1]['tool_calls'], rows
        result = copy.deepcopy(view[3])
        result.pop('message_uid')
        result.update(timestamp=50., content='second-result')
        engine.ingest(copy.deepcopy(view + [call, result]))
        rows = engine._store.get_session_messages('parent')
        assert len(rows) == 6 and rows[-2]['tool_calls'], rows
        assert rows[-1]['content'] == 'second-result'
        assert rows[-1]['tool_call_id'] == call['tool_calls'][0]['id']
    finally:
        engine.shutdown()


@pytest.mark.parametrize('uid_mode', [None, 'off'])
def test_cursor_zero_twin_uses_receipt_only_once(tmp_path, monkeypatch, uid_mode):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _view('wrapped-base64')
        engine.ingest(copy.deepcopy(view))
        call = copy.deepcopy(view[2])
        call.pop('message_uid')
        view = copy.deepcopy(view) + [call]
        view[0]['timestamp'] += 1000
        plan = engine._identity_anchor_prematch(view, view, 0)
        assert 2 in plan['replayed'] and 4 not in plan['replayed'], plan
        targets = [r['store_id'] for i in (2, 4) for r in plan['matched'].get(i, ())]
        assert targets == [3], plan
    finally:
        engine.shutdown()


@pytest.mark.parametrize('uid_mode', [None, 'off'])
def test_two_busy_ends_preserve_proven_prefix_call(tmp_path, monkeypatch, uid_mode):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _turn(0) + _turn(1, 'plain')
        engine.ingest(copy.deepcopy(view[:2]))
        _busy_end(engine, monkeypatch, view[:5])
        _busy_end(engine, monkeypatch, view)
        engine._retry_pending_session_tails()
        rows = engine._store.get_session_messages('parent')
        assert sum(bool(r.get('tool_calls')) for r in rows) == 1, rows
        assert sum(r['role'] == 'tool' for r in rows) == 1
        assert [r['observed_at'] for r in rows] == [m['timestamp'] for m in view]
        assert not engine._pending_session_tails
    finally:
        engine.shutdown()


@pytest.mark.parametrize('writer', ['ingest', 'append_batch'])
@pytest.mark.parametrize('uid_mode', [None, 'off'])
def test_prefix_call_before_unstored_result(tmp_path, monkeypatch, writer, uid_mode):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _turn(0) + _turn(1, 'plain')
        engine.ingest(copy.deepcopy(view[:2]))
        _busy_end(engine, monkeypatch, view)
        [tail] = engine._pending_session_tails
        if writer == 'ingest':
            engine.ingest(copy.deepcopy(view[:5]))
        else:
            engine._store.append_batch('parent', copy.deepcopy(view[2:5]), conversation_id='conv')
        proof = engine._match_pending_tail(tail)
        assert proof[0] == 3, proof
        engine._retry_pending_session_tails()
        rows = engine._store.get_session_messages('parent')
        assert sum(bool(r.get('tool_calls')) for r in rows) == 1, rows
        assert sum(r['role'] == 'tool' for r in rows) == 1
        assert not engine._pending_session_tails
    finally:
        engine.shutdown()


@pytest.mark.parametrize('uid_mode', [None, 'off'])
@pytest.mark.parametrize('veto', ['canonical-uid', 'conflicting-uidset'])
def test_prefix_uid_only_change_matches_as_main(engines, monkeypatch, veto, uid_mode, record_property):
    """Hermes may reissue a prefix occurrence with a new UID; content/stamp still match."""
    monkeypatch.delenv('LCM_HOST_MESSAGE_UID')
    if uid_mode is not None:
        monkeypatch.setenv('LCM_HOST_MESSAGE_UID', uid_mode)
    engine = engines()
    view = history()
    engine.ingest(copy.deepcopy([view[0], view[1], view[3]]))
    incoming = copy.deepcopy(view)
    incoming[2]['message_uid'] = 'reissued-C'
    tail = end_view(engine, incoming)
    assert 'ledger' not in tail and [m['content'] for m in tail['messages']] == ['C']
    [target] = replay(engine, [view[2]], with_view=veto == 'conflicting-uidset')
    if veto == 'canonical-uid':
        engine._store.add_host_uid_bindings('foreign-lineage', [(target, 'C', 'canonical', 'stored_new')])
        assert engine._store._conn.execute(
            "SELECT uid FROM host_uid_bindings WHERE store_id=? AND kind='canonical'", (target,)).fetchall() == [('C',)]
    else:
        conn = engine._store._conn
        occ, blob = conn.execute(
            "SELECT b.occ_id,v.entries FROM occurrence_bindings b JOIN ingest_views v "
            "ON v.lineage=b.lineage AND v.view_seq=b.view_seq AND v.kind='ingest' WHERE b.store_id=?",
            (target,)).fetchone()
        bound = next(e for e in ingest_ledger.decode_entries(blob) if e.ref == occ)
        [entry], _ = engine._ledger_view_entries(engine._store.ingest_ledger_salt(), [incoming[2]])
        assert bound.uidset_digest is not None and bound.uidset_digest != entry.uidset_digest
    proof = engine._match_pending_tail(tail)
    engine._retry_pending_session_tails()
    record_property('C_count', contents(engine).count('C'))
    record_property('total_rows', len(contents(engine)))
    assert contents(engine).count('C') == 1
    assert len(contents(engine)) == 4
    assert proof[0] == 1, proof
    assert not engine._pending_session_tails


@pytest.mark.parametrize('uid_mode', [None, 'off'])
def test_queued_digest_prefix_reissued_uid_without_view(engines, monkeypatch, uid_mode):
    monkeypatch.delenv('LCM_HOST_MESSAGE_UID')
    if uid_mode is not None:
        monkeypatch.setenv('LCM_HOST_MESSAGE_UID', uid_mode)
    engine = engines()
    view = history()
    engine.ingest(copy.deepcopy(view[:2]))
    incoming = copy.deepcopy(view)
    incoming[2]['message_uid'] = 'reissued-C'
    tail = end_view(engine, incoming)
    assert 'ledger' in tail
    replay(engine, view[2:], with_view=False)
    proof = engine._match_pending_tail(tail)
    assert proof[3] == 'digest', proof
    assert proof[0] == 2, proof
    engine._retry_pending_session_tails()
    assert contents(engine) == list('ABCD')
    assert not engine._pending_session_tails
