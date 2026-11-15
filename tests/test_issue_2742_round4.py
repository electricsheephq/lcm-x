"""Held receipt ownership and bounded pending-tail matching (I1/I8/I9)."""

import copy
import json
from collections import Counter

import pytest

from tests.invariants.test_issue_2742_warm_tail_refinal import _busy_end, _open, _row, _turn
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _view
from tests.invariants.test_p1e_decided_tails import contents, history, queue, replay
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_ingest_ledger_writes import engines as engines  # noqa: F401


@pytest.mark.parametrize('shape', ['wrapped-base64', 'data-uri-key'])
@pytest.mark.parametrize('uid_mode', [None, 'off'])
@pytest.mark.parametrize('audit_from', [0, 1, 2])
@pytest.mark.parametrize('edit', [None, 0])
def test_held_receipt_never_overtakes_twin(tmp_path, monkeypatch, shape, uid_mode, audit_from, edit):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        twin = copy.deepcopy(view[2])
        twin.pop('message_uid')
        engine.ingest(copy.deepcopy(view + [twin]))
        incoming = copy.deepcopy(view + [twin])
        if edit is not None:
            incoming[edit]['content'] += ' (edited by host)'
        plan = engine._identity_anchor_prematch(incoming, incoming, 4, audit_from=audit_from)
        targets = {i: [r['store_id'] for r in rows] for i, rows in plan['matched'].items()}
        assert targets[2] == [3], plan
        assert targets[4] == [5], plan
        assert len({sid for ids in targets.values() for sid in ids}) == sum(map(len, targets.values()))
    finally:
        engine.shutdown()


@pytest.mark.parametrize('shape', ['wrapped-base64', 'data-uri-key'])
@pytest.mark.parametrize('uid_mode', [None, 'off'])
@pytest.mark.parametrize('edit', [0, 1])
def test_two_engines_keep_each_call_receipt(tmp_path, monkeypatch, shape, uid_mode, edit, record_property):
    first = _open(tmp_path, monkeypatch, uid_mode)
    second = None
    try:
        view = _view(shape)
        first.ingest(copy.deepcopy(view))
        twin = copy.deepcopy(view[2])
        twin.pop('message_uid')
        second = _open(tmp_path, monkeypatch, uid_mode)
        second.ingest(copy.deepcopy(view + [twin]))
        assert len(first._store.get_session_messages('parent')) == 5
        incoming = copy.deepcopy(view + [twin])
        incoming[edit]['content'] += ' (edited by host)'
        first.ingest(copy.deepcopy(incoming))
        rows = first._store.get_session_messages('parent')
        counts = dict(rows=len(rows), calls=sum(bool(r.get('tool_calls')) for r in rows))
        print('OVERTAKE_COUNTS', json.dumps(dict(shape=shape, uid_mode=uid_mode, edit=edit, **counts)))
        record_property('counts', json.dumps(counts))
        assert counts == dict(rows=6, calls=2), counts
        assert rows[-1]['content'] == incoming[edit]['content']
        assert [r['store_id'] for r in rows if r.get('tool_calls')] == [3, 5]
    finally:
        if second is not None:
            second.shutdown()
        first.shutdown()


def test_past_gap_window_copy_is_named_duplicate_as_main(engines, record_property):
    """Residual: an unseen far copy cannot mark the queued occurrence as stored."""
    engine = engines()
    tail = queue(engine)
    # Pin the decided window independently of the implementation (including on main).
    foreign = [_m('user', f'foreign-{i}', 50. + i) for i in range(256 + len(tail['messages']))]
    replay(engine, foreign)
    [far] = replay(engine, history()[3:])
    proof = engine._match_pending_tail(tail)
    assert proof[0] == 0 and proof[1] == tail['after_store_id'], proof
    assert far > tail['after_store_id'] + len(tail['messages']) + 256
    assert proof[6] == {}, proof
    engine._retry_pending_session_tails()
    expected = list('AB') + [r['content'] for r in foreign] + list('DCD')
    assert contents(engine) == expected
    assert not engine._pending_session_tails
    engine._retry_pending_session_tails()
    assert contents(engine) == expected
    record_property('counts', json.dumps(dict(rows=len(expected), C=1, D=2)))
    print('PAST_WINDOW_COUNTS', json.dumps(dict(rows=len(expected), C=1, D=2)))


def test_pending_tail_load_is_bounded_and_drains_once(engines, monkeypatch):
    engine = engines()
    tail = queue(engine)
    # A near replay beyond main's old window, followed by a long unrelated suffix.
    foreign = [_m('user', f'foreign-{i}', 50. + i) for i in range(256 + 99)]
    replay(engine, foreign[:5])
    [near] = replay(engine, history()[3:])
    replay(engine, foreign[5:])
    assert len(engine._store.get_session_messages('S')) - 2 == 256 + 100
    get_range, loads = engine._store.get_range, []

    def bounded(*args, **kwargs):
        rows = get_range(*args, **kwargs)
        loads.append((kwargs['limit'], [r['store_id'] for r in rows]))
        assert 0 <= kwargs['limit'] <= len(tail['messages']) + 256
        assert len(rows) <= len(tail['messages']) + 256
        return rows

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, 'get_range', bounded)
        proof = engine._match_pending_tail(tail)
        assert set(proof[6]) == {1}, proof
        assert proof[6][1]['store_id'] == near
        assert all(r['store_id'] in loads[-1][1] for r in proof[6].values())
        engine._retry_pending_session_tails()
    assert loads and all(len(ids) == len(tail['messages']) + 256 for _, ids in loads)
    expected = list('AB') + [r['content'] for r in foreign[:5]] + ['D'] + [r['content'] for r in foreign[5:]] + ['C']
    assert contents(engine) == expected
    assert Counter(contents(engine)) == Counter({text: 1 for text in expected})
    assert not engine._pending_session_tails
    engine._retry_pending_session_tails()
    assert contents(engine) == expected


@pytest.mark.parametrize('uid_mode', [None, 'off'])
def test_2742_final_after_later_rows_stored_once(tmp_path, monkeypatch, uid_mode):
    """A sibling commits FINAL beyond the queued tail's old positional window."""
    first = _open(tmp_path, monkeypatch, uid_mode)
    second = None
    try:
        original = _turn(0)
        first.ingest(copy.deepcopy(original))
        second = _open(tmp_path, monkeypatch, uid_mode)
        incoming = copy.deepcopy(original)
        incoming[0]['timestamp'] += 1000
        final = _row('FINAL', 1003., 'assistant')
        _busy_end(first, monkeypatch, incoming + [final])
        [tail] = first._pending_session_tails
        later = [_row(f'later-{i}', 50. + i) for i in range(5)]
        second.ingest(copy.deepcopy(original + later + [final]))
        rows = first._store.get_session_messages('parent')
        [stored_final] = [r for r in rows if r['content'] == 'FINAL']
        assert stored_final['store_id'] > tail['after_store_id'] + len(tail['messages'])
        first._retry_pending_session_tails()
        rows = first._store.get_session_messages('parent')
        assert sum(r['content'] == 'FINAL' for r in rows) == 1, rows
        assert [r['observed_at'] for r in rows if r['content'] == 'question-0'] == [1., 1001.]
        assert all(sum(r['content'] == m['content'] for r in rows) == 1 for m in later)
        assert not first._pending_session_tails
    finally:
        if second is not None:
            second.shutdown()
        first.shutdown()
