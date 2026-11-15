"""Held receipt ownership across repeated calls and engines (I1/I8/I9)."""

import copy
import json

import pytest

from tests.invariants.test_issue_2742_warm_tail_refinal import _open
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _view


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
