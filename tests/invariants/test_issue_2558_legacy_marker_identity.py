"""#2558 (I1/I6): real v0.29.2 L1/L2 stores retain each replay occurrence once."""

import json
import tempfile

import pytest

from tests.invariants.test_issue_2539_compare_matrix import _restore, _v0292_matrix_store
from tests.test_issue_10_persisted_output_spillover import _marker
from tests.test_issue_2521_spillover_upgrade_replay import (
    CALL, NEW_TURN, _host_history, _read_store, _run_phase, _tool_rows,
)


@pytest.fixture(params=['L1', 'L2'])
def legacy(request, tmp_path, monkeypatch):
    temp = tmp_path / 'host-temp'
    temp.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(temp))
    preview, location, crlf = ('plain', 'hermes-results', True) if request.param == 'L1' else (
        'data-uri', 'spillover', False)
    marker, path, body = _v0292_matrix_store(tmp_path, temp, preview, location, crlf)
    return marker, path, body, _read_store(tmp_path)


def _logical(rows, tmp_path):
    storage = tmp_path / 'hermes' / 'externalized'
    return [(r['session_id'], r['conversation_id'], r['role'], _restore(r['content'], storage),
             r['tool_call_id'], r['tool_calls']) for r in rows]


def _control(original, history, tmp_path):
    # v0.29.2 -> v0.29.2 keeps the original four rows and appends only
    # the host suffix. Restore its placeholders for row-for-row comparison.
    expected = _logical(original, tmp_path)
    for m in history[4:]:
        expected.append((original[0]['session_id'], original[0]['conversation_id'],
                         m['role'], m['content'], m.get('tool_call_id'),
                         json.dumps(m['tool_calls']) if m.get('tool_calls') else None))
    return expected


@pytest.mark.parametrize('phase', ['first', 'new-turn', 'new-saved', 'twin', 'file-gone', 'third'])
def test_legacy_marker_upgrade_acceptance(legacy, phase, tmp_path):
    marker, path, body, original = legacy
    if phase == 'file-gone':
        path.rename(path.with_suffix('.gone'))
    extra = [('assistant', 'FINAL')]
    if phase in ('new-turn', 'new-saved', 'twin', 'third'):
        extra += list(NEW_TURN)
    if phase in ('new-saved', 'twin'):
        call_id = 'call_twin' if phase == 'twin' else 'call_new'
        saved = marker
        if phase == 'new-saved':
            new_path = path.with_name('new-result.txt')
            new_body = 'NEW_SAVED_RESULT:\n' + 'another result\n' * 500
            new_path.write_text(new_body)
            saved = _marker(new_path, new_body)
        calls = [{'id': call_id, 'type': 'function', 'function': {'name': 'dump', 'arguments': '{}'}}]
        extra += [('assistant', '', {'tool_calls': calls}),
                  ('tool', saved, {'tool_call_id': call_id}), ('assistant', 'Saved anew.')]
    history = _host_history(marker, extra)
    _run_phase(tmp_path, history)
    rows = _read_store(tmp_path)
    assert rows[:4] == original
    assert _tool_rows(rows)[CALL] == 1, f'first replay duplicated old result: rows={len(rows)}'
    if phase in ('new-saved', 'twin'):
        assert len(rows) == 10
        assert _tool_rows(rows) == {CALL: 1, call_id: 1}
        # A new saved result is recovered, even with identical marker text and a new id.
        new_row = next(r for r in rows if r['tool_call_id'] == call_id)
        assert _restore(new_row['content'], tmp_path / 'hermes' / 'externalized') == (
            body if phase == 'twin' else new_body)
        assert _logical(rows[:7], tmp_path) == _control(original, history[:7], tmp_path)
    else:
        assert len(rows) == (7 if phase in ('new-turn', 'third') else 5)
        assert _logical(rows, tmp_path) == _control(original, history, tmp_path)
    if phase == 'third':
        _run_phase(tmp_path, history)
        assert _read_store(tmp_path) == rows
        _run_phase(tmp_path, history)
        assert _read_store(tmp_path) == rows, 'third replay added rows'
