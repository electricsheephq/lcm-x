"""#2558 (I1/I6): real v0.29.2 L1/L2 stores retain each replay occurrence once."""

import copy
import json
import tempfile

import pytest

from tests.invariants.test_issue_2539_compare_matrix import (
    _host_rows, _match_host_rows, _restore, _snapshot, _v0292_matrix_store,
)
from tests.test_issue_10_persisted_output_spillover import _marker
from tests.test_issue_2521_spillover_upgrade_replay import (
    CALL, NEW_TURN, _host_history, _read_store, _run_phase, _tool_rows,
    _call_rows,
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


@pytest.mark.parametrize('shape', ['L1', 'L2', 'plain'])
def test_repeated_call_upgrade_replay_consumes_each_row_once(shape, tmp_path, monkeypatch):
    temp = tmp_path / 'host-temp'
    temp.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(temp))
    preview, location, crlf = {
        'L1': ('plain', 'hermes-results', True),
        'L2': ('data-uri', 'spillover', False),
        'plain': ('plain', 'spillover', False),
    }[shape]
    marker, _, body = _v0292_matrix_store(tmp_path, temp, preview, location, crlf)
    original = _read_store(tmp_path)
    history = _host_history(marker)
    second = copy.deepcopy(history[1:3])
    stamp = history[-1]['timestamp']
    for i, row in enumerate(second, 1):
        row['timestamp'] = stamp + i
    history += second + [{'role': 'assistant', 'content': 'second occurrence', 'timestamp': stamp + 3}]
    counts, snapshots = [len(original)], []
    for _ in range(3):
        _run_phase(tmp_path, history)
        rows = _read_store(tmp_path)
        counts.append(len(rows))
        snapshots.append(rows)
        assert rows[:4] == original
        storage, home = tmp_path / 'hermes' / 'externalized', tmp_path / 'hermes'
        multiset = _snapshot(rows, storage, home, temp, {})[3]
        extras, missing = _match_host_rows(multiset, _host_rows(history, body, home, temp))
        assert not missing, f'{shape}: missing host rows: {missing}'
        assert _tool_rows(rows)[CALL] >= 2 and _call_rows(rows)[CALL] >= 2
    print(f'REPEATED_CALL {shape} counts={counts}')
    assert counts == [4, 7, 7, 7], f'{shape}: later replay added rows: {counts}'
    assert not extras
    assert snapshots[0] == snapshots[1] == snapshots[2]


@pytest.mark.parametrize('shape', ['L1', 'L2', 'plain'])
def test_shifted_repeated_call_upgrade_replay(shape, tmp_path, monkeypatch):
    """#2558 review, R4-8 named limit: four earlier rows plus the last turn of a 208-row store, then the
    same call id and marker again. The second replay may store the repeat once more; main does the same for
    plain and ends at 213 for all three shapes. Bars: no host row missing, a third replay adds nothing,
    never above main's 213."""
    temp = tmp_path / 'host-temp'
    temp.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(temp))
    preview, location, crlf = {
        'L1': ('plain', 'hermes-results', True),
        'L2': ('data-uri', 'spillover', False),
        'plain': ('plain', 'spillover', False),
    }[shape]
    marker, path, body = _v0292_matrix_store(tmp_path, temp, preview, location, crlf)
    original = _host_history(marker)
    padding = [{'role': ('user', 'assistant')[i % 2], 'content': f'long row {i}'} for i in range(200)]
    history = original + padding + copy.deepcopy(original)
    for i, row in enumerate(history):
        row['timestamp'] = 1762700001 + i
    gone = path.with_suffix('.gone')
    path.rename(gone)
    _run_phase(tmp_path, history)
    gone.rename(path)
    before = _read_store(tmp_path)
    assert len(before) == 208
    history = copy.deepcopy(history[120:124] + history[-4:])
    second = copy.deepcopy(original[1:3])
    stamp = history[-1]['timestamp']
    for i, row in enumerate(second, 1):
        row['timestamp'] = stamp + i
    history += second + [{'role': 'assistant', 'content': 'second occurrence', 'timestamp': stamp + 3}]
    counts, snapshots = [len(before)], []
    for _ in range(3):
        _run_phase(tmp_path, history)
        rows = _read_store(tmp_path)
        assert rows[:208] == before
        counts.append(len(rows))
        snapshots.append(rows)
        storage, home = tmp_path / 'hermes' / 'externalized', tmp_path / 'hermes'
        multiset = _snapshot(rows, storage, home, temp, {})[3]
        _, missing = _match_host_rows(multiset, _host_rows(history, body, home, temp))
        assert not missing, f'{shape}: missing host rows: {missing}'
    print(f'SHIFTED_CALL {shape} counts={counts}')
    assert snapshots[1] == snapshots[2], f'{shape}: third replay added rows: {counts}'
    assert counts == [208, 211, 213, 213], f'{shape}: named-limit counts changed: {counts}'


def test_an_unreadable_tool_row_lookup_keeps_the_row_and_stores_every_host_row(legacy, tmp_path, monkeypatch):
    """CodeRabbit (#2572): a failed lookup in the legacy keep proves nothing; the row keeps its own identity."""
    from hermes_lcm.store import MessageStore

    marker, path, body, original = legacy
    calls = []

    def unreadable(self, *args, **kwargs):
        calls.append(1)
        raise RuntimeError('lookup failed')

    monkeypatch.setattr(MessageStore, 'find_session_tool_rows', unreadable)
    history = _host_history(marker, [('assistant', 'FINAL')])
    _run_phase(tmp_path, history)  # no exception escapes the ingest
    assert calls, 'the legacy keep did not reach the lookup'
    rows = _read_store(tmp_path)
    assert rows[:4] == original
    storage, home, temp = tmp_path / 'hermes' / 'externalized', tmp_path / 'hermes', tmp_path / 'host-temp'
    _, missing = _match_host_rows(_snapshot(rows, storage, home, temp, {})[3], _host_rows(history, body, home, temp))
    assert not missing, missing
