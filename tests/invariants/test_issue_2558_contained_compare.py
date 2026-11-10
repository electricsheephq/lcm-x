"""#2558 review (#2584): the replay's new bare-marker compare never costs a row that main stores.

Two sites in the replay use the bare-marker identity: the legacy keep and the ordered-prefix binding. A fault in
either falls back to main's behaviour, so the turn's new rows are still stored. The binding also requires each pair's
replay tool name (``tool_name`` or ``name``), so a re-sent tool row whose tool name changed is a new row, as on main.
"""

import copy

import pytest

from tests.test_issue_2521_spillover_upgrade_replay import (
    CALL, CONVERSATION, SESSION, _engine, _host_history, _read_store, _tool_rows, _v0292_store,
)

FINAL = {'role': 'assistant', 'content': 'FINAL', 'timestamp': 1762700010.0}


def _ingest(tmp_path, history, monkeypatch=None, fault=None):
    engine = _engine(tmp_path)
    engine.on_session_start(SESSION, platform='cli', context_length=128000, conversation_id=CONVERSATION)
    try:
        if fault:
            def raises(*_a, **_k):
                raise RuntimeError('injected compare fault')
            monkeypatch.setattr(engine, fault, raises)
        engine._ingest_messages(copy.deepcopy(history))
    finally:
        engine.shutdown()


@pytest.mark.parametrize('fault', ['_bare_marker_identity', '_session_end_rows_equal'])
def test_a_fault_in_the_bare_marker_compare_still_stores_the_new_rows(tmp_path, monkeypatch, fault):
    original, marker, _ = _v0292_store(tmp_path)
    history = _host_history(marker) + [FINAL]
    _ingest(tmp_path, history, monkeypatch, fault)
    rows = _read_store(tmp_path)
    assert rows[:len(original)] == original
    assert sum(row['content'] == 'FINAL' for row in rows) == 1, f'rows={len(rows)}'


def test_a_renamed_bare_marker_row_is_a_new_row(tmp_path):
    _, marker, _ = _v0292_store(tmp_path)
    history = _host_history(marker) + [FINAL]
    history[2]['name'] = 'different_tool'
    _ingest(tmp_path, history)
    rows = _read_store(tmp_path)
    assert _tool_rows(rows)[CALL] == 2, f'rows={len(rows)}'  # main: the renamed result is stored once more
    assert sum(row['content'] == 'FINAL' for row in rows) == 1


def test_a_renamed_plain_tool_row_behind_a_bare_marker_is_a_new_row(tmp_path):
    _, marker, _ = _v0292_store(tmp_path)
    extra = [{'role': 'assistant', 'content': '', 'tool_calls': [
                 {'id': 'p2', 'type': 'function', 'function': {'name': 'plain', 'arguments': '{}'}}]},
             {'role': 'tool', 'content': 'plain result', 'tool_call_id': 'p2'},
             {'role': 'assistant', 'content': 'second done'}]
    history = _host_history(marker) + extra
    _ingest(tmp_path, history)
    history[-2]['name'] = 'different_tool'
    _ingest(tmp_path, history + [FINAL])
    rows = _read_store(tmp_path)
    assert _tool_rows(rows)['p2'] == 2, f'rows={len(rows)}'  # main: the renamed result is stored once more
    assert sum(row['content'] == 'FINAL' for row in rows) == 1
