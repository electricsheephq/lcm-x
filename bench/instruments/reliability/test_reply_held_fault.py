import hashlib
import json

OLD_IDS = ['overflow-recovery-restart/in-place', 'baseline/in-place/acp', 'acp-trailing/in-place', 'repeat-identical-replies/in-place', 'crash-then-lcm-tool-in-merge-turn/in-place', 'gateway-second-restart/in-place', 'parallel-tool-group/in-place', 'lcm-tool-mid-turn/in-place', 'cancel-retry/in-place', 'long-80/in-place', 'multi-session-one-process/in-place', 'native-on-off/v0.23.3/in-place', 'native-on-off/v0.24.3/in-place', 'drain/hidden-backlog/in-place', 'drain/hidden-backlog-large/in-place', 'crash-after-compaction/in-place/acp-history', 'crash-after-compaction/in-place/gateway-reload', 'overflow-recovery-restart/rotation', 'baseline/rotation/acp', 'acp-trailing/rotation', 'repeat-identical-replies/rotation', 'crash-then-lcm-tool-in-merge-turn/rotation', 'gateway-second-restart/rotation', 'parallel-tool-group/rotation', 'lcm-tool-mid-turn/rotation', 'cancel-retry/rotation', 'long-80/rotation', 'multi-session-one-process/rotation', 'native-on-off/v0.23.3/rotation', 'native-on-off/v0.24.3/rotation', 'drain/hidden-backlog/rotation', 'drain/hidden-backlog-large/rotation', 'crash-after-compaction/rotation/acp-history', 'crash-after-compaction/rotation/gateway-reload', 'preflight-continue/in-place', 'publication-failure/rotation-child', 'publication-failure/rotation-child-persistent', 'publication-failure/pass-3-in-place', 'separator-heavy-retained/in-place', 'pressure-disagreement/in-place', 'window-1m/in-place', 'crash-after-rotation/rotation', 'upgrade-crash-after-rotation/rotation', 'crash-between-end-and-start/rotation', 'delegated-child/in-place', 'delegated-child/rotation', 'p8-control/archived', 'p8-control/other-active', 'p8-control/random-snapshot', 'p8-control/none']
OLD_SHA = "16e364c39241264ce704a0f9d5a31a3e960cd97c476c33c7e37ec73b928772e6"

import socket
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.instruments.reliability import cells, ci, fake_provider as FP, process_cell as PC, reply_fault as RF, run_matrix as RM


def fixture(tmp_path, depth=0, expected=None):
    run = object.__new__(PC.ProcessCell)
    run.home, run.d, run.transcript = tmp_path, tmp_path, tmp_path / 'transcript.jsonl'
    run.sid, run.phase, run.fired, run.log_mark = 'root', 'A', set(), 0
    run.host_log = lambda: ''
    run.proc = SimpleNamespace(kill=Mock())
    run.cell = cells.cell('fixture', [], in_place=True)
    run.text_tag = 'T05'
    run.reply_fault = None
    host = sqlite3.connect(tmp_path / 'state.db')
    host.executescript('CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT);'
                       'CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT);')
    host.executemany('INSERT INTO sessions VALUES (?, ?)',
                     [('root', None)] + [(f'child{i}', 'root' if i == 1 else f'child{i-1}') for i in range(1, depth + 1)])
    host.commit()
    host.close()
    store = sqlite3.connect(tmp_path / 'lcm.db')
    store.execute('CREATE TABLE messages(store_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)')
    store.close()
    held = RF.HeldReply(run, dict(kind=RF.KIND, turn=5, depth=depth if expected is None else expected),
                        PC.reply_text(run.cell, 'T', 5), timeout=0.01)
    run.reply_fault = held
    return run, held, f'child{depth}' if depth else 'root'


def insert(run, text, session='root', db='state.db'):
    con = sqlite3.connect(run.home / db)
    con.execute('INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)', (session, 'assistant', text))
    con.commit()
    con.close()


def released(held):
    assert held.done.is_set() and held.reservation is None and held.host is None
    con = sqlite3.connect(held.run.home / 'lcm.db', timeout=0)
    con.execute('BEGIN IMMEDIATE')
    con.close()


@pytest.mark.parametrize('depth', [0, 1, 2])
def test_valid_depth_and_crash_records(tmp_path, depth):
    run, held, session = fixture(tmp_path, depth)
    held.before_release()
    insert(run, held.text, session)
    held.after_release()
    released(held)
    assert held.record['valid'] and held.record['depth'] == depth
    run.proc.kill.assert_called_once()
    event = PC.read_jsonl(run.transcript)[0]
    assert event['event'] == 'crash' and event['fault'] == RF.KIND and event['phase'] == 'A'
    assert event['kill_monotonic_ns'] == held.record['kill_monotonic_ns']
    assert 'reply to' not in json.dumps(held.record)
    assert PC.read_jsonl(tmp_path / 'faults-fired.jsonl')[0]['kind'] == RF.KIND
    rec = {'verdict': 'FAIL'}
    held.verdict(rec, 'done')
    assert rec['verdict'] == 'FAIL'
    held.verdict(rec, 'error')
    assert rec['verdict'] == 'ERROR' and rec['reason'].startswith('reply-held fault INVALID: restart')


@pytest.mark.parametrize('shape,reason', [('root', 'wrong root'), ('depth', 'depth mismatch'),
    ('cycle', 'lineage cycle'), ('missing', 'lineage missing parent'), ('store', 'post-kill')])
def test_invalid_witnesses(tmp_path, shape, reason):
    run, held, session = fixture(tmp_path, 1, expected=2 if shape == 'depth' else 1)
    con = sqlite3.connect(tmp_path / 'state.db')
    if shape == 'root':
        run.sid = 'another-root'
    if shape in ('cycle', 'missing'):
        con.execute('UPDATE sessions SET parent_session_id = ? WHERE id = ?',
                    ('child1' if shape == 'cycle' else 'absent', 'child1'))
    con.commit()
    con.close()
    if shape == 'store':
        insert(run, held.text, db='lcm.db')
    held.before_release()
    insert(run, held.text, session)
    held.after_release()
    released(held)
    rec = {'verdict': 'PASS'}
    held.verdict(rec, 'done')
    assert rec['verdict'] == 'ERROR' and reason in rec['reason']


@pytest.mark.parametrize('shape', ['old', 'timeout', 'exception', 'busy', 'reservation'])
def test_cleanup_and_non_new_rows(tmp_path, shape):
    run, held, _ = fixture(tmp_path)
    if shape == 'old':
        insert(run, held.text)
    blocker = sqlite3.connect(tmp_path / 'lcm.db', isolation_level=None)
    if shape == 'reservation':
        blocker.execute('BEGIN IMMEDIATE')
    held.before_release()
    if shape in ('exception', 'busy'):
        original = held.host
        held.host = SimpleNamespace(execute=Mock(side_effect=ValueError if shape == 'exception' else sqlite3.OperationalError),
                                    close=original.close)
    held.after_release()
    blocker.close()
    released(held)
    assert not held.record['valid'] and held.record['invalid_reason']
    run.proc.kill.assert_not_called()
    if shape == 'busy':
        assert held.record['busy_poll_count'] == held.record['poll_count'] > 0


@pytest.mark.parametrize('hooks,stream', [(True, False), (True, True), (False, False)])
def test_provider_reserves_before_first_byte(tmp_path, hooks, stream, monkeypatch):
    run, held, _ = fixture(tmp_path)
    arrived, proceed, finished = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def before():
        held.before_release()
        arrived.set()
        assert proceed.wait(2)

    def after():
        held.release_failed()  # Ordering test only: no host to produce a row.
        finished.set()

    reply = {'content': 'normal reply'}
    if hooks:
        reply.update(before_release=before, after_release=after, release_failed=after)
    monkeypatch.setattr(FP.time, 'time', lambda: 123)
    provider = FP.FakeProvider(tmp_path / 'provider.jsonl', main=lambda _: reply).start()

    def client():
        try:
            with socket.create_connection(('127.0.0.1', provider.port), timeout=2) as sock:
                body = json.dumps(dict(model='rel/main', messages=[], stream=stream)).encode()
                sock.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n'
                             + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
                if hooks:
                    assert arrived.wait(2)
                    con = sqlite3.connect(tmp_path / 'lcm.db', timeout=0)
                    try:
                        with pytest.raises(sqlite3.OperationalError, match='database is locked'):
                            con.execute('BEGIN IMMEDIATE')
                        sock.settimeout(0.02)
                        with pytest.raises(TimeoutError):
                            sock.recv(1)
                    finally:
                        con.close()
                        proceed.set()
                        sock.settimeout(2)
                chunks = []
                while data := sock.recv(65536):
                    chunks.append(data)
                payload = b''.join(chunks).split(b'\r\n\r\n', 1)[1]
                if not stream:
                    expected = []
                    FP.FakeProvider.json_reply(SimpleNamespace(send_json=lambda status, data: expected.append(data)),
                                               'openai', 'rel/main', {'content': 'normal reply'}, 800, False)
                    assert payload == json.dumps(expected[0]).encode()
        except BaseException as exc:
            errors.append(exc)

    try:
        client()
        assert not errors, errors
        if hooks:
            assert finished.wait(2)
            released(held)
    finally:
        proceed.set()
        provider.stop()


@pytest.mark.parametrize('completion_timeout', [False, True])
def test_run_phase_waits_for_post_kill_completion(tmp_path, monkeypatch, completion_timeout):
    run, held, _ = fixture(tmp_path)
    run.phase_timeout, run.deadline, run.turn_timeout, run.last_turn = None, None, 1, 5
    run.argv, run.env, run.work, run.files = lambda: [], lambda: {}, tmp_path, tmp_path
    started, continue_validation = threading.Event(), threading.Event()
    original_validate = held.validate

    def validate(session):
        started.set()
        assert continue_validation.wait(2)
        original_validate(session)

    held.validate = validate
    if completion_timeout:
        held.done = Mock(wraps=held.done)
        held.done.wait.return_value = False
    stub = SimpleNamespace(killed=False, initialize=lambda _: None, new_session=lambda *_: 'root', close=lambda: -9)
    stub.kill = lambda: setattr(stub, 'killed', True)
    monkeypatch.setattr(PC.AD, 'AcpProcess', lambda *_: stub)
    monkeypatch.setattr(PC.P1, 'extend_turns', lambda *_: [5])

    def worker():
        held.before_release()
        insert(run, held.text)
        held.after_release()

    thread = threading.Thread(target=worker)

    def turn(_):
        thread.start()
        assert started.wait(2)
        raise PC.AD.ProcessGone('test crash')

    run.turn = turn
    result = []
    phase = threading.Thread(target=lambda: result.append(run.run_phase(5)))
    phase.start()
    assert started.wait(2)
    assert not held.done.is_set()
    if not completion_timeout:
        assert phase.is_alive() and not result
    continue_validation.set()
    phase.join(2)
    thread.join(2)
    assert result[0]['exit'] == ('error' if completion_timeout else 'crash')
    if not completion_timeout:
        assert result == [dict(exit='crash', next_turn=6, turn=5)]
    released(held)


def test_registry_unchanged_and_r1_rejects_early(tmp_path, monkeypatch):
    all_cells = cells.registry()
    new = [c for c in all_cells if c['id'].startswith('reply-held-crash/')]
    old = [c for c in all_cells if c not in new]
    assert [c['id'] for c in old] == OLD_IDS
    assert hashlib.sha256(json.dumps(old, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == OLD_SHA
    expected = [('rotation-turn', 12, 1, False, [2624]), ('plain-turn', 15, 1, False, [2624]),
                ('multi-hop', 22, 2, False, [2624]), ('pre-rotation', 5, 0, False, []), ('in-place', 15, 0, True, [])]
    assert len(new) == 5
    monkeypatch.setattr(RM, 'scratch_dir', Mock(side_effect=AssertionError('must reject before any probe')))
    for c, (name, turn, depth, ip, targets) in zip(new, expected):
        assert c['id'] == 'reply-held-crash/' + name
        assert c['faults'] == [dict(kind=RF.KIND, turn=turn, depth=depth)]
        assert (c['in_place'], c['transport'], c['targets']) == (ip, 'acp', targets)
        cells.validate(c)
        assert ci.in_gate_set(c['id']) and PC.unsupported(c, 'acp-process') is None
        result = RM.run_cell(c, 'stub', dict(sha='host', src='unused', python='unused'),
                             dict(sha='plugin', ref='stub'), tmp_path, 1, False, identity={'sha': 'host'})
        assert result['verdict'] == 'UNSUPPORTED' and result['reason'] == RF.R1_REASON


def test_scenario_arms_only_once_on_final_main_reply_in_phase_a(tmp_path):
    run, held, _ = fixture(tmp_path)
    run.cell['faults'] = [held.fault]
    run.reply_fault, run.crash_due = None, lambda _: False
    scenario = PC.Scenario(run)
    scenario.begin(4, 'normal', 'A', 1)
    assert 'before_release' not in scenario.main([dict(role='user', content='[T05] user')])
    scenario.begin(5, 'normal', 'B', 6)
    assert 'before_release' not in scenario.main([dict(role='user', content='[T05] user')])
    scenario.begin(5, 'normal', 'A', 1)
    reply = scenario.main([dict(role='user', content='[T05] user')])
    assert reply['content'] == held.text and callable(reply['before_release'])
    assert 'before_release' not in scenario.main([dict(role='user', content='[T05] user')])
