import hashlib
import json

OLD_IDS = ['overflow-recovery-restart/in-place', 'baseline/in-place/acp', 'acp-trailing/in-place', 'repeat-identical-replies/in-place', 'crash-then-lcm-tool-in-merge-turn/in-place', 'gateway-second-restart/in-place', 'parallel-tool-group/in-place', 'lcm-tool-mid-turn/in-place', 'cancel-retry/in-place', 'long-80/in-place', 'multi-session-one-process/in-place', 'native-on-off/v0.23.3/in-place', 'native-on-off/v0.24.3/in-place', 'drain/hidden-backlog/in-place', 'drain/hidden-backlog-large/in-place', 'crash-after-compaction/in-place/acp-history', 'crash-after-compaction/in-place/gateway-reload', 'overflow-recovery-restart/rotation', 'baseline/rotation/acp', 'acp-trailing/rotation', 'repeat-identical-replies/rotation', 'crash-then-lcm-tool-in-merge-turn/rotation', 'gateway-second-restart/rotation', 'parallel-tool-group/rotation', 'lcm-tool-mid-turn/rotation', 'cancel-retry/rotation', 'long-80/rotation', 'multi-session-one-process/rotation', 'native-on-off/v0.23.3/rotation', 'native-on-off/v0.24.3/rotation', 'drain/hidden-backlog/rotation', 'drain/hidden-backlog-large/rotation', 'crash-after-compaction/rotation/acp-history', 'crash-after-compaction/rotation/gateway-reload', 'preflight-continue/in-place', 'publication-failure/rotation-child', 'publication-failure/rotation-child-persistent', 'publication-failure/pass-3-in-place', 'separator-heavy-retained/in-place', 'pressure-disagreement/in-place', 'window-1m/in-place', 'crash-after-rotation/rotation', 'upgrade-crash-after-rotation/rotation', 'crash-between-end-and-start/rotation', 'delegated-child/in-place', 'delegated-child/rotation', 'p8-control/archived', 'p8-control/other-active', 'p8-control/random-snapshot', 'p8-control/none']
OLD_SHA = "a035220cba28ad23b3d548db07a618906022edaf0a5fcf7e3719ca28732d7035"  # #2624: crash-after-rotation no longer targets 2624

import socket
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.instruments.reliability import cells, ci, fake_provider as FP, process_cell as PC, reply_fault as RF, run_matrix as RM


@pytest.fixture(autouse=True)
def stub_host_group(monkeypatch):
    killpg = RF.os.killpg

    def signal_group(pid, sig):
        if pid == -2624:
            raise ProcessLookupError
        return killpg(pid, sig)

    monkeypatch.setattr(RF.os, 'killpg', signal_group)


def fixture(tmp_path, depth=0, expected=None):
    run = object.__new__(PC.ProcessCell)
    run.home, run.d, run.transcript = tmp_path, tmp_path, tmp_path / 'transcript.jsonl'
    run.sid, run.phase, run.fired, run.log_mark = 'root', 'A', set(), 0
    run.host_log = lambda: ''
    run.proc = SimpleNamespace(kill=Mock(), pid=-2624, process=SimpleNamespace(poll=lambda: -9))
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


@pytest.mark.parametrize('checks_fail', [False, True])
def test_kill_exit_checks_rollback_order(tmp_path, monkeypatch, checks_fail):
    run, held, _ = fixture(tmp_path)
    held.before_release()
    insert(run, held.text)
    order = []
    run.proc.kill.side_effect = lambda: order.append('kill')
    probe = Mock(side_effect=[None, PermissionError(), ProcessLookupError()])

    def group(pid, sig):
        assert order == ['kill'] and pid == run.proc.pid and sig == 0
        try:
            return probe()
        except ProcessLookupError:
            order.append('exit confirmed')
            raise

    validate = held.validate

    def checks(session):
        assert held.record['exit_confirmed_ms'] >= 0
        assert held.reservation.in_transaction
        order.append('checks')
        if checks_fail:
            raise ValueError('test validation failure')
        validate(session)

    held.reservation.set_trace_callback(lambda sql: order.append(sql) if sql == 'ROLLBACK' else None)
    monkeypatch.setattr(RF.os, 'killpg', group)
    monkeypatch.setattr(held, 'validate', checks)
    held.after_release()
    assert order == ['kill', 'exit confirmed', 'checks', 'ROLLBACK']
    assert held.record['valid'] is (not checks_fail)
    released(held)


@pytest.mark.parametrize('shape', ['present', 'permission', 'leader running'])
def test_exit_timeout_invalid_and_releases_reservation(tmp_path, monkeypatch, shape):
    run, held, _ = fixture(tmp_path)
    held.before_release()
    insert(run, held.text)
    clock = [0.0]
    monkeypatch.setattr(RF.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(RF.time, 'monotonic_ns', lambda: int(clock[0] * 1_000_000_000))

    def sleep(delay):
        assert delay == 0.005
        clock[0] += delay

    monkeypatch.setattr(RF.time, 'sleep', sleep)
    monkeypatch.setattr(RF.os, 'killpg', Mock(side_effect=(
        PermissionError if shape == 'permission' else ProcessLookupError if shape == 'leader running' else None)))
    if shape == 'leader running':
        run.proc.process.poll = lambda: None
    held.validate = Mock(side_effect=AssertionError('must skip checks'))
    held.after_release()
    assert 5 <= clock[0] < 5.01
    assert held.record['invalid_reason'] == 'host group did not exit'
    assert held.record['valid'] is False and held.record['exit_confirmed_ms'] is None
    held.validate.assert_not_called()
    released(held)


@pytest.mark.parametrize('delayed', [False, True])
def test_blocked_writer_cannot_commit_before_confirmed_exit(tmp_path, delayed):
    run, held, _ = fixture(tmp_path)
    held.before_release()
    script = ("import sqlite3, sys; con = sqlite3.connect(sys.argv[1], timeout=10); "
              "print('ready', flush=True); con.execute('BEGIN IMMEDIATE'); "
              "con.execute('INSERT INTO messages(content) VALUES (?)', (sys.argv[2],)); con.commit()")
    writer = subprocess.Popen([sys.executable, '-c', script, str(tmp_path / 'lcm.db'), held.text],
                              start_new_session=True, stdout=subprocess.PIPE, text=True)
    landed = threading.Event()

    def signal_writer():
        try:
            RF.os.killpg(writer.pid, 9)
        except ProcessLookupError:
            pass
        finally:
            landed.set()

    timer = threading.Timer(0.1, signal_writer)
    run.proc = SimpleNamespace(pid=writer.pid, process=writer, kill=timer.start if delayed else signal_writer)
    try:
        assert writer.stdout.readline().strip() == 'ready'
        assert writer.poll() is None
        insert(run, held.text)
        held.after_release()
        assert writer.poll() is not None and held.record['exit_confirmed_ms'] >= 0
        with sqlite3.connect(tmp_path / 'lcm.db') as store:
            assert store.execute('SELECT COUNT(*) FROM messages').fetchone()[0] == 0
        assert held.record['valid'] and held.record['store_has_row'] is False
        assert landed.wait(5)
        released(held)
    finally:
        if writer.poll() is None:
            signal_writer()
        if timer.ident is not None:
            timer.join(5)
            assert not timer.is_alive()
        writer.wait(timeout=5)
        writer.stdout.close()
        if not held.done.is_set():
            held.finish()


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
    assert event['polls'] == [[round(wall, 4), result] for wall, result in held.polls]
    assert len(event['polls']) == event['poll_count'] > 0
    assert 'witness_id' not in json.dumps(event)
    assert 'reply to' not in json.dumps(held.record)
    assert PC.read_jsonl(tmp_path / 'faults-fired.jsonl')[0]['kind'] == RF.KIND
    rec = {'verdict': 'FAIL'}
    held.verdict(rec, 'done')
    assert rec['verdict'] == 'FAIL'
    assert rec['reply_fault']['polls'] == event['polls']
    held.record['polls'].append([0, 'busy'])
    assert rec['reply_fault']['polls'] == event['polls']
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
        assert held.record['polls'] == [[round(wall, 4), 'busy'] for wall, _ in held.polls]


@pytest.mark.parametrize('hooks,stream', [(True, False), (True, True), (False, False), (False, True)])
def test_provider_reserves_before_first_byte(tmp_path, hooks, stream, monkeypatch):
    run, held, _ = fixture(tmp_path)
    arrived, proceed, finished = threading.Event(), threading.Event(), threading.Event()
    after_started, eof = threading.Event(), threading.Event()
    errors = []

    def before():
        held.before_release()
        arrived.set()
        assert proceed.wait(2)

    def after():
        after_started.set()
        try:
            if stream:
                assert eof.wait(2), 'client must receive EOF while after_release is running'
        finally:
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
                if hooks and stream:
                    assert after_started.wait(2) and not finished.is_set()
                    eof.set()
                payload = b''.join(chunks).split(b'\r\n\r\n', 1)[1]
                if stream:
                    assert payload.endswith(b'data: [DONE]\n\n')
                else:
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
        eof.set()
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
    stub = SimpleNamespace(killed=False, initialize=lambda _: None, new_session=lambda *_: 'root', close=lambda: -9,
                           pid=run.proc.pid, process=run.proc.process)
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


def test_stream_eof_while_after_release_runs_over_socketpair():
    """Exercise the real SSE writer and half-close without needing a listening port."""
    sender, client = socket.socketpair()
    client.settimeout(2)
    started, eof, finished = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def after():
        started.set()
        assert eof.wait(2)
        finished.set()

    provider = object.__new__(FP.FakeProvider)
    provider.next_rid, provider.log = lambda: 1, Mock()
    provider.continuity_context, provider.usage_scale = None, 1
    provider.reply_for = lambda *_: dict(content='normal reply', after_release=after)
    handler = SimpleNamespace(connection=sender, wfile=sender.makefile('wb'), path='/v1/chat/completions',
                              send_response=Mock(), send_header=Mock(), end_headers=Mock())

    def serve():
        try:
            provider.serve(handler, dict(model='rel/main', messages=[], stream=True), 'openai')
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        chunks = []
        while data := client.recv(65536):
            chunks.append(data)
        assert b''.join(chunks).endswith(b'data: [DONE]\n\n')
        assert started.wait(2) and not finished.is_set()
    finally:
        eof.set()
        thread.join(2)
        handler.wfile.close()
        sender.close()
        client.close()
    assert finished.is_set() and not thread.is_alive() and not errors


def test_registry_unchanged_and_r1_rejects_early(tmp_path, monkeypatch):
    all_cells = cells.registry()
    new = [c for c in all_cells if c['id'].startswith('reply-held-crash/')]
    old = [c for c in all_cells if c not in new]
    assert [c['id'] for c in old] == OLD_IDS
    assert hashlib.sha256(json.dumps(old, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == OLD_SHA
    expected = [('rotation-turn', 12, 1, False, []), ('plain-turn', 15, 1, False, []),
                ('multi-hop', 22, 2, False, []), ('pre-rotation', 5, 0, False, []), ('in-place', 15, 0, True, []),
                ('tool-turn', 15, 1, False, []), ('tool-loop', 15, 1, False, [])]
    assert len(new) == 7
    monkeypatch.setattr(RM, 'scratch_dir', Mock(side_effect=AssertionError('must reject before any probe')))
    for c, (name, turn, depth, ip, targets) in zip(new, expected):
        assert c['id'] == 'reply-held-crash/' + name
        assert c['faults'] == [dict(kind=RF.KIND, turn=turn, depth=depth,
                                  **({'at': 'tool_results'} if name == 'tool-loop' else {}))]
        assert (c['in_place'], c['transport'], c['targets']) == (ip, 'acp', targets)
        assert c['bars'] == list(cells.BARS)
        if name.startswith('tool-'):
            assert c['big_lines'] == 8 and c['user_text']['trailing_ws']
            assert c['tool_plan'] == [{'turns': [15], 'calls': [
                {'name': 'read_file', 'args': {'path': '{files}/small.txt'}},
                {'name': 'read_file', 'args': {'path': '{files}/big.txt'}}]}]
        cells.validate(c)
        assert ci.in_gate_set(c['id']) and PC.unsupported(c, 'acp-process') is None
        result = RM.run_cell(c, 'stub', dict(sha='host', src='unused', python='unused'),
                             dict(sha='plugin', ref='stub'), tmp_path, 1, False, identity={'sha': 'host'})
        assert result['verdict'] == 'UNSUPPORTED' and result['reason'] == RF.R1_REASON
    control_id = 'reply-held-crash-control/tool-loop-in-place'
    assert control_id not in {c['id'] for c in all_cells}
    assert control_id not in {c['id'] for c in PC.R2_PROBE_CELLS}
    assert control_id not in {c['id'] for c in cells.select('all', PC.R2_CELLS, PC.R2_PROBE_CELLS, PC.R2_CONTROL_CELLS)}
    assert control_id not in ci.expected_cells('acp-process')
    [control] = cells.select(control_id, PC.R2_CELLS, PC.R2_PROBE_CELLS, PC.R2_CONTROL_CELLS)
    assert 'controls=process_cell.R2_CONTROL_CELLS if a.transport else ()' in Path(RM.__file__).read_text()
    assert control['in_place'] and control['targets'] == []
    assert control['faults'] == [dict(kind=RF.KIND, turn=15, depth=0, at='tool_results')]


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


@pytest.mark.parametrize('name', ['tool-turn', 'tool-loop'])
@pytest.mark.parametrize('phase,already_fired', [('A', False), ('B', False), ('A', True)])
def test_tool_scenario_position_and_held_next_request(tmp_path, name, phase, already_fired):
    run, _, _ = fixture(tmp_path)
    run.cell = cells.held_tool_cell(name)
    run.reply_fault, run.crash_due, run.files, run.text_tag = None, lambda _: False, tmp_path, 'T15'
    if already_fired:
        run.fired.add(RF.KIND)
    scenario = PC.Scenario(run)
    scenario.begin(15, 'normal', phase, 16)
    messages = [dict(role='user', content='[T15] user')]
    first = scenario.main(messages)
    armed = phase == 'A' and not already_fired
    assert ('before_release' in first) == (armed and name == 'tool-loop')
    assert len(first['tool_calls']) == 2
    messages += [dict(role='tool', tool_call_id=c['id'], content='synthetic result') for c in first['tool_calls']]
    second = scenario.main(messages)
    assert ('before_release' in second) == (armed and name == 'tool-turn')
    if armed and name == 'tool-loop':
        assert run.reply_fault.ids == [c['id'] for c in first['tool_calls']]
        assert second == dict(hold_until_killed=True, log=dict(turn=15, kind='normal', step=1))
        assert scenario.main(messages)['hold_until_killed']
        assert not any(e['event'] == 'emit' for e in PC.read_jsonl(run.transcript))
        holds = [e for e in PC.read_jsonl(run.transcript) if e['event'] == 'reply_fault_hold']
        assert [(e['tag'], e['step'], e['session_prefix']) for e in holds] == [('T15', 1, 'T'), ('T15', 2, 'T')]
        assert all('text' not in e and 'content' not in e for e in holds)
    else:
        assert second['content'] == PC.reply_text(run.cell, 'T', 15)
        assert 'before_release' not in scenario.main(messages)
    assert len([e for e in PC.read_jsonl(run.transcript) if e['event'] == 'tool_seen']) == 2


def test_accounting_requires_held_tool_loop_request_event(tmp_path):
    from bench.instruments.reliability.scorers import bars, continuity
    from tests import test_reliability_scorers as fx

    events, _ = fx.tool_turn(15)
    events = fx.turn_events(15)[:1] + events
    hold = dict(phase='A', turn=15, event='reply_fault_hold', tag='T15', step=1, session_prefix='T')
    requests = [dict(rid=i, method='POST', path='/v1/chat/completions', role='main') for i in (1, 2)]
    (tmp_path / 'provider-requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
    assert not PC.accounting(tmp_path, events)['ok']
    accounted = PC.accounting(tmp_path, events + [hold])
    assert accounted['ok'] and accounted['main_implied']['reply_fault_holds'] == 1
    assert bars.attempts(events + [hold]) == bars.attempts(events)
    assert continuity.score_window(events + [hold]) == continuity.score_window(events)


def tool_fixture(tmp_path, depth=1):
    run, _, session = fixture(tmp_path, depth)
    for db in ('state.db', 'lcm.db'):
        with sqlite3.connect(tmp_path / db) as con:
            con.execute('ALTER TABLE messages ADD COLUMN tool_calls TEXT')
            con.execute('ALTER TABLE messages ADD COLUMN tool_call_id TEXT')
    held = RF.HeldReply(run, dict(kind=RF.KIND, turn=5, depth=depth, at='tool_results'), ['call_a', 'call_b'], timeout=1)
    run.reply_fault = held
    return run, held, session


def tool_row(run, session, *, cid=None, calls=None, db='state.db'):
    with sqlite3.connect(run.home / db) as con:
        con.execute('INSERT INTO messages(session_id, role, content, tool_calls, tool_call_id) VALUES (?,?,?,?,?)',
                    (session, 'tool' if cid else 'assistant', 'synthetic result' if cid else '',
                     json.dumps([{'id': i} for i in calls]) if calls else None, cid))


def test_tool_loop_waits_for_every_result_then_validates_depth(tmp_path, monkeypatch):
    run, held, session = tool_fixture(tmp_path, depth=2)
    held.before_release()
    tool_row(run, session, calls=held.ids)
    tool_row(run, session, cid=held.ids[0])
    original_sleep = RF.time.sleep

    def next_result(_delay):
        assert held.polls[-1][1] == 1
        run.proc.kill.assert_not_called()
        tool_row(run, session, cid=held.ids[1])

    monkeypatch.setattr(RF.time, 'sleep', next_result)
    held.after_release()
    monkeypatch.setattr(RF.time, 'sleep', original_sleep)
    released(held)
    assert [n for _, n in held.polls] == [1, 2]
    assert held.record['valid'] and held.record['depth'] == 2
    assert held.record['position'] == 'tool_results' and held.record['witness_rows'] == 2
    assert held.record['host_has_row'] and held.record['store_has_row'] is False
    assert held.record['exit_confirmed_ms'] is not None
    run.proc.kill.assert_called_once()
    assert RF.KIND in run.fired
    assert 'synthetic' not in json.dumps(held.record) and 'call_a' not in json.dumps(held.record)


def test_tool_loop_duplicate_result_cannot_replace_missing_id(tmp_path, monkeypatch):
    run, held, session = tool_fixture(tmp_path)
    held.before_release()
    tool_row(run, session, calls=held.ids)
    tool_row(run, session, cid=held.ids[0])
    tool_row(run, session, cid=held.ids[0])

    def missing_result(_delay):
        assert held.polls[-1][1] == 2
        run.proc.kill.assert_not_called()
        tool_row(run, session, cid=held.ids[1])

    monkeypatch.setattr(RF.time, 'sleep', missing_result)
    held.after_release()
    released(held)
    assert [n for _, n in held.polls] == [2, 3]
    assert held.record['valid'] and held.record['witness_rows'] == 3
    run.proc.kill.assert_called_once()


def test_tool_loop_post_exit_check_requires_every_distinct_id(tmp_path):
    run, held, session = tool_fixture(tmp_path)
    held.before_release()
    tool_row(run, session, calls=held.ids)
    for cid in held.ids:
        tool_row(run, session, cid=cid)

    def change_result_id():
        with sqlite3.connect(run.home / 'state.db') as con:
            con.execute('UPDATE messages SET tool_call_id = ? WHERE tool_call_id = ?', held.ids)

    run.proc.kill.side_effect = change_result_id
    held.after_release()
    released(held)
    assert not held.record['valid'] and held.record['host_has_row'] is False
    assert held.record['invalid_reason'] == 'post-kill host/store check'


@pytest.mark.parametrize('shape,reason', [('sessions', 'witness rows in several sessions'),
    ('missing-call', 'post-kill'), ('store-tool', 'post-kill'), ('store-call', 'post-kill')])
def test_tool_loop_invalid_host_or_store(tmp_path, shape, reason):
    run, held, session = tool_fixture(tmp_path)
    if shape.startswith('store-'):
        tool_row(run, session, db='lcm.db', **({'cid': held.ids[0]} if shape == 'store-tool' else {'calls': held.ids[:1]}))
    held.before_release()
    if shape != 'missing-call':
        tool_row(run, session, calls=held.ids)
    for i, cid in enumerate(held.ids):
        tool_row(run, 'root' if shape == 'sessions' and i else session, cid=cid)
    held.after_release()
    released(held)
    assert not held.record['valid'] and reason in held.record['invalid_reason']
    if shape == 'missing-call':
        assert held.record['host_has_row'] is False
    rec = dict(verdict='PASS')
    held.verdict(rec, 'done')
    assert rec['verdict'] == 'ERROR'


@pytest.mark.parametrize('shape', ['timeout', 'exception', 'busy'])
def test_tool_loop_poll_failure_kills_without_firing_and_releases(tmp_path, monkeypatch, shape):
    run, held, _ = tool_fixture(tmp_path)
    held.before_release()
    if shape != 'timeout':
        held.host = SimpleNamespace(close=held.host.close, execute=Mock(side_effect=(
            ValueError if shape == 'exception' else sqlite3.OperationalError)))
    clock = [0.0]
    monkeypatch.setattr(RF.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(RF.time, 'monotonic_ns', lambda: int(clock[0] * 1_000_000_000))
    monkeypatch.setattr(RF.time, 'sleep', lambda _: clock.__setitem__(0, clock[0] + 1))
    held.after_release()
    run.proc.kill.assert_called_once()
    assert not run.fired and held.record['valid'] is False
    assert held.record['exit_confirmed_ms'] is not None
    assert held.record['invalid_reason'].startswith('poll/validation' if shape == 'exception' else 'poll timeout')
    released(held)


@pytest.mark.parametrize('failure', ['reservation', 'release', 'release-after-reservation', 'invalid'])
def test_tool_loop_pre_poll_failure_kills_without_firing_and_releases(tmp_path, monkeypatch, failure):
    run, held, _ = tool_fixture(tmp_path)
    readonly = RF.readonly
    if failure in ('reservation', 'release-after-reservation'):
        monkeypatch.setattr(RF, 'readonly', Mock(side_effect=sqlite3.OperationalError))
    held.before_release()
    monkeypatch.setattr(RF, 'readonly', readonly)
    if failure == 'invalid':
        held.invalid('test invalid before poll')
    if failure.startswith('release'):
        held.release_failed()
    else:
        held.after_release()
    # Repeated callbacks must neither signal again nor fire a valid fault.
    held.after_release()
    run.proc.kill.assert_called_once()
    assert not run.fired and held.record['valid'] is False
    assert held.record['kill_wall'] is not None and held.record['exit_confirmed_ms'] is not None
    assert held.record['poll_count'] == 0
    assert held.record['reservation_result'] == ('OperationalError' if 'reservation' in failure else 'ok')
    released(held)


@pytest.mark.parametrize('stored_tools', [0, 1, 3])
def test_tool_turn_records_store_tool_rows_without_changing_validity(tmp_path, stored_tools):
    run, _, session = tool_fixture(tmp_path)
    run.cell = cells.held_tool_cell('tool-turn')
    held = RF.HeldReply(run, run.cell['faults'][0], PC.reply_text(run.cell, 'T', 15), timeout=1)
    prefix = 'call_T15_'
    # Matching assistant rows count once even when they name both calls.
    rows = [dict(calls=[prefix + '0_0', prefix + '0_1']), dict(cid=prefix + '0_0'), dict(cid=prefix + '0_1')]
    for row in rows[:stored_tools]:
        tool_row(run, session, db='lcm.db', **row)
    for row in [dict(cid='call_T14_0_0'), dict(calls=['call_T16_0_0'])]:
        tool_row(run, session, db='lcm.db', **row)
    held.before_release()
    insert(run, held.text, session)
    held.after_release()
    released(held)
    assert held.record['valid'] and held.record['store_has_row'] is False
    assert held.record['store_turn_tool_rows'] == stored_tools
    assert 'synthetic' not in json.dumps(held.record) and prefix not in json.dumps(held.record)


@pytest.mark.parametrize('shape', ['once', 'missing', 'duplicate', 'host-lacks', 'no-fault', 'no-fault-missing'])
def test_host_held_tool_rows_are_expected_only_for_held_faults(tmp_path, shape):
    from tests import test_reliability_scorers as fx

    ev, tools = fx.tool_turn(15)
    user = fx.U.format(15, 15)
    host_tools, stored_tools = list(tools), list(tools)
    if shape in ('missing', 'no-fault-missing'):
        stored_tools = tools[:-1]
    elif shape == 'duplicate':
        stored_tools += [tools[-1]]
        host_tools += [tools[-1]]  # Host multiplicity never licenses a second stored copy.
    elif shape == 'host-lacks':
        host_tools = tools[:-1]
    host = [('S0', r, c, *((rest + [None, None])[:2]), i)
            for i, (r, c, *rest) in enumerate([('user', user)] + host_tools, 1)]
    faults = [] if shape.startswith('no-fault') else [dict(kind=RF.KIND, turn=15)]
    out = fx.make(tmp_path, rows=[('user', user)] + stored_tools,
                  events=fx.turn_events(15)[:1] + ev, host_occurrences=host, faults=faults,
                  tool_plan=[{'turns': [15], 'calls': []}], bars=['B1', 'B2', 'B6'])
    numbers = out['numbers']['B2']
    assert numbers['tool_missing_rows'] == int(shape == 'missing')
    assert numbers['tool_surplus_rows'] == int(shape in ('duplicate', 'host-lacks'))
    assert out['verdict'] == ('FAIL' if shape in ('missing', 'duplicate', 'host-lacks') else 'PASS')
    if shape.startswith('no-fault'):
        assert 'host_held_tool_rows' not in numbers
    else:
        receipt = numbers['host_held_tool_rows']
        assert receipt['rows'] == (3 if shape == 'host-lacks' else 4)
        assert all(set(r) == {'tag', 'session', 'id', 'kind', 'key'} and len(r['key']) == 12
                   for r in receipt['records'])
        assert {r['kind'] for r in receipt['records']} == {'call', 'tool'}


@pytest.mark.parametrize('signature', ['issued', 'name', 'args', 'both'])
def test_host_held_call_expectation_uses_issued_signature(tmp_path, signature):
    from tests import test_reliability_scorers as fx

    ev, tools = fx.tool_turn(15)
    entries = json.loads(tools[0][2])
    issued = entries[0]
    altered = json.loads(json.dumps(issued))
    altered['function']['name' if signature == 'name' else 'arguments'] = (
        'other_tool' if signature == 'name' else json.dumps({'path': 'other.txt'}))
    host_tools, stored_tools = list(tools), list(tools)
    if signature in ('name', 'args'):
        host_tools[0] = stored_tools[0] = ('assistant', '', json.dumps([altered, entries[1]]))
    elif signature == 'both':
        host_tools += [('assistant', '', json.dumps([altered, issued]))]
    user = fx.U.format(15, 15)
    host = [('S0', r, c, *((rest + [None, None])[:2]), i)
            for i, (r, c, *rest) in enumerate([('user', user)] + host_tools, 1)]
    out = fx.make(tmp_path, rows=[('user', user)] + stored_tools, host_occurrences=host,
                  events=fx.turn_events(15)[:1] + ev, bars=['B1', 'B2', 'B6'],
                  tool_plan=[{'turns': [15], 'calls': []}], faults=[dict(kind=RF.KIND, turn=15)])
    numbers = out['numbers']['B2']
    mismatch = signature in ('name', 'args')
    assert numbers['tool_missing_rows'] == numbers['tool_surplus_rows'] == int(mismatch)
    assert out['verdict'] == ('FAIL' if mismatch else 'PASS')
    assert ('B2' in out['failed_bars']) == mismatch
    receipt = numbers['host_held_tool_rows']
    assert receipt['rows'] == 4
    call_records = [r for r in receipt['records'] if r['kind'] == 'call' and r['id'] == issued['id']]
    issued_key = ('call', issued['id'], issued['function']['name'],
                  json.dumps(json.loads(issued['function']['arguments']), sort_keys=True))
    assert call_records == [dict(tag='T15', session='chat', id=issued['id'], kind='call',
                                key=hashlib.sha256(json.dumps(issued_key).encode()).hexdigest()[:12])]


@pytest.mark.parametrize('stored_call', [False, True])
def test_host_absent_call_is_neither_expected_nor_loose(tmp_path, stored_call):
    from tests import test_reliability_scorers as fx

    ev, tools = fx.tool_turn(15)
    host_tools = [('assistant', '', json.dumps(json.loads(tools[0][2])[1:]))] + tools[1:]
    user = fx.U.format(15, 15)
    host = [('S0', r, c, *((rest + [None, None])[:2]), i)
            for i, (r, c, *rest) in enumerate([('user', user)] + host_tools, 1)]
    out = fx.make(tmp_path, rows=[('user', user)] + (tools if stored_call else host_tools),
                  host_occurrences=host, events=fx.turn_events(15)[:1] + ev, bars=['B1', 'B2', 'B6'],
                  tool_plan=[{'turns': [15], 'calls': []}], faults=[dict(kind=RF.KIND, turn=15)])
    numbers = out['numbers']['B2']
    assert numbers['tool_missing_rows'] == 0
    assert numbers['tool_surplus_rows'] == int(stored_call)
    assert out['verdict'] == ('FAIL' if stored_call else 'PASS')
    receipt = numbers['host_held_tool_rows']
    assert receipt['rows'] == 3
    assert not any(r['kind'] == 'call' and r['id'] == ev[0]['id'] for r in receipt['records'])


@pytest.mark.parametrize('held,shape', [(True, 'ok'), (True, 'failed'), (True, 'missing'),
    (True, 'name'), (True, 'args'), (True, 'short'), (True, 'extra'), (True, 'idless'),
    (True, 'reused'), (False, 'failed')])
def test_unfinished_held_turn_validates_dispatches(tmp_path, held, shape):
    from tests import test_reliability_scorers as fx

    calls = [('read_file', {'path': path}) for path in ('small.txt', 'big.txt')]
    ev, tools = fx.tool_turn(15, calls=calls)
    dispatches = [e for e in ev if e['event'] == 'tool_dispatch']
    first = dispatches[0]
    failure = None
    gap = 'T15: planned read_file (call_T15_0_0) was never dispatched by the host'
    if shape == 'failed':
        first.update(ok=False, detail='synthetic error')
        failure = 'T15: read_file (call_T15_0_0) failed: synthetic error'
    elif shape == 'missing':
        ev.remove(first)
    elif shape in ('name', 'args'):
        first[shape] = 'other_tool' if shape == 'name' else {'path': 'other.txt'}
        failure = f"T15: call_T15_0_0 planned read_file but the host ran {first['name']}"
    elif shape == 'short':
        ev[0]['expect'] = {'min_chars': first['chars'] + 1}
        failure = 'T15: read_file result 2 chars < 3'
    elif shape == 'extra':
        ev.append({**first, 'id': 'unplanned'})
        failure = 'T15: unplanned host dispatch of read_file'
    elif shape == 'idless':
        for e in dispatches:
            e.pop('id')
            e['args'] = json.dumps(e['args'])  # Canonical arguments bind id-less dispatches.
    elif shape == 'reused':
        ev.remove(dispatches[1])
        first.pop('id')
        ev[1]['args'] = ev[0]['args']
        entries = json.loads(tools[0][2])
        entries[1]['function']['arguments'] = json.dumps(ev[1]['args'])
        tools[0] = ('assistant', '', json.dumps(entries))  # Isolate dispatch reuse from call-signature drift.
        gap = 'T15: planned read_file (call_T15_0_1) was never dispatched by the host'
    user = fx.U.format(15, 15)
    host = [('S0', r, c, *((rest + [None, None])[:2]), i)
            for i, (r, c, *rest) in enumerate([('user', user)] + tools, 1)]
    out = fx.make(tmp_path, rows=[('user', user)] + tools, host_occurrences=host,
                  events=fx.turn_events(15)[:1] + ev, bars=['B1', 'B2', 'B6'],
                  tool_plan=[{'turns': [15], 'calls': []}],
                  faults=[dict(kind=RF.KIND, turn=15)] if held else [])
    assert out['numbers']['B2']['tool_missing_rows'] == out['numbers']['B2']['tool_surplus_rows'] == 0
    if held and shape in ('missing', 'reused'):
        assert out['verdict'] == 'UNSUPPORTED' and gap in out['reason']
    elif held and failure:
        assert out['verdict'] == 'FAIL' and set(out['failed_bars']) == {'B6'}
        assert out['failed_bars']['B6']['tool_failures'] == [failure]
    else:
        assert out['verdict'] == 'PASS' and not out['failed_bars']
    assert out['numbers']['B6']['tool_execution_failures'] == int(held and failure is not None)


@pytest.mark.parametrize('shape', ['missing-db', 'bad-db', 'query-error'])
@pytest.mark.parametrize('stored', [False, True])
@pytest.mark.parametrize('missing_user', [False, True])
def test_unavailable_host_tool_rows_never_pass(tmp_path, monkeypatch, shape, stored, missing_user):
    from bench.instruments.reliability.scorers import tool_calls
    from tests import test_reliability_scorers as fx

    ev, tools = fx.tool_turn(15)
    state = tmp_path / 'absent.db'
    if shape == 'bad-db':
        state.write_text('not sqlite')
    elif shape == 'query-error':
        with sqlite3.connect(state) as con:
            con.execute('CREATE TABLE messages(id INTEGER)')
    host_held = tool_calls.host_held
    monkeypatch.setattr(tool_calls, 'host_held', lambda a, b, _state, g, t, lineage: host_held(a, b, state, g, t, lineage))
    out = fx.make(tmp_path, rows=([] if missing_user else [('user', fx.U.format(15, 15))]) + (tools if stored else []),
                  events=fx.turn_events(15)[:1] + ev, bars=['B1', 'B2', 'B6'],
                  tool_plan=[{'turns': [15], 'calls': []}], faults=[dict(kind=RF.KIND, turn=15)])
    receipt = out['numbers']['B2']['host_held_tool_rows']
    assert receipt['unavailable'].startswith('state.db unreadable')
    assert 'B6' in out['applicable_bars'] and 'not_applicable' not in out['numbers']['B6']
    if not stored:
        assert out['verdict'] == 'UNSUPPORTED' and 'B6: no tool-call group was stored' in out['reason']
    if missing_user:
        assert out['verdict'] == ('FAIL' if stored else 'UNSUPPORTED') and 'B2' in out['failed_bars']
        assert 'B2' not in out['inconclusive_bars']
    else:
        assert out['verdict'] == ('INCONCLUSIVE' if stored else 'UNSUPPORTED') and not out['failed_bars']
        assert out['inconclusive_bars']['B2'] == 'host tool rows unavailable: ' + receipt['unavailable']


@pytest.mark.parametrize('held,stored,host_tools,missing_user', [
    (True, False, False, False), (True, False, False, True),
    (True, False, True, False), (True, False, True, True),
    (True, True, True, False), (False, False, True, False)])
def test_b6_no_group_exception_requires_host_held_tool_rows(tmp_path, held, stored, host_tools, missing_user):
    from tests import test_reliability_scorers as fx

    ev, tools = fx.tool_turn(15)
    user = fx.U.format(15, 15)
    host = [('S0', r, c, *((rest + [None, None])[:2]), i)
            for i, (r, c, *rest) in enumerate([('user', user)] + (tools if host_tools else []), 1)]
    out = fx.make(tmp_path, rows=([] if missing_user else [('user', user)]) + (tools if stored else []),
                  host_occurrences=host,
                  events=fx.turn_events(15)[:1] + ev, bars=['B1', 'B2', 'B6'],
                  tool_plan=[{'turns': [15], 'calls': []}], faults=[dict(kind=RF.KIND, turn=15)] if held else [])
    if held and host_tools:
        assert out['numbers']['B2']['host_held_tool_rows']['rows'] == 4
        assert ('B6' in out['applicable_bars']) == stored
        assert out['verdict'] == ('FAIL' if missing_user or not stored else 'PASS')
        assert out['numbers']['B2']['tool_missing_rows'] == (0 if stored else 4)
        if not stored:
            assert out['numbers']['B6']['not_applicable'] == 'no tool group stored'
        else:
            assert 'not_applicable' not in out['numbers']['B6']
    else:
        assert out['verdict'] == 'UNSUPPORTED' and 'B6: no tool-call group was stored' in out['reason']
        assert 'B6' in out['applicable_bars'] and 'not_applicable' not in out['numbers']['B6']
        if held:
            assert out['numbers']['B2']['host_held_tool_rows']['rows'] == 0


def test_b6_no_group_exception_requires_a_receipt(tmp_path):
    from tests import test_reliability_scorers as fx

    out = fx.make(tmp_path, rows=fx.clean_rows(1), events=fx.turn_events(1), host_occurrences=[],
                  bars=['B1', 'B2', 'B6'], tool_plan=[{'turns': [1], 'calls': []}],
                  faults=[dict(kind=RF.KIND, turn=1)])
    assert 'host_held_tool_rows' not in out['numbers']['B2']
    assert out['verdict'] == 'UNSUPPORTED' and 'B6: no tool-call group was stored' in out['reason']
    assert 'B6' in out['applicable_bars'] and 'not_applicable' not in out['numbers']['B6']


@pytest.mark.parametrize('failure', ['orphan', 'tool'])
def test_held_tool_cell_no_stored_group_keeps_b6_failures(tmp_path, failure):
    from tests import test_reliability_scorers as fx

    ev, tools = fx.tool_turn(15, ok=failure != 'tool')
    turn = fx.turn_events(15)
    events = turn[:1] + ev + (turn[1:] if failure == 'tool' else [])
    rows = [('user', fx.U.format(15, 15))] + ([('assistant', fx.R.format(15, 15))] if failure == 'tool' else [])
    host = [('S0', r, c, *((rest + [None, None])[:2]), i)
            for i, (r, c, *rest) in enumerate(tools, 1)] if failure == 'orphan' else []
    out = fx.make(tmp_path, rows=rows, events=events, host_occurrences=host,
                  phase={'counters': {'orphan_drops': int(failure == 'orphan')}}, bars=['B1', 'B2', 'B6'],
                  tool_plan=[{'turns': [15], 'calls': []}], faults=[dict(kind=RF.KIND, turn=15)])
    assert out['verdict'] == ('FAIL' if failure == 'orphan' else 'UNSUPPORTED')
    assert 'B6' in out['applicable_bars'] and 'B6' in out['failed_bars']
    assert out['numbers']['B6']['groups'] == 0 and 'not_applicable' not in out['numbers']['B6']
    assert out['numbers']['B6']['host_orphan_drops' if failure == 'orphan' else 'tool_execution_failures'] > 0


def test_held_tool_cell_still_fails_b6_for_a_stored_split_group(tmp_path):
    from tests import test_reliability_scorers as fx

    ev, tools = fx.tool_turn(15)
    user = fx.U.format(15, 15)
    out = fx.make(tmp_path, rows=[('user', user)] + tools, nodes=[[2]],
                  events=fx.turn_events(15)[:1] + ev, bars=['B1', 'B2', 'B6'],
                  tool_plan=[{'turns': [15], 'calls': []}], faults=[dict(kind=RF.KIND, turn=15)])
    assert out['verdict'] == 'FAIL' and set(out['failed_bars']) == {'B6'}
    assert out['numbers']['B6']['split_groups'] == 1


@pytest.mark.parametrize('change', ['delete', 'role', 'content'])
def test_post_kill_check_requires_the_witnessed_row(tmp_path, change):
    run, held, _ = fixture(tmp_path)
    held.before_release()
    insert(run, held.text)

    def kill():
        with sqlite3.connect(run.home / 'state.db') as con:
            if change == 'delete':
                con.execute('DELETE FROM messages WHERE id = 1')
            else:
                con.execute(f'UPDATE messages SET {change} = ? WHERE id = 1', ('changed',))
            con.execute('INSERT INTO messages VALUES (100, ?, ?, ?)', ('root', 'assistant', held.text))

    run.proc.kill.side_effect = kill
    held.after_release()
    released(held)
    assert not held.record['host_has_row'] and not held.record['valid']
    assert held.record['invalid_reason'] == 'post-kill host/store check'


@pytest.mark.parametrize('verdict', ['PASS', 'FAIL', 'ERROR'])
def test_invalid_overrides_scored_verdict_and_preserves_earlier_error(tmp_path, verdict):
    _, held, _ = fixture(tmp_path)
    held.invalid('test witness')
    rec = dict(verdict=verdict, reason='earlier failure ' + 'x' * 330, failed_bars={'B1': {}})
    old = rec['reason']
    held.verdict(rec, 'done')
    assert rec['verdict'] == 'ERROR' and rec['reply_fault']['valid'] is False
    assert rec['reason'] == 'reply-held fault INVALID: test witness' + (
        '; earlier: ' + old[:300] if verdict == 'ERROR' else '')


@pytest.mark.parametrize('failure', ['phase-A death before turn 5', 'network STOP'])
def test_never_armed_done_preserves_pre_arm_errors(tmp_path, monkeypatch, failure):
    cell = next(c for c in cells.registry() if c['id'] == 'reply-held-crash/pre-rotation')
    last = dict(exit='error', reason=failure) if failure != 'network STOP' else dict(exit='done')
    run = SimpleNamespace(reply_fault=None, phase='A', sid='root', provider=Mock(), proxy=Mock(),
                          scenario=SimpleNamespace(unexpected=[]), run_phase=Mock(return_value=last),
                          phase_record=lambda *_: dict(citations={}, network_attempts=(
                              [{'fixture': 'blocked'}] if failure == 'network STOP' else [])))
    monkeypatch.setattr(PC, 'ProcessCell', lambda *_args, **_kwargs: run)
    monkeypatch.setattr(PC, 'config_yaml', lambda *_: '')
    monkeypatch.setattr(PC.P1, 'write_tool_files', lambda *_: None)
    monkeypatch.setattr(PC.RM, 'copy_dbs', lambda *_: [])
    monkeypatch.setattr(PC.RM, 'release_scratch', lambda *_: None)
    monkeypatch.setattr(PC, 'accounting', lambda *_: dict(ok=True, unexpected_requests=[]))
    monkeypatch.setattr(PC.continuity, 'score_dir', lambda *_: {})
    monkeypatch.setattr(PC.RM, 'verdict_fields', Mock(side_effect=AssertionError('invalid run must not score')))
    rec = PC.run_cell_process(cell, 'fixture', dict(sha='host', src='unused', python='unused'),
                              dict(sha='plugin', ref='fixture', dir='fixture', tree=str(tmp_path)),
                              tmp_path, 1, False, identity=dict(sha='host'), scratch_root=tmp_path)
    expected = failure if failure != 'network STOP' else (
        "STOP: host attempted non-localhost network access: [{'fixture': 'blocked'}]")
    assert rec['verdict'] == 'ERROR' and rec['reply_fault']['valid'] is False
    assert rec['reason'] == 'reply-held fault INVALID: trigger never armed; earlier: ' + expected
    assert json.loads((Path(rec['dir']) / 'verdict.json').read_text()) == rec


@pytest.mark.parametrize('verdict', ['PASS', 'FAIL', 'ERROR'])
def test_offline_rescore_refuses_invalid_reply_fault(tmp_path, capsys, verdict):
    from bench.instruments.reliability.scorers import cli
    (tmp_path / 'cell.json').write_text('{}')
    (tmp_path / 'verdict.json').write_text(json.dumps(dict(verdict=verdict, reply_fault=dict(valid=False))))
    with pytest.raises(SystemExit) as exc:
        cli.main(['--cell-dir', str(tmp_path)])
    assert exc.value.code == 2
    assert 'reply-held fault INVALID (reply_fault.valid is false)' in capsys.readouterr().err


@pytest.mark.parametrize('evidence', [{}, {'reply_fault': None}, {'reply_fault': {}},
                                    {'reply_fault': {'valid': None}}, {'reply_fault': {'valid': 1}}])
def test_offline_rescore_requires_affirmative_reply_fault_validity(tmp_path, capsys, evidence):
    from bench.instruments.reliability.scorers import cli
    (tmp_path / 'cell.json').write_text(json.dumps({'faults': [{'kind': RF.KIND}]}))
    (tmp_path / 'verdict.json').write_text(json.dumps({'verdict': 'PASS', **evidence}))
    with pytest.raises(SystemExit) as exc:
        cli.main(['--cell-dir', str(tmp_path)])
    assert exc.value.code == 2
    assert 'reply-held fault validity is missing' in capsys.readouterr().err


@pytest.mark.parametrize('outcome', ['kill', 'invalid', 'timeout'])
def test_turn_waits_for_witness_before_advancing(tmp_path, monkeypatch, outcome):
    run, held, _ = fixture(tmp_path)
    run.phase_timeout, run.turn_timeout, run.last_turn = None, 1, 0
    run.argv, run.env, run.work, run.files = lambda: [], lambda: {}, tmp_path, tmp_path
    run.cell.update(final_compaction_check=False)
    run.scenario = PC.Scenario(run)
    prompted, waiting, allow_completion = threading.Event(), threading.Event(), threading.Event()
    turns, results, errors = [], [], []
    stub = SimpleNamespace(killed=False, initialize=lambda _: None, new_session=lambda *_: 'root', close=lambda: 0)
    stub.kill = lambda: setattr(stub, 'killed', True)

    def witness():
        try:
            assert allow_completion.wait(2)
            if outcome == 'kill':
                stub.kill()
                run.fire(RF.KIND, 5)
            else:
                held.invalid('poll timeout')
        except BaseException as exc:
            errors.append(exc)
        finally:
            held.done.set()

    worker = threading.Thread(target=witness)

    def prompt(_sid, text, _timeout):
        turns.append(text)
        if len(turns) == 1:
            if outcome != 'timeout':
                worker.start()
            prompted.set()
        return 'reply', 'end_turn'

    stub.prompt = prompt
    if outcome == 'timeout':
        held.done = Mock(wraps=held.done)
        held.done.wait.return_value = False
    else:
        original_wait = held.done.wait
        held.done.wait = lambda timeout: (waiting.set(), original_wait(timeout))[1]
    monkeypatch.setattr(PC.AD, 'AcpProcess', lambda *_: stub)
    monkeypatch.setattr(PC.P1, 'extend_turns', lambda *_: [5, 6])
    phase = threading.Thread(target=lambda: results.append(run.run_phase(5)))
    phase.start()
    try:
        assert prompted.wait(2)
        if outcome != 'timeout':
            assert waiting.wait(2)
            assert phase.is_alive() and not results and len(turns) == 1
    finally:
        allow_completion.set()
        phase.join(2)
        if worker.ident is not None:
            worker.join(2)
    assert not phase.is_alive() and not errors
    if outcome == 'kill':
        assert results == [dict(exit='crash', next_turn=6, turn=5)] and len(turns) == 1
    else:
        assert results[0]['exit'] == 'done' and len(turns) == 2
        assert held.record['invalid_reason'] == ('completion timeout' if outcome == 'timeout' else 'poll timeout')


@pytest.mark.parametrize('error', [PC.AD.JsonRpcError('session/prompt', {'code': -1}),
                                 PC.AD.ProcessGone('test crash')])
def test_prompt_error_during_witness_kill(tmp_path, monkeypatch, error):
    run, held, _ = fixture(tmp_path)
    run.phase_timeout, run.turn_timeout, run.last_turn = None, 1, 0
    run.argv, run.env, run.work, run.files = lambda: [], lambda: {}, tmp_path, tmp_path
    run.scenario = PC.Scenario(run)
    stub = SimpleNamespace(killed=False, initialize=lambda _: None, new_session=lambda *_: 'root', close=lambda: -9,
                           pid=run.proc.pid, process=run.proc.process)
    stub.kill = lambda: setattr(stub, 'killed', True)

    def prompt(*_):
        held.before_release()
        insert(run, held.text)
        held.after_release()
        raise error

    stub.prompt = prompt
    monkeypatch.setattr(PC.AD, 'AcpProcess', lambda *_: stub)
    monkeypatch.setattr(PC.P1, 'extend_turns', lambda *_: [5])
    result = run.run_phase(5)
    released(held)
    if isinstance(error, PC.AD.ProcessGone):
        assert result == dict(exit='crash', next_turn=6, turn=5)
        assert held.record['valid']
    else:
        assert result['exit'] == 'error' and result['reason'].startswith('JsonRpcError:')
        assert held.record['valid'] is False
        assert held.record['invalid_reason'].startswith('prompt error')
