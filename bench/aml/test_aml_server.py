"""Focused pytest tests: only synthetic messages and a disk-backed client stub."""
from contextlib import contextmanager
from email.message import Message
from http.client import HTTPResponse
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

import pytest

import aml_server
from aml_server import (APIError, MAX_BYTES, Server, Service, add_generation,
                        authenticated, digest, prune_generations, run_worker,
                        search_response, user_lock, validate)


class StubClient:
    calls = 0
    fail = False

    def __init__(self, root):
        self.root = root

    def add(self, messages, user_id, **kwargs):
        type(self).calls += 1
        folder = self.root / user_id
        folder.mkdir(exist_ok=True)
        path = folder / 'messages.json'
        previous = json.loads(path.read_text()) if path.exists() else []
        path.write_text(json.dumps(previous + messages))
        if type(self).fail:
            raise RuntimeError('simulated failure after partial write')


def stub_runner(root, operation, body, deadline):
    if operation == 'add':
        return add_generation(root, body, StubClient)
    path = root / digest(body['user_id']) / 'messages.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    return search_response('\n'.join(row['content'] for row in rows), digest(body['user_id']), body['top_k'])


def addition(user='user/a', request='request-1', session='session-1', content='synthetic moonflower'):
    return dict(user_id=user, request_id=request, session_id=session,
                messages=[dict(role='user', content=content, timestamp=1700000000000)])


def execute(service, op, body, deadline=None):
    if not service.slots.acquire(blocking=False):
        raise APIError(429, 'worker capacity exhausted; retry later')
    return service.execute(op, body, deadline or time.monotonic() + 10, service.slots)


@contextmanager
def running_server(tmp_path, service=None):
    service = service or Service(tmp_path, 'synthetic-token', runner=stub_runner)
    server = Server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:' + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def call(base, path, body=None, headers=None):
    request = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                      headers=headers or {})
    opener = build_opener(ProxyHandler({}))
    try:
        response = opener.open(request, timeout=5)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response), response.headers


@pytest.mark.parametrize('header,value', [('Authorization', 'Token synthetic-token'),
                                         ('Authorization', 'Bearer synthetic-token'),
                                         ('X-Api-Key', 'synthetic-token'),
                                         ('Authorization', 'bearer synthetic-token')])
def test_auth_schemes(header, value):
    headers = Message()
    headers[header] = value
    assert authenticated(headers, 'synthetic-token')


@pytest.mark.parametrize('value', ['', 'Token wrong', 'Basic synthetic-token', 'Token'])
def test_auth_rejects(value):
    headers = Message()
    headers['Authorization'] = value
    assert not authenticated(headers, 'synthetic-token')
    assert not authenticated(headers, '')


def test_durable_replay_is_original_and_no_reingest(tmp_path):
    StubClient.calls = 0
    service = Service(tmp_path, 'test', runner=stub_runner)
    original = execute(service, 'add', addition())
    assert original == dict(success=True, request_id='request-1', user_id='user/a', session_id='session-1')
    # New service = process restart; reused ID returns original even if labels change.
    service = Service(tmp_path, 'test', runner=stub_runner)
    assert execute(service, 'add', addition(session='other', content='different')) == original
    assert StubClient.calls == 1
    with sqlite3.connect(tmp_path / digest('user/a') / 'aml-seen.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM seen').fetchone()[0] == 1
    found = execute(service, 'search', dict(user_id='user/a', query='moonflower'))
    assert len(found['data']) == 1
    assert found['data'][0]['content'] == 'synthetic moonflower'


def test_failure_does_not_publish_and_retry_succeeds(tmp_path):
    service = Service(tmp_path, 'test', runner=stub_runner)
    execute(service, 'add', addition())
    before = (tmp_path / digest('user/a')).resolve()
    StubClient.fail = True
    try:
        with pytest.raises(RuntimeError):
            execute(service, 'add', addition(request='request-2', content='second'))
    finally:
        StubClient.fail = False
    assert (tmp_path / digest('user/a')).resolve() == before
    execute(service, 'add', addition(request='request-2', content='second'))
    result = execute(service, 'search', dict(user_id='user/a', query='anything'))
    assert [hit['content'] for hit in result['data']] == ['synthetic moonflower', 'second']


def test_users_collision_isolation_sessions_are_labels(tmp_path):
    service = Service(tmp_path, 'test', runner=stub_runner)
    for user in ('user/a', 'user?a', 'third'):
        execute(service, 'add', addition(user=user, content=user))
    execute(service, 'add', addition(request='next', session='next', content='another session'))
    for user in ('user/a', 'user?a', 'third'):
        result = execute(service, 'search', dict(user_id=user, query='all', session_id='ignored', space_id='ignored'))
        assert result['data'][0]['content'] == user
        assert len(result['data']) == (2 if user == 'user/a' else 1)


def test_concurrent_replay_single_add(tmp_path):
    service = Service(tmp_path, 'test', workers=4, runner=stub_runner)
    StubClient.calls = 0
    results = []
    threads = [threading.Thread(target=lambda: results.append(execute(service, 'add', addition()))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 4 and all(item == results[0] for item in results)
    assert StubClient.calls == 1


def test_capacity_and_user_lock_deadline(tmp_path):
    service = Service(tmp_path, 'test', workers=1, runner=stub_runner)
    service.slots.acquire()
    try:
        with pytest.raises(APIError) as error:
            execute(service, 'add', addition())
        assert error.value.status == 429
    finally:
        service.slots.release()
    lock = threading.Lock()
    service._locks[digest('user/a')] = lock
    lock.acquire()
    try:
        with pytest.raises(APIError) as error:
            execute(service, 'add', addition(), time.monotonic() + 0.01)
        assert error.value.status == 408
    finally:
        lock.release()


def test_response_shape_rank_and_limits():
    text = '\n'.join(f'hit {index}' for index in range(110))
    assert len(search_response(text, 'u', 500)['data']) == 100
    result = search_response(text, 'u', 2)
    assert len(result['data']) == 2
    assert set(result) == {'data'}
    assert set(result['data'][0]) == {'id', 'content', 'score'}
    assert result['data'][0]['score'] > result['data'][1]['score']
    assert result == search_response(text, 'u', 2)
    assert search_response('', 'u', 100) == {'data': []}
    with pytest.raises(APIError) as error:
        search_response('x' * MAX_BYTES, 'u', 1)
    assert error.value.status == 413


def test_validation_and_timestamp():
    assert validate(addition(), 'add')['messages'][0]['chat_time'] == '2023-11-14T22:13:20+00:00'
    assert validate(dict(user_id='u', query='q', top_k=999), 'search')['top_k'] == 100
    for bad in (0, -1, True, '100', 1.1):
        with pytest.raises(APIError):
            validate(dict(user_id='u', query='q', top_k=bad), 'search')
    for bad in (float('nan'), True, '1700000000000', 1e100):
        body = addition()
        body['messages'][0]['timestamp'] = bad
        with pytest.raises(APIError):
            validate(body, 'add')


def test_http_contract_and_private_logs(tmp_path, capsys):
    with running_server(tmp_path) as base:
        assert call(base, '/aml/health')[:2] == (200, {'status': 'ok'})
        assert call(base, '/aml/add', addition())[0] == 401
        for headers in ({'Authorization': 'Token synthetic-token'},
                        {'Authorization': 'Bearer synthetic-token'},
                        {'X-Api-Key': 'synthetic-token'}):
            assert call(base, '/aml/add', addition(), headers)[0] == 200
        headers = {'X-Api-Key': 'synthetic-token'}
        assert call(base, '/aml/search', dict(user_id='user/a', query='moonflower'), headers)[0] == 200
        status, error, _ = call(base, '/aml/search', dict(user_id='user/a'), headers)
        assert status == 422 and 'reason' in error['detail']
        assert call(base, '/private?token=do-not-log')[0] == 404
    logs = capsys.readouterr().out
    assert all(secret not in logs for secret in ('synthetic-token', 'moonflower', 'user/a', 'do-not-log'))
    receipts = [json.loads(line) for line in logs.splitlines()]
    assert len(receipts) == 8
    assert all(set(row) == {'method', 'path', 'user_hash', 'status', 'ms', 'request_id_hash', 'slow'} for row in receipts)


def test_worker_timeout_kills_process(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    processes = []

    def sleeping_worker(*args, **kwargs):
        process = real_popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, 'Popen', sleeping_worker)
    with pytest.raises(APIError) as error:
        run_worker(tmp_path, 'add', addition(), time.monotonic() + 0.05)
    assert error.value.status == 408
    assert processes[0].poll() is not None


def test_pruning_five_adds_preserves_original_replay(tmp_path):
    service = Service(tmp_path, 'test', runner=stub_runner)
    original = execute(service, 'add', addition())
    for index in range(2, 6):
        execute(service, 'add', addition(request=f'request-{index}', session=f'session-{index}'))
    generations = tmp_path / '.generations' / digest('user/a')
    assert len(list(generations.iterdir())) == 1
    assert list((tmp_path / '.staging').iterdir()) == []
    assert execute(service, 'add', addition(session='changed', content='changed')) == original
    with sqlite3.connect(tmp_path / digest('user/a') / 'aml-seen.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM seen').fetchone()[0] == 5


def test_startup_prunes_staging_and_only_inactive_generations(tmp_path):
    add_generation(tmp_path, validate(addition(), 'add'), StubClient)
    active = (tmp_path / digest('user/a')).resolve()
    orphan = active.parent / 'orphan'
    orphan.mkdir()
    inactive_user = tmp_path / '.generations' / digest('inactive') / 'unpublished'
    inactive_user.mkdir(parents=True)
    stage = tmp_path / '.staging' / 'interrupted'
    stage.mkdir()
    (stage / 'partial').write_text('synthetic')
    prune_generations(tmp_path)
    assert active.is_dir() and list(active.parent.iterdir()) == [active]
    assert not inactive_user.exists()
    assert list((tmp_path / '.staging').iterdir()) == []


def test_stale_base_aborts_without_publish(tmp_path):
    body = validate(addition(), 'add')
    add_generation(tmp_path, body, StubClient)
    active = tmp_path / digest(body['user_id'])
    before = active.resolve()
    replacement = before.parent / 'replacement'
    replacement.mkdir()

    class ChangingClient(StubClient):
        def add(self, *args, **kwargs):
            super().add(*args, **kwargs)
            link = tmp_path / '.test-replacement'
            link.symlink_to(replacement.relative_to(tmp_path))
            os.replace(link, active)

    with pytest.raises(RuntimeError, match='stale base'):
        add_generation(tmp_path, validate(addition(request='next'), 'add'), ChangingClient)
    assert active.resolve() == replacement
    assert before.is_dir()


def test_kernel_user_lock_excludes_another_process(tmp_path):
    script = ('import sys; from pathlib import Path; from aml_server import user_lock; '
              'print("waiting", flush=True); '
              '\nwith user_lock(Path(sys.argv[1]), "synthetic"): print("acquired", flush=True)')
    with user_lock(tmp_path, 'synthetic'):
        process = subprocess.Popen([sys.executable, '-B', '-c', script, str(tmp_path)],
                                   cwd=Path(__file__).resolve().parent,
                                   stdout=subprocess.PIPE, text=True)
        try:
            assert process.stdout.readline().strip() == 'waiting'
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=0.1)
        except BaseException:
            process.kill()
            process.wait()
            raise
    try:
        stdout, _ = process.communicate(timeout=5)
        assert process.returncode == 0 and stdout.strip() == 'acquired'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def raw_post(base, length, body=b'', finish=False):
    with socket.create_connection(('127.0.0.1', int(base.rsplit(':', 1)[1])), timeout=2) as connection:
        headers = ('POST /aml/add HTTP/1.0\r\nX-Api-Key: synthetic-token\r\n'
                   f'Content-Length: {length}\r\n\r\n').encode('latin-1')
        connection.sendall(headers + body)
        if finish:
            connection.shutdown(socket.SHUT_WR)
        response = HTTPResponse(connection)
        response.begin()
        return response.status, json.loads(response.read()), response.headers


def test_content_length_requires_bounded_ascii_digits(tmp_path):
    with running_server(tmp_path) as base:
        for length in ('²', '-1', '+2', '12345678901', '', '1.0'):
            assert raw_post(base, length)[0] == 400


def test_slot_precedes_body_read_and_is_released_on_every_path(tmp_path):
    service = Service(tmp_path, 'synthetic-token', workers=1, runner=stub_runner)
    with running_server(tmp_path, service) as base:
        service.slots.acquire()
        try:
            # No body is sent: capacity rejection must not wait for it.
            status, _, headers = raw_post(base, '100')
            assert status == 429 and headers['Retry-After'] == '60'
        finally:
            service.slots.release()
        for payload, status in ((b'x', 400), (b'{}', 422), (json.dumps(addition()).encode(), 200)):
            assert raw_post(base, str(len(payload)), payload)[0] == status
            assert service.slots.acquire(timeout=1)
            service.slots.release()
        assert raw_post(base, '2', b'{', finish=True)[0] == 408
        assert service.slots.acquire(timeout=1)
        service.slots.release()

        def failing_runner(*args):
            raise RuntimeError('synthetic failure')

        service.runner = failing_runner
        payload = json.dumps(addition()).encode()
        assert raw_post(base, str(len(payload)), payload)[0] == 500
        assert service.slots.acquire(timeout=1)
        service.slots.release()


def test_validate_forwards_only_contract_fields():
    body = dict(addition(), extra='ignored', space_id='ignored')
    assert set(validate(body, 'add')) == {'request_id', 'user_id', 'session_id', 'messages'}
    search = validate(dict(user_id='u', query='q', extra='ignored', session_id='ignored'), 'search')
    assert search == dict(user_id='u', query='q', top_k=100, options=[])
    assert validate(dict(search, options=['choice']), 'search')['options'] == ['choice']


def test_search_response_deduplicates_before_top_k():
    rows = search_response('first\nfirst\n\nsecond\nfirst\nthird', 'u', 2)['data']
    assert [row['content'] for row in rows] == ['first', 'second']
    assert len({row['id'] for row in rows}) == 2
    assert [row['score'] for row in rows] == [1.0, 0.5]


def test_t1_coding_bounds_order_and_ids():
    lines = [f'hit {index}: ' + 'λ' * 5000 for index in range(15)]
    text = '\n'.join([lines[0], '', lines[0], *lines[1:]])
    textual = search_response(text, 'u', 100)['data']
    coding = aml_server.coding_search_response(text, 'u', 100)['data']
    assert len(coding) == 10
    assert all(len(row['content']) <= 4000 for row in coding)
    assert sum(len(row['content']) for row in coding) <= 40000
    assert [row['content'] for row in coding] == [row['content'][:4000] for row in textual[:10]]
    assert [row['id'] for row in coding] == [row['id'] for row in textual[:10]]
    assert [row['score'] for row in coding] == [1.0 / (rank + 1) for rank in range(10)]


def test_t2_coding_single_empty_and_top_k():
    line = 'x' * 40001
    assert aml_server.coding_search_response(line, 'u', 100) == {
        'data': [dict(id=digest('u\0' + line), content=line[:4000], score=1.0)]}
    assert aml_server.coding_search_response('', 'u', 100) == {'data': []}
    rows = aml_server.coding_search_response('a\nb\nc\nd', 'u', 3)['data']
    assert [row['content'] for row in rows] == ['a', 'b', 'c']


def test_t3_textual_literal_unchanged():
    assert search_response('first\nfirst\n\nsecond\nfirst\nthird', 'u', 2) == {'data': [
        {'id': 'ff57647e46f19db31e8aea5b77eea364aa57fd302bf0aef0a555638da5dc26b1',
         'content': 'first', 'score': 1.0},
        {'id': '6de6d86025a19f3f2f0ef9aac4bb76b9ee351da6ab7a7517be24416d3883ae7b',
         'content': 'second', 'score': 0.5}]}


def gateway_environment(tmp_path, mode):
    root = tmp_path / 'db'
    (root / '.model-cache' / 'models--qdrant--bge-small-en-v1.5-onnx-q').mkdir(parents=True)
    factory = tmp_path / 'omni' / 'scripts' / 'client_factory'
    factory.mkdir(parents=True)
    (factory / 'lcmx_client.py').write_text(
        'class LcmxClient:\n'
        '    def search(self, query, user_id, top_k):\n'
        '        assert top_k == 100\n'
        '        return query\n')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, AML_DB_ROOT=str(root), AML_OMNIMEMEVAL_REPO=str(factory.parents[1]),
               AML_ADDSEARCH_TOKEN='synthetic-token', AML_PORT=str(port), PYTHONDONTWRITEBYTECODE='1')
    env.pop('AML_SEARCH_MODE', None)
    if mode is not None:
        env['AML_SEARCH_MODE'] = mode
    return env, 'http://127.0.0.1:' + str(port)


@contextmanager
def running_gateway(tmp_path, mode):
    env, base = gateway_environment(tmp_path, mode)
    with subprocess.Popen([sys.executable, '-B', aml_server.__file__], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            deadline = time.monotonic() + 5
            while True:
                assert process.poll() is None, process.communicate()
                try:
                    assert call(base, '/aml/v1/health')[0] == 200
                    break
                except OSError:
                    assert time.monotonic() < deadline, 'gateway startup timed out'
                    time.sleep(0.01)
            yield base, process
        finally:
            process.terminate()
            process.communicate(timeout=5)


def test_t4_parent_mode_reaches_real_worker(tmp_path):
    text = '\n'.join(f'hit {index}: ' + 'x' * 5000 for index in range(15))
    body = dict(user_id='synthetic-user', query=text, top_k=100)
    for mode in ('coding', None):
        with running_gateway(tmp_path / (mode or 'unset'), mode) as (base, _):
            status, response, _ = call(base, '/aml/v1/search', body, {'X-Api-Key': 'synthetic-token'})
            assert status == 200
            expected = search_response(text, digest(body['user_id']), 100)
            if mode == 'coding':
                expected['data'] = [dict(row, content=row['content'][:4000]) for row in expected['data'][:10]]
            assert response == expected


def test_t5_invalid_mode_and_textual_defaults(tmp_path):
    invalid = 'invalid-synthetic-value'
    env, _ = gateway_environment(tmp_path / 'invalid', invalid)
    result = subprocess.run([sys.executable, '-B', aml_server.__file__], env=env,
                            capture_output=True, text=True, timeout=2)
    assert result.returncode != 0
    assert 'AML_SEARCH_MODE must be textual or coding' in result.stderr
    assert 'Traceback' not in result.stderr and invalid not in result.stderr
    for mode in (None, ''):
        with running_gateway(tmp_path / ('unset' if mode is None else 'empty'), mode) as (base, _):
            text = 'x' * 4001
            status, response, _ = call(base, '/aml/v1/search', dict(user_id='u', query=text),
                                       {'X-Api-Key': 'synthetic-token'})
            assert status == 200 and response == search_response(text, digest('u'), 100)


def test_t6_startup_receipt_mode_and_privacy(tmp_path):
    for mode in ('coding', 'textual'):
        with running_gateway(tmp_path / mode, mode) as (_, process):
            line = process.stdout.readline()
            assert json.loads(line) == {'event': 'start', 'search_mode': mode}
            assert all(value not in line for value in ('synthetic-token', str(tmp_path)))
