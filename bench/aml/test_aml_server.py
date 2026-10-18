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
