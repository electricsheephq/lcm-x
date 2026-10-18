#!/usr/bin/env python3
"""Loopback-only AML gateway; retrieval/ingestion are the OmniMemEval adapter's.

Each operation runs in a killable subprocess. Adds publish a complete generation
(LCM-X data plus replay ledger) atomically; interrupted generations stay private.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

MAX_BYTES = 30 * 1024 * 1024
MAX_SECONDS = 1800


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


class APIError(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason


def authenticated(headers, token):
    """Always use constant-time comparison, including the missing-key case."""
    candidates = list(headers.get_all('X-Api-Key', []))
    for authorization in headers.get_all('Authorization', []):
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ('token', 'bearer'):
            candidates.append(parts[1])
    matched = False
    expected = token.encode('utf-8')
    for candidate in candidates or ['']:
        matched |= hmac.compare_digest(candidate.encode('utf-8'), expected)
    return bool(token) and matched


def validate(body, operation):
    if not isinstance(body, dict):
        raise APIError(422, 'body must be a JSON object')
    fields = ['user_id', 'request_id', 'session_id'] if operation == 'add' else ['user_id', 'query']
    for field in fields:
        if not isinstance(body.get(field), str) or not body[field].strip():
            raise APIError(422, field + ' must be a non-empty string')
    if operation == 'add':
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise APIError(422, 'messages must be a non-empty array')
        rows = []
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get('role') not in ('user', 'assistant')
                    or not isinstance(message.get('content'), str)):
                raise APIError(422, 'messages require role user|assistant and string content')
            row = {'role': message['role'], 'content': message['content']}
            if 'timestamp' in message:
                stamp = message['timestamp']
                if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                    raise APIError(422, 'timestamp must be finite Unix milliseconds')
                try:
                    row['chat_time'] = datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat()
                except (ValueError, OverflowError, OSError):
                    raise APIError(422, 'timestamp is out of range') from None
            rows.append(row)
        return dict(request_id=body['request_id'], user_id=body['user_id'],
                    session_id=body['session_id'], messages=rows)
    top_k = body.get('top_k', 100)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise APIError(422, 'top_k must be a positive integer')
    options = body.get('options', [])
    if not isinstance(options, list) or any(not isinstance(item, str) for item in options):
        raise APIError(422, 'options must be an array of strings')
    return dict(user_id=body['user_id'], query=body['query'],
                top_k=min(top_k, 100), options=options)


def search_response(text, user_key, top_k):
    # The adapter collapses stored newlines and emits one line per ranked hit.
    # It does not expose original scores/IDs: provide stable content IDs and
    # monotone rank scores without modifying its retrieval/reranking behavior.
    lines = list(dict.fromkeys(line for line in text.splitlines() if line.strip()))[:min(top_k, 100)]
    result = {'data': [dict(id=digest(user_key + '\0' + line), content=line,
                            score=1.0 / (rank + 1)) for rank, line in enumerate(lines)]}
    if len(encoded(result)) > MAX_BYTES:
        raise APIError(413, 'search response exceeds 30 MiB')
    return result


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(path):
    for folder, _, names in os.walk(path, topdown=False):
        for name in names:
            with open(Path(folder) / name, 'rb') as handle:
                os.fsync(handle.fileno())
        fsync_dir(folder)


def ledger_response(directory, request_key):
    ledger = directory / 'aml-seen.sqlite3'
    if not ledger.exists():
        return None
    with contextlib.closing(sqlite3.connect(f'{ledger.as_uri()}?mode=ro', uri=True)) as db:
        row = db.execute('SELECT response FROM seen WHERE request_id=?', (request_key,)).fetchone()
    return json.loads(row[0]) if row else None


@contextlib.contextmanager
def user_lock(root, user_key):
    locks = root / '.locks'
    locks.mkdir(exist_ok=True)
    with open(locks / (user_key + '.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def prune_generations(root):
    # Workers create their lock files before staging. Wait for existing orphans
    # before cleaning their private work or inspecting active generations.
    with contextlib.ExitStack() as held:
        for lock in sorted((root / '.locks').glob('*.lock')):
            held.enter_context(user_lock(root, lock.stem))
        for stage in (root / '.staging').glob('*'):
            if stage.is_dir() and not stage.is_symlink():
                shutil.rmtree(stage)
            else:
                stage.unlink()
        for generations in (root / '.generations').glob('*'):
            active = root / generations.name
            base = os.readlink(active) if active.is_symlink() else None
            for generation in generations.iterdir():
                if str(generation.relative_to(root)) != base:
                    shutil.rmtree(generation)


def add_generation(root, body, client_factory):
    """Hold kernel exclusion even if the parent dies while this worker runs."""
    with user_lock(root, digest(body['user_id'])):
        return _add_generation(root, body, client_factory)


def _add_generation(root, body, client_factory):
    user_key, request_key = digest(body['user_id']), digest(body['request_id'])
    active = root / user_key
    base = os.readlink(active) if active.is_symlink() else None
    previous = ledger_response(active, request_key)
    if previous is not None:
        return previous
    stage_root = root / '.staging' / uuid.uuid4().hex
    stage = stage_root / user_key
    stage_root.mkdir(parents=True)
    if active.exists():
        shutil.copytree(active, stage)
    else:
        stage.mkdir()
    client = client_factory(stage_root)
    # Like benchmark users: one store per user; each add is a session. Labels
    # never restrict search. Opaque hashes also avoid _safe() path collisions.
    client.add(body['messages'], user_key, session_id=digest(body['session_id']))
    response = {key: body[key] for key in ('request_id', 'user_id', 'session_id')}
    response = dict(success=True, **response)
    with contextlib.closing(sqlite3.connect(stage / 'aml-seen.sqlite3')) as db, db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS seen (request_id TEXT PRIMARY KEY, response TEXT NOT NULL)')
        db.execute('INSERT INTO seen VALUES (?, ?)', (request_key, encoded(response).decode('utf-8')))
    fsync_tree(stage)
    generations = root / '.generations' / user_key
    generations.mkdir(parents=True, exist_ok=True)
    fsync_dir(generations.parent)
    destination = generations / stage_root.name
    os.rename(stage, destination)
    fsync_dir(generations)
    fsync_dir(stage_root)
    link = root / ('.publish-' + stage_root.name)
    if (os.readlink(active) if active.is_symlink() else None) != base:
        raise RuntimeError('stale base')
    link.symlink_to(destination.relative_to(root), target_is_directory=True)
    os.replace(link, active)
    fsync_dir(root)
    os.rmdir(stage_root)
    if base:
        shutil.rmtree(root / base, ignore_errors=True)
    return response


def local_network_only():
    """Prevent accidental model downloads/telemetry, even from dependencies."""
    def audit(event, args):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind'):
            address = args[1]
            if not isinstance(address, tuple) or address[0] != '127.0.0.1':
                raise PermissionError('only 127.0.0.1 network access is permitted')
        elif event == 'socket.getaddrinfo' and args[0] != '127.0.0.1':
            raise PermissionError('external DNS is disabled')
        elif event in ('socket.sendto', 'socket.sendmsg'):
            raise PermissionError('datagram network access is disabled')
    sys.addaudithook(audit)


def configure_adapter(root):
    # Pin board settings regardless of inherited shell configuration.
    os.environ.update(LCMX_DB_ROOT=str(root), LCMX_CONFIG='fleet',
                      LCMX_EMBED_PROVIDER='fastembed', LCMX_EMBED_MODEL='BAAI/bge-small-en-v1.5',
                      LCMX_CHUNK_POLICY='full', LCMX_DELIVERY='full', LCMX_TOPK_CAP='100',
                      HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    for variable in ('LCMX_EMBED_CACHE', 'LCM_LONGMEMEVAL_EMBED_CACHE', 'LCMX_SEARCH_SCOPE_BIAS'):
        os.environ.pop(variable, None)
    sys.dont_write_bytecode = True
    omni_repo = os.environ.get('AML_OMNIMEMEVAL_REPO', '')
    if not omni_repo:
        raise RuntimeError('AML_OMNIMEMEVAL_REPO is required')
    OMNI_REPO = Path(omni_repo)
    if not OMNI_REPO.is_dir():
        raise RuntimeError('AML_OMNIMEMEVAL_REPO must be a directory')
    sys.path.insert(0, str(OMNI_REPO / 'scripts'))
    from client_factory.lcmx_client import LcmxClient
    return LcmxClient()


def prepare_model_cache(root):
    """Keep dependency cache writes inside AML_DB_ROOT; never download a model."""
    cache = root / '.model-cache'
    model = 'models--qdrant--bge-small-en-v1.5-onnx-q'
    if not (cache / model).is_dir():
        source = Path(os.environ.get('AML_EMBED_CACHE_SOURCE', str(Path.home() / '.cache/fastembed')))
        if not (source / model).is_dir():
            raise RuntimeError('offline BGE-small cache missing; downloads are forbidden')
        # Atomic directory publication also makes interrupted startup retryable.
        staging = root / ('.model-copy-' + uuid.uuid4().hex)
        shutil.copytree(source / model, staging / model, symlinks=False)
        os.rename(staging, cache)
    os.environ['LCM_LONGMEMEVAL_FASTEMBED_CACHE'] = str(cache)
    os.environ['FASTEMBED_CACHE_PATH'] = str(cache)


def worker():
    local_network_only()
    payload = json.load(sys.stdin)
    root, body = Path(payload['root']), payload['body']
    # Suppress third-party prints (which can include user identifiers/content).
    output = os.dup(sys.stdout.fileno())
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            if payload['operation'] == 'add':
                result = add_generation(root, body, configure_adapter)
            else:
                user_key = digest(body['user_id'])
                with user_lock(root, user_key):
                    client = configure_adapter(root)
                    result = search_response(client.search(body['query'], user_key, body['top_k']),
                                             user_key, body['top_k'])
            reply = {'status': 200, 'body': result}
        except APIError as error:
            reply = {'status': error.status, 'body': {'detail': {'reason': error.reason}}}
        except Exception:
            reply = {'status': 500, 'body': {'detail': {'reason': 'LCM-X operation failed'}}}
    with os.fdopen(output, 'wb') as handle:
        handle.write(encoded(reply))


def run_worker(root, operation, body, deadline):
    env = dict(os.environ)
    env.pop('AML_ADDSEARCH_TOKEN', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise APIError(408, 'request deadline exceeded')
    with subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          env=env) as process:
        try:
            stdout, _ = process.communicate(encoded(dict(root=str(root), operation=operation, body=body)),
                                            timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise APIError(408, 'request deadline exceeded') from None
    if process.returncode:
        raise APIError(500, 'LCM-X worker failed')
    reply = json.loads(stdout)
    if reply['status'] != 200:
        raise APIError(reply['status'], reply['body']['detail']['reason'])
    return reply['body']


class Service:
    def __init__(self, root, token, workers=4, timeout=MAX_SECONDS, runner=run_worker):
        if not token or workers < 1 or not 0 < timeout <= MAX_SECONDS:
            raise ValueError('token, positive worker count and timeout <= 1800 required')
        self.root, self.token = Path(root).resolve(), token
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout, self.runner = timeout, runner
        self.slots = threading.BoundedSemaphore(workers)
        self._locks, self._guard = {}, threading.Lock()

    def execute(self, operation, body, deadline, slot):
        # Ownership of the already-acquired request slot transfers here.
        try:
            body = validate(body, operation)
            with self._guard:
                lock = self._locks.setdefault(digest(body['user_id']), threading.Lock())
            if not lock.acquire(timeout=max(0, deadline - time.monotonic())):
                raise APIError(408, 'request deadline exceeded')
            try:
                return self.runner(self.root, operation, body, deadline)
            finally:
                lock.release()
        finally:
            slot.release()


class Handler(BaseHTTPRequestHandler):
    # One request per connection avoids keepalive/body framing ambiguities.
    protocol_version = 'HTTP/1.0'
    server_version = 'aml'
    sys_version = ''

    def log_message(self, *_args):
        pass

    def handle_one_request(self):
        self.started = time.monotonic()
        self.deadline = self.started + self.server.service.timeout
        self.logged, self.user_hash, self.request_hash = False, None, None
        timer = threading.Timer(self.server.service.timeout, self.expire)
        timer.daemon = True
        timer.start()
        self.connection.settimeout(min(30, self.server.service.timeout))
        try:
            super().handle_one_request()
        finally:
            timer.cancel()
            if not self.logged and getattr(self, 'raw_requestline', b''):
                self.receipt(408)

    def expire(self):
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)

    def receipt(self, status):
        if self.logged:
            return
        self.logged = True
        ms = round((time.monotonic() - self.started) * 1000, 3)
        path = getattr(self, 'path', '')
        path = path if path in ('/aml/health', '/aml/add', '/aml/search') else '<unknown>'
        method = getattr(self, 'command', '')
        method = method if method in ('GET', 'POST', 'HEAD', 'PUT', 'DELETE', 'OPTIONS', 'PATCH') else '<unknown>'
        print(json.dumps(dict(method=method, path=path, user_hash=self.user_hash,
                              status=status, ms=ms, request_id_hash=self.request_hash,
                              slow=ms >= self.server.slow_seconds * 1000)), flush=True)

    def respond(self, status, body):
        data = encoded(body)
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Connection', 'close')
            if status == 429:
                self.send_header('Retry-After', '60')
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            self.receipt(status)

    def send_error(self, code, message=None, explain=None):
        self.respond(code, {'detail': {'reason': 'invalid HTTP request'}})

    # AML's v0.27.0 binding uses /aml/v1/<op>; accept it as an alias of /aml/<op>.
    V1_ALIASES = {'/aml/v1/health': '/aml/health', '/aml/v1/add': '/aml/add', '/aml/v1/search': '/aml/search'}

    def do_GET(self):
        self.path = self.V1_ALIASES.get(self.path, self.path)
        if self.path == '/aml/health':
            self.respond(200, {'status': 'ok'})
        else:
            self.respond(404, {'detail': {'reason': 'unknown endpoint'}})

    def do_POST(self):
        self.path = self.V1_ALIASES.get(self.path, self.path)
        slot = None
        try:
            if self.path not in ('/aml/add', '/aml/search'):
                raise APIError(404, 'unknown endpoint')
            if not authenticated(self.headers, self.server.service.token):
                raise APIError(401, 'authentication required')
            lengths = self.headers.get_all('Content-Length', [])
            if (self.headers.get('Transfer-Encoding') or len(lengths) != 1
                    or not re.fullmatch(r'[0-9]{1,10}', lengths[0])):
                raise APIError(400, 'one Content-Length required; transfer encoding unsupported')
            length = int(lengths[0])
            if length > MAX_BYTES:
                raise APIError(413, 'request exceeds 30 MiB')
            if not self.server.service.slots.acquire(blocking=False):
                raise APIError(429, 'worker capacity exhausted; retry later')
            slot = self.server.service.slots
            self.connection.settimeout(max(0.001, self.deadline - time.monotonic()))
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise APIError(408, 'incomplete request body or deadline exceeded')
            try:
                body = json.loads(raw, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
            except (ValueError, UnicodeError, RecursionError):
                raise APIError(400, 'invalid JSON') from None
            if isinstance(body, dict):
                self.user_hash = digest(body['user_id']) if isinstance(body.get('user_id'), str) else None
                self.request_hash = digest(body['request_id']) if isinstance(body.get('request_id'), str) else None
            reserved_slot, slot = slot, None
            response = self.server.service.execute(self.path.rsplit('/', 1)[1], body,
                                                   self.deadline, reserved_slot)
            self.respond(200, response)
        except APIError as error:
            self.respond(error.status, {'detail': {'reason': error.reason}})
        except (TimeoutError, OSError):
            self.respond(408, {'detail': {'reason': 'request interrupted or deadline exceeded'}})
        except Exception:
            self.respond(500, {'detail': {'reason': 'internal service error'}})
        finally:
            if slot is not None:
                slot.release()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, service, port=8791, slow_seconds=10):
        self.service, self.slow_seconds = service, slow_seconds
        super().__init__(('127.0.0.1', port), Handler)

    def handle_error(self, request, client_address):
        # Never print request exception tracebacks or raw addresses/bodies.
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    local_network_only()
    os.umask(0o077)
    token = os.environ.get('AML_ADDSEARCH_TOKEN', '')
    root = Path(os.environ['AML_DB_ROOT']).expanduser().resolve()
    service = Service(root, token, int(os.environ.get('AML_WORKERS', '4')),
                      float(os.environ.get('AML_REQUEST_TIMEOUT_SECONDS', '1800')))
    # One server per database root; worker file locks also survive parent exit.
    with open(root / '.server.lock', 'a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prune_generations(root)
        prepare_model_cache(root)
        with Server(service, int(os.environ.get('AML_PORT', '8791')),
                    float(os.environ.get('AML_SLOW_SECONDS', '10'))) as server:
            server.serve_forever()


if __name__ == '__main__':
    main()
