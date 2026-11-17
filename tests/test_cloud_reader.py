"""Synthetic split-corpus proofs for the separate-process cloud reader."""
from __future__ import annotations

import ast
import fcntl
import io
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from hermes_lcm import claude_code_store as cc
from hermes_lcm import cloud_reader as cr
from hermes_lcm import cloud_view as cv
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from tests.test_claude_code_store import _entry, _transcript

ROOT = Path(__file__).resolve().parents[1]
COPIES = [str(uuid.UUID(int=n)) for n in (10, 11, 12)]
OWN = str(uuid.UUID(int=99))
PREFIX = COPIES[0].replace('-', '')
TOOLS = ['lcm_grep', 'lcm_recall', 'lcm_expand', 'lcm_load_session', 'lcm_describe']
BAD = re.compile(r'\b(?:store_id|node_id|after_store_id|anchor_store_id)=\d|(?<![0-9a-f]:)\blcm:\d+:|(?<![0-9a-f]:)\blcm-summary:\d')

# The wrapper tees the reader's actual protocol stream, records child pids and
# refuses network connections, including from background model preparation.
WRAPPER = r'''
import json, os, pathlib, runpy, sys, time
sys.path[:] = json.loads(os.environ['TEST_READER_PATH'])
log = pathlib.Path(sys.argv[sys.argv.index('--log-dir') + 1])
log.mkdir(parents=True, exist_ok=True)
home = pathlib.Path(sys.argv[sys.argv.index('--hermes-home') + 1])
label = home.parent.name + '-' + home.name
pid = os.getpid()
(log / (str(pid) + '.pid')).write_text(str(home))
def audit(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo'):
        with open(log / 'connections', 'a') as f: f.write(event + '\n')
        raise OSError('network refused by fixture')
sys.addaudithook(audit)
control = log / 'control.json'
def settings():
    return json.loads(control.read_text()) if control.exists() else {}
def record(request):
    with open(log / (str(pid) + '.requests'), 'a') as f: f.write(json.dumps(request) + '\n')
if os.environ.get('TEST_FAKE_READER'):
    for line in sys.stdin:
        request = json.loads(line)
        record(request)
        if 'id' not in request: continue
        args = request.get('params', {}).get('arguments', {})
        replica = home.name.startswith('view-')
        cfg = settings()
        if replica and cfg.get('stall') and request['method'] == 'tools/call':
            time.sleep(60)
        if request['method'] == 'initialize': result = {'serverInfo': {'name': 'fake'}}
        elif request['method'] == 'tools/list':
            result = {'tools': [{'name': name} for name in ('lcm_grep', 'lcm_recall', 'lcm_expand', 'lcm_load_session', 'lcm_describe')]}
        elif request['method'] == 'tools/call':
            hit = {'store_id': 1, 'session_id': 'remote', 'snippet': 'remote marker',
                   'unknown_handle': 'store_id=999', 'content': 'verbatim',
                   'transcript_content': 'verbatim store_id=3 lcm:3:0-5'}
            if cfg.get('missing'): hit.pop('session_id'); hit['store_id'] = 999999
            if cfg.get('empty'): hit['session_id'] = ''; hit['store_id'] = 999999
            body = {'store_uuid': 'f'*32, 'reader': {}, 'results': [hit], 'hits': [hit]}
            result = {'content': [{'type': 'text', 'text': json.dumps(body)}], 'isError': False}
        else: result = {}
        reply = json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}) + '\n'
        with open(log / (str(pid) + '.replies'), 'a') as f: f.write(reply)
        sys.stdout.write(reply); sys.stdout.flush()
else:
    runpy.run_path(os.environ['TEST_STOCK_SCRIPT'], run_name='fixture_bootstrap')
    from hermes_lcm import mcp_reader
    original = mcp_reader._isolate_stdio
    class Tee:
        def __init__(self, stream): self.stream = stream
        def write(self, value):
            with open(log / (str(pid) + '.replies'), 'a') as f: f.write(value)
            return self.stream.write(value)
        def flush(self): self.stream.flush()
    def isolate(path): return Tee(original(path))
    mcp_reader._isolate_stdio = isolate
    handle = mcp_reader.handle
    def traced(reader, request): record(request); return handle(reader, request)
    mcp_reader.handle = traced
    raise SystemExit(mcp_reader.main(sys.argv[1:]))
'''


MERGER_AUDIT_WRAPPER = r'''
import os, pathlib, runpy, sys
log = pathlib.Path(os.environ['TEST_MERGER_AUDIT_LOG'])
log.mkdir(parents=True, exist_ok=True)
def audit(event, args):
    if event in ('socket.__new__', 'socket.connect', 'socket.connect_ex', 'socket.getaddrinfo'):
        with open(log / 'connections', 'a') as f: f.write(event + '\n')
        raise OSError('socket refused by fixture')
sys.addaudithook(audit)
(log / 'merger-audit').write_text('installed\n')
runpy.run_path(sys.argv.pop(1), run_name='__main__')
'''


def make_store(base, session, text=None):
    home = base / session
    config = LCMConfig(database_path=str(home / 'lcm.db'), embeddings_enabled=False,
                       large_output_externalization_enabled=True,
                       large_output_externalization_threshold_chars=200)
    db = cc.admit(home, hermes_home=base / 'unused-hermes')
    store = cc.open_store(db, home, config)
    payload = 'payload marker ' + ('amber ' * 80)
    entries = [_entry('user', 'u1', text or f'{session} marker store_id=3 lcm:3:0-5', session),
               _entry('assistant', 'a1', [{'type': 'tool_use', 'id': 'tool-1', 'name': 'Read', 'input': {}}], session),
               _entry('user', 'r1', [{'type': 'tool_result', 'tool_use_id': 'tool-1', 'content': payload}], session),
               _entry('assistant', 'a2', f'{session} marker final', session)]
    transcript = _transcript(base / session / 'fixture', entries, session)
    assert cc.drain(store, cc.Lineage(session, None, transcript)) == 'drained'
    dag = SummaryDAG(db)
    leaf = dag.add_node(SummaryNode(session_id=session, summary=f'{session} marker summary',
                                   source_ids=[1, 2, 3, 4], source_type='messages', token_count=12,
                                   source_token_count=80, expand_hint='lcm_expand(node_id=1)',
                                   earliest_at=1_700_000_000, latest_at=1_700_000_001),
                        escalation_level=1, model='synthetic')
    dag.add_node(SummaryNode(session_id=session, depth=1, summary=f'{session} marker parent',
                             source_ids=[leaf], source_type='nodes', token_count=10,
                             source_token_count=80, expand_hint='lcm_expand(node_id=2)'))
    dag.close()
    assert cc.to_rollback_journal(store.connection)
    store.close()
    assert cc.leave_readable(db)
    assert db.stat().st_size < 5_000_000
    return home, payload


def optin(home):
    path = home / 'cloud' / 'cloud.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'copy_id': OWN}))
    path.chmod(0o600)


def copy_generation(source, local, copy_id, generation=1):
    folder = local / 'cloud' / 'replicas' / copy_id / f'gen-{generation}'
    folder.mkdir(parents=True)
    shutil.copy2(source / 'lcm.db', folder / 'lcm.db')
    payloads = source / 'lcm-large-outputs'
    if payloads.exists():
        shutil.copytree(payloads, folder / 'lcm-large-outputs')
    (folder.parent / 'current').write_text(str(generation) + '\n')
    cv.build(local, copy_id)
    return folder


def forgotten_ids(home):
    # During the red run the base still returns only the set.
    result = cr.forgotten(home / 'lcm.db')
    return result[0] if isinstance(result, tuple) else result


def view_folder(generation):
    pointer = json.loads((generation.parent / 'view.json').read_text())
    return generation.parent / pointer['folder']


def assert_view_readers(client):
    paths = [Path((client.log / f'{pid}.pid').read_text()) for pid in client.pids(True)]
    assert paths and all(path.name.startswith('view-') for path in paths)
    assert not any(Path(p.read_text()).name.startswith('gen-') for p in client.log.glob('*.pid'))


class Client:
    def __init__(self, base, local, *, fake=False, wait=30, budget=8, stock=False, missing=False, audit=False):
        self.local = local
        self.log = base / ('plain-logs' if stock else 'logs')
        wrapper = base / 'reader-wrapper.py'
        wrapper.write_text(WRAPPER)
        env = {k: v for k, v in os.environ.items() if not k.startswith('LCM_') and k != 'PYTHONPATH'}
        env.update(HF_HUB_OFFLINE='1', TEST_READER_PATH=json.dumps(sys.path),
                   TEST_STOCK_SCRIPT=str(ROOT / 'scripts' / 'lcm_x_mcp.py'), PYTHONDONTWRITEBYTECODE='1')
        if fake:
            env['TEST_FAKE_READER'] = '1'
        argv = [sys.executable, str(wrapper), '--store', str(local / 'lcm.db'),
                '--hermes-home', str(local), '--log-dir', str(self.log)]
        command = argv if stock else [sys.executable, '-I', '-S', str(ROOT / 'cloud_reader.py'),
                                     '--replica-wait-s', str(wait), '--budget-s', str(budget)]
        if audit and not stock:
            auditor = base / 'merger-audit-wrapper.py'
            auditor.write_text(MERGER_AUDIT_WRAPPER)
            env['TEST_MERGER_AUDIT_LOG'] = str(self.log)
            command.insert(3, str(auditor))
        if not missing:
            env['LCMX_LOCAL_READER_CMD'] = json.dumps(argv)
        self.before_pids = {p.stem for p in self.log.glob('*.pid')}
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, text=True, cwd=base)
        self.ident = 0

    def request(self, method, params=None):
        self.ident += 1
        req = {'jsonrpc': '2.0', 'id': self.ident, 'method': method}
        if params is not None:
            req['params'] = params
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            assert selector.select(45), 'merger did not answer'
        line = self.proc.stdout.readline()
        assert line, f'merger exited: {self.proc.poll()}'
        reply = json.loads(line)
        assert reply['id'] == self.ident
        return reply

    def initialize(self):
        assert 'serverInfo' in self.request('initialize', {'protocolVersion': '2025-06-18'})['result']

    def call(self, name='lcm_grep', **arguments):
        return self.request('tools/call', {'name': name, 'arguments': arguments})

    def body(self, name='lcm_grep', **arguments):
        return json.loads(self.call(name, **arguments)['result']['content'][0]['text'])

    def pids(self, replica=None):
        return [int(p.stem) for p in self.log.glob('*.pid')
                if p.stem not in self.before_pids
                and (replica is None or Path(p.read_text()).name.startswith('view-') == replica)]

    def local_reply(self):
        pids = self.pids(False)
        assert pids
        replies = [json.loads(line) for pid in pids
                   for line in (self.log / f'{pid}.replies').read_text().splitlines()]
        return next(reply for reply in reversed(replies) if reply['id'] == self.ident)

    def control(self, **values):
        self.log.mkdir(exist_ok=True)
        (self.log / 'control.json').write_text(json.dumps(values))

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc.stdout.close()
        self.proc.stderr.close()


@pytest.fixture
def corpus(tmp_path):
    local, _ = make_store(tmp_path, 'local')
    (local / '.env').write_text('LCM_EMBEDDINGS_ENABLED=false\n')
    optin(local)
    sources = [make_store(tmp_path, f'remote{i}')[0] for i in range(3)]
    return local, sources


@pytest.fixture
def clients():
    opened = []
    def start(base, local, **kwargs):
        client = Client(base, local, **kwargs)
        opened.append(client)
        client.initialize()
        return client
    yield start
    for client in opened:
        client.close()


def ready(client, count=1):
    deadline = time.monotonic() + 40
    while True:
        body = client.body(query='marker')
        entries = body.get('other_stores', [])
        assert len(entries) == count, body
        if all(e['status'] == 'ok' for e in entries):
            return body
        assert time.monotonic() < deadline, entries
        time.sleep(0.05)


def assert_local(client, reply):
    text = reply['result']['content'][0]['text']
    original = client.local_reply()['result']
    assert cr.strip_other_stores(text) == original['content'][0]['text']
    assert reply['result']['isError'] == original['isError']


def qualified(value, prefix=PREFIX):
    return f'{prefix}:{value}'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def eventually(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_01_local_bytes(tmp_path, corpus, clients):
    local, sources = corpus
    for count in (0, 1, 3):
        for i in range(count):
            if not (local / 'cloud' / 'replicas' / COPIES[i]).exists():
                copy_generation(sources[i], local, COPIES[i])
        client = clients(tmp_path, local)
        if count:
            ready(client, count)
        for tool, args in [('lcm_grep', {'query': 'marker'}), ('lcm_recall', {'query': 'marker'}),
                           ('lcm_expand', {'store_id': 1}), ('lcm_load_session', {'session_id': 'local'}),
                           ('lcm_describe', {'session_id': 'local'})]:
            assert_local(client, client.call(tool, **args))
        assert_local(client, client.request('tools/call', {'name': 'lcm_grep', 'arguments': ['invalid']}))
        assert_local(client, client.request('tools/call', 'invalid'))
        client.close()
    assert cr.strip_other_stores('{"other_stores": [], "x":  1}') == '{"x":  1}'
    assert cr.strip_other_stores('{"other_stores": []}') == '{}'
    assert cr.strip_other_stores('{"other_stores": [{"hits": [{"content": "} ,"}]}], "a":1}') == '{"a":1}'


def test_02_replica_only_value(tmp_path, corpus, clients):
    local, sources = corpus
    generation = copy_generation(sources[0], local, COPIES[0])
    assert not (generation / '.env').exists()
    plain = clients(tmp_path, generation, stock=True)
    assert plain.body('lcm_expand', store_id=1)['content'].startswith('remote0 marker')
    assert plain.body(query='marker')['results']
    assert not (plain.log / 'connections').exists()
    plain.close()
    client = clients(tmp_path, local)
    body = ready(client)
    hits = body['other_stores'][0]['hits']
    hit = next(h for h in hits if h.get('store_id') == qualified(1))
    assert 'remote0' in hit['snippet'] and 'marker' in hit['snippet']
    expanded = client.body('lcm_expand', store_id=hit['store_id'])
    assert expanded['content'] == 'remote0 marker store_id=3 lcm:3:0-5'
    assert not (client.log / 'connections').exists()


def test_03_wrong_route_visible(tmp_path, corpus, clients):
    local, sources = corpus
    for i in range(2):
        copy_generation(sources[i], local, COPIES[i])
    client = clients(tmp_path, local)
    ready(client, 2)
    for prefix, name in [(None, 'local'), (PREFIX, 'remote0'), (COPIES[1].replace('-', ''), 'remote1')]:
        body = client.body('lcm_expand', store_id=qualified(1, prefix) if prefix else 1)
        assert body['content'] == f'{name} marker store_id=3 lcm:3:0-5'


def scan(value, stored=False):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in cr.STORED:
                continue
            if key in cr.INTEGER_IDS | cr.STRING_IDS:
                assert isinstance(item, str) and re.fullmatch('[0-9a-f]{32}:.*', item)
            yield from scan(item)
    elif isinstance(value, list):
        for item in value:
            yield from scan(item)
    elif isinstance(value, str):
        assert not BAD.search(value), value
        yield value


def test_04_no_bare_id_leaves(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    body = ready(client)['other_stores']
    list(scan(body))
    for tool, args in [('lcm_expand', {'store_id': qualified(1), 'include_exact_ref': True}),
                       ('lcm_load_session', {'session_id': qualified('remote0'), 'limit': 1}),
                       ('lcm_describe', {'session_id': qualified('remote0')})]:
        result = client.body(tool, **args)
        strings = list(scan(result))
        assert all('f' * 32 not in s for s in strings)
    synthetic = {'store_id': True, 'node_id': '3', 'session_id': 4, 'next_cursor': None,
                 'lineage': 'subagent of parent', 'degraded_reason': 'store_id=3',
                 'returned_id_ranges': [[1, 2]], 'duplicate_store_ids': [1, 3],
                 'exact_ref': 'lcm-summary:3', 'novel_refs': ['lcm:1:0-5', 'unsafe'],
                 'expand_hint': 'lcm_expand(node_id=3, session_id="remote0")',
                 'session_hint': '__import__("os").system("no")'}
    rebuilt = cr.rebuild(synthetic, PREFIX)
    assert rebuilt['lineage'] == f'subagent of {PREFIX}:parent'
    assert rebuilt['returned_id_ranges'] == [[qualified(1), qualified(2)]]
    assert rebuilt['duplicate_store_ids'] == [qualified(1), qualified(3)]
    assert rebuilt['novel_refs'] == [qualified('lcm:1:0-5')]
    assert not {'store_id', 'node_id', 'session_id', 'next_cursor', 'degraded_reason', 'session_hint'} & rebuilt.keys()
    list(scan(rebuilt))
    assert cr.rebuild({'arms': ['fts'], 'omission_reason': None}, PREFIX) == {'arms': ['fts'], 'omission_reason': None}
    assert cr.error_body('Message store_id 3 not found', {'store_id': 3}, PREFIX) == {
        'error': f'Message store_id {PREFIX}:3 not found'}
    assert cr.replace_token('13 name3 ' + qualified(3) + ' 3', 3, qualified(3)) == (
        '13 name3 ' + qualified(3) + ' ' + qualified(3))
    for text in ('bad store_id=4', 'foreign ' + 'e' * 32):
        assert cr.error_body(text, {'store_id': 3}, PREFIX)['error'] == 'the other store could not answer this call'


def test_05_stored_text_verbatim(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    with sqlite3.connect(sources[0] / 'lcm.db') as conn:
        row = conn.execute("SELECT store_id, content FROM messages WHERE role='tool'").fetchone()
    assert row
    expanded = client.body('lcm_expand', store_id=qualified(row[0]))
    assert expanded['content'] == row[1]
    ref = expanded['externalized_refs'][0]
    assert ref.startswith(PREFIX + ':')
    payload = client.body('lcm_expand', externalized_ref=ref)
    assert payload['content'] == 'payload marker ' + 'amber ' * 80
    text = 'store_id=3 lcm:3:0-5 [Externalized LCM ingest payload: saved; ref=demo.json]'
    assert cr.rebuild({'content': text}, PREFIX) == {'content': text, 'externalized_refs': [qualified('demo.json')]}
    opaque = {'content': {'store_id': 3}, 'tool_calls': [{'arguments': text}], 'summary': [text]}
    rebuilt = cr.rebuild(opaque, PREFIX)
    assert all(rebuilt[key] == value for key, value in opaque.items())


def test_06_every_handle_works(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    hits = ready(client)['other_stores'][0]['hits']
    for hit in hits:
        if 'store_id' in hit:
            assert client.body('lcm_expand', store_id=hit['store_id'])['store'] == PREFIX
        if 'ref' in hit:
            assert client.body('lcm_expand', externalized_ref=hit['ref'])['store'] == PREFIX
        if 'session_id' in hit:
            assert client.body('lcm_load_session', session_id=hit['session_id'])['session_id'] == hit['session_id']
        if 'node_id' in hit:
            assert client.body('lcm_expand', node_id=hit['node_id'])['store'] == PREFIX
    page = client.body('lcm_load_session', session_id=qualified('remote0'), limit=1, include_exact_ref=True)
    assert page['next_cursor'].startswith(PREFIX + ':')
    next_page = client.body('lcm_load_session', session_id=qualified('remote0'), after_store_id=page['next_cursor'])
    assert next_page['messages'] and next_page['messages'][0]['store_id'] != page['messages'][0]['store_id']
    recalled = client.body('lcm_recall', query='marker', detail='answer_ready')
    for hit in recalled['other_stores'][0]['hits']:
        if 'exact_ref' in hit:
            ref = hit['exact_ref'].split(':', 1)[1]
            match = re.fullmatch(r'lcm:(\d+):(\d+)-(\d+)', ref)
            if match:
                store_id, start, end = map(int, match.groups())
                expanded = client.body('lcm_expand', store_id=qualified(store_id), content_offset=start)
                with sqlite3.connect(sources[0] / 'lcm.db') as conn:
                    content = conn.execute('SELECT content FROM messages WHERE store_id=?', (store_id,)).fetchone()[0]
                assert expanded['content'][:end - start] == content[start:end]
            else:
                node = int(ref.removeprefix('lcm-summary:'))
                assert client.body('lcm_expand', node_id=qualified(node))['store'] == PREFIX
        for key in ('expand_hint', 'session_hint'):
            if key in hit:
                call = ast.parse(hit[key], mode='eval').body
                args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                assert client.body(call.func.id, **args)['store'] == PREFIX
    result = client.body('lcm_grep', query='marker', conversation_id=qualified('remote0'))
    assert result['store'] == PREFIX and result.get('conversation_id') == qualified('remote0')


def test_07_schemas(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    reply = client.request('tools/list')
    plain = client.local_reply()
    for tool, original in zip(reply['result']['tools'], plain['result']['tools']):
        props = tool['inputSchema']['properties']
        for key, spec in original['inputSchema']['properties'].items():
            if key not in cr.SCALARS | cr.LISTS:
                assert props[key] == spec
                continue
            emitted = props[key]
            if spec.get('type') == 'integer':
                assert emitted['type'] == ['integer', 'string']
            assert emitted['description'].startswith(spec.get('description', ''))
            restored = dict(emitted)
            restored['description'] = spec.get('description', '')
            restored['type'] = spec['type']
            assert restored == spec
        tool['inputSchema'] = original['inputSchema']
        assert tool == original
    assert reply == plain


def test_08_unknown_and_own_prefixes(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    own_store = client.body('lcm_expand', store_id=1)['store_uuid']
    assert 'foreign_store' in client.body('lcm_expand', store_id=qualified(1, 'e' * 32))['error']
    assert 'foreign_store' in client.body('lcm_grep', query='marker', externalized_refs=[qualified('demo.json', 'e' * 32)])['error']
    for prefix in (own_store, OWN.replace('-', '')):
        assert client.body('lcm_expand', store_id=qualified(1, prefix))['content'].startswith('local marker')


def test_09_two_stores_no_calls(tmp_path, corpus, clients):
    local, sources = corpus
    for i in range(2):
        copy_generation(sources[i], local, COPIES[i])
    client = clients(tmp_path, local, fake=True)
    ready(client, 2)
    before = {p: p.read_bytes() for p in client.log.glob('*.requests')}
    body = client.body('lcm_expand', store_id=qualified(1), session_id=qualified('remote1', COPIES[1].replace('-', '')))
    assert body == {'error': 'ids from more than one store in one call'}
    assert {p: p.read_bytes() for p in client.log.glob('*.requests')} == before


def test_10_list_arguments(tmp_path, corpus, clients):
    local, sources = corpus
    for i in range(2):
        copy_generation(sources[i], local, COPIES[i])
    client = clients(tmp_path, local, fake=True)
    ready(client, 2)
    assert_view_readers(client)
    client.body('lcm_recall', query='marker', seen_refs=['lcm:1:0-5', qualified('lcm:2:0-5')],
                exclude_session_ids=['bare-session', qualified('remote0')])
    requests = []
    for path in client.log.glob('*.requests'):
        home = (client.log / (path.stem + '.pid')).read_text()
        args = json.loads(path.read_text().splitlines()[-1])['params']['arguments']
        requests.append((home, args))
    for home, args in requests:
        if not Path(home).name.startswith('view-'):
            assert args['seen_refs'] == ['lcm:1:0-5']
            assert args['exclude_session_ids'] == ['bare-session']
        elif COPIES[0] in home:
            assert args['seen_refs'] == ['lcm:2:0-5']
            assert args['exclude_session_ids'] == ['bare-session', 'remote0']
        else:
            assert args['seen_refs'] == []
            assert args['exclude_session_ids'] == ['bare-session']
    client.body('lcm_grep', query='marker', externalized_refs=['bare.json'])
    for entry in client.body('lcm_grep', query='marker', externalized_refs=['bare.json'])['other_stores']:
        assert entry['status'] == 'ok' and entry['hits'] == []


def key_paths(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield path + (key,)
            if key not in cr.STORED | cr.DROPPED:
                yield from key_paths(item, path + (key,))
    elif isinstance(value, list):
        for item in value:
            yield from key_paths(item, path + ('[]',))


def test_11_unknown_keys_and_inventory(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    fake = clients(tmp_path, local, fake=True)
    hit = ready(fake)['other_stores'][0]['hits'][0]
    assert 'unknown_handle' not in hit
    assert fake.body('lcm_expand', store_id=qualified(1))['results'][0].get('transcript_content') == 'verbatim store_id=3 lcm:3:0-5'
    fake.close()
    client = clients(tmp_path, local)
    ready(client)
    calls = [('lcm_grep', dict(query='marker', mode=mode, content_scope=scope))
             for mode in ('full_text', 'semantic', 'hybrid') for scope in ('history', 'externalized', 'both')]
    calls += [('lcm_recall', dict(query='marker', detail=detail, seen_refs=[]))
              for detail in ('snippets', 'answer_ready')]
    calls += [('lcm_recall', dict(query='marker', detail='answer_ready', include_occurrence_time=True)),
              ('lcm_expand', {'store_id': 1, 'include_exact_ref': True}),
              ('lcm_expand', {'store_id': 3}), ('lcm_expand', {'node_id': 1}),
              ('lcm_expand', {'node_id': 2, 'source_limit': 1}), ('lcm_expand', {'node_id': 0}),
              ('lcm_load_session', {'session_id': 'remote0', 'limit': 1, 'include_exact_ref': True}),
              ('lcm_load_session', {'session_id': 'remote0', 'anchor_store_id': 1}),
              ('lcm_describe', {'session_id': 'remote0'}), ('lcm_describe', {'node_id': 1}),
              ('lcm_describe', {'node_id': 2}), ('lcm_describe', {'node_id': 0})]
    generation = local / 'cloud' / 'replicas' / COPIES[0] / 'gen-1'
    plain = clients(tmp_path, generation, stock=True)
    paths = set()
    refs = list((generation / 'lcm-large-outputs').glob('*'))
    assert refs
    calls += [(tool, {'externalized_ref': refs[0].name}) for tool in ('lcm_expand', 'lcm_describe')]
    for tool, args in calls:
        body = plain.body(tool, **args)
        paths.update(key_paths(body))
    # Profile modes run on the source fixture, never by adding a .env to a generation.
    for unit in ('hits', 'session'):
        for guidance in ('v1', 'v2'):
            for strict in ('true', 'false'):
                (sources[0] / '.env').write_text(
                    'LCM_EMBEDDINGS_ENABLED=false\n'
                    f'LCM_RECALL_DELIVERY_UNIT={unit}\nLCM_RECALL_GUIDANCE={guidance}\n'
                    f'LCM_RECALL_REFERENCE_STRICT={strict}\nLCM_RECALL_SESSION_HINTS=true\n'
                    'LCM_RECALL_ADJACENT_REPLY=true\nLCM_RECALL_SNIPPET_WINDOW_HITS=2\n'
                    'LCM_RECALL_LATER_STATEMENTS=2\nLCM_RECALL_EXTRA_ENTRIES=1\n'
                    'LCM_RECALL_SUMMARY_EVENT_TIME=true\n')
                profile = clients(tmp_path, sources[0], stock=True)
                for detail in ('snippets', 'answer_ready'):
                    answer = profile.body('lcm_recall', query='marker', detail=detail, seen_refs=[])
                    assert 'hits' in answer, answer
                    paths.update(key_paths(answer))
                    rebuilt = cr.rebuild(answer, PREFIX)
                    list(scan(rebuilt))
                    if unit == 'session':
                        assert rebuilt['provenance']['delivery']['sessions']
                profile.close()
    print('INVENTORY_PATHS=' + json.dumps(sorted('.'.join(path) for path in paths)))
    temporal_dropped = {'reason', 'session_date', 'policy_version', 'support', 'quote',
                        'anchor_source', 'observation_is_not_occurrence'}
    unknown = [path for path in paths if path[-1] not in cr.CLASSIFIED | cr.DROPPED | temporal_dropped and not re.fullmatch(r'd\d+', path[-1])]
    assert unknown == [], sorted(unknown)
    temporal = dict(time_uncertain='carried', observed_at=1700000000, ingested_at=1700000001,
                    date='2023-11-14', stored_at=1700000001, occurred_at=1699920000,
                    event_at=1699920000, event_date='2023-11-14', precision='day')
    for container in ('occurrence_time', 'observation_time'):
        assert cr.rebuild({container: {**temporal, 'reason': 'free text'}}, PREFIX) == {container: temporal}
    assert {'results', 'hits', 'messages', 'depths', 'error'} <= {p[-1] for p in paths}


def test_12_dead_replica(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local, budget=2)
    ready(client)
    old = client.pids(True)[0]
    os.kill(old, signal.SIGKILL)
    reply = client.call(query='marker')
    assert_local(client, reply)
    assert json.loads(reply['result']['content'][0]['text'])['other_stores'][0]['status'] == 'unavailable'
    ready(client)
    assert len(client.pids(True)) == 2


def test_13_late_replica(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local, fake=True, wait=0.2, budget=2)
    ready(client)
    pid = client.pids(True)[0]
    client.control(stall=True)
    reply = client.call(query='marker')
    assert_local(client, reply)
    assert json.loads(reply['result']['content'][0]['text'])['other_stores'][0]['status'] == 'late'
    eventually(lambda: not alive(pid), timeout=8)
    client.control(stall=False)
    client.body(query='marker')
    ready(client)
    assert len(client.pids(True)) == 2


def test_14_bad_replicas(tmp_path, corpus, clients):
    local, sources = corpus
    generation = copy_generation(sources[0], local, COPIES[0])
    pointer = json.loads((generation.parent / 'view.json').read_text())
    (view_folder(generation) / 'lcm.db').unlink()
    (view_folder(generation) / 'lcm.db').write_bytes(b'corrupt')
    root = generation.parent.parent
    for name, folder in [(COPIES[1], 'view-2-1-' + 'a'*16), (COPIES[2], '../escape'), ('invalid', 'bad'), (OWN, 'bad')]:
        copy = root / name
        copy.mkdir()
        (copy / 'view.json').write_text(json.dumps(dict(pointer, folder=folder)))
    linked = root / str(uuid.UUID(int=44))
    linked.mkdir()
    (linked / pointer['folder']).symlink_to(sources[0], target_is_directory=True)
    (linked / 'view.json').write_text(json.dumps(pointer))
    client = clients(tmp_path, local)
    reply = client.call(query='marker')
    assert_local(client, reply)
    entries = json.loads(reply['result']['content'][0]['text'])['other_stores']
    assert all(e['status'] in ('unavailable', 'warming') and not e['hits'] for e in entries)
    assert all(e['copy_id'] not in ('invalid', OWN.replace('-', '')) for e in entries)
    assert not any(Path(p.read_text()).name.startswith('gen-') for p in client.log.glob('*.pid'))

    eventually(lambda: client.body(query='marker')['other_stores'][0]['status'] == 'unavailable')


def test_15_optin(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    path = local / 'cloud' / 'cloud.json'
    for mode in ('missing', 'writable'):
        if mode == 'missing':
            path.rename(path.with_suffix('.disabled'))
        else:
            path.with_suffix('.disabled').rename(path)
            path.chmod(0o620)
        client = clients(tmp_path, local)
        for method, params in [('tools/list', None), ('tools/call', {'name': 'lcm_grep', 'arguments': {'query': 'marker'}})]:
            reply = client.request(method, params)
            assert reply == client.local_reply()
        assert not client.pids(True)
        client.close()


@pytest.mark.parametrize('failure', ['startup', 'parse'])
def test_16_startup_fallback(tmp_path, corpus, clients, failure):
    local, _ = corpus
    (local / 'cloud' / 'replicas').write_text('not a directory')
    client = clients(tmp_path, local)
    assert_local(client, client.call(query='marker'))
    missing = Client(tmp_path, local, missing=True)
    try:
        _, err = missing.proc.communicate('', timeout=10)
        assert missing.proc.returncode == 2 and len(err.splitlines()) == 1
    finally:
        missing.close()
    # A bare-name reader works at Popen and must work at both exec fallbacks.
    stub = tmp_path / 'r12-stock-reader'
    stub.write_text(f'#!{sys.executable}\nprint("r12 PATH stock reader", flush=True)\n')
    stub.chmod(0o700)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ.get('PATH', ''),
               LCMX_LOCAL_READER_CMD=json.dumps([stub.name, '--store', str(local / 'lcm.db'),
                                                '--hermes-home', str(local)]))
    command = [sys.executable, '-I', '-S', str(ROOT / 'cloud_reader.py')]
    if failure == 'parse':
        command.append('--unknown-r12-flag')
    result = subprocess.run(command, input='', capture_output=True, text=True, env=env, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'r12 PATH stock reader\n'


def test_17_generation_change(tmp_path, corpus, clients):
    local, sources = corpus
    old = copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    assert_view_readers(client)
    pid = client.pids(True)[0]
    old_view = view_folder(old)
    first = client.body('lcm_expand', store_id=qualified(1))
    source = sources[0]
    store = cc.open_store(source / 'lcm.db', source)
    transcript = _transcript(tmp_path / 'extra', [_entry('user', 'new', 'newgeneration marker', 'newsession')], 'newsession')
    assert cc.drain(store, cc.Lineage('newsession', None, transcript)) == 'drained'
    assert cc.to_rollback_journal(store.connection)
    store.close()
    assert cc.leave_readable(source / 'lcm.db')
    copy_generation(source, local, COPIES[0], 2)
    eventually(lambda: client.body(query='marker')['other_stores'][0]['generation'] == 2, timeout=40)
    after = client.body('lcm_expand', store_id=qualified(1))
    assert first['content'] == after['content']
    entry = client.body(query='newgeneration')['other_stores'][0]
    assert entry['generation'] == 2 and any('newgeneration' in h.get('snippet', '') for h in entry['hits'])
    eventually(lambda: not alive(pid))
    assert old.exists() and old_view.exists()


def test_18_folder_lock(tmp_path, corpus, clients):
    local, sources = corpus
    generation = copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    fd = os.open(view_folder(generation), os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client.close()
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def forget(local, session):
    store = cc.open_store(local / 'lcm.db', local)
    store.delete_session_messages(session)
    assert cc.to_rollback_journal(store.connection)
    store.close()
    assert cc.leave_readable(local / 'lcm.db')


def test_19_forgotten_search(tmp_path, corpus, clients):
    local, sources = corpus
    # Record the tombstone through the capture writer's deletion hook.
    store = cc.open_store(local / 'lcm.db', local)
    transcript = _transcript(tmp_path / 'forgotten', [_entry('user', 'f1', 'remote0 marker', 'remote0')], 'remote0')
    assert cc.drain(store, cc.Lineage('remote0', None, transcript)) == 'drained'
    assert cc.to_rollback_journal(store.connection)
    store.close()
    forget(local, 'remote0')
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    for tool in ('lcm_grep', 'lcm_recall'):
        for args in ({}, {'session_id': 'remote0'}):
            entry = client.body(tool, query='marker', **args)['other_stores'][0]
            assert entry['hits'] == []


def test_20_forgotten_by_id(tmp_path, corpus, clients):
    local, sources = corpus
    # A capture deletion records a tombstone even when no rows remain locally.
    store = cc.open_store(local / 'lcm.db', local)
    path = _transcript(tmp_path / 'forget', [_entry('user', 'f', 'marker', 'remote0')], 'remote0')
    cc.drain(store, cc.Lineage('remote0', None, path))
    assert cc.to_rollback_journal(store.connection)
    store.close()
    forget(local, 'remote0')
    generation = copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    twin, _ = make_store(tmp_path / 'twin', 'unrelated')
    # Delete all rows from a separate synthetic replica to get stock missing answers.
    store = cc.open_store(twin / 'lcm.db', twin)
    store.delete_session_messages('unrelated')
    cc.to_rollback_journal(store.connection)
    store.close()
    cc.leave_readable(twin / 'lcm.db')
    plain = clients(tmp_path, twin, stock=True)
    ref = next((generation / 'lcm-large-outputs').iterdir()).name
    for tool, args in [('lcm_expand', {'store_id': 1}), ('lcm_expand', {'externalized_ref': ref}),
                       ('lcm_load_session', {'session_id': 'remote0'}),
                       ('lcm_load_session', {'session_id': 'remote0', 'anchor_store_id': 1}),
                       ('lcm_grep', {'query': 'marker', 'externalized_refs': [ref], 'content_scope': 'externalized'})]:
        routed_args = {key: ([qualified(v) for v in value] if isinstance(value, list) else qualified(value))
                       if key in cr.SCALARS | cr.LISTS else value for key, value in args.items()}
        actual = client.body(tool, **routed_args)
        raw_expected = {k: v for k, v in plain.body(tool, **args).items() if k not in ('reader', 'store_uuid')}
        if 'error' in raw_expected:
            text = raw_expected['error']
            values = [v for key, value in args.items() if key in cr.SCALARS | cr.LISTS
                      for v in (value if isinstance(value, list) else [value])]
            for value in values:
                text = re.sub(r'(?<![\w:.-])' + re.escape(str(value)) + r'(?![\w.-])', qualified(value), text)
            expected = {'error': text}
        else:
            expected = cr.rebuild(raw_expected, PREFIX)
        assert {k: v for k, v in actual.items() if k not in ('store', 'generation')} == expected




def snapshot(home):
    return {str(p.relative_to(home)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in home.rglob('*')}


def test_22_no_writes_or_connections(tmp_path, corpus, clients):
    local, sources = corpus
    forget(local, 'remote0')
    assert forgotten_ids(local) == {'remote0'}
    copy_generation(sources[0], local, COPIES[0])
    before = snapshot(local)
    client = clients(tmp_path, local, audit=True)
    ready(client)
    assert_view_readers(client)
    for tool, args in [('lcm_grep', {'query': 'marker'}), ('lcm_recall', {'query': 'marker'}),
                       ('lcm_expand', {'store_id': qualified(1)}), ('lcm_describe', {'session_id': qualified('remote0')}),
                       ('lcm_load_session', {'session_id': qualified('remote0')})]:
        client.body(tool, **args)
    client.close()
    assert snapshot(local) == before
    assert (client.log / 'merger-audit').exists(), 'the merger itself must run under the socket audit hook'
    assert (client.log / 'merger-audit').read_text() == 'installed\n'
    assert not (client.log / 'connections').exists()


def test_23_local_restart(tmp_path, corpus, clients):
    local, _ = corpus
    client = clients(tmp_path, local)
    pid = client.pids(False)[0]
    os.kill(pid, signal.SIGKILL)
    assert client.body('lcm_expand', store_id=1)['content'].startswith('local marker')
    assert len(client.pids(False)) == 2


def test_24_children_exit(tmp_path, corpus, clients):
    local, sources = corpus
    generations = [copy_generation(source, local, copy_id) for source, copy_id in zip(sources, COPIES)]
    client = clients(tmp_path, local)
    ready(client, 3)
    assert_view_readers(client)
    pids = client.pids()
    assert len(pids) == 4
    client.close()
    eventually(lambda: all(not alive(pid) for pid in pids))
    for generation in generations:
        fd = os.open(view_folder(generation), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


SLOW_READER = r'''
import json, pathlib, sys, time
home = pathlib.Path(sys.argv[sys.argv.index('--hermes-home') + 1])
for line in sys.stdin:
    req = json.loads(line)
    if 'id' not in req: continue
    if req['method'] == 'tools/call':
        if not home.name.startswith('view-'): time.sleep(1.5)
        result = {'content': [{'type': 'text', 'text': '{"answer": "slow local", "results": [], "hits": []}'}], 'isError': False}
    else:
        result = {'serverInfo': {'name': 'slow'}}
    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': result}) + '\n')
    sys.stdout.flush()
'''


def run_merger(argv, requests, *extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('LCM_') and k != 'PYTHONPATH'}
    env['LCMX_LOCAL_READER_CMD'] = json.dumps(argv)
    return subprocess.run([sys.executable, '-I', '-S', str(ROOT / 'cloud_reader.py'), *extra],
                          input=requests, capture_output=True, env=env, timeout=60)


@pytest.mark.parametrize('replicas', [False, True])
@pytest.mark.parametrize('tool', ['lcm_grep', 'lcm_recall'])
def test_r4_slow_local_has_no_budget(tmp_path, corpus, replicas, tool):
    local, sources = corpus
    if replicas:
        copy_generation(sources[0], local, COPIES[0])
    else:
        (local / 'cloud' / 'cloud.json').unlink()
    script = tmp_path / 'slow.py'
    script.write_text(SLOW_READER)
    argv = [sys.executable, str(script), '--store', str(local / 'lcm.db'),
            '--hermes-home', str(local), '--log-dir', str(tmp_path / 'logs')]
    requests = cr.request(1, 'initialize') + cr.request(2, 'tools/call', {'name': tool, 'arguments': {'query': 'x'}})
    plain = subprocess.run(argv, input=requests, capture_output=True, timeout=30)
    merged = run_merger(argv, requests, '--budget-s', '1')
    replies = merged.stdout.splitlines()
    assert merged.returncode == 0 and len(replies) == 2, 'slow local call lost at the replica budget'
    expected = json.loads(plain.stdout.splitlines()[1])
    actual = json.loads(replies[1])
    actual['result']['content'][0]['text'] = cr.strip_other_stores(actual['result']['content'][0]['text'])
    assert actual == expected
    if not replicas:
        assert merged.stdout == plain.stdout


def test_r4_unhashable_session_keeps_serving(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    plain = clients(tmp_path, local, stock=True)
    expected = plain.call(query='marker', session_id=['x'])['result']
    client = clients(tmp_path, local)
    ready(client)
    reply = client.call(query='marker', session_id=['x'])['result']
    reply['content'][0]['text'] = cr.strip_other_stores(reply['content'][0]['text'])
    assert reply == expected
    assert_local(client, client.call(query='marker'))


def test_r4_request_exceptions_fall_back_and_continue(monkeypatch):
    first = cr.request(1, 'ping')
    second = cr.request(2, 'ping')
    third = cr.request(3, 'ping')
    fallback = b'{"jsonrpc":"2.0","id":1,"result":{}}\n'
    success = b'{"jsonrpc":"2.0","id":3,"result":{}}\n'
    class BrokenMerger:
        def __init__(self, *args): pass
        def handle(self, raw):
            if raw == third:
                return success
            raise RuntimeError('handle failed')
        def local_call(self, raw, notification=False):
            if raw == first:
                return fallback
            raise RuntimeError('local failed')
        def close(self): pass
    output = io.BytesIO()
    monkeypatch.setattr(cr, 'Merger', BrokenMerger)
    monkeypatch.setattr(sys, 'argv', ['cloud_reader.py'])
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(first + second + b'invalid\n' + third)))
    monkeypatch.setattr(sys, 'stdout', SimpleNamespace(buffer=output))
    monkeypatch.setenv('LCMX_LOCAL_READER_CMD', json.dumps(['fixture-reader']))
    assert cr.main() == 0, 'a request exception must not end the merger'
    replies = output.getvalue().splitlines()
    assert replies[0] + b'\n' == fallback and replies[-1] + b'\n' == success
    for raw, ident in zip(replies[1:3], [2, None]):
        reply = json.loads(raw)
        assert reply['jsonrpc'] == '2.0' and reply['id'] == ident
        assert reply['error']['code'] == -32603
    assert len(replies) == 4


@pytest.mark.parametrize('prefix_kind', ['store_uuid', 'unknown'])
def test_r4_local_list_prefix_preserved(tmp_path, corpus, clients, prefix_kind):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    plain = clients(tmp_path, local, stock=True)
    prefix = plain.body('lcm_expand', store_id=1)['store_uuid'] if prefix_kind == 'store_uuid' else 'e' * 32
    args = {'query': 'marker', 'exclude_session_ids': [prefix + ':local']}
    expected = plain.call(**args)['result']['content'][0]['text']
    if prefix_kind == 'store_uuid':
        assert json.loads(expected)['results'] == []
    else:
        assert 'foreign_store' in expected
    client = clients(tmp_path, local)
    ready(client)
    actual = client.call(**args)['result']['content'][0]['text']
    assert cr.strip_other_stores(actual) == expected


def test_r4_argument_parse_execs_stock(tmp_path, corpus):
    local, _ = corpus
    script = tmp_path / 'stock.py'
    script.write_text(SLOW_READER)
    argv = [sys.executable, str(script), '--store', str(local / 'lcm.db'),
            '--hermes-home', str(local), '--log-dir', str(tmp_path / 'logs')]
    requests = cr.request(1, 'initialize')
    plain = subprocess.run(argv, input=requests, capture_output=True, timeout=30)
    merged = run_merger(argv, requests, '--unknown-flag')
    assert merged.returncode == plain.returncode == 0 and merged.stdout == plain.stdout


def test_r4_own_prefix_session_fanout(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    expected = client.body(query='marker', session_id='local')['other_stores']
    actual = client.body(query='marker', session_id=qualified('local', OWN.replace('-', '')))['other_stores']
    assert actual == expected and all(entry['status'] == 'ok' for entry in actual)
    requests = [json.loads(line) for pid in client.pids(True)
                for line in (client.log / f'{pid}.requests').read_text().splitlines()]
    assert requests[-1]['params']['arguments']['session_id'] == 'local'






@pytest.mark.parametrize('key', ['seen_refs', 'exclude_session_ids'])
def test_r4_list_entries_go_to_owner(key):
    merger = object.__new__(cr.Merger)
    merger.own = OWN.replace('-', '')
    second = COPIES[1].replace('-', '')
    merger.replicas = {PREFIX: None, second: None}
    local_uuid, unknown = 'd' * 32, 'e' * 32
    args = {key: ['bare', qualified('a'), qualified('b', second),
                  qualified('home', merger.own), qualified('origin', local_uuid), qualified('unknown', unknown)]}
    for prefix in (None, merger.own):
        assert merger.arguments(args, prefix, True)[key] == [
            'bare', 'home', qualified('origin', local_uuid), qualified('unknown', unknown)]
    for prefix, bare in [(PREFIX, 'a'), (second, 'b')]:
        assert merger.arguments(args, prefix, True)[key] == (['bare', bare] if key == 'exclude_session_ids' else [bare])




def test_r6_notification_double_failure_keeps_serving(monkeypatch):
    notification = b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
    second = cr.request(2, 'ping')
    success = b'{"jsonrpc":"2.0","id":2,"result":{}}\n'
    handled, fallbacks = [], []
    class BrokenMerger:
        def __init__(self, *args): pass
        def handle(self, raw):
            handled.append(raw)
            if raw == second:
                return success
            raise RuntimeError('handle failed')
        def local_call(self, raw, notification=False):
            fallbacks.append((raw, notification))
            raise RuntimeError('local failed')
        def close(self): pass
    output = io.BytesIO()
    monkeypatch.setattr(cr, 'Merger', BrokenMerger)
    monkeypatch.setattr(sys, 'argv', ['cloud_reader.py'])
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(notification + second)))
    monkeypatch.setattr(sys, 'stdout', SimpleNamespace(buffer=output))
    monkeypatch.setenv('LCMX_LOCAL_READER_CMD', json.dumps(['fixture-reader']))
    assert cr.main() == 0
    assert handled == [notification, second] and fallbacks == [(notification, True)]
    assert output.getvalue() == success, 'a failed notification must not emit a reply'


def r7_async(call):
    future = cr.Future()
    def run():
        try:
            future.set_result(call())
        except Exception as exc:
            future.set_exception(exc)
    threading.Thread(target=run, daemon=True).start()
    return future


@pytest.fixture
def direct_mergers(tmp_path, monkeypatch):
    wrapper = tmp_path / 'direct-reader.py'
    wrapper.write_text(WRAPPER)
    monkeypatch.setenv('TEST_READER_PATH', json.dumps(sys.path))
    monkeypatch.setenv('TEST_STOCK_SCRIPT', str(ROOT / 'scripts' / 'lcm_x_mcp.py'))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    opened = []
    def start(local, **kwargs):
        argv = [sys.executable, str(wrapper), '--store', str(local / 'lcm.db'),
                '--hermes-home', str(local), '--log-dir', str(tmp_path / 'direct-logs')]
        merger = cr.Merger(argv, **kwargs)
        opened.append(merger)
        merger.handle(cr.request(1, 'initialize', {'protocolVersion': '2025-06-18'}))
        return merger
    yield start
    for merger in opened:
        try:
            merger.close()
        except OSError:
            pass  # Preserve the regression assertion when the base cleanup fails too.


@pytest.mark.parametrize('tool,args', [('lcm_expand', {'store_id': qualified(1)}),
                                      ('lcm_grep', {'query': 'marker'})])
def test_r7_disabled_tool_stays_local(tmp_path, corpus, clients, tool, args):
    local, sources = corpus
    (local / '.env').write_text('LCM_EMBEDDINGS_ENABLED=false\nLCM_DISABLED_TOOLS=lcm_expand,lcm_grep\n')
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    eventually(lambda: client.body('lcm_recall', query='marker')['other_stores'][0]['status'] == 'ok', timeout=40)
    logs = [client.log / f'{pid}.requests' for pid in client.pids(True)]
    before = [path.read_bytes() for path in logs]
    plain = clients(tmp_path, local, stock=True)
    expected = plain.call(tool, **args)['result']
    reply = client.call(tool, **args)
    assert reply['result'] == expected, 'disabled calls must return the stock local error unchanged'
    assert reply['result']['isError'] is True
    assert [path.read_bytes() for path in logs] == before, 'disabled tool reached a replica'


def test_r7_exited_child_completes_future():
    child = cr.Child(['/usr/bin/true'], None)
    child.process.wait(timeout=5)
    try:
        with pytest.raises(Exception) as error:
            child.ask(cr.request(1, 'ping')).result(timeout=5)
        assert not isinstance(error.value, TimeoutError), 'dead reader left its Future pending'
    finally:
        try:
            child.close()
        except OSError:
            pass


def test_r7_broken_pipe_close_releases_all(tmp_path):
    child = cr.Child(['/usr/bin/true'], None, tmp_path)
    child.process.wait(timeout=5)
    fd = child.fd
    child.process.stdin.write(b'buffer retained until close\n')
    errors = []
    for _ in range(2):
        try:
            child.close()
        except OSError as exc:
            errors.append(exc)
    try:
        assert errors == [], 'close must tolerate broken-pipe flush and remain idempotent'
        assert child.process.stdin.closed and child.process.stdout.closed
        with pytest.raises(OSError):
            os.fstat(fd)
    finally:
        if child.fd is not None:
            os.close(child.fd)
        child.process.stdout.close()


@pytest.mark.parametrize('opted_in', [False, True])
def test_r7_reaped_local_restarts(corpus, direct_mergers, opted_in):
    local, _ = corpus
    if not opted_in:
        (local / 'cloud' / 'cloud.json').unlink()
    merger = direct_mergers(local)
    old = merger.local.process
    old.kill()
    old.wait(timeout=5)
    future = r7_async(lambda: merger.handle(cr.tool_request('lcm_expand', {'store_id': 1})))
    cr.wait([future], timeout=10)
    assert future.done(), 'reaped local reader stalled the request for 10 seconds'
    body, _ = cr.tool_body(future.result())
    assert body['content'].startswith('local marker')
    assert merger.local.process.pid != old.pid


def test_r7_reaped_replica_is_unavailable(corpus, direct_mergers, monkeypatch):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    merger = direct_mergers(local, replica_wait=5)
    eventually(lambda: cr.tool_body(merger.handle(cr.tool_request('lcm_grep', {'query': 'marker'})))[0]
               ['other_stores'][0]['status'] == 'ok', timeout=40)
    original = merger.replica_call
    def kill_before_write(replica, *args):
        replica.child.process.kill()
        replica.child.process.wait(timeout=5)
        return original(replica, *args)
    monkeypatch.setattr(merger, 'replica_call', kill_before_write)
    started = time.monotonic()
    body, _ = cr.tool_body(merger.handle(cr.tool_request('lcm_grep', {'query': 'marker'})))
    assert body['other_stores'][0]['status'] == 'unavailable', 'a reaped reader must never be late'
    assert time.monotonic() - started < 2, 'dead reader consumed the replica wait'


def test_r7_reaped_warming_probe_completes(tmp_path, monkeypatch):
    original = cr.Child
    children = []
    def exited_child(argv, budget, folder):
        child = original(['/usr/bin/true'], None, folder)
        child.process.wait(timeout=5)
        children.append(child)
        return child
    monkeypatch.setattr(cr, 'Child', exited_child)
    replica = cr.Replica(PREFIX, ['fixture-reader', '--store', 'unused', '--hermes-home', 'unused'], 8)
    assert 'digests' in __import__('inspect').signature(replica.refresh).parameters
    replica.refresh(tmp_path, 1, ('forgotten', 'removed'))
    try:
        cr.wait([replica.probe], timeout=5)
        assert replica.probe.done(), 'dead warming reader left its probe pending'
        with pytest.raises(Exception):
            replica.probe.result()
    finally:
        for child in children:
            try:
                child.close()
            except OSError:
                if child.fd is not None:
                    os.close(child.fd)
                child.process.stdout.close()


@pytest.mark.parametrize('tool', ['lcm_describe', 'lcm_grep'])
def test_r7_forgotten_session_equals_missing(tmp_path, corpus, clients, tool):
    local, sources = corpus
    forget(local, 'remote0')
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    args = {'query': 'marker'} if tool == 'lcm_grep' else {}
    missing = client.call(tool, session_id=qualified('never-existing'), **args)['result']
    actual = client.call(tool, session_id=qualified('remote0'), **args)['result']
    expected = json.loads(json.dumps(missing).replace(qualified('never-existing'), qualified('remote0')))
    assert actual == expected, 'forgotten session must have the stock missing-session success shape'
    assert actual['isError'] is False


@pytest.mark.parametrize('args', [{'externalized_refs': ['payload']}, {'session_id': 'forgotten'}])
def test_r7_busy_routed_precheck_fails_closed(args):
    replica = SimpleNamespace(prefix=PREFIX, status='ok', generation=1,
                              child=SimpleNamespace(ask=lambda raw: None))
    merger = object.__new__(cr.Merger)
    try:
        assert len(__import__('inspect').signature(merger.routed).parameters) == 3
        answer = merger.routed(replica, 'lcm_grep', args)
    except Exception as exc:
        answer = exc
    assert answer == {'error': 'the other store could not answer this call', 'store': PREFIX}


def test_r7_exclusive_generation_lock_never_stalls(corpus, direct_mergers):
    local, sources = corpus
    folder = copy_generation(sources[0], local, COPIES[0])
    fd = os.open(view_folder(folder), os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    future = r7_async(lambda: direct_mergers(local))
    try:
        cr.wait([future], timeout=2)
        assert future.done(), 'held generation lock stalled merger startup'
        merger = future.result()
        started = time.monotonic()
        raw = cr.tool_request('lcm_grep', {'query': 'marker'})
        body, reply = cr.tool_body(merger.handle(raw))
        assert time.monotonic() - started < 2
        assert body['other_stores'][0]['status'] == 'warming'
        assert merger.replicas[PREFIX].pending is None
        plain = json.loads(merger.local_call(raw))['result']['content'][0]['text']
        assert cr.strip_other_stores(reply['result']['content'][0]['text']) == plain
    finally:
        os.close(fd)
        future.result(timeout=10)  # Release a blocked base constructor before fixture cleanup.


def test_r8_startup_warms_only_with_optin(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    (local / 'cloud' / 'cloud.json').unlink()
    plain = clients(tmp_path, local)
    requests = [json.loads(line) for pid in plain.pids(False)
                for line in (plain.log / f'{pid}.requests').read_text().splitlines()]
    assert [req['method'] for req in requests] == ['initialize'], 'no opt-in must add no private tools/list'
    assert plain.pids(True) == []
    plain.close()
    optin(local)
    client = clients(tmp_path, local)
    deadline = time.monotonic() + 5
    while not client.pids(True) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert any(alive(pid) for pid in client.pids(True)), 'initialize alone must start a replica reader'
    assert_view_readers(client)


def test_r8_idle_start_first_grep_is_ok(corpus, direct_mergers):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    merger = direct_mergers(local)
    replica = merger.replicas[PREFIX]
    # Wait only for background warm-up; send no tool call to trigger it.
    if replica.probe is not None:
        cr.wait([replica.probe], timeout=20)
        assert replica.probe.done(), 'startup warm-up did not finish within its bounded wait'
        replica.probe.result()
    body, _ = cr.tool_body(merger.handle(cr.tool_request('lcm_grep', {'query': 'marker'})))
    assert body['other_stores'][0]['status'] == 'ok', 'first grep after idle warm-up must include the replica'
    assert body['other_stores'][0]['hits']
    assert replica.child.folder.name.startswith('view-')


def test_r8_recall_forgotten_session_equals_missing(tmp_path, corpus, clients):
    local, sources = corpus
    forget(local, 'remote0')
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    missing = client.call('lcm_recall', query='marker', session_id=qualified('never-existing'))['result']
    actual = client.call('lcm_recall', query='marker', session_id=qualified('remote0'))['result']
    expected = json.loads(json.dumps(missing).replace(qualified('never-existing'), qualified('remote0')))
    assert actual == expected, 'forgotten recall scope must equal the missing-session success answer'
    assert actual['isError'] is False
    body = json.loads(actual['content'][0]['text'])
    assert body['hits'] == [], 'no hit from the forgotten session may survive'


@pytest.mark.parametrize('text', [
    'visible1 marker commit 0123456789abcdef0123456789abcdef01234567',
    'visible1 marker store_id=3 lcm:3:0-5',
], ids=['commit_sha', 'id_patterns'])
def test_r9_recall_forgotten_with_visible_hits(tmp_path, corpus, clients, text):
    local, sources = corpus
    store = cc.open_store(sources[0] / 'lcm.db', sources[0])
    try:
        transcript = _transcript(tmp_path / 'extra-visible1',
                                 [_entry('user', 'x1', text, 'visible1')], 'visible1')
        assert cc.drain(store, cc.Lineage('visible1', None, transcript)) == 'drained'
        assert cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    assert cc.leave_readable(sources[0] / 'lcm.db')
    assert (sources[0] / 'lcm.db').stat().st_size < 5_000_000
    forget(local, 'remote0')
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    missing = client.call('lcm_recall', query='marker', session_id=qualified('never-existing'))['result']
    actual = client.call('lcm_recall', query='marker', session_id=qualified('remote0'))['result']
    expected = json.loads(json.dumps(missing).replace(qualified('never-existing'), qualified('remote0')))
    assert actual['isError'] is False, 'visible stored text must not turn forgotten recall into an error'
    assert actual == expected, 'forgotten recall must equal the missing-session answer with visible hits'
    hits = json.loads(actual['content'][0]['text'])['hits']
    assert any(hit.get('session_id') == qualified('visible1') for hit in hits)
    assert all(hit.get('session_id') != qualified('remote0') for hit in hits)


@pytest.mark.parametrize('limit', [None, 1], ids=['default_limit', 'limit_1'])
def test_r10_forgotten_cursor_equals_missing(tmp_path, corpus, clients, limit):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    before = clients(tmp_path, local)
    ready(before)
    page = before.body('lcm_load_session', session_id=qualified('remote0'), limit=1)
    assert page['has_more'] is True
    cursor = page['next_cursor']
    before.close()
    forget(local, 'remote0')
    client = clients(tmp_path, local)
    ready(client)
    extra = {'after_store_id': cursor, **({'limit': limit} if limit is not None else {})}
    missing = client.call('lcm_load_session', session_id=qualified('never-existing'), **extra)['result']
    actual = client.call('lcm_load_session', session_id=qualified('remote0'), **extra)['result']
    expected = json.loads(json.dumps(missing).replace(qualified('never-existing'), qualified('remote0')))
    assert actual['isError'] is False, 'a forgotten cursor must return the missing-session empty page'
    assert actual == expected
    body = json.loads(actual['content'][0]['text'])
    assert body['messages'] == [] and body['after_store_id'] == cursor


@pytest.mark.parametrize('query', [
    '0123456789abcdef0123456789abcdef01234567',
    'parser regression fixed in deadbeefcafebabe0123456789abcdef42 yesterday',
    ' 0123456789abcdef0123456789abcdef01234567 ',
    '0123456789abcdef0123456789abcdef01234567\n',
    'parser regression fixed in deadbeefcafebabe0123456789abcdef42 yesterday ',
    10**40,
], ids=['commit_sha', 'hex_in_sentence', 'padded_sha', 'newline_sha', 'padded_sentence', 'integer_query'])
def test_r10_forgotten_grep_echo_equals_missing(tmp_path, corpus, clients, query):
    local, sources = corpus
    forget(local, 'remote0')
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    missing = client.call('lcm_grep', query=query, session_id=qualified('never-existing'))['result']
    actual = client.call('lcm_grep', query=query, session_id=qualified('remote0'))['result']
    expected = json.loads(json.dumps(missing).replace(qualified('never-existing'), qualified('remote0')))
    assert actual['isError'] is False, 'the caller query must not trigger the replica error scan'
    assert actual == expected
    body = json.loads(actual['content'][0]['text'])
    assert body['query'] == str(query).strip() and body['results'] == []




@pytest.mark.parametrize('cursor', [-1, 10**40, 'abc'], ids=['negative', 'overflow', 'noninteger'])
@pytest.mark.parametrize('limit', [None, 1], ids=['default_limit', 'limit_1'])
def test_r11_forgotten_invalid_cursor_equals_missing(tmp_path, corpus, clients, cursor, limit):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    before = clients(tmp_path, local)
    ready(before)
    page = before.body('lcm_load_session', session_id=qualified('remote0'), limit=1)
    anchor = page['messages'][0]['store_id']
    assert page['messages'][0]['session_id'] == qualified('remote0')
    before.close()
    forget(local, 'remote0')
    client = clients(tmp_path, local)
    ready(client)
    for anchored in (False, True):
        extra = {'after_store_id': cursor, **({'limit': limit} if limit is not None else {}),
                 **({'anchor_store_id': anchor} if anchored else {})}
        missing = client.call('lcm_load_session', session_id=qualified('never-existing'), **extra)['result']
        actual = client.call('lcm_load_session', session_id=qualified('remote0'), **extra)['result']
        expected = json.loads(json.dumps(missing).replace(qualified('never-existing'), qualified('remote0')))
        assert actual == expected, (cursor, limit, anchored)
        assert actual['isError'] is True


@pytest.mark.parametrize('mode', ['full_text', 'semantic', 'hybrid'])
@pytest.mark.parametrize('shape', ['forgotten', 'padded', 'mixed'])
def test_r12_forgotten_conversation_equals_missing(tmp_path, corpus, clients, mode, shape):
    local, sources = corpus
    source = sources[0]
    store = cc.open_store(source / 'lcm.db', source)
    try:
        for session, conversation in [('r12-visible', 'r12-visible-conversation'),
                                      ('r12-mixed-visible', 'r12-mixed-conversation')]:
            transcript = _transcript(tmp_path / session,
                [_entry('user', 'r12', session + ' marker', session)], session)
            assert cc.drain(store, cc.Lineage(session, None, transcript)) == 'drained'
            store.connection.execute('UPDATE messages SET conversation_id=? WHERE session_id=?',
                                     (conversation, session))
        store.connection.execute("UPDATE messages SET conversation_id='r12-forgotten-conversation' WHERE session_id='remote0'")
        store.connection.execute("UPDATE messages SET conversation_id='r12-mixed-conversation' WHERE store_id=4")
        store.connection.commit()
        assert cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    assert cc.leave_readable(source / 'lcm.db')
    forget(local, 'remote0')
    generation = copy_generation(source, local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    plain = clients(tmp_path, generation, stock=True)
    extra = dict(mode=mode, content_scope='history', limit=1)
    visible = plain.body(query='marker', conversation_id='r12-visible-conversation', **extra)
    actual_visible = client.body(query='marker', conversation_id=qualified('r12-visible-conversation'), **extra)
    assert visible['results'] and visible['total_results'] == 1
    assert actual_visible == {'store': PREFIX, 'generation': 1, **cr.rebuild(visible, PREFIX)}
    if shape == 'mixed':
        mixed = client.body(query='marker', conversation_id=qualified('r12-mixed-conversation'), **extra)
        assert mixed['total_results'] == 1
        assert {h['session_id'] for h in mixed['results']} == {qualified('r12-mixed-visible')}
        assert_view_readers(client)
        return
    conversation = 'r12-mixed-conversation' if shape == 'mixed' else 'r12-forgotten-conversation'
    requested = ' ' + conversation + ' ' if shape == 'padded' else conversation
    missing = client.call(query='marker', conversation_id=qualified('r12-never-existing'), **extra)['result']
    actual = client.call(query='marker', conversation_id=qualified(requested), **extra)['result']
    body = json.loads(actual['content'][0]['text'])
    assert body['total_results'] == 0, 'forgotten conversation match count must be missing, not query dependent'
    expected = json.loads(json.dumps(missing).replace(qualified('r12-never-existing'), qualified(conversation)))
    assert actual == expected, 'trimmed conversation scope must equal missing, including its restored id echo'
    assert actual['isError'] is False
    no_match = client.call(query='r12unmatchable', conversation_id=qualified(requested), **extra)['result']
    assert no_match == json.loads(json.dumps(actual).replace('marker', 'r12unmatchable'))
    if shape == 'padded':
        assert actual == client.call(query='marker', conversation_id=qualified(conversation), **extra)['result']
    if shape == 'mixed':
        by_session = client.body(query='marker', session_id=qualified('r12-mixed-visible'), **extra)
        assert by_session['results'] and by_session['total_results'] == 1
        fanout = client.body(query='r12-mixed-visible', **extra)['other_stores'][0]
        assert any(h.get('session_id') == qualified('r12-mixed-visible') for h in fanout['hits'])


@pytest.mark.parametrize('detail', ['snippets', 'answer_ready'])
def test_r12_temporal_fields_survive_rebuild(tmp_path, corpus, clients, detail):
    local, sources = corpus
    source = sources[0]
    store = cc.open_store(source / 'lcm.db', source)
    try:
        store.connection.execute("UPDATE messages SET observed_at=1700000000, observed_at_source='host_message_timestamp' WHERE store_id=1")
        store.connection.commit()
        store.add_message_relations([[(1, 'carried_stamp', None, 0, 1700000000)]])
        assert cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    assert cc.leave_readable(source / 'lcm.db')
    generation = copy_generation(source, local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    plain = clients(tmp_path, generation, stock=True)
    args = dict(query='remote0', detail=detail, include='verbatim', limit=10)
    if detail == 'answer_ready':
        args['include_occurrence_time'] = True
    original = plain.body('lcm_recall', **args)
    local_hit = next(h for h in original['hits'] if h.get('store_id') == 1)
    hosted = client.body('lcm_recall', **args)['other_stores'][0]
    hit = next(h for h in hosted['hits'] if h.get('store_id') == qualified(1))
    assert local_hit['event_time'] == '2023-11-14T22:13:20Z' and local_hit['time_uncertain'] == 'carried'
    if detail == 'snippets':
        assert (hit.get('event_time'), hit.get('time_uncertain')) == (local_hit['event_time'], 'carried')
    else:
        # Existing resolver metadata remains dropped; compare every admitted value.
        omitted = {'reason', 'session_date', 'policy_version', 'support'}
        for key in ('occurrence_time', 'observation_time'):
            assert key in hit, 'hosted answer_ready evidence must retain its temporal objects'
            assert hit[key] == {k: v for k, v in local_hit[key].items() if k not in omitted}
            assert 'reason' not in hit[key]
        assert hit['time_uncertain'] == 'carried'
        assert 'reason' in local_hit['occurrence_time']


R13_HIDDEN = 'hiddensess'
R13_QUERY = 'quokkaswirl'


def r13_world(tmp_path, corpus, clients, mode='hits', nodes=False, conversation=False):
    """S1-S4: never-held and locally deleted replicas beside the forgotten one."""
    local, sources = corpus
    source = sources[0]
    homes, generations = {'merger': local}, {}
    for label in ('never', 'deleted'):
        home, _ = make_store(tmp_path / label, 'local')
        optin(home)
        homes[label] = home
    generations['never'] = copy_generation(source, homes['never'], COPIES[0])
    store = cc.open_store(source / 'lcm.db', source)
    try:
        tx = _transcript(tmp_path / 'hidden-transcript', [
            _entry('user', 'h1', f'{R13_QUERY} only lives here', R13_HIDDEN),
            _entry('assistant', 'h2', f'{R13_QUERY} echoed back', R13_HIDDEN)], R13_HIDDEN)
        assert cc.drain(store, cc.Lineage(R13_HIDDEN, None, tx)) == 'drained'
        hidden_ids = [r[0] for r in store.connection.execute(
            'SELECT store_id FROM messages WHERE session_id=? ORDER BY store_id', (R13_HIDDEN,))]
        assert len(hidden_ids) == 2
        if conversation:
            store.connection.execute('UPDATE messages SET conversation_id=? WHERE session_id=?', ('r13-conversation', R13_HIDDEN))
            store.connection.commit()
        leaves = {}
        if nodes:
            dag = SummaryDAG(source / 'lcm.db')
            for label, ids in ([('mixed', [1, 2] + hidden_ids), ('visible', [1, 2])] if nodes == 'mixed' else [('visible', [1, 2])]):
                leaves[label] = dag.add_node(SummaryNode(
                    session_id='remote0', summary='synthetic visible leaf', source_ids=ids,
                    source_type='messages', token_count=5, source_token_count=40),
                    escalation_level=1, model='synthetic')
            dag.close()
        assert cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    assert cc.leave_readable(source / 'lcm.db')
    forget(homes['deleted'], R13_HIDDEN)
    generations['deleted'] = copy_generation(source, homes['deleted'], COPIES[0])
    forget(local, R13_HIDDEN)
    assert R13_HIDDEN in forgotten_ids(local)
    generations['merger'] = copy_generation(source, local, COPIES[0])
    if mode == 'session':
        for generation in generations.values():
            (view_folder(generation) / '.env').write_text('LCM_RECALL_DELIVERY_UNIT=session\n')
    readers = {}
    for label, home in homes.items():
        base = tmp_path if label == 'merger' else tmp_path / label
        readers[label] = clients(base, home)
        ready(readers[label])
    readers['stock'] = clients(tmp_path, generations['merger'], stock=True)
    return SimpleNamespace(**readers, gen=generations['merger'], leaves=leaves)


R13_SCORE = re.compile(r'(^|_)score$')


def r13_normalize(value):
    """Probe comparison: every field except named timing fields and scores (r13_same compares scores)."""
    if isinstance(value, dict):
        return {k: r13_normalize(v) for k, v in value.items()
                if not re.search(r'(^|_)ms$|elapsed|latency|duration|seconds', k) and not R13_SCORE.search(k)}
    if isinstance(value, list):
        return [r13_normalize(v) for v in value]
    return value


def r13_scores(value):
    """Every numeric score field, in document order."""
    if isinstance(value, dict):
        return [s for k, v in value.items()
                for s in ([v] if R13_SCORE.search(k) and isinstance(v, (int, float)) else r13_scores(v))]
    if isinstance(value, list):
        return [s for v in value for s in r13_scores(v)]
    return []


def r13_same(actual, expected):
    """Equal apart from timing fields; scores equal within 1e-4.

    Scores are recency-adjusted against the wall clock, so two calls a moment apart can differ in the sixth
    decimal (CI run 43966940518). A changed rank moves a score by far more than the tolerance.
    """
    assert r13_normalize(actual) == r13_normalize(expected)
    got, want = r13_scores(actual), r13_scores(expected)
    assert len(got) == len(want) and all(abs(a - b) <= 1e-4 for a, b in zip(got, want)), (got, want)


@pytest.mark.parametrize('mode', ['hits', 'session', 'answer_ready', 'grep'])
def test_r13_forgotten_excluded_at_query_time(tmp_path, corpus, clients, mode):
    world = r13_world(tmp_path, corpus, clients, mode)
    tool = 'lcm_grep' if mode == 'grep' else 'lcm_recall'
    if mode == 'grep':
        ref = next((world.gen / 'lcm-large-outputs').iterdir()).name
        args = dict(externalized_refs=[qualified(ref)], content_scope='both', session_scope='all')
        stock_args = {**args, 'externalized_refs': [ref]}
        assert world.merger.body(tool, query='amber', **args)['results'], 'visible ref precheck control'
    else:
        args = dict(session_id=qualified('remote0'))
        if mode == 'answer_ready':
            args['detail'] = 'answer_ready'
        stock_args = {**args, 'session_id': 'remote0'}
    key = 'results' if mode == 'grep' else 'hits'
    stock = world.stock.body(tool, query=R13_QUERY, **stock_args)
    assert any(h.get('session_id') == R13_HIDDEN for h in stock[key])
    assert_view_readers(world.merger)
    assert not world.stock.body(tool, query='wombatplume', **stock_args)[key]
    assert world.merger.body(tool, query='marker', **args)[key], 'visible hit control'
    actual = world.merger.body(tool, query=R13_QUERY, **args)
    assert actual['total_results'] == 0, 'replica counts must exclude forgotten rows before retrieval'
    for oracle in (world.never, world.deleted):
        r13_same(actual, oracle.body(tool, query=R13_QUERY, **args))
    assert R13_HIDDEN not in json.dumps(actual), 'no hidden id or exclusion list echo'
    no_match = world.merger.body(tool, query='wombatplume', **args)
    r13_same(json.loads(json.dumps(actual).replace(R13_QUERY, 'wombatplume')), no_match)
    # A named payload from an excluded session is a stock-reader scope error;
    # use history scope to test caller exclusions with an empty success answer.
    exclude_args = dict(session_id=qualified('remote0'), session_scope='session', content_scope='history') if mode == 'grep' else args
    excluded = world.merger.body(tool, query='marker', exclude_session_ids=[qualified('remote0')], **exclude_args)
    assert not excluded[key] and excluded['total_results'] == 0, 'caller exclusion still applies'
    unchanged = world.never.body(tool, query='marker', **args)
    never_excluded = world.never.body(tool, query='marker', exclude_session_ids=[qualified(R13_HIDDEN)], **args)
    r13_same(unchanged, never_excluded)
    world.never.close()
    forget(world.never.local, R13_HIDDEN)
    (tmp_path / 'never-reopened').mkdir()
    never_hidden = clients(tmp_path / 'never-reopened', world.never.local)
    ready(never_hidden)
    r13_same(unchanged, never_hidden.body(tool, query='marker', **args))




def test_r13_expand_counts_match_local_forget(tmp_path, corpus, clients):
    world = r13_world(tmp_path, corpus, clients, nodes=True)
    label, expected_rows = 'visible', 2
    args = {'node_id': qualified(world.leaves[label])}
    actual = world.merger.body('lcm_expand', **args)
    oracle = world.deleted.body('lcm_expand', **args)
    assert_view_readers(world.merger)
    assert len(actual['expanded']) == len(oracle['expanded']) == expected_rows
    assert actual['pagination'] == oracle['pagination']
    assert actual['pagination']['returned_sources'] == len(actual['expanded'])
    stock = world.stock.body('lcm_expand', node_id=world.leaves[label])
    assert actual['pagination'] == stock['pagination']


def test_r13_recall_coverage_survives_rebuild(tmp_path, corpus, clients):
    world = r13_world(tmp_path, corpus, clients)
    args = dict(query='wombatplume', session_id=qualified('remote0'))
    expected = world.never.body('lcm_recall', **args)['provenance']['coverage']
    stock = world.stock.body('lcm_recall', query='wombatplume', session_id='remote0')
    assert stock['provenance']['coverage'] == expected
    assert_view_readers(world.merger)
    actual = world.merger.body('lcm_recall', **args)['provenance']
    assert actual.get('coverage') == expected, {
        'direct_coverage': expected, 'types': {k: type(v).__name__ for k, v in expected.items()},
        'rebuilt_coverage': actual.get('coverage')}


@pytest.mark.parametrize('tool', ['lcm_grep', 'lcm_recall'])
@pytest.mark.parametrize('case', ['messages', 'mode', 'exclude', 'healthy', 'unavailable', 'warming', 'late'])
def test_r14_fanout_forgotten_equals_missing(tmp_path, corpus, direct_mergers, monkeypatch, tool, case):
    local, sources = corpus
    hidden, missing = 'deadbeef-session', '00000000-s0ssion'
    source = sources[0]
    store = cc.open_store(source / 'lcm.db', source)
    try:
        tx = _transcript(tmp_path / 'hidden-transcript',
                         [_entry('user', 'h1', 'hidden marker', hidden)], hidden)
        assert cc.drain(store, cc.Lineage(hidden, None, tx)) == 'drained'
        assert cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    assert cc.leave_readable(source / 'lcm.db')
    # A local tombstone hides a session that the replica still holds.
    store = cc.open_store(local / 'lcm.db', local)
    try:
        tx = _transcript(tmp_path / 'local-forget', [_entry('user', 'f1', 'local copy', hidden)], hidden)
        assert cc.drain(store, cc.Lineage(hidden, None, tx)) == 'drained'
        assert cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    forget(local, hidden)
    copy_generation(source, local, COPIES[0])
    merger = direct_mergers(local, replica_wait=30)
    replica = merger.replicas[PREFIX]
    if replica.probe is not None:
        replica.probe.result(timeout=20)
    merger.refresh()
    assert replica.status == 'ok'
    sent = []
    ask = replica.child.ask
    def traced(raw):
        sent.append(json.loads(raw)['params']['arguments'])
        return ask(raw)
    monkeypatch.setattr(replica.child, 'ask', traced)
    def entry(**args):
        raw = cr.tool_request(tool, args)
        body, reply = cr.tool_body(merger.handle(raw))
        plain, _ = cr.tool_body(merger.local_call(raw))
        assert json.loads(cr.strip_other_stores(reply['result']['content'][0]['text'])) == plain
        return body['other_stores'][0]
    # Controls retain the existing arguments, exclusions, local body, and hits.
    for scope in ({'session_id': 'remote0'}, {}):
        sent.clear()
        control = entry(query='marker', session_scope='session' if scope else 'all', **scope)
        assert control['status'] == 'ok' and control['hits']
        assert all(hit['session_id'] == qualified('remote0') for hit in control['hits'])
        expected_control = {'query': 'marker', 'session_scope': 'session' if scope else 'all',
                            **scope}
        assert sent == [expected_control]
    sent.clear()
    assert entry(query='marker', session_id=hidden, externalized_refs=['local-only.json']) == {
        'copy_id': PREFIX, 'generation': 1, 'status': 'ok', 'hits': []}
    assert sent == [], 'externalized_refs still skips every replica'
    extra = {'messages': {'content_scope': 'messages'}, 'mode': {'mode': 'nonsense'},
             'exclude': {'exclude_session_ids': 'x'}}.get(case, {})
    pending = []
    if case in {'unavailable', 'warming'}:
        monkeypatch.setattr(merger, 'refresh', lambda *args: None)
        replica.status = case
    elif case == 'late':
        def synthetic(raw):
            args = json.loads(raw)['params']['arguments']
            sent.append(args)
            future = cr.Future()
            pending.append(future)
            return future
        monkeypatch.setattr(replica.child, 'ask', synthetic)
        # An unresolved Future and zero wait make lateness independent of scheduling.
        merger.replica_wait = 0
    sent.clear()
    args = dict(query='wombatplume', session_scope='session', **extra)
    expected = entry(session_id=missing, **args)
    actual = entry(session_id=hidden, **args)
    expected = json.loads(cr.replace_token(json.dumps(expected), qualified(missing), qualified(hidden)))
    assert actual == expected, 'every entry field must equal the missing-id entry after swapping ids'
    if case in {'unavailable', 'warming', 'late'} or case == 'exclude' or (tool == 'lcm_grep' and case in {'messages', 'mode'}):
        status = case if case in {'unavailable', 'warming', 'late'} else 'unavailable'
        assert actual['status'] == status and actual['hits'] == []
    else:
        assert actual['status'] == 'ok'
    if case not in {'unavailable', 'warming'}:
        assert sent == [{**args, 'session_id': missing}, {**args, 'session_id': hidden}], 'caller ids reach the view unchanged'
    if case == 'late':
        assert len(pending) == 2 and all(not future.done() for future in pending)
        for future in pending:
            future.set_result(cr.Merger.answer({'id': 1}, {'results': [], 'hits': []}))
        assert actual['status'] == 'late', 'a discarded late response cannot change the returned entry'


# The round-14 review's complete differential corpus, adapted to this module.
R15_H = '7c9e6679-7425-40de-944b-e07fc1f90ae7'
R15_M = '9b2d4e6f-1111-4222-8333-944455566677'
R15_NIL = '00000000-0000-0000-0000-000000000000'
# T1-208: realistic removed IDs avoid stock metadata-key substrings.
R15_SHORT_H, R15_SHORT_M = '20261117_120000_a1b2c3d4', 'c'
R15_OWNP = OWN.replace('-', '')
R15_P0, R15_P1 = COPIES[0].replace('-', ''), COPIES[1].replace('-', '')


def r15_drain(home, session, entries, tmp):
    store = cc.open_store(home / 'lcm.db', home)
    try:
        tx = _transcript(tmp / f'tx-{session}-{home.name}', entries, session)
        assert cc.drain(store, cc.Lineage(session, None, tx)) == 'drained'
        assert cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    assert cc.leave_readable(home / 'lcm.db')


def r15_build(tmp, nil_session=False):
    local, _ = make_store(tmp, 'local')
    (local / '.env').write_text('LCM_EMBEDDINGS_ENABLED=false\n')
    optin(local)
    src = make_store(tmp, 'remotex', text=f'needle {R15_NIL} tail')[0]
    for sid in (R15_H, R15_SHORT_H):
        r15_drain(src, sid, [_entry('user', 'h1', 'hidden needle words', sid)], tmp)
        r15_drain(local, sid, [_entry('user', 'f1', 'local copy', sid)], tmp)
        forget(local, sid)
    if nil_session:
        r15_drain(src, R15_NIL, [_entry('user', 'n1', 'nil session needle words', R15_NIL)], tmp)
    assert {R15_H, R15_SHORT_H} <= forgotten_ids(local)
    with sqlite3.connect(f'file:{src / "lcm.db"}?mode=ro&immutable=1', uri=True) as conn:
        conv = {sid: [r[0] for r in conn.execute('SELECT DISTINCT conversation_id FROM messages WHERE session_id=?', (sid,))]
                for sid in (R15_H, 'remotex')}
    copy_generation(src, local, COPIES[0])
    src2 = make_store(tmp, 'remotey')[0]
    copy_generation(src2, local, COPIES[1])
    return local, conv


def r15_swap(entry, old, new):
    text = json.dumps(entry)
    for p in (R15_P0, R15_P1):
        text = cr.replace_token(text, f'{p}:{old}', f'{p}:{new}')
    return json.loads(cr.replace_token(text, old, new))


def r15_replica_args(client):
    out = {}
    for pid in client.pids(True):
        path = client.log / f'{pid}.requests'
        if path.exists():
            out[pid] = [json.loads(line) for line in path.read_text().splitlines()]
    return out


def r15_new_calls(before, after):
    return [r['params']['arguments'] for pid, reqs in after.items() for r in reqs[len(before.get(pid, [])):]
            if r.get('method') == 'tools/call']


def r15_pairs(tmp_path, clients):
    local, conv = r15_build(tmp_path)
    client = clients(tmp_path, local)
    ready(client, 2)
    hconv = conv[R15_H][0] if conv[R15_H] else None
    pairs = {
        'grep_session': ('lcm_grep', dict(query='needle', session_scope='session', session_id=R15_H), dict(query='needle', session_scope='session', session_id=R15_M)),
        'grep_session_default_scope': ('lcm_grep', dict(query='needle', session_id=R15_H), dict(query='needle', session_id=R15_M)),
        'recall_session_id': ('lcm_recall', dict(query='needle', session_id=R15_H), dict(query='needle', session_id=R15_M)),
        'recall_session_id_nil_query': ('lcm_recall', dict(query=R15_NIL, session_id=R15_H), dict(query=R15_NIL, session_id=R15_M)),
        'recall_own_prefix': ('lcm_recall', dict(query='needle', session_id=f'{R15_OWNP}:{R15_H}'), dict(query='needle', session_id=f'{R15_OWNP}:{R15_M}')),
        'grep_own_prefix': ('lcm_grep', dict(query='needle', session_scope='session', session_id=f'{R15_OWNP}:{R15_H}'), dict(query='needle', session_scope='session', session_id=f'{R15_OWNP}:{R15_M}')),
        'recall_short_hex': ('lcm_recall', dict(query='needle', session_id=R15_SHORT_H), dict(query='needle', session_id=R15_SHORT_M)),
        'grep_short_hex': ('lcm_grep', dict(query='needle', session_scope='session', session_id=R15_SHORT_H), dict(query='needle', session_scope='session', session_id=R15_SHORT_M)),
        'grep_padded': ('lcm_grep', dict(query='needle', session_scope='session', session_id=f' {R15_H} '), dict(query='needle', session_scope='session', session_id=f' {R15_M} ')),
        'recall_padded': ('lcm_recall', dict(query='needle', session_id=f' {R15_H} '), dict(query='needle', session_id=f' {R15_M} ')),
        'grep_excl_self': ('lcm_grep', dict(query='needle', session_scope='session', session_id=R15_H, exclude_session_ids=[R15_H]), dict(query='needle', session_scope='session', session_id=R15_M, exclude_session_ids=[R15_M])),
        'recall_excl_self': ('lcm_recall', dict(query='needle', session_id=R15_H, exclude_session_ids=[R15_H]), dict(query='needle', session_id=R15_M, exclude_session_ids=[R15_M])),
        'grep_excl_standin': ('lcm_grep', dict(query='needle', session_scope='session', session_id=R15_H, exclude_session_ids=[R15_NIL]), dict(query='needle', session_scope='session', session_id=R15_M, exclude_session_ids=[R15_NIL])),
        'grep_all_excl_forgotten': ('lcm_grep', dict(query='needle', session_scope='all', exclude_session_ids=[R15_H]), dict(query='needle', session_scope='all', exclude_session_ids=[R15_M])),
        'recall_excl_forgotten': ('lcm_recall', dict(query='needle', exclude_session_ids=[R15_H]), dict(query='needle', exclude_session_ids=[R15_M])),
        'grep_conv': ('lcm_grep', dict(query='needle', session_scope='all', conversation_id=hconv or R15_H), dict(query='needle', session_scope='all', conversation_id='missingconv')),
        'grep_limit1': ('lcm_grep', dict(query='needle', session_scope='session', session_id=R15_H, limit=1), dict(query='needle', session_scope='session', session_id=R15_M, limit=1)),
        'grep_int_like': ('lcm_grep', dict(query='needle', session_scope='session', session_id=7), dict(query='needle', session_scope='session', session_id=8)),
    }
    out = {'conv': conv}
    for label, (tool, fa, ma) in pairs.items():
        before = r15_replica_args(client)
        f = client.body(tool, **fa).get('other_stores')
        mid = r15_replica_args(client)
        sent_f = r15_new_calls(before, mid)
        m = client.body(tool, **ma).get('other_stores')
        sent_m = r15_new_calls(mid, r15_replica_args(client))
        old = ma.get('session_id') if isinstance(ma.get('session_id'), str) else None
        new = fa.get('session_id') if isinstance(fa.get('session_id'), str) else None
        ms = m
        if old and new:
            o, n = old.strip().split(':')[-1], new.strip().split(':')[-1]
            ms = r15_swap(m, o, n)
        diffs = []
        for fe, me in zip(f or [], ms or []):
            for k in sorted(set(fe) | set(me)):
                if fe.get(k) != me.get(k):
                    diffs.append({'copy': fe.get('copy_id'), 'key': k, 'forgotten': fe.get(k), 'missing_swapped': me.get(k)})
        out[label] = {'equal': f == ms, 'diffs': diffs, 'forgotten': f, 'missing': m,
                      'status_f': [e['status'] for e in f or []], 'status_m': [e['status'] for e in m or []],
                      'hits_f': [len(e['hits']) for e in f or []], 'hits_m': [len(e['hits']) for e in m or []],
                      'sent_f_session_ids': [a.get('session_id') for a in sent_f], 'sent_m_session_ids': [a.get('session_id') for a in sent_m],
                      'sent_f_conv': [a.get('conversation_id') for a in sent_f],
                      'forgotten_id_as_session_arg': any(isinstance(a.get('session_id'), str) and a['session_id'].strip() in (R15_H, R15_SHORT_H) for a in sent_f)}
    return out


def r15_two_stores_one_down(tmp_path, direct_mergers, monkeypatch):
    local, _ = r15_build(tmp_path)
    merger = direct_mergers(local, replica_wait=30)
    for r in merger.replicas.values():
        if r.probe is not None:
            r.probe.result(timeout=20)
    merger.refresh()
    assert all(r.status == 'ok' for r in merger.replicas.values())
    monkeypatch.setattr(merger, 'refresh', lambda *args: None)
    merger.replicas[R15_P1].status = 'unavailable'
    out = {}
    for tool, extra in (('lcm_grep', {'session_scope': 'session'}), ('lcm_recall', {})):
        res = {}
        for sid in (R15_H, R15_M):
            body, _ = cr.tool_body(merger.handle(cr.tool_request(tool, dict(query='needle', session_id=sid, **extra))))
            res[sid] = body['other_stores']
        ms = r15_swap(res[R15_M], R15_M, R15_H)
        out[tool] = {'equal': res[R15_H] == ms, 'forgotten': res[R15_H], 'missing': res[R15_M]}
    return out


def r15_standin_is_visible_session(tmp_path, clients):
    local, _ = r15_build(tmp_path, nil_session=True)
    client = clients(tmp_path, local)
    ready(client, 2)
    out = {}
    for tool, extra in (('lcm_grep', {'session_scope': 'session'}), ('lcm_recall', {})):
        f = client.body(tool, query='needle', session_id=R15_H, **extra)['other_stores']
        m = client.body(tool, query='needle', session_id=R15_M, **extra)['other_stores']
        out[tool] = {'equal': f == r15_swap(m, R15_M, R15_H), 'forgotten': f, 'missing': m}
    return out


def r15_extra_shapes(tmp_path, clients):
    local, _ = make_store(tmp_path, 'local')
    (local / '.env').write_text('LCM_EMBEDDINGS_ENABLED=false\n')
    optin(local)
    src = make_store(tmp_path, 'remotex', text='needle visible tail')[0]
    for sid in ('work', '12345'):
        r15_drain(src, sid, [_entry('user', 'h1', 'hidden needle words', sid)], tmp_path)
        r15_drain(local, sid, [_entry('user', 'f1', 'local copy', sid)], tmp_path)
        forget(local, sid)
    assert {'work', '12345'} <= forgotten_ids(local)
    copy_generation(src, local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client, 1)
    unknown = 'f' * 32
    pairs = {
        'grep_nohex': ('lcm_grep', dict(query='needle', session_scope='session', session_id='work'), dict(query='needle', session_scope='session', session_id='nope')),
        'recall_nohex': ('lcm_recall', dict(query='needle', session_id='work'), dict(query='needle', session_id='nope')),
        'grep_int_forgotten': ('lcm_grep', dict(query='needle', session_scope='session', session_id=12345), dict(query='needle', session_scope='session', session_id=67890)),
        'grep_str_digits_forgotten': ('lcm_grep', dict(query='needle', session_scope='session', session_id='12345'), dict(query='needle', session_scope='session', session_id='67890')),
        'recall_int_forgotten': ('lcm_recall', dict(query='needle', session_id=12345), dict(query='needle', session_id=67890)),
        'grep_unknown_prefix': ('lcm_grep', dict(query='needle', session_scope='session', session_id=f'{unknown}:work'), dict(query='needle', session_scope='session', session_id=f'{unknown}:nope')),
    }
    out = {}
    for label, (tool, fa, ma) in pairs.items():
        before = r15_replica_args(client)
        f = client.body(tool, **fa).get('other_stores')
        mid = r15_replica_args(client)
        m = client.body(tool, **ma).get('other_stores')
        after = r15_replica_args(client)
        o, n = str(ma['session_id']).split(':')[-1], str(fa['session_id']).split(':')[-1]
        ms = r15_swap(m, o, n)
        out[label] = {'equal': f == ms, 'forgotten': f, 'missing': m,
                      'sent_f': [a.get('session_id') for a in r15_new_calls(before, mid)],
                      'sent_m': [a.get('session_id') for a in r15_new_calls(mid, after)]}
    return out


def test_r15_reviewer_pairs(tmp_path, clients, direct_mergers, monkeypatch):
    # Keep all reviewer pairs, including the two explicitly deferred classes.
    groups = {
        'pairs': r15_pairs(tmp_path / 'pairs', clients),
        'extra': r15_extra_shapes(tmp_path / 'extra', clients),
        'two_stores': r15_two_stores_one_down(tmp_path / 'two', direct_mergers, monkeypatch),
        'standin_visible': r15_standin_is_visible_session(tmp_path / 'collision', clients),
    }
    actual = {(group, label): result['equal'] for group, rows in groups.items()
              for label, result in rows.items() if label != 'conv'}
    expected = dict.fromkeys(actual, True)
    assert len(actual) == 28
    assert actual == expected, 'every forgotten/missing reviewer pair must agree'


@pytest.mark.parametrize('tool', ['lcm_grep', 'lcm_recall'])
def test_r15_standin_text_unchanged(tmp_path, direct_mergers, monkeypatch, tool):
    local, _ = r15_build(tmp_path)
    merger = direct_mergers(local, replica_wait=30)
    replica = merger.replicas[PREFIX]
    for r in merger.replicas.values():
        if r.probe is not None:
            r.probe.result(timeout=20)
    merger.refresh()
    assert all(r.status == 'ok' for r in merger.replicas.values())
    # Use a real visible grep hit; replay it to pin rebuilding independently of scope.
    future = replica.child.ask(cr.tool_request('lcm_grep', {'query': 'needle', 'session_scope': 'all'}))
    stock, _ = cr.tool_body(future.result(timeout=20))
    hit = next(hit for hit in stock['results'] if R15_NIL in hit['snippet'])
    original = hit['snippet']
    def replay(raw):
        future = cr.Future()
        future.set_result(cr.Merger.answer({'id': 1}, {'results': [hit], 'hits': [hit]}))
        return future
    monkeypatch.setattr(replica.child, 'ask', replay)
    def entries(sid):
        body, _ = cr.tool_body(merger.handle(cr.tool_request(tool, {'query': 'needle', 'session_id': sid})))
        return body['other_stores']
    missing, forgotten = entries(R15_M), entries(R15_H)
    assert json.dumps(forgotten).encode() == json.dumps(missing).encode(), 'entry bytes must match'
    assert forgotten[0]['hits'][0]['snippet'] == original
    assert R15_NIL in original






# Part B: view construction, rather than argument-specific forgetting.
def b_world(tmp_path, *, lineage=False, forget_hidden=True):
    local, _ = make_store(tmp_path, 'local')
    (local / '.env').write_text('LCM_EMBEDDINGS_ENABLED=false\n')
    optin(local)
    source, _ = make_store(tmp_path, R15_H)
    r15_drain(source, 'visible', [_entry('user', 'v1', 'visible marker words', 'visible')], tmp_path)
    if lineage:
        with sqlite3.connect(source / 'lcm.db') as conn:
            conn.execute('INSERT INTO metadata VALUES (?, ?)', ('rotation_predecessor_session:visible', json.dumps(R15_H)))
            conn.execute('CREATE TABLE IF NOT EXISTS session_lineage (child_session_id TEXT PRIMARY KEY, parent_session_id TEXT, child_subagent_id TEXT, parent_subagent_id TEXT, child_role TEXT, source TEXT, created_at REAL)')
            conn.execute("INSERT INTO session_lineage VALUES ('visible', ?, NULL, NULL, NULL, 'subagent', 1700000000)", (R15_H,))
    if forget_hidden:
        forget(local, R15_H)
    generation = copy_generation(source, local, COPIES[0])
    ref = next((generation / 'lcm-large-outputs').iterdir()).name
    return local, source, generation, ref


def b_equal(client, tool, hidden, missing, swaps):
    actual = client.call(tool, **hidden)['result']
    expected = client.call(tool, **missing)['result']
    body = json.loads(expected['content'][0]['text'])
    for old, new in swaps:
        body = json.loads(json.dumps(body).replace(str(old), str(new)))
    got = json.loads(actual['content'][0]['text'])
    r13_same(got, body)
    assert actual.get('isError') == expected.get('isError')
    return got


B1_SHAPES = [(pad, shape) for pad in (' {}', '{} ', '\t{}\n', ' {} ') for shape in range(9)] + [('\u00a0{}', 0), ('{}\u00a0', 1)]


@pytest.mark.parametrize('pad,shape', B1_SHAPES)
def test_b1_padded_sessions(tmp_path, clients, pad, shape):
    local, _, _, ref = b_world(tmp_path)
    client = clients(tmp_path, local)
    ready(client)
    h, m = qualified(pad.format(R15_H)), qualified(pad.format(R15_M))
    pairs = [
        ('lcm_describe', {}, {}), ('lcm_load_session', {}, {}),
        ('lcm_grep', {'query': 'marker', 'session_scope': 'session'}, {'query': 'marker', 'session_scope': 'session'}),
        ('lcm_load_session', {'after_store_id': qualified(1)}, {'after_store_id': qualified(1)}),
        ('lcm_expand', {'node_id': qualified(1)}, {'node_id': qualified(99999)}),
        ('lcm_expand', {'node_id': qualified(1)}, {'node_id': qualified(1)}),
        ('lcm_describe', {'node_id': qualified(1)}, {'node_id': qualified(99999)}),
        ('lcm_expand', {'externalized_ref': qualified(ref)}, {'externalized_ref': qualified('missing.json')}),
        ('lcm_expand', {'node_id': qualified(1)}, {'node_id': qualified(99999)}),
    ]
    tool, ha, ma = pairs[shape]
    if shape == 8:
        h, m = pad.format(R15_H), pad.format(R15_M)
    b_equal(client, tool, dict(ha, session_id=h), dict(ma, session_id=m),
            [(R15_M, R15_H), ('99999', '1'), ('missing.json', ref)])


@pytest.mark.parametrize('pad', [' {}', '{} ', '\t{}\n', '{}\u00a0'])
@pytest.mark.parametrize('tool', ['lcm_expand', 'lcm_describe'])
def test_b2_padded_refs(tmp_path, clients, pad, tool):
    local, _, _, ref = b_world(tmp_path)
    client = clients(tmp_path, local)
    ready(client)
    b_equal(client, tool, {'externalized_ref': qualified(pad.format(ref))},
            {'externalized_ref': qualified(pad.format('missing.json'))}, [('missing.json', ref)])


def test_b3_empty_conversation(tmp_path, clients):
    local, _, _, _ = b_world(tmp_path)
    client = clients(tmp_path, local)
    ready(client)
    body = client.body(query='visible', conversation_id=qualified(''), session_scope='all')
    assert body.get('results') and body['results'][0]['session_id'] == qualified('visible')
    assert not client.body(query='visible', conversation_id=qualified('missing'), session_scope='all')['results']


def test_b4_arguments_unchanged(tmp_path, clients):
    local, _, _, _ = b_world(tmp_path)
    forget(local, 'work')
    cv.build(local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    before = r15_replica_args(client)
    h = qualified(' ' + R15_H + ' ')
    client.body('lcm_describe', session_id=h)
    sent = r15_new_calls(before, r15_replica_args(client))
    assert sent == [{'session_id': ' ' + R15_H + ' '}]
    assert R15_NIL not in json.dumps(sent)
    body = b_equal(client, 'lcm_load_session', {'session_id': qualified('work')},
                   {'session_id': qualified('nope')}, [('nope', 'work')])
    assert R15_NIL not in json.dumps(body)


def test_b5_mixed_summary_refuses(tmp_path, corpus, direct_mergers):
    # Do not start clients for this refused world: inspect the builder and each merger path.
    local, source = corpus[0], corpus[1][0]
    r15_drain(source, R13_HIDDEN, [_entry('user', 'h1', 'hidden marker', R13_HIDDEN)], tmp_path)
    with sqlite3.connect(source / 'lcm.db') as conn:
        hidden = conn.execute('SELECT MAX(store_id) FROM messages').fetchone()[0]
    dag = SummaryDAG(source / 'lcm.db')
    mixed = dag.add_node(SummaryNode(session_id='remote0', summary='mixed marker', source_ids=[1, hidden], source_type='messages', token_count=3, source_token_count=4))
    dag.close()
    store = cc.open_store(source / 'lcm.db', source)
    assert cc.to_rollback_journal(store.connection)
    store.close()
    assert cc.leave_readable(source / 'lcm.db')
    forget(local, R13_HIDDEN)
    generation = copy_generation(source, local, COPIES[0])
    result = json.loads((generation.parent / 'view-result.json').read_text())
    assert result['code'] == 'view_summary_spans_forgotten'
    merger = direct_mergers(local)
    replica = merger.replicas[PREFIX]
    assert replica.child is None and replica.pending is None, 'a refused view must start no reader'
    for tool, args in [('lcm_grep', {'query': 'marker'}), ('lcm_recall', {'query': 'marker'}),
                       ('lcm_expand', {'node_id': qualified(mixed)}),
                       ('lcm_describe', {'session_id': qualified('remote0')}),
                       ('lcm_load_session', {'session_id': qualified('remote0')})]:
        answer, _ = cr.tool_body(merger.handle(cr.tool_request(tool, args)))
        assert 'expanded' not in answer
        if tool in ('lcm_grep', 'lcm_recall'):
            assert answer['other_stores'][0]['status'] == 'warming'
            assert answer['other_stores'][0]['hits'] == []
        else:
            assert answer['error'] == 'the other store could not answer this call'
    assert replica.child is None and replica.pending is None


def test_b5_partial_and_search_error(tmp_path, corpus, direct_mergers, monkeypatch):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    merger = direct_mergers(local)
    replica = b_ready(merger)
    partial = {'payloads_searched': 0, 'payloads_total': 1, 'reasons': ['byte_cap'], 'prefix_bytes': 100}
    body = {'partial': partial, 'results': [], 'hits': []}
    def ask(raw):
        future = cr.Future()
        future.set_result(cr.Merger.answer({'id': 1}, body))
        return future
    monkeypatch.setattr(replica.child, 'ask', ask)
    for session in (R15_H, R15_M):
        routed, _ = cr.tool_body(merger.handle(cr.tool_request('lcm_grep', {'query': 'marker', 'session_id': qualified(session)})))
        fan, _ = cr.tool_body(merger.handle(cr.tool_request('lcm_grep', {'query': 'marker', 'session_id': session})))
        assert routed.get('partial') == partial and fan['other_stores'][0].get('partial') == partial
    body.update(message_search_error='private exception marker', results=[{'session_id': 'remote0', 'snippet': 'unsafe marker'}])
    routed, _ = cr.tool_body(merger.handle(cr.tool_request('lcm_grep', {'query': 'marker', 'session_id': qualified('remote0')})))
    fan, _ = cr.tool_body(merger.handle(cr.tool_request('lcm_grep', {'query': 'marker'})))
    assert 'private exception marker' not in json.dumps([routed, fan])
    assert fan['other_stores'][0]['status'] == 'unavailable' and fan['other_stores'][0]['hits'] == []
    assert 'error' in routed and 'unsafe marker' not in json.dumps(routed)


def test_b6_payload_scan_counts(tmp_path, clients):
    local, _, generation, _ = b_world(tmp_path)
    client = clients(tmp_path, local)
    ready(client)
    args = {'query': 'amber', 'content_scope': 'externalized', 'session_scope': 'session'}
    body = b_equal(client, 'lcm_grep', dict(args, session_id=qualified(R15_H)),
                   dict(args, session_id=qualified(R15_M)), [(R15_M, R15_H)])
    assert isinstance(body.get('externalized_scan'), dict), 'a discovery scan must return its counts'
    # The oracle was born with only V; H and H's payload have never been here.
    oracle = tmp_path / 'oracle'
    db = cc.admit(oracle, hermes_home=tmp_path / 'unused')
    store = cc.open_store(db, oracle, LCMConfig(database_path=str(db), embeddings_enabled=False))
    tx = _transcript(tmp_path / 'oracle-tx', [_entry('user', 'v1', 'visible marker words', 'visible')], 'visible')
    assert cc.drain(store, cc.Lineage('visible', None, tx)) == 'drained'
    assert cc.to_rollback_journal(store.connection)
    store.close()
    assert cc.leave_readable(db)
    (oracle / 'lcm-large-outputs').mkdir(exist_ok=True)
    stock = clients(tmp_path, oracle, stock=True)
    plain = stock.body('lcm_grep', **args, session_id=R15_H)
    assert body['externalized_scan'] == plain['externalized_scan']
    assert not list((view_folder(generation) / 'lcm-large-outputs').iterdir())


def test_b7_lineage(tmp_path, clients):
    local, _, _, ref = b_world(tmp_path, lineage=True)
    client = clients(tmp_path, local)
    ready(client)
    for session in ('visible', 'visible '):
        b_equal(client, 'lcm_expand', {'externalized_ref': qualified(ref), 'session_id': qualified(session)},
                {'externalized_ref': qualified('missing.json'), 'session_id': qualified(session)}, [('missing.json', ref)])
    calls = [('lcm_grep', {'query': 'visible', 'session_scope': 'all'}), ('lcm_recall', {'query': 'visible'}),
             ('lcm_grep', {'query': 'visible', 'session_scope': 'session', 'session_id': qualified('visible')}),
             ('lcm_describe', {'session_id': qualified('visible')}), ('lcm_load_session', {'session_id': qualified('visible')})]
    for tool, args in calls:
        body = client.body(tool, **args)
        assert R15_H not in json.dumps(body)
        assert 'visible' in json.dumps(body)


def b_ready(merger, prefix=PREFIX):
    def ready_now():
        merger.refresh()
        return merger.replicas[prefix].status == 'ok'
    eventually(ready_now, timeout=40)
    return merger.replicas[prefix]


@pytest.mark.parametrize('change', ['forgotten', 'removed', 'rules', 'format', 'folder', 'digest'])
def test_b8_stale_view_never_called(tmp_path, direct_mergers, monkeypatch, change):
    local, _, generation, _ = b_world(tmp_path, forget_hidden=False)
    merger = direct_mergers(local)
    replica = b_ready(merger)
    old = replica.child
    sent = []
    original = old.ask
    def traced(raw):
        sent.append(raw)
        return original(raw)
    monkeypatch.setattr(old, 'ask', traced)
    pointer_path = generation.parent / 'view.json'
    pointer = json.loads(pointer_path.read_text())
    if change == 'forgotten':
        forget(local, R15_H)
    elif change == 'removed':
        (generation.parent / 'removed.json').write_text(json.dumps({'format': 1, 'session_ids': [R15_H], 'row_ids': []}))
    else:
        pointer[{'digest': 'forgotten_sha256'}.get(change, change)] = {'rules': 77, 'format': 2, 'folder': '../gen-1', 'digest': 'wrong'}[change]
        pointer_path.write_text(json.dumps(pointer))
    builds = []
    real_popen = cr.subprocess.Popen
    class Blocked:
        def poll(self): return None
    def popen(argv, **kwargs):
        if 'view' in argv:
            builds.append((argv, kwargs))
            return Blocked()
        return real_popen(argv, **kwargs)
    monkeypatch.setattr(cr.subprocess, 'Popen', popen)
    for _ in range(2):
        for tool in TOOLS:
            args = {'query': 'marker'} if tool in ('lcm_grep', 'lcm_recall') else {'session_id': qualified('visible')}
            raw = cr.tool_request(tool, args)
            body, reply = cr.tool_body(merger.handle(raw))
            if tool in ('lcm_grep', 'lcm_recall'):
                assert body['other_stores'][0]['status'] == 'warming'
                plain = json.loads(merger.local_call(raw))['result']['content'][0]['text']
                assert cr.strip_other_stores(reply['result']['content'][0]['text']) == plain
            else:
                assert 'error' in body
    assert not sent and len(builds) == 1
    argv, kwargs = builds[0]
    assert argv == [sys.executable, '-I', '-S', '-B', str(ROOT / 'scripts/lcm_x_cloud.py'), 'view', '--home', str(local), '--copy-id', COPIES[0]]
    assert kwargs['start_new_session'] is True
    assert all(kwargs[key] == subprocess.DEVNULL for key in ('stdin', 'stdout', 'stderr'))
    # Publish the repaired view: no call to the stale reader during warm-up either.
    assert cv.build(local, COPIES[0])['code'] == 'ok'
    b_ready(merger)
    assert not sent
    assert replica.child.folder == view_folder(generation)


@pytest.mark.parametrize('code,age,retry', [('view_unsupported_table', 500, False), ('view_unsupported_key', 500, False),
                                           ('view_summary_spans_forgotten', 500, False), ('view_busy', 59, False),
                                           ('view_busy', 61, True), ('view_disk_floor', 59, False), ('view_disk_floor', 61, True)])
def test_b8_build_result_retry(tmp_path, direct_mergers, monkeypatch, code, age, retry):
    local, _, generation, _ = b_world(tmp_path)
    folder = generation.parent
    now = time.time()
    monkeypatch.setattr(time, 'time', lambda: now)
    record = json.loads((folder / 'view-result.json').read_text())
    (folder / 'view.json').unlink()
    record.update(code=code, ended_at=time.time()-age)
    (folder / 'view-result.json').write_text(json.dumps(record))
    builds = []
    real_popen = cr.subprocess.Popen
    class Ended:
        def poll(self): return 2
    def popen(argv, **kwargs):
        if 'view' in argv:
            builds.append(argv)
            # An actual builder writes the result before it exits.
            record['ended_at'] = time.time()
            (folder / 'view-result.json').write_text(json.dumps(record))
            return Ended()
        return real_popen(argv, **kwargs)
    monkeypatch.setattr(cr.subprocess, 'Popen', popen)
    merger = direct_mergers(local)
    for _ in range(3):
        merger.handle(cr.tool_request('lcm_grep', {'query': 'marker'}))
    assert len(builds) == int(retry)
    record['rules'] += 1
    (folder / 'view-result.json').write_text(json.dumps(record))
    merger.handle(cr.tool_request('lcm_grep', {'query': 'marker'}))
    assert len(builds) == int(retry) + 1
    assert merger.replicas[PREFIX].status == 'warming'


def test_b8_digest_contract(tmp_path, corpus):
    local, sources = corpus
    generation = copy_generation(sources[0], local, COPIES[0])
    ids = {'0b1c2d3e-1111-4222-8333-444455556666', '3f2a1b0c-5555-4666-8777-888899990000'}
    for session in ids:
        forget(local, session)
    result = cr.forgotten(local / 'lcm.db')
    assert isinstance(result, tuple), 'forgotten must return the set and its F13 digest'
    assert result == (ids, 'd32b75cc4f118d8e709d786a1d10998880f9517537e7df297302a09905e6f88c')
    assert cr.RULES == cv.RULES
    assert cr.FORGOTTEN_QUERY == cv.FORGOTTEN_QUERY
    assert cr.forgotten_digest(set()) == cv.forgotten_digest(set())
    for raw in (None, b'{"format":1,"session_ids":[],"row_ids":[]}\n'):
        if raw is not None:
            (generation.parent / 'removed.json').write_bytes(raw)
        assert cr.removed_sha256(generation.parent) == cv.removed_sha256(generation.parent)


def test_b10_vanished_view(tmp_path, direct_mergers, monkeypatch):
    local, _, generation, _ = b_world(tmp_path)
    copy_generation(make_store(tmp_path, 'other')[0], local, COPIES[1])
    merger = direct_mergers(local)
    b_ready(merger)
    b_ready(merger, COPIES[1].replace('-', ''))
    # Force a new folder, so its lock is acquired after reading a fresh pointer.
    old_pointer = json.loads((generation.parent / 'view.json').read_text())
    target = generation.parent / old_pointer['folder'].replace('view-1-', 'view-2-')
    shutil.copytree(view_folder(generation), target)
    shutil.copytree(generation, generation.parent / 'gen-2')
    (generation.parent / 'current').write_text('2\n')
    pointer = dict(old_pointer, folder=target.name, generation=2)
    (generation.parent / 'view.json').write_text(json.dumps(pointer))
    opened = []
    original = cr.os.open
    def vanish(path, flags, *args, **kwargs):
        if Path(path) == target:
            opened.append(flags)
            subprocess.run([sys.executable, '-c', 'import shutil,sys; shutil.rmtree(sys.argv[1])', str(target)], check=True)
        return original(path, flags, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(cr.os, 'open', vanish)
        raw = cr.tool_request('lcm_grep', {'query': 'marker'})
        body, reply = cr.tool_body(merger.handle(raw))
    assert body['other_stores'][0]['status'] == 'warming'
    assert body['other_stores'][1]['status'] == 'ok' and body['other_stores'][1]['hits']
    plain = json.loads(merger.local_call(raw))['result']['content'][0]['text']
    assert cr.strip_other_stores(reply['result']['content'][0]['text']) == plain
    assert len(opened) == 1 and opened[0] & os.O_ACCMODE == os.O_RDONLY
    assert cv.build(local, COPIES[0])['code'] == 'ok'
    eventually(lambda: b_ready(merger).child.folder == view_folder(generation), timeout=40)


def test_b11_view_lock_lifetime(tmp_path, corpus, direct_mergers):
    local, sources = corpus
    generation = copy_generation(sources[0], local, COPIES[0])
    merger = direct_mergers(local)
    replica = b_ready(merger)
    x = view_folder(generation)
    fd = os.open(x, os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A changed generation builds Y while X is still held by the reader.
        gen2 = generation.parent / 'gen-2'
        shutil.copytree(generation, gen2)
        (generation.parent / 'current').write_text('2\n')
        phases = []
        def seam(name):
            if name in ('matched', 'published'):
                phases.append(name)
                assert x.exists(), 'sweep reclaimed a serving view'
        assert cv.build(local, COPIES[0], seam=seam)['code'] == 'ok'
        y = view_folder(gen2)
        assert phases == ['matched', 'published'] and y != x and x.exists()
        old = replica.child
        eventually(lambda: b_ready(merger).child.folder == y, timeout=40)
        assert old.process.poll() is not None
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert cv.build(local, COPIES[0])['code'] == 'ok'
        assert not x.exists() and y.exists()
    finally:
        os.close(fd)


@pytest.mark.parametrize('replacement', ['gone', 'symlink', 'different'])
def test_b10_lock_before_trust(tmp_path, monkeypatch, replacement):
    folder = tmp_path / 'view'
    folder.mkdir()
    (folder / 'lcm.db').write_bytes(b'fixture')
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'lcm.db').write_bytes(b'fixture')
    original = cr.fcntl.flock
    def mutate(fd, flags):
        original(fd, flags)
        folder.rename(tmp_path / 'moved')
        if replacement == 'symlink':
            folder.symlink_to(other, target_is_directory=True)
        elif replacement == 'different':
            folder.mkdir()
            (folder / 'lcm.db').write_bytes(b'fixture')
    monkeypatch.setattr(cr.fcntl, 'flock', mutate)
    child = None
    try:
        with pytest.raises((OSError, ValueError)):
            child = cr.Child([sys.executable, '-c', 'import time; time.sleep(60)'], 8, folder)
    finally:
        if child:
            child.close()


def test_b8_real_build_child(tmp_path, clients):
    local, _, generation, _ = b_world(tmp_path)
    (generation.parent / 'view.json').unlink()
    (generation.parent / 'view-result.json').unlink()
    client = clients(tmp_path, local)
    # The isolated builder's reply never enters the MCP stream. All calls parse as JSON-RPC.
    for _ in range(3):
        reply = client.call(query='marker')
        assert_local(client, reply)
        body = json.loads(reply['result']['content'][0]['text'])
        assert body['other_stores'][0]['status'] in ('warming', 'ok')
    eventually(lambda: (generation.parent / 'view.json').is_file(), timeout=40)
    ready(client)
    assert_view_readers(client)


def test_b8_local_read_failure(tmp_path, corpus, direct_mergers, monkeypatch):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    merger = direct_mergers(local)
    replica = b_ready(merger)
    assert replica.child.folder.name.startswith('view-')
    sent, reads = [], []
    ask, read = replica.child.ask, cr.forgotten
    def traced(raw):
        sent.append(raw)
        return ask(raw)
    def forgotten(database):
        reads.append(database)
        return read(database)
    monkeypatch.setattr(replica.child, 'ask', traced)
    monkeypatch.setattr(cr, 'forgotten', forgotten)
    merger.handle(cr.tool_request('lcm_grep', {'query': 'marker'}))
    assert len(reads) == 1, 'one forgotten snapshot per call'
    sent.clear()
    with (local / 'lcm.db').open('r+b') as handle:
        handle.seek(18)
        handle.write(b'\x02\x02')
    body, _ = cr.tool_body(merger.handle(cr.tool_request('lcm_grep', {'query': 'marker'})))
    assert body['other_stores'][0]['status'] == 'unavailable' and not sent
    body, _ = cr.tool_body(merger.handle(cr.tool_request('lcm_describe', {'session_id': qualified('remote0')})))
    assert 'error' in body and not sent


def test_b_r2_short_id_refuses_view(tmp_path):
    # T1-208: short IDs still fail closed when they collide with stock keys.
    local, _ = make_store(tmp_path, 'local')
    source, _ = make_store(tmp_path, 'a')
    r15_drain(local, 'a', [_entry('user', 'f1', 'local copy', 'a')], tmp_path)
    forget(local, 'a')
    assert 'a' in forgotten_ids(local)
    source_bytes = (source / 'lcm.db').read_bytes()
    with sqlite3.connect(f'file:{source / "lcm.db"}?mode=ro', uri=True) as conn:
        assert conn.execute("SELECT key FROM metadata WHERE key='schema_version'").fetchone()
    generation = copy_generation(source, local, COPIES[0])
    result = cv.build(local, COPIES[0])
    assert result['code'] == 'view_unsupported_key'
    assert not (generation.parent / 'view.json').exists()
    assert (source / 'lcm.db').read_bytes() == source_bytes
    assert (generation / 'lcm.db').read_bytes() == source_bytes
