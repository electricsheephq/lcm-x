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
import time
from types import SimpleNamespace
import uuid

import pytest

from hermes_lcm import claude_code_store as cc
from hermes_lcm import cloud_reader as cr
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
        replica = home.name.startswith('gen-')
        cfg = settings()
        if replica and cfg.get('stall') and request['method'] == 'tools/call':
            time.sleep(60)
        if request['method'] == 'initialize': result = {'serverInfo': {'name': 'fake'}}
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
    return folder


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
                and (replica is None or p.read_text().endswith('gen-1') == replica)]

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
    client.body('lcm_recall', query='marker', seen_refs=['lcm:1:0-5', qualified('lcm:2:0-5')],
                exclude_session_ids=['bare-session', qualified('remote0')])
    requests = []
    for path in client.log.glob('*.requests'):
        home = (client.log / (path.stem + '.pid')).read_text()
        args = json.loads(path.read_text().splitlines()[-1])['params']['arguments']
        requests.append((home, args))
    for home, args in requests:
        if 'gen-1' not in home:
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
    calls += [('lcm_expand', {'store_id': 1, 'include_exact_ref': True}),
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
    unknown = [path for path in paths if path[-1] not in cr.CLASSIFIED | cr.DROPPED and not re.fullmatch(r'd\d+', path[-1])]
    assert unknown == [], sorted(unknown)
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
    folder = copy_generation(sources[0], local, COPIES[0])
    (folder / 'lcm.db').write_bytes(b'corrupt')
    root = folder.parent.parent
    for name, pointer in [(COPIES[1], '2'), (COPIES[2], 'letters'), ('invalid', '1'), (OWN, '1')]:
        copy = root / name
        copy.mkdir()
        (copy / 'current').write_text(pointer)
    symlink_id = str(uuid.UUID(int=44))
    linked = root / symlink_id
    linked.mkdir()
    (linked / 'gen-1').symlink_to(sources[0], target_is_directory=True)
    (linked / 'current').write_text('1')
    client = clients(tmp_path, local)
    reply = client.call(query='marker')
    assert_local(client, reply)
    entries = json.loads(reply['result']['content'][0]['text'])['other_stores']
    assert all(e['status'] in ('unavailable', 'warming') and e['hits'] == [] for e in entries)
    assert all(e['copy_id'] not in ('invalid', OWN.replace('-', '')) for e in entries)
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


def test_16_startup_fallback(tmp_path, corpus, clients):
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


def test_17_generation_change(tmp_path, corpus, clients):
    local, sources = corpus
    old = copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    pid = client.pids(True)[0]
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
    assert old.exists()


def test_18_folder_lock(tmp_path, corpus, clients):
    local, sources = corpus
    generation = copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local)
    ready(client)
    fd = os.open(generation, os.O_RDONLY)
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


def test_21_fail_closed(tmp_path, corpus, clients):
    local, sources = corpus
    copy_generation(sources[0], local, COPIES[0])
    client = clients(tmp_path, local, fake=True)
    ready(client)
    for mode in ('missing', 'empty'):
        client.control(**{mode: True})
        assert client.body(query='marker')['other_stores'][0]['hits']
    store = cc.open_store(local / 'lcm.db', local)
    store.delete_session_messages('local')
    cc.to_rollback_journal(store.connection)
    store.close()
    cc.leave_readable(local / 'lcm.db')
    for mode in ('missing', 'empty'):
        client.control(**{mode: True})
        assert client.body(query='marker')['other_stores'][0]['hits'] == []
    database = local / 'lcm.db'
    with database.open('r+b') as handle:
        handle.seek(18)
        handle.write(b'\x02\x02')
    assert client.body(query='marker')['other_stores'][0]['status'] == 'unavailable'


def snapshot(home):
    return {str(p.relative_to(home)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in home.rglob('*')}


def test_22_no_writes_or_connections(tmp_path, corpus, clients):
    local, sources = corpus
    forget(local, 'remote0')
    assert cr.forgotten(local / 'lcm.db') == {'remote0'}
    copy_generation(sources[0], local, COPIES[0])
    before = snapshot(local)
    client = clients(tmp_path, local, audit=True)
    ready(client)
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
    pids = client.pids()
    assert len(pids) == 4
    client.close()
    eventually(lambda: all(not alive(pid) for pid in pids))
    for generation in generations:
        fd = os.open(generation, os.O_RDONLY)
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
        if not home.name.startswith('gen-'): time.sleep(1.5)
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


@pytest.mark.parametrize('error', ['Message store_id 0 not found; 0 matches'])
def test_r4_ambiguous_integer_standin_is_generic(monkeypatch, error):
    class MissingChild:
        folder = None
        def ask(self, raw):
            assert json.loads(raw)['params']['arguments']['store_id'] == 0
            future = cr.Future()
            future.set_result(cr.Merger.answer({'id': 1}, {'error': error}))
            return future
    monkeypatch.setattr(cr, 'visible_item', lambda *args: False)
    replica = SimpleNamespace(prefix=PREFIX, status='ok', child=MissingChild(), generation=1)
    merger = object.__new__(cr.Merger)
    assert merger.routed(replica, 'lcm_expand', {'store_id': 1}, {'forgotten'}) == {
        'error': 'the other store could not answer this call', 'store': PREFIX}


def test_r5_integer_standin_without_zero_passes_unchanged(monkeypatch):
    error = 'No matching message'
    class MissingChild:
        folder = None
        def ask(self, raw):
            assert json.loads(raw)['params']['arguments']['store_id'] == 0
            future = cr.Future()
            future.set_result(cr.Merger.answer({'id': 1}, {'error': error}))
            return future
    monkeypatch.setattr(cr, 'visible_item', lambda *args: False)
    replica = SimpleNamespace(prefix=PREFIX, status='ok', child=MissingChild(), generation=1)
    merger = object.__new__(cr.Merger)
    assert merger.routed(replica, 'lcm_expand', {'store_id': 1}, {'forgotten'}) == {
        'error': error, 'store': PREFIX, 'generation': 1}


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
