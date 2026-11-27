"""Packet 5 Part A: synthetic directory-bucket restore proofs."""
import ast
from contextlib import closing
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from hermes_lcm import claude_code_store as cc, cloud_copy as cp, cloud_guard as cg, cloud_restore as cr
from tests.test_cloud_copy import (COPY, PREFIX, ROOT, capture_store, checked, entry,
                                  fingerprint, fixture as capture_fixture, grow, pub)


@pytest.fixture
def fixture(tmp_path):
    return capture_fixture.__wrapped__(tmp_path)


def restore(f, target, generation=1, **kw):
    return cr.restore(f[1].inner, PREFIX, COPY, generation, target, **kw)


def head_path(f, generation):
    return f[1].inner.root / PREFIX / 'gen' / f'{generation}.json'


def object_path(f, key):
    return f[1].inner.root / PREFIX / key


def refusal(code, call):
    with pytest.raises(cp.Refused) as caught:
        call()
    assert caught.value.code == code
    return caught.value.detail


def test_A1_rebuild(fixture, tmp_path):
    for generation in range(1, 5):
        if generation > 1:
            grow(fixture)
        manifest = pub(fixture, part_size=8192, max_generations=3)
        assert len(manifest['database']['deltas']) == (generation - 1 if generation < 4 else 0)
        expected = checked(fixture).read_bytes()
        target = tmp_path / f'restore-{generation}'
        result = restore(fixture, target, generation, origin=fixture[0],
                         owner_file=fixture[0] / cc.OWNER_FILE,
                         transcripts=[f's1={fixture[2]}'], objects_dir=tmp_path / f'objects-{generation}',
                         reader_argv=successful_reader(tmp_path))
        assert result['code'] == 'done' and result['check3'] == 'done'
        assert (target / 'lcm.db').read_bytes() == expected
        assert (target / 'manifest.json').read_bytes() == head_path(fixture, generation).read_bytes()
        assert sorted(p.name for p in (target / cp.PAYLOAD_FOLDER).iterdir()) == sorted(
            r['name'] for r in manifest['payloads'])
        assert sorted(p.name for p in target.iterdir()) == ['lcm-large-outputs', 'lcm.db', 'manifest.json']
        assert target.stat().st_mode & 0o777 == 0o700


def test_v2_restores_v1_replica(fixture, tmp_path, monkeypatch):
    with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn, conn:
        conn.execute('DROP INDEX idx_msg_session_tool_call')
    with monkeypatch.context() as old:
        old.setattr(cp, 'CONTRACT_VERSION', 1)
        old.setattr(cp, 'CONTRACT_DIGEST', 'sha256:76644a990e50b9613697704dcb05b349a952d44ea22187bd768b0eab9a9badc9')
        manifest = pub(fixture)
    assert manifest['stamp']['contract']['version'] == 1 and cp.CONTRACT_VERSION == 2
    target = tmp_path / 'restored-v1'
    result = restore(fixture, target, transcripts=[f's1={fixture[2]}'])
    assert result['code'] == 'done' and result['check3'] == 'done'
    assert (target / 'lcm.db').read_bytes() == checked(fixture).read_bytes()


def test_v1_refuses_v2_replica(fixture, tmp_path, monkeypatch):
    assert pub(fixture)['stamp']['contract']['version'] == 2
    monkeypatch.setattr(cp, 'CONTRACT_VERSION', 1)
    fixture[1].calls.clear()
    refusal('replica_newer_contract', lambda: cr.restore(fixture[1], PREFIX, COPY, 1, tmp_path / 'refused-v2'))
    assert fixture[1].calls == [('get', f'{PREFIX}/gen/1.json')]


@pytest.mark.parametrize('fault,code', [
    ('delta', 'object_missing'), ('part', 'object_missing'), ('changed', 'object_sha256'),
    ('short', 'object_bytes'), ('digest', 'rebuild_digest_mismatch'),
    ('all_gone', 'object_missing'), ('payload', 'object_missing'),
    ('payload_changed', 'object_sha256'), ('payload_short', 'object_bytes'),
    ('payload_digest', 'object_sha256'), ('manifest', 'object_missing')])
def test_A2_proof_shape(fixture, tmp_path, fault, code):
    manifests = []
    for generation in range(1, 5):
        if generation > 1:
            grow(fixture)
        manifests.append(pub(fixture, part_size=8192, max_generations=3))
    reader = successful_reader(tmp_path)
    restore(fixture, tmp_path / 'newest', 4, reader_argv=reader)
    newest = manifests[-1]
    head_path(fixture, 4).unlink()
    old_keys = {r['key'] for r in manifests[2]['database']['base']['parts']}
    for key in {ref['key'] for ref in newest['database']['base']['parts']} - old_keys:
        object_path(fixture, key).unlink()
    restore(fixture, tmp_path / 'preceding', 3, reader_argv=reader)
    m = copy.deepcopy(manifests[2])
    ref = (m['payloads'][0] if fault.startswith('payload') else
           m['database']['deltas'][0]['objects'][0] if fault == 'delta' else m['database']['base']['parts'][0])
    key = PREFIX + '/' + ref['key']
    path = object_path(fixture, ref['key'])
    if fault in ('delta', 'part', 'payload'):
        path.unlink()
    elif fault in ('changed', 'payload_changed'):
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif fault in ('short', 'payload_short'):
        path.write_bytes(path.read_bytes()[:-1])
    elif fault == 'digest':
        m['database']['file_sha256'] = '0' * 64
    elif fault == 'payload_digest':
        ref['sha256'] = '0' * 64
    elif fault == 'all_gone':
        for folder in ('parts', 'deltas', 'payloads'):
            for p in (fixture[1].inner.root / PREFIX / folder).iterdir():
                p.unlink()
    if fault == 'manifest':
        head_path(fixture, 3).unlink()
        key = PREFIX + '/gen/3.json'
    else:
        head_path(fixture, 3).write_text(json.dumps(m))
    target = tmp_path / 'broken'
    detail = refusal(code, lambda: restore(fixture, target, 3, objects_dir=tmp_path / 'objects'))
    if fault == 'digest':
        assert detail['generation'] == 3
    else:
        assert detail['key'] == key
    assert not (target / 'lcm.db').exists()
    assert not list(target.iterdir())
    assert not list((tmp_path / 'objects').rglob('*')) if (tmp_path / 'objects').exists() else True


@pytest.mark.parametrize('marker', ['nonempty', 'capture-owner.json', 'config.yaml', 'state.db', 'env', 'default'])
@pytest.mark.parametrize('nested', [False, True])
def test_A3_target_refusals(fixture, tmp_path, monkeypatch, marker, nested):
    assert cr.OWNER_FILE == cc.OWNER_FILE and cr.HERMES_MARKERS == cc.HERMES_MARKERS
    home = tmp_path / 'unsafe'
    home.mkdir()
    if marker == 'env':
        monkeypatch.setenv('HERMES_HOME', str(home))
    elif marker == 'default':
        monkeypatch.delenv('HERMES_HOME', raising=False)
        monkeypatch.setenv('HOME', str(tmp_path))
        home = tmp_path / '.hermes'
        home.mkdir()
    else:
        (home / ('data' if marker == 'nonempty' else marker)).write_text('fixture')
    target = home / 'child' if nested else home
    fixture[1].calls.clear()
    if marker == 'nonempty' and nested:
        target.mkdir()
        (target / 'data').write_text('fixture')
    code = 'restore_target_not_empty' if marker == 'nonempty' else 'restore_target_is_home'
    refusal(code, lambda: cr.restore(fixture[1], PREFIX, COPY, 1, target))
    assert not fixture[1].calls


def fake_reader(tmp_path, body):
    path = tmp_path / 'reader.py'
    path.write_text('import json, sys\nfrom pathlib import Path\n' + body)
    return [sys.executable, '-I', '-S', '-B', str(path)]


def successful_reader(tmp_path):
    # Isolate checks 1-3 and object faults from the reader check.
    return fake_reader(tmp_path, 'for line in sys.stdin:\n'
                       ' r = json.loads(line)\n'
                       ' print(json.dumps({"jsonrpc":"2.0", "id":r["id"], "result":'
                       '{"content":[{"type":"text","text":"{}"}],"isError":False}}), flush=True)\n')


@pytest.mark.parametrize('check', [1, 2, 3, 4])
def test_A3_checks_fail_independently(fixture, tmp_path, monkeypatch, check):
    pub(fixture)
    target = tmp_path / 'restored'
    if check == 1:
        real = cr.materialize
        def corrupt(*args, **kw):
            real(*args, **kw)
            (target / 'lcm.db').write_bytes(b'not sqlite')
        monkeypatch.setattr(cr, 'materialize', corrupt)
    options = dict(transcripts=[f's1={fixture[2]}'])
    if check == 2:
        m = json.loads(head_path(fixture, 1).read_bytes())
        m['unresolved_quoted'].append({'name': 'one-sided', 'sessions': ['s1']})
        head_path(fixture, 1).write_text(json.dumps(m))
    if check == 3:
        bad = tmp_path / 'wrong.jsonl'
        bad.write_text(json.dumps(entry('user', 'wrong', 'different')) + '\n')
        options['transcripts'] = [f's1={bad}']
    if check == 4:
        options['reader_argv'] = fake_reader(tmp_path, 'raise SystemExit(1)\n')
    detail = refusal('restore_check_failed', lambda: restore(fixture, target, **options))
    assert detail['check'] == check
    assert (target / 'lcm.db').exists() and (target / 'manifest.json').exists()


@pytest.mark.parametrize('transcripts', [[], ['other=missing']])
def test_A3_check3_not_run(fixture, tmp_path, transcripts):
    pub(fixture)
    assert restore(fixture, tmp_path / 'restored', transcripts=transcripts,
                   reader_argv=successful_reader(tmp_path))['check3'] == 'check3_not_run'


@pytest.mark.parametrize('kind', ['unresolved', 'quoted', 'empty', 'missing_session'])
def test_A3_special_stores(fixture, tmp_path, kind):
    with capture_store(fixture[0] / 'lcm.db', fixture[0], fixture[3]) as store:
        if kind in ('empty', 'unresolved', 'quoted'):
            store.connection.execute('DELETE FROM message_relations')
            store.connection.execute('DELETE FROM messages')
            store.connection.execute('DELETE FROM capture_events')
            store.connection.commit()
        if kind == 'missing_session':
            store.connection.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES ('s2','user','second session',1)")
            store.connection.commit()
        elif kind in ('unresolved', 'quoted'):
            name = '20261101_120000_lost_0123456789ab_1234.json' if kind == 'unresolved' else 'example'
            store.connection.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES ('s1','user',?,1)",
                                     (f'[Externalized tool output: absent; ref={name}]',))
            store.connection.commit()
    m = pub(fixture)
    result = restore(fixture, tmp_path / 'restored', origin=fixture[0],
                     transcripts=[f's1={fixture[2]}'] if kind == 'missing_session' else [],
                     reader_argv=successful_reader(tmp_path) if kind == 'missing_session' else None)
    assert result['code'] == 'done' and result['check3'] == 'check3_not_run'
    assert result['unresolved_minted'] == len(m['unresolved_minted'])
    assert result['unresolved_quoted'] == len(m['unresolved_quoted'])
    if kind in ('unresolved', 'quoted'):
        assert result['unresolved_' + ('minted' if kind == 'unresolved' else 'quoted')] == 1


@pytest.mark.parametrize('payload_mode', ['link', 'copy', 'empty'])
@pytest.mark.parametrize('reader_fails', [False, True])
def test_A3_fresh_reader_home(fixture, tmp_path, monkeypatch, payload_mode, reader_fails):
    if payload_mode == 'empty':
        with capture_store(fixture[0] / 'lcm.db', fixture[0], fixture[3]) as store:
            store.connection.execute("UPDATE messages SET content='synthetic plain message'")
            store.connection.commit()
    manifest = pub(fixture)
    if payload_mode == 'copy':
        def no_link(*args, **kw):
            raise OSError('fixture: hard links unavailable')
        monkeypatch.setattr(cr.os, 'link', no_link)
    record = tmp_path / 'observed.json'
    reader = fake_reader(tmp_path, 'import hashlib\n'
                         'home = Path(sys.argv[sys.argv.index("--hermes-home") + 1])\n'
                         'store = Path(sys.argv[sys.argv.index("--store") + 1])\n'
                         'folder = home / "lcm-large-outputs"\n'
                         'payloads = {p.name: [p.is_symlink(), hashlib.sha256(p.read_bytes()).hexdigest(),\n'
                         ' p.samefile(store.parent / "lcm-large-outputs" / p.name)]\n'
                         ' for p in folder.iterdir()} if folder.exists() else {}\n'
                         f'Path({str(record)!r}).write_text(json.dumps([str(home), str(store),\n'
                         ' sorted(p.name for p in home.iterdir()), payloads, home.is_symlink()]))\n'
                         f'if {reader_fails!r}: raise SystemExit(1)\n'
                         'for line in sys.stdin:\n'
                         ' r = json.loads(line)\n'
                         ' print(json.dumps({"jsonrpc":"2.0", "id":r["id"], "result":'
                         '{"content":[{"type":"text","text":"{}"}],"isError":False}}), flush=True)\n')
    real_reader = cr._reader
    snapshots = []
    def observed_reader(to, *args):
        def snapshot():
            return {p.relative_to(to).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns,
                    hashlib.sha256(p.read_bytes()).hexdigest()) for p in to.rglob('*') if p.is_file()}
        before = snapshot()
        try:
            return real_reader(to, *args)
        finally:
            snapshots.append((before, snapshot()))
    monkeypatch.setattr(cr, '_reader', observed_reader)
    target = tmp_path / 'restored'
    if reader_fails:
        assert refusal('restore_check_failed', lambda: restore(fixture, target, reader_argv=reader))['check'] == 4
    else:
        assert restore(fixture, target, reader_argv=reader)['code'] == 'done'
    name, database, entries, payloads, symlink = json.loads(record.read_text())
    assert Path(name).parent == target.parent and Path(name) != target and not symlink
    assert Path(database) == target / 'lcm.db'
    assert entries == ['lcm-large-outputs'] and not Path(name).exists()
    assert payloads == {r['name']: [False, r['sha256'], payload_mode == 'link'] for r in manifest['payloads']}
    assert snapshots and snapshots[0][0] == snapshots[0][1]
    assert sorted(p.name for p in target.iterdir()) == ['lcm-large-outputs', 'lcm.db', 'manifest.json']


def test_A4_isolation(fixture, tmp_path):
    tree = ast.parse((ROOT / 'cloud_restore.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split('.')[0] in sys.stdlib_module_names for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.level == 1 and node.module is None and
                    [n.name for n in node.names] == ['cloud_copy', 'cloud_delta']) or (
                    node.level == 0 and node.module.split('.')[0] in sys.stdlib_module_names)
    # The isolated CLI exercises the stock reader on a restored payload too.
    pub(fixture)
    before = fingerprint(fixture[0])
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(tmp_path))
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'scripts/lcm_x_cloud.py'),
                             'restore', '--copy-id', COPY, '--bucket-dir', str(fixture[1].inner.root),
                             '--prefix', 'copies', '--generation', '1', '--to', str(tmp_path / 'restored')],
                            capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(result.stdout.splitlines()) == 1 and json.loads(result.stdout)['code'] == 'done'
    assert (tmp_path / 'restored/lcm.db').read_bytes() == checked(fixture).read_bytes()
    assert (tmp_path / 'restored/manifest.json').read_bytes() == head_path(fixture, 1).read_bytes()
    assert fingerprint(fixture[0]) == before


@pytest.mark.parametrize('force_copy', [False, True])
def test_A3_stock_payload_reader(fixture, tmp_path, monkeypatch, force_copy):
    manifest = pub(fixture)
    assert manifest['payloads']
    if force_copy:
        def no_link(*args, **kw):
            raise OSError('fixture: hard links unavailable')
        monkeypatch.setattr(cr.os, 'link', no_link)
    before = fingerprint(fixture[0])
    target = tmp_path / 'restored'
    result = restore(fixture, target)
    assert result['code'] == 'done' and (target / 'lcm.db').exists()
    assert fingerprint(fixture[0]) == before


@pytest.mark.parametrize('where', ['hermes', 'capture', 'equal', 'inside', 'contains'])
def test_objects_dir_refusals(fixture, tmp_path, monkeypatch, where):
    pub(fixture)
    hermes = tmp_path / 'hermes'
    hermes.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(hermes))
    target = tmp_path / 'restored'
    objects = {'hermes': hermes / 'objects', 'capture': fixture[0] / 'objects',
               'equal': target, 'inside': target / 'objects', 'contains': tmp_path / 'cache'}[where]
    if where == 'contains':
        target = objects / 'restored'
    reader = successful_reader(tmp_path)
    def snapshot():
        return {p.relative_to(tmp_path).as_posix(): (p.stat().st_mode,
                p.read_bytes() if p.is_file() else None) for p in tmp_path.rglob('*')}
    before = snapshot()
    fixture[1].calls.clear()
    try:
        code = cr.restore(fixture[1], PREFIX, COPY, 1, target, objects_dir=objects,
                          reader_argv=reader)['code']
    except cp.Refused as error:
        code = error.code
    assert code == ('restore_target_is_home' if where in ('hermes', 'capture') else 'restore_objects_overlap')
    assert snapshot() == before
    assert not fixture[1].calls
    assert not target.exists() and not objects.exists()


def test_tilde_hermes_home(fixture, tmp_path, monkeypatch):
    pub(fixture)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', '~/hermes-x')
    fixture[1].calls.clear()
    target = tmp_path / 'hermes-x' / 'restored'
    refusal('restore_target_is_home', lambda: cr.restore(
        fixture[1], PREFIX, COPY, 1, target, reader_argv=successful_reader(tmp_path)))
    assert not target.parent.exists() and not fixture[1].calls


@pytest.mark.parametrize('path', ['direct', 'after_create'])
def test_p8_default_path_race(fixture, path):
    if path == 'direct':
        pub(fixture)
    saved = []
    def vanish(method, key):
        if method == 'get' and key == PREFIX + '/gen/1.json' and not saved:
            manifest = fixture[1].inner.root / key
            saved.append((manifest, manifest.read_bytes()))
            manifest.unlink()
    fixture[1].hook = vanish
    try:
        if path == 'direct':
            try:
                cp._head(fixture[1], PREFIX, COPY)
                outcome = 'returned'
            except Exception as error:
                outcome = type(error).__name__
            assert outcome == 'FileNotFoundError'
        else:
            hooks = cg.install(fixture[0], COPY, fixture[1], PREFIX, uploader=cp, hardware_id='synthetic-host')
            try:
                outcome = cp.publish(fixture[0], COPY, fixture[1], PREFIX, **hooks)['generation']
            except cp.Refused as error:
                outcome = error.code
            assert outcome == 1
            state = json.loads((fixture[0] / 'cloud/state.json').read_text())
            assert 'last_pair' in state and 'snapshot_signature' in state
        assert saved
    finally:
        for manifest, data in saved:
            manifest.write_bytes(data)


def test_duplicate_payload_name(fixture, tmp_path):
    manifest = pub(fixture)
    manifest['payloads'].append(dict(manifest['payloads'][0]))
    head_path(fixture, 1).write_text(json.dumps(manifest))
    fixture[1].calls.clear()
    target, objects = tmp_path / 'restored', tmp_path / 'objects'
    try:
        cr.restore(fixture[1], PREFIX, COPY, 1, target, objects_dir=objects)
        outcome = 'returned'
    except Exception as error:
        outcome = getattr(error, 'code', type(error).__name__)
    assert outcome == 'bad_name'
    assert fixture[1].calls == [('get', PREFIX + '/gen/1.json')]
    assert not list(target.iterdir())
    assert not objects.exists()


def test_reader_child_environment(fixture, tmp_path, monkeypatch):
    manifest = pub(fixture)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'parent-hermes'))
    monkeypatch.setenv('LCM_PARENT_SENTINEL', 'parent-only')
    monkeypatch.setenv('RESTORE_PARENT_SENTINEL', 'parent-only')
    record = tmp_path / 'reader-env.json'
    reader = fake_reader(tmp_path, 'import os\n'
                         f'Path({str(record)!r}).write_text(json.dumps(dict(os.environ)))\n'
                         'for line in sys.stdin:\n'
                         ' r = json.loads(line)\n'
                         ' print(json.dumps({"id":r["id"],"result":'
                         '{"content":[{"type":"text","text":"{}"}]}}), flush=True)\n')
    target = tmp_path / 'restored'
    assert restore(fixture, target, reader_argv=reader)['payloads'] == len(manifest['payloads'])
    env = json.loads(record.read_text())
    assert env.get('HERMES_HOME') != str(tmp_path / 'parent-hermes')
    assert not any(k.startswith('LCM_') for k in env) and 'RESTORE_PARENT_SENTINEL' not in env
    home = Path(env['HERMES_HOME'])
    assert env['HOME'] == str(home) and home.parent == target.parent and not home.exists()
    assert env['PATH'] == os.defpath
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
    # CPython and CoreFoundation may add locale metadata after process launch.
    assert set(env) <= {'PATH', 'HOME', 'HERMES_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                        'LC_CTYPE', '__CF_USER_TEXT_ENCODING'}


def test_checker_child_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'parent-hermes'))
    monkeypatch.setenv('LCM_PARENT_SENTINEL', 'parent-only')
    monkeypatch.setenv('RESTORE_PARENT_SENTINEL', 'parent-only')
    record = tmp_path / 'checker-env.json'
    script = tmp_path / 'checker.py'
    script.write_text('import json, os, sys\nfrom pathlib import Path\n'
                      'Path(sys.argv[1]).write_text(json.dumps(dict(os.environ)))\n')
    monkeypatch.setattr(cr, 'SCRIPTS', tmp_path)
    cr._run(script.name, [str(record)])
    env = json.loads(record.read_text())
    assert 'HERMES_HOME' not in env
    assert not any(k.startswith('LCM_') for k in env) and 'RESTORE_PARENT_SENTINEL' not in env
    assert env['PATH'] == os.defpath and set(env) <= {'PATH', 'LC_CTYPE', '__CF_USER_TEXT_ENCODING'}


def tree_bytes(home):
    return {p.relative_to(home).as_posix(): (p.stat().st_mode, p.read_bytes() if p.is_file() else None)
            for p in [home, *home.rglob('*')]}


@pytest.mark.parametrize('kind', ['parts', 'deltas', 'file', 'deep', 'root'])
@pytest.mark.parametrize('marker', ['config.yaml', 'capture-owner.json'])
def test_cache_symlinks(fixture, tmp_path, kind, marker):
    manifest = pub(fixture, part_size=8192)
    if kind == 'deltas':
        grow(fixture)
        manifest = pub(fixture, part_size=8192)
        assert manifest['database']['deltas']
    home = tmp_path / 'synthetic-home'
    home.mkdir(mode=0o755)
    (home / marker).write_text('synthetic marker')
    objects = tmp_path / 'objects'
    objects.mkdir()
    ref = manifest['database']['base']['parts'][0]
    if kind == 'file':
        (objects / 'parts').mkdir()
        source = home / 'existing-object'
        source.write_bytes(object_path(fixture, ref['key']).read_bytes())
        (objects / ref['key']).symlink_to(source)
    elif kind in ('deep', 'root'):
        link = objects / 'nested' if kind == 'deep' else tmp_path / 'cache-link'
        link.symlink_to(home, target_is_directory=True)
        objects = link / 'cache' if kind == 'deep' else link
    else:
        (objects / kind).symlink_to(home, target_is_directory=True)
    before = tree_bytes(home)
    try:
        code = restore(fixture, tmp_path / 'restored', manifest['generation'], objects_dir=objects,
                       reader_argv=successful_reader(tmp_path))['code']
    except cp.Refused as error:
        code = error.code
    assert code == 'restore_objects_symlink'
    assert tree_bytes(home) == before
    assert home.stat().st_mode & 0o777 == 0o755


def test_target_busy(fixture, tmp_path, monkeypatch):
    pub(fixture)
    target = tmp_path / 'restored'
    real_run, observed = cr._run, []
    def competing(*args):
        before = tree_bytes(target)
        try:
            code = restore(fixture, target)['code']
        except cp.Refused as error:
            code = error.code
        observed.append(code)
        assert tree_bytes(target) == before
        return real_run(*args)
    monkeypatch.setattr(cr, '_run', competing)
    assert restore(fixture, target, reader_argv=successful_reader(tmp_path))['code'] == 'done'
    assert observed == ['restore_target_busy']


@pytest.mark.parametrize('failing', [True, False])
def test_target_swap(fixture, tmp_path, failing):
    manifest = pub(fixture)
    target, moved, home = (tmp_path / name for name in ('target', 'moved', 'synthetic-home'))
    target.mkdir()
    home.mkdir(mode=0o755)
    (home / 'config.yaml').write_text('synthetic configuration')
    (home / 'keep').mkdir()
    (home / 'keep/data').write_bytes(b'untouched')
    before, swapped = tree_bytes(home), []
    def fetch(key):
        if not swapped:
            target.rename(moved)
            target.symlink_to(home, target_is_directory=True)
            swapped.append(True)
            if failing:
                raise KeyError(key)
        return fixture[1].inner.get(PREFIX + '/' + key)
    try:
        cr.materialize(manifest, head_path(fixture, 1).read_bytes(), fetch, target)
        outcome = 'done'
    except Exception as error:
        outcome = getattr(error, 'code', type(error).__name__)
    assert tree_bytes(home) == before
    assert outcome == ('object_missing' if failing else 'done')
    assert list(moved.iterdir()) == [] if failing else (moved / 'lcm.db').read_bytes() == checked(fixture).read_bytes()


@pytest.mark.parametrize('other', ['good', 'short', 'wrong'])
def test_cache_create_race(fixture, tmp_path, monkeypatch, other):
    manifest = pub(fixture, part_size=8192)
    target, objects = tmp_path / 'target', tmp_path / 'objects'
    target.mkdir()
    ref = manifest['database']['base']['parts'][0]
    needed = objects / ref['key']
    original = object_path(fixture, ref['key']).read_bytes()
    competing = original if other == 'good' else original[:-1] if other == 'short' else b'x' * len(original)
    real_link, raced = os.link, []
    def win_first(source, path, *args, **kw):
        if Path(path) == needed and not raced:
            raced.append(True)
            needed.write_bytes(competing)
        return real_link(source, path, *args, **kw)
    monkeypatch.setattr(cr.os, 'link', win_first)
    def fetch(key):
        if raced and key != ref['key']:
            raise KeyError(key)
        return fixture[1].inner.get(PREFIX + '/' + key)
    try:
        cr.materialize(manifest, head_path(fixture, 1).read_bytes(), fetch, target, objects_dir=objects)
        code = 'done'
    except Exception as error:
        code = getattr(error, 'code', type(error).__name__)
    assert raced and needed.exists(), 'rollback deleted the competing cache object'
    assert needed.read_bytes() == competing
    assert code == {'good': 'object_missing', 'short': 'object_bytes', 'wrong': 'object_sha256'}[other]
    assert not list(target.iterdir())


@pytest.mark.parametrize('home_kind', ['hermes', 'capture', 'nested_capture'])
def test_cache_contains_home(fixture, tmp_path, monkeypatch, home_kind):
    pub(fixture, part_size=8192)
    objects = tmp_path / 'objects'
    home = objects / ('parts' if home_kind == 'hermes' else 'deltas')
    if home_kind == 'nested_capture':
        home /= 'unused/deeper'
    home.mkdir(mode=0o755, parents=True)
    (home / 'keep').write_bytes(b'synthetic protected data')
    if home_kind == 'hermes':
        monkeypatch.setenv('HERMES_HOME', str(home))
    else:
        (home / 'capture-owner.json').write_text('synthetic owner')
    target = tmp_path / 'restored'
    before = tree_bytes(tmp_path)
    fixture[1].calls.clear()
    try:
        code = cr.restore(fixture[1], PREFIX, COPY, 1, target, objects_dir=objects)['code']
    except cp.Refused as error:
        code = error.code
    assert code == 'restore_target_is_home'
    assert tree_bytes(tmp_path) == before
    assert home.stat().st_mode & 0o777 == 0o755
    assert not fixture[1].calls


@pytest.mark.parametrize('home_kind', ['hermes', 'capture', 'target'])
def test_cache_object_destination_guard(fixture, tmp_path, monkeypatch, home_kind):
    manifest = pub(fixture, part_size=8192)
    objects = tmp_path / 'objects'
    home = objects / 'parts'
    home.mkdir(mode=0o755, parents=True)
    target = home if home_kind == 'target' else tmp_path / 'restored'
    target.mkdir(exist_ok=True)
    before = tree_bytes(objects)
    def fetch(key):
        if home_kind == 'hermes':
            monkeypatch.setenv('HERMES_HOME', str(home))
        elif home_kind == 'capture':
            (home / 'capture-owner.json').write_text('synthetic owner')
        return fixture[1].inner.get(PREFIX + '/' + key)
    try:
        cr.materialize(manifest, head_path(fixture, 1).read_bytes(), fetch, target, objects_dir=objects)
        code = 'done'
    except cp.Refused as error:
        code = error.code
    assert code == ('restore_objects_overlap' if home_kind == 'target' else 'restore_target_is_home')
    if home_kind == 'capture':
        (home / 'capture-owner.json').unlink()
    if home_kind == 'target':
        # The claim legitimately sets the restore target's mode before the fetch.
        before['parts'] = (home.stat().st_mode, None)
    assert tree_bytes(objects) == before
    assert not list(target.iterdir())


def test_cache_atomic_publish_race(fixture, tmp_path, monkeypatch):
    manifest = pub(fixture, part_size=8192)
    objects = tmp_path / 'objects'
    targets = [tmp_path / name for name in ('first', 'second')]
    for target in targets:
        target.mkdir()
    ref = manifest['database']['base']['parts'][0]
    needed = objects / ref['key']
    original = object_path(fixture, ref['key']).read_bytes()
    stored = head_path(fixture, 1).read_bytes()
    def fetch(key):
        return fixture[1].inner.get(PREFIX + '/' + key)
    real_open, real_link, real_sync = os.open, os.link, os.fsync
    raced, outcomes, synced, published = [], [], set(), []
    def sync(fd):
        real_sync(fd)
        info = os.fstat(fd)
        synced.add((info.st_dev, info.st_ino))
    def link(source, destination, *args, **kw):
        if Path(destination).parent == needed.parent:
            info = Path(source).stat()
            assert (info.st_dev, info.st_ino) in synced
            assert Path(source) != Path(destination)
            published.append(Path(source))
        return real_link(source, destination, *args, **kw)
    def half_written(path, flags, *args, **kw):
        fd = real_open(path, flags, *args, **kw)
        if flags & os.O_EXCL and Path(path).parent == needed.parent and not raced:
            raced.append(Path(path))
            os.write(fd, original[:len(original) // 2])
            try:
                cr.materialize(manifest, stored, fetch, targets[1], objects_dir=objects)
                outcomes.append('done')
            except cp.Refused as error:
                outcomes.append(error.code)
            finally:
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
        return fd
    monkeypatch.setattr(cr.os, 'open', half_written)
    monkeypatch.setattr(cr.os, 'link', link)
    monkeypatch.setattr(cr.os, 'fsync', sync)
    cr.materialize(manifest, stored, fetch, targets[0], objects_dir=objects)
    assert raced and outcomes == ['done'], 'the second restore validated an in-progress cache write'
    assert raced[0] != needed and published and all(not path.exists() for path in published)
    assert needed.read_bytes() == original
    assert all((target / 'lcm.db').read_bytes() == checked(fixture).read_bytes() for target in targets)
    assert {p.relative_to(objects).as_posix() for p in objects.rglob('*') if p.is_file()} == {
        row['key'] for row in manifest['database']['base']['parts']}


@pytest.mark.parametrize('check', [2, 3, 4])
@pytest.mark.parametrize('timeout', [True, False])
def test_check_timeout(fixture, tmp_path, monkeypatch, check, timeout):
    pub(fixture)
    real_run = cr._run
    def failure():
        if timeout:
            raise subprocess.TimeoutExpired('synthetic slow check', 60)
        raise ValueError('synthetic mismatch')
    def run(script, args):
        if (check == 2 and script == 'cloud_copy_outside_in.py') or (
                check == 3 and script == 'capture_outside_in.py'):
            failure()
        return real_run(script, args)
    monkeypatch.setattr(cr, '_run', run)
    if check == 4:
        monkeypatch.setattr(cr, '_reader', lambda *args: failure())
    target = tmp_path / 'restored'
    detail = refusal('restore_check_timeout' if timeout else 'restore_check_failed', lambda: restore(
        fixture, target, transcripts=[f's1={fixture[2]}'], reader_argv=successful_reader(tmp_path)))
    assert detail['check'] == check and detail['generation'] == 1
    assert (target / 'manifest.json').exists()


@pytest.mark.parametrize('where', ['equal', 'inside', 'contains', 'hermes', 'unicode'])
def test_path_aliases(fixture, tmp_path, monkeypatch, where):
    probe = tmp_path / 'CaseProbe'
    probe.mkdir()
    if not (tmp_path / 'caseprobe').exists():
        pytest.skip('test volume is case-sensitive (probed CaseProbe/caseprobe)')
    pub(fixture)
    target = tmp_path / 'Restored'
    target.mkdir()
    alias = tmp_path / 'RESTORED'
    options = {'equal': alias, 'inside': alias / 'cache', 'contains': tmp_path / 'CACHE'}
    if where == 'hermes':
        monkeypatch.setenv('HERMES_HOME', str(target))
        target = alias / 'child'
    elif where == 'unicode':
        target = tmp_path / 'caf\u00e9'
        target.mkdir()
        alias = tmp_path / 'cafe\u0301'
        if not alias.exists():
            pytest.skip('test volume distinguishes composed and decomposed Unicode (probed)')
        options[where] = alias / 'cache'
    elif where == 'contains':
        cache = tmp_path / 'cache'
        cache.mkdir()
        target = cache / 'child'
    before = tree_bytes(tmp_path)
    try:
        code = restore(fixture, target, objects_dir=options.get(where),
                       reader_argv=None)['code']
    except cp.Refused as error:
        code = error.code
    assert code == ('restore_target_is_home' if where == 'hermes' else 'restore_objects_overlap')
    assert tree_bytes(tmp_path) == before
