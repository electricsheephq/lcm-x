"""Packet 5 Part A: synthetic directory-bucket restore proofs."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
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
