"""Lane 2 synthetic capture stores and injected publication races."""
import ast
import base64
from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import zlib

import pytest

from hermes_lcm import claude_code_store as cc, cloud_copy as cp, cloud_delta as cd
from hermes_lcm.config import LCMConfig

COPY = 'abcdef01-2345-6789-abcd-0123456789ab'
PREFIX = 'copies/' + COPY
TOKEN = '1' * 32
ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def entry(kind, uid, content):
    return dict(type=kind, uuid=uid, sessionId='s1', timestamp='2026-11-01T00:00:00.000Z',
                message=dict(role=kind, content=content))


class Spy:
    def __init__(self, root):
        self.inner = cp.DirTransport(root, PREFIX)
        self.calls = []
        self.hook = lambda method, key: None

    def __getattr__(self, method):
        def call(*args):
            key = args[0] if args else None
            self.calls.append((method, key))
            self.hook(method, key)
            return getattr(self.inner, method)(*args)
        return call


@contextmanager
def capture_store(database, home, config):
    with closing(cc.open_store(database, home, config=config)) as store:
        try:
            yield store
        finally:
            assert cc.to_rollback_journal(store.connection)


@pytest.fixture
def fixture(tmp_path):
    home = tmp_path / 'home'
    database = cc.admit(home, hermes_home=tmp_path / 'unused-host')
    config = LCMConfig(database_path=str(database), embeddings_enabled=False,
                       large_output_externalization_enabled=True,
                       large_output_externalization_threshold_chars=200)
    path = tmp_path / 'projects' / 'work' / 's1.jsonl'
    path.parent.mkdir(parents=True)
    rows = [entry('user', 'u1', 'synthetic question'),
            entry('assistant', 'a1', [dict(type='tool_use', id='c1', name='Read', input={})]),
            entry('user', 'r1', [dict(type='tool_result', tool_use_id='c1', content='ochre ' * 700)])]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with capture_store(database, home, config) as store:
        assert cc.drain(store, cc.Lineage('s1', None, path)) == 'drained'
        store.add_message_relations([[(1, 'proof', 2, 0, 1.0)]])
    cloud = home / 'cloud'
    cloud.mkdir(mode=0o700)
    (cloud / 'cloud.json').write_text(json.dumps(dict(copy_id=COPY)))
    bucket = Spy(tmp_path / 'bucket')
    return home, bucket, path, config


def grow(f, count=1):
    home, _, path, config = f
    offset = len(path.read_text().splitlines())
    with path.open('a') as handle:
        for index in range(count):
            handle.write(json.dumps(entry('user', f'u{offset + index}', 'one more synthetic turn ' * 40)) + '\n')
    with capture_store(home / 'lcm.db', home, config) as store:
        assert cc.drain(store, cc.Lineage('s1', None, path)) == 'drained'


def pub(f, **kw):
    home, bucket, _, _ = f
    result = cp.publish(home, COPY, bucket, PREFIX, guard=lambda head, db: None,
                        forget_items=kw.pop('forget_items', lambda head, db: []), **kw)
    assert isinstance(result, dict), 'publish must produce a manifest'
    assert staged(f).stat().st_size < 5_000_000
    return result


def staged(f):
    return f[0] / 'cloud' / 'staging' / 'lcm.db'


def keys(f):
    root = f[1].inner.root
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()) if root.exists() else []


def refused(code, call, *args, **kw):
    with pytest.raises(cd.DeltaRefused) as caught:
        call(*args, **kw)
    assert caught.value.code == code
    return caught.value


def objects(manifest):
    db = manifest['database']
    return db['base']['parts'] + [r for e in db['deltas'] for r in e['objects']] + manifest['payloads']


def fetch(f, key):
    return f[1].get(PREFIX + '/' + key)


def rebuild(f, manifest, name):
    path = f[0].parent / name
    cd.rebuild(path, manifest['database'], manifest['generation'], lambda key: fetch(f, key))
    return path.read_bytes()


def state(f):
    return f[0] / 'cloud' / 'state.json'


def last(f):
    return f[0] / 'cloud' / 'pages' / (COPY + '.last')


def ready(f, now=100):
    m = pub(f, clock=lambda: now)
    data = json.loads(state(f).read_text())
    data.update(last_pair=[m['generation'], m['attempt']['token']],
                pending_pair=[m['generation'], m['attempt']['token']],
                retention_done_generation=m['generation'], unpublished_decisions=[])
    state(f).write_text(json.dumps(data))
    return m


def cli(f, *extra):
    return subprocess.run([sys.executable, '-I', '-S', str(ROOT / 'scripts/lcm_x_cloud.py'),
                           'publish', '--home', str(f[0]), '--copy-id', COPY,
                           '--bucket-dir', str(f[1].inner.root), '--prefix', 'copies', *extra],
                          capture_output=True, text=True)


def test_01_generation_1_is_a_whole_base(fixture):
    junk = fixture[0] / cp.PAYLOAD_FOLDER / 'unreferenced.json'
    junk.parent.mkdir(exist_ok=True)
    junk.write_text('{"content":"unused"}')
    m = pub(fixture, part_size=8192)
    assert rebuild(fixture, m, 'rebuilt') == staged(fixture).read_bytes()
    assert m['payloads'] and all(p['name'] != junk.name for p in m['payloads'])


def test_02_both_placeholder_families(fixture):
    home, _, path, config = fixture
    payload = base64.b64encode(bytes(range(256)) * 16).decode()
    with path.open('a') as handle:
        handle.write(json.dumps(entry('user', 'media', 'prefix\n' + payload + '\nsuffix')) + '\n')
    with capture_store(home / 'lcm.db', home, config) as store:
        cc.drain(store, cc.Lineage('s1', None, path))
        assert '[Externalized LCM ingest payload:' in store.connection.execute(
            'SELECT content FROM messages ORDER BY store_id DESC LIMIT 1').fetchone()[0]
    m = pub(fixture)
    assert len(m['payloads']) >= 2
    assert all(fetch(fixture, p['key']) for p in m['payloads'])


def test_03_bucket_holds_nothing_else(fixture):
    m = pub(fixture)
    allowed = {PREFIX + '/' + r['key'] for r in objects(m)} | {PREFIX + '/gen/1.json'}
    assert set(keys(fixture)) == allowed
    for path in (fixture[0] / 'cloud').rglob('*'):
        if path.name != 'cloud.json':
            assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    for path in fixture[1].inner.root.rglob('*'):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)


def test_04_records_sidecar(fixture):
    (fixture[0] / 'lcm.db.records').touch()
    refused('records_pair_unsupported', pub, fixture)
    assert not keys(fixture)


def test_05_part_size(fixture):
    assert cp.DEFAULT_PART_SIZE == 33554432
    m = pub(fixture, part_size=8192)
    parts = m['database']['base']['parts']
    assert len(parts) > 1 and all(p['bytes'] == 8192 for p in parts[:-1])
    assert all(sha(fetch(fixture, p['key'])) == p['key'].split('/')[1][:64] for p in parts)


def test_06_a_second_publish_is_a_delta(fixture):
    first = pub(fixture)
    grow(fixture)
    old_payload = first['payloads'][0]
    new_name = 'same-content.json'
    shutil.copyfile(fixture[0] / cp.PAYLOAD_FOLDER / old_payload['name'], fixture[0] / cp.PAYLOAD_FOLDER / new_name)
    with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn, conn:
        conn.execute('UPDATE messages SET content=? WHERE store_id=(SELECT max(store_id) FROM messages)',
                     ('[Externalized tool output: quoted; ref=' + new_name + ']',))
    fixture[1].calls.clear()
    def exclusive(method, key):
        if method == 'put_new':
            assert not (fixture[1].inner.root / key).exists()
    fixture[1].hook = exclusive
    second = pub(fixture, token='2' * 32)
    puts = [key for method, key in fixture[1].calls if method == 'put_new']
    assert not any('/parts/' in key for key in puts)
    assert any('/deltas/' in key and key.endswith('2' * 32) for key in puts)
    by_name = {p['name']: p for p in second['payloads']}
    assert by_name[old_payload['name']] == old_payload
    assert by_name[new_name]['key'].endswith('2' * 32)


def test_07_resumed_attempt(fixture):
    def crash(method, key):
        if method == 'list' and key.endswith('/parts/'):
            raise OSError('injected outage')
    fixture[1].hook = crash
    with pytest.raises(OSError):
        pub(fixture, token=TOKEN, clock=lambda: 100)
    existing = set(keys(fixture))
    assert existing
    fixture[1].hook = lambda method, key: None
    fixture[1].calls.clear()
    pub(fixture, token=TOKEN, clock=lambda: 100)
    assert not existing.intersection(key for method, key in fixture[1].calls if method == 'put_new')


def test_08_stamp(fixture):
    db = fixture[0] / 'lcm.db'
    before = db.read_bytes()
    m = pub(fixture)
    with closing(sqlite3.connect(staged(fixture))) as conn:
        assert json.loads(conn.execute("SELECT value FROM metadata WHERE key='import:origin'").fetchone()[0]) == m['stamp']
        assert not conn.execute("SELECT 1 FROM metadata WHERE key='store_uuid'").fetchone()
    assert db.read_bytes() == before
    with pytest.raises(cc.AdmissionError, match='store_unstamped'):
        cc.admit(staged(fixture).parent, hermes_home=fixture[0].parent / 'unused-host')


def test_09_lock(fixture):
    lock = fixture[0] / 'cloud' / 'upload.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refused('lock_held', pub, fixture)
        assert not staged(fixture).exists()
    leftovers = staged(fixture).parent
    leftovers.mkdir(exist_ok=True)
    (leftovers / 'leftover').touch()
    pub(fixture)
    assert not (leftovers / 'leftover').exists()


def test_10_writer_during_the_copy(fixture):
    done = []
    def progress(*args):
        if not done:
            done.append(True)
            with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn, conn:
                conn.executemany('INSERT INTO messages(session_id,role,content,timestamp) VALUES (?,?,?,?)',
                                 [('race', 'user', str(i), 1) for i in range(4)])
    pub(fixture, progress=progress)
    with closing(sqlite3.connect(staged(fixture))) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        assert conn.execute("SELECT count(*) FROM messages WHERE session_id='race'").fetchone()[0] in (0, 4)
    assert done


def test_11_journal_mode(fixture):
    with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute("UPDATE metadata SET value=value WHERE key='schema_version'")
        conn.commit()
        before = (fixture[0] / 'lcm.db').read_bytes()
        pub(fixture)
        assert staged(fixture).read_bytes()[18] == 1
        assert (fixture[0] / 'lcm.db').read_bytes() == before


def test_12_payload_staged_before_upload(fixture):
    def remove(method, key):
        if method == 'put_new' and '/payloads/' in key:
            for path in (fixture[0] / cp.PAYLOAD_FOLDER).iterdir():
                path.unlink()
    fixture[1].hook = remove
    m = pub(fixture)
    assert m['payloads'] and all(fetch(fixture, p['key']) for p in m['payloads'])


def test_13_unresolved_references(fixture):
    minted = '20261101_000000_tool_0123456789ab_a.json'
    with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn, conn:
        conn.execute('UPDATE messages SET content=?,tool_calls=NULL', (' '.join(
            '[Externalized tool output: x; ref=' + n + ']' for n in ('...', '{ref}', minted)),))
    m = pub(fixture)
    assert [r['name'] for r in m['unresolved_minted']] == [minted]
    assert [r['name'] for r in m['unresolved_quoted']] == ['...', '{ref}']
    assert all(r['sessions'] == ['s1'] for r in m['unresolved_quoted']) and not m['payloads']


def test_14_forget_before_the_manifest(fixture):
    fired = []
    def forget(method, key):
        if method == 'list' and key.endswith('/parts/') and not fired:
            fired.append(True)
            with capture_store(fixture[0] / 'lcm.db', fixture[0], fixture[3]) as store:
                store.delete_session_messages('s1')
    fixture[1].hook = forget
    m = pub(fixture)
    rebuilt = rebuild(fixture, m, 'forgotten')
    assert rebuilt == staged(fixture).read_bytes() and not m['payloads']
    with closing(sqlite3.connect(staged(fixture))) as conn:
        assert conn.execute('SELECT count(*) FROM messages').fetchone() == (0,)


def test_15_manifest_last_created_once(fixture):
    first = pub(fixture)
    grow(fixture)
    before_state, before_last = state(fixture).read_bytes(), last(fixture).read_bytes()
    before = {k: fetch(fixture, k[len(PREFIX)+1:]) for k in keys(fixture)}
    original = fixture[1].inner.put_new
    def collide(key, data):
        if key.endswith('/gen/2.json'):
            original(key, b'{}')
        return original(key, data)
    fixture[1].inner.put_new = collide
    refused('chain_conflict', pub, fixture)
    assert state(fixture).read_bytes() == before_state and last(fixture).read_bytes() == before_last
    assert all(fetch(fixture, k[len(PREFIX)+1:]) == v for k, v in before.items())
    assert first['generation'] == 1
    assert not any(method == 'delete' for method, _ in fixture[1].calls)


def test_16_crash_before_the_manifest(fixture):
    first = pub(fixture)
    grow(fixture)
    def crash(method, key):
        if method == 'put_new' and key.endswith('/gen/2.json'):
            raise OSError('injected crash')
    fixture[1].hook = crash
    previous_state, previous_last = state(fixture).read_bytes(), last(fixture).read_bytes()
    with pytest.raises(OSError):
        pub(fixture)
    assert state(fixture).read_bytes() == previous_state and last(fixture).read_bytes() == previous_last
    assert PREFIX + '/gen/2.json' not in keys(fixture)
    assert json.loads(fetch(fixture, 'gen/1.json')) == first
    fixture[1].hook = lambda method, key: None
    assert pub(fixture)['generation'] == 2


def test_17_disk_floor(fixture):
    refused('disk_floor', pub, fixture, free_bytes=lambda _: 0)
    assert not staged(fixture).exists()
    pub(fixture)
    grow(fixture)
    readings = iter([10**9, 0])
    fixture[1].calls.clear()
    refused('disk_floor', pub, fixture, free_bytes=lambda _: next(readings))
    assert not any(method == 'put_new' for method, _ in fixture[1].calls)


def fingerprint(home):
    paths = [home / 'lcm.db']
    for name in ('lcm-large-outputs', 'spool'):
        if (home / name).exists():
            paths.extend((home / name).rglob('*'))
    return {p.relative_to(home).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns, sha(p.read_bytes()))
            for p in paths if p.is_file()}


def test_18_live_home_untouched(fixture):
    spool = fixture[0] / 'spool'
    spool.mkdir(exist_ok=True)
    (spool / 'pending.json').write_text('{}')
    before = fingerprint(fixture[0])
    pub(fixture)
    assert fingerprint(fixture[0]) == before and not (fixture[0] / 'capture.lock').exists()
    with (fixture[0] / 'capture.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pub(fixture, quiet_s=0)
    assert fingerprint(fixture[0]) == before


def test_19_quiet_window(fixture):
    home = fixture[0]
    pub(fixture, quiet_s=0)
    with closing(sqlite3.connect(home / 'lcm.db')) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('SELECT count(*) FROM messages').fetchone()
        refused('not_quiet', pub, fixture, quiet_s=60, clock=lambda: (home / 'lcm.db').stat().st_mtime + 100)
        assert cc.to_rollback_journal(conn)
    pub(fixture, quiet_s=60, clock=lambda: (home / 'lcm.db').stat().st_mtime + 100)
    for suffix in ('spool', 'wal', 'journal', 'mtime'):
        with pytest.MonkeyPatch.context() as patch:
            if suffix == 'spool':
                (home / 'spool').mkdir(exist_ok=True)
                path = home / 'spool' / 'pending.claimed'
                path.touch()
            elif suffix in ('wal', 'journal'):
                path = home / ('lcm.db-' + suffix)
                path.touch()
            else:
                path = None
            refused('not_quiet', pub, fixture, quiet_s=60, clock=lambda: (home / 'lcm.db').stat().st_mtime + (0 if suffix == 'mtime' else 100))
            if path:
                path.unlink()
            patch.setattr(cp, '_after_backup', lambda: (home / 'spool' / 'race.json').touch())
            if suffix == 'mtime':
                refused('not_quiet', pub, fixture, quiet_s=60, clock=lambda: (home / 'lcm.db').stat().st_mtime + 100)


def test_20_wal_without_sidecars(fixture):
    path = fixture[0] / 'lcm.db'
    data = bytearray(path.read_bytes())
    data[18:20] = b'\2\2'
    path.write_bytes(data)
    before = sorted(p.name for p in fixture[0].iterdir())
    refused('wal_without_sidecars', pub, fixture)
    assert sorted(p.name for p in fixture[0].iterdir()) == before


def test_21_no_guard_no_second_generation(fixture):
    pub(fixture)
    for kw in ({}, {'guard': lambda *args: None}, {'forget_items': lambda *args: []}):
        refused('guard_not_installed', cp.publish, fixture[0], COPY, fixture[1], PREFIX, **kw)
    result = cli(fixture)
    assert result.returncode == 2 and json.loads(result.stdout)['code'] == 'guard_not_installed'


def test_22_guard_block(fixture):
    pub(fixture)
    db = staged(fixture)
    payloads = db.parent / cp.PAYLOAD_FOLDER
    expected = cp.guard_block(db, payloads)
    copied = fixture[0].parent / 'guard-copy.db'
    shutil.copyfile(db, copied)
    assert expected and expected == cp.guard_block(copied, payloads)
    for sql in ('DELETE FROM messages WHERE store_id=(SELECT min(store_id) FROM messages)',
                "UPDATE messages SET content='changed' WHERE store_id=(SELECT min(store_id) FROM messages)",
                'DELETE FROM host_uid_bindings', 'DELETE FROM message_relations', 'DELETE FROM capture_events',
                "UPDATE capture_events SET raw=? WHERE raw IS NOT NULL"):
        copied = fixture[0].parent / ('guard-' + str(len(sql)) + '.db')
        shutil.copyfile(db, copied)
        with closing(sqlite3.connect(copied)) as conn, conn:
            if 'raw=?' in sql:
                conn.execute(sql, (zlib.compress(b'cut'),))
            else:
                conn.execute(sql)
        assert cp.guard_block(copied, payloads) != expected
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("UPDATE host_uid_bindings SET first_check='yes',disagree_seen=9")
        conn.execute("UPDATE capture_events SET notes='mutable'")
    assert cp.guard_block(db, payloads) == expected
    path = next(payloads.iterdir())
    data = json.loads(path.read_text())
    data['content'] = 'changed payload'
    path.unlink()  # keep the live hard link untouched
    path.write_text(json.dumps(data))
    assert cp.guard_block(db, payloads) != expected


def test_23_id_set_codec():
    for ids in ([], [1, 2, 4], list(range(1, 1000001)), list(range(1, 100000001, 1000))):
        encoded = cp.encode_ids(ids)
        assert cp.decode_ids(encoded) == ids


def test_24_names(fixture):
    refused('bad_name', fixture[1].get, 'outside/' + PREFIX + '/gen/1.json')
    refused('bad_name', cp.publish, fixture[0], '../copy', fixture[1], PREFIX)
    pub(fixture)
    path = fixture[1].inner.root / PREFIX / 'gen/1.json'
    m = json.loads(path.read_text())
    m['payloads'][0]['name'] = 'nested/name'
    path.write_text(json.dumps(m))
    refused('bad_name', pub, fixture)


def test_26_newer_head(fixture):
    pub(fixture)
    path = fixture[1].inner.root / PREFIX / 'gen/1.json'
    original = path.read_bytes()
    for key, code in (('layout', 'newer_layout'), ('contract', 'replica_newer_contract')):
        m = json.loads(original)
        if key == 'layout':
            m['layout'] = 2
        else:
            m['stamp']['contract']['version'] = 2
        path.write_text(json.dumps(m))
        fixture[1].calls.clear()
        refused(code, pub, fixture)
        assert not any(method == 'put_new' for method, _ in fixture[1].calls)


def test_27_database_block_of_a_base(fixture):
    m = pub(fixture)
    db = m['database']
    data = staged(fixture).read_bytes()
    p = int.from_bytes(data[16:18], 'big')
    assert db['file_sha256'] == sha(data)
    assert db['page_list_sha256'] == sha(b''.join(hashlib.sha256(data[i:i+p]).digest() for i in range(0, len(data), p)))
    assert db['base']['reason'] == 'first' and db['base']['generation'] == m['generation']
    assert db['base']['file_bytes'] == db['file_bytes'] and db['base']['file_sha256'] == db['file_sha256']
    assert not db['deltas'] and m['forgotten'] == []


def test_28_database_block_of_a_delta(fixture):
    pub(fixture)
    grow(fixture)
    first = pub(fixture)
    grow(fixture)
    m = pub(fixture)
    db = m['database']
    assert db['base'] == first['database']['base']
    assert db['deltas'][:-1] == first['database']['deltas']
    tail = db['deltas'][-1]
    assert tail['generation'] == m['generation'] and tail['file_pages'] * db['page_size'] == db['file_bytes']
    for obj in tail['objects']:
        data = fetch(fixture, obj['key'])
        assert len(data) == obj['bytes'] and int.from_bytes(data[12:16], 'big') == obj['records']


def test_29_every_generation_rebuilds(fixture):
    lengths = []
    for i in range(4):
        if i == 2:
            with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn, conn:
                conn.execute("UPDATE messages SET content='same size update' WHERE store_id=1")
        elif i:
            grow(fixture, count=40 if i == 1 else 1)
        m = pub(fixture)
        lengths.append(m['database']['file_bytes'])
        assert rebuild(fixture, m, f'rebuilt-{i}') == staged(fixture).read_bytes()
        assert len(m['database']['deltas']) == i
    assert lengths[1] > lengths[0] and lengths[2] == lengths[1]


def test_30_the_page_list_decides(fixture):
    pub(fixture)
    last(fixture).unlink()
    grow(fixture)
    fixture[1].calls.clear()
    m = pub(fixture)
    assert m['database']['base']['reason'] == 'page_list'
    assert not any(method == 'get' and '/parts/' in key for method, key in fixture[1].calls)
    grow(fixture)
    assert pub(fixture)['database']['deltas']
    other = fixture[0].parent / 'other-staging.db'
    shutil.copyfile(staged(fixture), other)
    with closing(sqlite3.connect(other)) as conn, conn:
        conn.execute("UPDATE messages SET content='another staging copy' WHERE store_id=1")
    cd.save_pending_list(last(fixture).parent, COPY, cd.hash_pages(other).digests)
    cd.promote_pending_list(last(fixture).parent, COPY)
    grow(fixture)
    assert pub(fixture)['database']['base']['reason'] == 'page_list'


def test_31_the_page_lists_order(fixture, monkeypatch):
    pub(fixture)
    grow(fixture)
    old = last(fixture).read_bytes()
    def check(method, key):
        if method == 'put_new':
            assert last(fixture).with_suffix('.pending').exists()
            assert last(fixture).read_bytes() == old
    fixture[1].hook = check
    with monkeypatch.context() as patch:
        patch.setattr(cd, 'promote_pending_list', lambda *args: (_ for _ in ()).throw(OSError('crash after manifest')))
        with pytest.raises(OSError):
            pub(fixture)
    fixture[1].hook = lambda *args: None
    grow(fixture)
    m = pub(fixture)
    assert m['generation'] == 3 and len(m['database']['deltas']) == 2


def remove_at_check(f, key):
    fired = []
    def hook(method, prefix):
        if method == 'list' and prefix.endswith('/parts/') and not fired:
            fired.append(True)
            (f[1].inner.root / PREFIX / key).unlink()
    f[1].hook = hook


def test_32_the_chain_check(fixture):
    m = pub(fixture)
    for index, kind in enumerate(('part', 'old-delta', 'own-delta')):
        current_token = f'{index + 10:032x}'
        grow(fixture)
        if kind == 'old-delta':
            m = pub(fixture)
            grow(fixture)
        fixture[1].calls.clear()
        if kind == 'own-delta':
            fired = []
            def own(method, prefix):
                if method == 'list' and prefix.endswith('/parts/') and not fired:
                    fired.append(True)
                    path = next(p for p in (fixture[1].inner.root / PREFIX / 'deltas').iterdir() if p.name.endswith(current_token))
                    path.unlink()
            fixture[1].hook = own
        else:
            key = m['database']['base']['parts'][0]['key'] if kind == 'part' else m['database']['deltas'][0]['objects'][0]['key']
            remove_at_check(fixture, key)
        m = pub(fixture, token=current_token)
        assert bool(m['database']['deltas']) == (kind == 'own-delta')
        if kind != 'own-delta':
            assert m['database']['base']['reason'] == 'missing_object'
        assert all(PREFIX + '/' + r['key'] in keys(fixture) for r in objects(m))
        lists = [key.rsplit('/', 2)[-2] for method, key in fixture[1].calls if method == 'list' and not key.endswith('/gen/')]
        assert lists == ['parts', 'deltas', 'payloads'] * (1 if kind == 'own-delta' else 2)
        assert not any(method == 'stat' for method, _ in fixture[1].calls)
        fixture[1].hook = lambda *args: None


ITEM = dict(key='s1', kind='forgotten', session='s1', chain_may_hold=True)


def test_33_fixed_once_last(fixture, monkeypatch):
    m = pub(fixture)
    grow(fixture)
    remove_at_check(fixture, m['database']['base']['parts'][0]['key'])
    calls = []
    def items(*args):
        calls.append(True)
        return [] if len(calls) == 1 else [ITEM]
    reasons = []
    original = cd.base_reason
    def decide(*args, **kw):
        answer = original(*args, **kw)
        reasons.append(answer)
        return answer
    monkeypatch.setattr(cd, 'base_reason', decide)
    m = pub(fixture, forget_items=items)
    assert reasons == [None, None, 'missing_object']
    assert len(calls) == 2 and not m['database']['deltas']
    assert m['database']['base']['file_sha256'] == m['database']['file_sha256']
    assert m['forgotten'][0]['base_generation'] == m['generation']


def test_34_size_and_length(fixture):
    pub(fixture)
    grow(fixture)
    fixture[1].calls.clear()
    m = pub(fixture, size_factor=0.000001)
    assert m['database']['base']['reason'] == 'size'
    assert not any(method == 'put_new' and '/deltas/' in key for method, key in fixture[1].calls)
    grow(fixture)
    assert pub(fixture, max_generations=1)['database']['base']['reason'] == 'length'


def test_35_staged_file_not_written_after_hash(fixture, monkeypatch):
    real = cd.hash_pages
    observed = []
    def hashed(path, **kw):
        result = real(path, **kw)
        observed.append((path.read_bytes(), path.stat().st_mtime_ns))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(cd, 'hash_pages', hashed)
        pub(fixture)
    assert observed == [(staged(fixture).read_bytes(), staged(fixture).stat().st_mtime_ns)]
    for delta, code in ((False, 'staging_changed'), (True, 'delta_selfcheck')):
        grow(fixture)
        def tamper(path, **kw):
            result = real(path, **kw)
            data = bytearray(path.read_bytes())
            previous = last(fixture).read_bytes()
            changed = cd.changed_pages(previous, result.digests)
            index = changed[0] if delta else next(i for i in range(result.pages) if i not in changed)
            data[(index + 1) * result.page_size - 1] ^= 1
            path.write_bytes(data)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(cd, 'hash_pages', tamper)
            refused(code, pub, fixture, **({} if delta else {'size_factor': 0}))
        assert PREFIX + '/gen/2.json' not in keys(fixture)


def test_36_a_forget_races_a_delta(fixture):
    pub(fixture)
    grow(fixture)
    calls = []
    def items(*args):
        calls.append(True)
        return [] if len(calls) == 1 else [ITEM]
    m = pub(fixture, forget_items=items)
    assert len(calls) == 4 and m['database']['base']['reason'] == 'forget'
    assert m['forgotten'][0]['base_generation'] == m['generation']


def test_37_a_forget_races_a_base(fixture):
    calls = []
    def items(*args):
        calls.append(True)
        return [] if len(calls) == 1 else [ITEM]
    m = pub(fixture, forget_items=items)
    assert len(calls) == 2 and m['forgotten'][0]['base_generation'] == 1


def test_38_items_that_prune_nothing(fixture):
    item = ITEM | dict(key='decision', kind='as-delete', chain_may_hold=False)
    m = pub(fixture, forget_items=lambda *args: [item])
    assert m['forgotten'][0]['base_generation'] is None
    grow(fixture)
    m = pub(fixture, forget_items=lambda *args: [item | dict(key='decision2')])
    assert m['database']['deltas'] and m['forgotten'][-1]['base_generation'] is None


def test_39_forgotten_is_cumulative(fixture):
    first = pub(fixture, forget_items=lambda *args: [ITEM | dict(key='z', chain_may_hold=False)])
    grow(fixture)
    m = pub(fixture, forget_items=lambda *args: [ITEM | dict(key='a', kind='as-delete', chain_may_hold=False)])
    assert m['forgotten'][-1] == first['forgotten'][0]
    assert [(r['kind'], r['key']) for r in m['forgotten']] == [('as-delete', 'a'), ('forgotten', 'z')]


def test_40_the_timers_skip(fixture):
    ready(fixture)
    fixture[1].calls.clear()
    before = staged(fixture).stat().st_mtime_ns
    refused('unchanged', pub, fixture, timer=True, clock=lambda: 101)
    assert not fixture[1].calls and staged(fixture).stat().st_mtime_ns == before
    data = json.loads(state(fixture).read_text())
    data['last_publish_s'] = time.time()
    state(fixture).write_text(json.dumps(data))
    result = cli(fixture, '--timer')
    assert result.returncode == 3 and json.loads(result.stdout)['code'] == 'unchanged'
    assert pub(fixture, clock=lambda: 102)['generation'] == 2
    grow(fixture)
    assert pub(fixture, timer=True, clock=lambda: 103)['generation'] == 3


def test_41_each_blocker_publishes(fixture):
    ready(fixture)
    for field, value, blocker in (('unpublished_decisions', ['d'], 'decision_pending'),
                                  ('pending_pair', [99, TOKEN], 'attempt_pending'),
                                  ('retention_done_generation', 99, 'retention_pending'),
                                  ('last_publish_s', -86400, 'ceiling')):
        ready(fixture)
        data = json.loads(state(fixture).read_text())
        data[field] = value
        state(fixture).write_text(json.dumps(data))
        m = pub(fixture, timer=True, clock=lambda: 101)
        assert m['blocker'] == blocker
    data = json.loads(state(fixture).read_text())
    for field in ('last_pair', 'pending_pair', 'retention_done_generation', 'unpublished_decisions'):
        data.pop(field, None)
    state(fixture).write_text(json.dumps(data))
    assert pub(fixture, timer=True, clock=lambda: 102)['blocker'] == 'attempt_pending'


def test_42_state_json(fixture, monkeypatch):
    assert not state(fixture).exists()
    signature = cd.store_signature(fixture[0])
    pub(fixture, clock=lambda: 100)
    assert json.loads(state(fixture).read_text()) == dict(copy_id=COPY, snapshot_signature=signature, last_publish_s=100)
    for identity in ({'copy_id': 'another'}, {}):
        state(fixture).write_text(json.dumps(identity | dict(unknown=True, last_pair=[1, TOKEN])))
        pub(fixture, timer=True, clock=lambda: 101)
        assert set(json.loads(state(fixture).read_text())) == {'copy_id', 'snapshot_signature', 'last_publish_s'}
    ready(fixture)
    data = json.loads(state(fixture).read_text()) | {'unknown': {'keep': [1, 2]}}
    state(fixture).write_text(json.dumps(data))
    before = state(fixture).read_bytes()
    refused('unchanged', pub, fixture, timer=True, clock=lambda: 101)
    assert state(fixture).read_bytes() == before
    def deny(*args):
        raise cp.Refused('as_delete_rows_present')
    refused('as_delete_rows_present', cp.publish, fixture[0], COPY, fixture[1], PREFIX,
            guard=deny, forget_items=lambda *args: [])
    assert state(fixture).read_bytes() == before
    def fail_upload(method, key):
        if method == 'put_new' and '/gen/' in key:
            raise OSError('injected failed publish')
    fixture[1].hook = fail_upload
    with pytest.raises(OSError):
        pub(fixture)
    assert state(fixture).read_bytes() == before
    fixture[1].hook = lambda *args: None
    pub(fixture, clock=lambda: 102)
    after = json.loads(state(fixture).read_text())
    assert all(after[k] == v for k, v in data.items() if k not in ('snapshot_signature', 'last_publish_s'))
    before = state(fixture).read_bytes()
    real = os.replace
    def fail(source, dest):
        if Path(dest) == state(fixture):
            raise OSError('injected state replace')
        return real(source, dest)
    with monkeypatch.context() as patch:
        patch.setattr(cp.os, 'replace', fail)
        with pytest.raises(OSError):
            pub(fixture)
    assert state(fixture).read_bytes() == before
    state(fixture).write_text('[]')
    refused('state_unreadable', pub, fixture)
    assert state(fixture).read_text() == '[]'


def test_43_codecs_refusals_pass_through(fixture, monkeypatch, capsys):
    def fail(*args, **kw):
        raise cd.DeltaRefused('page_size')
    monkeypatch.setattr(cd, 'hash_pages', fail)
    refused('page_size', pub, fixture)
    from scripts import lcm_x_cloud
    assert lcm_x_cloud.main(['snapshot', '--home', str(fixture[0]), '--copy-id', COPY,
                             '--bucket-dir', str(fixture[1].inner.root), '--prefix', 'copies']) == 2
    assert json.loads(capsys.readouterr().out)['code'] == 'page_size'
    assert not keys(fixture)


def test_44_imports(fixture):
    tree = ast.parse((ROOT / 'cloud_copy.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split('.')[0] in sys.stdlib_module_names for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.level == 1 and node.module is None and [n.name for n in node.names] == ['cloud_delta']) or (
                node.level == 0 and node.module.split('.')[0] in sys.stdlib_module_names)
    result = subprocess.run([sys.executable, '-I', '-S', str(ROOT / 'scripts/lcm_x_cloud.py'), '--help'], capture_output=True)
    assert result.returncode == 0
    result = subprocess.run([sys.executable, '-I', '-S', str(ROOT / 'scripts/lcm_x_cloud.py'),
        'snapshot', '--home', str(fixture[0]), '--copy-id', COPY, '--bucket-dir', str(fixture[1].inner.root),
        '--prefix', 'copies'], capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)['code'] == 'done'
    assert staged(fixture).exists() and not keys(fixture) and not state(fixture).exists()


def test_45_malformed_head_refused_before_upload(fixture):
    pub(fixture)
    grow(fixture)
    pub(fixture)
    path = fixture[1].inner.root / PREFIX / 'gen/2.json'
    original = path.read_bytes()
    for broken, code in (('generation', 'chain_generation'), ('base', 'bad_database_block')):
        m = json.loads(original)
        if broken == 'base':
            del m['database']['base']
        else:
            m['database']['deltas'][-1]['generation'] = 99
        path.write_text(json.dumps(m))
        before = keys(fixture)
        fixture[1].calls.clear()
        refused(code, pub, fixture)
        assert keys(fixture) == before and not any(method == 'put_new' for method, _ in fixture[1].calls)
        result = cli(fixture)
        assert result.returncode == 2 and json.loads(result.stdout)['code'] == code


def test_46_uploaded_bytes_are_checked_bytes(fixture, monkeypatch):
    pub(fixture)
    grow(fixture)
    real = cd.encode_delta
    rewritten = []
    def change(*args, **kw):
        result = real(*args, **kw)
        item = result[0]
        data = item.path.read_bytes()
        data = data[:16] + zlib.compress(zlib.decompress(data[16:]), 0)
        item.path.write_bytes(data)
        rewritten.append(data)
        return result
    monkeypatch.setattr(cd, 'encode_delta', change)
    refused('staging_changed', pub, fixture)
    assert rewritten and not any(fetch(fixture, key[len(PREFIX)+1:]) == rewritten[0] for key in keys(fixture))
