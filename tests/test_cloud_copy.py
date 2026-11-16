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
        self.checked = Path(root).parent / 'checked-copy'

    def __getattr__(self, method):
        def call(*args):
            key = args[0] if args else None
            self.calls.append((method, key))
            if method == 'put_new' and '/gen/' in key:
                source = self.inner.root.parent / 'home' / 'cloud' / 'staging'
                shutil.copytree(source, self.checked, dirs_exist_ok=True)
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
    assert result['database']['file_bytes'] < 5_000_000
    return result


def staged(f):
    return f[0] / 'cloud' / 'staging' / 'lcm.db'


def checked(f):
    # Saved at manifest creation, before publish's required staging cleanup.
    return f[1].checked / 'lcm.db'


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
    return subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'scripts/lcm_x_cloud.py'),
                           'publish', '--home', str(f[0]), '--copy-id', COPY,
                           '--bucket-dir', str(f[1].inner.root), '--prefix', 'copies', *extra],
                          capture_output=True, text=True)


def test_01_generation_1_is_a_whole_base(fixture):
    junk = fixture[0] / cp.PAYLOAD_FOLDER / 'unreferenced.json'
    junk.parent.mkdir(exist_ok=True)
    junk.write_text('{"content":"unused"}')
    m = pub(fixture, part_size=8192)
    assert rebuild(fixture, m, 'rebuilt') == checked(fixture).read_bytes()
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
    with pytest.raises(cp.Refused) as caught:
        pub(fixture)
    assert caught.value.code == 'records_pair_unsupported'
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
    with closing(sqlite3.connect(checked(fixture))) as conn:
        assert json.loads(conn.execute("SELECT value FROM metadata WHERE key='import:origin'").fetchone()[0]) == m['stamp']
        assert not conn.execute("SELECT 1 FROM metadata WHERE key='store_uuid'").fetchone()
    assert db.read_bytes() == before
    with pytest.raises(cc.AdmissionError, match='store_unstamped'):
        cc.admit(checked(fixture).parent, hermes_home=fixture[0].parent / 'unused-host')


def test_09_lock(fixture):
    lock = fixture[0] / 'cloud' / 'upload.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(cp.Refused) as caught:
            pub(fixture)
        assert caught.value.code == 'lock_held'
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
    with closing(sqlite3.connect(checked(fixture))) as conn:
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
        assert checked(fixture).read_bytes()[18] == 1
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
    assert rebuilt == checked(fixture).read_bytes() and not m['payloads']
    with closing(sqlite3.connect(fixture[0].parent / 'forgotten')) as conn:
        for table in ('messages', 'capture_events'):
            assert conn.execute(f'SELECT count(*) FROM {table} WHERE session_id=?', ('s1',)).fetchone() == (0,)
    assert 's1' not in m['guard']['sessions']
    assert not m['guard']['payloads']
    assert m['unresolved_minted'] == m['unresolved_quoted'] == m['payloads'] == []
    assert not m['database']['deltas']
    assert json.loads(fetch(fixture, 'gen/1.json')) == m
    assert fired == [True]


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
    with pytest.raises(cp.Refused) as caught:
        pub(fixture, free_bytes=lambda _: 0)
    assert caught.value.code == 'disk_floor'
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
    with pytest.raises(cp.Refused) as caught:
        pub(fixture)
    assert caught.value.code == 'wal_without_sidecars'
    assert sorted(p.name for p in fixture[0].iterdir()) == before


def test_21_no_guard_no_second_generation(fixture):
    pub(fixture)
    for kw in ({}, {'guard': lambda *args: None}, {'forget_items': lambda *args: []}):
        refused('guard_not_installed', cp.publish, fixture[0], COPY, fixture[1], PREFIX, **kw)
    result = cli(fixture)
    assert result.returncode == 2 and json.loads(result.stdout)['code'] == 'guard_not_installed'


def test_22_guard_block(fixture):
    pub(fixture)
    db = checked(fixture)
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
        with pytest.raises(cp.Refused) as caught:
            pub(fixture)
        assert caught.value.code == code
        assert not any(method == 'put_new' for method, _ in fixture[1].calls)


def test_27_database_block_of_a_base(fixture):
    m = pub(fixture)
    db = m['database']
    data = checked(fixture).read_bytes()
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
        assert rebuild(fixture, m, f'rebuilt-{i}') == checked(fixture).read_bytes()
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
    shutil.copyfile(checked(fixture), other)
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
    assert observed == [(checked(fixture).read_bytes(), checked(fixture).stat().st_mtime_ns)]
    for delta, code in ((False, 'staging_changed'), (True, 'delta_selfcheck')):
        grow(fixture)
        def tamper(path, **kw):
            result = real(path, **kw)
            data = bytearray(path.read_bytes())
            previous = last(fixture).read_bytes()
            changed = cd.changed_pages(previous, result.digests)
            index = changed[0] if delta else next(i for i in range(result.pages) if i not in changed)
            data[(index + 1) * result.page_size - 1] ^= 1
            metadata = path.stat()
            path.write_bytes(data)
            os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            return result
        with monkeypatch.context() as patch:
            patch.setattr(cd, 'hash_pages', tamper)
            with pytest.raises(cp.Refused) as caught:
                pub(fixture, **({} if delta else {'size_factor': 0}))
            assert caught.value.code == code
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


def test_40_the_timers_skip(fixture, monkeypatch):
    ready(fixture)
    fixture[1].calls.clear()
    stage_calls, real_stage = [], cp._stage
    def spy(*args, **kw):
        stage_calls.append(True)
        return real_stage(*args, **kw)
    monkeypatch.setattr(cp, '_stage', spy)
    refused('unchanged', pub, fixture, timer=True, clock=lambda: 101)
    assert not fixture[1].calls and not stage_calls
    data = json.loads(state(fixture).read_text())
    data['last_publish_s'] = time.time()
    state(fixture).write_text(json.dumps(data))
    result = cli(fixture, '--timer')
    assert result.returncode == 3 and json.loads(result.stdout)['code'] == 'unchanged'
    assert pub(fixture, clock=lambda: 102)['generation'] == 2
    grow(fixture)
    assert pub(fixture, timer=True, clock=lambda: 103)['generation'] == 3


def test_41_each_blocker_publishes(fixture, monkeypatch, capsys):
    from scripts import lcm_x_cloud
    real = cp.publish
    def allowed(*args, **kw):
        return real(*args, **(kw | dict(guard=lambda *a: None, forget_items=lambda *a: [], clock=lambda: 101)))
    monkeypatch.setattr(cp, 'publish', allowed)
    for field, value, blocker in (('unpublished_decisions', ['d'], 'decision_pending'),
                                  ('pending_pair', [99, TOKEN], 'attempt_pending'),
                                  ('retention_done_generation', 99, 'retention_pending'),
                                  ('last_publish_s', -86400, 'ceiling'),
                                  ('missing_pairs', None, 'attempt_pending')):
        ready(fixture)
        data = json.loads(state(fixture).read_text())
        if field == 'missing_pairs':
            for key in ('last_pair', 'pending_pair', 'retention_done_generation', 'unpublished_decisions'):
                data.pop(key, None)
        else:
            data[field] = value
        state(fixture).write_text(json.dumps(data))
        assert lcm_x_cloud.main(['publish', '--home', str(fixture[0]), '--copy-id', COPY,
            '--bucket-dir', str(fixture[1].inner.root), '--prefix', 'copies', '--timer']) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed['blocker'] == blocker
        manifest = json.loads(fetch(fixture, f"gen/{printed['generation']}.json"))
        assert 'blocker' not in manifest


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


def test_44_imports(fixture, monkeypatch):
    real = subprocess.run
    def isolated(command, **kw):
        assert '-I' in command and '-S' in command and '-B' in command
        return real(command, **kw)
    monkeypatch.setattr(subprocess, 'run', isolated)
    tree = ast.parse((ROOT / 'cloud_copy.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split('.')[0] in sys.stdlib_module_names for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.level == 1 and node.module is None and [n.name for n in node.names] == ['cloud_delta']) or (
                node.level == 0 and node.module.split('.')[0] in sys.stdlib_module_names)
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'scripts/lcm_x_cloud.py'), '--help'], capture_output=True)
    assert result.returncode == 0
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'scripts/lcm_x_cloud.py'),
        'snapshot', '--home', str(fixture[0]), '--copy-id', COPY, '--bucket-dir', str(fixture[1].inner.root),
        '--prefix', 'copies'], capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)['code'] == 'done'
    assert staged(fixture).exists() and not keys(fixture) and not state(fixture).exists()


@pytest.mark.parametrize('broken,code', [
    ('generation', 'chain_generation'), ('base', 'bad_database_block'),
    ('database_null', 'bad_database_block'), ('database_absent', 'bad_database_block'),
    ('database_list', 'bad_database_block'), ('head_list', 'bad_database_block'),
    ('stamp_absent', 'bad_head'), ('stamp_list', 'bad_head'), ('contract_null', 'bad_head'),
    ('version_list', 'bad_head'), ('layout_absent', 'bad_head'), ('layout_list', 'bad_head'),
    ('copy_id_absent', 'bad_head'), ('generation_absent', 'bad_head'), ('generation_list', 'bad_head'),
    ('payloads_absent', 'bad_head'), ('payloads_null', 'bad_head'), ('payload_list', 'bad_head'),
    ('payload_sha_absent', 'bad_head'), ('guard_list', 'bad_head'), ('guard_mark_list', 'bad_head'),
    ('forgotten_list', 'bad_head'), ('invalid_json', 'bad_head'),
])
def test_45_malformed_head_refused_before_upload(fixture, monkeypatch, broken, code):
    pub(fixture)
    grow(fixture)
    pub(fixture)
    path = fixture[1].inner.root / PREFIX / 'gen/2.json'
    m = json.loads(path.read_bytes())
    if broken == 'base':
        del m['database']['base']
    elif broken == 'generation':
        m['database']['deltas'][-1]['generation'] = 99
    elif broken == 'head_list':
        m = [m]
    elif broken.endswith('_absent'):
        field = broken.removesuffix('_absent')
        del (m['payloads'][0] if field == 'payload_sha' else m)['sha256' if field == 'payload_sha' else field]
    elif broken in ('database_null', 'database_list', 'stamp_list', 'layout_list', 'generation_list', 'payloads_null', 'guard_list'):
        m[broken.rsplit('_', 1)[0]] = None if broken.endswith('_null') else []
    elif broken == 'contract_null':
        m['stamp']['contract'] = None
    elif broken == 'version_list':
        m['stamp']['contract']['version'] = []
    elif broken == 'payload_list':
        m['payloads'] = [[]]
    elif broken == 'guard_mark_list':
        m['guard']['mark'] = []
    elif broken == 'forgotten_list':
        m['forgotten'] = [[]]
    path.write_text('{' if broken == 'invalid_json' else json.dumps(m))
    before = keys(fixture)
    fixture[1].calls.clear()
    result = cli(fixture)
    assert result.returncode == 2 and json.loads(result.stdout)['code'] == code
    assert json.loads(result.stdout)['generation'] == 2
    stage_calls, real_stage = [], cp._stage
    def spy(*args, **kw):
        stage_calls.append(True)
        return real_stage(*args, **kw)
    monkeypatch.setattr(cp, '_stage', spy)
    refused(code, pub, fixture)
    assert keys(fixture) == before and not any(method == 'put_new' for method, _ in fixture[1].calls)
    assert not stage_calls


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
    with pytest.raises(cp.Refused) as caught:
        pub(fixture)
    assert caught.value.code == 'staging_changed'
    assert rewritten and not any(fetch(fixture, key[len(PREFIX)+1:]) == rewritten[0] for key in keys(fixture))


@pytest.mark.parametrize('when', ['upload', 'backup'])
def test_r2_writer_closes_before_live_open(fixture, monkeypatch, when):
    home = fixture[0]
    staging = staged(fixture).parent
    staging.mkdir(exist_ok=True)
    (staging / 'leftover').touch()
    writer = sqlite3.connect(home / 'lcm.db')
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute("UPDATE metadata SET value=value WHERE key='schema_version'")
    writer.commit()
    def files():
        return {p.relative_to(home).as_posix(): (p.stat().st_size, sha(p.read_bytes()))
                for p in home.rglob('*') if p.is_file() and 'cloud' not in p.relative_to(home).parts}
    closed = []
    def close():
        if not closed:
            writer.close()
            closed.append(files())
            assert not (home / 'lcm.db-wal').exists() and not (home / 'lcm.db-shm').exists()
    if when == 'upload':
        fixture[1].hook = lambda method, key: close() if method == 'put_new' else None
    else:
        real = cp.shutil.rmtree
        def clear(path, **kw):
            real(path, **kw)
            close()
        monkeypatch.setattr(cp.shutil, 'rmtree', clear)
    try:
        with pytest.raises(cp.Refused) as caught:
            pub(fixture)
        assert caught.value.code == 'wal_without_sidecars'
        assert closed and files() == closed[0]
        assert not (home / 'lcm.db-wal').exists() and not (home / 'lcm.db-shm').exists()
        assert PREFIX + '/gen/1.json' not in keys(fixture)
    finally:
        writer.close()


def test_r2_transport_symlinked_ancestor(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    transport = cp.DirTransport(alias / 'deep' / 'bucket', PREFIX)
    key = PREFIX + '/gen/1.json'
    transport.put_new(key, b'{}')
    assert transport.root == (real / 'deep' / 'bucket').resolve()
    assert transport.get(key) == b'{}' and transport.list(PREFIX + '/gen/') == [key]
    outside = tmp_path / 'outside'
    outside.mkdir()
    (transport.root / PREFIX / 'parts').symlink_to(outside, target_is_directory=True)
    refused('bad_name', transport.list, PREFIX + '/parts/')
    refused('bad_name', transport.put_new, PREFIX + '/parts/' + 'a' * 64 + '.' + TOKEN, b'x')
    (transport.root / PREFIX / 'gen/2.json').symlink_to(outside / 'secret')
    for call in (transport.get, transport.stat):
        refused('bad_name', call, PREFIX + '/gen/2.json')
    refused('bad_name', transport.list, PREFIX + '/gen/')
    assert list(outside.iterdir()) == []


def test_r2_transport_ignores_dot_files(fixture):
    transport = fixture[1].inner
    key = PREFIX + '/gen/1.json'
    transport.put_new(key, b'{}')
    folder = transport.root / PREFIX / 'gen'
    (folder / '.upload-abandoned').write_bytes(b'partial')
    (folder / '.other').mkdir()
    assert transport.list(PREFIX + '/gen/') == [key]
    (folder / 'bad.json').write_bytes(b'{}')
    refused('bad_name', transport.list, PREFIX + '/gen/')


@pytest.mark.parametrize('generation', [1, 2])
@pytest.mark.parametrize('projected', [False, True], ids=['ledger-only', 'message-control'])
def test_r3_forgotten_ledger_session(fixture, monkeypatch, generation, projected):
    home, bucket, path, config = fixture
    planted = 'synthetic-ledger-only-round-three'
    rows = [dict(type='system', uuid='s2-system', subtype='informational', content=planted),
            dict(type='file-history-snapshot', messageId='s2-file', snapshot={'note': planted})]
    if projected:
        rows.append(entry('user', 's2-user', planted))
    transcript = path.with_name('s2.jsonl')
    transcript.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with capture_store(home / 'lcm.db', home, config) as store:
        assert cc.drain(store, cc.Lineage('s2', None, transcript)) == 'drained'
        assert store.connection.execute("SELECT count(*) FROM messages WHERE session_id='s2'").fetchone()[0] == int(projected)
        assert store.connection.execute("SELECT count(*) FROM capture_events WHERE session_id='s2'").fetchone()[0] == len(rows)
    if generation == 2:
        pub(fixture)
        grow(fixture)
    backups, fired = [], []
    monkeypatch.setattr(cp, '_after_backup', lambda: backups.append(True))
    def forget(method, key):
        if method == 'list' and key.endswith('/parts/') and not fired:
            fired.append(True)
            with capture_store(home / 'lcm.db', home, config) as store:
                store.delete_session_messages('s2')
    def items(head, db):
        with closing(cp._ro(home / 'lcm.db')) as live:
            exists = live.execute('SELECT 1 FROM metadata WHERE key=?', (cp.FORGOTTEN_PREFIX + 's2',)).fetchone()
        return [dict(key='s2', kind='forgotten', session='s2', chain_may_hold=True)] if exists else []
    bucket.hook = forget
    m = pub(fixture, forget_items=items, **({'size_factor': 0} if generation == 2 else {}))
    assert fired == [True] and len(backups) == 2
    assert m['generation'] == generation and m['database']['base']['generation'] == generation
    assert m['forgotten'] == [dict(key='s2', kind='forgotten', session='s2', base_generation=generation)]
    rebuild(fixture, m, 'ledger-rebuilt.db')
    with closing(sqlite3.connect(home.parent / 'ledger-rebuilt.db')) as conn:
        for table in ('messages', 'capture_events'):
            assert conn.execute(f'SELECT count(*) FROM {table} WHERE session_id=?', ('s2',)).fetchone() == (0,)
        for (raw,) in conn.execute('SELECT raw FROM capture_events WHERE raw IS NOT NULL'):
            assert planted.encode() not in zlib.decompress(raw)
    assert not staged(fixture).parent.exists()


@pytest.mark.parametrize('snapshot_only', [False, True], ids=['publish', 'snapshot'])
@pytest.mark.parametrize('setting', ['absolute', 'relative', 'export-double', 'single', 'mixed-first', 'mixed-last',
                                         'bom', 'bom-export', 'export-tab', 'quoted-key', 'formfeed', 'u2028'])
def test_r3_payload_folder_refused(fixture, snapshot_only, setting):
    home, bucket, _, _ = fixture
    key = 'LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH'
    other, default = str(home.parent / 'elsewhere'), str(home / cp.PAYLOAD_FOLDER)
    lines = {'absolute': f'{key}={other}', 'relative': f'{key}=relative/folder',
             'export-double': f'export {key}="{other}"', 'single': f"{key}='{other}'",
             'mixed-first': f'{key}={other}\n{key}={default}',
             'mixed-last': f'{key}={default}\n{key}={other}',
             'bom': f'\ufeff{key}={other}', 'bom-export': f'\ufeffexport {key}={other}',
             'export-tab': f'export\t{key}={other}', 'quoted-key': f"'{key}'={other}",
             'formfeed': f'{key}={default}\x0c/sub', 'u2028': f'{key}={default}\u2028/sub'}
    (home / '.env').write_text(lines[setting] + '\n')
    # Exercise both CLI exit classification and the library's pre-stage boundary.
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'scripts/lcm_x_cloud.py'),
        'snapshot' if snapshot_only else 'publish', '--home', str(home), '--copy-id', COPY,
        '--bucket-dir', str(bucket.inner.root), '--prefix', 'copies'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and json.loads(result.stdout)['code'] == 'payload_folder_configured'
    assert not staged(fixture).parent.exists() and not keys(fixture)
    with pytest.raises(cp.Refused) as caught:
        pub(fixture, snapshot_only=snapshot_only)
    assert caught.value.code == 'payload_folder_configured'
    assert not staged(fixture).parent.exists() and not keys(fixture)
    assert not any(method == 'put_new' for method, _ in bucket.calls)


def test_r3_payload_folder_defaults_proceed(fixture, monkeypatch):
    home = fixture[0]
    monkeypatch.setenv('HOME', str(home.parent))
    key = 'LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH'
    default = str(home / cp.PAYLOAD_FOLDER)
    env = home / '.env'
    settings = [None, '# comment\nUNRELATED=value', f'# {key}=elsewhere', f'{key}=',
                f'{key}= "" ', f"{key}=''", f'{key}={default}', f'{key}=~/home/{cp.PAYLOAD_FOLDER}',
                f' export {key}="{default}" ', f"{key}='{default}'"]
    for content in settings:
        if content is not None:
            env.write_text(content + '\n')
        m = pub(fixture, snapshot_only=True)
        assert m['payloads'] and staged(fixture).is_file() and not keys(fixture)
    pub(fixture)
    assert not staged(fixture).parent.exists()


def test_r3_disk_floor_counts_wal(fixture):
    home = fixture[0]
    with closing(sqlite3.connect(home / 'lcm.db')) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('PRAGMA wal_autocheckpoint=0')
        writer.executemany('INSERT INTO metadata(key,value) VALUES (?,?)',
                           [(f'synthetic-wal:{i}', 'x' * 4000) for i in range(32)])
        writer.commit()
        payload = sum(p.stat().st_size for p in (home / cp.PAYLOAD_FOLDER).iterdir())
        old_budget = (home / 'lcm.db').stat().st_size + payload + cp.DISK_MARGIN
        wal_bytes = (home / 'lcm.db-wal').stat().st_size
        assert wal_bytes > 0
        with pytest.raises(cp.Refused) as caught:
            pub(fixture, free_bytes=lambda _: old_budget + wal_bytes // 2)
        assert caught.value.code == 'disk_floor'
        assert not staged(fixture).parent.exists() and not keys(fixture)
        assert pub(fixture, free_bytes=lambda _: old_budget + wal_bytes + 1)['generation'] == 1


@pytest.mark.parametrize('outcome', ['success', 'staging_changed', 'chain_conflict', 'guard_not_installed',
                                    'guard_loss', 'disk_floor', 'schema_newer', 'page_size', 'transport_error'])
def test_r3_publish_cleans_staging(fixture, monkeypatch, outcome):
    home, bucket, _, _ = fixture
    if outcome in ('guard_not_installed', 'guard_loss', 'disk_floor'):
        pub(fixture)
        grow(fixture)
    # Snapshot is the control: it must keep its staged copy even after publish cleanup is added.
    pub(fixture, snapshot_only=True)
    assert staged(fixture).is_file()
    live = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (home / cp.PAYLOAD_FOLDER).iterdir()}
    real = cp.shutil.rmtree
    cleaned = []
    def clear(path, **kw):
        assert Path(path) == staged(fixture).parent
        with (home / 'cloud' / 'upload.lock').open('a') as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if kw.get('ignore_errors'):
            cleaned.append(True)
        return real(path, **kw)
    monkeypatch.setattr(cp.shutil, 'rmtree', clear)
    kw = dict(guard=lambda *args: None, forget_items=lambda *args: [])
    if outcome == 'guard_not_installed':
        kw = {}
    elif outcome == 'guard_loss':
        def guard(*args):
            raise cp.Refused('guard_loss')
        kw['guard'] = guard
    elif outcome == 'disk_floor':
        readings = iter([10**9, 0])
        kw['free_bytes'] = lambda _: next(readings)
    elif outcome == 'schema_newer':
        monkeypatch.setattr(cp, 'SCHEMA_VERSION', 0)
    elif outcome == 'page_size':
        def bad_hash(*args, **kwargs):
            raise cp.Refused('page_size')
        monkeypatch.setattr(cd, 'hash_pages', bad_hash)
    elif outcome == 'staging_changed':
        def tamper(method, key):
            if method == 'list' and key.endswith('/parts/'):
                with staged(fixture).open('ab') as handle:
                    handle.write(b'x')
        bucket.hook = tamper
    elif outcome == 'chain_conflict':
        def conflict(method, key):
            if method == 'put_new' and key.endswith('/gen/1.json'):
                bucket.inner.put_new(key, b'{}')
        bucket.hook = conflict
    elif outcome == 'transport_error':
        def outage(method, key):
            if method == 'put_new' and '/parts/' in key:
                raise OSError('synthetic outage')
        bucket.hook = outage
    if outcome == 'success':
        cp.publish(home, COPY, bucket, PREFIX, **kw)
    elif outcome == 'transport_error':
        with pytest.raises(OSError, match='synthetic outage'):
            cp.publish(home, COPY, bucket, PREFIX, **kw)
    else:
        with pytest.raises(cp.Refused) as caught:
            cp.publish(home, COPY, bucket, PREFIX, **kw)
        assert caught.value.code == outcome
    assert not staged(fixture).parent.exists()
    assert cleaned == [True]
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in live} == live


@pytest.mark.parametrize('field,code', [('layout', 'newer_layout'), ('contract', 'replica_newer_contract')])
def test_r3_head_check_order(fixture, field, code):
    pub(fixture)
    path = fixture[1].inner.root / PREFIX / 'gen/1.json'
    m = json.loads(path.read_bytes())
    m['database'] = None
    if field == 'layout':
        m['layout'] = 2
    else:
        m['stamp']['contract']['version'] = 2
    path.write_text(json.dumps(m))
    before = keys(fixture)
    fixture[1].calls.clear()
    with pytest.raises(cp.Refused) as caught:
        pub(fixture)
    assert caught.value.code == code and caught.value.detail['generation'] == 1
    assert keys(fixture) == before and not any(method == 'put_new' for method, _ in fixture[1].calls)


def test_r3_wal_skip_preserves_snapshot(fixture):
    pub(fixture, snapshot_only=True)
    db = fixture[0] / 'lcm.db'
    data = bytearray(db.read_bytes())
    data[18:20] = b'\2\2'
    db.write_bytes(data)
    assert not db.with_name('lcm.db-wal').exists() and not db.with_name('lcm.db-shm').exists()
    def contents():
        return {p.relative_to(staged(fixture).parent): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in staged(fixture).parent.rglob('*') if p.is_file()}
    before = contents()
    with pytest.raises(cp.Refused) as caught:
        pub(fixture, snapshot_only=True)
    assert caught.value.code == 'wal_without_sidecars'
    assert contents() == before
    assert staged(fixture).stat().st_size > 0


@pytest.mark.parametrize('code', ['bad_head', 'payload_folder_configured'])
def test_r3_documented_refusal_codes(code):
    assert code in cp.__doc__
    command_doc = ast.get_docstring(ast.parse((ROOT / 'scripts/lcm_x_cloud.py').read_text()))
    assert code in command_doc
    assert code not in cp.SKIPS


def test_r4_payload_folder_worker_parity(fixture, monkeypatch):
    from hermes_lcm import externalize, mcp_reader
    home = fixture[0]
    monkeypatch.delenv('LCM_HERMES_BASE_DIR', raising=False)
    key = 'LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH'
    other, default = str(home.parent / 'elsewhere'), str(home / cp.PAYLOAD_FOLDER)
    cases = {
        'bom_first_line_other': b'\xef\xbb\xbf' + f'{key}={other}\n'.encode(),
        'bom_first_line_default': b'\xef\xbb\xbf' + f'{key}={default}\n'.encode(),
        'bom_export_other': b'\xef\xbb\xbf' + f'export {key}={other}\n'.encode(),
        'export_tab_other': f'export\t{key}={other}\n'.encode(),
        'export_two_spaces_other': f'export  {key}={other}\n'.encode(),
        'leading_ws_export_other': f'   export {key}={other}\n'.encode(),
        'crlf_other': f'A=b\r\n{key}={other}\r\n'.encode(),
        'crlf_default': f'A=b\r\n{key}={default}\r\n'.encode(),
        'cr_only_other': f'A=b\r{key}={other}\r'.encode(),
        'latin1_unrelated_default': b'OTHER=caf\xe9\n' + f'{key}={default}\n'.encode(),
        'latin1_unrelated_other': b'OTHER=caf\xe9\n' + f'{key}={other}\n'.encode(),
        'latin1_unrelated_nokey': b'OTHER=caf\xe9\n',
        'inline_comment_default': f'{key}={default} # note\n'.encode(),
        'inline_comment_other': f'{key}={other} # note\n'.encode(),
        'dquote_escape_other': f'{key}="{other}\\tx"\n'.encode(),
        'dquote_default_trailing_comment': f'{key}="{default}" # c\n'.encode(),
        'spaces_around_eq_other': f'{key} = {other}\n'.encode(),
        'spaces_around_eq_default': f'{key} = {default}\n'.encode(),
        'tab_before_eq_other': f'{key}\t={other}\n'.encode(),
        'lowercase_key_other': f'{key.lower()}={other}\n'.encode(),
        'single_quoted_key_other': f"'{key}'={other}\n".encode(),
        'bare_key_unsets_after_other': f'{key}={other}\n{key}\n'.encode(),
        'formfeed_in_value': f'{key}={default}\x0c/sub\n'.encode(),
        'u2028_in_value': f'{key}={default}\u2028/sub\n'.encode(),
        'nbsp_leading_other': f'\u00a0{key}={other}\n'.encode(),
        'dollar_var_other': f'{key}=${{HOME}}/x\n'.encode(),
        'dquote_space_inside_default': f'{key}=" {default}"\n'.encode(),
    }
    failures = []
    assert len(cases) == 27
    for name, data in cases.items():
        (home / '.env').write_bytes(data)
        config = LCMConfig(large_output_externalization_path=mcp_reader.read_profile_env(home).get(key, ''))
        worker_folder = externalize.get_large_output_storage_dir(config, hermes_home=str(home), create=False)
        try:
            cp._before(home, 0, time.time, lambda _: 10**12)
        except cp.Refused as error:
            if error.code != 'payload_folder_configured':
                failures.append((name, error.code))
        except Exception as error:
            failures.append((name, type(error).__name__))
        else:
            if worker_folder != (home / cp.PAYLOAD_FOLDER).resolve():
                failures.append((name, 'false_proceed'))
    assert not failures, failures


def test_r4_payload_folder_latin1(fixture):
    home = fixture[0]
    key = 'LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH'
    failures = []
    for bom in (b'', b'\xef\xbb\xbf'):
        for moved in (False, True):
            data = bom + b'OTHER=caf\xe9\n'
            if moved:
                data += f'{key}={home.parent / "elsewhere"}\n'.encode()
            (home / '.env').write_bytes(data)
            try:
                cp._before(home, 0, time.time, lambda _: 10**12)
                outcome = 'proceed'
            except cp.Refused as error:
                outcome = error.code
            except Exception as error:
                outcome = type(error).__name__
            if outcome != ('payload_folder_configured' if moved else 'proceed'):
                failures.append((bool(bom), moved, outcome))
    assert not failures, failures


def test_r4_pre_stage_refusal_preserves_snapshot(fixture):
    home, bucket, _, _ = fixture
    pub(fixture, snapshot_only=True)
    folder = staged(fixture).parent
    def contents():
        return {p.relative_to(folder): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in folder.rglob('*') if p.is_file()}
    before = contents()
    config = home / 'cloud' / 'cloud.json'
    config.write_text(json.dumps(dict(copy_id='other')))
    with pytest.raises(cp.Refused) as caught:
        pub(fixture)
    assert caught.value.code == 'bad_name'
    assert contents() == before
    config.write_text(json.dumps(dict(copy_id=COPY)))
    pub(fixture)
    pub(fixture, snapshot_only=True)
    with pytest.raises(cp.Refused) as caught:
        cp.publish(home, COPY, bucket, PREFIX)
    assert caught.value.code == 'guard_not_installed'
    assert not folder.exists()


def test_r4_disk_floor_reclaims_staging(fixture, tmp_path):
    home = fixture[0]
    folder = staged(fixture).parent
    payload_bytes = sum(p.stat().st_size for p in (home / cp.PAYLOAD_FOLDER).iterdir())
    budget = (home / 'lcm.db').stat().st_size + payload_bytes + cp.DISK_MARGIN
    pub(fixture, snapshot_only=True)
    occupied = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    capacity = budget + occupied // 2
    # One staged copy fits in total capacity; keeping both would not fit.
    free = capacity - occupied
    assert free < budget <= capacity
    try:
        pub(fixture, snapshot_only=True, free_bytes=lambda _: free)
        outcome = 'proceed'
    except cp.Refused as error:
        outcome = error.code
    assert outcome == 'proceed'
    shutil.rmtree(folder)
    assert pub(fixture, snapshot_only=True, free_bytes=lambda _: capacity)['generation'] == 1
    occupied = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    refused('disk_floor', pub, fixture, snapshot_only=True, free_bytes=lambda _: budget - occupied - 1)
    assert {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()} == before
    shutil.rmtree(folder)
    refused('disk_floor', pub, fixture, snapshot_only=True, free_bytes=lambda _: budget - 1)
    # Nested regular files count; links to files or directories never add credit.
    outside = tmp_path / 'unowned'
    outside.mkdir()
    (outside / 'large').write_bytes(b'x' * 4096)
    folder.mkdir()
    (folder / 'nested').mkdir()
    (folder / 'nested' / 'regular').write_bytes(b'x' * 32)
    (folder / 'file-link').symlink_to(outside / 'large')
    (folder / 'dir-link').symlink_to(outside, target_is_directory=True)
    cp._before(home, 0, time.time, lambda _: budget - 32)
    refused('disk_floor', cp._before, home, 0, time.time, lambda _: budget - 33)
    shutil.rmtree(folder)
    folder.symlink_to(outside, target_is_directory=True)
    refused('disk_floor', cp._before, home, 0, time.time, lambda _: budget - 1)
    assert (outside / 'large').read_bytes() == b'x' * 4096
