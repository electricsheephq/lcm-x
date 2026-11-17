"""Lane 3 proofs on synthetic stores and create-only directory transport."""
import ast
from contextlib import closing
import copy
import hashlib
import json
import os
import shutil
import sqlite3
import sys

import pytest

from hermes_lcm import cloud_copy as cp, cloud_guard as cg, cloud_delta as cd, claude_code_store as cc
from tests.test_cloud_copy import (COPY, PREFIX, TOKEN, ROOT, grow, keys, objects,
                                  capture_store, entry, fetch, refused, state)

from tests.test_cloud_copy import fixture as _capture_fixture

fixture = _capture_fixture


OWNED = {
    **dict.fromkeys(('a1', 'a2', 'a3', 'a4'), ('test_forget_retention_and_rebuild',)),
    **dict.fromkeys(('b', 'e'), ('test_loss_surfaces', 'test_control_publisher_and_state')),
    **dict.fromkeys(('c', 'd', 'g2'), ('test_decisions_pins_and_pruning', 'test_pin_bucket_clock_expiry_and_missing_date')),
    **dict.fromkeys(('f1', 'j4', 'j5'), ('test_twin_interleavings',)),
    'f2': ('test_recreated_number_below_head_never_retains',),
    'h': ('test_newest_forget_then_capture',),
    **dict.fromkeys(('i1', 'i2'), ('test_rollback_before_after_mark', 'test_reuse_priority', 'test_forgotten_returned_priority')),
    'i3-retention': ('test_retention_window_chain_and_other_prefix',),
    **dict.fromkeys(('j1', 'j2'), ('test_chain_priority', 'test_crash_after_create_pending_pair_recovers', 'test_decision_consumed_after_winning_crash')),
    'j6': ('test_two_stall_residual_candidates',),
    'j7': ('test_pending_crash_and_abandon',),
    **dict.fromkeys(('L1', 'L2', 'L3', 'L4', 'L5', 'L6'), ('test_delta_lifecycle', 'test_ledger_only_accepted_session_forget_race')),
}


def installed(f, host='synthetic-host'):
    return cg.install(f[0], COPY, f[1], PREFIX, uploader=cp, hardware_id=host)


def pub(f, **kw):
    return cp.publish(f[0], COPY, f[1], PREFIX, **(installed(f) | kw))


def sql(f, text, args=()):
    with closing(sqlite3.connect(f[0] / 'lcm.db')) as conn, conn:
        conn.execute(text, args)


def forget(f):
    with capture_store(f[0] / 'lcm.db', f[0], f[3]) as store:
        store.delete_session_messages('s1')


def lost(f):
    sql(f, "DELETE FROM messages WHERE session_id='s1'")
    sql(f, "DELETE FROM host_uid_bindings")
    sql(f, "DELETE FROM message_relations")
    sql(f, "DELETE FROM capture_events")


def accept(f, head, kind='accept-loss'):
    m = cp.publish(f[0], COPY, f[1], PREFIX, snapshot_only=True)
    loss = cg.losses(head['guard'], m['guard'], unresolved=m['unresolved_minted'] + m['unresolved_quoted'])
    decision = dict(id=os.urandom(16).hex(), kind=kind, generation=head['generation'],
                    manifest_sha256=hashlib.sha256(cp._json(head)).hexdigest(), losses=loss)
    if kind == 'as-delete':
        decision.update(session='s1', ids=loss.get('messages', {}).get('s1', cp.encode_ids([])))
    cg.record_decision(f[0], COPY, decision, update_state=getattr(cp, '_update_state', None))
    return decision


def release(f, generation):
    d = dict(id=os.urandom(16).hex(), kind='release', pin=generation)
    cg.record_decision(f[0], COPY, d, update_state=getattr(cp, '_update_state', None))
    return d


def rebuild(f, m, name='rebuilt'):
    path = f[0].parent / name
    cd.rebuild(path, m['database'], m['generation'], lambda k: fetch(f, k))
    return path


def assert_empty(f, m):
    with closing(sqlite3.connect(rebuild(f, m, 'rebuilt-' + str(m['generation'])))) as conn:
        for table in ('messages', 'capture_events'):
            assert conn.execute(f'SELECT count(*) FROM {table} WHERE session_id=?', ('s1',)).fetchone() == (0,)
    assert not m['payloads']
    for k in keys(f):
        if '/gen/' in k:
            assert 's1' not in json.loads(f[1].get(k))['guard']['sessions']


def test_control_publisher_and_state(fixture):
    m = pub(fixture)
    assert m['publisher'] == hashlib.sha256((COPY + '\nsynthetic-host').encode()).hexdigest()
    data = json.loads(state(fixture).read_text())
    assert data['last_pair'] == data['pending_pair'] == [1, m['attempt']['token']]
    assert data['retention_done_generation'] == 1
    assert not {'pins', 'superseded'} & m['guard'].keys()
    assert pub(fixture)['generation'] == 2


@pytest.mark.parametrize('missing', ['guard', 'forget_items', 'publisher', 'chain_check', 'after_create', 'prepare_manifest'])
def test_missing_hook_refuses_first_generation(fixture, missing):
    hooks = installed(fixture)
    hooks.pop(missing, None)
    refused('guard_not_installed', cp.publish, fixture[0], COPY, fixture[1], PREFIX, **hooks)
    assert not keys(fixture) and not state(fixture).exists()


@pytest.mark.parametrize('why', ['null', 'hardware', 'lost-state', 'other-pair'])
def test_chain_priority(fixture, why):
    m = pub(fixture)
    if why == 'null':
        m['publisher'] = None
        (fixture[1].inner.root / PREFIX / 'gen/1.json').write_bytes(cp._json(m))
    elif why == 'lost-state':
        state(fixture).unlink()
    elif why == 'other-pair':
        data = json.loads(state(fixture).read_text())
        data.update(last_pair=[9, TOKEN], pending_pair=[8, TOKEN])
        state(fixture).write_bytes(cp._json(data))
    lost(fixture)
    before = keys(fixture), state(fixture).read_bytes() if state(fixture).exists() else None
    refused('chain_conflict', cp.publish, fixture[0], COPY, fixture[1], PREFIX,
            **installed(fixture, 'other-host' if why == 'hardware' else 'synthetic-host'))
    assert before == (keys(fixture), state(fixture).read_bytes() if state(fixture).exists() else None)


@pytest.mark.parametrize('surface', ['message', 'binding', 'relation', 'ledger-key', 'raw', 'payload-missing', 'payload-content'])
def test_loss_surfaces(fixture, surface):
    pub(fixture)
    if surface == 'message':
        sql(fixture, 'DELETE FROM messages WHERE store_id=1')
    elif surface == 'binding':
        sql(fixture, 'DELETE FROM host_uid_bindings')
    elif surface == 'relation':
        sql(fixture, 'DELETE FROM message_relations')
    elif surface == 'ledger-key':
        sql(fixture, 'DELETE FROM capture_events')
    elif surface == 'raw':
        import zlib
        sql(fixture, 'UPDATE capture_events SET raw=? WHERE raw IS NOT NULL', (zlib.compress(b'cut'),))
    else:
        for p in (fixture[0] / cp.PAYLOAD_FOLDER).iterdir():
            if surface == 'payload-missing':
                p.unlink()
            else:
                data = json.loads(p.read_text())
                data['content'] = 'changed synthetic payload'
                p.write_text(json.dumps(data))
    before = keys(fixture), state(fixture).read_bytes()
    error = refused('guard_loss', pub, fixture)
    assert error.detail == {'generation': 1}
    assert before == (keys(fixture), state(fixture).read_bytes())


@pytest.mark.parametrize('mode', ['sequence', 'changed-row', 'removed-id'])
def test_reuse_priority(fixture, mode):
    first = pub(fixture)
    if mode == 'sequence':
        sql(fixture, "UPDATE sqlite_sequence SET seq=0 WHERE name='messages'")
    elif mode == 'changed-row':
        sql(fixture, "UPDATE messages SET content='changed synthetic row' WHERE store_id=1")
    else:
        lost(fixture)
        accept(fixture, first)
        pub(fixture)
        sql(fixture, "INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES (1,'other','user','reused',1)")
    refused('store_id_reused', pub, fixture)


@pytest.mark.parametrize('advance', [False, True])
def test_rollback_before_after_mark(fixture, advance):
    old = (fixture[0] / 'lcm.db').read_bytes()
    pub(fixture)
    grow(fixture, 2)
    pub(fixture)
    (fixture[0] / 'lcm.db').write_bytes(old)
    if advance:
        sql(fixture, "INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES (4,'new','user','reused fourth id',1)")
        sql(fixture, "INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES (50,'new','user','new',1)")
    refused('store_id_reused', pub, fixture)


def test_forgotten_returned_priority(fixture):
    old = (fixture[0] / 'lcm.db').read_bytes()
    pub(fixture)
    forget(fixture)
    pub(fixture)
    (fixture[0] / 'lcm.db').write_bytes(old)
    refused('forgotten_session_returned', pub, fixture)


@pytest.mark.parametrize('crash', ['none', 'before-retention', 'between-deletes', 'after-staging'])
def test_forget_retention_and_rebuild(fixture, monkeypatch, crash):
    pub(fixture, part_size=32768)
    grow(fixture)
    pub(fixture, part_size=32768)
    if crash == 'after-staging':
        fired = []
        def race(method, key):
            if method == 'list' and key.endswith('/parts/') and not fired:
                fired.append(True)
                forget(fixture)
        fixture[1].hook = race
    else:
        forget(fixture)
    if crash in ('before-retention', 'between-deletes'):
        if crash == 'before-retention':
            real = cg.retention
            def die(*a, **kw):
                raise RuntimeError('synthetic crash before retention')
            monkeypatch.setattr(cg, 'retention', die)
        else:
            real = fixture[1].inner.delete
            def die(key):
                if '/gen/' in key:
                    raise RuntimeError('synthetic crash after keys')
                real(key)
            monkeypatch.setattr(fixture[1].inner, 'delete', die)
        with pytest.raises(RuntimeError):
            pub(fixture, part_size=32768)
        if crash == 'before-retention':
            monkeypatch.setattr(cg, 'retention', real)
        else:
            monkeypatch.setattr(fixture[1].inner, 'delete', real)
    m = pub(fixture, part_size=32768)
    assert m['forgotten'] and m['forgotten'][0]['base_generation'] is not None
    assert_empty(fixture, m)
    assert json.loads(state(fixture).read_text())['retention_done_generation'] == m['generation']


@pytest.mark.parametrize('kind', ['accept-loss', 'as-delete'])
def test_decisions_pins_and_pruning(fixture, kind):
    head = pub(fixture)
    lost(fixture)
    d = accept(fixture, head, kind)
    assert json.loads(state(fixture).read_text()).get('unpublished_decisions') == [d]
    m = pub(fixture)
    assert json.loads(state(fixture).read_text())['unpublished_decisions'] == []
    assert set(cp.decode_ids(m['guard']['removed_ids'])) == {1, 2, 3}
    if kind == 'accept-loss':
        assert m['pins'][0]['decision'] == d['id']
        with closing(sqlite3.connect(rebuild(fixture, head))) as conn:
            assert conn.execute('SELECT count(*) FROM messages').fetchone()[0] == 3
        forget(fixture)
        assert pub(fixture)['pins']
        release(fixture, 1)
        m = pub(fixture)
        assert not m['pins']
    assert_empty(fixture, m)


def test_acceptance_is_exact_and_ledger_always_compared(fixture):
    head = pub(fixture)
    sql(fixture, 'DELETE FROM message_relations')
    d = accept(fixture, head)
    assert d['losses'].get('relations')
    sql(fixture, 'DELETE FROM capture_events')
    refused('guard_loss', pub, fixture)
    accept(fixture, head)
    assert pub(fixture)['pins']


def test_pin_bucket_clock_expiry_and_missing_date(fixture):
    t = [1000]
    fixture[1].inner.clock = lambda: t[0]
    head = pub(fixture)
    lost(fixture)
    d = accept(fixture, head)
    m = pub(fixture, clock=lambda: 40 * 86400)
    assert m['pins'] == [dict(generation=1, decision=d['id'], **{'from': 1000, 'until': 1000 + 30 * 86400})]
    t[0] += 29 * 86400
    for _ in range(3):
        assert pub(fixture)['pins']
    t[0] = None
    assert pub(fixture)['pins']
    t[0] = 1000 + 30 * 86400
    assert pub(fixture)['pins'] == []
    assert PREFIX + '/gen/1.json' not in keys(fixture)


@pytest.mark.parametrize('field', ['pins', 'superseded'])
def test_malformed_pin_or_superseded_refuses(fixture, field):
    m = pub(fixture)
    m[field] = [{}]
    (fixture[1].inner.root / PREFIX / 'gen/1.json').write_bytes(cp._json(m))
    refused('bad_head', pub, fixture)


def test_withdraw_and_locked_state_merge(fixture):
    head = pub(fixture)
    lost(fixture)
    d = accept(fixture, head, 'as-delete')
    assert json.loads(state(fixture).read_text()).get('unpublished_decisions') == [d]
    with cp._locked(fixture[0]):
        refused('lock_held', cg.withdraw, fixture[0], COPY, d['id'], update_state=cp._update_state)
    cg.withdraw(fixture[0], COPY, d['id'], update_state=cp._update_state)
    assert not json.loads(state(fixture).read_text())['unpublished_decisions']
    refused('guard_loss', pub, fixture)


def test_as_delete_rollback_staging_and_final(fixture):
    head = pub(fixture)
    old = (fixture[0] / 'lcm.db').read_bytes()
    lost(fixture)
    d = accept(fixture, head, 'as-delete')
    assert d.get('ids') != cp.encode_ids([])
    (fixture[0] / 'lcm.db').write_bytes(old)
    before = list(keys(fixture))
    refused('as_delete_rows_present', pub, fixture)
    assert before == keys(fixture)
    lost(fixture)
    fired = []
    def race(method, key):
        if method == 'list' and key.endswith('/parts/') and not fired:
            fired.append(True)
            sql(fixture, "INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES (1,'s1','user','rollback',1)")
    # The final predicate checks the frozen db, not a live rollback after staging.
    fixture[1].hook = race
    assert pub(fixture)['forgotten'][0]['kind'] == 'as-delete'


def test_inherited_forget_and_first_generation(fixture):
    head = {'forgotten': [dict(key='s1', kind='forgotten', session='s1', base_generation=3),
                          dict(key='d', kind='as-delete', session='s1', base_generation=3)]}
    items = cg.inherited_items(head)
    assert items == [dict(key='s1', kind='forgotten', session='s1')]
    fn = cg.forget_items_for(fixture[0], COPY, inherited=items)
    refused('forgotten_session_returned', fn, None, fixture[0] / 'lcm.db')
    forget(fixture)
    m = cp.publish(fixture[0], COPY, fixture[1], PREFIX, **(installed(fixture) | {'forget_items': fn}))
    assert m['forgotten'] == [dict(key='s1', kind='forgotten', session='s1', base_generation=1)]


@pytest.mark.parametrize('lifecycle', ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])
def test_delta_lifecycle(fixture, monkeypatch, lifecycle):
    head = pub(fixture)
    lost(fixture)
    d = accept(fixture, head, 'as-delete' if lifecycle in ('L3', 'L4', 'L6') else 'accept-loss')
    assert d['losses']
    if lifecycle == 'L6':
        sql(fixture, "INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES (1,'s1','user','rollback',1)")
        refused('as_delete_rows_present', pub, fixture)
        return
    if lifecycle in ('L3', 'L4'):
        pub(fixture)
        sql(fixture, "INSERT INTO messages(session_id,role,content,timestamp) VALUES ('s1','user','new rows',1)")
        head = pub(fixture)
        if lifecycle == 'L4':
            lost(fixture)
            accept(fixture, head, 'as-delete')
        else:
            forget(fixture)
    else:
        pub(fixture)
        fired = []
        def race(method, key):
            if method == 'list' and key.endswith('/parts/') and not fired:
                fired.append(True)
                forget(fixture)
        fixture[1].hook = race
    stages = []
    real = cp._stage
    def stage(*a, **kw):
        stages.append(True)
        return real(*a, **kw)
    monkeypatch.setattr(cp, '_stage', stage)
    if lifecycle == 'L2':
        real_retention = cg.retention
        def die(*a, **kw):
            raise RuntimeError('kill before retention')
        monkeypatch.setattr(cg, 'retention', die)
        with pytest.raises(RuntimeError):
            pub(fixture)
        monkeypatch.setattr(cg, 'retention', real_retention)
    m = pub(fixture)
    assert m['database']['base']['reason'] == 'forget'
    if lifecycle in ('L1', 'L5'):
        assert len(stages) == 2
    release(fixture, 1)
    assert_empty(fixture, pub(fixture))


def test_newest_forget_then_capture(fixture):
    pub(fixture)
    forget(fixture)
    sql(fixture, "INSERT INTO messages(session_id,role,content,timestamp) VALUES ('s2','user','new row',1)")
    m = pub(fixture)
    assert m['guard']['removed_sessions'] == ['s1']
    assert cp.decode_ids(m['guard']['removed_ids']) == [1, 2, 3]


def test_pending_crash_and_abandon(fixture):
    real = fixture[1].inner.put_new
    seen = []
    def stop(key, data):
        if '/gen/' in key:
            seen.append(json.loads(data))
            raise RuntimeError('before create')
        real(key, data)
    fixture[1].inner.put_new = stop
    with pytest.raises(RuntimeError):
        pub(fixture)
    assert state(fixture).exists(), 'pending state precedes the first upload'
    data = json.loads(state(fixture).read_text())
    assert 'pending_pair' in data and 'pending_started' in data
    fixture[1].inner.put_new = real
    m = pub(fixture)
    assert m['attempt'] == seen[0]['attempt']
    fixture[1].inner.put_new = stop
    with pytest.raises(RuntimeError):
        pub(fixture)
    t = json.loads(state(fixture).read_text())['pending_started'] + 3.5 * 86400 + 1
    fixture[1].inner.clock = lambda: t
    fixture[1].inner.put_new = real
    assert pub(fixture)['attempt']['token'] != seen[-1]['attempt']['token']


def test_crash_after_create_pending_pair_recovers(fixture, monkeypatch):
    real = cd.promote_pending_list
    def die(*a):
        raise RuntimeError('after create')
    monkeypatch.setattr(cd, 'promote_pending_list', die)
    with pytest.raises(RuntimeError):
        pub(fixture)
    assert state(fixture).exists(), 'winning create retains pending state across a crash'
    data = json.loads(state(fixture).read_text())
    assert data.get('pending_pair') and not data.get('last_pair')
    monkeypatch.setattr(cd, 'promote_pending_list', real)
    assert pub(fixture)['generation'] == 2


def test_retention_order_failure_prefix_and_orphans(fixture):
    head = pub(fixture, part_size=32768)
    assert head['publisher'] is not None
    stray = 'parts/' + 'a' * 64 + '.' + TOKEN
    fixture[1].put_new(PREFIX + '/' + stray, b'unreferenced synthetic bytes')
    path = fixture[1].inner.root / PREFIX / stray
    os.utime(path, (100, 100))
    fixture[1].inner.clock = lambda: 100 + 7 * 86400
    forget(fixture)
    calls = []
    real = fixture[1].inner.delete
    def fail(key):
        calls.append(key)
        if '/parts/' in key and not key.endswith(stray):
            raise OSError('injected delete failure')
        real(key)
    fixture[1].inner.delete = fail
    m = pub(fixture)
    assert PREFIX + '/gen/1.json' in keys(fixture)
    assert PREFIX + '/' + stray in keys(fixture)
    assert json.loads(state(fixture).read_text()).get('retention_done_generation') != m['generation']
    fixture[1].inner.delete = real
    fixture[1].inner.clock = lambda: 101 + 7 * 86400
    m = pub(fixture)
    assert PREFIX + '/gen/1.json' not in keys(fixture)
    assert PREFIX + '/' + stray not in keys(fixture)
    assert all(k.startswith(PREFIX + '/') for k in calls)


@pytest.mark.parametrize('restart', [0, 1, 2])
@pytest.mark.parametrize('with_head', [False, True])
def test_forget_refusal_orphans_wait_for_grace(fixture, monkeypatch, restart, with_head):
    # Force an unresolvable live-forget race at each pass without a real bucket.
    original = cp._stage
    fixture[1].inner.clock = lambda: 100
    if with_head:
        pub(fixture)
    existing = set(keys(fixture))
    passes = []
    def stage(*a, **kw):
        if len(passes) >= restart:
            sql(fixture, "DELETE FROM metadata WHERE key='capture:forgotten:s1'")
        m = original(*a, **kw)
        passes.append(True)
        return m
    monkeypatch.setattr(cp, '_stage', stage)
    def mark(method, key):
        if method == 'list' and key.endswith('/parts/') and len(passes) > restart:
            sql(fixture, "INSERT OR REPLACE INTO metadata(key,value) VALUES ('capture:forgotten:s1','1')")
    fixture[1].hook = mark
    # A prior-pass restart is forced by the same live marker, never by token cleanup.
    original_items = installed(fixture)['forget_items']
    def items(head, db):
        answer = original_items(head, db)
        if len(passes) <= restart:
            sql(fixture, "INSERT OR REPLACE INTO metadata(key,value) VALUES ('capture:forgotten:s1','1')")
        return answer
    refused('forgotten_during_publish', pub, fixture, forget_items=items)
    uploaded = sorted(set(keys(fixture)) - existing)
    assert uploaded and not any('/gen/' in k for k in uploaded)
    assert {k for k in keys(fixture) if '/gen/' in k} == {k for k in existing if '/gen/' in k}
    for key in existing:
        if '/gen/' in key:
            named = {PREFIX + '/' + r['key'] for r in objects(json.loads(fixture[1].get(key)))}
            assert not named & set(uploaded)
    # The base had no guard/retention: this assertion is its red proof.
    assert len(passes) == 3
    for k in uploaded:
        os.utime(fixture[1].inner.root / k, (100, 100))
    monkeypatch.setattr(cp, '_stage', original)
    fixture[1].hook = lambda *a: None
    forget(fixture)
    fixture[1].inner.clock = lambda: 100 + 7 * 86400
    pub(fixture)
    assert set(uploaded) <= set(keys(fixture))
    fixture[1].inner.clock = lambda: 101 + 7 * 86400
    pub(fixture)
    assert not set(uploaded) & set(keys(fixture))


def test_payload_exemption_and_fast_path():
    old = {'mark': 1, 'sessions': {'s': dict(messages=[[1, 'x']], bindings=[], relations=[], ledger=[['line', ['k'], 'd', 'd', None]], digest='same')}, 'payloads': {'p': 'sha'}}
    new = copy.deepcopy(old)
    new['sessions']['s']['ledger'] = []
    new['payloads'] = {}
    got = cg.losses(old, new, unresolved=[])
    assert got.get('ledger') == {'s': [['line', ['k']]]}
    assert not got.get('payloads')
    assert cg.losses(old, new, unresolved=[{'name': 'p'}])['payloads'] == ['p']


def test_import_rule_and_ownership():
    assert cg.publisher(COPY, 'synthetic-host') is not None
    assert set(OWNED) == {'a1', 'a2', 'a3', 'a4', 'b', 'c', 'd', 'e', 'f1', 'f2', 'g2', 'h', 'i1', 'i2', 'i3-retention', 'j1', 'j2', 'j4', 'j5', 'j6', 'j7', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6'}
    for clause, names in OWNED.items():
        assert names, clause
        for name in names:
            assert callable(globals().get(name)), f'{clause}: missing {name}'
    text = (ROOT / 'scripts/maintainer_gate.py').read_text()
    assert '"cloud_guard.py",' in text
    for name in ('cloud_guard.py', 'cloud_copy.py'):
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node, ast.Import):
                assert all(n.name.split('.')[0] in sys.stdlib_module_names for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.level == 1 and node.module is None and [n.name for n in node.names] == ['cloud_delta']) or (node.level == 0 and node.module.split('.')[0] in sys.stdlib_module_names)


@pytest.mark.parametrize('pause', ['parts', 'part-check', 'manifest', 'retention'])
@pytest.mark.parametrize('copied_pair', ['last', 'pending'])
@pytest.mark.parametrize('machine', ['same', 'other'])
@pytest.mark.parametrize('upload_kind', ['base', 'delta'])
def test_twin_interleavings(fixture, pause, copied_pair, machine, upload_kind):
    pub(fixture)
    grow(fixture)
    twin_home = fixture[0].parent / 'twin'
    if copied_pair == 'last':
        shutil.copytree(fixture[0], twin_home)
    fired, outcomes = [], []
    def race(method, key):
        matches = ((pause == 'parts' and method == 'put_new' and ('/parts/' if upload_kind == 'base' else '/deltas/') in key) or
                   (pause == 'part-check' and method == 'list' and key.endswith('/parts/')) or
                   (pause == 'manifest' and method == 'put_new' and '/gen/' in key) or
                   (pause == 'retention' and method == 'get' and '/gen/2.json' in key))
        if fired or not matches:
            return
        fired.append(True)
        if copied_pair == 'pending':
            shutil.copytree(fixture[0], twin_home)
        twin = (twin_home, fixture[1], fixture[2], fixture[3])
        fixture[1].hook = lambda *a: None
        try:
            outcomes.append(pub(twin, **installed(twin, 'synthetic-host' if machine == 'same' else 'other-host')))
        except cp.Refused as e:
            outcomes.append(e.code)
        fixture[1].hook = race
    fixture[1].hook = race
    try:
        outcomes.append(pub(fixture, **({'size_factor': 0} if upload_kind == 'base' else {})))
    except cp.Refused as e:
        outcomes.append(e.code)
    assert fired
    manifests = [json.loads(fixture[1].get(k)) for k in keys(fixture) if '/gen/' in k]
    assert all(m['publisher'] is not None for m in manifests)
    assert len({m['generation'] for m in manifests}) == len(manifests)
    for m in manifests:
        assert all(PREFIX + '/' + r['key'] in keys(fixture) for r in objects(m))
    assert any(isinstance(o, dict) for o in outcomes)
    # A copied pending pair may adopt a winning create; the next divergent attempt is fenced.
    fixture[1].hook = lambda *a: None
    if pause != 'retention':
        assert 'chain_conflict' in outcomes


@pytest.mark.parametrize('candidate', ['built', 'reread', 'digest'])
def test_two_stall_residual_candidates(fixture, candidate):
    import concurrent.futures
    import threading
    head = pub(fixture)
    assert head['publisher'] is not None
    clock = [100.0]
    fixture[1].inner.clock = lambda: clock[0]
    grow(fixture)
    twin_ready, finish_twin = threading.Event(), threading.Event()
    twin_home = fixture[0].parent / 'twin-stall'
    real_put, real_get, real_stat = fixture[1].inner.put_new, fixture[1].inner.get, fixture[1].inner.stat
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future, twin_manifest, local_snapshots = [], [], {}
    local_snapshots[fixture[0]] = (fixture[0] / 'lcm.db').read_bytes()
    initial_thread = threading.get_ident()
    def put(key, data):
        if '/gen/' in key and candidate == 'reread':
            latest, _ = cp._head(fixture[1], PREFIX, COPY)
            assert latest['generation'] == int(key.rsplit('/', 1)[1][:-5]) - 1
        if '/gen/' in key and candidate == 'digest':
            home = fixture[0] if threading.get_ident() == initial_thread else twin_home
            cp._update_state(home, COPY, pending_manifest_sha256=hashlib.sha256(data).hexdigest())
        if key.endswith('/gen/3.json'):
            twin_manifest.append(json.loads(data))
            twin_ready.set()
            assert finish_twin.wait(5), 'bounded twin stall'
        return real_put(key, data)
    launched = []
    def get(key):
        data = real_get(key)
        if key.endswith('/gen/2.json') and threading.get_ident() == initial_thread and not launched:
            launched.append(True)
            shutil.copytree(fixture[0], twin_home)
            twin = (twin_home, fixture[1], fixture[2], fixture[3])
            sql(twin, "INSERT INTO messages(session_id,role,content,timestamp) VALUES ('new','user','twin new row',1)")
            local_snapshots[twin_home] = (twin_home / 'lcm.db').read_bytes()
            if candidate == 'digest':
                assert json.loads(state(twin).read_text())['pending_manifest_sha256'] == hashlib.sha256(data).hexdigest()
            future.append(pool.submit(pub, twin))
            assert twin_ready.wait(5), 'bounded retention stall'
            clock[0] = 8 * 86400
        return data
    def stat(key):
        result = real_stat(key)
        if result:
            result['time'] = 100
        return result
    fixture[1].inner.put_new, fixture[1].inner.get, fixture[1].inner.stat = put, get, stat
    try:
        pub(fixture)
        finish_twin.set()
        m = future[0].result(timeout=5)
    finally:
        finish_twin.set()
        pool.shutdown(wait=True)
        fixture[1].inner.put_new, fixture[1].inner.get, fixture[1].inner.stat = real_put, real_get, real_stat
    assert all((home / 'lcm.db').read_bytes() == data for home, data in local_snapshots.items())
    missing = [r['key'] for r in objects(m) if PREFIX + '/' + r['key'] not in keys(fixture)]
    assert missing, 'recorded two-stall residual must remain visible'
    with pytest.raises((FileNotFoundError, cp.Refused)):
        rebuild(fixture, m)
    twin = (twin_home, fixture[1], fixture[2], fixture[3])
    twin = (twin_home, fixture[1].inner, fixture[2], fixture[3])
    fixed = pub(twin)
    assert fixed['database']['base']['reason'] == 'missing_object'
    rebuild(twin, fixed, 'whole-again')
    refused('chain_conflict', pub, fixture)


def test_recreated_number_below_head_never_retains(fixture):
    pub(fixture)
    grow(fixture)
    outer_deletes, fired = [], []
    def pause(method, key):
        if method == 'put_new' and key.endswith('/gen/2.json') and not fired:
            fired.append(True)
            twin_home = fixture[0].parent / 'twin-recreate'
            shutil.copytree(fixture[0], twin_home)
            twin = (twin_home, fixture[1], fixture[2], fixture[3])
            fixture[1].hook = lambda *a: None
            pub(twin)
            forget(twin)
            pub(twin)
            pub(twin)
            assert PREFIX + '/gen/2.json' not in keys(fixture)
            fixture[1].hook = lambda method, key: outer_deletes.append(key) if method == 'delete' else None
    fixture[1].hook = pause
    recreated = pub(fixture)
    assert recreated['generation'] == 2 and not outer_deletes
    assert cp._head(fixture[1], PREFIX, COPY)[0]['generation'] == 4
    with pytest.raises((FileNotFoundError, cp.Refused)):
        rebuild(fixture, recreated)
    refused('chain_conflict', pub, fixture)


def test_state_merge_preserves_new_decision(fixture):
    pub(fixture)
    release_decision = dict(id='a' * 32, kind='release', pin=99)
    fired = []
    def pending_write(method, key):
        if method == 'put_new' and '/gen/' in key and not fired:
            fired.append(True)
            # Simulate a writer already holding this same publish lock, never a second locker.
            getattr(cp, '_update_state', lambda *a, **kw: None)(fixture[0], COPY, unpublished_decisions=[release_decision], unknown={'keep': True})
    fixture[1].hook = pending_write
    pub(fixture)
    data = json.loads(state(fixture).read_text())
    assert data.get('unpublished_decisions') == [release_decision] and data.get('unknown') == {'keep': True}


def test_final_as_delete_predicate_raises_without_restart(fixture, monkeypatch):
    pub(fixture)
    fn = cg.forget_items_for(fixture[0], COPY)
    ids = cp.encode_ids([1])
    fired, stages = [], []
    real = cp._stage
    def stage(*a, **kw):
        stages.append(True)
        return real(*a, **kw)
    monkeypatch.setattr(cp, '_stage', stage)
    def pending(method, key):
        if method == 'list' and key.endswith('/parts/') and not fired:
            fired.append(True)
            getattr(cp, '_update_state', lambda *a, **kw: None)(fixture[0], COPY, unpublished_decisions=[dict(id='d' * 32, kind='as-delete', ids=ids, session='s1')])
    fixture[1].hook = pending
    refused('as_delete_rows_present', pub, fixture, forget_items=fn)
    assert len(stages) == 1 and PREFIX + '/gen/2.json' not in keys(fixture)


def test_hardware_id_unavailable():
    refused('hardware_id_unavailable', cg.publisher, COPY, '')


def test_decision_binds_stored_manifest_bytes(fixture):
    head = pub(fixture)
    raw = json.dumps(head, indent=2).encode()
    (fixture[1].inner.root / PREFIX / 'gen/1.json').write_bytes(raw)
    lost(fixture)
    d = accept(fixture, head)
    assert d['losses']
    refused('guard_loss', pub, fixture)
    cg.withdraw(fixture[0], COPY, d['id'], update_state=cp._update_state)
    d['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    cg.record_decision(fixture[0], COPY, d, update_state=cp._update_state)
    assert pub(fixture)['pins']


def test_pin_dated_at_winning_create(fixture):
    fixture[1].inner.clock = lambda: 100
    head = pub(fixture)
    lost(fixture)
    accept(fixture, head)
    def tick(method, key):
        if method == 'list' and key.endswith('/parts/'):
            fixture[1].inner.clock = lambda: 1000
    fixture[1].hook = tick
    m = pub(fixture)
    assert m['pins'] and m['pins'][0]['from'] == 1000


def test_decision_consumed_after_winning_crash(fixture, monkeypatch):
    head = pub(fixture)
    lost(fixture)
    d = accept(fixture, head)
    assert d['losses']
    real = cd.promote_pending_list
    def die(*a):
        raise RuntimeError('winning decision not yet recorded')
    monkeypatch.setattr(cd, 'promote_pending_list', die)
    with pytest.raises(RuntimeError):
        pub(fixture)
    assert json.loads(state(fixture).read_text())['unpublished_decisions'] == [d]
    monkeypatch.setattr(cd, 'promote_pending_list', real)
    assert pub(fixture)['pins']
    assert json.loads(state(fixture).read_text())['unpublished_decisions'] == []


@pytest.mark.parametrize('failure', ['head-changed', 'list-failed'])
def test_timer_reads_bucket_generation_and_failed_list_publishes(fixture, failure):
    first = pub(fixture, clock=lambda: 100)
    # Seed the lane-2-compatible skip preconditions on the base as well.
    data = json.loads(state(fixture).read_text())
    data.update(last_pair=[1, first['attempt']['token']], pending_pair=[1, first['attempt']['token']],
                retention_done_generation=1)
    state(fixture).write_bytes(cp._json(data))
    if failure == 'head-changed':
        pub(fixture, clock=lambda: 100)
        state(fixture).write_bytes(cp._json(data))
        refused('chain_conflict', pub, fixture, timer=True, clock=lambda: 101)
    else:
        real, fired = fixture[1].inner.list, []
        def fail_once(key):
            if key.endswith('/gen/') and not fired:
                fired.append(True)
                raise OSError('list timeout')
            return real(key)
        fixture[1].inner.list = fail_once
        try:
            answer = pub(fixture, timer=True, clock=lambda: 101)
        except cp.Refused as error:
            answer = error.code
        assert isinstance(answer, dict) and answer['generation'] == 2


def test_ledger_only_accepted_session_forget_race(fixture, monkeypatch):
    sql(fixture, "UPDATE capture_events SET session_id='ledger-only'")
    head = pub(fixture)
    sql(fixture, 'DELETE FROM capture_events')
    d = accept(fixture, head)
    assert d['losses'].get('ledger', {}).get('ledger-only')
    m = pub(fixture)
    assert 'ledger-only' in m['guard']['removed_sessions']
    fired, stages = [], []
    real = cp._stage
    def stage(*a, **kw):
        stages.append(True)
        return real(*a, **kw)
    monkeypatch.setattr(cp, '_stage', stage)
    def forget_only(method, key):
        if method == 'list' and key.endswith('/parts/') and not fired:
            fired.append(True)
            sql(fixture, "INSERT INTO metadata(key,value) VALUES ('capture:forgotten:ledger-only','1')")
    fixture[1].hook = forget_only
    m = pub(fixture)
    assert len(stages) == 2 and m['database']['base']['reason'] == 'forget'


def test_as_delete_staging_preempts_chain_code(fixture):
    head = pub(fixture)
    before = (fixture[0] / 'lcm.db').read_bytes()
    lost(fixture)
    d = accept(fixture, head, 'as-delete')
    assert d['ids'] != cp.encode_ids([])
    (fixture[0] / 'lcm.db').write_bytes(before)
    head['publisher'] = None
    (fixture[1].inner.root / PREFIX / 'gen/1.json').write_bytes(cp._json(head))
    refused('as_delete_rows_present', pub, fixture)


def test_retention_window_chain_and_other_prefix(fixture):
    first = pub(fixture)
    assert first['publisher'] is not None
    other_id = 'abcdef01-2345-6789-abcd-0123456789ac'
    other_prefix = 'copies/' + other_id
    other = cp.DirTransport(fixture[1].inner.root, other_prefix)
    foreign_key = other_prefix + '/parts/' + 'a' * 64 + '.' + TOKEN
    other.put_new(foreign_key, b'synthetic other-copy object')
    other.put_new(other_prefix + '/gen/1.json', b'synthetic other-copy manifest')
    first['superseded'] = [dict(copy_id=other_id, generation=1, manifest_sha256='a' * 64)]
    (fixture[1].inner.root / PREFIX / 'gen/1.json').write_bytes(cp._json(first))
    for _ in range(5):
        grow(fixture)
        m = pub(fixture)
        assert m['superseded'] == first['superseded']
    assert {k for k in keys(fixture) if k.startswith(PREFIX + '/gen/')} == {PREFIX + f'/gen/{i}.json' for i in (4, 5, 6)}
    for i in (4, 5, 6):
        manifest = json.loads(fetch(fixture, f'gen/{i}.json'))
        rebuild(fixture, manifest, 'chain-' + str(i))
    assert other.get(foreign_key) == b'synthetic other-copy object'
    assert other.get(other_prefix + '/gen/1.json') == b'synthetic other-copy manifest'


def fork_capture(f):
    rows = [dict(json.loads(line), sessionId='f1') for line in f[2].read_text().splitlines()]
    rows.append(dict(entry('user', 'fork-new', 'synthetic fork turn'), sessionId='f1'))
    path = f[2].parent / 'f1.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with capture_store(f[0] / 'lcm.db', f[0], f[3]) as store:
        store.update_metadata_json(cc.FORK_PREFIX + 'f1', lambda _: True)
        assert cc.drain(store, cc.Lineage('f1', None, path)) == 'drained'
        assert store.connection.execute("SELECT count(*) FROM message_relations WHERE kind='fork_copy_of'").fetchone()[0]


def remove_message(f, ident):
    sql(f, 'DELETE FROM host_uid_bindings WHERE store_id=?', (ident,))
    sql(f, 'DELETE FROM message_relations WHERE store_id=? OR related_store_id=?', (ident, ident))
    sql(f, 'DELETE FROM messages WHERE store_id=?', (ident,))


def publish_answer(f, **kw):
    try:
        return pub(f, **kw)
    except (cp.Refused, OSError) as error:
        return error


@pytest.mark.parametrize('action', ['forget-parent', 'forget-fork', 'as-delete-parent'])
def test_cross_session_relation_exempt_at_either_end(fixture, action):
    fork_capture(fixture)
    head = pub(fixture)
    if action == 'forget-fork':
        assert set(head['guard']['sessions']['s1']['relations']) & set(head['guard']['sessions']['f1']['relations']), 'both endpoints must name each cross-session relation'
    before = (fixture[0] / 'lcm.db').read_bytes()
    # A vanished relation without either exemption still refuses.
    sql(fixture, "DELETE FROM message_relations WHERE kind='fork_copy_of'")
    refused('guard_loss', pub, fixture)
    (fixture[0] / 'lcm.db').write_bytes(before)
    if action == 'as-delete-parent':
        with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn:
            ids = [r[0] for r in conn.execute("SELECT store_id FROM messages WHERE session_id='s1'")]
        for ident in ids:
            remove_message(fixture, ident)
        sql(fixture, "DELETE FROM capture_events WHERE session_id='s1'")
        refused('guard_loss', pub, fixture)
        accept(fixture, head, 'as-delete')
    else:
        sid = 's1' if action == 'forget-parent' else 'f1'
        with capture_store(fixture[0] / 'lcm.db', fixture[0], fixture[3]) as store:
            store.delete_session_messages(sid)
    answer = publish_answer(fixture)
    assert isinstance(answer, dict), f'{action} must publish: {answer}'
    assert answer['generation'] == 2
    assert not answer['pins']
    remaining = 's1' if action == 'forget-fork' else 'f1'
    assert answer['guard']['sessions'][remaining]['messages']
    assert not answer['guard']['sessions'][remaining]['relations'] or action == 'forget-fork'


@pytest.mark.parametrize('kind', ['accept-loss', 'as-delete'])
def test_held_accepted_loss_compares_to_previous_generation(fixture, kind):
    # Synthetic held entry covers two message rows, with no raw bytes of its own.
    sql(fixture, "UPDATE capture_events SET raw=NULL,raw_from=NULL,disposition='held',store_ids='[1,2]' "
        'WHERE event_id=(SELECT min(event_id) FROM capture_events)')
    head = pub(fixture)
    remove_message(fixture, 1)
    refused('guard_loss', pub, fixture)
    accept(fixture, head, kind)
    assert pub(fixture)['generation'] == 2
    answer = publish_answer(fixture)
    assert isinstance(answer, dict), f'an unchanged accepted held entry must publish: {answer}'
    assert answer['generation'] == 3
    remove_message(fixture, 2)
    refused('guard_loss', pub, fixture)


def test_raw_accepted_cut_compares_to_previous_generation(fixture):
    import zlib
    head = pub(fixture)
    statement = 'UPDATE capture_events SET raw=? WHERE event_id=(SELECT min(event_id) FROM capture_events WHERE raw IS NOT NULL)'
    sql(fixture, statement, (zlib.compress(b'synthetic accepted cut'),))
    refused('guard_loss', pub, fixture)
    accept(fixture, head)
    assert pub(fixture)['generation'] == 2
    answer = publish_answer(fixture)
    assert isinstance(answer, dict), f'an unchanged accepted raw cut must publish: {answer}'
    assert answer['generation'] == 3
    sql(fixture, statement, (zlib.compress(b'further cut'),))
    refused('guard_loss', pub, fixture)


def test_unrecorded_create_then_failed_attempt_recovers(fixture, monkeypatch):
    pub(fixture)
    grow(fixture)
    original_promote = cd.promote_pending_list
    def killed(*args):
        raise RuntimeError('killed after winning create')
    monkeypatch.setattr(cd, 'promote_pending_list', killed)
    with pytest.raises(RuntimeError):
        pub(fixture)
    monkeypatch.setattr(cd, 'promote_pending_list', original_promote)
    won = json.loads(fetch(fixture, 'gen/2.json'))['attempt']
    grow(fixture)
    original_put = fixture[1].inner.put_new
    def failed(key, data):
        if '/gen/' in key:
            raise OSError('failed before next create')
        return original_put(key, data)
    monkeypatch.setattr(fixture[1].inner, 'put_new', failed)
    with pytest.raises(OSError):
        pub(fixture)
    data = json.loads(state(fixture).read_text())
    assert data['last_pair'] == [won['generation'], won['token']], 'preserve the recognized pending head before replacing pending'
    monkeypatch.setattr(fixture[1].inner, 'put_new', original_put)
    assert pub(fixture)['generation'] == 3
    before = json.loads(state(fixture).read_text())
    grow(fixture)
    monkeypatch.setattr(fixture[1].inner, 'put_new', failed)
    with pytest.raises(OSError):
        pub(fixture)
    after = json.loads(state(fixture).read_text())
    assert {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)} <= {'pending_pair', 'pending_started'}


@pytest.mark.parametrize('fault', ['head-list', 'head-get', 'retention-get', 'retention-list', 'retention-stat', 'retention-now'])
def test_won_create_io_failure_still_records_state(fixture, fault):
    head = pub(fixture)
    lost(fixture)
    decision = accept(fixture, head)
    stray = PREFIX + '/parts/' + 'e' * 64 + '.' + TOKEN
    fixture[1].put_new(stray, b'synthetic never-named object')
    created, injected, deleted = [], [], []
    def hook(method, key):
        if method == 'put_new' and key.endswith('/gen/2.json'):
            created.append(True)
        if method == 'delete':
            deleted.append(key)
        matches = ((fault == 'head-list' and method == 'list' and key.endswith('/gen/'))
                   or (fault == 'head-get' and method == 'get' and key.endswith('/gen/2.json'))
                   or (fault == 'retention-get' and method == 'get' and key.endswith('/gen/1.json'))
                   or (fault == 'retention-list' and method == 'list' and key.endswith('/parts/'))
                   or (fault == 'retention-stat' and method == 'stat' and key == stray)
                   or (fault == 'retention-now' and method == 'now'))
        if created and matches and not injected:
            injected.append(True)
            raise OSError('transient post-create I/O error')
    fixture[1].hook = hook
    answer = publish_answer(fixture)
    assert isinstance(answer, dict), f'a won create must record state despite {fault}: {answer}'
    assert injected and not deleted
    data = json.loads(state(fixture).read_text())
    assert data['last_pair'] == data['pending_pair'] == [2, answer['attempt']['token']]
    assert data['snapshot_signature'] == cd.store_signature(fixture[0])
    assert data['unpublished_decisions'] == [] and decision['id'] in {p['decision'] for p in answer['pins']}
    assert data.get('retention_done_generation') != 2
    fixture[1].hook = lambda *args: None
    assert pub(fixture)['generation'] == 3
    assert json.loads(state(fixture).read_text())['retention_done_generation'] == 3


@pytest.mark.parametrize('sid', ['s1', 'new-session'])
@pytest.mark.parametrize('damage', ['cut-raw', 'held-row-gone'])
def test_new_ledger_key_judged_at_first_appearance(fixture, sid, damage):
    import zlib
    head = pub(fixture)
    with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn:
        before = conn.execute('SELECT max(event_id) FROM capture_events').fetchone()[0]
    grow(fixture)
    if sid != 's1':
        sql(fixture, 'UPDATE messages SET session_id=? WHERE store_id>?', (sid, head['guard']['mark']))
        sql(fixture, 'UPDATE capture_events SET session_id=? WHERE event_id>?', (sid, before))
    with closing(sqlite3.connect(fixture[0] / 'lcm.db')) as conn:
        event = conn.execute("SELECT event_id FROM capture_events WHERE event_id>? AND source_kind='line' "
                             'AND raw IS NOT NULL ORDER BY event_id DESC LIMIT 1', (before,)).fetchone()[0]
        ident = conn.execute('SELECT max(store_id) FROM messages').fetchone()[0]
    if damage == 'held-row-gone':
        sql(fixture, "UPDATE capture_events SET raw=NULL,raw_from=NULL,disposition='held',store_ids=? WHERE event_id=?",
            (json.dumps([ident]), event))
    intact = (fixture[0] / 'lcm.db').read_bytes()
    if damage == 'cut-raw':
        sql(fixture, 'UPDATE capture_events SET raw=? WHERE event_id=?', (zlib.compress(b'synthetic cut'), event))
    else:
        remove_message(fixture, ident)
    staged = cp.publish(fixture[0], COPY, fixture[1], PREFIX, snapshot_only=True)
    old_keys = {cg._json(row[:2]) for row in head['guard']['sessions'].get(sid, {}).get('ledger', [])}
    new_keys = [row[:2] for row in staged['guard']['sessions'][sid]['ledger'] if cg._json(row[:2]) not in old_keys]
    loss = cg.losses(head['guard'], staged['guard'])
    assert loss.get('ledger', {}).get(sid) == new_keys, 'damage must be judged before the new key is published'
    assert not cg.losses(head['guard'], staged['guard'], forgotten=[sid])
    unchanged = keys(fixture), state(fixture).read_bytes()
    refused('guard_loss', pub, fixture)
    assert unchanged == (keys(fixture), state(fixture).read_bytes())
    # Both raw and held intact first appearances publish.
    (fixture[0] / 'lcm.db').write_bytes(intact)
    assert pub(fixture)['generation'] == 2
    # A new damaged key in a forgotten session is exempt, even on the digest fast path.
    exempt = copy.deepcopy(staged['guard'])
    exempt['sessions'][sid]['digest'] = head['guard']['sessions'].get(sid, {}).get('digest')
    sql(fixture, 'INSERT INTO metadata(key,value) VALUES (?,?)', (cg.FORGOTTEN + sid, '1'))
    assert cg.guard(head, fixture[0] / 'lcm.db', staged | {'guard': exempt}) == set()
    with capture_store(fixture[0] / 'lcm.db', fixture[0], fixture[3]) as store:
        store.delete_session_messages(sid)
    assert pub(fixture)['generation'] == 3


@pytest.mark.parametrize('recorded', ['last-1', 'last-2', 'pending-2'])
def test_empty_published_bucket_conflicts(fixture, recorded):
    # First publish and a resumed first attempt remain valid controls.
    original_put = fixture[1].inner.put_new
    def fail(key, data):
        raise OSError('interrupted first upload')
    fixture[1].inner.put_new = fail
    with pytest.raises(OSError):
        pub(fixture)
    pending = json.loads(state(fixture).read_text())['pending_pair']
    assert pending[0] == 1
    fixture[1].inner.put_new = original_put
    first = pub(fixture)
    assert [1, first['attempt']['token']] == pending
    if recorded != 'last-1':
        pub(fixture)
    if recorded == 'pending-2':
        data = json.loads(state(fixture).read_text())
        data.pop('last_pair')
        state(fixture).write_bytes(cp._json(data))
    for key in keys(fixture):
        fixture[1].inner.delete(key)
    before = state(fixture).read_bytes()
    fixture[1].calls.clear()
    answer = publish_answer(fixture)
    assert isinstance(answer, cp.Refused) and answer.code == 'chain_conflict', 'an emptied published copy must refuse'
    assert not keys(fixture) and state(fixture).read_bytes() == before
    assert not any(method in ('put_new', 'delete') for method, key in fixture[1].calls)


def test_as_delete_preserves_other_session_payload_loss(tmp_path):
    db = tmp_path / 'staged.db'
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute('CREATE TABLE messages(store_id INTEGER, session_id TEXT)')
        conn.execute('CREATE TABLE metadata(key TEXT, value TEXT)')
        conn.execute("INSERT INTO messages VALUES (2,'s2')")
    s1 = dict(messages=[[1, 'h1']], bindings=[], relations=[], ledger=[], digest='d1')
    s2 = dict(messages=[[2, 'h2']], bindings=[], relations=[], ledger=[], digest='d2')
    old = dict(mark=2, sessions={'s1': s1, 's2': s2}, payloads={'P.json': 'aa'},
               removed_ids=cg._ids([]), removed_sessions=[])
    new = dict(old, sessions={'s2': copy.deepcopy(s2)}, payloads={})
    head = dict(generation=1, forgotten=[], pins=[], guard=old)
    m = dict(guard=new, sequence={'messages': 2}, unresolved_minted=[{'name': 'P.json', 'sessions': ['s2']}], unresolved_quoted=[])
    loss = cg.losses(old, new, unresolved=m['unresolved_minted'])
    d = dict(id='0' * 32, kind='as-delete', generation=1, manifest_sha256='f' * 64,
             session='s1', ids=cg._ids([1]), losses=loss)
    outcome = None
    try:
        cg.guard(head, db, m, decisions=[d], bucket_now=100, head_sha='f' * 64)
    except cp.Refused as error:
        outcome = error.code
    assert outcome == 'guard_loss', 'as-delete cannot accept a surviving session payload loss'
    d['kind'] = 'accept-loss'
    cg.guard(head, db, m, decisions=[d], bucket_now=100, head_sha='f' * 64)
    assert len(m['pins']) == 1


def test_ownership_detects_renamed_proof(monkeypatch):
    monkeypatch.delitem(globals(), 'test_delta_lifecycle')
    detected = False
    try:
        test_import_rule_and_ownership()
    except AssertionError as error:
        detected = 'test_delta_lifecycle' in str(error)
    assert detected, 'ownership must detect a missing test_delta_lifecycle'
