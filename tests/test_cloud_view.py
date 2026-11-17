"""Replica views remove ownership, independently checked through rows and readers."""
import ast
from contextlib import closing, ExitStack
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading

import pytest

from hermes_lcm import claude_code_store as cc, cloud_copy as cp, cloud_view as cv
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.db_bootstrap import ensure_chunk_tables, ensure_embedding_tables
from tests.test_claude_code_store import _entry, _transcript
from tests.test_cloud_copy import fixture as _upload_fixture, grow, pub

upload_fixture = _upload_fixture

H = '0b1c2d3e-1111-4222-8333-444455556666'
V = '3f2a1b0c-5555-4666-8777-888899990000'
COPY = 'aaaaaaaa-1111-4222-8333-444455556666'
WORD = 'synthetic_private_ochre'
PARENT = 'synthetic-kept-parent-distinctive-session'
FORK = 'synthetic-kept-fork-distinctive-session'
ROOT = Path(__file__).resolve().parents[1]


def sql(db, statement, args=()):
    with closing(sqlite3.connect(db)) as conn, conn:
        return conn.execute(statement, args).fetchall()


def rows(db):
    with closing(sqlite3.connect(db)) as conn:
        return {t: conn.execute(f'SELECT * FROM "{t}"').fetchall() for (t,) in
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                if not t.endswith(('_data', '_idx'))}


def scanner(db, payloads, allowed=()):
    # No view-map knowledge: inspect every column, and query the FTS indexes too.
    allowed_columns = {prefix + session for session in allowed for prefix in
                       ('', 'claude-code:', 'capture:cursor:claude-code:',
                        'capture:pending:claude-code:', 'capture:ended:claude-code:',
                        'capture:forgotten:', 'capture:compact:', 'capture:fork:',
                        'capture:local_gc:', 'rotation_predecessor_session:')}
    fts = {'messages_fts': ('messages', 'store_id'), 'messages_fts_stem': ('messages', 'store_id'),
           'nodes_fts': ('summary_nodes', 'node_id')}
    for table, records in rows(db).items():
        if table in fts:
            parent, key = fts[table]
            for rowid, *record in sql(db, f'SELECT rowid,* FROM {table}'):
                if H in repr(record) or WORD in repr(record):
                    assert sql(db, f'SELECT session_id FROM {parent} WHERE {key}=?', (rowid,))[0][0] in allowed
            continue
        for record in records:
            text = repr(record)
            if H in text or WORD in text:
                assert any(isinstance(value, str) and value in allowed_columns for value in record), (table, record)
    for table, (parent, key) in fts.items():
        for (rowid,) in sql(db, f'SELECT rowid FROM {table} WHERE {table} MATCH ?', (WORD,)):
            assert sql(db, f'SELECT session_id FROM {parent} WHERE {key}=?', (rowid,))[0][0] in allowed
    for path in payloads.iterdir():
        body = json.loads(path.read_bytes())
        assert H not in repr(body) and WORD not in repr(body) or body.get('session_id') in allowed


def make_store(tmp_path, sessions=(H, V), features=False):
    home = tmp_path / 'source'
    db = cc.admit(home, hermes_home=tmp_path / 'unused-host')
    config = LCMConfig(database_path=str(db), embeddings_enabled=False,
                       large_output_externalization_enabled=True,
                       large_output_externalization_threshold_chars=200)
    ids, hidden_entries = {}, []
    with closing(cc.open_store(db, home, config)) as store:
        for index, session in enumerate(sessions):
            text = WORD if session == H else 'visible amber control'
            entries = [_entry('user', f'u{index}', text, session),
                       _entry('assistant', f'a{index}', [{'type': 'tool_use', 'id': f't{index}', 'name': 'Read', 'input': {}}], session),
                       _entry('user', f'r{index}', [{'type': 'tool_result', 'tool_use_id': f't{index}', 'content': (text + ' ') * 80}], session)]
            if session == H:
                hidden_entries = entries
            if session == FORK:
                entries = [{**entry, 'sessionId': session} for entry in hidden_entries] + entries
                store.update_metadata_json(cc.FORK_PREFIX + session, lambda _: True)
            transcript = _transcript(tmp_path / f'fixture{index}', entries, session)
            parent, mark, agent = session.partition('/agent:')
            assert cc.drain(store, cc.Lineage(parent, agent if mark else None, transcript)) == 'drained'
            ids[session] = [r[0] for r in store.connection.execute('SELECT store_id FROM messages WHERE session_id=?', (session,))]
            if features:
                for prefix in ('capture:compact:', 'capture:fork:', 'capture:local_gc:', 'capture:forgotten:'):
                    if prefix != 'capture:forgotten:' or session == H:
                        store.connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (prefix + session, '[]'))
                for prefix in ('capture:pending:', 'capture:ended:'):
                    store.connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (prefix + 'claude-code:' + session, '{}'))
                store.connection.commit()
        if features:
            store.ensure_ingest_ledger()
            for session in sessions:
                mid = ids[session][0]
                lineage = 'claude-code:' + session
                occ = store.connection.execute("INSERT INTO occurrence_bindings(lineage,ordinal,store_id,identity_digest,disposition,view_seq) VALUES (?,0,?,?,'stored',1)", (lineage, mid, b'synthetic')).lastrowid
                store.connection.execute("INSERT INTO stamp_observations(lineage,occ_id,stamp,kind,view_seq) VALUES (?,?,'synthetic','alt_stamp',1)", (lineage, occ))
                store.connection.execute("INSERT INTO ledger_emissions(lineage,view_seq,output_index,descriptor_digest,kind,source_store_ids,state,recovered_store_id) VALUES (?,1,0,?,'projection',?,'adopted',?)", (lineage, b'synthetic', json.dumps([mid]), mid))
                store.connection.execute("INSERT INTO ingest_views VALUES (?,1,'ingest','ready',?,?)", (lineage, b'[]', b'synthetic'))
                store.connection.execute('INSERT INTO lineage_marks(lineage,max_store_id) VALUES (?,?)', (lineage, mid))
            store.connection.commit()
        store.record_session_lineage(V, H, source='synthetic')
        store.connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('rotation_predecessor_session:' + V, H))
        store.connection.execute('INSERT OR REPLACE INTO lcm_lifecycle_state(conversation_id,current_session_id,last_finalized_session_id) VALUES (?,?,?)', (V, V, H))
        store._ensure_identity_anchor_schema()
        store.connection.execute('INSERT INTO message_relations(store_id,kind,related_store_id) VALUES (?,?,?)', (ids[V][0], 'fork_copy_of', ids[H][0]))
        store.connection.commit()
        dag = SummaryDAG(db)
        for session in sessions:
            leaf = dag.add_node(SummaryNode(session_id=session, summary=WORD if session == H else 'visible summary', source_ids=ids[session], token_count=5), escalation_level=1, model='synthetic')
            dag.add_node(SummaryNode(session_id=session, summary=WORD if session == H else 'visible parent', depth=1, source_ids=[leaf], source_type='nodes', token_count=5))
        dag.close()
        if features:
            ensure_embedding_tables(store.connection)
            ensure_chunk_tables(store.connection)
            for session in sessions:
                mid = ids[session][0]
                nid = store.connection.execute('SELECT node_id FROM summary_nodes WHERE session_id=? ORDER BY node_id', (session,)).fetchone()[0]
                store.connection.execute('INSERT INTO lcm_embedding_meta(embedded_id,embedded_kind,identity_hash) VALUES (?,\'summary\',\'synthetic\')', (str(nid),))
                for table, column, value in [('lcm_embedding_vectors', 'vec', b'v'), ('lcm_embedding_binary', 'bits', b'b')]:
                    store.connection.execute(f'INSERT INTO {table} VALUES (?,\'synthetic\',?)', (str(nid), value))
                store.connection.execute('INSERT INTO lcm_chunk_meta(chunk_id,identity_hash,store_id) VALUES (?,\'synthetic\',?)', (f'chunk{mid}', mid))
                store.connection.execute('INSERT INTO lcm_chunk_vectors VALUES (?,\'synthetic\',?)', (f'chunk{mid}', b'v'))
                store.connection.execute('INSERT INTO lcm_chunk_binary VALUES (?,\'synthetic\',?)', (f'chunk{mid}', b'b'))
            store.connection.execute("INSERT OR REPLACE INTO metadata VALUES ('chunk_gapfill_scan:synthetic','preserved')")
            store.connection.commit()
        assert cc.to_rollback_journal(store.connection)
    assert cc.leave_readable(db) and db.stat().st_size < 5_000_000
    return home, ids


@pytest.fixture
def world(tmp_path):
    source, ids = make_store(tmp_path, features=True)
    home = tmp_path / 'local'
    home.mkdir()
    shutil.copyfile(source / 'lcm.db', home / 'lcm.db')
    copy = home / 'cloud' / 'replicas' / COPY
    copy.mkdir(parents=True)
    shutil.copytree(source, copy / 'gen-1')
    (copy / 'current').write_text('1\n')
    forget(home, H)
    return home, copy, ids


def forget(home, *sessions):
    for session in sessions:
        sql(home / 'lcm.db', 'INSERT OR REPLACE INTO metadata VALUES (?,?)', ('capture:forgotten:' + session, 'true'))


def built(world, **kwargs):
    home, copy, _ = world
    result = cv.build(home, COPY, **kwargs)
    assert result['code'] == 'ok', result
    pointer = json.loads((copy / 'view.json').read_bytes())
    return copy / pointer['folder'], pointer


def test_a1_residue(world):
    home, copy, ids = world
    before = rows(copy / 'gen-1/lcm.db')
    folder, _ = built(world)
    scanner(folder / 'lcm.db', folder / 'lcm-large-outputs')
    after = rows(folder / 'lcm.db')
    for table in ('messages', 'summary_nodes', 'capture_events'):
        assert after[table] == [r for r in before[table] if V in r]
    hidden_nodes = {r[0] for r in before['summary_nodes'] if H in r}
    hidden_chunks = {r[0] for r in before['lcm_chunk_meta'] if r[2] in ids[H]}
    hidden_occurrences = {r[0] for r in before['occurrence_bindings'] if r[3] in ids[H]}
    for table in ('summary_node_provenance', 'lcm_embedding_meta', 'lcm_embedding_vectors', 'lcm_embedding_binary'):
        assert after[table] == [r for r in before[table] if int(r[0]) not in hidden_nodes]
    for table in ('lcm_chunk_vectors', 'lcm_chunk_binary'):
        assert after[table] == [r for r in before[table] if r[0] not in hidden_chunks]
    assert after['stamp_observations'] == [r for r in before['stamp_observations'] if r[1] not in hidden_occurrences]
    for table in ('occurrence_bindings', 'ledger_emissions', 'ingest_views', 'lineage_marks'):
        assert after[table] == [r for r in before[table] if 'claude-code:' + H not in r]
    for table in ('host_uid_bindings', 'lcm_chunk_meta'):
        assert after[table] == [r for r in before[table] if r[0] not in ids[H] and r[2] not in ids[H]]
    assert ('chunk_gapfill_scan:synthetic', 'preserved') in after['metadata']
    assert rows(copy / 'gen-1/lcm.db') == before
    assert cv.removed_set(home) == {H}


@pytest.mark.parametrize('plant', [False, True])
def test_a1b_whole_names(tmp_path, plant):
    child = H + '/agent:A'
    collision = PARENT + '/agent:' + H
    source, _ = make_store(tmp_path, (H, V, PARENT, child, collision, FORK), features=True)
    home = tmp_path / 'local'
    home.mkdir()
    shutil.copyfile(source / 'lcm.db', home / 'lcm.db')
    copy = home / 'cloud/replicas' / COPY
    copy.mkdir(parents=True)
    shutil.copytree(source, copy / 'gen-1')
    (copy / 'current').write_text('1')
    db = copy / 'gen-1/lcm.db'
    assert sql(db, 'SELECT parent_session_id FROM session_lineage WHERE child_session_id=?', (FORK,)) == [(H,)]
    sql(db, 'INSERT OR REPLACE INTO session_lineage VALUES (?,?,NULL,NULL,NULL,\'subagent\',0)', (child, H))
    assert json.loads(sql(db, 'SELECT value FROM metadata WHERE key=?', ('capture:cursor:claude-code:' + FORK,))[0][0])['fork_parent'] == H
    before = rows(db)
    forget(home, H)
    folder, _ = built((home, copy, {}))
    after = rows(folder / 'lcm.db')
    for table in ('messages', 'summary_nodes', 'capture_events', 'metadata', 'host_uid_bindings',
                  'occurrence_bindings', 'stamp_observations', 'ledger_emissions', 'ingest_views', 'lineage_marks'):
        for row in before[table]:
            if any(s in row or any(isinstance(x, str) and s in x for x in row) for s in (child, collision, 'claude-code:' + FORK, 'claude-code:' + PARENT)) or table in ('messages', 'summary_nodes', 'capture_events') and any(s in row for s in (FORK, PARENT)):
                assert row in after[table], (table, row)
    assert after['session_lineage'] == [r for r in before['session_lineage'] if r[0] != H and r[1] != H]
    assert all(r[1] != 'fork_copy_of' for r in sql(folder / 'lcm.db', 'SELECT store_id,kind FROM message_relations'))
    allowed = (child, collision, FORK)
    scanner(folder / 'lcm.db', folder / 'lcm-large-outputs', allowed)
    if plant:
        reject_planted_event(db, folder, allowed)


def test_a3_unknown_table(world):
    home, copy, _ = world
    sql(copy / 'gen-1/lcm.db', 'CREATE TABLE future_feature(value TEXT)')
    sql(copy / 'gen-1/lcm.db', "INSERT INTO future_feature VALUES ('control')")
    assert cv.build(home, COPY)['code'] == 'view_unsupported_table'
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    folder, _ = built(world)
    assert folder.joinpath('lcm.db').stat().st_ino == copy.joinpath('gen-1/lcm.db').stat().st_ino


@pytest.mark.parametrize('kind', ['metadata', 'lineage'])
def test_a3_unsupported_key(world, kind):
    home, copy, _ = world
    db = copy / 'gen-1/lcm.db'
    if kind == 'metadata':
        sql(db, 'INSERT INTO metadata VALUES (?,?)', ('future:' + H, 'x'))
    else:
        sql(db, 'UPDATE host_uid_bindings SET lineage_key=? WHERE store_id=(SELECT MIN(store_id) FROM messages)', ('future:' + H,))
    assert cv.build(home, COPY)['code'] == 'view_unsupported_key'


def test_a3_summary_spans(world):
    home, copy, ids = world
    sql(copy / 'gen-1/lcm.db', 'UPDATE summary_nodes SET source_ids=? WHERE session_id=? AND depth=0', (json.dumps([ids[H][0]]), V))
    assert cv.build(home, COPY)['code'] == 'view_summary_spans_forgotten'


def test_a3_trigger_recheck(world):
    home, copy, ids = world
    sql(copy / 'gen-1/lcm.db', f"CREATE TRIGGER residue AFTER DELETE ON capture_events BEGIN INSERT OR IGNORE INTO messages(store_id,session_id,role,timestamp) VALUES ({ids[H][0]},'{V}','user',0); END")
    assert cv.build(home, COPY)['code'] == 'view_unsupported_key'
    assert not (copy / 'view.json').exists()


def test_a4_payload_ownership(world):
    _, copy, ids = world
    payloads = copy / 'gen-1/lcm-large-outputs'
    cases = [('hidden.json', H, (V,), False), ('visible.json', V, (H, V), True),
             ('unowned-hidden.json', '', (H,), False), ('shared.json', '', (H, V), True),
             ('owner-only.json', H, (), False)]
    for name, owner, sessions, _ in cases:
        (payloads / name).write_text(json.dumps({'session_id': owner, 'content': 'synthetic'}))
        for session in sessions:
            sql(copy / 'gen-1/lcm.db', 'UPDATE messages SET content=content || ? WHERE store_id=?', (f' [Externalized payload: x; ref={name}]', ids[session][0]))
    folder, _ = built(world)
    for name, _, _, kept in cases:
        target = folder / 'lcm-large-outputs' / name
        assert target.exists() == kept
        if kept:
            assert target.stat().st_ino == (payloads / name).stat().st_ino


def test_a4_identity(world, monkeypatch):
    home, copy, _ = world
    payload = copy / 'gen-1/lcm-large-outputs/owner-only.json'
    payload.write_text(json.dumps({'session_id': 'only-owner', 'content': 'owner-only control'}))
    folder, pointer = built(world)
    forget(home, 'absent')
    again, p2 = built(world)
    assert again == folder and p2['forgotten_sha256'] != pointer['forgotten_sha256']
    forget(home, 'only-owner')
    owned, _ = built(world)
    assert owned != folder and not (owned / 'lcm-large-outputs/owner-only.json').exists()
    sql(home / 'lcm.db', 'DELETE FROM metadata WHERE key=?', ('capture:forgotten:' + H,))
    restored, _ = built(world)
    assert restored != folder and sql(restored / 'lcm.db', 'SELECT 1 FROM messages WHERE session_id=?', (H,))
    sql(home / 'lcm.db', 'DELETE FROM metadata WHERE key=?', ('capture:forgotten:only-owner',))
    restored, _ = built(world)
    assert restored.joinpath('lcm.db').stat().st_ino == copy.joinpath('gen-1/lcm.db').stat().st_ino
    p = json.loads((copy / 'view.json').read_bytes())
    p['rules'] = 0
    (copy / 'view.json').write_text(json.dumps(p))
    assert built(world)[1]['rules'] == 1
    monkeypatch.setattr(cv, 'RULES', 2)
    changed, rules_pointer = built(world)
    assert changed != restored and rules_pointer['rules'] == 2


@pytest.mark.parametrize('kind', ['session', 'row'])
def test_a4_removed_file(world, kind):
    home, copy, ids = world
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    if kind == 'row':
        # Capture stores do not compact; test removed message ids without mixed summaries.
        sql(copy / 'gen-1/lcm.db', 'DELETE FROM summary_nodes')
    old, _ = built(world)
    data = {'format': 1, 'session_ids': [H] if kind == 'session' else [], 'row_ids': ids[H] if kind == 'row' else []}
    (copy / 'removed.json').write_text(json.dumps(data))
    folder, pointer = built(world)
    assert folder != old
    assert not sql(folder / 'lcm.db', 'SELECT 1 FROM messages WHERE session_id=?', (H,))
    assert pointer['removed_sha256'] == hashlib.sha256((copy / 'removed.json').read_bytes()).hexdigest()


def test_a4_digest():
    assert cv.forgotten_digest([V, H, H]) == 'd32b75cc4f118d8e709d786a1d10998880f9517537e7df297302a09905e6f88c'
    assert cv.forgotten_digest([]) == hashlib.sha256(b'').hexdigest()


@pytest.mark.parametrize('seam', ['generation', 'matched', 'reuse', 'database', 'deleted', 'payloads', 'synced', 'rechecked', 'renamed', 'published'])
def test_a5_crash(world, seam):
    home, copy, _ = world
    def crash(step):
        if step == seam:
            raise InterruptedError('synthetic crash')
    # reuse is reached on a second call; all other seams on a fresh build.
    if seam == 'reuse':
        built(world)
    with pytest.raises(InterruptedError):
        cv.build(home, COPY, seam=crash)
    if (copy / 'view.json').exists():
        p = json.loads((copy / 'view.json').read_bytes())
        scanner(copy / p['folder'] / 'lcm.db', copy / p['folder'] / 'lcm-large-outputs')
    built(world)
    assert len([p for p in copy.glob('view-*') if p.is_dir()]) == 1
    assert not list(copy.glob('*.tmp-*'))


@pytest.mark.parametrize('clone', [False, True])
def test_a5_disk_floor(world, clone):
    home, copy, _ = world
    assert cv.build(home, COPY, free_space=lambda _: 0, clone=lambda *_: clone)['code'] == 'view_disk_floor'
    assert not (copy / 'view.json').exists() and not [p for p in copy.glob('view-*') if p.is_dir()]
    def try_clone(source, target):
        if clone:
            shutil.copyfile(source, target)
        return clone
    space = cv.FLOOR
    result = cv.build(home, COPY, free_space=lambda _: space, clone=try_clone)
    assert result['code'] == 'view_disk_floor'
    space += 2 * copy.joinpath('gen-1/lcm.db').stat().st_size
    assert cv.build(home, COPY, free_space=lambda _: space, clone=try_clone)['code'] == 'ok'
    assert cv.build(home, COPY, free_space=lambda _: 0)['code'] == 'ok'
    old = json.loads((copy / 'view.json').read_bytes())
    forget(home, V)
    if not clone:
        assert cv.build(home, COPY, free_space=lambda _: cv.FLOOR, clone=try_clone)['code'] == 'view_disk_floor'
        assert (copy / 'view.json').exists(), 'a disk-floor refusal must preserve the old pointer'
        assert json.loads((copy / 'view.json').read_bytes()) == old
        assert (copy / old['folder']).exists()
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    assert cv.build(home, COPY, free_space=lambda _: 0)['code'] == 'ok'


def test_a5_isolated_command(world):
    home, copy, _ = world
    assert cv.wait_lock is getattr(cp, 'wait_lock', None), 'builders and uploads share the bounded wait'
    allowed = {'cloud_view.py': ('cloud_copy',), 'cloud_copy.py': (None,), 'cloud_delta.py': ()}
    for filename, siblings in allowed.items():
        for node in ast.walk(ast.parse((ROOT / filename).read_text())):
            if isinstance(node, ast.Import):
                assert all(n.name.split('.')[0] in sys.stdlib_module_names for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.level == 1 and node.module in siblings and
                        (node.module is not None or [n.name for n in node.names] == ['cloud_delta'])) or (
                            node.level == 0 and node.module.split('.')[0] in sys.stdlib_module_names)
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(ROOT / 'scripts/lcm_x_cloud.py'), 'view', '--home', str(home), '--copy-id', COPY], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['code'] == 'ok'
    assert len(result.stdout.splitlines()) == 1 and (copy / 'view.json').exists()


def test_a6_waiting_builders(world):
    home, copy, _ = world
    entered, release = threading.Event(), threading.Event()
    answers, errors = [], []
    def first_seam(step):
        if step == 'matched':
            entered.set()
            assert release.wait(30)
    def run(**kwargs):
        try:
            answers.append(cv.build(home, COPY, **kwargs))
        except BaseException as exc:
            errors.append(exc)
    first = threading.Thread(target=run, kwargs={'seam': first_seam})
    first.start()
    try:
        assert entered.wait(30)
        second = threading.Thread(target=run)
        second.start()
        forget(home, V)
    finally:
        release.set()
        first.join(30)
        if 'second' in locals():
            second.join(30)
    assert not errors and len(answers) == 2 and all(r['code'] == 'ok' for r in answers)
    folder, p = built(world)
    assert p['forgotten_sha256'] == cv.forgotten_digest({H, V})
    assert not sql(folder / 'lcm.db', 'SELECT * FROM messages')


def test_a6_busy(world):
    home, copy, _ = world
    with (copy / 'view.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert cv.build(home, COPY, lock_wait=lambda *_: False)['code'] == 'view_busy'


def test_a6_races_and_locks(world):
    home, copy, _ = world
    once = []
    def interleave(step):
        if step == 'deleted' and not once:
            once.append(True)
            forget(home, V)
            shutil.copytree(copy / 'gen-1', copy / 'gen-2')
            (copy / 'current').write_text('2')
        if step == 'generation':
            gen = (copy / 'current').read_text().strip()
            with ExitStack() as stack:
                lock = os.open(copy / ('gen-' + gen), os.O_RDONLY)
                stack.callback(os.close, lock)
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder, p = built(world, seam=interleave)
    assert p['generation'] == 2 and not sql(folder / 'lcm.db', 'SELECT * FROM messages')
    with ExitStack() as stack:
        lock = os.open(folder, os.O_RDONLY)
        stack.callback(os.close, lock)
        fcntl.flock(lock, fcntl.LOCK_SH)
        sql(home / 'lcm.db', 'DELETE FROM metadata WHERE key=?', ('capture:forgotten:' + V,))
        built(world)
        assert folder.exists()
    built(world)
    assert not folder.exists()


def test_a7_result_record(world):
    home, copy, _ = world
    result = cv.build(home, COPY)
    record = json.loads((copy / 'view-result.json').read_bytes()) if (copy / 'view-result.json').exists() else {}
    assert record.get('code') == 'ok', result
    assert set(record) == {'code', 'generation', 'rules', 'forgotten_sha256', 'removed_sha256', 'ended_at'}
    assert record['rules'] == 1 and record['generation'] == 1 and record['ended_at'] > 0
    assert record['forgotten_sha256'] == cv.forgotten_digest({H})
    assert record['removed_sha256'] == hashlib.sha256(b'').hexdigest()
    sql(copy / 'gen-1/lcm.db', 'CREATE TABLE future_feature(x)')
    sql(copy / 'gen-1/lcm.db', 'INSERT INTO future_feature VALUES (1)')
    for kwargs, code in [({}, 'view_unsupported_table'), ({'lock_wait': lambda *_: False}, 'view_busy'), ({'free_space': lambda _: 0}, 'view_unsupported_table')]:
        if code == 'view_busy':
            with (copy / 'view.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                assert cv.build(home, COPY, **kwargs)['code'] == code
        else:
            # Retire the old complete folder so the classification is rerun.
            forget(home, V)
            assert cv.build(home, COPY, **kwargs)['code'] == code
        assert json.loads((copy / 'view-result.json').read_bytes())['code'] == code


def test_a2_reader_pairs(world, tmp_path):
    _, _, ids = world
    folder, _ = built(world)
    missing = '99999999-9999-4999-8999-999999999999'
    payload = next(p.name for p in world[1].joinpath('gen-1/lcm-large-outputs').iterdir() if json.loads(p.read_bytes()).get('session_id') == H)
    pairs = []
    for tool in ('lcm_recall', 'lcm_grep', 'lcm_expand', 'lcm_load_session', 'lcm_describe'):
        for session in (H, ' \t' + H + '\n', '\u00a0' + H + '\u00a0'):
            extra = {'query': WORD} if tool in ('lcm_recall', 'lcm_grep') else {}
            pairs.append((tool, dict(extra, session_id=session), {session: session.replace(H, missing)}))
    for tool, key, value, other in [('lcm_expand', 'store_id', ids[H][0], 999999), ('lcm_expand', 'node_id', 1, 999999),
                                    ('lcm_load_session', 'anchor_store_id', ids[H][0], 999999),
                                    ('lcm_grep', 'conversation_id', H, missing),
                                    ('lcm_recall', 'seen_refs', ['lcm:1:0-5'], ['lcm:999999:0-5'])]:
        pairs.append((tool, {key: value, **({'query': WORD} if tool in ('lcm_recall', 'lcm_grep') else {})}, {str(value): other}))
    for session in (None, V, H):
        for ref in (payload, '\u00a0' + payload + '\u00a0'):
            args = {'externalized_ref': ref}
            if session:
                args['session_id'] = session
            pairs.append(('lcm_expand', args, {ref: ref.replace(payload, 'never-existed.json')}))
    requests = []
    swaps = []
    for tool, args, changes in pairs:
        twin = dict(args)
        for key, value in args.items():
            if str(value) in changes:
                twin[key] = changes[str(value)]
        for item in (args, twin):
            requests.append({'jsonrpc': '2.0', 'id': len(requests), 'method': 'tools/call', 'params': {'name': tool, 'arguments': item}})
        swaps.append(changes)
    for tool in ('lcm_recall', 'lcm_grep', 'lcm_expand', 'lcm_load_session', 'lcm_describe'):
        args = {'session_id': V}
        if tool in ('lcm_recall', 'lcm_grep'):
            args['query'] = 'visible'
        if tool == 'lcm_expand':
            args['node_id'] = 3
        requests.append({'jsonrpc': '2.0', 'id': len(requests), 'method': 'tools/call', 'params': {'name': tool, 'arguments': args}})
    env = {k: v for k, v in os.environ.items() if not k.startswith('LCM_') and k != 'PYTHONPATH'}
    env['HF_HUB_OFFLINE'] = '1'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/lcm_x_mcp.py'), '--store', str(folder / 'lcm.db'), '--hermes-home', str(folder), '--log-dir', str(tmp_path / 'logs')], input=''.join(json.dumps(r) + '\n' for r in requests), capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    answers = [json.loads(line)['result'] for line in result.stdout.splitlines()]
    assert len(answers) == len(requests)
    for i, changes in enumerate(swaps):
        left, right = answers[2*i:2*i+2]
        left_body = json.loads(left['content'][0]['text'])
        right_body = json.loads(right['content'][0]['text'])
        def swap(value):
            if isinstance(value, dict):
                return {k: swap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [swap(v) for v in value]
            if isinstance(value, str):
                for before, after in changes.items():
                    if before.isdigit():
                        value = re.sub(r'(?<![A-Za-z0-9_])' + before + r'(?![A-Za-z0-9_])', str(after), value)
                    else:
                        value = value.replace(before.strip(), str(after).strip())
            return value
        assert swap(left_body) == right_body and left['isError'] == right['isError'], (pairs[i], left, right)
    assert all(H not in json.dumps(reply) and WORD not in json.dumps(reply) for reply in answers[-5:])


def test_a5_wal_copy(world):
    home, copy, _ = world
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    with closing(sqlite3.connect(copy / 'gen-1/lcm.db')) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES (100,?,\'user\',\'wal control\',0)', (V,))
        writer.commit()
        folder, _ = built(world)
        assert sql(folder / 'lcm.db', 'SELECT content FROM messages WHERE store_id=100') == [('wal control',)]
        assert folder.joinpath('lcm.db').stat().st_ino != copy.joinpath('gen-1/lcm.db').stat().st_ino
        assert not list(folder.glob('lcm.db-*'))


def test_a6_stale_before_copy(world):
    home, copy, _ = world
    old, _ = built(world)
    forget(home, V)
    def check(step):
        if step == 'database':
            assert not old.exists()
            if (copy / 'view.json').exists():
                p = json.loads((copy / 'view.json').read_bytes())
                assert (copy / p['folder'] / 'lcm.db').is_file()
    built(world, seam=check)


def test_a7_busy_record(world):
    home, copy, _ = world
    with (copy / 'view.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        result = cv.build(home, COPY, lock_wait=lambda *_: False)
    assert result['code'] == 'view_busy'
    record = json.loads((copy / 'view-result.json').read_bytes())
    assert record['generation'] is None
    assert record['forgotten_sha256'] == cv.forgotten_digest({H})
    assert record['removed_sha256'] == hashlib.sha256(b'').hexdigest()


def reject_planted_event(source_db, folder, allowed):
    record = sql(source_db, 'SELECT * FROM capture_events WHERE session_id=? ORDER BY event_id LIMIT 1', (H,))[0]
    sql(folder / 'lcm.db', 'INSERT INTO capture_events VALUES (' + ','.join('?' for _ in record) + ')', record)
    with pytest.raises(AssertionError):
        scanner(folder / 'lcm.db', folder / 'lcm-large-outputs', allowed)


@pytest.mark.parametrize('failure', ['residue', 'rollup', 'integrity', 'sidecar', 'unexpected', 'disk'])
def test_r3_cleanup(world, monkeypatch, failure):
    home, copy, ids = world
    db = copy / 'gen-1/lcm.db'
    if failure == 'residue':
        sql(db, f"CREATE TRIGGER residue AFTER DELETE ON capture_events BEGIN INSERT OR IGNORE INTO messages(store_id,session_id,role,timestamp) VALUES ({ids[H][0]},'{V}','user',0); END")
    if failure == 'rollup':
        from hermes_lcm.db_bootstrap import ensure_temporal_rollup_tables
        with closing(sqlite3.connect(db)) as conn:
            ensure_temporal_rollup_tables(conn)
    if failure == 'integrity':
        original = cv.connect
        class BadIntegrity(sqlite3.Connection):
            def execute(self, statement, *args):
                return super().execute("SELECT 'synthetic corrupt'" if statement == 'PRAGMA integrity_check' else statement, *args)
        def connect(database, removed, readonly=True):
            if readonly:
                return original(database, removed, readonly)
            conn = sqlite3.connect(database, factory=BadIntegrity)
            conn.create_function('lineage_hit', 1, lambda v: cv.lineage(v) in removed)
            conn.create_function('metadata_hit', 1, lambda k: any(k == p + v or k == p + 'claude-code:' + v for p in cv.METADATA for v in removed))
            return conn
        monkeypatch.setattr(cv, 'connect', connect)
    def seam(step):
        if step == 'deleted' and failure == 'unexpected':
            raise RuntimeError('synthetic failure')
        if step == 'deleted' and failure == 'sidecar':
            next(copy.glob('*.tmp-*')).joinpath('lcm.db-journal').write_bytes(b'synthetic')
    count = len([p for p in copy.iterdir() if p.is_dir()])
    for _ in range(3):
        result = None
        try:
            result = cv.build(home, COPY, seam=seam, free_space=lambda _: 0 if failure == 'disk' else 10**12)
        except Exception:
            pass  # The assertion below pins unexpected-exception cleanup too.
        assert not list(copy.glob('*.tmp-*')), failure
        assert result and result['code'] != 'ok', result
        assert len([p for p in copy.iterdir() if p.is_dir()]) == count


@pytest.mark.parametrize('malformed', ['removed', 'payload'])
def test_r3_malformed_command(world, capsys, malformed):
    from hermes_lcm.scripts import lcm_x_cloud as command
    home, copy, _ = world
    path = copy / ('removed.json' if malformed == 'removed' else 'gen-1/lcm-large-outputs/bad.json')
    path.write_text('[]')
    assert command.main(['view', '--home', str(home), '--copy-id', COPY]) == 2
    output = capsys.readouterr()
    assert output.err == '' and len(output.out.splitlines()) == 1
    assert json.loads(output.out)['code'] == 'bad_database_block'
    assert json.loads((copy / 'view-result.json').read_bytes())['code'] == 'bad_database_block'


@pytest.mark.parametrize('window', ['source', 'locked'])
@pytest.mark.parametrize('churn', [False, True])
def test_r3_generation_retry(world, window, churn):
    home, copy, _ = world
    calls = []
    def seam(step):
        if step == window and (churn or not calls):
            old = (copy / 'current').read_text().strip()
            new = str(int(old) + 1)
            calls.append(int(old))
            shutil.copytree(copy / ('gen-' + old), copy / ('gen-' + new))
            (copy / 'current').write_text(new)
            shutil.rmtree(copy / ('gen-' + old))
    result = cv.build(home, COPY, seam=seam)
    assert calls == ([1, 2, 3] if churn else [1]), calls
    assert result['code'] == ('view_busy' if churn else 'ok'), result
    record = json.loads((copy / 'view-result.json').read_bytes())
    assert record['generation'] == (3 if churn else 2)
    if churn:
        assert not (copy / 'view.json').exists()
    else:
        assert json.loads((copy / 'view.json').read_bytes())['generation'] == 2


def test_r3_sweep_private():
    assert not hasattr(cv, 'sweep')


@pytest.mark.parametrize('clone_ok', [False, True])
def test_r3_floor_before_clone(world, clone_ok):
    home, copy, _ = world
    attempted = []
    size = (copy / 'gen-1/lcm.db').stat().st_size
    def clone(source, target):
        attempted.append(True)
        if clone_ok:
            shutil.copyfile(source, target)
        return clone_ok
    result = cv.build(home, COPY, free_space=lambda _: cv.FLOOR + 2 * size - 1, clone=clone)
    assert result['code'] == 'view_disk_floor', result
    assert not attempted and not list(copy.glob('*.tmp-*'))
    assert cv.build(home, COPY, free_space=lambda _: cv.FLOOR + 2 * size, clone=clone)['code'] == 'ok'
    assert attempted == [True]


def test_r3_byte_wal_guard(world):
    home, copy, _ = world
    db = copy / 'gen-1/lcm.db'
    with db.open('r+b') as handle:
        handle.seek(18)
        handle.write(b'\x02\x02')
    before = db.read_bytes()
    result = cv.build(home, COPY)
    assert not list(db.parent.glob('lcm.db-*'))
    assert result['code'] == 'bad_database_block'
    assert db.read_bytes() == before


@pytest.mark.parametrize('codes,expected', [(['view_busy', 'view_unsupported_key'], 2),
    (['bad_database_block', 'view_disk_floor'], 2), (['ok', 'view_busy'], 3), (['ok', 'ok'], 0)])
def test_r3_command_list_status(tmp_path, monkeypatch, capsys, codes, expected):
    from hermes_lcm.scripts import lcm_x_cloud as command
    records = [dict(code=code) for code in codes]
    monkeypatch.setattr(cv, 'build', lambda *_: records)
    assert command.main(['view', '--home', str(tmp_path)]) == expected
    output = capsys.readouterr()
    assert output.err == '' and len(output.out.splitlines()) == 1
    assert json.loads(output.out) == records


def hold_lock(path):
    code = "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); sys.stdin.read()"
    return subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', code, str(path)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.mark.parametrize('name', ['view.lock', 'space.lock'])
def test_r3_command_busy(world, monkeypatch, capsys, name):
    from hermes_lcm.scripts import lcm_x_cloud as command
    home, copy, _ = world
    held = hold_lock(copy / name if name == 'view.lock' else copy.parent / name)
    original = cv.wait_lock
    monkeypatch.setattr(cv, 'wait_lock', lambda fd, _: original(fd, 0))
    try:
        assert held.stdout.readline().strip() == 'held'
        # The injected wait is also checked directly, independent of the CLI defaults.
        assert cv.build(home, COPY, lock_wait=cv.wait_lock)['code'] == 'view_busy'
        assert command.main(['view', '--home', str(home), '--copy-id', COPY]) == 3
        output = capsys.readouterr()
        assert output.err == '' and len(output.out.splitlines()) == 1
        assert json.loads(output.out)['code'] == 'view_busy'
        assert json.loads((copy / 'view-result.json').read_bytes())['code'] == 'view_busy'
        assert not list(copy.glob('*.tmp-*'))
    finally:
        held.communicate('', timeout=30)
        assert held.returncode == 0


@pytest.mark.parametrize('message_index', [0, 1])
def test_r3_payload_row_removal(world, message_index):
    home, copy, ids = world
    db = copy / 'gen-1/lcm.db'
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    sql(db, 'DELETE FROM summary_nodes')
    removed_id, kept_id = ids[V][message_index:message_index + 2]
    # One ledger event can name both messages: the event leaves, message 6 stays.
    sql(db, 'UPDATE capture_events SET store_ids=? WHERE session_id=? AND store_ids=?',
        (json.dumps([removed_id, kept_id]), V, json.dumps([removed_id])))
    before_rows = rows(db)
    assert (removed_id, kept_id) == ((4, 5) if message_index == 0 else (5, 6))
    assert any(json.loads(row[0]) == [removed_id, kept_id] for row in sql(db, 'SELECT store_ids FROM capture_events WHERE session_id=?', (V,)))
    (copy / 'removed.json').write_text(json.dumps(dict(format=1, session_ids=[], row_ids=[removed_id])))
    row_view, _ = built(world)
    assert_row_id_diff(db, row_view / 'lcm.db', before_rows, removed_id)
    assert sql(row_view / 'lcm.db', 'SELECT * FROM messages WHERE store_id=?', (kept_id,)) == sql(db, 'SELECT * FROM messages WHERE store_id=?', (kept_id,))
    (copy / 'removed.json').unlink()
    cases = [('kept-owner.json', V, [removed_id], False),
             ('ownerless.json', None, [removed_id], False),
             ('shared-owner.json', V, [removed_id, kept_id], True),
             ('removed-session.json', V, [ids[H][0]], False)]
    forget(home, H)
    for name, owner, references, _ in cases:
        body = dict(content='synthetic', **({'session_id': owner} if owner else {}))
        (db.parent / 'lcm-large-outputs' / name).write_text(json.dumps(body))
        for row_id in references:
            sql(db, 'UPDATE messages SET content=content || ? WHERE store_id=?', (f' [Externalized payload: x; ref={name}]', row_id))
    before = sql(db, 'SELECT * FROM messages WHERE session_id=? AND store_id!=?', (V, removed_id))
    first, _ = built(world)
    (copy / 'removed.json').write_text(json.dumps(dict(format=1, session_ids=[], row_ids=[removed_id])))
    folder, _ = built(world)
    assert folder != first
    assert not sql(folder / 'lcm.db', 'SELECT 1 FROM messages WHERE store_id=?', (removed_id,))
    assert sql(folder / 'lcm.db', 'SELECT * FROM messages WHERE session_id=?', (V,)) == before
    for name, _, _, present in cases:
        assert (folder / 'lcm-large-outputs' / name).exists() == present, name
    # Every newly omitted file contributes to e, independently of the same row omission.
    previous = folder
    for name in ('kept-owner.json', 'ownerless.json'):
        (db.parent / 'lcm-large-outputs' / name).unlink()
        folder, _ = built(world)
        assert folder != previous, name
        previous = folder


def assert_row_id_diff(source, view, before, removed_id):
    """Independent V3 oracle: compare every logical table, including kept rows."""
    after = rows(view)
    assert after.keys() == before.keys()
    occurrences = {r[0] for r in sql(source, 'SELECT occ_id FROM occurrence_bindings WHERE store_id=?', (removed_id,))}
    chunks = {r[0] for r in sql(source, 'SELECT chunk_id FROM lcm_chunk_meta WHERE store_id=?', (removed_id,))}
    keys = {'messages': ('store_id', {removed_id}), 'host_uid_bindings': ('store_id', {removed_id}),
            'occurrence_bindings': ('store_id', {removed_id}), 'stamp_observations': ('occ_id', occurrences),
            'lcm_chunk_meta': ('store_id', {removed_id}), 'lcm_chunk_vectors': ('chunk_id', chunks),
            'lcm_chunk_binary': ('chunk_id', chunks), 'messages_fts_docsize': ('id', {removed_id}),
            'messages_fts_stem_docsize': ('id', {removed_id})}
    for table, records in before.items():
        columns = [r[1] for r in sql(source, f'PRAGMA table_info("{table}")')]
        def leaves(row):
            fields = dict(zip(columns, row))
            if table in keys:
                column, values = keys[table]
                return fields[column] in values
            if table == 'capture_events':
                return removed_id in json.loads(fields['store_ids'])
            if table == 'message_relations':
                return removed_id in (fields['store_id'], fields['related_store_id'])
            if table == 'ledger_emissions':
                return fields['recovered_store_id'] == removed_id or fields['next_occ_id'] in occurrences or removed_id in json.loads(fields['source_store_ids'])
            return False
        if table in ('messages_fts', 'messages_fts_stem'):
            # FTS content is external; assert its visible rowids and bytes, not free pages.
            expected = [r for r in sql(source, f'SELECT rowid,* FROM {table}') if r[0] != removed_id]
            assert sql(view, f'SELECT rowid,* FROM {table}') == expected, table
        else:
            assert after[table] == [r for r in records if not leaves(r)], table
    assert rows(source) == before, 'the byte copy stays untouched'


@pytest.mark.parametrize('kind', ['entity', 'unknown'])
@pytest.mark.parametrize('removal', ['local', 'session', 'row'])
@pytest.mark.parametrize('existing', [False, True])
def test_r4_tables_before_reuse(world, kind, removal, existing):
    home, copy, _ = world
    db = copy / 'gen-1/lcm.db'
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    if kind == 'entity':
        from hermes_lcm.entity_store import EntityStore
        with closing(EntityStore(db, profile_key='synthetic-profile')) as entities:
            entities.create_entity('synthetic entity')
        assert cc.leave_readable(db)
    else:
        sql(db, 'CREATE TABLE future_feature(value TEXT)')
        sql(db, "INSERT INTO future_feature VALUES ('synthetic')")
    if existing:
        old_view, _ = built(world)
        old = (copy / 'view.json').read_bytes()
    else:
        old = None
    absent = 'synthetic-forgotten-with-no-classified-row'
    if removal == 'local':
        forget(home, absent)
    else:
        (copy / 'removed.json').write_text(json.dumps(dict(format=1,
            session_ids=[absent] if removal == 'session' else [], row_ids=[99999] if removal == 'row' else [])))
    before = rows(db)
    result = cv.build(home, COPY)
    assert result['code'] == 'view_unsupported_table', result
    assert ((copy / 'view.json').read_bytes() if (copy / 'view.json').exists() else None) == old
    assert sorted(p.name for p in copy.glob('view-*') if p.is_dir()) == ([old_view.name] if existing else [])
    assert not list(copy.glob('*.tmp-*'))
    assert rows(db) == before


@pytest.mark.parametrize('kind', ['entity', 'unknown'])
def test_r4_empty_removed_links(world, kind):
    home, copy, _ = world
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    db = copy / 'gen-1/lcm.db'
    if kind == 'entity':
        from hermes_lcm.entity_store import EntityStore
        with closing(EntityStore(db, profile_key='synthetic-profile')) as entities:
            entities.create_entity('synthetic entity')
        assert cc.leave_readable(db)
    else:
        sql(db, 'CREATE TABLE future_feature(value TEXT)')
        sql(db, "INSERT INTO future_feature VALUES ('synthetic')")
    before = rows(db)
    folder, _ = built(world, free_space=lambda _: 0)
    assert (folder / 'lcm.db').stat().st_ino == db.stat().st_ino
    assert rows(folder / 'lcm.db') == before


@pytest.mark.parametrize('changed', ['none', 'forgotten', 'removed', 'rules'])
def test_r4_failed_successor(world, changed):
    home, copy, _ = world
    old_view, _ = built(world)
    old = (copy / 'view.json').read_bytes()
    shutil.copytree(copy / 'gen-1', copy / 'gen-2')
    (copy / 'current').write_text('2')
    if changed == 'forgotten':
        forget(home, 'synthetic-absent-session')
    elif changed == 'removed':
        (copy / 'removed.json').write_text(json.dumps(dict(format=1, session_ids=[], row_ids=[99999])))
    elif changed == 'rules':
        pointer = json.loads(old)
        pointer['rules'] = cv.RULES + 1
        (copy / 'view.json').write_text(json.dumps(pointer))
    steps = []
    def seam(step):
        if step == 'database':  # After the floor and retirement, before publication.
            steps.append(step)
            raise RuntimeError('synthetic successor failure')
    result = cv.build(home, COPY, seam=seam, free_space=lambda _: 10**12)
    assert steps == ['database'] and result['code'] == 'bad_database_block', result
    assert old_view.exists() == (changed == 'none')
    assert (copy / 'view.json').exists() == (changed == 'none')
    if changed == 'none':
        assert (copy / 'view.json').read_bytes() == old
        assert sql(old_view / 'lcm.db', 'PRAGMA integrity_check') == [('ok',)]
    assert not list(copy.glob('*.tmp-*'))
    folder, pointer = built(world)
    assert pointer['generation'] == 2 and folder != old_view
    assert not old_view.exists()


@pytest.mark.parametrize('busy', [False, True])
def test_r4_retirement_lock_order(world, monkeypatch, busy):
    home, copy, _ = world
    old_view, _ = built(world)
    forget(home, V)
    events, waited = [], []
    original = fcntl.flock
    original_retire = cv.retire
    def retire(*args):
        remove = original_retire(*args)
        def attempt():
            events.append('retire')
            return remove()
        return attempt if remove else None
    monkeypatch.setattr(cv, 'retire', retire)
    with ExitStack() as stack:
        names = {}
        for name, path in [('view', copy / 'view.lock'), ('generation', copy / 'gen-1'),
                           ('stale', old_view), ('space', copy.parent / 'space.lock')]:
            path.touch(exist_ok=True) if name == 'space' else None
            names[os.stat(path).st_ino] = name
        if busy:
            held = os.open(old_view, os.O_RDONLY)
            stack.callback(os.close, held)
            original(held, fcntl.LOCK_SH)
        def flock(fd, operation):
            name = names.get(os.fstat(fd).st_ino, 'other')
            events.append((name, operation))
            return original(fd, operation)
        def lock_wait(fd, seconds):
            waited.append(names[os.fstat(fd).st_ino])
            return cv.wait_lock(fd, 0)
        monkeypatch.setattr(cv.fcntl, 'flock', flock)
        result = cv.build(home, COPY, lock_wait=lock_wait, free_space=lambda _: 10**12)
        assert result['code'] == 'ok', result
        assert waited == ['view', 'space']
        assert events[:5] == [('view', fcntl.LOCK_EX | fcntl.LOCK_NB),
                             ('generation', fcntl.LOCK_SH),
                             ('space', fcntl.LOCK_EX | fcntl.LOCK_NB), 'retire',
                             ('stale', fcntl.LOCK_EX | fcntl.LOCK_NB)]
        assert old_view.exists() == busy
    # A busy stale view is left for a later nonblocking sweep after publication.
    built(world)
    assert not old_view.exists()


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('kind', ['metadata', 'lineage'])
def test_r3_keys_before_reuse(world, existing, kind):
    home, copy, _ = world
    db = copy / 'gen-1/lcm.db'
    absent = 'synthetic-absent-forgotten-session'
    if kind == 'metadata':
        sql(db, 'INSERT INTO metadata VALUES (?,?)', ('future:' + absent, 'synthetic'))
    else:
        sql(db, "INSERT INTO ingest_views VALUES (?,1,'ingest','ready',?,?)", ('future:' + absent, b'[]', b'synthetic'))
    if existing:
        built(world)
        old = (copy / 'view.json').read_bytes()
    else:
        sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    forget(home, absent)
    assert cv.build(home, COPY)['code'] == 'view_unsupported_key'
    if existing:
        assert (copy / 'view.json').read_bytes() == old
    else:
        assert not (copy / 'view.json').exists()


def test_r3_space_lock_floor(world):
    home, copy, _ = world
    checked = []
    def free_space(_):
        with (copy.parent / 'space.lock').open('a') as handle:
            busy = False
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy = True
            assert busy, 'free-space read must hold space.lock'
        checked.append(True)
        return 10**12
    def seam(step):
        if step in ('database', 'deleted', 'payloads'):
            with (copy.parent / 'space.lock').open('a') as handle:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if step in ('rechecked', 'renamed', 'published'):
            with (copy.parent / 'space.lock').open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = cv.build(home, COPY, free_space=free_space, seam=seam)
    assert checked and result['code'] == 'ok', result


def test_r3_payload_copy_floor(world, monkeypatch):
    home, copy, _ = world
    size = (copy / 'gen-1/lcm.db').stat().st_size
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    def no_links(*_):
        raise OSError('synthetic no hard links')
    original = os.link
    def payload_links(source, target):
        return no_links() if Path(target).parent.name == 'lcm-large-outputs' else original(source, target)
    monkeypatch.setattr(cv.os, 'link', payload_links)
    payload_bytes = sum(p.stat().st_size for p in (copy / 'gen-1/lcm-large-outputs').iterdir())
    floor = cv.FLOOR + 2 * size + payload_bytes
    result = cv.build(home, COPY, free_space=lambda _: floor - 1)
    assert result['code'] == 'view_disk_floor', result
    assert not list(copy.glob('*.tmp-*')) and not (copy / 'view.json').exists()
    assert cv.build(home, COPY, free_space=lambda _: floor)['code'] == 'ok'


def test_r3_scanner_column_values(world):
    folder, _ = built(world)
    allowed = ('synthetic-kept-fork-with-distinctive-id',)
    sql(folder / 'lcm.db', 'INSERT INTO metadata VALUES (?,?)',
        ('future:' + H, 'prefix-' + allowed[0] + '-suffix'))
    with pytest.raises(AssertionError):
        scanner(folder / 'lcm.db', folder / 'lcm-large-outputs', allowed)


def test_r3_retry_cleanup(world):
    home, copy, _ = world
    passes = []
    def seam(step):
        if step == 'deleted':
            passes.append(True)
            forget(home, 'synthetic-changing-absent-session-' + str(len(passes)))
    result = cv.build(home, COPY, seam=seam)
    assert result['code'] == 'view_busy' and len(passes) == 3, result
    assert not list(copy.glob('*.tmp-*'))
    assert not (copy / 'view.json').exists()



def test_dangling_pointer_with_stale_digest(world):
    """Review F1: external removal of a folder must not wedge its stale pointer."""
    home, copy, _ = world
    folder, _ = built(world)
    forget(home, V)
    shutil.rmtree(folder)
    codes = [cv.build(home, COPY)['code'] for _ in range(3)]
    assert codes == ['ok'] * 3, codes
    pointer = json.loads((copy / 'view.json').read_bytes())
    assert (copy / pointer['folder'] / 'lcm.db').is_file()
    assert pointer['forgotten_sha256'] == cv.forgotten_digest({H, V})


@pytest.mark.parametrize('kind', ['Externalized payload', "GC'd externalized tool output", 'Externalized LCM ingest payload'])
@pytest.mark.parametrize('column', ['content', 'tool_calls'])
def test_r5_payload_reference_matching(world, kind, column):
    from hermes_lcm.mcp_reader import StoreReader
    home, copy, ids = world
    db = copy / 'gen-1/lcm.db'
    # Row removal in a kept session, with neither summary nor other real reference.
    sql(db, 'DELETE FROM summary_nodes')
    rid, kept = ids[V][:2]
    names = ('p.json', 'kept.json', 'split.json')
    folder = copy / 'gen-1/lcm-large-outputs'
    for name in names:
        (folder / name).write_text(json.dumps({'session_id': V, 'content': WORD if name != 'kept.json' else 'visible payload control'}))
    genuine = ' '.join(f'[{kind}: real; ref={name}]' for name in names)
    sql(db, f'UPDATE messages SET {column}=? WHERE store_id=?', (genuine, rid))
    text = f'[{kind}:\nquoted; ref=p.json] [{kind}: real; ref=kept.json]'
    # Joining columns would manufacture a reference that the uploader never sees.
    values = (text + f' [{kind}: split;', 'ref=split.json]') if column == 'content' else (f'[{kind}: split;', 'ref=split.json] ' + text)
    sql(db, 'UPDATE messages SET content=?,tool_calls=? WHERE store_id=?', (*values, kept))
    (copy / 'removed.json').write_text(json.dumps({'format': 1, 'session_ids': [], 'row_ids': [rid]}))
    view, _ = built(world)
    assert not (view / 'lcm-large-outputs/p.json').exists(), 'a multiline quote is not a kept reference'
    assert not (view / 'lcm-large-outputs/split.json').exists(), 'text columns are matched separately'
    assert (view / 'lcm-large-outputs/kept.json').read_bytes() == (folder / 'kept.json').read_bytes()
    reader = StoreReader(view / 'lcm.db', view, config=LCMConfig(embeddings_enabled=False), environ={})
    missing, missing_error = reader.call('lcm_expand', {'externalized_ref': 'never-existed.json'})
    for name in ('p.json', 'split.json'):
        answer, is_error = reader.call('lcm_expand', {'externalized_ref': name})
        assert answer.replace(name, 'never-existed.json') == missing and is_error == missing_error
        assert WORD not in answer
    answer, is_error = reader.call('lcm_expand', {'externalized_ref': 'kept.json'})
    assert not is_error and 'visible payload control' in answer


def space_is_held(home):
    path = home / 'cloud/replicas/space.lock'
    if not path.exists():
        return False
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


@pytest.mark.parametrize('phase', ['stage', 'delta'])
def test_r5_upload_space_lock_lifetime(upload_fixture, monkeypatch, phase):
    home, transport, _, _ = upload_fixture
    if phase == 'delta':
        pub(upload_fixture)
        grow(upload_fixture)
    reads, writes, network = [], [], []
    def free_bytes(_):
        reads.append(True)
        assert space_is_held(home), 'uploader free-disk checks must hold space.lock'
        assert home.joinpath('cloud/replicas').stat().st_mode & 0o777 == 0o700
        with home.joinpath('cloud/upload.lock').open('a') as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return 10**12
    references, encode = cp._references, cp.cloud_delta.encode_delta
    def payloads(*args):
        assert space_is_held(home), 'payload writes stay inside the reservation'
        writes.append('payloads')
        return references(*args)
    def delta(*args):
        assert space_is_held(home), 'delta objects stay inside the reservation'
        writes.append('delta')
        return encode(*args)
    def transfer(method, key):
        network.append(method)
        assert not space_is_held(home), 'no network transfer while holding space.lock'
    monkeypatch.setattr(cp, '_references', payloads)
    monkeypatch.setattr(cp.cloud_delta, 'encode_delta', delta)
    transport.hook = transfer
    result = pub(upload_fixture, free_bytes=free_bytes)
    assert len(reads) == (2 if phase == 'delta' else 1)
    assert writes == (['payloads', 'delta'] if phase == 'delta' else ['payloads'])
    assert network and not space_is_held(home)
    assert bool(result['database']['deltas']) == (phase == 'delta')


@pytest.mark.parametrize('phase', ['stage', 'delta'])
def test_r5_upload_space_busy(upload_fixture, monkeypatch, phase):
    home = upload_fixture[0]
    if phase == 'delta':
        pub(upload_fixture)
        grow(upload_fixture)
    root = home / 'cloud/replicas'
    root.mkdir(mode=0o700, exist_ok=True)
    state = home / 'cloud/state.json'
    before = state.read_bytes() if state.exists() else None
    held, waits = [], []
    def take():
        process = hold_lock(root / 'space.lock')
        held.append(process)
        assert process.stdout.readline().strip() == 'held'
    if phase == 'stage':
        take()
    else:
        hash_pages = cp.cloud_delta.hash_pages
        def hashed(*args, **kwargs):
            result = hash_pages(*args, **kwargs)
            take()
            return result
        monkeypatch.setattr(cp.cloud_delta, 'hash_pages', hashed)
    def bounded(fd, seconds):
        waits.append(seconds)
        return cv.wait_lock(fd, 0)
    monkeypatch.setattr(cp, 'wait_lock', bounded, raising=False)
    try:
        code = 'ok'
        try:
            pub(upload_fixture)
        except cp.Refused as error:
            code = error.code
            assert error.detail == {}
        assert code == 'space_busy', code
        assert 'space_busy' in cp.SKIPS
        assert waits == ([600] if phase == 'stage' else [600, 600])
        assert not home.joinpath('cloud/staging').exists()
        assert (state.read_bytes() if state.exists() else None) == before
    finally:
        for process in held:
            process.communicate('', timeout=30)
            assert process.returncode == 0


def test_r5_upload_competes_with_view(world, monkeypatch):
    home, copy, _ = world
    sql(home / 'lcm.db', "DELETE FROM metadata WHERE key LIKE 'capture:forgotten:%'")
    # Keep the forget out of the uploader's origin; it belongs to this replica.
    (copy / 'removed.json').write_text(json.dumps({'format': 1, 'session_ids': [H], 'row_ids': []}))
    size = (copy / 'gen-1/lcm.db').stat().st_size
    available = cv.FLOOR + 2 * size
    codes, waits, remaining = [], [], []
    stage = cp._stage
    def bounded(fd, seconds):
        waits.append(seconds)
        return cv.wait_lock(fd, 0)
    def staged(*args):
        result = stage(*args)
        # Try the second operation while staging still holds its reservation.
        codes.append(cv.build(home, COPY, lock_wait=bounded, free_space=lambda _: available)['code'])
        return result
    monkeypatch.setattr(cp, '_stage', staged)
    transport = cp.DirTransport(home.parent / 'bucket', 'copies/' + COPY)
    put = transport.put_new
    def upload(key, data):
        if not remaining:
            used = sum(p.stat().st_size for p in home.joinpath('cloud/staging').rglob('*')
                       if p.is_file() and p.stat().st_nlink == 1)
            remaining.append(available - used)
            codes.append(cv.build(home, COPY, free_space=lambda _: remaining[0])['code'])
        return put(key, data)
    monkeypatch.setattr(transport, 'put_new', upload)
    result = cp.publish(home, COPY, transport, 'copies/' + COPY, free_bytes=lambda _: available,
                        guard=lambda *a: None, forget_items=lambda *a: [], publisher=lambda: None,
                        chain_check=lambda *a: None, after_create=lambda m: {}, prepare_manifest=lambda m: None)
    assert codes == ['view_busy', 'view_disk_floor'], codes
    assert waits == [600, 600]
    assert remaining and cp.DISK_MARGIN <= remaining[0] < cv.FLOOR + 2 * size
    assert result['generation'] == 1 and not (copy / 'view.json').exists()


@pytest.mark.parametrize('reader_holds_view', [False, True])
def test_r6_abandoned_build_before_floor(world, tmp_path, reader_holds_view):
    home, copy, _ = world
    old_view, _ = built(world)
    pointer = (copy / 'view.json').read_bytes()
    old_database = (old_view / 'lcm.db').read_bytes()
    old_inode = (old_view / 'lcm.db').stat().st_ino
    shutil.copytree(copy / 'gen-1', copy / 'gen-2')
    (copy / 'current').write_text('2\n')
    name = 'view-2-1-' + 'a' * 16 + '.tmp-' + 'b' * 32
    abandoned = copy / name
    abandoned.mkdir()
    # Direct writes model SIGKILL: no builder ExitStack gets to clean this up.
    shutil.copyfile(copy / 'gen-2/lcm.db', abandoned / 'lcm.db')
    abandoned_bytes = (abandoned / 'lcm.db').stat().st_size
    assert 0 < abandoned_bytes < 5_000_000
    unmatched = copy / (name + 'x')
    unmatched.mkdir()
    (unmatched / 'keep').write_bytes(b'unmatched control')
    outside = tmp_path / 'symlink-target'
    outside.mkdir()
    (outside / 'keep').write_bytes(b'symlink control')
    symlink = copy / ('view-2-1-' + 'c' * 16 + '.tmp-' + 'd' * 32)
    symlink.symlink_to(outside, target_is_directory=True)
    floor = cv.FLOOR + 2 * (copy / 'gen-2/lcm.db').stat().st_size
    checks, steps = [], []
    def free_space(_):
        available = floor - (abandoned_bytes if abandoned.exists() else 0)
        checks.append(available)
        return available
    def seam(step):
        assert old_view.exists()
        assert (old_view / 'lcm.db').read_bytes() == old_database
        assert (old_view / 'lcm.db').stat().st_ino == old_inode
        assert symlink.is_symlink() and (outside / 'keep').read_bytes() == b'symlink control'
        assert (unmatched / 'keep').read_bytes() == b'unmatched control'
        if step != 'published':
            assert (copy / 'view.json').read_bytes() == pointer
        steps.append(step)
    with ExitStack() as stack:
        if reader_holds_view:
            held = os.open(old_view, os.O_RDONLY)
            stack.callback(os.close, held)
            fcntl.flock(held, fcntl.LOCK_SH)
        result = cv.build(home, COPY, free_space=free_space, seam=seam, clone=lambda *_: False)
        assert result['code'] == 'ok', result
        assert checks == [floor] and 'database' in steps and 'published' in steps
        assert not abandoned.exists()
        assert old_view.exists() == reader_holds_view
        successor = json.loads((copy / 'view.json').read_bytes())
        assert successor['generation'] == 2 and (copy / successor['folder'] / 'lcm.db').is_file()
    assert symlink.is_symlink() and (outside / 'keep').read_bytes() == b'symlink control'
    assert (unmatched / 'keep').read_bytes() == b'unmatched control'


@pytest.mark.parametrize('enough_free', [False, True])
def test_r6_abandoned_build_removal_failure(world, monkeypatch, capsys, enough_free):
    home, copy, _ = world
    abandoned = copy / ('view-1-1-' + 'e' * 16 + '.tmp-' + 'f' * 32)
    abandoned.mkdir()
    shutil.copyfile(copy / 'gen-1/lcm.db', abandoned / 'lcm.db')
    original = shutil.rmtree
    def fail_removal(path, *args, **kwargs):
        if Path(path) == abandoned:
            raise PermissionError('synthetic removal refusal')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(cv.shutil, 'rmtree', fail_removal)
    floor = cv.FLOOR + 2 * (copy / 'gen-1/lcm.db').stat().st_size
    checks = []
    def free_space(_):
        checks.append(True)
        return floor if enough_free else floor - 1
    result = cv.build(home, COPY, free_space=free_space, clone=lambda *_: False)
    assert result['code'] == ('ok' if enough_free else 'view_disk_floor'), result
    assert checks == [True] and abandoned.is_dir()
    captured = capsys.readouterr()
    assert 'view_temp_cleanup_failed' in captured.err and not captured.out
    assert 'synthetic removal refusal' not in captured.err
    assert (copy / 'view.json').exists() == enough_free


def r7_advance(copy, generation):
    shutil.copytree(copy / 'gen-1', copy / f'gen-{generation}', copy_function=os.link)
    (copy / 'current').write_text(f'{generation}\n')


def r7_kill_after_rename(home):
    child = '''
import os, signal, sys
import tests.conftest
from hermes_lcm import cloud_view as cv
def seam(step):
    if step == 'renamed':
        os.kill(os.getpid(), signal.SIGKILL)
cv.build(sys.argv[1], sys.argv[2], seam=seam, clone=lambda *_: False)
'''
    result = subprocess.run([sys.executable, '-c', child, str(home), COPY],
                            cwd=ROOT, capture_output=True)
    import signal
    assert result.returncode == -signal.SIGKILL, result.stderr


def r7_bytes(folder):
    return sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())


def test_r7_renamed_orphan_before_floor(world):
    home, copy, _ = world
    current, _ = built(world)
    r7_advance(copy, 2)
    r7_kill_after_rename(home)
    orphan, = copy.glob('view-2-*')
    assert re.fullmatch(cv.VIEW, orphan.name) and current.is_dir()
    orphan_bytes = r7_bytes(orphan)
    assert 0 < orphan_bytes < 5_000_000
    r7_advance(copy, 3)
    floor = cv.FLOOR + 2 * (copy / 'gen-3/lcm.db').stat().st_size
    checks = []
    pointer = (copy / 'view.json').read_bytes()
    def free_space(_):
        assert current.is_dir() and (copy / 'view.json').read_bytes() == pointer
        available = floor - (orphan_bytes if orphan.exists() else 0)
        checks.append(available)
        return available
    result = cv.build(home, COPY, free_space=free_space, clone=lambda *_: False)
    assert result['code'] == 'ok', result
    assert checks == [floor] and not orphan.exists()
    assert json.loads((copy / 'view.json').read_bytes())['generation'] == 3


def test_r7_same_identity_adopts_orphan(world):
    home, copy, _ = world
    current, _ = built(world)
    r7_advance(copy, 2)
    r7_kill_after_rename(home)
    orphan, = copy.glob('view-2-*')
    inode = (orphan / 'lcm.db').stat().st_ino
    database = (orphan / 'lcm.db').read_bytes()
    stale = copy / ('view-9-1-' + 'e' * 16)
    stale.mkdir()
    (stale / 'keep').write_bytes(b'reclaimable control')
    checks, reused = [], []
    def free_space(_):
        checks.append(True)
        return 0
    def seam(step):
        if step == 'reuse':
            reused.append(True)
            assert current.is_dir() and not stale.exists()
            assert orphan.is_dir() and (orphan / 'lcm.db').stat().st_ino == inode
    result = cv.build(home, COPY, free_space=free_space, seam=seam, clone=lambda *_: False)
    assert result['code'] == 'ok', result
    assert reused == [True] and checks == [] and not stale.exists()
    assert json.loads((copy / 'view.json').read_bytes())['folder'] == orphan.name
    assert (orphan / 'lcm.db').stat().st_ino == inode
    assert (orphan / 'lcm.db').read_bytes() == database


def test_r7_released_reader_before_floor(world):
    home, copy, _ = world
    old, _ = built(world)
    with ExitStack() as stack:
        held = os.open(old, os.O_RDONLY)
        stack.callback(os.close, held)
        fcntl.flock(held, fcntl.LOCK_SH)
        r7_advance(copy, 2)
        current, _ = built(world)
        assert old.is_dir()
        old_bytes = r7_bytes(old)
        r7_advance(copy, 3)
        floor = cv.FLOOR + 2 * (copy / 'gen-3/lcm.db').stat().st_size
        pointer = (copy / 'view.json').read_bytes()
        checks = []
        def free_space(_):
            assert current.is_dir() and (copy / 'view.json').read_bytes() == pointer
            available = floor - (old_bytes if old.exists() else 0)
            checks.append(available)
            return available
        result = cv.build(home, COPY, free_space=free_space, clone=lambda *_: False)
        assert result['code'] == 'view_disk_floor' and old.is_dir()
        assert checks == [floor - old_bytes]
    checks.clear()
    result = cv.build(home, COPY, free_space=free_space, clone=lambda *_: False)
    assert result['code'] == 'ok', result
    assert checks == [floor] and not old.exists()


def test_r7_held_temp_before_floor(world, tmp_path):
    home, copy, _ = world
    current, _ = built(world)
    r7_advance(copy, 2)
    name = 'view-2-1-' + 'a' * 16 + '.tmp-' + 'b' * 32
    held_temp = copy / name
    held_temp.mkdir()
    (held_temp / 'keep').write_bytes(b'held temporary control')
    free_temp = copy / ('view-2-1-' + 'c' * 16 + '.tmp-' + 'd' * 32)
    free_temp.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep').write_bytes(b'outside control')
    symlink = copy / ('view-2-1-' + 'e' * 16)
    symlink.symlink_to(outside, target_is_directory=True)
    temp_symlink = copy / ('view-2-1-' + 'f' * 16 + '.tmp-' + 'a' * 32)
    temp_symlink.symlink_to(outside, target_is_directory=True)
    near_misses = [copy / (name + suffix) for suffix in ('x', '\n')]
    near_misses += [copy / 'gen-9', copy / ('view-0-1-' + 'a' * 16),
                    copy / (name[:-1]), copy / 'view.json.tmp-control']
    for path in near_misses:
        path.mkdir()
        (path / 'keep').write_bytes(b'near miss control')
    regular = copy / ('view-3-1-' + 'a' * 16)
    regular.write_bytes(b'regular file control')
    floor = cv.FLOOR + 2 * (copy / 'gen-2/lcm.db').stat().st_size
    checks = []
    def free_space(_):
        checks.append(True)
        assert current.is_dir() and held_temp.is_dir() and not free_temp.exists()
        return floor - 1
    with ExitStack() as stack:
        held = os.open(held_temp, os.O_RDONLY)
        stack.callback(os.close, held)
        fcntl.flock(held, fcntl.LOCK_SH)
        result = cv.build(home, COPY, free_space=free_space, clone=lambda *_: False)
        assert result['code'] == 'view_disk_floor', result
        assert checks == [True] and (held_temp / 'keep').read_bytes() == b'held temporary control'
    assert all((p / 'keep').read_bytes() == b'near miss control' for p in near_misses)
    assert symlink.is_symlink() and temp_symlink.is_symlink()
    assert (outside / 'keep').read_bytes() == b'outside control'
    assert regular.read_bytes() == b'regular file control'
    checks.clear()
    cv._sweep(copy, temporary_only=True)
    assert not held_temp.exists() and current.is_dir()


@pytest.mark.parametrize('stderr_mode', ['none', 'raises'])
@pytest.mark.parametrize('enough_free', [False, True])
def test_r7_cleanup_stderr_cannot_affect_build(world, monkeypatch, capsys, stderr_mode, enough_free):
    home, copy, _ = world
    abandoned = copy / ('view-1-1-' + 'e' * 16 + '.tmp-' + 'f' * 32)
    abandoned.mkdir()
    original = shutil.rmtree
    def fail_removal(path, *args, **kwargs):
        if Path(path) == abandoned:
            raise PermissionError('synthetic removal refusal')
        return original(path, *args, **kwargs)
    class BrokenStderr:
        def write(self, _):
            raise RuntimeError('synthetic stderr refusal')
    floor = cv.FLOOR + 2 * (copy / 'gen-1/lcm.db').stat().st_size
    checks = []
    def free_space(_):
        checks.append(True)
        return floor if enough_free else floor - 1
    with monkeypatch.context() as patch:
        patch.setattr(cv.shutil, 'rmtree', fail_removal)
        patch.setattr(cv.sys, 'stderr', None if stderr_mode == 'none' else BrokenStderr())
        result = cv.build(home, COPY, free_space=free_space, clone=lambda *_: False)
    assert result['code'] == ('ok' if enough_free else 'view_disk_floor'), result
    assert checks == [True] and abandoned.is_dir()
    assert not capsys.readouterr().out
    assert (copy / 'view.json').exists() == enough_free
    assert json.loads((copy / 'view-result.json').read_bytes())['code'] == result['code']
