"""PR-S format 1: only test writers register a records table (I5/I6)."""
from contextlib import closing
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_lcm import store as sm
from hermes_lcm import maintenance
from tests.harness import engine_context

TABLE = 'test_records'
DDL = ('CREATE TABLE IF NOT EXISTS test_records ('
       'owner_store_id INTEGER, owner_timestamp REAL, owner_ingested_at REAL, owner_content TEXT, payload TEXT)')
COLUMNS = dict(owner_store_id='store_id', owner_timestamp='timestamp',
               owner_ingested_at='ingested_at', owner_content='content')


def register():
    sm.register_records_table(TABLE, DDL, COLUMNS)


@pytest.fixture
def pair(tmp_path):
    old = dict(sm._RECORDS_TABLES)
    register()
    store = sm.MessageStore(tmp_path / 'lcm.db')
    sid = store.append('S', dict(role='user', content='original owner', timestamp=1.0), conversation_id='conv')
    columns = ','.join(COLUMNS.values())
    values = dict(zip(COLUMNS, store.connection.execute(
        f'SELECT {columns} FROM messages WHERE store_id=?', (sid,)).fetchone()), payload='record')
    writer = sm.open_records_writer(store)
    try:
        assert writer.insert(TABLE, values) is None
        yield store, writer, values
    finally:
        writer.close()
        store.close()
        sm._RECORDS_TABLES.clear()
        sm._RECORDS_TABLES.update(old)


def read(store):
    with sm.records_read_cut(store) as cut:
        return sm.read_records(cut, TABLE)


def counts(writer):
    return (writer.connection.execute(f'SELECT count(*) FROM {TABLE}').fetchone()[0],
            writer.connection.execute('SELECT count(*) FROM _records_purge_intents').fetchone()[0])


def test_store_never_creates_records(tmp_path):
    store = sm.MessageStore(tmp_path / 'lcm.db')
    try:
        assert not sm.records_path(store).exists()
        store.append('S', dict(role='user', content='canonical'))
        assert read(store) == ([], 0)
    finally:
        store.close()
    assert not (tmp_path / 'lcm.db.records').exists()


def test_store_constructor_never_touches_records_path(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('MessageStore.__init__ touched records path')
    monkeypatch.setattr(sm, 'records_path', forbidden)
    store = sm.MessageStore(tmp_path / 'lcm.db')
    store.close()


def test_engine_ingest_compact_close_never_creates_records(tmp_path, monkeypatch):
    import hermes_lcm.engine as em
    monkeypatch.setattr(em, 'summarize_with_escalation', lambda **kw: ('Earlier turns.', 1))
    with engine_context(tmp_path, window=4000, embeddings_enabled=False, fresh_tail_count=2,
                        fresh_tail_max_tokens=500, leaf_chunk_tokens=400,
                        background_leaf_prep_enabled=False) as engine:
        messages = [dict(role='user' if i % 2 == 0 else 'assistant', content=('word ' * 200), timestamp=float(i))
                    for i in range(30)]
        engine.ingest(messages)
        engine.compress(messages, current_tokens=12000, bypass_cooldown=True)
        assert engine._dag.get_session_nodes('S'), 'the never-created proof must exercise real compaction'
        assert not sm.records_path(engine._store).exists()
    assert not (tmp_path / 'lcm.db.records').exists()


def test_claude_code_capture_and_audit_never_create_records(tmp_path):
    from hermes_lcm import claude_code_store as cc, claude_code_audit as ca
    from tests.test_claude_code_store import _entry, _transcript
    home = tmp_path / 'capture'
    transcript = _transcript(tmp_path, [_entry('user', 'u1', 'canonical capture')])
    store = cc.open_store(cc.admit(home, hermes_home=tmp_path / 'hermes-reference'), home)
    try:
        assert cc.drain(store, cc.Lineage('s1', None, transcript)) == 'drained'
        cc.to_rollback_journal(store.connection)
    finally:
        store.close()
    ca.audit(home)
    assert not (home / 'lcm.db.records').exists()


def test_two_stores_append_suffix_and_freeze_binding(tmp_path):
    first = sm.MessageStore(tmp_path / 'lcm.db')
    second = sm.MessageStore(tmp_path / 'lcm.sqlite3')
    alias = tmp_path / 'alias'
    alias.symlink_to(tmp_path / 'lcm.db')
    writer = sm.open_records_writer(first)
    try:
        assert sm.records_path(first) == tmp_path / 'lcm.db.records'
        assert sm.records_path(second) == tmp_path / 'lcm.sqlite3.records'
        assert sm.records_path(alias) == sm.records_path(first)
        bound = writer.path
        alias.unlink()
        alias.symlink_to(tmp_path / 'lcm.sqlite3')
        first.db_path = second.db_path  # profile configuration rebind
        assert sm.records_path(first) == bound
        assert writer.path == bound
        assert not sm.records_path(second).exists()
    finally:
        writer.close()
        first.close()
        second.close()


def test_memory_has_no_records(tmp_path):
    store = sm.MessageStore(':memory:')
    try:
        assert sm.records_path(store) is None
        with pytest.raises(ValueError, match='in-memory'):
            sm.open_records_writer(store)
        assert read(store) == ([], 0)
    finally:
        store.close()


def test_format_and_zero_busy_timeout(pair):
    _, writer, _ = pair
    assert writer.connection.execute('PRAGMA user_version').fetchone()[0] == 1
    assert writer.connection.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'
    assert writer.connection.execute('PRAGMA busy_timeout').fetchone()[0] == 0


def test_newer_format_is_refused_unchanged_and_doctor_reports(tmp_path):
    store = sm.MessageStore(tmp_path / 'lcm.db')
    path = sm.records_path(store)
    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.execute('PRAGMA user_version=2')
            conn.execute('CREATE TABLE future(value)')
            conn.commit()
        before = path.read_bytes()
        with pytest.raises(sm.RecordsFormatError, match='refused format 2'):
            sm.open_records_writer(store)
        assert read(store) == ([], 0)
        assert sm.records_doctor(store) == 'records: refused format 2'
        assert path.read_bytes() == before
    finally:
        store.close()


def test_records_busy_is_isolated_from_canonical(pair):
    store, writer, values = pair
    before = store.get_session_messages('S')
    with closing(sqlite3.connect(writer.path, timeout=0)) as other:
        other.execute('BEGIN IMMEDIATE')
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            writer.insert(TABLE, values)
        other.rollback()
    assert store.get_session_messages('S') == before
    assert not store.connection.in_transaction
    assert counts(writer) == (1, 0)


def test_canonical_write_lock_does_not_block_records(pair):
    store, writer, values = pair
    with closing(sqlite3.connect(store.db_identity, timeout=0)) as other:
        other.execute('BEGIN IMMEDIATE')
        assert writer.insert(TABLE, values) is None
        other.rollback()
    assert counts(writer) == (2, 0)


def test_attachment_uses_owner_current_scope(pair):
    store, writer, values = pair
    rows, unattached = read(store)
    assert unattached == 0 and rows[0]['payload'] == 'record'
    assert rows[0]['conversation_id'] == 'conv'
    store.reassign_session_messages('S', 'new-session')
    assert read(store)[0][0]['session_id'] == 'new-session'
    writer.maintain()
    assert counts(writer) == (1, 0)


def test_restored_copy_reused_owner_id_is_hidden_counted_and_retained(pair, tmp_path):
    store, writer, values = pair
    older = sm.MessageStore(tmp_path / 'older.db')
    try:
        reused = older.append('S', dict(role='user', content='different occurrence', timestamp=2.0))
        assert reused == values['owner_store_id']
        older.connection.backup(store.connection)  # restoring only lcm.db, beside newer records
    finally:
        older.close()
    assert read(store) == ([], 1)
    writer.maintain()
    assert read(store) == ([], 1)
    assert counts(writer) == (1, 0)


def test_read_cut_pins_records_before_canonical_and_never_writes(pair):
    store, writer, values = pair
    with sm.records_read_cut(store) as cut:
        assert cut.records.in_transaction and cut.canonical.in_transaction
        store.reassign_session_messages('S', 'later')
        assert sm.read_records(cut, TABLE)[0][0]['session_id'] == 'S'
        assert cut.canonical.execute('SELECT session_id FROM messages').fetchone()[0] == 'S'
        with pytest.raises(sqlite3.OperationalError):
            cut.records.execute(f'DELETE FROM {TABLE}')
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            writer.insert(TABLE, values)
    assert counts(writer) == (1, 0)


def test_uncommitted_owner_is_refused(pair):
    store, writer, values = pair
    with store.atomic(immediate=True):
        sid = store.append('S', dict(role='user', content='uncommitted', timestamp=2.0))
        row = store.connection.execute('SELECT ' + ','.join(COLUMNS.values()) + ' FROM messages WHERE store_id=?',
                                       (sid,)).fetchone()
        candidate = dict(zip(COLUMNS, row), payload='must not appear')
        assert writer.insert(TABLE, candidate) == 'owner not committed or evidence mismatch'
    assert counts(writer) == (1, 0)


def test_purge_commit_removes_records_and_intent(pair):
    store, writer, _ = pair
    assert store.delete_session_messages('S') == 1
    assert counts(writer) == (0, 0)
    assert store._transaction_owner.callback_failures == 0


def test_purge_rollback_keeps_records_and_clears_intent(pair):
    store, writer, _ = pair
    with pytest.raises(RuntimeError, match='rollback'):
        with store.atomic(immediate=True):
            store.delete_session_messages('S')
            raise RuntimeError('rollback')
    assert len(store.get_session_messages('S')) == 1
    assert counts(writer) == (1, 0)


def test_busy_intent_refuses_purge_with_canonical_unchanged(pair):
    store, writer, _ = pair
    with closing(sqlite3.connect(writer.path, timeout=0)) as other:
        other.execute('BEGIN IMMEDIATE')
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            store.delete_session_messages('S')
        other.rollback()
    assert len(store.get_session_messages('S')) == 1
    assert counts(writer) == (1, 0)


@pytest.mark.parametrize('rollback', [False, True])
def test_open_purge_second_connection_refuses_insert_and_maintenance_drops_nothing(pair, rollback):
    store, writer, values = pair
    with closing(sm.open_records_writer(store)) as second:
        try:
            with store.atomic(immediate=True):
                sm.run_session_purges(store.connection, 'SELECT ?', ('S',))
                second.maintain()
                assert counts(writer) == (1, 1), 'maintenance dropped an active purge intent'
                assert second.insert(TABLE, values) == 'pending purge intent'
                numeric_alias = dict(values, owner_store_id=float(values['owner_store_id']))
                assert second.insert(TABLE, numeric_alias) == 'pending purge intent'
                if rollback:
                    raise RuntimeError('rollback')
                store.connection.execute("DELETE FROM messages WHERE session_id='S'")
        except RuntimeError:
            assert rollback
    assert counts(writer) == ((1, 0) if rollback else (0, 0))


def test_crash_before_commit_leaves_intent_until_second_purge(pair):
    store, writer, values = pair
    with store.atomic(immediate=True):
        sm.run_session_purges(store.connection, 'SELECT ?', ('S',))
        store._transaction_owner.on_commit.clear()  # crash before canonical delete, no callbacks
        store._transaction_owner.on_rollback.clear()
    assert counts(writer) == (1, 1)
    assert writer.insert(TABLE, values) == 'pending purge intent'
    assert 'pending purge intents=1' in sm.records_doctor(store)
    writer.maintain()
    assert counts(writer) == (1, 1)
    store.delete_session_messages('S')
    assert counts(writer) == (0, 0)


@pytest.mark.parametrize('maintenance_point', ['insert', 'maintain'])
def test_crash_after_commit_is_completed_at_next_records_write_or_maintenance(pair, maintenance_point):
    store, writer, values = pair
    with store.atomic(immediate=True):
        store.delete_session_messages('S')
        store._transaction_owner.on_commit.clear()
    assert counts(writer) == (1, 1)
    if maintenance_point == 'maintain':
        writer.maintain()
    else:
        sid = store.append('T', dict(role='user', content='other', timestamp=3.0))
        row = store.connection.execute('SELECT ' + ','.join(COLUMNS.values()) + ' FROM messages WHERE store_id=?',
                                       (sid,)).fetchone()
        assert writer.insert(TABLE, dict(zip(COLUMNS, row), payload='new')) is None
    assert counts(writer) == ((1, 0) if maintenance_point == 'insert' else (0, 0))


def test_commit_callback_busy_keeps_intent_for_maintenance(pair):
    store, writer, _ = pair
    with closing(sqlite3.connect(writer.path, timeout=0)) as other:
        with store.atomic(immediate=True):
            store.delete_session_messages('S')
            other.execute('BEGIN IMMEDIATE')
        assert not store.get_session_messages('S')
        assert store._transaction_owner.callback_failures == 1
        other.rollback()
    assert counts(writer) == (1, 1)
    writer.maintain()
    assert counts(writer) == (0, 0)


@pytest.mark.parametrize('kind', ['direct', 'timestamped', 'rotate'])
def test_backup_copies_pair_and_failed_second_copy_keeps_previous_pair(pair, tmp_path, monkeypatch, kind):
    store, writer, values = pair
    root = tmp_path / 'backups'
    root.mkdir()
    engine = SimpleNamespace(_store=store, _dag=SimpleNamespace(_conn=store.connection),
                             backup_dir=lambda: root, rotate_backup_path=lambda: root / 'rotate.sqlite3')
    target = root / 'direct.sqlite3'
    def backup():
        if kind == 'direct':
            with closing(sqlite3.connect(target)) as dest:
                store.backup(dest)
            return dict(ok=True, backup_path=target)
        return (maintenance.backup_database if kind == 'timestamped' else maintenance.rotate_backup_database)(engine)
    result = backup()
    assert result['ok']
    target = result['backup_path']
    assert read(target) == read(store)
    previous = (target.read_bytes(), Path(str(target) + '.records').read_bytes())
    store.append('T', dict(role='user', content='after snapshot', timestamp=5.0))
    # Fail the canonical (second) SQLite copy, after the sidecar copy succeeded.
    original = sm.records_read_cut
    class BrokenCanonical:
        def backup(self, dest):
            raise sqlite3.OperationalError('injected second copy failure')
    from contextlib import contextmanager
    @contextmanager
    def broken_cut(db):
        with original(db) as cut:
            cut.canonical = BrokenCanonical()
            yield cut
    monkeypatch.setattr(sm, 'records_read_cut', broken_cut)
    if kind == 'direct':
        with pytest.raises(sqlite3.OperationalError, match='second copy'):
            backup()
    else:
        assert not backup()['ok']
    assert (target.read_bytes(), Path(str(target) + '.records').read_bytes()) == previous
    monkeypatch.setattr(sm, 'records_read_cut', original)
    restored = tmp_path / 'without-records.db'
    with closing(sqlite3.connect(target)) as source, closing(sqlite3.connect(restored)) as dest:
        source.backup(dest)
    assert read(restored) == ([], 0)
    assert read(target)[0][0]['payload'] == 'record'


def test_doctor_absent_present_and_message_count_stays_canonical(pair, tmp_path):
    store, writer, values = pair
    other = sm.MessageStore(tmp_path / 'absent.db')
    try:
        assert sm.records_doctor(other) == 'records: absent'
    finally:
        other.close()
    assert writer.insert(TABLE, dict(values, owner_content='wrong')) == 'owner not committed or evidence mismatch'
    detail = sm.records_doctor(store)
    assert 'format 1; bytes=' in detail
    assert 'test_records attached=1 unattached=0' in detail
    assert 'pending purge intents=0' in detail
    assert store.connection.execute('SELECT count(*) FROM messages').fetchone()[0] == 1


def test_container_creates_no_product_table_and_absent_table_reads_nothing(tmp_path):
    store = sm.MessageStore(tmp_path / 'lcm.db')
    try:
        with closing(sm.open_records_writer(store)) as writer:
            tables = {r[0] for r in writer.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert tables == {'_records_purge_intents'}
        with sm.records_read_cut(store) as cut:
            assert sm.read_records(cut, 'not_registered_or_present') == ([], 0)
    finally:
        store.close()


@pytest.mark.parametrize('version', [None, 1, 2])
def test_tool_doctor_has_one_records_line_and_counts_only_messages(tmp_path, version):
    import json
    from hermes_lcm.tools import lcm_doctor
    old_registry = dict(sm._RECORDS_TABLES)
    register()
    try:
        with engine_context(tmp_path, window=4000, embeddings_enabled=False,
                            background_leaf_prep_enabled=False) as engine:
            engine.ingest([dict(role='user', content='doctor canonical', timestamp=1.)])
            if version == 1:
                row = engine._store.connection.execute('SELECT ' + ','.join(COLUMNS.values()) + ' FROM messages').fetchone()
                with closing(sm.open_records_writer(engine._store)) as writer:
                    assert writer.insert(TABLE, dict(zip(COLUMNS, row), payload='separate record')) is None
            if version == 2:
                with closing(sqlite3.connect(sm.records_path(engine._store))) as conn:
                    conn.execute('PRAGMA user_version=2')
            report = json.loads(lcm_doctor({}, engine=engine))
            checks = [row for row in report['checks'] if row['check'] == 'records']
            assert len(checks) == 1
            expected = ('records: absent' if version is None else
                        'records: refused format 2' if version == 2 else 'records: format 1; bytes=')
            assert checks[0]['detail'].startswith(expected)
            fts = next(row['detail'] for row in report['checks'] if row['check'] == 'fts_index_sync')
            assert '1 session messages' in fts
    finally:
        sm._RECORDS_TABLES.clear()
        sm._RECORDS_TABLES.update(old_registry)


def test_newer_format_with_uncheckpointed_wal_is_refused_without_checkpoint(tmp_path):
    import subprocess
    import sys
    store = sm.MessageStore(tmp_path / 'lcm.db')
    path = sm.records_path(store)
    create_future = '''
import os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA wal_autocheckpoint=0')
conn.execute('PRAGMA user_version=2')
conn.execute('CREATE TABLE future(value)')
conn.execute("INSERT INTO future VALUES ('future data')")
conn.commit()
os._exit(0)  # leave committed format-2 pages in WAL, without a close-time checkpoint
'''
    try:
        subprocess.run([sys.executable, '-B', '-c', create_future, str(path)], check=True)
        before = path.read_bytes()
        wal = Path(str(path) + '-wal')
        before_wal = wal.read_bytes()
        with pytest.raises(sm.RecordsFormatError):
            sm.open_records_writer(store)
        assert path.read_bytes() == before, 'refusing a newer format must not checkpoint its WAL'
        assert wal.read_bytes() == before_wal
        assert sm.records_doctor(store) == 'records: refused format 2'
        assert path.read_bytes() == before
    finally:
        store.close()


def test_completion_refuses_a_newer_format_and_keeps_its_bytes(pair):
    store, writer, _ = pair
    with store.atomic(immediate=True):
        store.delete_session_messages('S')
        writer.connection.execute('PRAGMA user_version=2')
        writer.connection.commit()
        before = writer.path.read_bytes()
    assert not store.get_session_messages('S')
    assert store._transaction_owner.callback_failures == 1
    assert writer.path.read_bytes() == before
    assert counts(writer) == (1, 1)
    assert sm.records_doctor(store) == 'records: refused format 2'


def test_error_after_commit_does_not_misclassify_the_intent_as_rolled_back(pair, monkeypatch):
    store, writer, _ = pair
    real = store.connection

    class ErrorAfterCommit:
        def __getattr__(self, name):
            return getattr(real, name)

        def commit(self):
            real.commit()
            raise RuntimeError('error after durable commit')

    monkeypatch.setattr(store, '_conn', ErrorAfterCommit())
    with pytest.raises(RuntimeError, match='error after durable commit'):
        store.delete_session_messages('S')
    assert not store.get_session_messages('S')
    assert counts(writer) == (1, 1), 'a finished purge must keep its intent until records are deleted'
    writer.maintain()
    assert counts(writer) == (0, 0)
