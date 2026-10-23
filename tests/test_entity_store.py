"""Lane S: gated source schema, lifecycle and atomic scoped primitives."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib
import re
import sqlite3
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from hermes_lcm import db_bootstrap as db
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


@pytest.fixture
def api():
    # Kept lazy so the base records individual red acceptance probes.
    return importlib.import_module("hermes_lcm.entity_store")


@pytest.fixture
def store(tmp_path, api):
    with closing(MessageStore(tmp_path / "lcm.db")), closing(SummaryDAG(tmp_path / "lcm.db")), closing(
            LifecycleStateStore(tmp_path / "lcm.db")), closing(api.EntityStore(
            tmp_path / "lcm.db", profile_key="profile-A")) as instance:
        yield instance


def rows(conn, sql, values=()):
    return [tuple(r) for r in conn.execute(sql, values)]


def test_derived_key_writers_transaction_scope_unique_and_no_log(store, api):
    entity = store.create_entity('key owner')
    before = store.events_after()
    for call in (lambda: store.add_key('person:alex', entity, 'alias_match', [1]),
                 lambda: store.retire_key(1), lambda: store.restore_key(1)):
        with pytest.raises(RuntimeError, match='transaction'):
            call()
    with store.transaction():
        key = store.add_key('person:alex', entity, 'alias_match', [1])
        assert store.keys_for_entity(entity)[0]['scope_key'] == store.scope_key
        with pytest.raises(sqlite3.IntegrityError, match='UNIQUE'):
            store.add_key('person:alex', entity, 'override', [1])
        store.retire_key(key)
        second = store.add_key('person:alex', entity, 'override', [1])
        with pytest.raises(sqlite3.IntegrityError, match='UNIQUE'):
            store.restore_key(key)
        store.retire_key(second)
        store.restore_key(key)
    assert store.events_after() == before
    with closing(api.EntityStore(store.db_path, profile_key='profile-B')) as other:
        with other.transaction():
            for call in (lambda: other.retire_key(key), lambda: other.restore_key(key)):
                with pytest.raises(KeyError):
                    call()
            other.add_key('person:alex', entity, 'alias_match', [1])
        assert other.keys_for_entity(entity)[0]['scope_key'] == 'profile-B'
    assert [r['key_id'] for r in store.keys_for_entity(entity)] == [key]


def test_flag_off(tmp_path, monkeypatch):
    trace, connect = [], sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(trace.append)
        return conn

    monkeypatch.setattr(sqlite3, 'connect', traced_connect)
    monkeypatch.delenv("LCM_ENTITIES_ENABLED", raising=False)
    assert not LCMConfig.from_env().entities_enabled
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
                       hermes_home=str(tmp_path / "home"))
    opening_trace = list(trace)
    try:
        assert not any(token in sql.lower() for sql in opening_trace
                       for token in ('lcm_entit', 'lcm_memory_', 'entity_memory_v1'))
        flag_off_steps = {r[0] for r in engine._store.connection.execute(
            'SELECT step_name FROM lcm_migration_state')}
        assert db.ENTITY_MIGRATION_STEP not in flag_off_steps
        assert engine._entities is None
        assert not rows(engine._store.connection, "SELECT name FROM sqlite_master WHERE name LIKE 'lcm_entit%' OR name LIKE 'lcm_memory_%'")
        assert engine._store.connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == '5'
    finally:
        engine.shutdown()
    enabled = LCMEngine(config=LCMConfig(database_path=str(tmp_path / 'enabled.db'),
                                        entities_enabled=True),
                        hermes_home=str(tmp_path / 'enabled-home'))
    try:
        flag_on_steps = {r[0] for r in enabled._store.connection.execute(
            'SELECT step_name FROM lcm_migration_state')}
        assert flag_on_steps == flag_off_steps | {db.ENTITY_MIGRATION_STEP}
    finally:
        enabled.shutdown()


def test_lazy_marker_and_fast_path(store, api, monkeypatch):
    conn, path = store.connection, store.db_path
    cookie = conn.execute("PRAGMA schema_version").fetchone()[0]
    # SQLite stamps markers with strftime, so use a sentinel rather than a clock mock.
    conn.execute("UPDATE lcm_migration_state SET completed_at=1 WHERE step_name=?",
                 (db.ENTITY_MIGRATION_STEP,))
    trace, original = [], sqlite3.connect

    def connect(*a, **kw):
        result = original(*a, **kw)
        result.set_trace_callback(trace.append)
        return result

    monkeypatch.setattr(sqlite3, "connect", connect)
    with closing(api.EntityStore(path, profile_key="profile-A")) as other:
        assert db.verify_entity_schema(other.connection) == []
        assert other.connection.execute("PRAGMA schema_version").fetchone()[0] == cookie
        assert other.connection.execute("SELECT count(*) FROM lcm_migration_state WHERE step_name='entity_memory_v1'").fetchone()[0] == 1
        assert other.connection.execute("SELECT completed_at FROM lcm_migration_state WHERE step_name=?",
                                        (db.ENTITY_MIGRATION_STEP,)).fetchone()[0] == 1
    assert not any('lcm_migration_state' in sql.lower() and
                   re.search(r'\b(?:INSERT|UPDATE|REPLACE|DELETE)\b', sql, re.IGNORECASE)
                   for sql in trace)
    assert not any('BEGIN IMMEDIATE' in s or s.lstrip().upper().startswith(('CREATE', 'ALTER', 'DROP'))
                   and any(name in s for name in ('lcm_entit', 'lcm_memory_')) for s in trace)


def test_read_only_never_mutates(tmp_path, api):
    path = tmp_path / 'legacy.db'
    with closing(MessageStore(path)) as messages:
        before = rows(messages.connection, "SELECT type,name,sql FROM sqlite_master ORDER BY type,name")
    with pytest.raises(api.EntityStoreUnavailableError):
        api.EntityStore(path, profile_key='profile-A', read_only=True)
    with closing(sqlite3.connect(path)) as conn:
        assert rows(conn, "SELECT type,name,sql FROM sqlite_master ORDER BY type,name") == before
    with closing(api.EntityStore(path, profile_key='profile-A')) as instance:
        instance.connection.execute('ALTER TABLE lcm_memory_notes ADD COLUMN unknown TEXT')
        malformed = rows(instance.connection, "SELECT type,name,sql FROM sqlite_master ORDER BY type,name")
    with pytest.raises(api.EntityStoreUnavailableError):
        api.EntityStore(path, profile_key='profile-A', read_only=True)
    with closing(sqlite3.connect(path)) as conn:
        assert rows(conn, "SELECT type,name,sql FROM sqlite_master ORDER BY type,name") == malformed


@pytest.mark.parametrize('mutation', [
    "ALTER TABLE lcm_memory_notes ADD COLUMN extra TEXT",
    "CREATE INDEX idx_lcm_entit_x ON lcm_entities(display_name)",
    "ALTER TABLE lcm_entity_aliases RENAME COLUMN alias_kind TO renamed_kind",
    "DROP INDEX idx_lcm_memory_host_file_rev_content; CREATE INDEX idx_lcm_memory_host_file_rev_content ON lcm_memory_host_file_revisions(path)",
])
def test_strict_verifier(store, mutation):
    for sql in mutation.split(';'):
        store.connection.execute(sql)
    assert db.verify_entity_schema(store.connection)
    store.connection.execute("UPDATE metadata SET value='6' WHERE key='schema_version'")
    assert db.classify_version_mismatch(store.connection) == 'genuinely_newer'


def test_dropped_check_is_rejected(store):
    conn = store.connection
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='lcm_entities'").fetchone()[0]
    conn.execute('PRAGMA writable_schema=ON')
    conn.execute("UPDATE sqlite_master SET sql=? WHERE name='lcm_entities'",
                 (sql.replace("CHECK(status IN ('active','merged','forgotten'))", ''),))
    conn.execute('PRAGMA writable_schema=OFF')
    assert db.verify_entity_schema(conn)


def test_preserved_classifier_and_assertion_remediation(store):
    conn = store.connection
    entity = store.create_entity('kept')
    db.ensure_assertion_tables(conn)
    conn.execute('DROP TABLE lcm_assertion_relations')
    conn.execute("UPDATE metadata SET value='6' WHERE key='schema_version'")
    assert db.classify_version_mismatch(conn) == 'interim_stamp'
    drops = db._interim_family_drops(conn)
    assert not any('lcm_entit' in str(x) or 'lcm_memory_' in str(x) for x in drops)
    result = db.remediate_interim_schema_stamp(conn, apply=True)
    assert result
    assert db.verify_entity_schema(conn) == []
    assert store.get_entity(entity)['display_name'] == 'kept'


def test_concurrent_first_engine_open(tmp_path):
    barrier = threading.Barrier(2)
    config = LCMConfig(database_path=str(tmp_path / 'lcm.db'), entities_enabled=True)

    def open_it(_):
        barrier.wait(timeout=10)
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / 'home'))
        try:
            assert db.verify_entity_schema(engine._entities.connection) == []
        finally:
            engine.shutdown()

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(open_it, range(2)))
    with closing(sqlite3.connect(tmp_path / 'lcm.db')) as conn:
        assert conn.execute("SELECT count(*) FROM lcm_migration_state WHERE step_name='entity_memory_v1'").fetchone()[0] == 1


def test_fifty_rebinds_close_connections(tmp_path):
    engine = LCMEngine(config=LCMConfig(entities_enabled=True), hermes_home=str(tmp_path / 'home-0'))
    connections = []
    try:
        for i in range(50):
            connections.append(engine._entities.connection)
            assert engine._rebind_storage_for_home(str(tmp_path / f'home-{i+1}'))
            for conn in connections:
                with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                    conn.execute('SELECT 1')
            assert engine._entities.connection.execute('SELECT 1').fetchone()[0] == 1
    finally:
        last = engine._entities.connection
        engine.shutdown()
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        last.execute('SELECT 1')


@pytest.mark.parametrize('resolution', ['names', 'missing', 'none'])
def test_shared_database_profile_isolation(tmp_path, monkeypatch, resolution):
    host = SimpleNamespace() if resolution == 'missing' else SimpleNamespace(
        profile_name_for_home=lambda home: Path(home).name if resolution == 'names' else None)
    monkeypatch.setitem(sys.modules, 'hermes_constants', host)
    config = LCMConfig(database_path=str(tmp_path / 'lcm.db'), entities_enabled=True)
    engines = [LCMEngine(config=config, hermes_home=str(tmp_path / p)) for p in ('A', 'B')]
    try:
        stores = [engine._entities for engine in engines]
        expected = ['A', 'B'] if resolution == 'names' else [str((tmp_path / p).resolve()) for p in ('A', 'B')]
        assert [s.scope_key for s in stores] == expected
        assert stores[0].scope_key != stores[1].scope_key
        ids = [s.create_entity('same platform person') for s in stores]
        for s, key in zip(stores, ids):
            s.add_alias(key, 'Telegram', 'platform_id', alias_norm='telegram:AbC')
            with pytest.raises(sqlite3.IntegrityError):
                s.add_alias(key, 'duplicate', 'platform_id', alias_norm='telegram:AbC')
        for i, s in enumerate(stores):
            assert s.get_entity(ids[1-i]) is None
            assert len(s.aliases_by_norm('telegram:AbC')) == 1
            assert len(s.events_after()) == 2
        assert engines[0]._rebind_storage_for_home(str(tmp_path / 'B'))
        assert engines[0]._entities.get_entity(ids[0]) is None
        assert engines[0]._entities.get_entity(ids[1])
    finally:
        for e in engines:
            e.shutdown()


def test_ids_clock_and_normalisation(store, api):
    entity = store.create_entity('Alex', id_factory=lambda: 'a'*32, now=lambda: 7.0)
    assert entity == 'ent_'+'a'*32
    assert store.get_entity(entity)['created_at'] == 7.0
    with store.transaction():
        keys = {store.create_entity('unique', log_event=False) for _ in range(10000)}
    assert len(keys) == 10000 and all(re.fullmatch(r'ent_[0-9a-f]{32}', k) for k in keys)
    assert api.normalize_alias('Straße') == api.normalize_alias('STRASSE')
    assert api.normalize_alias('Ａｌｅｘ') == api.normalize_alias('Alex')
    assert api.normalize_alias('“  Alex,  Smith! ”') == 'alex, smith'
    assert api.normalize_platform_id('Telegram', 'AbC') == 'telegram:AbC'


def test_atomicity_cites_and_append_only(store, monkeypatch):
    before = store.events_after()
    with pytest.raises(sqlite3.IntegrityError):
        store.add_note('uncited')
    with monkeypatch.context() as patch:
        def fail(*a, **kw):
            raise RuntimeError('after insert before event')
        patch.setattr(store, 'append_event', fail)
        with pytest.raises(RuntimeError, match='after insert'):
            store.add_note('rolled back', cites=[1])
    assert rows(store.connection, 'SELECT note_id FROM lcm_memory_notes') == []
    assert store.events_after() == before
    store.append_event('create_entity', {}, now=lambda: 1.0)
    for sql in ('UPDATE lcm_memory_events SET actor=actor', 'DELETE FROM lcm_memory_events'):
        with pytest.raises(sqlite3.IntegrityError, match='lcm_memory_events is append-only'):
            store.connection.execute(sql)


def test_all_primitives_log_and_compose(store):
    with pytest.raises(RuntimeError, match='transaction'):
        store.create_entity('denied', log_event=False)
    with store.transaction():
        entity = store.create_entity('composite', log_event=False)
        assert store.events_after() == []
        store.append_event('create_entity', {'entity_id': entity}, now=lambda: 1.0)
    alias = store.add_alias(entity, 'Alex')
    store.retire_alias(alias)
    store.restore_alias(alias)
    store.set_entity_status(entity, 'forgotten', None)
    note = store.add_note('fact', cites=[1])
    store.set_note_superseded(note, None)
    revision = store.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash')
    assert re.fullmatch(r'note_[0-9a-f]{32}', note) and re.fullmatch(r'rev_[0-9a-f]{32}', revision)
    before = store.events_after()
    assert store.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash') == revision
    assert store.events_after() == before
    assert [r['op'] for r in before] == ['create_entity', 'add_alias', 'retire_alias', 'restore_alias',
                                        'set_entity_status', 'add_note', 'supersede_note', 'host_file_revision']
    with store.transaction():
        store.retire_alias(alias, log_event=False)
        store.restore_alias(alias, log_event=False)
        store.set_entity_status(entity, 'active', None, log_event=False)
        store.set_note_superseded(note, note, log_event=False)
        store.add_note('composite fact', cites=[1], log_event=False)
        store.add_alias(entity, 'silent', log_event=False)
        store.insert_host_file_revision('/fixture/USER.md', 'sha2', 'hash2', log_event=False)
    assert store.events_after() == before


def test_equal_timestamps_and_undo_order(store):
    first = store.append_event('split', {'touches': ['E1', 'E2']}, now=lambda: 1.0)
    second = store.append_event('split', {'touches': ['E2', 'E3']}, now=lambda: 1.0)
    store.append_event('undo', {}, target_event_id=second, now=lambda: 0.0)
    store.append_event('undo', {}, target_event_id=first, now=lambda: 0.0)
    events = store.events_after()
    assert [e['event_id'] for e in events] == sorted(e['event_id'] for e in events)
    assert [e['target_event_id'] for e in events[2:]] == [second, first]
    assert store.events_after(first) == events[1:]
    # W1 consumes this deterministic order; this lane implements no undo logic.


def test_visibility_and_one_statement_per_read(tmp_path, api):
    path = tmp_path / 'lcm.db'
    with closing(MessageStore(path)) as messages:
        messages.connection.execute('ALTER TABLE messages ADD COLUMN access_scope TEXT')
        for i, scope in ((1, 'A'), (2, 'B'), (3, None)):
            messages.connection.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp,access_scope) VALUES(?,'S','user','fixture',1,?)", (i, scope))
        messages.connection.commit()
        with closing(api.EntityStore(path, profile_key='profile-A')) as s:
            entity = s.create_entity('visible')
            mixed = s.add_note('mixed', entity_id=entity, cites=[1, 2])
            legacy = s.add_note('legacy', entity_id=entity, cites=[3])
            deleted = s.add_note('deleted', entity_id=entity, cites=[99])
            revision = s.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash')
            file_note = s.add_note('owner file', entity_id=entity, cites_file_revisions=[revision])
            alias = s.add_alias(entity, 'owner alias')
            assert s.get_note(mixed, access_scope='A') is None
            assert s.get_note(legacy, access_scope='A')
            assert s.get_note(deleted) is None and s.get_note(deleted, access_scope='A') is None
            for owner in (False, True):
                access = dict(access_scope='A', principal_is_owner=owner)
                assert bool(s.get_note(file_note, **access)) == owner
                assert bool(s.get_alias(alias, **access)) == owner
            assert s.get_note(file_note) and s.get_alias(alias)
            with closing(api.EntityStore(path, profile_key='profile-B')) as other:
                assert other.get_note(file_note, access_scope='A', principal_is_owner=True) is None
                assert other.get_host_file_revision(revision) is None
            messages.connection.execute('DELETE FROM messages WHERE store_id=3')
            messages.connection.commit()
            assert s.get_note(legacy) is None
            calls = [lambda: s.get_entity(entity), lambda: s.get_note(file_note), lambda: s.get_alias(alias),
                     lambda: s.aliases_by_norm('owner alias'), lambda: s.aliases_for_entity(entity),
                     lambda: s.notes_for_entity(entity), lambda: s.keys_for_entity(entity),
                     lambda: s.events_after(), lambda: s.get_host_file_revision(revision)]
            for call in calls:
                trace = []
                s.connection.set_trace_callback(trace.append)
                call()
                s.connection.set_trace_callback(None)
                assert len(trace) == 1, trace


@pytest.mark.parametrize('access', [
    {'access_scope': 'A'}, {'access_scope': None},
    {'principal_is_owner': True}, {'principal_is_owner': False},
])
def test_events_are_trusted_replay_only(store, access):
    store.create_entity('private replay payload')
    assert store.events_after()
    with pytest.raises(ValueError, match='trusted replay only'):
        store.events_after(**access)


@pytest.mark.parametrize('owner', [None, 0, 1, 'owner', []])
def test_principal_owner_requires_bool(store, owner):
    revision = store.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash')
    note = store.add_note('private file fact', cites_file_revisions=[revision])
    with pytest.raises(TypeError, match='principal_is_owner'):
        store.get_note(note, access_scope='A', principal_is_owner=owner)
    # A correctly typed non-owner still cannot read the file-derived fact.
    assert store.get_note(note, access_scope='A', principal_is_owner=False) is None


@pytest.mark.parametrize('owner', [False, True])
def test_empty_principal_denies_all_guarded_reads(store, owner):
    store.connection.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES(1,'S','user','fixture',1)")
    entity = store.create_entity('private entity')
    alias = store.add_alias(entity, 'cited alias', source_store_ids=[1])
    store.add_alias(entity, 'owner alias')
    revision = store.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash')
    note = store.add_note('private fact', entity_id=entity, cites=[1], cites_file_revisions=[revision])
    with store.transaction():
        store.connection.execute("INSERT INTO lcm_entity_keys(subject_key,entity_id,method,source_store_ids,scope_key,created_at) VALUES('private',?,'alias_match','[1]',?,1)", (entity, store.scope_key))
    access = dict(access_scope='', principal_is_owner=owner)
    assert store.get_entity(entity, **access) is None
    assert store.get_alias(alias, **access) is None
    assert store.get_note(note, **access) is None
    assert store.get_host_file_revision(revision, **access) is None
    assert store.aliases_by_norm('cited alias', **access) == []
    assert store.aliases_for_entity(entity, **access) == []
    assert store.notes_for_entity(entity, **access) == []
    assert store.keys_for_entity(entity, **access) == []


def test_note_cannot_cite_another_profiles_revision(store, api):
    with closing(api.EntityStore(store.db_path, profile_key='profile-Q')) as other:
        revision = other.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash')
    note = store.add_note('foreign revision', cites_file_revisions=[revision])
    assert store.get_note(note, access_scope='A', principal_is_owner=True) is None
    assert store.get_note(note) is None


def test_revision_read_requires_owner(store):
    revision = store.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash')
    assert store.get_host_file_revision(revision)
    assert store.get_host_file_revision(revision, access_scope='A', principal_is_owner=True)
    assert store.get_host_file_revision(revision, access_scope='A') is None


def test_entity_requires_visible_evidence(store):
    store.connection.execute('ALTER TABLE messages ADD COLUMN access_scope TEXT')
    store._stamp = 'm.access_scope'
    store.connection.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp,access_scope) VALUES(1,'S','user','fixture',1,'B')")
    entity = store.create_entity('foreign evidence')
    store.add_alias(entity, 'foreign alias', source_store_ids=[1])
    store.add_note('foreign fact', entity_id=entity, cites=[1])
    assert store.get_entity(entity)
    assert store.get_entity(entity, access_scope='A') is None
    assert store.get_entity(entity, access_scope='B')


@pytest.mark.parametrize('evidence', ['alias', 'note'])
def test_other_profiles_evidence_cannot_expose_entity(store, api, evidence):
    store.connection.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES(1,'S','user','fixture',1)")
    entity = store.create_entity('no local evidence')
    with closing(api.EntityStore(store.db_path, profile_key='profile-Q')) as other:
        if evidence == 'alias':
            key = other.add_alias(entity, 'foreign alias', source_store_ids=[1])
            assert other.get_alias(key, access_scope='A')
        else:
            key = other.add_note('foreign note', entity_id=entity, cites=[1])
            assert other.get_note(key, access_scope='A')
    assert store.get_entity(entity)
    assert store.aliases_for_entity(entity) == []
    assert store.notes_for_entity(entity) == []
    assert store.get_entity(entity, access_scope='A') is None


def test_entity_visible_with_only_note_evidence(store):
    store.connection.execute("INSERT INTO messages(store_id,session_id,role,content,timestamp) VALUES(1,'S','user','fixture',1)")
    entity = store.create_entity('note evidence only')
    note = store.add_note('visible fact', entity_id=entity, cites=[1])
    assert store.aliases_for_entity(entity) == []
    assert store.get_note(note, access_scope='A')
    assert store.get_entity(entity, access_scope='A') == store.get_entity(entity)


def test_empty_key_sources_hidden_even_from_owner(store):
    entity = store.create_entity('owner alias is not key evidence')
    store.add_alias(entity, 'owner alias')
    with store.transaction():
        store.connection.execute("INSERT INTO lcm_entity_keys(subject_key,entity_id,method,source_store_ids,scope_key,created_at) VALUES('empty',?,'alias_match','[]',?,1)", (entity, store.scope_key))
    assert store.keys_for_entity(entity)
    assert store.aliases_for_entity(entity, access_scope='A', principal_is_owner=True)
    assert store.keys_for_entity(entity, access_scope='A', principal_is_owner=True) == []


@pytest.mark.parametrize('relocate', ['move', 'copy'])
def test_profile_name_survives_home_relocation(tmp_path, monkeypatch, relocate):
    monkeypatch.setitem(sys.modules, 'hermes_constants', SimpleNamespace(
        profile_name_for_home=lambda home: Path(home).name))
    home, moved = tmp_path / 'old' / 'alpha', tmp_path / 'new' / 'alpha'
    engine = LCMEngine(config=LCMConfig(entities_enabled=True), hermes_home=str(home))
    try:
        entity = engine._entities.create_entity('same named profile')
        revision = engine._entities.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash')
    finally:
        engine.shutdown()
    moved.parent.mkdir()
    if relocate == 'move':
        shutil.move(str(home), str(moved))
    else:
        shutil.copytree(home, moved)
    engine = LCMEngine(config=LCMConfig(entities_enabled=True), hermes_home=str(moved))
    try:
        assert engine._entities.get_entity(entity)
        assert engine._entities.get_host_file_revision(revision)['profile_key'] == 'alpha'
    finally:
        engine.shutdown()


@pytest.mark.parametrize('resolution', ['empty', 'raises', 'missing', 'none'])
def test_profile_scope_falls_back_to_resolved_home(tmp_path, monkeypatch, resolution):
    def resolve(home):
        if resolution == 'raises':
            raise RuntimeError('host helper unavailable')
        return None if resolution == 'none' else ''

    host = SimpleNamespace() if resolution == 'missing' else SimpleNamespace(profile_name_for_home=resolve)
    monkeypatch.setitem(sys.modules, 'hermes_constants', host)
    home = tmp_path / 'profile'
    engine = LCMEngine(config=LCMConfig(entities_enabled=True), hermes_home=str(home))
    try:
        assert engine._entities.scope_key == str(home.resolve())
    finally:
        engine.shutdown()
