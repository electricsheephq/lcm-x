"""Lane S forward and rollback proof against explicitly pinned release writers."""

from contextlib import closing
import hashlib
import importlib
import json
import sqlite3

import pytest

from hermes_lcm import db_bootstrap as db
from hermes_lcm.engine import LCMEngine
from hermes_lcm.schema_families import COMPAT_REGISTRY
from tests.compat.helpers import PREVIOUS_GA, READERS, config_for, run_old

ENTITY_READERS = ('v0.26.2', 'v0.28.0', 'v0.28.1', 'v0.25.0-rc1')
assert READERS['v0.28.1'] == '77b2b4f795fb6a4f84c55abf6d0441da4469971e'
OLD_WRITE = r'''
options.update(fresh_tail_count=2, leaf_chunk_tokens=100)
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    engine.on_session_start(mode, platform='cli', context_length=200000, conversation_id=mode)
    view = []
    for i in range(12):
        view.extend([{'role':'user','content':f'Question {i} ' + 'detail '*200, 'timestamp':100.+i*2},
                     {'role':'assistant','content':f'Answer {i} ' + 'reply '*200, 'timestamp':101.+i*2}])
    engine.ingest(view)
    engine.compress(view, current_tokens=100000, force=True)
    assert engine._dag.get_session_nodes(mode), 'compression must publish a summary'
    assert engine._store.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    print(json.dumps({'ingested':len(engine._store.get_session_messages(mode)), 'compressed':True}))
finally:
    engine.shutdown()
'''


def digest(rows):
    return hashlib.sha256(json.dumps(rows, default=str, sort_keys=True).encode()).hexdigest()


def core_snapshot(path, prior=None):
    """Hash every pre-existing source/DAG row, allowing the old writer new rows."""
    with closing(sqlite3.connect(path)) as conn:
        result = {}
        for table in ('messages', 'summary_nodes', 'lcm_lifecycle_state'):
            columns = prior[table][0] if prior else [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]
            select = ','.join('"'+c+'"' for c in columns)
            records = conn.execute(f'SELECT rowid,{select} FROM {table} ORDER BY rowid').fetchall()
            if prior:
                ids = set(prior[table][1])
                records = [r for r in records if r[0] in ids]
            result[table] = (columns, [r[0] for r in records], digest(records))
        return result


def family_snapshot(path):
    with closing(sqlite3.connect(path)) as conn:
        return {table: digest(conn.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall())
                for table in db._ENTITY_TABLE_COLUMNS}


@pytest.mark.parametrize('reader', ENTITY_READERS)
def test_entity_forward(reader, tmp_path, old_tree, host_root, monkeypatch):
    assert PREVIOUS_GA == 'v0.28.1'
    assert run_old(old_tree(reader), tmp_path, host_root, OLD_WRITE, mode='write')['compressed']
    path = tmp_path / 'lcm.db'
    before = core_snapshot(path)
    monkeypatch.setattr(db, 'dispatch_background_stem_backfill', lambda *a: None)
    cookie = None
    for opening in range(2):
        engine = LCMEngine(config=config_for(tmp_path, entities_enabled=True), hermes_home=str(tmp_path / 'home'))
        try:
            assert db.verify_entity_schema(engine._entities.connection) == []
            current = engine._entities.connection.execute('PRAGMA schema_version').fetchone()[0]
            if opening:
                assert current == cookie
                trace = []
                engine._entities.connection.set_trace_callback(trace.append)
                db.ensure_entity_tables(engine._entities.connection)
                assert not any('BEGIN IMMEDIATE' in sql or sql.lstrip().startswith('CREATE') for sql in trace)
            cookie = current
        finally:
            engine.shutdown()
        assert core_snapshot(path, before) == before


@pytest.mark.parametrize('reader', ENTITY_READERS)
def test_entity_rollback(reader, tmp_path, old_tree, host_root):
    api = importlib.import_module('hermes_lcm.entity_store')
    # Seed real ingest/compress data before creating all six feature tables.
    assert run_old(old_tree('v0.28.1'), tmp_path, host_root, OLD_WRITE, mode='write')['compressed']
    engine = LCMEngine(config=config_for(tmp_path, entities_enabled=True), hermes_home=str(tmp_path / 'home'))
    try:
        s = engine._entities
        entity = s.create_entity('synthetic')
        s.add_alias(entity, 'synthetic', source_store_ids=[1])
        rev = s.insert_host_file_revision('/fixture/MEMORY.md', 'sha', 'hash')
        s.add_note('synthetic cited fact', entity_id=entity, cites=[1], cites_file_revisions=[rev])
        with s.transaction():
            s.connection.execute("INSERT INTO lcm_entity_keys(subject_key,entity_id,method,source_store_ids,scope_key,created_at) VALUES('person:synthetic',?,'alias_match','[1]',?,1)", (entity, s.scope_key))
        assert all(s.connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in db._ENTITY_TABLE_COLUMNS)
        assert isinstance(s, api.EntityStore)
    finally:
        engine.shutdown()
    before, family = core_snapshot(tmp_path / 'lcm.db'), family_snapshot(tmp_path / 'lcm.db')
    # Pre-D11 readers reopen/ingest/compress; forget is not purge and those
    # readers would expose forgotten source text again. No hiding claim here.
    result = run_old(old_tree(reader), tmp_path, host_root, OLD_WRITE, mode='rollback')
    assert result['ingested'] == 24 and result['compressed']
    assert family_snapshot(tmp_path / 'lcm.db') == family
    assert core_snapshot(tmp_path / 'lcm.db', before) == before
    engine = LCMEngine(config=config_for(tmp_path, entities_enabled=True), hermes_home=str(tmp_path / 'home'))
    try:
        assert db.verify_entity_schema(engine._entities.connection) == []
        assert engine._entities.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    finally:
        engine.shutdown()


def test_entity_registry():
    for name, prefix in (('entity', 'lcm_entit'), ('memory', 'lcm_memory_')):
        family = COMPAT_REGISTRY[name]
        assert family.prefix == prefix and family.marker == 'entity_memory_v1'
        assert family.oldest_reader == 'v0.25.0-rc1' and family.removal_version is None
        assert family.tables == tuple(table for table in db._ENTITY_TABLE_COLUMNS if table.startswith(prefix))
        assert family.indexes == tuple(index for index in db._ENTITY_INDEX_SHAPES if index.startswith('idx_' + prefix))
        assert prefix in db._PRESERVED_FEATURE_FAMILIES
        assert not any(f['prefix'] == prefix for f in db._INTERIM_FEATURE_FAMILIES)
        assert db._family_verifier(prefix) is db.verify_entity_schema

    for test in (test_entity_forward, test_entity_rollback):
        assert any(mark.name == 'parametrize' and mark.args[0] == 'reader'
                   and set(mark.args[1]) == set(ENTITY_READERS) for mark in test.pytestmark)
