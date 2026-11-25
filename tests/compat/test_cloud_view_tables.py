"""View-map coverage must include enabled feature tables, not just capture core."""

from contextlib import closing
import json
import sqlite3

import pytest

from hermes_lcm import claude_code_store as cc, cloud_view
from hermes_lcm.config import LCMConfig
from hermes_lcm.entity_store import EntityStore
from tests.test_claude_code_store import _entry, _transcript


HIDDEN = '0b1c2d3e-1111-4222-8333-444455556666'
VISIBLE = '3f2a1b0c-5555-4666-8777-888899990000'
PLANTED = 'synthetic_private_ochre'


def capture_store(tmp_path):
    home = tmp_path / 'capture'
    database = cc.admit(home, hermes_home=tmp_path / 'unused-host')
    config = LCMConfig(database_path=str(database), embeddings_enabled=False,
                       large_output_externalization_enabled=True,
                       large_output_externalization_threshold_chars=200)
    ids = {}
    with closing(cc.open_store(database, home, config)) as store:
        for session, text in ((HIDDEN, PLANTED), (VISIBLE, 'visible control')):
            transcript = _transcript(tmp_path, [_entry('user', session, text, session)], session)
            assert cc.drain(store, cc.Lineage(session, None, transcript)) == 'drained'
            ids[session] = store.connection.execute(
                'SELECT store_id FROM messages WHERE session_id=?', (session,)).fetchone()[0]
        assert cc.to_rollback_journal(store.connection)
    assert cc.leave_readable(database)
    assert database.stat().st_size < 5_000_000
    return database, ids


def assert_mapped(database):
    with closing(sqlite3.connect(database)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    unmapped = sorted(table for table in tables
                      if table not in cloud_view.TABLES
                      and not (table.startswith('sqlite_') and cloud_view.TABLES.get('sqlite_*') == 'global'))
    assert not unmapped, f'view class map lacks tables: {unmapped}'


def test_a3_table_map_capture(tmp_path):
    database, _ = capture_store(tmp_path)
    assert_mapped(database)


def test_a3_table_map_entity_memory(tmp_path):
    database, ids = capture_store(tmp_path)
    with closing(EntityStore(database, profile_key='synthetic-profile')) as entities:
        entity = entities.create_entity('synthetic entity')
        entities.add_alias(entity, PLANTED, source_store_ids=[ids[HIDDEN]])
        entities.add_note(PLANTED, entity_id=entity, cites=[ids[HIDDEN]])
        entities.add_note(PLANTED + ' and visible control', entity_id=entity,
                          cites=[ids[HIDDEN], ids[VISIBLE]])
        notes = entities.connection.execute('SELECT text,cites FROM lcm_memory_notes ORDER BY rowid').fetchall()
        assert [(row[0], json.loads(row[1])) for row in notes] == [
            (PLANTED, [ids[HIDDEN]]),
            (PLANTED + ' and visible control', [ids[HIDDEN], ids[VISIBLE]])]
        assert any(PLANTED in row[0] for row in entities.connection.execute(
            'SELECT payload_json FROM lcm_memory_events'))
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            entities.connection.execute('DELETE FROM lcm_memory_events')
    assert cc.leave_readable(database)
    assert database.stat().st_size < 5_000_000
    assert_mapped(database)

    from tests.test_cloud_view import COPY, built, forget, sql
    import shutil
    home = tmp_path / 'local'
    home.mkdir()
    shutil.copyfile(database, home / 'lcm.db')
    copy = home / 'cloud/replicas' / COPY
    copy.mkdir(parents=True)
    shutil.copytree(database.parent, copy / 'gen-1')
    (copy / 'current').write_text('1')
    entity_tables = ('lcm_entities', 'lcm_entity_aliases', 'lcm_entity_keys',
                     'lcm_memory_notes', 'lcm_memory_events', 'lcm_memory_host_file_revisions')
    assert all(cloud_view.TABLES.get(t) == 'unsupported' for t in entity_tables)
    before = {t: sql(database, f'SELECT * FROM {t}') for t in entity_tables}
    folder, _ = built((home, copy, {}))
    assert (folder / 'lcm.db').stat().st_ino == (copy / 'gen-1/lcm.db').stat().st_ino
    forget(home, HIDDEN)
    assert cloud_view.build(home, COPY)['code'] == 'view_unsupported_table'
    assert {t: sql(copy / 'gen-1/lcm.db', f'SELECT * FROM {t}') for t in entity_tables} == before


@pytest.mark.parametrize('plant', [False, True])
def test_a3_every_capture_feature(tmp_path, plant):
    from tests.test_cloud_view import COPY, FORK, PARENT, built, forget, make_store, scanner, reject_planted_event
    import shutil
    child = HIDDEN + '/agent:A'
    collision = PARENT + '/agent:' + HIDDEN
    source, _ = make_store(tmp_path, (HIDDEN, VISIBLE, PARENT, child, collision, FORK), features=True)
    assert_mapped(source / 'lcm.db')
    home = tmp_path / 'local'
    home.mkdir()
    shutil.copyfile(source / 'lcm.db', home / 'lcm.db')
    copy = home / 'cloud/replicas' / COPY
    copy.mkdir(parents=True)
    shutil.copytree(source, copy / 'gen-1')
    (copy / 'current').write_text('1')
    forget(home, HIDDEN)
    folder, _ = built((home, copy, {}))
    allowed = (child, collision, FORK)
    scanner(folder / 'lcm.db', folder / 'lcm-large-outputs', allowed)
    if plant:
        reject_planted_event(copy / 'gen-1/lcm.db', folder, allowed)


def test_a3_empty_entity_tables(tmp_path):
    from hermes_lcm.db_bootstrap import ensure_entity_tables
    from tests.test_cloud_view import COPY, built, forget
    import shutil
    database, _ = capture_store(tmp_path)
    with closing(sqlite3.connect(database)) as conn:
        ensure_entity_tables(conn)
    home = tmp_path / 'local'
    home.mkdir()
    shutil.copyfile(database, home / 'lcm.db')
    copy = home / 'cloud/replicas' / COPY
    copy.mkdir(parents=True)
    shutil.copytree(database.parent, copy / 'gen-1')
    (copy / 'current').write_text('1')
    forget(home, HIDDEN)
    folder, _ = built((home, copy, {}))
    assert not sql_hidden(folder / 'lcm.db')


def sql_hidden(database):
    with closing(sqlite3.connect(database)) as conn:
        return conn.execute('SELECT 1 FROM messages WHERE session_id=?', (HIDDEN,)).fetchall()
