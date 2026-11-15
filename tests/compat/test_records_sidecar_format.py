"""Records format 1 is separate from canonical schema; reader floor v0.30.2 (I5/I6)."""
from pathlib import Path
import shutil
import sqlite3
from contextlib import closing

import pytest

from hermes_lcm import store as sm
from tests.compat.helpers import run_old
from tests.test_records_sidecar import TABLE, register, read, COLUMNS

# The oldest GA reader. CI's checkout has main's history only, so release-branch pins such as v0.30.1-rc3
# (98651d61) are not available there; any pre-v0.30.2 reader ignores lcm.db.records the same way.
PINS = {'v0.30.0': '6260c31f'}
TURN = r'''
store = store_mod.MessageStore(db, hermes_home=home)
try:
    store.append('S', {'role': 'user', 'content': 'old reader turn', 'timestamp': 7.0})
    rows = store.get_session_messages('S')
    print(json.dumps({'rows': [[r['store_id'], r['role'], r['content']] for r in rows]}))
finally:
    store.close()
'''


@pytest.mark.parametrize('reader', PINS)
def test_forward_old_store_opens_without_records(tmp_path, old_tree, host_root, reader):
    result = run_old(old_tree(reader, PINS), tmp_path, host_root, TURN)
    store = sm.MessageStore(tmp_path / 'lcm.db')
    try:
        assert [[r['store_id'], r['role'], r['content']] for r in store.get_session_messages('S')] == result['rows']
        assert not sm.records_path(store).exists()
    finally:
        store.close()


@pytest.mark.parametrize('reader', PINS)
def test_rollback_ignores_records_and_reupgrade_keeps_them(tmp_path, old_tree, host_root, reader):
    before_registry = dict(sm._RECORDS_TABLES)
    register()
    with_records = tmp_path / 'with'
    without_records = tmp_path / 'without'
    without_records.mkdir()
    store = sm.MessageStore(with_records / 'lcm.db')
    try:
        sid = store.append('S', dict(role='user', content='candidate turn', timestamp=1.0))
        row = store.connection.execute('SELECT ' + ','.join(COLUMNS.values()) + ' FROM messages WHERE store_id=?',
                                       (sid,)).fetchone()
        with closing(sm.open_records_writer(store)) as writer:
            assert writer.insert(TABLE, dict(zip(COLUMNS, row), payload='intact')) is None
    finally:
        store.close()
    path = sm.records_path(with_records / 'lcm.db')
    before = path.read_bytes()
    shutil.copyfile(with_records / 'lcm.db', without_records / 'lcm.db')
    try:
        tree = old_tree(reader, PINS)
        with_result = run_old(tree, with_records, host_root, TURN)
        without_result = run_old(tree, without_records, host_root, TURN)
        assert with_result == without_result
        assert path.read_bytes() == before
        reopened = sm.MessageStore(with_records / 'lcm.db')
        try:
            assert [r['content'] for r in reopened.get_session_messages('S')] == ['candidate turn', 'old reader turn']
            rows, unattached = read(reopened)
            assert unattached == 0 and rows[0]['payload'] == 'intact'
            assert path.read_bytes() == before
        finally:
            reopened.close()
    finally:
        sm._RECORDS_TABLES.clear()
        sm._RECORDS_TABLES.update(before_registry)


def test_negative_newer_records_format_never_downgrades(tmp_path):
    store = sm.MessageStore(tmp_path / 'lcm.db')
    path = sm.records_path(store)
    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.execute('PRAGMA user_version=2')
        original = Path(path).read_bytes()
        with pytest.raises(sm.RecordsFormatError):
            sm.open_records_writer(store)
        with sm.records_read_cut(store) as cut:
            assert cut.refused_format == 2 and cut.records is None
        assert path.read_bytes() == original
        store.append('S', dict(role='user', content='canonical still available'))
        assert path.read_bytes() == original
    finally:
        store.close()
