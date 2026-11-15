"""Receipt reservation across the audited cursor and reissued occurrences (I1/I8/I9)."""
import copy
import json

import pytest

from hermes_lcm import ingest_ledger
from tests.invariants.test_issue_2742_warm_tail_refinal import _open
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _view


@pytest.mark.parametrize('uid_mode', [None, 'off'])
@pytest.mark.parametrize('shape', ['plain', 'wrapped-base64', 'data-uri-key'])
@pytest.mark.parametrize('edit', [0, 1])
def test_edit_reissued_call_and_no_uid_twin(tmp_path, monkeypatch, shape, edit, uid_mode):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        incoming = copy.deepcopy(view)
        incoming[edit]['content'] += ' (edited by host)'
        incoming[2]['message_uid'] = 'reissued-call-uid'
        twin = copy.deepcopy(view[2])
        twin.pop('message_uid')
        engine.ingest(copy.deepcopy(incoming + [twin]))
        rows = engine._store.get_session_messages('parent')
        assert len(rows) == 6, rows
        assert sum(bool(r.get('tool_calls')) for r in rows) == 2
        assert rows[-1]['tool_calls'] and rows[-1]['observed_at'] == twin['timestamp']
        assert rows[-2]['content'] == incoming[edit]['content']
    finally:
        engine.shutdown()


@pytest.mark.parametrize('uid_mode', [None, 'off'])
@pytest.mark.parametrize('shape', ['wrapped-base64', 'data-uri-key'])
def test_reissued_call_occurrence_table(tmp_path, monkeypatch, shape, uid_mode, record_property):
    engine = _open(tmp_path, monkeypatch, uid_mode)
    try:
        view = _view(shape)
        engine.ingest(copy.deepcopy(view))
        incoming = copy.deepcopy(view)
        incoming[2]['message_uid'] = 'reissued-call-uid'
        twin = copy.deepcopy(view[2])
        twin.pop('message_uid')
        engine.ingest(copy.deepcopy(incoming + [twin]))
        rows = engine._store.get_session_messages('parent')
        conn = engine._store._conn
        blob = conn.execute("SELECT entries FROM ingest_views WHERE kind='ingest' ORDER BY view_seq DESC LIMIT 1").fetchone()[0]
        entries = ingest_ledger.decode_entries(blob)
        table = [(i, conn.execute('SELECT store_id FROM occurrence_bindings WHERE occ_id=?', (e.ref,)).fetchone())
                 for i, e in enumerate(entries)]
        # Prefix stubs deliberately have no binding ref. Resolve their unique,
        # unchanged content/stamp identities directly from the stored rows.
        salt = engine._store.ingest_ledger_salt()
        for i in range(2):
            matches = [r['store_id'] for r in rows if r['observed_at'] == incoming[i]['timestamp']
                       and ingest_ledger.identity_digest(salt, r) == ingest_ledger.identity_digest(salt, incoming[i])]
            assert len(matches) == 1, (i, matches)
            assert table[i][1] is None
            table[i] = (i, (matches[0],))
        table = [(i, row[0] if row else None) for i, row in table]
        print('OCCURRENCE_TABLE', json.dumps(dict(shape=shape, uid_mode=uid_mode, table=table,
              rows=[(r['store_id'], r['role'], bool(r.get('tool_calls'))) for r in rows])))
        record_property('occurrence_table', json.dumps(table))
        assert len(rows) == 5, rows
        assert sum(bool(r.get('tool_calls')) for r in rows) == 2
        assert len(table) == 5 and all(sid is not None for _, sid in table), table
        assert len({sid for _, sid in table}) == 5, table
        assert {sid for _, sid in table} == {r['store_id'] for r in rows}, table
    finally:
        engine.shutdown()
