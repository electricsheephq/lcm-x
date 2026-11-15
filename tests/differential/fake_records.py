"""Test-only records-v1 adapter over the unchanged main engine."""
import json
import sqlite3
from types import SimpleNamespace

from .record_contract import expected_record, projection, writer_fault_stages
from .session import RECORD_COLUMNS, canonical_rows, queued_rows


WRITE_STYLE = 'batch'


def install(engine, entries, patch):
    conn = engine._store._conn
    conn.execute('CREATE TABLE IF NOT EXISTS queued_user_rows (' + ', '.join(RECORD_COLUMNS) + ')')
    from .session import freeze
    def fake_rows():
        return [freeze(dict(zip(RECORD_COLUMNS, row))) for row in conn.execute(
            'SELECT ' + ', '.join(RECORD_COLUMNS) + ' FROM queued_user_rows ORDER BY owner_store_id, ordinal')]
    patch.setattr(engine, '_fake_queued_rows', fake_rows, raising=False)
    def write(rows):
        existing = {tuple(row) for row in conn.execute('SELECT * FROM queued_user_rows')}
        for row in rows:
            values = tuple(row[k] for k in RECORD_COLUMNS)
            if values not in existing:
                conn.execute('INSERT INTO queued_user_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', values)
                existing.add(values)
        conn.commit()
    patch.setattr(engine, '_store_queued_user_rows', write, raising=False)
    original, pending = engine.ingest, []
    def ingest(*args, **kwargs):
        result = original(*args, **kwargs)
        stage = len(engine._paired_trace.ingests) + 1
        stored = {(dict(r)['owner_store_id'], dict(r)['ordinal']) for r in queued_rows(engine)}
        pending[:] = [expected_record(e) for e in entries if e['stage'] <= stage
                      and (e['owner_store_id'], e['predecessor_ordinal']) not in stored]
        batches = [[r for r in pending if r['owner_store_id'] == owner]
                   for owner in sorted({r['owner_store_id'] for r in pending})] if WRITE_STYLE == 'per-owner' else [pending]
        for batch in batches:
            for _ in range(2 if WRITE_STYLE == 'retry' else 1):
                try:
                    engine._store_queued_user_rows(rows=batch)
                except (ValueError, RuntimeError, sqlite3.Error):
                    assert stage in writer_fault_stages(engine._paired_trace.case)
                else:
                    break
        witness = engine._paired_trace.baseline_stages[stage - 1]['reload_witness']
        for owner, sid in witness.items():
            conn.execute('UPDATE queued_user_rows SET superseded_by=? WHERE owner_store_id=?', (sid, owner))
        conn.commit()
        return result
    patch.setattr(engine, 'ingest', ingest)
    def rows():
        return [dict(row) for row in projection(dict(canonical=canonical_rows(engine), records=queued_rows(engine)))]
    def grep(args, **_):
        text = args['query'].strip('"')
        hits = [dict(r, reference=dict(tool='lcm_expand', args=dict(queued_user_ref=[r['owner_store_id'], r['ordinal']])))
                for r in map(dict, queued_rows(engine)) if r['content'] == text]
        return json.dumps(dict(results=hits))
    def expand(args, **_):
        key = args['queued_user_ref']
        return json.dumps(next(dict(r) for r in queued_rows(engine)
                               if [dict(r)['owner_store_id'], dict(r)['ordinal']] == key))
    def load(args, **_):
        start = args.get('record_cursor', 0)
        data = rows()
        return json.dumps(dict(messages=data[start:start + 1], next_cursor={'record_cursor': start + 1} if start + 1 < len(data) else None))
    return SimpleNamespace(lcm_grep=grep, lcm_expand=expand, lcm_load_session=load)
