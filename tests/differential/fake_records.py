"""Test-only records-v1 adapter over the unchanged main engine."""
import json
import sqlite3
from types import SimpleNamespace

from .record_contract import expected_record, writer_fault_stages
from .session import RECORD_COLUMNS, canonical_rows, queued_rows


WRITE_STYLE = 'batch'


def withholding(engine):
    """Fake implementation: scan its own canonical and physical record rows."""
    rows = [dict(row) for row in canonical_rows(engine)]
    result = []
    for raw in queued_rows(engine):
        record = dict(raw)
        if record['superseded_by'] is not None or not record['content'].strip():
            continue
        owner = next(r for r in rows if r['store_id'] == record['owner_store_id'])
        needle = record['content'].strip().split('\n\n')
        for row in rows:
            if row['session_id'] != owner['session_id'] or row['store_id'] <= owner['store_id'] or row['role'] != 'user':
                continue
            segments = row['content'].split('\n\n')
            if any(segments[start:start + len(needle)] == needle for start in range(len(segments))):
                result.append(dict(record, state='overlap', reference=reference(record)))
                break
    return result


def reference(row):
    return dict(tool='lcm_expand', args=dict(store_id=row['owner_store_id']))


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
        witness = engine._paired_trace.baseline_stages[stage - 1]['supersession_witness']
        for owner, sid in witness.items():
            conn.execute('UPDATE queued_user_rows SET superseded_by=? WHERE owner_store_id=?', (sid, owner))
        conn.commit()
        return result
    patch.setattr(engine, 'ingest', ingest)
    def record_view(row):
        return dict(row, kind='record', reference=reference(row))
    def grep(args, **_):
        text = args['query'].strip('"')
        hidden = {(r['owner_store_id'], r['ordinal']) for r in withholding(engine)}
        hits = [record_view(r) for r in map(dict, queued_rows(engine))
                if r['content'] == text and (r['owner_store_id'], r['ordinal']) not in hidden]
        canonical = [dict(r) for r in canonical_rows(engine) if text in dict(r)['content']]
        return json.dumps(dict(results=canonical, queued_user_record_hits=hits))
    def expand(args, **_):
        owner = args['store_id']
        canonical = next(dict(r) for r in canonical_rows(engine) if dict(r)['store_id'] == owner)
        hidden = {(r['owner_store_id'], r['ordinal']): r['state'] for r in withholding(engine)}
        records = [dict(record_view(r), state=hidden.get((r['owner_store_id'], r['ordinal']), 'visible'))
                   for r in map(dict, queued_rows(engine)) if r['owner_store_id'] == owner]
        return json.dumps(dict(canonical, queued_user_records=records))
    def load(args, **_):
        remaining = [dict(r) for r in canonical_rows(engine) if dict(r)['store_id'] > args.get('after_store_id', 0)]
        page = remaining[:args.get('limit', 1)]
        owners = {r['store_id'] for r in page}
        locators = [dict(owner_store_id=r['owner_store_id'], ordinal=r['ordinal'], state=r['state'],
                         reference=reference(r)) for r in withholding(engine) if r['owner_store_id'] in owners]
        hidden = {(r['owner_store_id'], r['ordinal']) for r in locators}
        visible = [record_view(r) for r in map(dict, queued_rows(engine)) if r['owner_store_id'] in owners
                   and r['superseded_by'] is None and (r['owner_store_id'], r['ordinal']) not in hidden]
        reply = dict(messages=page, next_cursor=page[-1]['store_id'] if len(remaining) > len(page) else None)
        if visible:
            reply['queued_user_records'] = visible
        if locators:
            counts = {state: sum(r['state'] == state for r in locators) for state in ('overlap', 'unscanned')}
            reply['queued_user_records_withheld'] = dict(counts=counts, locators=locators)
        return json.dumps(reply)
    return SimpleNamespace(lcm_grep=grep, lcm_expand=expand, lcm_load_session=load)
