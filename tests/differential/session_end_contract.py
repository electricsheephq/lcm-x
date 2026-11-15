"""Strict canonical-only session-end observations, independent of #2707 projections.

Code citations are at a42baa75: prefix_matching.py:272-281 compares the prefix;
engine.py:5342 and 5376-5390 refuse mismatches; :12114-12122 queues a cold
view; :12252-12300 selects detached, went-on and current retries; :12398-12429
compares/appends; :12431-12506 shares the retry budget; :12619-12679 drains
then closes. maintenance_scheduler.py:421-430 is the at-exit drain.
"""
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from types import SimpleNamespace

from .session import assert_faults, audit_fault, freeze
from .session_end_expected import FAULTS, INTENDED_DIFFERENCES, ROUTES, SHRINKING_TABLES

# Local copies keep this gate independent of the historical pin modules.
FULL_OUTPUT = "SPILLOVER_RECOVERY_NEEDLE:\n" + ("é文abcdef\n" * 600)


def _b64(n):
    return base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(n))).decode("ascii")


def _call(shape):
    if shape == "wrapped-base64":
        long = _b64(3420)
        wrapped = "\n".join(long[i:i + 76] for i in range(0, len(long), 76))
        arguments = json.dumps({"path": "a.b64", "content": "hdr\n" + wrapped + "\nend"})
    elif shape == "data-uri-key":
        arguments = json.dumps({"data:image\\/png;base64," + _b64(1200): "v", "b": "x"})
    else:  # plain control
        arguments = json.dumps({"path": "a.txt", "content": "hdr\nplain body\nend"})
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "m1", "type": "function", "function": {"name": "write_file", "arguments": arguments}}]}


def _history(tool_content):
    rows = [("user", "A"), ("assistant", "B"), ("user", "C")]
    msgs = [{"role": r, "content": t, "timestamp": float(i + 1)} for i, (r, t) in enumerate(rows)]
    msgs.append({"role": "assistant", "content": "", "timestamp": 4.0, "tool_calls": [
        {"id": "m1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
    msgs.append({"role": "tool", "tool_call_id": "m1", "content": tool_content, "timestamp": 5.0})
    for i, (r, t) in enumerate((("assistant", "D"), ("user", "E"), ("assistant", "F"), ("user", "G"),
                                ("assistant", "H"))):
        msgs.append({"role": r, "content": t, "timestamp": float(6 + i)})
    return msgs


def _state_db(tmp_path, sessions) -> None:
    """Host state.db next to lcm.db: (id, parent_session_id, end_reason, source, model_config)."""
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, "
                 "end_reason TEXT, source TEXT, model_config TEXT)")
    conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
                     [tuple(row) + (None,) * (5 - len(row)) for row in sessions])
    conn.commit()
    conn.close()


def _marker(path, full=FULL_OUTPUT, *, count=None, preview=None):
    """Hermes f42f579c tools/tool_result_storage.py::_build_persisted_message shape."""
    count = len(full) if count is None else count
    preview = full[:30] if preview is None else preview
    return (
        "<persisted-output>\n"
        f"This tool result was too large ({count:,} characters, {count / 1024:.1f} KB).\n"
        f"Full output saved to: {path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n"
        "Recovery: page through the saved file with read_file (offset/limit) or "
        "process it with execute_code — do NOT re-request the same data from the "
        "remote API; the full result is already on disk.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        f"{preview}\n...\n"
        "</persisted-output>"
    )


def _write_output(path, full=FULL_OUTPUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(full.encode("utf-8"))
    return _marker(path, full)


# Excluded by column NAME only. observed_at and host stamps remain semantic.
CLOCK_COLUMNS = frozenset(('timestamp', 'ingested_at', 'created_at', 'updated_at', 'completed_at',
    'last_finalized_at', 'debt_updated_at', 'last_maintenance_attempt_at', 'last_rollover_at', 'last_reset_at'))
# All deadline reads on these paths, including shutdown's child-store drain.
DEADLINE_READS = {'engine.py': (12076, 12079, 12365, 12437, 12443, 12449, 12500, 12513, 12574, 12640),
                  'maintenance_scheduler.py': (204, 216, 228, 234, 243, 282, 301, 318, 422),
                  'aux_session.py': (963, 968), 'db_bootstrap.py': (179, 186)}


class Clock:
    def __init__(self, mode='instant'):
        self.now, self.mode, self.slow = 1000., mode, False
        self.deadlines = []

    def monotonic(self):
        return self.now

    def install(self, package, patch):
        proxy = SimpleNamespace(**{key: getattr(time, key) for key in dir(time) if not key.startswith('__')})
        proxy.monotonic = proxy.perf_counter = self.monotonic
        proxy.time = lambda: 1700001000.
        proxy.time_ns = lambda: 1700001000000000000
        proxy.gmtime = lambda: time.gmtime(1700001000)
        for suffix in ('engine', 'maintenance_scheduler', 'aux_session', 'store', 'dag', 'db_bootstrap',
                       'lifecycle_state', 'compaction', 'externalize', 'reconcile', 'ingest_ledger'):
            module = importlib.import_module(package + '.' + suffix)
            if hasattr(module, 'time'):
                patch.setattr(module, 'time', proxy)
        store = importlib.import_module(package + '.store')
        patch.setattr(store, 'secrets', SimpleNamespace(token_bytes=lambda size: b'l' * size))

    def busy(self, engine):
        if self.mode == 'busy-timeout':
            self.now += sys.modules[engine.__module__]._SESSION_END_BUSY_TIMEOUT_MS / 1000

    def cold(self):
        if self.mode == 'slow-cold' and not self.slow:
            self.now += 10.
            self.slow = True


def slow_seam(dotted, clock, patch):
    """Opt-in only: wrap any named function, expiring every caller deadline.

    No product seam is registered by this PR. A later WARNING PR may call this
    with its dotted function name; it must still pass the same canonical judge.
    """
    parts = dotted.split('.')
    for count in range(len(parts) - 1, 0, -1):
        try:
            owner = importlib.import_module('.'.join(parts[:count]))
        except ModuleNotFoundError:
            continue
        for part in parts[count:-1]:
            owner = getattr(owner, part)
        break
    else:
        raise AssertionError(f'slow seam target absent: {dotted}')
    original = getattr(owner, parts[-1])

    def slowed(*args, **kwargs):
        frame, deadlines = sys._getframe(1), list(clock.deadlines)
        while frame:
            value = frame.f_locals.get('deadline')
            if isinstance(value, (int, float)):
                deadlines.append(value)
            frame = frame.f_back
        clock.now = max([clock.now, *deadlines]) + 1.
        return original(*args, **kwargs)
    patch.setattr(owner, parts[-1], slowed)


def fault(engine, owner, target, kind, clock, patch, *, error=sqlite3.OperationalError, after=False):
    audit = audit_fault(engine, kind, hasattr(owner, target))
    assert audit.present, f'{kind}: target absent {target}'
    original = getattr(owner, target)

    def injected(*args, **kwargs):
        audit.called += 1
        if after:
            original(*args, **kwargs)
        clock.busy(engine)
        audit.raised += 1
        raise error('database is locked' if issubclass(error, sqlite3.Error) and kind != 'failed-ledger-write' else kind)
    patch.setattr(owner, target, injected)
    return audit


def floor(traces, expected=FAULTS):
    assert traces, 'floor skipped'
    for kind in expected:
        for tree in ('main', 'candidate'):
            audits = [f for pair in traces for f in pair[tree].faults if f.kind == kind]
            assert audits and all(f.present and f.called and f.raised for f in audits), \
                f'{tree}: declared fault {kind} never entered/raised'
    for pair in traces:
        assert_faults('session-end', pair)


def payloads(root):
    result = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            try:
                metadata = json.loads(data)
            except (ValueError, UnicodeError):
                metadata = None
            result.append((str(path.relative_to(root)), len(data), hashlib.sha256(data).hexdigest(), freeze(metadata)))
    return result


def database(path):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
        schema = conn.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name').fetchall()
        tables = {}
        for kind, name, *_ in schema:
            if kind != 'table':
                continue
            quoted = '"' + name.replace('"', '""') + '"'
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info({quoted})')]
            keep = [i for i, col in enumerate(cols) if col not in CLOCK_COLUMNS]
            rows = conn.execute(f'SELECT * FROM {quoted}').fetchall()
            tables[name] = (tuple(cols[i] for i in keep), sorted([tuple(r[i] for i in keep) for r in rows], key=repr))
    return schema, tables


def snapshot(engines, root, paths):
    schemas, tables, canonical = {}, {}, {}
    for path in sorted(paths):
        key = str(path.relative_to(root))
        schemas[key], tables[key] = database(path)
        canonical[key] = tables[key].get('messages')  # Includes store_id and token estimates.
    def value(item):
        if hasattr(item, 'db_path'):
            return ('store', str(Path(item.db_path).relative_to(root)))
        if isinstance(item, dict):
            return {key: value(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [value(val) for val in item]
        return item
    return dict(canonical=canonical, schema=schemas, tables=tables, payload=payloads(root / 'payloads'),
        pending=[freeze(value(e._pending_session_tails)) for e in engines],
        ledger=[dict(e.__dict__.get('_ingest_ledger_counters', {})) for e in engines],
        # Empty by construction at a42baa75; lifecycle state is covered by tables.
        lifecycle=[{k: value(v) for k, v in e.__dict__.items() if k.startswith('_lifecycle_') and not callable(v)}
                   for e in engines])


def _stored_counts(snapshot, key):
    # Clock columns are already dropped: genuinely identical rows share a key;
    # restoring two identical rows main lost would read as doubled.
    counts = Counter()
    for path, data in snapshot[key].items():
        messages = data.get('messages') if key == 'tables' else data
        if messages is not None:
            columns, rows = messages
            counts.update((path, freeze({col: value for col, value in zip(columns, row) if col != 'store_id'}))
                          for row in rows)
    return counts


def _stored_order(snapshot, key):
    ordered = {}
    for path, data in snapshot[key].items():
        messages = data.get('messages') if key == 'tables' else data
        if messages is not None:
            columns, rows = messages
            for row in sorted(rows, key=lambda row: row[columns.index('store_id')]):
                values = dict(zip(columns, row))
                group = (path, values['session_id'])
                ordered.setdefault(group, []).append(freeze({k: v for k, v in values.items() if k != 'store_id'}))
    return ordered


def judge(main, candidate, stage, *, case=None, mode=None, seen=None):
    for key in ('canonical', 'schema', 'tables', 'payload', 'pending', 'ledger', 'lifecycle'):
        entries = [entry for entry in INTENDED_DIFFERENCES
                   if entry[0] == case and entry[1] in ('*', mode) and entry[2:4] == (stage, key)]
        if entries:
            assert candidate[key] != main[key], f'{case}/{mode}/{stage}: stale intended difference {key}'
            if key in ('canonical', 'tables'):
                before, after = _stored_counts(main, key), _stored_counts(candidate, key)
                for row in before.keys() | after.keys():
                    assert after[row] >= before[row], f'{stage}: {key} lost stored row'
                    assert after[row] <= max(before[row], 1), f'{stage}: {key} doubled stored row'
                ordered = _stored_order(candidate, key)
                for group, rows in _stored_order(main, key).items():
                    remaining = iter(ordered.get(group, ()))
                    assert all(any(row == value for value in remaining) for row in rows), \
                        f'{stage}: {key} reordered stored row'
            if key == 'tables':
                for path, tables in main[key].items():
                    # Skip actual FTS5 shadows: their row counts follow segment merges; external content has no _content shadow.
                    shadows = {name + suffix for kind, name, _, sql in main['schema'].get(path, ())
                               if kind == 'table' and re.match(
                                   r'\s*CREATE\s+VIRTUAL\s+TABLE\b.*?\bUSING\s+fts5\b', sql or '', re.I | re.S)
                               # content_rowid= and contentless_delete= do not count as content=.
                               for suffix in ('_data', '_idx', '_docsize', '_config', *(
                                   ('_content',) if re.search(r'\bcontent\s*=', sql, re.I) is None else ()))}
                    for table, (_, rows) in tables.items():
                        if table != 'messages' and table not in SHRINKING_TABLES and table not in shadows:
                            after_rows = candidate[key].get(path, {}).get(table, ((), ()))[1]
                            assert len(after_rows) >= len(rows), f'{stage}: tables lost rows in {table}'
            if seen is not None:
                seen.update(entries)
        else:
            assert candidate[key] == main[key], f'{stage}: {key} changed'


class Proof:
    """Observe pre-protection host input and the storing attempt's index->id map.

    No digest is inferred from protected text or candidate output. A rollback's
    tentative map is not a receipt: only committed rows are inspected at stages.
    """
    def __init__(self):
        self.origins, self.counts, self.stages = {}, Counter(), {}
        self.route = None

    def install(self, engine, patch):
        original = engine._ingest_messages_transaction
        def transaction(messages, *args, **kwargs):
            host = deepcopy(messages)
            result = original(messages, *args, **kwargs)
            prepared = kwargs.get('prepared') or {}
            for index, sid in prepared.get('stored_at', {}).items():
                self.origins[(str(engine._store.db_path), sid)] = (self.route or 'ordinary', host[index])
            return result
        patch.setattr(engine, '_ingest_messages_transaction', transaction)
        batch_append = engine._append_ingest_batch
        def batch(batch):
            ids = batch_append(batch)
            for sid, host in zip(ids, batch['metadata_messages']):
                self.origins[(str(batch['store'].db_path), sid)] = (self.route or 'ordinary', deepcopy(host))
            return ids
        patch.setattr(engine, '_append_ingest_batch', batch)
        append = engine._append_off_current_session_end_suffix
        def s3(*args, **kwargs):
            ids = append(*args, **kwargs)
            store = kwargs.get('store', engine._store)
            for sid in ids:
                self.origins[(str(store.db_path), sid)] = ('S3', None)
            return ids
        patch.setattr(engine, '_append_off_current_session_end_suffix', s3)

    def check(self, engine, stage, paths, special=None):
        counts = Counter({route: 0 for route in ROUTES})
        ledger = importlib.import_module(engine.__module__.rsplit('.', 1)[0] + '.ingest_ledger')
        for path in paths:
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                salt = conn.execute("SELECT value FROM ledger_meta WHERE key='digest_salt'").fetchone() if 'ledger_meta' in names else None
                for row in conn.execute('SELECT * FROM messages ORDER BY store_id'):
                    sid = row['store_id']
                    route, host = self.origins.get((str(path), sid), ('unknown', None))
                    bindings = list(conn.execute('SELECT * FROM occurrence_bindings WHERE store_id=?', (sid,))) if salt else []
                    dispositions = {b['disposition'] for b in bindings}
                    if route == 'S3':
                        pass
                    elif dispositions == {'migrated'}:
                        route = 'migrated'
                    elif 'composite_part' in dispositions:
                        route = 'composite-part'
                    elif route in ('ledger-off', 'before-ledger', 'failed-ledger-write') and not bindings:
                        pass
                    else:
                        assert route == 'ordinary' and len(bindings) == 1 and dispositions == {'stored'}, \
                            f'STOP proof route outside named routes: {stage} row={sid} route={route} bindings={len(bindings)}'
                        assert bindings[0]['identity_digest'] == ledger.identity_digest(salt[0], host), \
                            f'STOP proof digest mismatch: {stage} row={sid}'
                        route = 'stored'
                    counts[route] += 1
        self.stages[stage] = dict(counts)
        self.counts.update(counts)
