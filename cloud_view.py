"""Immutable replica views. Only this standard-library process edits private copies."""
from contextlib import closing, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

from .cloud_copy import EXTERNALIZED_REF_RE, INGEST_PLACEHOLDER_RE, wait_lock

RULES = 1
FORGOTTEN_QUERY = "SELECT key FROM metadata WHERE key LIKE 'capture:forgotten:%'"
TABLES = {'sqlite_*': 'global', 'messages': 'session', 'summary_nodes': 'session',
          'capture_events': 'session', 'session_lineage': 'link', 'lcm_lifecycle_state': 'link',
          'message_relations': 'link', 'metadata': 'metadata', 'host_uid_bindings': 'message',
          'occurrence_bindings': 'message', 'stamp_observations': 'message', 'ledger_emissions': 'message',
          'ingest_views': 'lineage', 'lineage_marks': 'lineage', 'summary_node_provenance': 'node',
          'lcm_embedding_profile': 'global', 'lcm_migration_state': 'global', 'ledger_meta': 'global'}
for _name in ('lcm_embedding_meta', 'lcm_embedding_vectors', 'lcm_embedding_binary'):
    TABLES[_name] = 'node'
for _name in ('lcm_chunk_meta', 'lcm_chunk_vectors', 'lcm_chunk_binary'):
    TABLES[_name] = 'message'
for _name in ('lcm_entities', 'lcm_entity_aliases', 'lcm_entity_keys', 'lcm_memory_notes',
              'lcm_memory_events', 'lcm_memory_host_file_revisions'):
    TABLES[_name] = 'unsupported'
for _index in ('messages_fts', 'messages_fts_stem', 'nodes_fts'):
    TABLES[_index] = 'full-text'
    for _suffix in ('data', 'idx', 'content', 'docsize', 'config'):
        TABLES[f'{_index}_{_suffix}'] = 'full-text'
METADATA = {**dict.fromkeys(('capture:forgotten:', 'capture:compact:', 'capture:fork:',
                            'capture:local_gc:', 'rotation_predecessor_session:'), 'session'),
            **dict.fromkeys(('capture:cursor:', 'capture:pending:', 'capture:ended:'), 'lineage')}
LINEAGES = ('claude-code:',)
COPY_ID = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
VIEW = r'view-[1-9][0-9]*-[1-9][0-9]*-[0-9a-f]{16}'
FLOOR = 5_000_000_000


class Refused(Exception):
    def __init__(self, code):
        self.code = code


def forgotten_digest(ids):
    return hashlib.sha256(b''.join(x + b'\n' for x in sorted({s.encode('utf-8') for s in ids}))).hexdigest()


def clean(database):
    with database.open('rb') as handle:
        handle.seek(18)
        rollback = handle.read(2) == b'\x01\x01'
    return rollback and not any(Path(str(database) + s).exists() for s in ('-wal', '-shm', '-journal'))


def guard(database):
    with database.open('rb') as handle:
        handle.seek(18)
        header = handle.read(2)
    if len(header) != 2 or max(header) >= 2 and not all(Path(str(database) + s).exists() for s in ('-wal', '-shm')):
        raise Refused('bad_database_block')


def removed_set(home):
    database = Path(home) / 'lcm.db'
    guard(database)
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        return {row[0][len('capture:forgotten:'):] for row in conn.execute(FORGOTTEN_QUERY)}


def removed_bytes(folder):
    try:
        return (folder / 'removed.json').read_bytes()
    except FileNotFoundError:
        return b''


def removed_sha256(copy_folder):
    return hashlib.sha256(removed_bytes(Path(copy_folder))).hexdigest()


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def lineage(value):
    return next((value[len(p):] for p in LINEAGES if isinstance(value, str) and value.startswith(p) and value[len(p):]), None)


def plan(conn, removed, row_ids):
    """Collect immutable key sets, then return predicates used for reads AND deletes."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name in ('R', 'M', 'N', 'C', 'O'):
        conn.execute(f'CREATE TEMP TABLE {name}(id PRIMARY KEY)')
    conn.executemany('INSERT INTO R VALUES (?)', [(s,) for s in removed])
    for target, table, key in [('M', 'messages', 'store_id'), ('N', 'summary_nodes', 'node_id')]:
        conn.execute(f'INSERT INTO {target} SELECT {key} FROM {table} WHERE session_id IN R')
        if target == 'M':  # F12 removed_ids are message store_ids, not summary node_ids.
            conn.executemany(f'INSERT OR IGNORE INTO {target} VALUES (?)', [(n,) for n in row_ids])
    if 'lcm_chunk_meta' in tables:
        conn.execute('INSERT OR IGNORE INTO C SELECT chunk_id FROM lcm_chunk_meta WHERE store_id IN M')
    if 'occurrence_bindings' in tables:
        conn.execute('INSERT OR IGNORE INTO O SELECT occ_id FROM occurrence_bindings WHERE store_id IN M OR lineage_hit(lineage)')
    conditions = {'messages': 'session_id IN R OR store_id IN M',
                  'summary_nodes': 'session_id IN R OR node_id IN N',
                  'capture_events': 'session_id IN R OR lineage_hit(lineage_key) OR EXISTS (SELECT 1 FROM json_each(store_ids) WHERE value IN M)',
                  'session_lineage': 'child_session_id IN R OR parent_session_id IN R',
                  'lcm_lifecycle_state': 'current_session_id IN R OR last_finalized_session_id IN R',
                  'message_relations': 'store_id IN M OR related_store_id IN M',
                  'metadata': "metadata_hit(key) OR (key LIKE 'rotation_predecessor_session:%' AND value IN R)",
                  'host_uid_bindings': 'store_id IN M OR lineage_hit(lineage_key)',
                  'occurrence_bindings': 'store_id IN M OR lineage_hit(lineage)',
                  'stamp_observations': 'occ_id IN O OR lineage_hit(lineage)',
                  'ledger_emissions': 'recovered_store_id IN M OR next_occ_id IN O OR lineage_hit(lineage) OR EXISTS (SELECT 1 FROM json_each(source_store_ids) WHERE value IN M)',
                  'ingest_views': 'lineage_hit(lineage)', 'lineage_marks': 'lineage_hit(lineage)',
                  'summary_node_provenance': 'node_id IN N', 'lcm_chunk_meta': 'store_id IN M'}
    for table in ('lcm_embedding_meta', 'lcm_embedding_vectors', 'lcm_embedding_binary'):
        conditions[table] = 'CAST(embedded_id AS INTEGER) IN N'
    for table in ('lcm_chunk_vectors', 'lcm_chunk_binary'):
        conditions[table] = 'chunk_id IN C'
    conditions = {t: p for t, p in conditions.items() if t in tables}
    validate_keys(conn, conditions, removed)
    matches = {t: [r[0] for r in conn.execute(f'SELECT rowid FROM {quote(t)} WHERE {p}')] for t, p in conditions.items()}
    if removed or row_ids:
        for table in tables:
            if not table.startswith('sqlite_') and TABLES.get(table, 'unsupported') == 'unsupported' and conn.execute(f'SELECT 1 FROM {quote(table)} LIMIT 1').fetchone():
                raise Refused('view_unsupported_table')
    for node, kind, sources in conn.execute('SELECT node_id,source_type,source_ids FROM summary_nodes WHERE NOT (' + conditions['summary_nodes'] + ')'):
        target = 'M' if kind == 'messages' else 'N'
        if conn.execute(f'SELECT 1 FROM json_each(?) WHERE value IN {target} LIMIT 1', (sources,)).fetchone():
            raise Refused('view_summary_spans_forgotten')
    return tables, conditions, matches



def validate_keys(conn, conditions, removed):
    for table, predicate in conditions.items():
        cols = {r[1] for r in conn.execute(f'PRAGMA table_info({quote(table)})')}
        for col in ('lineage', 'lineage_key'):
            if col in cols:
                for (value,) in conn.execute(f'SELECT {col} FROM {quote(table)}'):
                    if lineage(value) is None and any(s in str(value) for s in removed):
                        raise Refused('view_unsupported_key')
    for (key,) in conn.execute('SELECT key FROM metadata'):
        family = next((p for p in METADATA if key.startswith(p)), None)
        if (family is None or METADATA[family] == 'lineage' and lineage(key[len(family):]) is None) and any(s in key for s in removed):
            raise Refused('view_unsupported_key')

def connect(database, removed, readonly=True):
    conn = sqlite3.connect(database.resolve().as_uri() + ('?mode=ro' if readonly else ''), uri=True)
    conn.create_function('lineage_hit', 1, lambda v: lineage(v) in removed)
    def metadata_hit(key):
        for prefix, kind in METADATA.items():
            if key.startswith(prefix):
                value = key[len(prefix):]
                return (lineage(value) if kind == 'lineage' else value) in removed
        return False
    conn.create_function('metadata_hit', 1, metadata_hit)
    return conn


def payload_plan(source, conn, removed, row_ids):
    references = {}
    for row_id, session, content, calls in conn.execute('SELECT store_id,session_id,content,tool_calls FROM messages'):
        for text in (content, calls):
            for pattern in (EXTERNALIZED_REF_RE, INGEST_PLACEHOLDER_RE):
                for ref in pattern.findall(text or ''):
                    references.setdefault(ref, set()).add(session in removed or row_id in row_ids)
    kept, omitted = [], []
    folder = source / 'lcm-large-outputs'
    for path in sorted(folder.iterdir()) if folder.exists() else []:
        if path.is_symlink() or not path.is_file() or not (1 <= len(path.name.encode()) <= 255) or '\\' in path.name:
            raise Refused('bad_name')
        body = json.loads(path.read_bytes())
        owner = body.get('session_id')
        refs = references.get(path.name, set())
        (omitted if owner in removed or refs and all(refs) else kept).append(path.name)
    return kept, omitted


def atomic(folder, name, body):
    temp = folder / (name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('w') as handle:
        json.dump(body, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(folder / name)
    sync(folder)


def sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sweep(copy_folder, *, temporary_only=False, keep=None):
    folder = Path(copy_folder)
    try:
        current = json.loads((folder / 'view.json').read_bytes())['folder']
    except (OSError, ValueError, KeyError):
        current = None
    for path in folder.iterdir():
        if path.is_symlink() or not path.is_dir():
            continue
        temporary = re.fullmatch(VIEW + r'\.tmp-[0-9a-f]{32}', path.name)
        if path.name in (current, keep) or not temporary and (temporary_only or not re.fullmatch(VIEW, path.name)):
            continue
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            shutil.rmtree(path)
        except BlockingIOError:
            pass
        except OSError:
            try:
                if sys.stderr is not None:
                    print('view_temp_cleanup_failed', file=sys.stderr)
            except Exception:
                pass
        finally:
            if fd is not None:
                os.close(fd)



def retire(folder, pointer):
    """Return a stale-view removal to try only under space.lock, after the floor."""
    try:
        previous = json.loads((folder / 'view.json').read_bytes())
        if all(previous.get(key) == pointer[key] for key in ('forgotten_sha256', 'removed_sha256', 'rules')):
            return None
        name = previous['folder']
        if not re.fullmatch(VIEW, name) or (folder / name).is_symlink():
            return None
        def remove():
            try:
                fd = os.open(folder / name, os.O_RDONLY)
            except FileNotFoundError:
                (folder / 'view.json').unlink(missing_ok=True)
                sync(folder)
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                (folder / 'view.json').unlink()
                sync(folder)
                shutil.rmtree(folder / name)
            except BlockingIOError:
                pass
            finally:
                os.close(fd)
        return remove
    except (OSError, ValueError, KeyError):
        return None


def clone_database(source, target):
    return sys.platform == 'darwin' and subprocess.run(['cp', '-c', str(source), str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def build(home, copy_id=None, *, seam=lambda _: None, free_space=lambda p: shutil.disk_usage(p).free,
          lock_wait=None, clone=clone_database):
    home = Path(home)
    lock_wait = lock_wait or wait_lock
    if copy_id is None:
        root = home / 'cloud/replicas'
        return [build(home, p.name, seam=seam, free_space=free_space, lock_wait=lock_wait, clone=clone)
                for p in sorted(root.iterdir()) if re.fullmatch(COPY_ID, p.name)] if root.exists() else []
    if not re.fullmatch(COPY_ID, copy_id):
        return {'code': 'bad_name'}
    folder = home / 'cloud/replicas' / copy_id
    if folder.is_symlink() or not folder.is_dir():
        return {'code': 'bad_name'}
    result = dict(code='bad_database_block', generation=None, rules=RULES, forgotten_sha256=None, removed_sha256=None)
    lock = os.open(folder / 'view.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not lock_wait(lock, 600):
            raise Refused('view_busy')
        _sweep(folder, temporary_only=True)
        for attempt in range(3):
            generation = (folder / 'current').read_text('ascii').strip('\n')
            if not re.fullmatch(r'[1-9][0-9]*', generation):
                raise Refused('bad_name')
            result['generation'] = int(generation)
            source = folder / ('gen-' + generation)
            if source.is_symlink():
                raise Refused('bad_name')
            with ExitStack() as stack:
                seam('source')
                try:
                    held = os.open(source, os.O_RDONLY)
                except FileNotFoundError:
                    continue
                stack.callback(os.close, held)
                fcntl.flock(held, fcntl.LOCK_SH)
                seam('locked')
                if not source.is_dir() or generation != (folder / 'current').read_text('ascii').strip('\n'):
                    continue
                seam('generation')
                local, raw = removed_set(home), removed_bytes(folder)
                removed = json.loads(raw) if raw else {'format': 1, 'session_ids': [], 'row_ids': []}
                if removed.get('format') != 1 or not isinstance(removed['session_ids'], list) or not isinstance(removed['row_ids'], list) or not all(isinstance(s, str) for s in removed['session_ids']) or not all(type(n) is int and n > 0 for n in removed['row_ids']):
                    raise Refused('bad_name')
                sessions = local | set(removed['session_ids'])
                result.update(forgotten_sha256=forgotten_digest(local), removed_sha256=hashlib.sha256(raw).hexdigest())
                database = source / 'lcm.db'
                guard(database)
                with closing(connect(database, sessions)) as conn:
                    tables, conditions, matches = plan(conn, sessions, removed['row_ids'])
                    kept, omitted = payload_plan(source, conn, sessions, set(removed['row_ids']))
                identity = hashlib.sha256(json.dumps([matches, omitted], sort_keys=True).encode()).hexdigest()[:16]
                name = f'view-{generation}-{RULES}-{identity}'
                target = folder / name
                _sweep(folder, keep=name)
                pointer = dict(format=1, generation=int(generation), rules=RULES, folder=name,
                               forgotten_sha256=result['forgotten_sha256'], removed_sha256=result['removed_sha256'])
                seam('matched')
                if target.is_dir() and not target.is_symlink():
                    seam('reuse')
                else:
                    filtering = any(matches.values()) or bool(omitted)
                    linking = not filtering and clean(database)
                    temp = folder / (name + '.tmp-' + uuid.uuid4().hex)
                    temp.mkdir(mode=0o700)
                    stack.callback(shutil.rmtree, temp, ignore_errors=True)
                    output, payloads = temp / 'lcm.db', temp / 'lcm-large-outputs'
                    payloads.mkdir(mode=0o700)
                    with ExitStack() as space:
                        copying = []
                        for item in kept:
                            try:
                                os.link(source / 'lcm-large-outputs' / item, payloads / item)
                            except OSError:
                                copying.append(item)
                        retirement = retire(folder, pointer)
                        if not linking or copying or retirement:
                            budget = os.open(folder.parent / 'space.lock', os.O_CREAT | os.O_RDWR, 0o600)
                            space.callback(os.close, budget)
                            if not lock_wait(budget, 600):
                                raise Refused('view_busy')
                        if not linking or copying:
                            extra = sum((source / 'lcm-large-outputs' / item).stat().st_size for item in kept) if copying else 0
                            if free_space(folder) < FLOOR + 2 * database.stat().st_size + extra:
                                raise Refused('view_disk_floor')
                        if retirement:
                            retirement()
                        if linking:
                            os.link(database, output)
                        elif not clean(database):
                            with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as src, closing(sqlite3.connect(output)) as dst:
                                src.backup(dst)
                        elif not clone(database, output):
                            shutil.copyfile(database, output)
                        seam('database')
                        if not linking:
                            with closing(connect(output, sessions, False)) as conn:
                                conn.execute('PRAGMA journal_mode=DELETE')
                                _, predicates, _ = plan(conn, sessions, removed['row_ids'])
                                metadata = [(k, v) for k, v in conn.execute('SELECT key,value FROM metadata WHERE NOT (' + predicates['metadata'] + ')')]
                                for table, predicate in predicates.items():
                                    conn.execute(f'DELETE FROM {quote(table)} WHERE {predicate}')
                                conn.executemany('INSERT OR IGNORE INTO metadata VALUES (?,?)', metadata)
                                if any(conn.execute(f'SELECT 1 FROM {quote(t)} WHERE {p} LIMIT 1').fetchone() for t, p in predicates.items()):
                                    raise Refused('view_unsupported_key')
                                validate_keys(conn, predicates, sessions)
                                if filtering and any(TABLES.get(t, 'unsupported') == 'unsupported' and not t.startswith('sqlite_') and conn.execute(f'SELECT 1 FROM {quote(t)} LIMIT 1').fetchone() for t in tables):
                                    raise Refused('view_unsupported_table')
                                conn.commit()
                                if conn.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                                    raise Refused('bad_database_block')
                        seam('deleted')
                        if not clean(output):
                            raise Refused('bad_database_block')
                        for item in copying:
                            shutil.copyfile(source / 'lcm-large-outputs' / item, payloads / item)
                        for item in kept:
                            sync(payloads / item)
                        seam('payloads')
                        sync(output)
                        sync(payloads)
                        sync(temp)
                        seam('synced')
                if local != removed_set(home) or raw != removed_bytes(folder) or generation != (folder / 'current').read_text('ascii').strip('\n'):
                    _sweep(folder, keep=name)
                    continue
                seam('rechecked')
                if not target.exists():
                    temp.rename(target)
                    sync(folder)
                seam('renamed')
                atomic(folder, 'view.json', pointer)
                seam('published')
                result['code'] = 'ok'
                _sweep(folder)
                return result
        raise Refused('view_busy')
    except Refused as exc:
        result['code'] = exc.code
        return result
    except InterruptedError:
        raise
    except Exception:
        result['code'] = 'bad_database_block'
        return result
    finally:
        if result['forgotten_sha256'] is None:
            try:
                result.update(forgotten_sha256=forgotten_digest(removed_set(home)),
                              removed_sha256=removed_sha256(folder))
            except Exception:
                pass
        atomic(folder, 'view-result.json', dict(result, ended_at=time.time()))
        os.close(lock)
