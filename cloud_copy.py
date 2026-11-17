"""Read-only origin snapshots and create-only hosted copies, format 1.

Guard encoding: canonical JSON row hashes preserve SQL types (blobs use base64).
Sessions carry sorted message [id, hash] pairs, binding/relation row hashes and a
combined digest, plus ledger [identity, line_digest, raw_sha256, held_ids] entries.
Identity excludes event_id; raw_from is resolved before hashing. Payload hashes
cover the decoded content string. Id sets are sorted delta unsigned varints,
zlib-compressed and base64 encoded. The injected guard authors cumulative removed
sets and the manifest's top-level pins.
No origin mutation or capture-lock access is permitted here. Exit-2 refusals include
bad_head (otherwise malformed head) and payload_folder_configured (moved payloads).
"""
import base64
from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
import zlib

from . import cloud_delta

DEFAULT_PART_SIZE = 33554432
CONTRACT_VERSION = 1
CONTRACT_DIGEST = 'sha256:76644a990e50b9613697704dcb05b349a952d44ea22187bd768b0eab9a9badc9'
SCHEMA_VERSION = 5
OWNER_KEY = 'capture:owner'
FORGOTTEN_PREFIX = 'capture:forgotten:'
PAYLOAD_FOLDER = 'lcm-large-outputs'
EXTERNALIZED_REF_RE = re.compile(r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]")
INGEST_PLACEHOLDER_RE = re.compile(r'\[Externalized LCM ingest payload:.*?;\s*ref=([^;\]\s]+)\]')
GENERATED_PAYLOAD_REF_RE = re.compile(r'\d{8}_\d{6}_.+_[0-9a-f]{12}_[0-9a-f]+\.json')
_COPY = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
_TOKEN = r'[0-9a-f]{32}'
_OBJECT = r'(?:parts|deltas|payloads)/[0-9a-f]{64}\.' + _TOKEN
Refused = cloud_delta.DeltaRefused
DISK_MARGIN = 1024 * 1024
SKIPS = {'space_busy', 'lock_held', 'not_quiet', 'wal_without_sidecars', 'disk_floor', 'forgotten_during_publish', 'unchanged'}
TABLE_COLUMNS = {
    'capture_events': ['event_id', 'session_id', 'lineage_key', 'source_kind', 'generation', 'position', 'uid', 'entry_type', 'entry_class', 'line_digest', 'content_digest', 'disposition', 'store_ids', 'notes', 'raw', 'raw_from', 'projection_version', 'captured_at'],
    'host_uid_bindings': ['store_id', 'uid', 'lineage_key', 'kind', 'binding_version', 'proof_kind', 'created_at', 'first_check', 'disagree_seen'],
    'ingest_views': ['lineage', 'view_seq', 'kind', 'state', 'entries', 'prefix_chain'],
    'lcm_lifecycle_state': ['conversation_id', 'current_session_id', 'last_finalized_session_id', 'current_frontier_store_id', 'last_finalized_frontier_store_id', 'debt_kind', 'debt_size_estimate', 'current_bound_at', 'last_finalized_at', 'debt_updated_at', 'last_maintenance_attempt_at', 'last_rollover_at', 'last_reset_at', 'updated_at'],
    'lcm_migration_state': ['step_name', 'completed_at'],
    'ledger_emissions': ['emission_id', 'lineage', 'view_seq', 'output_index', 'descriptor_digest', 'kind', 'source_store_ids', 'state', 'next_occ_id', 'observed_stamp', 'recovered_store_id'],
    'ledger_meta': ['key', 'value'],
    'lineage_marks': ['lineage', 'max_store_id', 'legacy_proof_digest', 'max_leaf_id', 'stamp_policy'],
    'message_relations': ['relation_id', 'store_id', 'kind', 'related_store_id', 'ordinal', 'observed_at', 'created_at'],
    'messages': ['store_id', 'session_id', 'source', 'conversation_id', 'role', 'content', 'tool_call_id', 'tool_calls', 'tool_name', 'timestamp', 'token_estimate', 'pinned', 'ingested_at', 'observed_at', 'observed_at_source', 'sender_id', 'sender_source'],
    'metadata': ['key', 'value'],
    'occurrence_bindings': ['occ_id', 'lineage', 'ordinal', 'store_id', 'identity_digest', 'stamp', 'disposition', 'tag', 'key_rank', 'view_seq'],
    'session_lineage': ['child_session_id', 'parent_session_id', 'child_subagent_id', 'parent_subagent_id', 'child_role', 'source', 'created_at'],
    'stamp_observations': ['lineage', 'occ_id', 'stamp', 'kind', 'view_seq', 'source'],
    'summary_nodes': ['node_id', 'session_id', 'depth', 'summary', 'token_count', 'source_token_count', 'source_ids', 'source_type', 'created_at', 'earliest_at', 'latest_at', 'expand_hint'],
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      default=lambda v: {'blob': base64.b64encode(v).decode('ascii')}).encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _name(value, pattern):
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise Refused('bad_name')
    return value


def _payload_name(value):
    try:
        size = len(value.encode('utf-8')) if isinstance(value, str) else 0
    except UnicodeError:
        return False
    return (isinstance(value, str) and value not in ('', '.', '..') and size <= 255
            and not any(c in value for c in '/\\\0'))


def _mkdir(path):
    if not path.parent.exists():
        _mkdir(path.parent)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _write(path, data):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def check_key(prefix, key, *, listing=False):
    """Validate the shared hosted prefix and one object or folder key."""
    _name(prefix, r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*')
    _name(prefix.rsplit('/', 1)[-1], _COPY)
    suffix = key.removeprefix(prefix + '/') if isinstance(key, str) else ''
    grammar = r'(?:parts|deltas|payloads|gen)/' if listing else _OBJECT + r'|gen/[1-9][0-9]*\.json'
    if not isinstance(key, str) or not key.startswith(prefix + '/'):
        raise Refused('bad_name')
    _name(suffix, grammar)
    return key


class DirTransport:
    """Keys include the configured prefix and copy id; listings return full keys."""
    def __init__(self, root, prefix, *, clock=time.time):
        self.root, self.prefix, self.clock = Path(root).resolve(), prefix, clock
        check_key(prefix, f'{prefix}/gen/', listing=True)

    def _path(self, key, listing=False):
        check_key(self.prefix, key, listing=listing)
        path = self.root / key
        if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(self.root)):
            raise Refused('bad_name')
        return path

    def get(self, key):
        return self._path(key).read_bytes()

    def stat(self, key):
        try:
            info = self._path(key).stat()
            return {'bytes': info.st_size, 'time': info.st_mtime}
        except FileNotFoundError:
            return None

    def list(self, prefix):
        folder = self._path(prefix, listing=True)
        if not folder.exists():
            return []
        result = []
        for entry in os.scandir(folder):
            if entry.name.startswith('.'):
                continue
            key = prefix + entry.name
            self._path(key)
            if not entry.is_file(follow_symlinks=False):
                raise Refused('bad_name')
            result.append(key)
        return sorted(result)

    def put_new(self, key, data):
        path = self._path(key)
        _mkdir(path.parent)
        fd, temp = tempfile.mkstemp(prefix='.upload-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                if hasattr(data, 'read'):
                    shutil.copyfileobj(data, handle)
                else:
                    handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)

    def now(self):
        return self.clock()


def encode_ids(ids):
    data, previous = bytearray(), 0
    for value in sorted(set(ids)):
        if type(value) is not int or value < previous:
            raise ValueError('invalid id')
        delta, previous = value - previous, value
        while delta >= 128:
            data.append((delta & 127) | 128)
            delta >>= 7
        data.append(delta)
    return base64.b64encode(zlib.compress(data)).decode('ascii')


def decode_ids(encoded):
    result, value, shift, previous = [], 0, 0, 0
    for byte in zlib.decompress(base64.b64decode(encoded, validate=True)):
        value |= (byte & 127) << shift
        if byte < 128:
            previous += value
            result.append(previous)
            value, shift = 0, 0
        else:
            shift += 7
    if shift:
        raise ValueError('truncated id')
    return result


def _ro(path):
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _rows(conn, table, columns=None):
    cols = ','.join('"' + c + '"' for c in (columns or TABLE_COLUMNS[table]))
    return [dict(r) for r in conn.execute(f'SELECT {cols} FROM "{table}"')]


def guard_block(db, payloads, parent=None):
    parent = parent or {}
    with closing(_ro(db)) as conn:
        tables, sessions, owner = _tables(conn), {}, {}
        def session(name):
            return sessions.setdefault(name, {'ids': [], 'messages': [], 'bindings': [], 'relations': [], 'ledger': []})
        for row in _rows(conn, 'messages'):
            sid, ident = row['session_id'], row['store_id']
            owner[ident] = sid
            session(sid)['ids'].append(ident)
            session(sid)['messages'].append([ident, _sha(_json(row))])
        for table, field, cols in (('host_uid_bindings', 'bindings', ['store_id', 'uid', 'lineage_key', 'kind']),
                                    ('message_relations', 'relations', TABLE_COLUMNS['message_relations'])):
            if table in tables:
                for row in _rows(conn, table, cols):
                    ends = {owner.get(row['store_id'], '')}
                    if field == 'relations':
                        ends.add(owner.get(row['related_store_id'], ''))
                    for sid in ends:
                        session(sid)[field].append(_sha(_json(row)))
        events = _rows(conn, 'capture_events') if 'capture_events' in tables else []
        by_id = {r['event_id']: r for r in events}
        for row in events:
            raw, visited = row, set()
            while raw['raw_from'] is not None:
                if raw['event_id'] in visited or raw['raw_from'] not in by_id:
                    raise ValueError('invalid ledger raw')
                visited.add(raw['event_id'])
                raw = by_id[raw['raw_from']]
            key = ([row['lineage_key'], row['generation'], row['position']] if row['source_kind'] == 'line'
                   else [row['lineage_key'], row['uid'], row['content_digest'], row['position']])
            try:
                digest = _sha(zlib.decompress(raw['raw'])) if raw['raw'] is not None else None
            except zlib.error:
                raise ValueError('invalid ledger raw') from None
            session(row['session_id'])['ledger'].append([row['source_kind'], key, row['line_digest'], digest,
                                                        row['store_ids'] if row['disposition'] == 'held' else None])
        for data in sessions.values():
            for field in ('messages', 'bindings', 'relations', 'ledger'):
                data[field].sort(key=_json)
            data['digest'] = _sha(_json({k: data[k] for k in ('messages', 'bindings', 'relations')}))
            data['ids'] = encode_ids(data['ids'])
        seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='messages'").fetchone() if 'sqlite_sequence' in tables else None
        mark = max(parent.get('mark', 0), seq[0] if seq else 0, max(owner, default=0))
    hashes = {}
    for path in sorted(Path(payloads).iterdir()):
        if not _payload_name(path.name) or path.is_symlink() or not path.is_file():
            raise Refused('bad_name')
        content = json.loads(path.read_text(encoding='utf-8'))['content']
        hashes[path.name] = _sha(content.encode('utf-8'))
    return dict(mark=mark, sessions=sessions, payloads=hashes,
                **{k: parent.get(k, [] if k != 'removed_ids' else encode_ids([]))
                   for k in ('removed_sessions', 'removed_ids')})


def _header_wal(db):
    with db.open('rb') as handle:
        handle.seek(18)
        return any(b >= 2 for b in handle.read(2))


def _check_wal(home):
    db = home / 'lcm.db'
    if _header_wal(db) and any(not (home / ('lcm.db-' + s)).exists() for s in ('wal', 'shm')):
        raise Refused('wal_without_sidecars')


def _live_ro(home):
    _check_wal(home)
    return _ro(home / 'lcm.db')


def _quiet(home, quiet_s, clock):
    db, spool = home / 'lcm.db', home / 'spool'
    pending = spool.exists() and any(not p.name.startswith('.') and p.name.endswith(('.json', '.claimed')) for p in spool.iterdir())
    return not (pending or _header_wal(db) or (home / 'lcm.db-wal').exists()
                or (home / 'lcm.db-journal').exists() or clock() - db.stat().st_mtime < quiet_s)


def wait_lock(fd, seconds):
    deadline = time.monotonic() + seconds
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


@contextmanager
def space_lock(home):
    root = home / 'cloud' / 'replicas'
    _mkdir(root)
    fd = os.open(root / 'space.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not wait_lock(fd, 600):
            raise Refused('space_busy')
        yield
    finally:
        os.close(fd)


def _before(home, quiet_s, clock, free_bytes):
    db = home / 'lcm.db'
    if (home / 'lcm.db.records').exists():
        raise Refused('records_pair_unsupported')
    _check_wal(home)
    env = home / '.env'
    if env.is_file():
        raw = env.read_bytes()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw.removeprefix(b'\xef\xbb\xbf').decode('latin-1')
        for line in re.split(r'\r\n|\n|\r', text):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if 'LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH' not in line:
                continue
            key, _, value = line.removeprefix('export ').partition('=')
            if key.strip() != 'LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH':
                raise Refused('payload_folder_configured')
            value = value.strip()
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            path = Path(os.path.expanduser(value))
            if value and ('#' in value or '\\' in value or '\x00' in value or not path.is_absolute() or path.resolve() != (home / PAYLOAD_FOLDER).resolve()):
                raise Refused('payload_folder_configured')
    try:
        wal_bytes = (home / 'lcm.db-wal').stat().st_size
    except FileNotFoundError:
        wal_bytes = 0
    folder = home / PAYLOAD_FOLDER
    payload_bytes = sum(p.stat().st_size for p in folder.iterdir() if p.is_file() and not p.is_symlink()) if folder.exists() else 0
    staging, staging_bytes = home / 'cloud' / 'staging', 0
    if staging.is_dir() and not staging.is_symlink():
        for root, _, files in os.walk(staging, followlinks=False):
            for name in files:
                path = Path(root) / name
                if path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1:
                    staging_bytes += path.stat().st_size
    if free_bytes(home) + staging_bytes < db.stat().st_size + wal_bytes + payload_bytes + DISK_MARGIN:
        raise Refused('disk_floor')
    if quiet_s > 0 and not _quiet(home, quiet_s, clock):
        raise Refused('not_quiet')


def _after_backup():
    """Injection seam after backup, before the second quiet check."""


def _unit():
    package = Path(__file__).resolve().parent
    bundle, version = {}, None
    path = package.parent / 'bundle.json'
    if path.is_file() and not path.is_symlink():
        bundle = json.loads(path.read_text(encoding='utf-8'))
    path = package / 'plugin.yaml'
    if path.is_file():
        match = re.search(r'^version:\s*[\'\"]?([^\s\'\"]+)', path.read_text(encoding='utf-8'), re.M)
        version = match[1] if match else None
    return dict(version=bundle.get('unit'), core_commit=bundle.get('core_commit'), core_version=version)


def _references(conn, home, staging):
    refs = {}
    try:
        for row in conn.execute('SELECT session_id,content,tool_calls FROM messages'):
            for text in (row['content'], row['tool_calls']):
                if isinstance(text, str):
                    for pattern in (EXTERNALIZED_REF_RE, INGEST_PLACEHOLDER_RE):
                        for match in pattern.finditer(text):
                            refs.setdefault(match[1], set()).add(row['session_id'])
    except sqlite3.OperationalError:
        raise Refused('undecodable_text') from None
    payloads, minted, quoted = [], [], []
    folder = staging / PAYLOAD_FOLDER
    _mkdir(folder)
    for name, sessions in sorted(refs.items()):
        valid = _payload_name(name)
        source = home / PAYLOAD_FOLDER / name if valid else None
        if valid and not source.is_symlink() and source.is_file():
            target = folder / name
            try:
                os.link(source, target)
            except OSError:
                with source.open('rb') as handle:
                    _write(target, handle.read())
            data = target.read_bytes()
            payloads.append(dict(name=name, sha256=_sha(data), bytes=len(data)))
        else:
            rows = minted if valid and GENERATED_PAYLOAD_REF_RE.fullmatch(name) else quoted
            rows.append(dict(name=name, sessions=sorted(sessions)))
    return payloads, minted, quoted


def _stage(home, copy_id, generation, head, head_bytes, clock, progress):
    staging = home / 'cloud' / 'staging'
    if staging.exists():
        shutil.rmtree(staging)
    _mkdir(staging)
    signature = cloud_delta.store_signature(home)
    db = staging / 'lcm.db'
    _write(db, b'')
    with closing(_live_ro(home)) as source, closing(sqlite3.connect(db)) as target:
        source.backup(target, pages=16, progress=progress)
    _after_backup()
    with closing(sqlite3.connect(db)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=DELETE')
        tables = _tables(conn)
        meta = dict(conn.execute('SELECT key,value FROM metadata'))
        schema = int(meta.get('schema_version', 0))
        if schema > SCHEMA_VERSION:
            raise Refused('schema_newer')
        owner = json.loads(meta[OWNER_KEY])
        projections = sorted({r[0] for r in conn.execute('SELECT projection_version FROM capture_events')}) if 'capture_events' in tables else []
        maximum = conn.execute('SELECT coalesce(max(store_id),0) FROM messages').fetchone()[0]
        stamp = dict(format=1, origin_id=owner['store_uuid'], copy_id=copy_id, origin_harness=owner['host'],
                     generation=generation, parent=None if head is None else dict(generation=head['generation'], manifest_sha256=_sha(head_bytes)),
                     contract=dict(version=CONTRACT_VERSION, digest=CONTRACT_DIGEST), schema_version=schema,
                     projection_versions=projections, unit=_unit(), max_store_id=maximum, created_at=clock())
        conn.execute("INSERT INTO metadata(key,value) VALUES ('import:origin',?)", (_json(stamp).decode('utf-8'),))
        conn.commit()
        if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise sqlite3.DatabaseError('integrity check failed')
        page_size = conn.execute('PRAGMA page_size').fetchone()[0]
        payloads, minted, quoted = _references(conn, home, staging)
        counts = {name: conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0] for name in TABLE_COLUMNS if name in tables}
        seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='messages'").fetchone() if 'sqlite_sequence' in tables else None
    manifest = dict(layout=1, copy_id=copy_id, generation=generation, created_at=stamp['created_at'], stamp=stamp,
                    payloads=payloads, unresolved_minted=minted, unresolved_quoted=quoted, tables=counts,
                    sequence={'messages': seq[0] if seq else None}, publisher=None,
                    pins=(head or {}).get('pins', []), superseded=(head or {}).get('superseded', []))
    manifest['guard'] = guard_block(db, staging / PAYLOAD_FOLDER, (head or {}).get('guard'))
    return manifest, signature, page_size


@contextmanager
def _locked(home):
    cloud = home / 'cloud'
    _mkdir(cloud)
    fd = os.open(cloud / 'upload.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused('lock_held') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _state(home, copy_id):
    try:
        value = json.loads((home / 'cloud' / 'state.json').read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError()
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        raise Refused('state_unreadable') from None
    return value if value.get('copy_id') == copy_id else {}


def _save_state(home, value):
    path = home / 'cloud' / 'state.json'
    temp = path.with_name('state.' + os.urandom(16).hex())
    try:
        _write(temp, _json(value))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _update_state(home, copy_id, *, consumed_decisions=(), **keys):
    """Merge named keys under the caller's upload.lock; preserve unknown state."""
    value = _state(home, copy_id)
    if consumed_decisions:
        value['unpublished_decisions'] = [d for d in value.get('unpublished_decisions', [])
                                          if d['id'] not in consumed_decisions]
    _save_state(home, value | keys | dict(copy_id=copy_id))


def _head(transport, prefix, copy_id):
    keys = transport.list(prefix + '/gen/')
    for key in keys:
        _name(key, re.escape(prefix) + r'/gen/[1-9][0-9]*\.json')
    if not keys:
        return None, b''
    key = max(keys, key=lambda k: int(k.rsplit('/', 1)[-1][:-5]))
    data = transport.get(key)
    generation = int(key.rsplit('/', 1)[-1][:-5])
    try:
        head = json.loads(data)
        if not isinstance(head, dict):
            raise Refused('bad_database_block', generation=generation)
        if type(head['layout']) is not int or head['layout'] < 1:
            raise ValueError()
        if head['layout'] > 1:
            raise Refused('newer_layout')
        version = head['stamp']['contract']['version']
        if type(version) is not int or version < 1:
            raise ValueError()
        if version > CONTRACT_VERSION:
            raise Refused('replica_newer_contract')
        if not isinstance(head.get('database'), dict):
            raise Refused('bad_database_block', generation=generation)
        if type(head['generation']) is not int or not isinstance(head['copy_id'], str):
            raise ValueError()
        if head['copy_id'] != copy_id or head['generation'] != generation:
            raise Refused('bad_name')
        if not isinstance(head['payloads'], list):
            raise ValueError()
        for row in head['payloads']:
            if not _payload_name(row['name']):
                raise Refused('bad_name')
            _name(row['key'], r'payloads/[0-9a-f]{64}\.' + _TOKEN)
            _name(row['sha256'], r'[0-9a-f]{64}')
            if type(row['bytes']) is not int or row['bytes'] < 0:
                raise ValueError()
        parent_guard = head.get('guard', {})
        if not isinstance(parent_guard, dict) or type(parent_guard.get('mark', 0)) is not int:
            raise ValueError()
        for field in ('pins', 'superseded', 'forgotten'):
            if not isinstance(head.get(field, []), list):
                raise ValueError()
        for item in head.get('forgotten', []):
            if not all(isinstance(item[k], str) for k in ('kind', 'key')):
                raise ValueError()
        # Apply the codec's existing F10 validation before any staging work.
        cloud_delta._shape(head['database'], generation)
    except Refused as error:
        error.detail['generation'] = generation
        raise
    except (ValueError, TypeError, KeyError, AttributeError):
        raise Refused('bad_head', generation=generation) from None
    return head, data


def _read_part(path, offset, size):
    with path.open('rb') as handle:
        handle.seek(offset)
        return handle.read(size)


def publish(home, copy_id, transport, prefix, *, quiet_s=0, part_size=DEFAULT_PART_SIZE,
            guard=None, forget_items=None, publisher=None, chain_check=None, after_create=None, prepare_manifest=None,
            token=None, timer=False, clock=time.time,
            free_bytes=lambda p: shutil.disk_usage(p).free, progress=None,
            size_factor=cloud_delta.SIZE_FACTOR, max_generations=cloud_delta.MAX_CHAIN_GENERATIONS,
            snapshot_only=False, result=None):
    """Return a manifest; refusals and skips raise Refused with a content-free code.

    snapshot_only stages a generation-1 base without transport or page/state writes.
    A guard receives (parent manifest, staged database, candidate manifest).
    result receives printable status fields after success, outside the manifest.
    """
    _name(copy_id, _COPY)
    _name(prefix, r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*')
    if prefix.rsplit('/', 1)[-1] != copy_id or type(part_size) is not int or part_size <= 0 or quiet_s < 0:
        raise Refused('bad_name')
    explicit_token = token is not None
    resumed = explicit_token
    token = token or _sha(os.urandom(16) + str(time.time_ns()).encode() + str(os.getpid()).encode())[:32]
    _name(token, _TOKEN)
    home = Path(home)
    with _locked(home):
        staging_started = False
        try:
            config = home / 'cloud' / 'cloud.json'
            if config.exists() and json.loads(config.read_text(encoding='utf-8')).get('copy_id') != copy_id:
                raise Refused('bad_name')
            state, blocker = _state(home, copy_id), None
            if not snapshot_only and any(fn is None for fn in (guard, forget_items, publisher, chain_check, after_create, prepare_manifest)):
                staging_started = True
                raise Refused('guard_not_installed')
            if timer and not snapshot_only:
                pair = state.get('last_pair')
                try:
                    listed = transport.list(prefix + '/gen/')
                    for key in listed:
                        _name(key, re.escape(prefix) + r'/gen/[1-9][0-9]*\.json')
                    head_generation = max((int(k.rsplit('/', 1)[1][:-5]) for k in listed), default=0)
                except OSError:
                    head_generation = None
                blocker = cloud_delta.skip_blocker(manual=False, signature=cloud_delta.store_signature(home),
                    recorded_signature=state.get('snapshot_signature'), decision_pending=bool(state.get('unpublished_decisions')),
                    last_pair=pair, pending_pair=state.get('pending_pair'), retention_done_generation=state.get('retention_done_generation'),
                    head_generation=head_generation, last_publish_s=state.get('last_publish_s'), now_s=clock())
                if blocker is None:
                    raise Refused('unchanged')
            head, head_bytes = (None, b'') if snapshot_only else _head(transport, prefix, copy_id)
            generation = head['generation'] + 1 if head else 1
            publisher_id = None if snapshot_only else publisher()
            bucket_now = None if snapshot_only else transport.now()
            pending = state.get('pending_pair')
            started = state.get('pending_started')
            if not explicit_token and pending and pending[0] == generation and bucket_now is not None and started is not None and 0 <= bucket_now - started <= 3.5 * 86400:
                token, resumed = pending[1], True
                _name(token, _TOKEN)
            elif not snapshot_only and pending and pending[0] == generation and explicit_token and token == pending[1]:
                if bucket_now is None or started is None or bucket_now - started > 3.5 * 86400:
                    token, resumed = _sha(os.urandom(16) + str(time.time_ns()).encode() + str(os.getpid()).encode())[:32], False
            pages_dir, db = home / 'cloud' / 'pages', home / 'cloud' / 'staging' / 'lcm.db'
            for restart in range(3):
                with space_lock(home):
                    _before(home, quiet_s, clock, free_bytes)
                    staging_started = True
                    m, signature, page_size = _stage(home, copy_id, generation, head, head_bytes, clock, progress)
                if quiet_s > 0 and not _quiet(home, quiet_s, clock):
                    shutil.rmtree(db.parent)
                    raise Refused('not_quiet')
                frozen = (db.stat().st_size, db.stat().st_mtime_ns)
                new = cloud_delta.hash_pages(db, expected_page_size=page_size)
                old = head['database'] if head else None
                usable = old and type(old.get('page_size')) is int and old['page_size'] > 0 and type(old.get('file_bytes')) is int and isinstance(old.get('page_list_sha256'), str)
                prev = cloud_delta.load_head_list(pages_dir, copy_id, old['page_list_sha256'], old['file_bytes']//old['page_size']) if usable else None
                items = forget_items(head, db) if forget_items else []
                if not snapshot_only:
                    recovered = chain_check(head, state, publisher_id, head_bytes)
                args = dict(head_generation=head['generation'] if head else None, page_size=page_size, file_bytes=new.file_bytes,
                            list_matches=prev is not None, forget=any(i['chain_may_hold'] for i in items), object_missing=False,
                            delta_bytes=0, size_factor=size_factor, max_generations=max_generations)
                try:
                    reason = cloud_delta.base_reason(old, **args)
                except Refused as error:
                    error.detail['generation'] = head['generation']
                    raise
                m['publisher'] = publisher_id
                if head and not snapshot_only:
                    guard(head, db, m)
                encoded = []
                if reason is None:
                    count = len(cloud_delta.changed_pages(prev, new.digests))
                    cap = cloud_delta.max_records(page_size)
                    with space_lock(home):
                        if free_bytes(home) < count * (4 + page_size) + 16 * ((count + cap - 1)//cap) + DISK_MARGIN:
                            raise Refused('disk_floor')
                        _mkdir(db.parent / 'deltas')
                        encoded = cloud_delta.encode_delta(db, prev, new, db.parent / 'deltas')
                    args['delta_bytes'] = sum(o.bytes for o in encoded)
                    reason = cloud_delta.base_reason(old, **args)
                if not snapshot_only:
                    _update_state(home, copy_id, pending_pair=[generation, token],
                                  pending_started=started if resumed and started is not None else bucket_now,
                                  **(recovered or {}))
                    cloud_delta.save_pending_list(pages_dir, copy_id, new.digests)
                sent, buffers = set(), {}
                def upload(key, data, force=False):
                    _name(key, _OBJECT)
                    if _sha(data) != key.split('/')[1][:64]:
                        raise Refused('staging_changed')
                    if snapshot_only or (key in sent and not force):
                        return
                    if resumed and not force:
                        try:
                            if len(transport.get(prefix + '/' + key)) == len(data):
                                sent.add(key)
                                return
                        except FileNotFoundError:
                            pass
                    try:
                        transport.put_new(prefix + '/' + key, data)
                    except FileExistsError:
                        if len(transport.get(prefix + '/' + key)) != len(data):
                            raise Refused('staging_changed') from None
                    sent.add(key)
                previous_payloads = {p['name']: p for p in (head or {}).get('payloads', [])}
                for payload in m['payloads']:
                    data = (db.parent / PAYLOAD_FOLDER / payload['name']).read_bytes()
                    if _sha(data) != payload['sha256']:
                        raise Refused('staging_changed')
                    previous = previous_payloads.get(payload['name'])
                    reused = previous is not None and previous['sha256'] == payload['sha256']
                    key = previous['key'] if reused else f"payloads/{payload['sha256']}.{token}"
                    payload['key'] = key
                    buffers[key] = lambda path=db.parent / PAYLOAD_FOLDER / payload['name']: path.read_bytes()
                    if not reused:
                        upload(key, data)
                database = {k: getattr(new, k) for k in ('page_size', 'file_bytes', 'file_sha256', 'page_list_sha256')}
                def base():
                    previous = {r['key'].split('/')[1][:64]: r['key'] for r in old['base']['parts']} if old else {}
                    refs, whole = [], hashlib.sha256()
                    with db.open('rb') as handle:
                        while data := handle.read(part_size):
                            digest = _sha(data)
                            whole.update(data)
                            key = previous.get(digest, f'parts/{digest}.{token}')
                            refs.append(dict(key=key, bytes=len(data)))
                            buffers[key] = lambda offset=handle.tell()-len(data), size=len(data): _read_part(db, offset, size)
                            if key not in previous.values():
                                upload(key, data)
                    if whole.hexdigest() != new.file_sha256:
                        raise Refused('staging_changed')
                    database.update(base=dict(generation=generation, file_bytes=new.file_bytes, file_sha256=new.file_sha256,
                                              reason=reason, part_size=part_size, parts=refs), deltas=[])
                if reason is not None:
                    base()
                else:
                    refs = []
                    for obj in encoded:
                        data = obj.path.read_bytes()
                        if len(data) != obj.bytes or _sha(data) != obj.sha256:
                            raise Refused('staging_changed')
                        key = f'deltas/{obj.sha256}.{token}'
                        buffers[key] = lambda path=obj.path: path.read_bytes()
                        upload(key, data)
                        refs.append(dict(key=key, bytes=obj.bytes, records=obj.records))
                    database.update(base=old['base'], deltas=old['deltas'] + [dict(generation=generation, file_pages=new.pages, objects=refs)])
                for check in range(2):
                    present = set()
                    if not snapshot_only:
                        for folder in ('parts', 'deltas', 'payloads'):
                            listed = transport.list(prefix + '/' + folder + '/')
                            for key in listed:
                                _name(key, re.escape(prefix) + '/' + _OBJECT)
                            present.update(key[len(prefix)+1:] for key in listed)
                    refs = database['base']['parts'] + [r for e in database['deltas'] for r in e['objects']] + m['payloads']
                    missing = [r for r in refs if r['key'] not in present] if not snapshot_only else []
                    old_missing = reason is None and any(r['key'] not in buffers for r in missing)
                    if old_missing:
                        args['object_missing'] = True
                        reason = cloud_delta.base_reason(old, **args)
                        base()
                        continue
                    for ref in missing:
                        key = ref['key']
                        replacement = key.split('.')[0] + '.' + token
                        data = buffers[key]()
                        if len(data) != ref['bytes']:
                            raise Refused('staging_changed')
                        upload(replacement, data, force=True)
                        ref['key'] = replacement
                    break
                if (db.stat().st_size, db.stat().st_mtime_ns) != frozen:
                    raise Refused('staging_changed')
                with closing(_live_ro(home)) as live, closing(_ro(db)) as staged:
                    forgotten = {r[0][len(FORGOTTEN_PREFIX):] for r in live.execute('SELECT key FROM metadata') if r[0].startswith(FORGOTTEN_PREFIX)}
                    query = 'SELECT session_id FROM messages'
                    if 'capture_events' in _tables(staged):
                        query += ' UNION SELECT session_id FROM capture_events'
                    sessions = {r[0] for r in staged.execute(query)}
                final_items = forget_items(head, db) if forget_items else []
                if sessions & forgotten or (reason is None and any(i['chain_may_hold'] for i in final_items)):
                    shutil.rmtree(db.parent)
                    if restart == 2:
                        raise Refused('forgotten_during_publish')
                    continue
                m.update(database=database, attempt=dict(generation=generation, token=token),
                         forgotten=sorted((head or {}).get('forgotten', []) + [
                             {k: i[k] for k in ('key', 'kind', 'session')} | dict(base_generation=generation if reason is not None and i['chain_may_hold'] else None)
                             for i in final_items], key=lambda i: (i['kind'], i['key'])))
                if snapshot_only:
                    return m
                prepare_manifest(m)
                try:
                    transport.put_new(prefix + f'/gen/{generation}.json', _json(m))
                except FileExistsError:
                    raise Refused('chain_conflict') from None
                cloud_delta.promote_pending_list(pages_dir, copy_id)
                changes = after_create(m)
                _update_state(home, copy_id, snapshot_signature=signature, last_publish_s=clock(), **changes)
                if blocker and result is not None:
                    result['blocker'] = blocker
                return m
        finally:
            if staging_started and not snapshot_only:
                shutil.rmtree(home / 'cloud' / 'staging', ignore_errors=True)
