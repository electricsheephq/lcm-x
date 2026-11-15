"""Recall-only records for a host's overwritten queued user messages (I1/I5/I6).

Capture never writes. Canonical ingest only reports its committed object/id map;
records transactions and read cuts never participate in canonical recovery.
"""
from collections import Counter, defaultdict, deque
from contextlib import closing
from dataclasses import dataclass
import hashlib
import logging
import time

from . import ingest_protection as protection
from .store import _records_ro, open_records_writer, read_records, records_path, records_read_cut

logger = logging.getLogger(__name__)
FLOOR_ROWS, FLOOR_BYTES = 64, 1048576
SCAN_ROWS, SCAN_BYTES = 200000, 64 * 1048576
DRAIN_ENTRIES, DRAIN_BYTES, DRAIN_SECONDS = 256, 1048576, .05
_EXIT_ENGINES = set()


def digest(text):
    return hashlib.sha256((text or '').encode('utf-8')).hexdigest()


def count(engine, reason, amount=1):
    if not hasattr(engine, '_queued_counts'):
        engine._queued_counts = Counter()
    engine._queued_counts[reason] += amount


@dataclass(frozen=True)
class Capture:
    turn: str
    binding: tuple
    objects: tuple
    members: tuple  # (raw text, host timestamp, host UID), frozen before host mutation
    clean: str
    created: float


@dataclass(frozen=True)
class Entry:
    identity: str
    path: object
    values: tuple


def binding(engine):
    return engine._store.db_identity, engine._session_id, engine._conversation_id


def capture(engine, payload):
    """Freeze occurrence evidence, not an owner's not-yet-committed identity."""
    count(engine, 'capture_attempt', 0)
    turn = str(payload.get('turn_id') or '')
    if payload.get('session_id') != engine._session_id:
        count(engine, 'capture_binding')
        return
    captures = engine.__dict__.setdefault('_queued_captures', {})
    if turn and turn in captures:
        return
    run = []
    for row in reversed(payload.get('conversation_history') or []):
        if (not isinstance(row, dict) or row.get('role') != 'user' or 'display_kind' in row
                or not isinstance(row.get('content'), str) or engine._is_scaffold_shaped_user_message(row)):
            break
        run.insert(0, row)
    if len(run) < 2:
        count(engine, 'capture_boundary')
        return
    seen = engine.__dict__.setdefault('_queued_seen_heads', deque(maxlen=1024))
    if any(run[0] is old for old in seen):
        count(engine, 'reused_head')
        return
    seen.append(run[0])
    # Legacy adapters omit turn_id; the empty key is scoped to this binding and
    # consumed once after commit. Object evidence still gates the owner.
    captures[turn] = Capture(turn, binding(engine), tuple(run),
        tuple((r['content'], r.get('timestamp'), r.get('message_uid')) for r in run),
        payload.get('user_message'), time.time())


def retire(engine, turn=None):
    captures = engine.__dict__.get('_queued_captures', {})
    if turn:
        captures.pop(str(turn), None)
    else:
        captures.clear()


def blank_member(previous, row):
    """Raw join-member boundaries, including members containing delimiters (3j)."""
    return (row == previous or row.startswith(previous + '\n\n')
            or row.endswith('\n\n' + previous) or '\n\n' + previous + '\n\n' in row)


def backward(canonical, owner, texts):
    used = 0
    rows = canonical.execute("SELECT content FROM messages INDEXED BY idx_msg_session "
        "WHERE session_id=? AND role='user' AND store_id<? ORDER BY store_id DESC",
        (owner['session_id'], owner['store_id']))
    for index, row in enumerate(rows):
        size = len((row[0] or '').encode('utf-8'))
        if index >= FLOOR_ROWS or used + size > FLOOR_BYTES:
            return 'backward_unscanned'
        used += size
        for text in texts:
            if not text.strip():
                if blank_member(text, row[0] or ''):
                    return 'backward_overlap'
            elif {p.strip() for p in text.split('\n\n') if p.strip()} & {
                    p.strip() for p in (row[0] or '').split('\n\n')}:
                return 'joined_overlap' if '\n\n' in text else 'backward_overlap'
    return None


def after_commit(engine, messages, stored_at, *, post_llm=False):
    """Only run after canonical commit; never affect its success/fault handling."""
    for turn, item in list(engine.__dict__.get('_queued_captures', {}).items()):
        if item.binding != binding(engine):
            count(engine, 'binding_changed')
            retire(engine, turn)
            continue
        touched = [(i, sid) for i, sid in stored_at.items()
                   if any(messages[i] is obj for obj in item.objects)]
        if not post_llm:
            if touched:
                count(engine, 'earlier_ingest')
                retire(engine, turn)
            continue
        reason = None
        head = item.objects[0]
        index = next((i for i, row in enumerate(messages) if row is head), None)
        if index not in stored_at:
            reason = 'owner_not_stored'
        elif any(any(row is obj for row in messages) for obj in item.objects[1:]):
            reason = 'run_object_survived'
        elif head.get('_merged_turn_prefix') or head.get('_absorbed_message_uids'):
            reason = 'prefix_host'
        elif not isinstance(item.clean, str) or head.get('content') != item.clean:
            reason = 'head_not_clean'
        joined = ''
        for text, _, _ in item.members:
            joined = joined + '\n\n' + text if joined and text else joined or text
        if reason is None and joined == item.clean:
            reason = 'join_unchanged'
        if reason is None and any(protection.contains_data_uri_base64(t) or protection.contains_long_base64_run(t)
                                  for t, _, _ in item.members):
            reason = 'inline_payload'
        if (reason is None and engine._config.large_output_externalization_enabled
                and protection._non_tool_text_over_floor(joined, engine._config,
                                                        context_window_tokens=engine.context_length)):
            reason = 'whole_row_payload'
        if reason is None:
            with closing(_records_ro(engine._store.db_identity)) as canonical:
                canonical.execute('BEGIN')
                owner = canonical.execute('SELECT * FROM messages WHERE store_id=?', (stored_at[index],)).fetchone()
                reason = 'owner_missing' if owner is None else backward(canonical, owner, [m[0] for m in item.members[:-1]])
                if reason is None:
                    queue = engine.__dict__.setdefault('_queued_entries', deque())
                    for ordinal, (text, stamp, uid) in enumerate(item.members[:-1]):
                        message = dict(role='user', content=text, timestamp=stamp)
                        if engine._matches_ignore_message_patterns(message):
                            count(engine, 'ignore_pattern')
                            continue
                        protected = protection.protect_message_for_ingest(message, config=engine._config,
                            hermes_home=engine._hermes_home, session_id=owner['session_id'],
                            context_window_tokens=engine.context_length)
                        values = dict(owner_store_id=owner['store_id'], owner_timestamp=owner['timestamp'],
                            owner_ingested_at=owner['ingested_at'], owner_content_digest=digest(owner['content']),
                            owner_message_uid=item.members[0][2], ordinal=ordinal, role='user', content=protected['content'],
                            content_digest=digest(protected['content']), observed_at=stamp,
                            observed_at_source='host_message_timestamp', captured_uid=uid,
                            captured_at=item.created, superseded_by=None, session_id=owner['session_id'],
                            conversation_id=owner['conversation_id'], profile=str(item.binding[0]))
                        queue.append(Entry(item.binding[0], records_path(engine._store), tuple(values.items())))
                    if queue:
                        _EXIT_ENGINES.add(engine)
        if reason:
            count(engine, reason)
        retire(engine, turn)


def committed(engine, messages, stored_at):
    try:
        after_commit(engine, messages, stored_at, post_llm=bool(getattr(engine, '_queued_post_llm', False)))
    except Exception:
        count(engine, 'capture_error')
        logger.debug('queued records capture failed', exc_info=True)


def write(engine, entries):
    """Frozen retry keys; equal retries are no-ops, different bytes never replace."""
    with closing(open_records_writer(entries[0].identity)) as writer:
        for entry in entries:
            values = dict(entry.values)
            if writer.path != entry.path:
                raise ValueError('records path changed')
            if writer.connection.execute("SELECT 1 FROM sqlite_master WHERE name='queued_user_rows'").fetchone():
                existing = writer.connection.execute('SELECT * FROM queued_user_rows WHERE owner_store_id=? AND ordinal=?',
                    (values['owner_store_id'], values['ordinal'])).fetchone()
                if existing is not None:
                    names = [r[1] for r in writer.connection.execute('PRAGMA table_info(queued_user_rows)')]
                    if dict(zip(names, existing)) != values:
                        raise ValueError('records retry has different bytes')
                    continue
            refused = writer.insert('queued_user_rows', values)
            if refused:
                raise ValueError(refused)


def drain(engine, *, exiting=False):
    queue = engine.__dict__.get('_queued_entries', ())
    end, used, done = time.monotonic() + DRAIN_SECONDS, 0, 0
    while queue and done < DRAIN_ENTRIES and time.monotonic() < end:
        entry = queue[0]
        size = len(dict(entry.values)['content'].encode('utf-8'))
        if used + size > DRAIN_BYTES:
            break
        try:
            engine._store_queued_user_rows([entry])
        except Exception:
            count(engine, 'drain_failed')
            break
        queue.popleft()
        done, used = done + 1, used + size
    if not queue:
        _EXIT_ENGINES.discard(engine)
    if exiting:
        count(engine, 'exit_kept', len(queue))
        count(engine, 'exit_dropped', 0)
        logger.warning('queued records exit: kept=%d dropped=0 counts=%s', len(queue), dict(engine._queued_counts))


def flush_at_exit():
    for engine in tuple(_EXIT_ENGINES):
        drain(engine, exiting=True)



def reference(record):
    return dict(tool='lcm_expand', args=dict(queued_user_ref=[record['owner_store_id'], record['ordinal']]))


def record_view(row):
    keys = ('owner_store_id', 'ordinal', 'role', 'content', 'observed_at', 'observed_at_source',
            'session_id', 'conversation_id', 'captured_uid', 'superseded_by')
    return dict({key: row[key] for key in keys}, kind='record', reference=reference(row))


def decide(cut, session):
    records, _ = read_records(cut, 'queued_user_rows', session)
    records.sort(key=lambda r: (r['owner_store_id'], r['ordinal']))
    states, pending = {}, defaultdict(list)
    for r in records:
        if r['content'].strip():
            pending[r['content'].strip().split('\n\n')[0]].append(r)
    def check(sid, text, subset=None):
        parts = (text or '').split('\n\n')
        for start, part in enumerate(parts):
            for r in pending.get(part, ()):
                key = r['owner_store_id'], r['ordinal']
                needle = r['content'].strip().split('\n\n')
                if sid > key[0] and (subset is None or key[0] == subset) and parts[start:start + len(needle)] == needle:
                    states[key] = 'overlap'
    # Every owner has a floor of its own, even past the session scan budget.
    floors = {}
    for owner in sorted({r['owner_store_id'] for r in records}):
        used, last, beyond = 0, owner, False
        for index, (sid, text) in enumerate(cut.canonical.execute(
                "SELECT store_id,content FROM messages INDEXED BY idx_msg_session "
                "WHERE session_id=? AND role='user' AND store_id>? ORDER BY store_id", (session, owner))):
            size = len((text or '').encode('utf-8'))
            if index >= FLOOR_ROWS or used + size > FLOOR_BYTES:
                beyond = True
                break
            used, last = used + size, sid
            check(sid, text, owner)
        floors[owner] = last, beyond
    used = rows = 0
    last = min(floors, default=0)
    exhausted = False
    if pending:
        for sid, text in cut.canonical.execute(
                "SELECT store_id,content FROM messages INDEXED BY idx_msg_session "
                "WHERE session_id=? AND role='user' AND store_id>? ORDER BY store_id", (session, last)):
            size = len((text or '').encode('utf-8'))
            if rows >= SCAN_ROWS or used + size > SCAN_BYTES:
                exhausted = True
                break
            rows, used, last = rows + 1, used + size, sid
            check(sid, text)
    for group in pending.values():
        for r in group:
            key = r['owner_store_id'], r['ordinal']
            if key not in states and exhausted and floors[key[0]][1]:
                states[key] = 'unscanned'
    return records, states

def listing(cut, args, engine, response, original):
    """Canonical pages and records share a stable owner/ordinal cursor."""
    import json
    session = args['session_id']
    records, states = decide(cut, session) if cut.records is not None else ([], {})
    limit = response['limit']
    hidden = [dict(owner_store_id=r['owner_store_id'], ordinal=r['ordinal'], state=states[(r['owner_store_id'], r['ordinal'])],
                   reference=reference(r)) for r in records if (r['owner_store_id'], r['ordinal']) in states]
    offset = args.get('withheld_cursor', 0)
    if offset is not None and (type(offset) is not int or offset < 0):
        return dict(error='withheld_cursor must be a non-negative integer or null')
    response['withheld'] = dict(counts={s: sum(r['state'] == s for r in hidden) for s in ('overlap', 'unscanned')},
        locators=hidden[offset:offset + limit] if offset is not None else [],
        next_cursor=dict(withheld_cursor=offset + limit) if offset is not None and offset + limit < len(hidden) else None)
    if not records and 'record_cursor' not in args:
        return response
    cursor = args.get('record_cursor', [args.get('after_store_id', 0), 2147483647])
    if cursor is None:
        response.update(messages=[], next_cursor=None, returned_messages=0, has_more=False)
        return response
    if not isinstance(cursor, list) or len(cursor) != 2 or any(type(v) is not int or v < 0 for v in cursor):
        return dict(error='record_cursor must be [owner_store_id, ordinal] or null')
    base_args = dict(args, after_store_id=max(0, cursor[0] - 1))
    base_args.pop('record_cursor', None)
    canonical_page = json.loads(original(base_args, engine=engine))
    if 'error' in canonical_page:
        return canonical_page
    rows = []
    by_owner = defaultdict(list)
    for r in records:
        if (r['owner_store_id'], r['ordinal']) not in states:
            by_owner[r['owner_store_id']].append(r)
    for message in canonical_page['messages']:
        sid = message['store_id']
        for r in by_owner.get(sid, ()):
            if not args.get('roles') or 'user' in args['roles']:
                rows.append(((sid, r['ordinal']), record_view(r)))
        rows.append(((sid, 2147483647), message))
    rows = [(key, row) for key, row in rows if key > tuple(cursor)]
    more = len(rows) > limit or canonical_page.get('has_more', False)
    selected = rows[:limit]
    response.update(messages=[row for _, row in selected], returned_messages=len(selected), has_more=more,
        next_cursor=dict(record_cursor=list(selected[-1][0])) if more and selected else None)
    return response


def search_records(cut, args, engine):
    """A bounded literal/term recall arm over attached, visible records only."""
    import re
    query = str(args.get('query') or '').strip()
    if not query or args.get('content_scope') == 'externalized':
        return []
    session = args.get('session_id') or engine.current_session_id
    scope = args.get('session_scope', 'all' if getattr(engine, '_mcp_reader', False) else 'current')
    sessions = [session] if scope != 'all' else [r[0] for r in cut.canonical.execute('SELECT DISTINCT session_id FROM messages')]
    excluded = set(args.get('_excluded_session_ids') or args.get('exclude_session_ids') or [])
    terms = re.findall(r'\w+', query.casefold())
    hits = []
    for session in sessions:
        if session in excluded:
            continue
        records, states = decide(cut, session)
        for row in records:
            text = row['content'].casefold()
            match = query[1:-1].casefold() in text if query.startswith('"') and query.endswith('"') else (
                bool(terms) and all(term in text for term in terms))
            if match and (row['owner_store_id'], row['ordinal']) not in states:
                hits.append(dict(record_view(row), snippet=row['content'][:4000]))
    return hits[:max(1, min(200, int(args.get('limit', 10))))]


def surface(name):
    """Use one records-first cut for attachment, decisions and canonical output."""
    import copy
    import functools
    import json
    def decorate(original):
        @functools.wraps(original)
        def wrapped(args, **kwargs):
            engine = kwargs.get('engine')
            store = getattr(engine, '_store', None)
            if store is None or not getattr(store, 'db_identity', None) or records_path(store) is None:
                result = json.loads(original(args, **kwargs))
                if name == 'load' and 'error' not in result:
                    result['withheld'] = dict(counts=dict(overlap=0, unscanned=0), locators=[], next_cursor=None)
                return json.dumps(result)
            with records_read_cut(store) as cut:
                proxy, readonly = copy.copy(engine), copy.copy(store)
                readonly._conn = cut.canonical
                proxy._store = readonly
                if name == 'expand' and 'queued_user_ref' in args:
                    key = args['queued_user_ref']
                    if (not isinstance(key, list) or len(key) != 2
                            or any(type(v) is not int or v < 0 for v in key)
                            or any(k in args for k in ('store_id', 'node_id', 'externalized_ref'))):
                        return json.dumps(dict(error='queued_user_ref must be [owner_store_id, ordinal], exclusively'))
                    owner = cut.canonical.execute('SELECT session_id FROM messages WHERE store_id=?', (key[0],)).fetchone()
                    rows, _ = read_records(cut, 'queued_user_rows', owner[0]) if owner else ([], 0)
                    row = next((r for r in rows if [r['owner_store_id'], r['ordinal']] == key), None)
                    return json.dumps(record_view(row) if row else dict(error='record not attached'))
                result = json.loads(original(args, **dict(kwargs, engine=proxy)))
                if 'error' in result:
                    return json.dumps(result)
                if name == 'load':
                    result = listing(cut, args, proxy, result, original)
                elif name in ('grep', 'recall') and cut.records is not None:
                    hits = search_records(cut, dict(args, session_scope='all') if name == 'recall' else args, proxy)
                    result['results'] = [*hits, *result.get('results', [])][:result.get('limit', args.get('limit', 10))]
                    result['total_results'] = result.get('total_results', 0) + len(hits)
                return json.dumps(result)
        return wrapped
    return decorate
