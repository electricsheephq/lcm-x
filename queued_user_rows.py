"""Recall-only records for a host's overwritten queued user messages (I1/I5/I6).

Capture never writes. Canonical ingest only reports its committed object/id map;
records transactions and read cuts never participate in canonical recovery.
"""
from collections import Counter, defaultdict, deque
from contextlib import closing, suppress
from dataclasses import dataclass
import hashlib
import logging
import sqlite3
import time

from . import ingest_protection as protection
from .store import _normalize_observed_at, _records_ro, open_records_writer, read_records, records_path, records_read_cut

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
                            content_digest=digest(protected['content']), observed_at=_normalize_observed_at(stamp),
                            observed_at_source='host_message_timestamp', captured_uid=uid,
                            captured_at=item.created, superseded_by=None, session_id=owner['session_id'],
                            conversation_id=owner['conversation_id'], profile=str(item.binding[0]))
                        if len(protected['content'].encode('utf-8')) > DRAIN_BYTES:
                            count(engine, 'refused_oversize')
                            continue
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
    for entry in tuple(queue):
        if done >= DRAIN_ENTRIES or time.monotonic() >= end:
            break
        size = len(dict(entry.values)['content'].encode('utf-8'))
        if used + size > DRAIN_BYTES:
            break
        try:
            engine._store_queued_user_rows([entry])
        except Exception as exc:
            count(engine, 'drain_failed')
            if isinstance(exc, sqlite3.OperationalError) and any(t in str(exc).lower() for t in ('busy', 'locked')):
                break
            values = dict(entry.values)
            permanent = str(exc) == 'pending purge intent'
            if str(exc) == 'owner not committed or evidence mismatch':
                with closing(_records_ro(entry.identity)) as canonical:
                    permanent = not canonical.execute('SELECT 1 FROM messages WHERE store_id=?',
                        (values['owner_store_id'],)).fetchone() and not any(
                            t['session_id'] == values['session_id'] and t['store'].db_identity == entry.identity
                            for t in engine._pending_session_tails)
            if permanent:
                with suppress(ValueError):
                    queue.remove(entry)
                    count(engine, 'dropped_permanent')
                    logger.warning('queued records permanent refusal: %s', exc)
            done, used = done + 1, used + size
            continue
        with suppress(ValueError):
            queue.remove(entry)
        done, used = done + 1, used + size
    if not queue:
        _EXIT_ENGINES.discard(engine)
    if exiting:
        count(engine, 'exit_kept', len(queue))
        logger.warning('queued records exit: discarded_at_exit=%d counts=%s', len(queue), dict(engine._queued_counts))


def flush_at_exit():
    for engine in tuple(_EXIT_ENGINES):
        try:
            with engine._storage_lock:
                drain(engine, exiting=True)
        except Exception as exc:
            logger.warning('queued records exit failed: %s', type(exc).__name__)



def reference(record):
    return dict(tool='lcm_expand', args=dict(store_id=record['owner_store_id']))


def observed_at(row):
    return _normalize_observed_at(row['observed_at'])


def record_view(row):
    keys = ('owner_store_id', 'ordinal', 'role', 'content', 'observed_at_source',
            'session_id', 'conversation_id', 'captured_uid', 'superseded_by')
    return dict({key: row[key] for key in keys}, kind='record', observed_at=observed_at(row), reference=reference(row))


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
    """Only attach records owned by canonical rows on this unmodified page."""
    records, states = decide(cut, response['session_id'])
    owners = {row['store_id'] for row in response['messages']}
    visible, hidden = [], []
    for r in records:
        if r['owner_store_id'] not in owners:
            continue
        state = states.get((r['owner_store_id'], r['ordinal']))
        if state:
            hidden.append(dict(owner_store_id=r['owner_store_id'], ordinal=r['ordinal'], state=state, reference=reference(r)))
        else:
            visible.append(record_view(r))
    if visible:
        response['queued_user_records'] = visible
    if hidden:
        response['queued_user_records_withheld'] = dict(counts=dict(dict(overlap=0, unscanned=0), **Counter(r['state'] for r in hidden)), locators=hidden)
    return response


def search_records(cut, args, engine):
    """A bounded literal/term recall arm over attached, visible records only."""
    import re
    from . import tools
    allowed = {'query', 'limit', 'mode', 'sort', 'session_id', 'session_scope', '_excluded_session_ids',
               'exclude_current_session', 'exclude_session_ids', 'role', 'conversation_id', 'time_from', 'time_to'}
    if any(k not in allowed and v not in (None, '', [], False) for k, v in args.items()):
        return []
    role, _ = tools._parse_grep_role(args.get('role'))
    if role is not None and role != 'user':
        return []
    start, _ = tools._parse_optional_timestamp(args.get('time_from'), 'time_from')
    end, _ = tools._parse_optional_timestamp(args.get('time_to'), 'time_to')
    query = str(args.get('query') or '').strip()
    if not query or args.get('content_scope') == 'externalized':
        return []
    scope = str(args.get('session_scope', 'current')).lower()
    if scope not in ('current', 'all', 'session'):
        scope = 'current'
    session = str(args.get('session_id') or '').strip() if scope == 'session' else engine.current_session_id
    sessions = [session] if scope != 'all' else [r[0] for r in cut.canonical.execute('SELECT DISTINCT session_id FROM messages')]
    excluded, _ = tools._parse_retrieval_excluded_session_ids(args, engine.current_session_id)
    excluded.update(args.get('_excluded_session_ids') or ())
    terms = re.findall(r'\w+', query.casefold())
    hits = []
    for session in sessions:
        if session in excluded:
            continue
        records, states = decide(cut, session)
        for row in records:
            stamp = observed_at(row)
            if (str(args.get('conversation_id') or '').strip() and row['conversation_id'] != str(args['conversation_id']).strip()
                    or start is not None and (stamp is None or stamp < start)
                    or end is not None and (stamp is None or stamp > end)):
                continue
            text = row['content'].casefold()
            match = query[1:-1].casefold() in text if query.startswith('"') and query.endswith('"') else (
                bool(terms) and all(term in text for term in terms))
            if match and (row['owner_store_id'], row['ordinal']) not in states:
                hits.append(record_view(row))
    return sorted(hits, key=lambda r: observed_at(r) or 0, reverse=True)


def surface(name):
    """Canonical handler first, with the real engine; isolated read-only additions."""
    import functools
    import json
    from . import tools
    def decorate(original):
        @functools.wraps(original)
        def wrapped(args, **kwargs):
            raw = original(args, **kwargs)
            try:
                return augment(raw, args, **kwargs)
            except Exception as exc:
                logger.warning('queued records read failed: %s', type(exc).__name__)
                with suppress(TypeError, ValueError):
                    result = json.loads(raw)
                    if isinstance(result, dict):
                        result['queued_user_records_status'] = dict(available=False,
                            reason='recovery_pending' if str(exc) == 'recovery_pending' else 'read_failed')
                        return json.dumps(result)
                return raw
        def augment(raw, args, **kwargs):
            engine = kwargs.get('engine')
            store = getattr(engine, '_store', None)
            if store is not None:
                path = records_path(store)
                if path is None or not path.exists():
                    return raw
                result = json.loads(raw)
                if 'error' in result:
                    return raw
                with records_read_cut(store, _read_only=True) as cut:
                    if cut.refused_format is not None:
                        raise ValueError('unsupported_format')
                    if name == 'load':
                        listing(cut, args, engine, result, original)
                    elif name == 'expand' and args.get('store_id') is not None and result.get('session_id'):
                        rows, states = decide(cut, result.get('session_id'))
                        rows = [dict(record_view(r), state=states.get((r['owner_store_id'], r['ordinal']), 'visible'))
                                for r in rows if r['owner_store_id'] == result.get('store_id')]
                        if rows:
                            result['queued_user_records'] = rows
                    elif name in ('grep', 'recall'):
                        hits = search_records(cut, dict(args, session_scope='all') if name == 'recall' else args, engine)
                        if hits:
                            result['queued_user_record_hits'] = hits
            else:
                return raw
            for key in ('queued_user_records', 'queued_user_record_hits'):
                for row in result.get(key, []):
                    if name == 'expand':
                        row.update(tools._slice_content_for_response(row['content'],
                            tools._parse_positive_int(args.get('max_tokens', 4000), 4000),
                            tools._parse_non_negative_int(args.get('content_offset', 0), 0)))
                    else:
                        bound = result['max_content_chars'] if name == 'load' else tools._LCM_RECALL_SNIPPET_CHARS
                        row.update(tools._slice_loaded_content(row['content'], bound))
                        if name in ('grep', 'recall'):
                            row['snippet'] = row['content']
            cap = dict(grep=tools._LCM_GREP_RESPONSE_CHAR_CAP, recall=tools._LCM_RECALL_RESPONSE_CHAR_CAP).get(name)
            omitted = 0
            # Each cap in main's own measure: grep counts unescaped characters (tools.py:1841),
            # recall the escaped reply (recall_delivery.py:486-511).
            while cap and len(json.dumps(result, ensure_ascii=name != 'grep')) > cap:
                entries = result.get('queued_user_record_hits', [])
                if not entries:
                    break
                entries.pop()
                omitted += 1
                result['queued_user_records_omitted'] = omitted
                if not entries:
                    del result['queued_user_record_hits']
            # Written ASCII-escaped, as main's grep and default recall write their replies.
            return raw if result == json.loads(raw) else json.dumps(result)
        return wrapped
    return decorate
