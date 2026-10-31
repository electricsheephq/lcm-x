"""Default-off, query-time read-only entity cards over existing profile rows."""
import json
import unicodedata
from datetime import datetime, timezone

from . import tools as _tools
from .assertion_store import CURRENT_EXTRACTION_VERSION
from .search_query import _ASCII_CASE_FOLD, requires_like_fallback, sanitize_fts5_query

# ADR D6: event/conversation and unknown future namespaces are not bridged.
_NAMESPACES = ('person', 'user', 'assistant', 'organization', 'project', 'place',
               'object', 'account', 'document')


def _name(value):
    text = ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    while text and (text[0].isspace() or unicodedata.category(text[0]).startswith('P')):
        text = text[1:]
    while text and (text[-1].isspace() or unicodedata.category(text[-1]).startswith('P')):
        text = text[:-1]
    return text.strip()


def _time(row):
    event = _tools._normalize_observed_at(row.get('observed_at'))
    stamp = event if event is not None else row['timestamp']
    return {'time': datetime.fromtimestamp(stamp, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'time_source': 'event_time' if event is not None else 'write_timestamp',
            'store_id': row['store_id'],
            **{k: v for k, v in _tools._lcm_recall_event_time(row).items() if k == 'time_uncertain'}}


def _bounds(rows):
    def key(r):
        observed = _tools._normalize_observed_at(r.get('observed_at'))
        return (observed if observed is not None else r['timestamp'], r['store_id'])
    return [_time(min(rows, key=key)), _time(max(rows, key=key))] if rows else []


def _rank(row):
    return (row.get('_memory_like', False), row.get('search_rank') or 0, row['store_id'])


def _mention(row):
    return {**_time(row), 'snippet': row.get('snippet') or row['content'][:200]}


def _assertions(store, names, limit):
    conn = store.connection
    conn.create_function('_lcm_entity_name', 1, _name, deterministic=True)
    # Use the same active-source/version join as AssertionStore.query_assertions.
    base = '''SELECT a.*, m.content AS current_content,
              substr(a.subject_key,1,instr(a.subject_key,':')-1) AS namespace
              FROM lcm_assertions a JOIN lcm_assertion_sources s
              ON s.source_store_id=a.source_store_id AND s.extraction_version=a.extraction_version
              AND s.source_content_sha256=a.source_content_sha256
              JOIN messages m ON m.store_id=a.source_store_id
              WHERE s.invalidated_at IS NULL AND a.extraction_version=?'''
    if names is None:
        rows = conn.execute(f'''WITH active AS ({base})
            SELECT namespace, subject_key, count(*) AS count, min(source_store_id) AS store_id
            FROM active GROUP BY namespace, subject_key ORDER BY namespace, subject_key LIMIT ?''',
            (CURRENT_EXTRACTION_VERSION, limit)).fetchall()
        return {'entities': [dict(row) for row in rows]}
    marks = ','.join('?' for _ in names)
    namespaces = ','.join('?' for _ in _NAMESPACES)
    rows = conn.execute(f'''WITH active AS ({base}), matching AS (
        SELECT * FROM active WHERE namespace IN ({namespaces}) AND
        _lcm_entity_name(substr(subject_key,instr(subject_key,':')+1)) IN ({marks}))
        SELECT *, (SELECT count(DISTINCT namespace) FROM matching) AS namespace_count
        FROM matching ORDER BY observed_at DESC, assertion_id LIMIT 201''',
        [CURRENT_EXTRACTION_VERSION, *_NAMESPACES, *names]).fetchall()
    facts = []
    for row in rows[:200]:
        store._validate_source_provenance(
            content=row['current_content'], expected_sha256=row['source_content_sha256'],
            span_start=row['source_span_start'], span_end=row['source_span_end'],
            quote=row['source_quote'], label='entity fact')
        facts.append({**{key: row[key] for key in ('subject_key', 'predicate_key', 'value_text',
                                                  'observed_at', 'source_store_id')},
                      'store_id': row['source_store_id']})
    return {'facts': facts, 'facts_status': 'enabled', 'facts_truncated': len(rows) > 200,
            'ambiguous': bool(rows and rows[0]['namespace_count'] > 1)}


def lcm_memory(args, **kwargs):
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({'error': 'LCM engine not initialized'})
    if not engine._config.entities_enabled:
        return json.dumps({'status': 'disabled', 'enable_with': 'LCM_ENTITIES_ENABLED=1'})
    action = args.get('action')
    if action not in ('lookup', 'list'):
        return json.dumps({'error': 'action must be lookup or list'})
    default, cap = (_tools._LCM_MEMORY_LOOKUP_LIMITS if action == 'lookup'
                    else _tools._LCM_MEMORY_LIST_LIMITS)
    limit, error = _tools._parse_strict_int(args.get('limit', default), 'limit')
    if error or limit is None or limit <= 0:
        return json.dumps({'error': error or 'limit must be a positive integer'})
    limit = min(limit, cap)
    assertions = getattr(engine, '_assertions', None)
    if action == 'list':
        return json.dumps(_assertions(assertions, None, limit) if assertions is not None else
                          {'entities': [], 'status': 'no_entity_registry_until_e2'})
    name = args.get('name')
    aliases = args.get('aliases', [])
    if not isinstance(name, str) or not _name(name):
        return json.dumps({'error': 'name must be a non-empty string'})
    if not isinstance(aliases, list) or len(aliases) > 8 or any(
            not isinstance(a, str) or not _name(a) for a in aliases):
        return json.dumps({'error': 'aliases must be an array of at most 8 non-empty strings'})
    question = args.get('question')
    if question is not None and not isinstance(question, str):
        return json.dumps({'error': 'question must be a string'})
    names = list(dict.fromkeys([name.strip(), *[a.strip() for a in aliases]]))
    with engine._store._read_snapshot():
        return _lookup(engine, assertions, names, question, limit)


def _lookup(engine, assertions, names, question, limit):
    forms = list(dict.fromkeys(f for n in names for f in (n, _name(n))))
    hits, matches, searched = {}, {}, set()
    for value in forms:
        # Literal phrases: caller text cannot inject FTS operators. Each search is bounded.
        query = '"' + value.replace('"', '""') + '"'
        safe = sanitize_fts5_query(query)
        # Only identical FTS streams under ASCII folding are provably redundant.
        fold = (query,) if requires_like_fallback(query, safe) else safe.translate(_ASCII_CASE_FOLD)
        if fold in searched:
            continue
        searched.add(fold)
        metadata = {'rows': []}
        found = engine._store.search(query, limit=limit + 1, sort='relevance', _memory=metadata)
        form_matches = {r['store_id']: r for r in metadata['rows']}
        if any(row['store_id'] not in form_matches for row in found):
            return json.dumps({'error': 'Memory lookup metadata is inconsistent'})
        for row in found:
            row['_memory_like'] = metadata.get('like', False)
            if row['store_id'] not in hits or _rank(row) < _rank(hits[row['store_id']]):
                hits[row['store_id']] = row
        matches.update(form_matches)
    rows = sorted(hits.values(), key=_rank)
    truncated = len(matches) > limit
    rows = rows[:limit]
    mentions = engine._store.search(question, limit=5, sort='relevance', matching_names=forms) if question else []
    carried = engine._store.get_carried_stamp_ids(list(matches), batched=True)
    for row in [*matches.values(), *rows, *mentions]:
        if row['store_id'] in carried:
            row['time_uncertain'] = 'carried'
    groups = {row['session_id']: [] for row in rows}
    for row in matches.values():
        group = groups.setdefault(row['session_id'], [])
        group.append(row)
    sessions = []
    for session, group in groups.items():
        times = _bounds(group)
        sessions.append({'session_id': session, 'count': len(group), 'first_seen': times[0],
                         'last_seen': times[-1], 'store_id': times[0]['store_id'],
                         'snippets': [_mention(row) for row in rows if row['session_id'] == session][:3]})
    times = _bounds(matches.values())
    result = {'names_searched': names, 'mention_count': len(matches), 'truncated': truncated,
              'first_seen': times[0] if times else None, 'last_seen': times[-1] if times else None,
              'sessions': sessions, 'facts': [], 'facts_status': 'assertions_disabled',
              'ambiguous': False}
    if assertions is not None:
        result.update(_assertions(assertions, [_name(n) for n in names], limit))
    if question:
        result['mentions'] = [_mention(row) for row in mentions]
    return json.dumps(result)
