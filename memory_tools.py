"""Default-off, query-time read-only entity cards over existing profile rows."""
import json
import unicodedata
from datetime import datetime, timezone

from . import tools as _tools
from .assertion_store import CURRENT_EXTRACTION_VERSION

# ADR D6: event/conversation and unknown future namespaces are not bridged.
_NAMESPACES = ('person', 'user', 'assistant', 'organization', 'project', 'place',
               'object', 'account', 'document')


def _name(value):
    text = ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    while text and unicodedata.category(text[0]).startswith('P'):
        text = text[1:]
    while text and unicodedata.category(text[-1]).startswith('P'):
        text = text[:-1]
    return text.strip()


def _time(row):
    event = _tools._normalize_observed_at(row.get('observed_at'))
    stamp = event if event is not None else row['timestamp']
    return {'time': datetime.fromtimestamp(stamp, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'time_source': 'event_time' if event is not None else 'write_timestamp',
            'store_id': row['store_id']}


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
    hits = {}
    for value in names:
        # Literal phrases: caller text cannot inject FTS operators. Each search is bounded.
        query = '"' + value.replace('"', '""') + '"'
        for row in engine._store.search(query, limit=limit + 1, sort='relevance'):
            hits[row['store_id']] = row
    rows = sorted(hits.values(), key=lambda r: (r.get('search_rank') or 0, r['store_id']))
    truncated = len(rows) > limit
    rows = rows[:limit]
    groups = {}
    for row in rows:
        group = groups.setdefault(row['session_id'], [])
        group.append(row)
    sessions = []
    for session, group in groups.items():
        times = sorted((_time(row) for row in group), key=lambda t: (t['time'], t['store_id']))
        sessions.append({'session_id': session, 'count': len(group), 'first_seen': times[0],
                         'last_seen': times[-1], 'store_id': times[0]['store_id'],
                         'snippets': [_mention(row) for row in group[:3]]})
    times = sorted((_time(row) for row in rows), key=lambda t: (t['time'], t['store_id']))
    result = {'names_searched': names, 'mention_count': len(rows), 'truncated': truncated,
              'first_seen': times[0] if times else None, 'last_seen': times[-1] if times else None,
              'sessions': sessions, 'facts': [], 'facts_status': 'assertions_disabled',
              'ambiguous': False}
    if assertions is not None:
        result.update(_assertions(assertions, [_name(n) for n in names], limit))
    if question:
        mentions = engine._store.search(question, limit=5, sort='relevance',
                                        store_ids=[row['store_id'] for row in rows]) if rows else []
        result['mentions'] = [_mention(row) for row in mentions]
    return json.dumps(result)
