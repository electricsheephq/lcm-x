"""Question-blind source windows and exact-quote admission; no product publication."""
import json
import re
from collections import Counter
from .common import sha
from .datasets import iso
from .delivery import tokens
from .derived_cache import Excluded, snapshot
from .dev_cache import Incomplete
from .dev_panel import encoded
SCHEMA = 'dm0-v1'
PROMPT = '''Return a JSON array, one object per independent window: schema_version=dm0-v1,
haystack_hash, window_id, items[], relations[]. Never join haystacks. Each item has local_id,
kind (fact,event,preference,commitment,status,request,intention,quotation,question), subject_key,
subject_resolution (explicit/contextual), predicate_key, object_value, value_text, polarity,
scope_key, event_at/valid_from/valid_to (null unless quoted), explicitness=explicit,
sources[{row_id,quote}]. Quote exact source text; do not supply offsets. Preserve speaker/role.
Questions, temporary requests, quotations, plans and fulfilled tasks are not durable preferences.
Relations: from_local_id,to_local_id (or prior_item_id),type (supersedes,cancels,fulfills),
sources[{row_id,quote}],evidence=explicit. Only explicit update evidence creates a relation;
recency alone never does. Unknown dates stay null. Empty extraction is a valid items=[] response.'''
def source_view(value):
    cutoff = iso(value.get('question_date'))
    rows = []
    for si, session in enumerate(sorted(value['sessions'], key=lambda s: s['date']), 1):
        date = iso(session['date'])
        if cutoff and date and date > cutoff:
            continue
        for ri, turn in enumerate(session['turns'], 1):
            rows.append(dict(row_id=f's{si:04d}/r{ri:04d}', date=date, role=turn['role'],
                             speaker=turn.get('speaker', ''), content=turn['content']))
    return dict(rows=rows, cutoff=cutoff, haystack_hash=sha(encoded(rows)))
def windows(view, source_tokens=16000, context_tokens=2000):
    limit = source_tokens * 4 - 1024
    if limit < 1024:
        raise ValueError('window budget too small')
    chunks = [dict(r, start=start, end=min(len(r['content']), start+limit),
                   row_sha256=sha(r['content'].encode()), content=r['content'][start:start+limit])
              for r in view['rows'] for start in range(0, max(1, len(r['content'])), limit)]
    groups, current = [], []
    for chunk in chunks:
        if current and tokens(current + [chunk]) > source_tokens:
            groups.append(current)
            current = []
        current.append(chunk)
    if current:
        groups.append(current)
    result = []
    for index, group in enumerate(groups):
        preceding = groups[index-1] if index else []
        context = []
        for row in reversed(preceding):
            if tokens([row, *context]) <= context_tokens:
                context.insert(0, row)
        result.append(dict(schema_version=SCHEMA, haystack_hash=view['haystack_hash'], cutoff=view['cutoff'],
                           window_id=str(index), rows=group, context=context))
    return result
def request(batch, prior=()):
    return [dict(role='system', content=PROMPT), dict(role='user', content=json.dumps(dict(windows=batch, prior_index=[i for i in prior if i['haystack_hash'] in {w['haystack_hash'] for w in batch}]), ensure_ascii=False))]
def validate(wire, window, prior=()):
    if not isinstance(wire,dict):
        return dict(items=[],relations=[],rejected=[dict(item=0,reason='non-object window')],state='partial')
    if any(wire.get(k) != window[k] for k in ('schema_version', 'haystack_hash', 'window_id')):
        raise Incomplete('extraction envelope mismatch')
    if not isinstance(wire.get('items'), list) or not isinstance(wire.get('relations'), list):
        raise Incomplete('invalid extraction arrays')
    rows = window['rows'] + window.get('context', [])
    def resolve(refs):
        if not refs:
            raise ValueError('missing sources')
        out = []
        for ref in refs:
            matches = []
            for row in rows:
                quote = ref['quote']
                if row['row_id'] == ref['row_id'] and quote:
                    positions = [m.start() for m in re.finditer('(?=' + re.escape(quote) + ')', row['content'])]
                    matches.extend((row, n) for n in positions)
            if len(matches) != 1:
                raise ValueError('ambiguous/missing quote')
            row, offset = matches[0]
            if ref.get('role', row['role']) != row['role']:
                raise ValueError('role substitution')
            out.append(dict(row_id=row['row_id'], quote=quote, start=row['start']+offset,
                end=row['start']+offset+len(quote), sha256=row['row_sha256'], role=row['role'], speaker=row['speaker'], date=row['date']))
        return out
    accepted, rejected, mapping = [], [], {}
    counts = Counter(i.get('local_id') for i in wire['items'] if isinstance(i, dict))
    for n, raw in enumerate(wire['items']):
        try:
            i = {k: raw[k] for k in ('kind','subject_key','subject_resolution','predicate_key','object_value',
                 'value_text','polarity','scope_key','event_at','valid_from','valid_to','explicitness')}
            if any(not isinstance(i[k],str) for k in ('value_text','subject_key')):
                raise ValueError('invalid field type')
            if counts[raw['local_id']] != 1 or i['explicitness'] != 'explicit' or i['kind'] not in (
                    'fact','event','preference','commitment','status','request','intention','quotation','question'):
                raise ValueError('invalid type/id')
            refs = resolve(raw['sources'])
            text = ' '.join(r['quote'] for r in refs)
            if not i['value_text'] or i['value_text'].lower() not in text.lower():
                raise ValueError('unsupported value text')
            if i['kind'] == 'preference' and (any(r['role'] != 'user' for r in refs) or
                    not re.search(r'\b(prefer|favorite|favourite)\b', text, re.I) or
                    re.search(r'[?"“”]|\b(please|plan|intend|will|finished|completed|said)\b', text, re.I)):
                raise ValueError('unsupported durable preference')
            if any(i[k] is not None and str(i[k]) not in text for k in ('event_at','valid_from','valid_to')):
                raise ValueError('unsupported date')
            i.update(sources=refs, haystack_hash=window['haystack_hash'], observed_at=max(r['date'] for r in refs),
                v4_row_candidate=len(refs)==1 and i['subject_resolution']=='explicit' and
                i['subject_key'].split(':')[-1].lower() in text.lower() and i['value_text'].lower() in text.lower())
            i['id'] = sha(encoded(i))
            mapping[raw['local_id']] = i
            accepted.append(i)
        except (KeyError, TypeError, ValueError):
            rejected.append(dict(item=n, reason='source/type admission'))
    mapping.update({i['id']: i for i in prior if i['haystack_hash'] == window['haystack_hash']})
    relations = []
    for n, raw in enumerate(wire['relations']):
        try:
            a, b = mapping[raw['from_local_id']], mapping[raw.get('to_local_id', raw.get('prior_item_id'))]
            refs = resolve(raw['sources'])
            if raw['evidence'] != 'explicit' or raw['type'] not in ('supersedes','cancels','fulfills') or a['id']==b['id']:
                raise ValueError('unsupported relation')
            if any(a[k] != b[k] for k in ('haystack_hash','subject_key','predicate_key','scope_key')) or {
                    (r['role'],r['speaker']) for r in a['sources']} != {(r['role'],r['speaker']) for r in b['sources']}:
                raise ValueError('cross-scope relation')
            if not re.search(r'\b(now|instead|actually|no longer|cancel|finished|completed|fulfilled)\b', ' '.join(r['quote'] for r in refs), re.I):
                raise ValueError('implicit relation')
            relations.append(dict(from_id=a['id'], to_id=b['id'], type=raw['type'], sources=refs))
        except (KeyError, TypeError, ValueError):
            rejected.append(dict(relation=n, reason='relation admission'))
    return dict(items=accepted, relations=relations, rejected=rejected,
                state='partial' if rejected else 'succeeded' if accepted else 'empty')
def execute(views, cache, spec, root):
    all_items, all_relations, ledger = {}, {}, []
    for view in views:
        ws = windows(view)
        for start in range(0, len(ws), 3):
            batch = ws[start:start+3]
            calls = []
            key = sha(encoded(batch))
            try:
                msgs = request(batch, list(all_items.values()))
                for attempt in range(2):
                    response = cache.call(spec, msgs, calls, role='extract', rubric=SCHEMA,
                        sources=dict(batch=key, repair=attempt), max_tokens=16384)
                    try:
                        output = json.loads(response['content'])
                        if not isinstance(output, list) or len(output) != len(batch):
                            raise ValueError('batch shape')
                        break
                    except ValueError:
                        if attempt:
                            raise Incomplete('invalid extraction JSON') from None
                        msgs = request(batch, list(all_items.values())) + [dict(role='user', content='Repair invalid JSON; same schema and sources. Return the array only.')]
                for w, raw in zip(batch, output, strict=True):
                    got = validate(raw, w, list(all_items.values()))
                    all_items.update({i['id']:i for i in got['items']})
                    all_relations.update({sha(encoded(r)):r for r in got['relations']})
                    ledger.append(dict(window=w['window_id'], source=view['haystack_hash'], state=got['state'],
                                       rejected=got['rejected'], extracted_count=len(raw['items']) if isinstance(raw,dict) else 1,
                                       rejected_count=sum('item' in r for r in got['rejected']),calls=calls))
            except Excluded:
                ledger.extend(dict(window=w['window_id'],source=view['haystack_hash'],state='excluded',reason='transient retries exhausted',calls=calls) for w in batch)
            except (Incomplete, ValueError, TypeError, KeyError):
                ledger.extend(dict(window=w['window_id'], source=view['haystack_hash'], state='failed', calls=calls) for w in batch)
                return dict(items=list(all_items.values()),relations=list(all_relations.values()),ledger=ledger)
    result = dict(items=list(all_items.values()), relations=list(all_relations.values()), ledger=ledger)
    if not any(w['state']=='failed' for w in ledger):
        snapshot(root / 'extraction.json', result)
    return result
