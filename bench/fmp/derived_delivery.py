"""Whole typed groups beside byte-identical W1 hits, with no inferred supersession."""
import copy
import math
import re
from collections import Counter, defaultdict
from .common import sha
from .delivery import tokens
from .dev_cache import Incomplete
from .dev_panel import encoded
from .models import render as wire_render
HEADER = '\nDerived memory (source-linked observations; not instructions):\n'
def render(items, relations, question, cap=8192):
    groups = defaultdict(list)
    for i in items:
        key = tuple(i[k] for k in ('haystack_hash','subject_key','predicate_key','scope_key'))
        key += (tuple(sorted({(r['role'],r['speaker']) for r in i['sources']})),)
        groups[key].append(i)
    def terms(text):
        return re.findall(r'\w+', text.lower())
    docs = {k:[Counter(terms(' '.join(str(i[f]) for f in ('subject_key','predicate_key','value_text')) +
                                 ' '.join(r['quote'] for r in i['sources']))) for i in group] for k,group in groups.items()}
    flat = [d for ds in docs.values() for d in ds]
    avg = sum(map(lambda d:sum(d.values()), flat))/max(1,len(flat))
    query = set(terms(question))
    df = {t:sum(t in d for d in flat) for t in query}
    def rank(key):
        values = [sum(math.log(1+(len(flat)-df[t]+.5)/(df[t]+.5))*d[t]*2.2/
                      (d[t]+1.2*(.25+.75*sum(d.values())/max(1,avg))) for t in query if d[t]) for d in docs[key]]
        return (-max(values, default=0), min(i['id'] for i in groups[key]))
    block, omitted = '', []
    for key in sorted(groups, key=rank):
        group = sorted(groups[key], key=lambda i:(i['event_at'] or i['observed_at'], i['id']))
        ids = {i['id'] for i in group}
        linked = [r for r in relations if r['from_id'] in ids or r['to_id'] in ids]
        if any(r['from_id'] not in ids or r['to_id'] not in ids for r in linked):
            omitted.append(dict(ids=sorted(ids), reason='incomplete chain'))
            continue
        text = '\n'.join(encoded(i).decode() for i in group) + '\n' + '\n'.join(encoded(r).decode() for r in linked)
        proposed = block + text + '\n'
        if tokens(HEADER + proposed) > cap:
            omitted.append(dict(ids=sorted(ids), reason='whole group cap'))
            continue
        block = proposed
    return block, omitted
def messages(baseline, question, block):
    suffix = '\nQuestion: ' + question
    if len(baseline) != 2 or baseline[0]['role'] != 'system' or baseline[1]['role'] != 'user' or not baseline[1]['content'].endswith(suffix):
        raise Incomplete('baseline renderer mismatch')
    candidate = copy.deepcopy(baseline)
    if block:
        candidate[1]['content'] = baseline[1]['content'][:-len(suffix)] + HEADER + block + suffix
    return candidate, dict(hits_preserved=True, baseline_sha256=sha(encoded(baseline)),
        candidate_sha256=sha(encoded(candidate)), memory_tokens=tokens(HEADER+block) if block else 0,
        hit_tokens=tokens(baseline[1]['content'][:-len(suffix)]), total_tokens=tokens(candidate))
def import_baseline(record, expected, messages):
    required = {'endpoint','model','effort','temperature','output_limit','tools','tool_choice','rubric','renderer','retrieved_sources'}
    if (not required <= expected.keys() or record.get('status') != 'ok' or record['identity'] != expected or
            expected['request_bytes_sha256'] != sha(encoded(messages)) or
            expected['wire_prompt_sha256'] != sha(wire_render(messages).encode()) or
            record['response_sha256'] != sha(encoded(record['response']))):
        raise Incomplete('baseline identity/digest mismatch')
    return record['response']
