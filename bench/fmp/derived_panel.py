"""DM0 population, canonical aliases and outcome-blind inventory admission."""
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from .common import sha, digest_file
from .derived_extract import source_view
from .dev_cache import Incomplete
from .dev_panel import encoded, fingerprint
EXPECTED = dict(total=886, target=314, guard=195, multi_hop=268, preference=25, other=84)
def history(q):
    return source_view(asdict(q) | {'question_date':''})['haystack_hash']
def select(questions, w1, dev_ids, dev_histories, strict=True):
    aliases = {getattr(q,'canonical_id',q.qid):q for q in questions}
    if len(aliases) != len(questions):
        raise Incomplete('duplicate canonical IDs')
    originals = {e['id']:e for e in w1['items']}
    if len(originals) != len(w1['items']):
        raise Incomplete('duplicate W1 IDs')
    if any(i not in aliases or fingerprint(aliases[i]) != e['sha256'] for i,e in originals.items()):
        raise Incomplete('normalized question fingerprint drift')
    entries = []
    for alias,q in aliases.items():
        if q.qid in dev_ids or history(q) in dev_histories:
            continue
        if alias not in originals and not (q.dataset=='locomo' and q.qtype in ('1','2') and alias.split('-q')[0] in {f'locomo-{i}' for i in range(3,11)}):
            continue
        guard = q.is_abstention
        category = ('locomo-adversarial' if q.dataset=='locomo' else 'lme-abstention') if guard else q.qtype
        entries.append(dict(id=alias, normalized_id=q.qid, dataset=q.dataset, category=category, guard=guard,
            target=not guard and q.qtype in ('knowledge-update','temporal-reasoning','2'),
            cluster=alias.split('-q')[0] if q.dataset=='locomo' else history(q), sha256=fingerprint(q)))
    if strict:
        counts = dict(total=len(entries), target=sum(e['target'] for e in entries), guard=sum(e['guard'] for e in entries),
            multi_hop=sum(e['category'] in ('multi-session','1') for e in entries),
            preference=sum(e['category']=='single-session-preference' for e in entries),
            other=sum(not e['target'] and not e['guard'] and e['category'] not in ('multi-session','1','single-session-preference') for e in entries))
        if counts != EXPECTED or Counter(e['dataset'] for e in entries) != {'lme-s':179,'locomo':707}:
            raise Incomplete('DM0 population counts drift')
    return sorted(entries, key=lambda e:(not e['target'], sha(('1191:'+e['id']).encode())))
def local25(receipt, product, cache=None):
    if receipt.get('status') != 'COMPLETE' or receipt.get('product') != product or not receipt.get('rows'):
        raise Incomplete('W1 LOCAL/25 incomplete')
    if any(not r.get('sources') or not r.get('cache_key') or r.get('observed',{}).get('limit') != 25 for r in receipt['rows']):
        raise Incomplete('W1 LOCAL/25 retrieval identity incomplete')
    if cache is not None:
        root=Path(cache)
        stores={digest_file(p):p.parent for p in (root/'stores').glob('*/*/build-manifest.json')}
        if len(receipt['rows'])!=450 or len({r['id'] for r in receipt['rows']})!=450:
            raise Incomplete('W1 LOCAL/25 population incomplete')
        for r in receipt['rows']:
            rec=json.loads((root/'retrieval'/(r['cache_key']+'.json')).read_bytes())
            identity,value=rec['identity'],rec['value']
            if (sha(encoded(identity))!=r['cache_key'] or identity['product']!=product or
                    identity['config'].get('recall_depth')!=25 or sha(encoded(value))!=rec['value_sha256'] or
                    sha(value['evidence'].encode())!=r['delivered_sha256'] or identity['store']!=r['sources']):
                raise Incomplete('W1 retrieval digest/identity drift')
            directory=stores[r['sources']['manifest_sha256']]
            for name,key in (('lcm.db','database_sha256'),('speakers.json','speakers_sha256')):
                if digest_file(directory/name)!=r['sources'][key]:
                    raise Incomplete('W1 store digest drift')
            if json.loads((directory/'build-manifest.json').read_bytes()).get('config')!='local':
                raise Incomplete('W1 is not LOCAL')
    return receipt['rows']
def inventory(receipt, product, cache):
    counts = {name:sum(1 for _ in (Path(cache)/name).glob('*.json')) for name in ('responses','retrieval')}
    try:
        rows = local25(receipt, product, cache)
        status = 'COMPLETE'
        n = len(rows)
    except (Incomplete, OSError, KeyError, ValueError):
        status = 'INCOMPLETE'
        n = 0
    return dict(status=status, local25_rows=n, local25_retrieval_hits=n, local25_retrieval_misses=450-n, **counts, expected=EXPECTED,
                planned_calls=dict(extraction=[450,650], total_max_reuse=[5040,5240], cold_extra=858))
