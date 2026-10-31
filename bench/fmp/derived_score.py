"""Predeclared two-family DM0 investment gate, never issue acceptance."""
import math
from functools import lru_cache
from .dev import compare
from .score import mcnemar
def holm(ps):
    result, last = [0.0]*len(ps), 0
    for rank,index in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        last = min(1, max(last, (len(ps)-rank)*ps[index]))
        result[index] = last
    return result
@lru_cache(maxsize=4096)
def cp(k, n):
    def cdf(k, p):
        return sum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)+
                   j*math.log(p)+(n-j)*math.log1p(-p)) for j in range(k+1))
    def invert(k, target):
        lo, hi = 0., 1.
        for _ in range(55):
            mid = (lo+hi)/2
            if cdf(k,mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    return (invert(k-1,.975) if k else 0., invert(k,.025) if k<n else 1.)
def paired_bounds(gains, losses, n):
    if n <= 0:
        return [-1.,1.]
    g, loss = cp(gains,n), cp(losses,n)
    return [g[0]-loss[1],g[1]-loss[0]]
def table(entries, rows, judge):
    selected = {(r['arm'],r['id']):r for r in rows if r['judge']==judge}
    if not entries:
        return dict(n=0,delta=None,ci95=[-1,1],paired95=[-1,1],p=1)
    mapped = {(arm,e['id']):selected[(src,e['id'])] for e in entries for arm,src in (('baseline','A0'),('candidate','M8192'))}
    result = compare(entries,mapped)
    a,b = [[selected[(arm,e['id'])]['correct'] for e in entries] for arm in ('A0','M8192')]
    exact = mcnemar(a,b)
    return result | exact | dict(paired95=paired_bounds(exact['n01'],exact['n10'],len(entries)),
        both_correct=sum(x and y for x,y in zip(a,b)), both_wrong=sum(not x and not y for x,y in zip(a,b)))
def passes(tables):
    target,overall,guard = [tables[k] for k in ('target','overall','guard')]
    adjusted = holm([target['p'],overall['p']])
    return (target['delta'] is not None and target['delta']>=.05 and overall['delta']>=.02 and
        all(p<.05 for p in adjusted) and target['ci95'][0]>0 and overall['ci95'][0]>0 and
        all(t['delta'] is None or t['delta']>=-.02 for t in tables['categories'].values()) and
        min(guard['ci95'][0],guard['paired95'][0])>-.02)
def evaluate(entries, rows, *, complete=True, semantic_valid=True, drift_valid=True, exclusions=(),**admission):
    from .derived import ARMS, JUDGES
    population=len(entries)
    category_names=sorted({e['category'] for e in entries})
    excluded={r['id'] if isinstance(r,dict) else r for r in exclusions} & {e['id'] for e in entries}
    entries=[e for e in entries if e['id'] not in excluded]
    rows=[r for r in rows if r['id'] not in excluded]
    keys = {(r['id'],r['arm'],r['judge']) for r in rows if r.get('correct') is None or type(r['correct']) is bool}
    expected = {(e['id'],a,j) for e in entries for a in ARMS for j in JUDGES}
    missing = len(expected-keys)
    result = dict(verdict='PARK-INCOMPLETE', reasons=[], judges={}, missing=missing,
        exclusions=len(excluded), claim_class='advisory', stage='Stage 0 only; GO funds Stage 1, never closes #1191')
    invalid={j:{a:sum(r['correct'] is None for r in rows if r['judge']==j and r['arm']==a) for a in ARMS} for j in JUDGES}
    result.update(admission,exclusion_details=list(exclusions),invalid_verdicts=invalid)
    if missing or len(keys)!=len(rows) or keys-expected or not semantic_valid or not drift_valid or len(excluded)>population*.03 or any(
            n>len(entries)*.02 for counts in invalid.values() for n in counts.values()):
        result['reasons']=[admission.get('incomplete_reason') or 'missing/duplicate judge pairs, semantic/coverage, drift, >3% exclusions or >2% invalid verdicts']
        return result
    for judge in JUDGES:
        invalid_ids={r['id'] for r in rows if r['judge']==judge and r['correct'] is None}
        valid=[e for e in entries if e['id'] not in invalid_ids]
        target_entries = [e for e in valid if e['target']]
        categories = {c:table([e for e in valid if e['category']==c],rows,judge) for c in category_names}
        if any(t['n']==0 for t in categories.values()):
            result['reasons']=['declared category has no evaluable items: '+judge]
            return result
        tables = {name:table(group,rows,judge) for name,group in (
            ('target',target_entries),('overall',valid),('guard',[e for e in valid if e['guard']]))}
        tables['categories'] = categories
        tables['holm'] = holm([tables['target']['p'],tables['overall']['p']])
        tables['datasets'] = {d:table([e for e in valid if e['dataset']==d],rows,judge) for d in sorted({e['dataset'] for e in valid})}
        tables['dataset_balanced_delta'] = sum(t['delta'] for t in tables['datasets'].values())/max(1,len(tables['datasets']))
        result['judges'][judge] = tables
    if not complete:
        if admission.get('incomplete_reason'):
            result['reasons'] = [admission['incomplete_reason']]
            return result
        futile = all(max(t['target']['ci95'][1],t['target']['paired95'][1])<.05 for t in result['judges'].values())
        result['verdict'] = 'PARK-FUTILITY' if futile else 'PARK-INCOMPLETE'
        result['reasons'] = ['fixed target interim; never promotes']
        return result
    good = all(passes(t) for t in result['judges'].values())
    uncertain = any(min(t[k]['ci95'][0],t[k]['paired95'][0])<=threshold<=max(t[k]['ci95'][1],t[k]['paired95'][1])
                    for t in result['judges'].values() for k,threshold in (('target',.05),('overall',.02)))
    result['verdict'] = 'GO' if good else 'PARK-NO-SIGNAL' if uncertain else 'PARK-NO-GO'
    result['reasons'] = ['two-family conjunction' if good else 'predeclared conjunction not met']
    result['caveats'] = ['Nominal McNemar assumes independent items; LoCoMo is clustered.',
        'Combined guard does not certify the 25-row LME guard alone.', 'Synthetic/source admission is not extraction quality.']
    return result
