"""DM0 frozen-hit instrument. Inventory is the default and never dispatches calls."""
import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from . import datasets, delivery, judges, models
from . import derived_delivery as memory
from . import derived_extract as extraction
from . import derived_panel as panel
from . import derived_score as scoring
from .closeout import card_tool
from .common import digest_file, load_lcm, sha
from .derived_cache import Cache, Deadline, Excluded, Paused, immutable, snapshot
from .dev import judge_messages
from .dev_cache import Incomplete, endpoint, request_identity, source_identity
from .dev_panel import encoded
READER = 'codex:gpt-6.1-sol@medium'
JUDGES = ('codex:gpt-6.1-sol@low','glm:glm-5.3')
ARMS = ('A0','M8192')
RULES = dict(target=.05, overall=.02, category=-.02, guard_strict=-.02, alpha=.05,
             bootstrap_draws=10000, schema='dm0-v1', dose=8192, multiplicity='Holm: target,overall per judge')
def baseline_messages(q, hits):
    return [dict(role='system',content=delivery.INSTRUCTIONS.format(date=q.question_date)+delivery.EVIDENCE),
            dict(role='user',content='History evidence:\n'+hits+'\nQuestion: '+q.question)]
def implementation():
    return {p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')}
def bind(bundle):
    if bundle['w1_cache'] is None:
        raise Incomplete('W1 LOCAL/25 cache required')
    rows=panel.local25(bundle['w1_receipt'],bundle['product'],bundle['w1_cache'])
    if {r['id'] for r in rows}!={r['id'] for r in bundle.get('w1_panel',{}).get('items',[])}:
        raise Incomplete('W1 panel identity mismatch')
    if bundle.get('added_receipt'):
        rows+=panel.local25(bundle['added_receipt'],bundle['product'],bundle['w1_cache'],population=None)
    verified={r['id']:r for r in rows}
    if len(verified)!=len(rows):
        raise Incomplete('duplicate baseline provenance')
    for raw in bundle['questions']:
        q=datasets.Question(**raw)
        r=verified.get(q.qid)
        if r is None:
            raise Incomplete('missing verified baseline ID')
        rec=json.loads((Path(bundle['w1_cache'])/'retrieval'/(r['cache_key']+'.json')).read_bytes())
        if (bundle['baselines'][q.qid]!=baseline_messages(q,rec['value']['evidence']) or
                rec['identity']['query_sha256']!=sha(q.question.encode()) or rec['identity']['date']!=q.question_date):
            raise Incomplete('A0 bytes/question identity mismatch')
        if q.qid in bundle.get('controls',{}):
            saved=bundle['controls'][q.qid]
            sources=dict(store=r['sources'],retrieval=rec['value']['source_sha256'])
            if bundle['w1_sources'][q.qid]!=sources:
                raise Incomplete('control source identity mismatch')
            original=json.loads((Path(bundle['w1_cache'])/'responses'/(sha(encoded(saved['identity']))+'.json')).read_bytes())
            if saved['identity']!=original['identity'] or saved['response']!=original['response']:
                raise Incomplete('control record mismatch')
            expected=request_identity(READER,bundle['baselines'][q.qid],endpoint=bundle['w1_endpoint'],effort='medium',temperature=None,
                max_tokens=32000,tools=None,tool_choice='none',rubric='fmp-single',renderer=bundle['w1_renderer'],sources=sources)
            memory.import_baseline(original,expected,bundle['baselines'][q.qid])
def freeze(root, bundle):
    root = Path(root)
    if bundle.get('schema') != 'dm0-v1' or not bundle.get('product') or not bundle.get('questions'):
        raise Incomplete('unresolved input identities')
    if not bundle.get('synthetic'):
        repeats=bundle.get('repeat_ids',[])
        if len(repeats)!=44 or len(set(repeats))!=44:
            raise Incomplete('repeat freeze requires 44 unique IDs')
        if not set(repeats)<={e['id'] for e in bundle['panel']}:
            raise Incomplete('repeat ID outside panel')
        for ident in repeats:
            if not {'identity','response','response_sha256','status'}<=bundle.get('controls',{}).get(ident,{}).keys():
                raise Incomplete('missing repeat control: '+ident)
            if (ident not in bundle.get('baselines',{}) or ident not in bundle.get('w1_sources',{}) or
                    not {'w1_endpoint','w1_renderer'}<=bundle.keys()):
                raise Incomplete('missing repeat baseline identity: '+ident)
        bind(bundle)
        if len(bundle['panel']) != 886 or len(bundle.get('repeat_ids',[])) != 44:
            raise Incomplete('DM0 panel/repeat freeze incomplete')
    if set(bundle['baselines']) != {e['id'] for e in bundle['panel']}:
        raise Incomplete('missing baseline')
    for q in bundle['questions']:
        memory.messages(bundle['baselines'][q['qid']],q['question'],'')
    immutable(root/'inputs.json',bundle)
    immutable(root/'run-manifest.json',dict(input_sha256=sha(encoded(bundle)),rules=RULES,reader=READER,judges=JUDGES,
        prompt_sha256=sha(extraction.PROMPT.encode()), schema='dm0-v1', implementation=implementation(),renderer=bundle['renderer'],endpoints=bundle['endpoints']))
def inputs(root):
    b = json.loads((root/'inputs.json').read_bytes())
    m = json.loads((root/'run-manifest.json').read_bytes())
    current=implementation()
    changed=sorted(k for k in set(m['implementation']) | set(current) if m['implementation'].get(k)!=current.get(k))
    if changed:
        raise Incomplete('implementation changed: '+', '.join(changed))
    if (m['input_sha256'] != sha(encoded(b)) or m['rules'] != RULES or m['prompt_sha256'] != sha(extraction.PROMPT.encode()) or
            m['renderer']!=b['renderer'] or m['endpoints']!=b['endpoints']):
        raise Incomplete('frozen input/source corruption')
    return b
def execute(root, backend, stage='all'):
    root = Path(root)
    b = inputs(root)
    if (root/'report.json').exists():
        for path in (root/'responses').glob('*.ok.json'):
            saved=json.loads(path.read_bytes())
            if sha(encoded(saved['response']))!=saved['response_sha256']:
                raise Incomplete('cache digest mismatch')
        return snapshot(root/'report.json')
    cache = Cache(root/'responses',backend,b['endpoints'],b['deadline'],b['renderer'])
    views = {sha(encoded(extraction.source_view(q))):extraction.source_view(q) for q in b['questions']}
    try:
        extracted = (snapshot(root/'extraction.json') if (root/'extraction.json').exists() else
                     extraction.execute(list(views.values()),cache,READER,root))
    except Paused:
        return dict(status='PAUSED',reason='paused: quota')
    if any(w['state']=='failed' for w in extracted['ledger']) and not extracted.get('terminal_reason'):
        return dict(verdict='PARK-INCOMPLETE',reasons=['unsettled extraction; resume same run'])
    rows, deliveries = [], []
    rejected=sum(w.get('rejected_count',0) for w in extracted['ledger'])
    total=sum(w.get('extracted_count',0) for w in extracted['ledger'])
    semantic=b.get('semantic_valid',False) and bool(extracted['items']) and bool(extracted['ledger']) and rejected/max(1,total)<=.05 and not extracted.get('terminal_reason')
    evaluation=dict(complete=True,semantic_valid=semantic,drift_valid=b.get('drift_valid',False) if b.get('synthetic') else False)
    evaluation.update(rejection_rate=rejected/max(1,total),rejected_items=rejected,extracted_items=total,exclusions=[])
    excluded_sources={w['source'] for w in extracted['ledger'] if w['state']=='excluded'}
    evaluation['exclusions']=[dict(id=q['qid'],reason='extraction transient retries exhausted') for q in b['questions']
        if extraction.source_view(q)['haystack_hash'] in excluded_sources and q['qid'] in {e['id'] for e in b['panel']}]
    if extracted.get('terminal_reason'):
        evaluation['incomplete_reason']=extracted['terminal_reason']
    ordered = sorted(b['panel'],key=lambda e:(not e['target'],e['id']))
    if not semantic or len(evaluation['exclusions'])>len(ordered)*.03:
        report=scoring.evaluate(ordered,rows,**evaluation)
        snapshot(root/'evaluation-inputs.json',dict(entries=ordered,parameters=evaluation))
        snapshot(root/'rows.json',rows)
        snapshot(root/'deliveries.json',deliveries)
        snapshot(root/'report.json',report)
        return report
    if stage=='extract':
        return dict(status='COMPLETE',windows=len(extracted['ledger']))
    try:
        if not b.get('synthetic'):
            evaluation['drift_valid']=drift(b,cache,root,evaluation['exclusions'])
        ordered = sorted(b['panel'],key=lambda e:(not e['target'],e['id']))
        qs = {q['qid']:datasets.Question(**q) for q in b['questions']}
        # Invalid drift settles below without spending any main-panel calls.
        for index,e in enumerate(ordered if evaluation['drift_valid'] else []):
            q=qs[e['id']]
            view=extraction.source_view(asdict(q))
            h=view['haystack_hash']
            group=[i for i in extracted['items'] if i['haystack_hash']==h]
            rel=[r for r in extracted['relations'] if r['from_id'] in {i['id'] for i in group}]
            block,omitted=memory.render(group,rel,q.question)
            a=b['baselines'][e['id']]
            m,meta=memory.messages(a,q.question,block)
            deliveries.append(dict(id=e['id'],omitted=omitted,**meta))
            for arm in (ARMS if index%2==0 else ARMS[::-1]):
                if e['id'] in {r['id'] for r in evaluation['exclusions']}:
                    break
                msgs=a if arm=='A0' else m
                calls=[]
                sources=dict(product=b['product'],source=h)
                saved=b.get('controls',{}).get(e['id']) if arm=='A0' or m==a else None
                if saved:
                    expected=request_identity(READER,a,endpoint=b['w1_endpoint'],effort='medium',temperature=None,
                        max_tokens=32000,tools=None,tool_choice='none',rubric='fmp-single',renderer=b['w1_renderer'],sources=b['w1_sources'][e['id']])
                    response=memory.import_baseline(saved,expected,a)
                else:
                    try:
                        response=cache.call(READER,msgs,calls,role='reader',rubric='fmp-single',sources=sources,max_tokens=32000)
                    except Excluded:
                        evaluation['exclusions'].append(dict(id=e['id'],reason='reader transient retries exhausted'))
                        break
                for judge in JUDGES:
                    jm,ds=judge_messages(q,response['content'])
                    try:
                        correct=verdict(cache,judge,jm,ds,calls,sources)
                    except Excluded:
                        evaluation['exclusions'].append(dict(id=e['id'],reason='judge transient retries exhausted'))
                        break
                    rows.append(dict(id=e['id'],arm=arm,judge=judge,correct=correct,answer_sha256=sha(encoded(response)),calls=list(calls)))
            if e['target'] and (index+1==len(ordered) or not ordered[index+1]['target']):
                interim=scoring.evaluate(ordered[:index+1],rows,**(evaluation | dict(complete=False)))
                immutable(root/'interim.json',interim)
                if interim['verdict']=='PARK-FUTILITY':
                    snapshot(root/'evaluation-inputs.json',dict(entries=ordered[:index+1],parameters=evaluation | dict(complete=False)))
                    snapshot(root/'rows.json',rows)
                    snapshot(root/'report.json',interim)
                    return interim
            if len({x['id'] for x in evaluation['exclusions']} & {e['id'] for e in ordered})>len(ordered)*.03:
                break
        report=scoring.evaluate(ordered,rows,**evaluation)
    except Paused:
        return dict(status='PAUSED',reason='paused: quota')
    except Deadline:
        evaluation.update(complete=False,drift_valid=False,incomplete_reason='deadline expired')
        report=scoring.evaluate(ordered,rows,**evaluation)
    except (Incomplete,ValueError,KeyError,TypeError):
        return dict(verdict='PARK-INCOMPLETE',reasons=['unsettled call; resume same run'])
    snapshot(root/'evaluation-inputs.json',dict(entries=ordered,parameters=evaluation))
    snapshot(root/'rows.json',rows)
    snapshot(root/'deliveries.json',deliveries)
    snapshot(root/'report.json',report)
    return report
def verdict(cache,judge,jm,ds,calls,sources):
    for replay in range(2):
        out=cache.call(judge,jm,calls,role='judge',rubric=sha(encoded(jm)),sources=sources,
            max_tokens=4096,temperature=None if ds.startswith('lme') else 0,replay=replay)
        correct=judges.parse(out['content'],ds)
        if correct is not None:
            break
    return correct
def drift(b,cache,root,exclusions=None):
    """Outcome-blind repeat with symmetric exclusions and per-arm invalid budgets."""
    exclusions=[] if exclusions is None else exclusions
    exclusions[:]=list({e['id']:e for e in exclusions}.values())
    if (root/'drift-admission.json').exists():
        saved=snapshot(root/'drift-admission.json')
        snapshot(root/'drift.json')
        exclusions[:]=list({e['id']:e for e in [*exclusions,*saved['exclusions']]}.values())
        return saved['valid']
    rows,drift_exclusions=[],[]
    extraction_excluded={e['id'] for e in exclusions}
    for ident in b['repeat_ids']:
        if ident in extraction_excluded:
            continue
        q=datasets.Question(**next(q for q in b['questions'] if q['qid']==ident))
        saved=b['controls'][ident]
        old=memory.import_baseline(saved,saved['identity'],b['baselines'][ident])
        calls=[]
        try:
            fresh=cache.call(READER,b['baselines'][ident],calls,role='reader',rubric='fmp-single',
                sources=dict(repeat=ident,product=b['product']),max_tokens=32000)
            for judge in JUDGES:
                pair=[]
                for response in (old,fresh):
                    jm,ds=judge_messages(q,response['content'])
                    pair.append(verdict(cache,judge,jm,ds,calls,dict(repeat=ident)))
                rows.append(dict(id=ident,judge=judge,pair=pair,flip=None if None in pair else pair[0]!=pair[1],
                    calls=[{k:v for k,v in c.items() if k!='cache'} for c in calls]))
        except Excluded:
            drift_exclusions.append(dict(id=ident,reason='drift transient retries exhausted'))
    exclusions.extend(drift_exclusions)
    excluded={e['id'] for e in exclusions}
    rows=[r for r in rows if r['id'] not in excluded]
    snapshot(root/'drift.json',rows)
    repeat_ids=set(b['repeat_ids'])
    n=len(repeat_ids-excluded)
    valid=n>0 and len(drift_exclusions)<=len(repeat_ids)*.03 and all(
        any(r['flip'] is not None for r in rows if r['judge']==j) and
        all(sum(r['pair'][a] is None for r in rows if r['judge']==j)<=max(1,math.ceil(n*.02)) for a in range(2)) and
        sum(r['flip'] is True for r in rows if r['judge']==j)/max(1,sum(r['flip'] is not None for r in rows if r['judge']==j))<=.05
        for j in JUDGES)
    snapshot(root/'drift-admission.json',dict(valid=valid,exclusions=drift_exclusions,
        extraction_skipped=len(extraction_excluded & repeat_ids)))
    return valid
def card(root):
    report=snapshot(root/'report.json')
    tool=card_tool()
    def invoke(*args):
        if tool.execute(tool.parser().parse_args([args[0],str(root),*args[1:]])):
            raise Incomplete('result card')
    if (root/'RESULT-MANIFEST.json').exists():
        invoke('check')
        return
    invoke('init','--project','lcm-x','--kind','benchmark','--owner-lane','dm0','--code-dir',str(Path(__file__).resolve().parents[2]),
           '--command','python -m bench.fmp.derived score --product PRODUCT --source INPUT --run RUN','--claim-class','advisory','--local-raw','false')
    (root/'RESULT.md').write_text('# DM0 instrument\n\n| Result | Receipt |\n|---|---|\n| '+report['verdict']+
        ' | report.json |\n\nAdmission: '+json.dumps({k:report.get(k) for k in ('rejection_rate','rejected_items','extracted_items','exclusions','invalid_verdicts')})+
        '\n\nPanel length: '+str(len(inputs(root)['panel']))+'; evaluated prefix length: '+str(len(snapshot(root/'evaluation-inputs.json')['entries']))+
        '\n\nRe-score: python -m bench.fmp.derived score --product PRODUCT --source inputs.json --run RUN\n\n'
        'Not proven: extraction quality, benchmark gain, issue acceptance, release/runtime/customer readiness.\nT1: unpublished. T2: unpublished.\n')
    for judge in JUDGES:
        invoke('headline','--benchmark',('DM0/synthetic' if inputs(root).get('synthetic') else 'DM0-886-v1')+'/'+judge,'--metric','go_to_stage1',
               '--value',str(int(report['verdict']=='GO')),'--n',str(report['judges'].get(judge,{}).get('overall',{}).get('n',0)),'--source','report.json')
    invoke('add','inputs.json','run-manifest.json','responses','extraction.json','evaluation-inputs.json','rows.json','report.json','RESULT.md','--role','per-item')
    for name in ('deliveries.json','interim.json','drift.json','drift-admission.json'):
        if (root/name).exists():
            invoke('add',name,'--role','per-item')
    invoke('rescore','--command','python -m bench.fmp.derived score --product PRODUCT --source inputs.json --run RUN','--input','inputs.json','evaluation-inputs.json','rows.json')
    invoke('finalize')
    invoke('check')
    invoke('status','settled')
def normalized(bundle, product):
    """Rebuild the panel from hash-pinned source files; never retrieve or model here."""
    lme=load_lcm(product)
    questions=[]
    if bundle['normalizer_sha256']!=digest_file(Path(datasets.__file__)):
        raise Incomplete('normalizer drift')
    for label,path in bundle['source_files'].items():
        if digest_file(path)!=bundle['dataset_sha256'][label]:
            raise Incomplete('dataset drift')
        raw=json.loads(Path(path).read_bytes())
        if label=='lme-s':
            questions.extend(datasets.normalize_lme(r,label,lme) for r in raw)
        else:
            for ci,item in enumerate(raw,1):
                for q in datasets.normalize_locomo(item,adversarial=True):
                    q.canonical_id=f'locomo-{ci}-q{q.qid.rsplit(":",1)[1]}'
                    questions.append(q)
    dev=bundle['dev_ids']
    histories=[panel.history(q) for q in questions if q.qid in dev or getattr(q,'canonical_id','').startswith(('locomo-1-q','locomo-2-q'))]
    entries=panel.select(questions,bundle['w1_panel'],dev,histories)
    by_id={getattr(q,'canonical_id',q.qid):q for q in questions}
    selected=[]
    for e in entries:
        q=asdict(by_id[e['id']])
        q['qid']=e['id']
        selected.append(q)
    return bundle | dict(panel=entries,questions=selected)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('inventory','freeze','extract','answer','score'),nargs='?',default='inventory')
    for name in ('product','source','run'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cache',type=Path)
    p.add_argument('--activate',action='store_true')
    a=p.parse_args()
    b=json.loads(a.source.read_bytes())
    try:
        if a.command=='inventory':
            result=panel.inventory(b.get('w1_receipt',b),b.get('product'),a.cache or a.source.parent)
            print(json.dumps(result))
            return 0 if result['status']=='COMPLETE' else 2
        if not b.get('synthetic'):
            head=subprocess.check_output(['git','-C',str(a.product),'rev-parse','HEAD'],text=True).strip()
            if head!=b['product_commit'] or source_identity(a.product)!=b['product']:
                raise Incomplete('product identity drift')
        if a.command=='freeze':
            freeze(a.run,b if b.get('synthetic') else normalized(b,a.product))
            return 0
        if a.command=='score':
            inputs(a.run)
            rows=snapshot(a.run/'rows.json')
            evaluation=snapshot(a.run/'evaluation-inputs.json')
            report=scoring.evaluate(evaluation['entries'],rows,**evaluation['parameters'])
            if report!=snapshot(a.run/'report.json'):
                raise Incomplete('re-score differs; use immutable successor evidence')
            card(a.run)
            return 0
        if not a.activate:
            raise Incomplete('separate live-run activation required')
        frozen=inputs(a.run)
        actual={'codex':endpoint(),'glm':{'transport':'claude-code/glm-plan','version':subprocess.check_output(['claude','--version'],text=True).strip()}}
        if frozen['endpoints']!=actual:
            raise Incomplete('endpoint identity drift')
        backend=models.Backends(a.run/'transport',models.Quota(a.run/'quota.json'),[READER,*JUDGES])
        try:
            result=execute(a.run,backend,'extract' if a.command=='extract' else 'all')
        finally:
            backend.close()
        print(json.dumps(dict(verdict=result.get('verdict'),status=result.get('status'))))
        return 0
    except (Incomplete,KeyError,ValueError) as exc:
        reason=str(exc) if str(exc).startswith(('implementation changed: ','repeat freeze','repeat ID','missing repeat')) else 'input, identity or activation preflight'
        print(json.dumps(dict(status='INCOMPLETE',reason=reason)))
        return 2
if __name__=='__main__':
    raise SystemExit(main())
