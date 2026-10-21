"""DM0 frozen-hit instrument. Inventory is the default and never dispatches calls."""
import argparse
import json
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
from .derived_cache import Cache, immutable, snapshot
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
def freeze(root, bundle):
    root = Path(root)
    if bundle.get('schema') != 'dm0-v1' or not bundle.get('product') or not bundle.get('questions'):
        raise Incomplete('unresolved input identities')
    if not bundle.get('synthetic'):
        panel.local25(bundle['w1_receipt'],bundle['product'],bundle['w1_cache'])
        if len(bundle['panel']) != 886 or len(bundle.get('repeat_ids',[])) != 44:
            raise Incomplete('DM0 panel/repeat freeze incomplete')
    if set(bundle['baselines']) != {e['id'] for e in bundle['panel']}:
        raise Incomplete('missing baseline')
    for q in bundle['questions']:
        memory.messages(bundle['baselines'][q['qid']],q['question'],'')
    immutable(root/'inputs.json',bundle)
    immutable(root/'run-manifest.json',dict(input_sha256=sha(encoded(bundle)),rules=RULES,reader=READER,judges=JUDGES,
        prompt_sha256=sha(extraction.PROMPT.encode()), schema='dm0-v1', implementation={p.name:digest_file(p) for p in Path(__file__).parent.glob('derived*.py')}))
def inputs(root):
    b = json.loads((root/'inputs.json').read_bytes())
    m = json.loads((root/'run-manifest.json').read_bytes())
    if (m['input_sha256'] != sha(encoded(b)) or m['rules'] != RULES or m['prompt_sha256'] != sha(extraction.PROMPT.encode()) or
            m['implementation'] != {p.name:digest_file(p) for p in Path(__file__).parent.glob('derived*.py')}):
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
    extracted = (snapshot(root/'extraction.json') if (root/'extraction.json').exists() else
                 extraction.execute(list(views.values()),cache,READER,root))
    if stage=='extract':
        return dict(status='COMPLETE',windows=len(extracted['ledger']))
    rows, deliveries = [], []
    semantic=b.get('semantic_valid',False) and bool(extracted['items']) and bool(extracted['ledger']) and all(w['state'] in ('succeeded','empty') for w in extracted['ledger'])
    evaluation=dict(complete=True,semantic_valid=semantic,drift_valid=b.get('drift_valid',False) if b.get('synthetic') else False)
    ordered = sorted(b['panel'],key=lambda e:(not e['target'],e['id']))
    try:
        if not b.get('synthetic'):
            evaluation['drift_valid']=drift(b,cache,root)
            if not evaluation['drift_valid']:
                raise Incomplete('historical control drift')
        ordered = sorted(b['panel'],key=lambda e:(not e['target'],e['id']))
        qs = {q['qid']:datasets.Question(**q) for q in b['questions']}
        for index,e in enumerate(ordered):
            q=qs[e['id']]
            view=extraction.source_view(asdict(q))
            h=view['haystack_hash']
            group=[i for i in extracted['items'] if i['haystack_hash']==h]
            rel=[r for r in extracted['relations'] if r['from_id'] in {i['id'] for i in group}]
            block,omitted=memory.render(group,rel,q.question)
            if omitted and not block:
                evaluation['semantic_valid']=False
            a=b['baselines'][e['id']]
            m,meta=memory.messages(a,q.question,block)
            deliveries.append(dict(id=e['id'],omitted=omitted,**meta))
            for arm in (ARMS if index%2==0 else ARMS[::-1]):
                msgs=a if arm=='A0' else m
                calls=[]
                sources=dict(product=b['product'],source=h)
                saved=b.get('controls',{}).get(e['id']) if arm=='A0' else None
                if saved:
                    expected=request_identity(READER,a,endpoint=b['w1_endpoint'],effort='medium',temperature=None,
                        max_tokens=32000,tools=None,tool_choice='none',rubric='fmp-single',renderer=b['w1_renderer'],sources=b['w1_sources'][e['id']])
                    response=memory.import_baseline(saved,expected,a)
                else:
                    response=cache.call(READER,msgs,calls,role='reader',rubric='fmp-single',sources=sources,max_tokens=32000)
                for judge in JUDGES:
                    jm,ds=judge_messages(q,response['content'])
                    judgment=cache.call(judge,jm,calls,role='judge',rubric=sha(encoded(jm)),sources=sources,
                        max_tokens=4096,temperature=None if ds.startswith('lme') else 0)
                    correct=judges.parse(judgment['content'],ds)
                    if correct is None:
                        raise Incomplete('invalid required judge')
                    rows.append(dict(id=e['id'],arm=arm,judge=judge,correct=correct,answer_sha256=sha(encoded(response)),calls=list(calls)))
            if e['target'] and (index+1==len(ordered) or not ordered[index+1]['target']):
                interim=scoring.evaluate(ordered[:index+1],rows,**(evaluation | dict(complete=False)))
                immutable(root/'interim.json',interim)
                if interim['verdict']=='PARK-FUTILITY':
                    snapshot(root/'evaluation-inputs.json',dict(entries=ordered[:index+1],parameters=evaluation | dict(complete=False)))
                    snapshot(root/'rows.json',rows)
                    snapshot(root/'report.json',interim)
                    return interim
        report=scoring.evaluate(ordered,rows,**evaluation)
    except (Incomplete,ValueError,KeyError,TypeError):
        evaluation['semantic_valid']=False
        report=scoring.evaluate(ordered,rows,**evaluation)
    snapshot(root/'evaluation-inputs.json',dict(entries=ordered,parameters=evaluation))
    snapshot(root/'rows.json',rows)
    snapshot(root/'deliveries.json',deliveries)
    snapshot(root/'report.json',report)
    return report
def drift(b,cache,root):
    """Outcome-blind 44-row repeat; stale historical controls never promote."""
    rows=[]
    for ident in b['repeat_ids']:
        raw=next(q for q in b['questions'] if q['qid']==ident)
        q=datasets.Question(**raw)
        saved=b['controls'][ident]
        old=memory.import_baseline(saved,saved['identity'],b['baselines'][ident])
        calls=[]
        fresh=cache.call(READER,b['baselines'][ident],calls,role='reader',rubric='fmp-single',
            sources=dict(repeat=ident,product=b['product']),max_tokens=32000)
        for judge in JUDGES:
            pair=[]
            for response in (old,fresh):
                jm,ds=judge_messages(q,response['content'])
                out=cache.call(judge,jm,calls,role='judge',rubric=sha(encoded(jm)),sources=dict(repeat=ident),
                    max_tokens=4096,temperature=None if ds.startswith('lme') else 0)
                pair.append(judges.parse(out['content'],ds))
            if None in pair:
                raise Incomplete('drift judge incomplete')
            rows.append(dict(id=ident,judge=judge,flip=pair[0]!=pair[1],calls=list(calls)))
    snapshot(root/'drift.json',rows)
    return len(rows)==88 and all(sum(r['flip'] for r in rows if r['judge']==j)/44<=.05 for j in JUDGES)
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
        ' | report.json |\n\nRe-score: python -m bench.fmp.derived score --product PRODUCT --source inputs.json --run RUN\n\n'
        'Not proven: extraction quality, benchmark gain, issue acceptance, release/runtime/customer readiness.\nT1: unpublished. T2: unpublished.\n')
    invoke('headline','--benchmark','DM0/synthetic' if inputs(root).get('synthetic') else 'DM0-886-v1','--metric','go_to_stage1',
           '--value',str(int(report['verdict']=='GO')),'--n',str(len(inputs(root)['panel'])),'--source','report.json')
    invoke('add','inputs.json','run-manifest.json','responses','extraction.json','evaluation-inputs.json','rows.json','report.json','RESULT.md','--role','per-item')
    for name in ('deliveries.json','interim.json','drift.json'):
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
    except (Incomplete,KeyError,ValueError):
        print(json.dumps(dict(status='INCOMPLETE',reason='input, identity or activation preflight')))
        return 2
if __name__=='__main__':
    raise SystemExit(main())
