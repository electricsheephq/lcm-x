"""Final DM0 admission/settlement regressions; real CLI shapes, fabricated text."""
import copy
import json
import subprocess
import pytest
from bench.fmp import models
from bench.fmp.tests.test_derived import bundle, cache, item, run, score, wire
from bench.fmp.dev_cache import Incomplete

@pytest.fixture
def stream(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.shutil, 'disk_usage', lambda p: type('D', (), {'free':100*1024**3})())
    state=dict(sends=0, mode='valid')
    def runner(cmd, stdin=None, **kw):
        state['sends']+=1
        prompt=stdin.read()
        if state['mode']=='transient':
            events=[dict(type='turn.failed',error=dict(message='status 503'))]
            return subprocess.CompletedProcess([],1,json.dumps(events[0]),'')
        text='yes'
        if '"windows"' in prompt:
            batch=json.JSONDecoder().raw_decode(prompt[prompt.index('{"windows"'):])[0]['windows']
            items=[] if state['mode']=='empty' else [item()]
            if state['mode']=='rejected':
                items+=[item('bad') | dict(value_text='unquoted')]
            text='invalid' if state['mode']=='invalid' else json.dumps([wire(w,items) for w in batch])
        events=[dict(type='item.completed',item=dict(type='agent_message',text=text)),dict(type='turn.completed')]
        return subprocess.CompletedProcess([],0,'\n'.join(map(json.dumps,events)),'')
    backend=models.Codex(tmp_path/'transport',models.Quota(tmp_path/'quota.json'),runner=runner)
    yield backend,state
    backend.close()

def test_fix4_terminal_extraction_resume(tmp_path, stream):
    backend,state=stream
    state['mode']='invalid'
    run.freeze(tmp_path,bundle())
    result=run.execute(tmp_path,backend)
    assert result['reasons']==['extraction invalid after retry']
    assert cache.snapshot(tmp_path/'extraction.json')['terminal_reason']==result['reasons'][0]
    assert cache.snapshot(tmp_path/'report.json')==result
    evaluation=cache.snapshot(tmp_path/'evaluation-inputs.json')
    assert score.evaluate(evaluation['entries'],[],**evaluation['parameters'])==result
    assert state['sends']==2
    assert run.execute(tmp_path,backend)==result and state['sends']==2

@pytest.mark.parametrize('fault,reason', [('duplicate','44 unique'),('outside','repeat ID outside panel'),
    ('control','missing repeat control'),('identity','missing repeat baseline identity')])
def test_fix4_repeat_freeze_admission(tmp_path, monkeypatch, fault, reason):
    b=bundle() | dict(synthetic=False, repeat_ids=[f'r{i}' for i in range(44)],controls={},
        w1_endpoint={},w1_renderer={},w1_sources={})
    sample=b['panel'][0]
    b['panel']=[sample | dict(id=f'r{i}') for i in range(886)]
    b['baselines']={e['id']:b['baselines']['q1'] for e in b['panel']}
    for i,q in enumerate(b['questions']):
        q['qid']=f'r{i}'
    for ident in b['repeat_ids']:
        b['controls'][ident]=dict(identity={},response={},response_sha256='fixture',status='ok')
        b['w1_sources'][ident]={}
    if fault=='duplicate':
        b['repeat_ids'][-1]=b['repeat_ids'][0]
    if fault=='outside':
        b['repeat_ids'][-1]='outside'
    if fault=='control':
        b['controls'].pop('r0')
    if fault=='identity':
        b['w1_sources'].pop('r0')
    monkeypatch.setattr(run,'bind',lambda b:None)  # isolate repeat admission from LOCAL/25 disk provenance
    with pytest.raises(Incomplete,match=reason):
        run.freeze(tmp_path,b)
    assert not (tmp_path/'inputs.json').exists()

@pytest.mark.parametrize('excluded',[True,False])
def test_fix4_frozen_category_coverage(excluded):
    entries=[dict(id=str(i),target=True,guard=False,dataset='lme-s',category='rare' if i==0 else 'common',cluster=str(i)) for i in range(100)]
    rows=[dict(id=e['id'],arm=a,judge=j,correct=None if e['id']=='0' and not excluded else True)
          for e in entries for a in run.ARMS for j in run.JUDGES]
    result=score.evaluate(entries,rows,exclusions=['0'] if excluded else [])
    assert result['verdict']=='PARK-INCOMPLETE' and 'category' in ' '.join(result['reasons'])

@pytest.mark.parametrize('mode',['false','absent','empty','rejected'])
@pytest.mark.parametrize('stage',['all','extract'])
def test_fix4_semantic_stops_before_drift(tmp_path, monkeypatch, stream, mode, stage):
    backend,state=stream
    b=bundle()
    if mode=='false':
        b['semantic_valid']=False
    if mode=='absent':
        b.pop('semantic_valid')
    state['mode']=mode
    run.freeze(tmp_path,b)
    frozen=copy.deepcopy(b) | dict(synthetic=False)
    monkeypatch.setattr(run,'inputs',lambda root:frozen)
    drift_calls=[]
    monkeypatch.setattr(run,'drift',lambda *a:drift_calls.append(a) or True)
    result=run.execute(tmp_path,backend,stage)
    assert result['verdict']=='PARK-INCOMPLETE'
    sends=2 if mode=='rejected' else 1  # Rejected items now receive one repair attempt.
    assert not drift_calls and state['sends']==sends  # extraction only; zero drift/main sends
    assert cache.snapshot(tmp_path/'rows.json')==[]
    assert cache.snapshot(tmp_path/'report.json')==result
    assert run.execute(tmp_path,backend)==result and state['sends']==sends

@pytest.mark.parametrize('attempt',[1,2,3])
def test_fix4_deadline_after_retry_sleep(tmp_path, monkeypatch, stream, attempt):
    backend,state=stream
    state['mode']='transient'
    clock=[1000.]
    monkeypatch.setattr(cache.time,'time',lambda:clock[0])
    sleeps=[]
    def sleep(delay):
        sleeps.append(delay)
        clock[0]=10000. if len(sleeps)==attempt else clock[0]+delay
    monkeypatch.setattr(cache.time,'sleep',sleep)
    c=cache.Cache(tmp_path/'responses',backend,{'codex':{'version':'fixture'}},5000.,{})
    with pytest.raises(Incomplete):
        c.call(run.READER,[],[],role='reader',rubric='r',sources={},max_tokens=32)
    assert state['sends']==attempt
    assert len(list(c.root.glob('*.send-*.json')))==attempt
    assert sleeps==[60,180,600][:attempt]
