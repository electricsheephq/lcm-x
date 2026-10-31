"""Extraction repair/terminal regressions with fabricated CLI-shaped responses."""
import json
import subprocess
import pytest
from bench.fmp import models
from bench.fmp.tests.test_derived import bundle, cache, item, run, score, wire

@pytest.fixture
def stream(tmp_path):
    state=dict(sends=0, prompts=[], mode='int_window', repair=False)
    def runner(cmd, stdin=None, **kw):
        state['sends']+=1
        prompt=stdin.read()
        state['prompts'].append(prompt)
        batch=json.JSONDecoder().raw_decode(prompt[prompt.index('{"windows"'):])[0]['windows']
        out=[wire(w,[item()]) for w in batch]
        if not (state['repair'] and state['sends']==2):
            for raw in out:
                if state['mode']=='int_window':
                    raw['window_id']=int(raw['window_id'])
                elif state['mode']=='no_schema':
                    raw.pop('schema_version')
                elif state['mode']=='rel_null':
                    raw['relations']=None
                elif state['mode']=='items_dict':
                    raw['items']={}
        events=[dict(type='item.completed',item=dict(type='agent_message',text=json.dumps(out))),
                dict(type='turn.completed')]
        return subprocess.CompletedProcess([],0,'\n'.join(map(json.dumps,events)),'')
    backend=models.Codex(tmp_path/'transport',models.Quota(tmp_path/'quota.json'),runner=runner)
    original=backend.call
    def call(*args,**kw):
        response=original(*args,**kw)
        if state['mode']=='non_string' and not (state['repair'] and state['sends']==2):
            response['content']={'invalid':True}  # json.loads TypeError after cache admission
        return response
    backend.call=call
    yield backend,state
    backend.close()

@pytest.mark.parametrize('mode',['int_window','no_schema','rel_null','items_dict','non_string'])
def test_fix5_terminal_validation_resume(tmp_path, stream, mode):
    backend,state=stream
    state['mode']=mode
    run.freeze(tmp_path,bundle())
    results=[run.execute(tmp_path,backend) for _ in range(3)]
    assert all(r['verdict']=='PARK-INCOMPLETE' and r['reasons']==['extraction invalid after retry'] for r in results)
    extracted=cache.snapshot(tmp_path/'extraction.json')
    assert extracted['terminal_reason']=='extraction invalid after retry'
    assert all(w['state']=='failed' for w in extracted['ledger'])
    assert cache.snapshot(tmp_path/'report.json')==results[0]
    evaluation=cache.snapshot(tmp_path/'evaluation-inputs.json')
    assert score.evaluate(evaluation['entries'],[],**evaluation['parameters'])==results[0]
    assert state['sends']==2 and 'Repair invalid JSON' in state['prompts'][1]
    assert len(list((tmp_path/'responses').glob('*.send-*.json')))==2
    warm=dict(sends=state['sends'],prompts=list(state['prompts']))
    assert run.execute(tmp_path,backend)==results[0]
    assert state['sends']==warm['sends'] and state['prompts']==warm['prompts']

@pytest.mark.parametrize('mode',['int_window','no_schema','rel_null','items_dict','non_string'])
def test_fix5_validation_repair_succeeds(tmp_path, stream, mode):
    backend,state=stream
    state.update(mode=mode,repair=True)
    run.freeze(tmp_path,bundle())
    assert run.execute(tmp_path,backend,stage='extract')['status']=='COMPLETE'
    extracted=cache.snapshot(tmp_path/'extraction.json')
    assert 'terminal_reason' not in extracted and len(extracted['items'])==1
    assert state['sends']==2 and 'Repair invalid JSON' in state['prompts'][1]
    assert run.execute(tmp_path,backend,stage='extract')['status']=='COMPLETE'
    assert state['sends']==2
