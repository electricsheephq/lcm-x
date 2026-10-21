"""Provider-free DM0 contract probes; all text and identities are fabricated."""
import copy
import json
import time
from dataclasses import asdict
import pytest
from bench.fmp import derived as run
from bench.fmp import derived_cache as cache
from bench.fmp import derived_delivery as delivery
from bench.fmp import derived_extract as extract
from bench.fmp import derived_panel as panel
from bench.fmp import derived_score as score
from bench.fmp.common import sha
from bench.fmp.datasets import Question
from bench.fmp.dev import judge_messages
from bench.fmp.dev_cache import Incomplete, request_identity
from bench.fmp.dev_panel import encoded, fingerprint
def qid(i="q", **kw):
    return Question(i, "lme-s", "knowledge-update", "Which drink?", "tea", "2024-03-01", False,
        [dict(session_id="answer_leak", date="2024-01-01", turns=[dict(role="user", content="I prefer tea")])], **kw)
def item(local="a", quote="I prefer tea", **kw):
    return dict(local_id=local, kind="preference", subject_key="person:alex", subject_resolution="explicit",
        predicate_key="drink", object_value="tea", value_text="tea", polarity="positive", scope_key="morning",
        event_at=None, valid_from=None, valid_to=None, explicitness="explicit",
        sources=[dict(row_id="s0001/r0001", quote=quote)], **kw)
def wire(w, items=None, relations=None):
    return dict(schema_version="dm0-v1", haystack_hash=w["haystack_hash"], window_id=w["window_id"],
                items=items or [], relations=relations or [])
def window():
    return extract.windows(extract.source_view(asdict(qid())))[0]
class Fake:
    def __init__(self):
        self.calls = 0
    def call(self, spec, messages, records, **kw):
        self.calls += 1
        records.append(dict(status="ok", requested_model=spec, input_tokens=1, output_tokens=1, latency_s=.001))
        text = messages[-1]["content"]
        if kw["max_tokens"] == 16384:
            batches = json.loads(text)["windows"]
            content = json.dumps([wire(w, [item()]) for w in batches])
        elif "judge" in text.lower() or kw["max_tokens"] == 4096:
            content = "yes"
        else:
            content = "tea"
        return dict(content=content)
def cached(tmp_path, backend=None, **kw):
    return cache.Cache(tmp_path, backend or Fake(), {"codex": {"version": "fixture"}, "glm": {"version": "fixture"}},
                       time.time() + 60, {"renderer": "fixture"}, **kw)
def test_01_source_whitelist_and_cutoff():
    a = asdict(qid())
    b = copy.deepcopy(a)
    b.update(question="changed", gold="decoy", qtype="other", evidence_ids=["leak"])
    assert extract.source_view(a) == extract.source_view(b)
    assert "answer_leak" not in json.dumps(extract.source_view(a))
    for field, value in (("role", "assistant"), ("content", "I prefer coffee")):
        b = copy.deepcopy(a)
        b["sessions"][0]["turns"][0][field] = value
        assert extract.source_view(a) != extract.source_view(b)
    b = copy.deepcopy(a)
    b["sessions"][0]["date"] = "2025-01-01"
    assert extract.source_view(b)["rows"] == []
    b = copy.deepcopy(a)
    b["question_date"] = "2024-02-01"
    assert extract.source_view(a) != extract.source_view(b)
def test_02_unicode_split_quotes_and_sibling_filter():
    a = asdict(qid())
    text = "λ" * 25001 + "I prefer tea" + "尾" * 9000
    a["sessions"][0]["turns"][0]["content"] = text
    windows = extract.windows(extract.source_view(a), source_tokens=7000, context_tokens=100)
    w = next(w for w in windows if "I prefer tea" in json.dumps(w, ensure_ascii=False))
    result = extract.validate(wire(w, [item(), item("bad", "missing")]), w)
    assert len(result["items"]) == 1 and result["state"] == "partial"
    assert result["items"][0]["sources"][0]["start"] == 25001
    a["sessions"][0]["turns"][0]["content"] = "I prefer tea; I prefer tea"
    w = extract.windows(extract.source_view(a))[0]
    assert not extract.validate(wire(w, [item()]), w)["items"]
    assert not extract.validate(wire(w, [item("a"), item("bad", "missing")],
        [dict(from_local_id="a", to_local_id="bad", type="supersedes", evidence="explicit", sources=item()["sources"])]), w)["relations"]
def test_03_semantic_hard_cases():
    texts = [("request", "Please bring tea today"), ("quotation", 'He said "prefer tea"'),
             ("question", "Do I prefer tea?"), ("intention", "I plan to buy tea"),
             ("preference", "I now prefer tea"), ("status", "I finished buying tea")]
    for kind, text in texts:
        a = asdict(qid())
        a["sessions"][0]["turns"][0]["content"] = text
        w = extract.windows(extract.source_view(a))[0]
        i = item(quote=text)
        i["kind"] = kind
        got = extract.validate(wire(w, [i]), w)
        assert got["items"][0]["kind"] == kind
        if kind != "preference":
            i["kind"] = "preference"
            assert not extract.validate(wire(w, [i]), w)["items"]
def test_04_cache_identity_resume_and_uncertain(tmp_path):
    backend = Fake()
    c = cached(tmp_path, backend)
    args = dict(role="reader", rubric="r", sources={"hash": "a"}, max_tokens=32)
    msg = [dict(role="user", content="synthetic")]
    assert c.call(run.READER, msg, [], **args) == c.call(run.READER, msg, [], **args)
    assert backend.calls == 1
    for field in ("sources", "rubric"):
        c.call(run.READER, msg, [], **(args | {field: "changed"}))
    c.call("codex:gpt-6.1-sol@low", msg, [], **args)
    c.endpoints["codex"]["version"] = "v2"
    c.call(run.READER, msg, [], **args)
    assert backend.calls == 5
    c.call(run.READER, [dict(role='user',content='changed prompt')], [], **args)
    c.renderer={'schema':'changed'}
    c.call(run.READER,msg,[],**args)
    assert backend.calls==7
    backend.call = lambda *a, **kw: (_ for _ in ()).throw(TimeoutError())
    msg[0]["content"] = "uncertain"
    with pytest.raises(Incomplete):
        c.call(run.READER, msg, [], **args)
    backend.call = Fake().call
    with pytest.raises(Incomplete):
        c.call(run.READER, msg, [], **args)
def test_05_supersession_scopes_and_complete_group():
    w = window()
    old = extract.validate(wire(w, [item()]), w)["items"][0]
    new = copy.deepcopy(old)
    new.update(id="new", value_text="coffee", object_value="coffee", observed_at="2024-02-01")
    unrelated = [dict(old, id=str(i), subject_key=str(i)) for i in range(30)]
    block, omitted = delivery.render([old, *unrelated, new], [], "drink tea")
    assert old["id"] in block and "coffee" in block and "supersedes" not in block
    rel = dict(from_id="new", to_id=old["id"], type="supersedes", sources=old["sources"])
    block, _ = delivery.render([old, new], [rel], "drink")
    assert "supersedes" in block and "tea" in block and "coffee" in block
    assert delivery.render([old, new], [rel], "drink", cap=3)[0] == ""
    assert delivery.render([old, new], [rel], "drink", cap=3)[1]
def test_06_delivery_exact_bytes_and_import():
    q = qid()
    a = run.baseline_messages(q, "HIT α\n")
    assert delivery.messages(a, q.question, "")[0] == a
    m, meta = delivery.messages(a, q.question, "MEM")
    assert m[0] == a[0] and m[1]["content"].replace(delivery.HEADER + "MEM", "") == a[1]["content"]
    ident = request_identity(run.READER, a, endpoint={"version":"fixture"}, effort="medium", temperature=None,
        max_tokens=32000, tools=None, tool_choice="none", rubric="fmp-single", renderer={"r":"1"}, sources={"s":"1"})
    rec = dict(identity=ident, status="ok", response={"content":"tea"}, response_sha256=sha(encoded({"content":"tea"})))
    assert delivery.import_baseline(rec, ident, a)["content"] == "tea"
    with pytest.raises(Incomplete):
        delivery.import_baseline(rec, ident | {"output_limit": 1}, a)
    assert meta["hits_preserved"]
    w=window()
    assert extract.validate(wire(w),w)["state"]=="empty"
    unsupported=item()
    unsupported["value_text"]="unquoted coffee"
    assert not extract.validate(wire(w,[unsupported]),w)["items"]
def test_07_panel_mapping_exclusions_and_incomplete_local25():
    a = qid()
    b = qid("dev")
    c = qid("shared")
    loc = qid("sample:0")
    loc.dataset="locomo"
    loc.qtype="1"
    loc.canonical_id="locomo-3-q0"
    loc.sessions[0]["turns"][0]["content"] = "A different synthetic conversation"
    w1 = {"items": [dict(id=q.qid, sha256=fingerprint(q)) for q in (a,b,c)]}
    p = panel.select([a,b,c,loc], w1, ["dev"], [], strict=False)
    assert {e["id"] for e in p} == {"q", "shared", "locomo-3-q0"}
    assert panel.select([a,b,c,loc], w1, [], [panel.history(a)], strict=False) == [e for e in p if e["id"]=="locomo-3-q0"]
    with pytest.raises(Incomplete):
        panel.local25(dict(status="INCOMPLETE", error="Incomplete", frames=[]), "product")
    w1["items"][0]["sha256"] = "drift"
    with pytest.raises(Incomplete):
        panel.select([a,b,c,loc], w1, [], [], strict=False)
def test_08_judge_family_keys_blindness_and_dedup(tmp_path):
    q = qid()
    q.dataset="locomo"
    q.is_abstention=True
    q.gold="ADVERSARIAL-DECOY"
    jm, ds = judge_messages(q, "unknown")
    assert ds == "lme-s" and "ADVERSARIAL-DECOY" not in json.dumps(jm)
    backend = Fake()
    c = cached(tmp_path, backend)
    for spec in run.JUDGES:
        for _ in range(2):
            c.call(spec, jm, [], role="judge", rubric="blind", sources={}, max_tokens=4096)
    assert backend.calls == 2
def test_09_statistics_exact_cp_clusters_and_gate():
    entries = [dict(id=str(i), dataset="locomo", cluster="one", category="target", target=True, guard=False) for i in range(4)]
    rows = [dict(id=str(i), arm=arm, judge=j, correct=(arm=="M8192")) for i in range(4) for arm in run.ARMS for j in run.JUDGES]
    s = score.table(entries, rows, run.JUDGES[0])
    assert s["p"] == .125 and s["delta"] == 1 and s["clusters"] == 1
    assert -.02 < score.paired_bounds(0, 0, 195)[0] < 0
    assert score.paired_bounds(0, 0, 25)[0] < -.02
    assert score.holm([.025,.049]) == pytest.approx([.05,.05])
    assert score.evaluate(entries, rows, complete=False)["verdict"] == "PARK-INCOMPLETE"
    assert score.evaluate(entries, rows[:-1], complete=True)["verdict"] == "PARK-INCOMPLETE"
def bundle():
    qs = [qid("q1"), qid("q2")]
    return dict(schema="dm0-v1", synthetic=True, product="fixture", endpoints={"codex":{"version":"fixture"},"glm":{"version":"fixture"}},
        renderer={"r":"1"}, deadline=time.time()+60, semantic_valid=True, drift_valid=True,
        questions=[asdict(q) for q in qs], panel=[dict(id=q.qid,dataset=q.dataset,cluster=q.qid,
            category=q.qtype,target=True,guard=False) for q in qs],
        baselines={q.qid:run.baseline_messages(q,"HIT") for q in qs})

@pytest.fixture(autouse=True)
def fake_disk_and_wait(monkeypatch):
    monkeypatch.setattr(cache.shutil, 'disk_usage', lambda p: type('Disk', (), {'free':100*1024**3})())
    monkeypatch.setattr(cache.time, 'sleep', lambda seconds: None)

@pytest.mark.parametrize('returned,accepted', [(['glm-5.3'],True),(['wrong'],False),(['glm-5.3','wrong'],False)])
def test_fix1_glm_identity(tmp_path, returned, accepted):
    class GLM(Fake):
        def call(self, *args, **kw):
            response=super().call(*args, **kw)
            args[2][-1]['returned_model']=returned
            return response
    c=cached(tmp_path,GLM())
    def call():
        return c.call(run.JUDGES[1],[dict(role='user',content='judge')],[],role='judge',rubric='r',sources={},max_tokens=4096)
    if accepted:
        assert call()['content']=='yes'
    else:
        with pytest.raises(Incomplete):
            call()

@pytest.mark.parametrize('bad', [dict(value_text=5),dict(value_text=['tea']),dict(subject_key=None),None])
def test_fix1_malformed(bad):
    w=window()
    got=extract.validate(wire(w,[item(),item('bad') | bad if bad is not None else 5]),w)
    assert len(got['items'])==1 and got['rejected'][0]['reason']
    assert extract.validate(5,w)['rejected']

def test_fix1_partial_threshold(tmp_path):
    class Partial(Fake):
        def call(self,*args,**kw):
            response=super().call(*args,**kw)
            if kw['max_tokens']==16384:
                data=json.loads(response['content'])
                data[0]['items']=[item(str(i)) | dict(scope_key=str(i)) for i in range(20)]+[item('bad') | dict(value_text=5)]
                response['content']=json.dumps(data)
            return response
    run.freeze(tmp_path,bundle())
    report=run.execute(tmp_path,Partial())
    assert report['verdict']!='PARK-INCOMPLETE'
    assert report['rejection_rate']==pytest.approx(1/21)

def test_fix1_retry_and_resume(tmp_path,monkeypatch):
    delays=[]
    monkeypatch.setattr(cache.time,'sleep',delays.append)
    class Flaky(Fake):
        def call(self,*args,**kw):
            if not getattr(self,'tripped',False):
                self.tripped=True
                raise TimeoutError()
            return super().call(*args,**kw)
    run.freeze(tmp_path,bundle() | dict(deadline=time.time()+3600))
    backend=Flaky()
    assert run.execute(tmp_path,backend)['verdict']!='PARK-INCOMPLETE'
    assert delays==[60]
    warm=Fake()
    run.execute(tmp_path,warm)
    assert warm.calls==0

def test_fix1_parse_retry(tmp_path):
    class Invalid(Fake):
        def call(self,*args,**kw):
            response=super().call(*args,**kw)
            if kw['max_tokens']==4096 and not getattr(self,'tripped',False):
                self.tripped=True
                response['content']='yes and no'
            return response
    run.freeze(tmp_path,bundle())
    assert run.execute(tmp_path,Invalid())['verdict']!='PARK-INCOMPLETE'

@pytest.mark.parametrize('null_cache',[True,False])
def test_fix1_live_binding(tmp_path,monkeypatch,null_cache):
    repeats=['q1','q2']+[f'r{i}' for i in range(42)]
    b=bundle() | dict(synthetic=False,w1_cache=None if null_cache else str(tmp_path),repeat_ids=repeats,w1_receipt={},
        controls={i:dict(identity={},response={},response_sha256='fixture',status='ok') for i in repeats},
        w1_sources={i:{} for i in repeats},w1_endpoint={},w1_renderer={})
    b['panel']*=443
    b['panel'][:44]=[b['panel'][0] | dict(id=i) for i in repeats]
    b['baselines'].update({i:b['baselines']['q1'] for i in repeats[2:]})
    b['w1_panel']=dict(items=[dict(id='q1')])
    (tmp_path/'retrieval').mkdir()
    (tmp_path/'retrieval/k.json').write_text(json.dumps(dict(value=dict(evidence='WRONG'),identity=dict(
        query_sha256=sha(b'Which drink?'),date='2024-03-01'))))
    monkeypatch.setattr(panel,'local25',lambda *a,**kw:[dict(id='q1',delivered_sha256=sha(b'WRONG'),sources={},cache_key='k')])
    with pytest.raises(Incomplete,match='cache required|A0 bytes'):
        run.freeze(tmp_path,b)

def test_fix1_empty_control(tmp_path,monkeypatch):
    b=bundle()
    a=b['baselines']['q1']
    b.update(w1_endpoint={'v':'w1'},w1_renderer={'r':'w1'},w1_sources={'q1':{'s':'w1'}})
    ident=request_identity(run.READER,a,endpoint=b['w1_endpoint'],effort='medium',temperature=None,max_tokens=32000,
        tools=None,tool_choice='none',rubric='fmp-single',renderer=b['w1_renderer'],sources=b['w1_sources']['q1'])
    response=dict(content='historical')
    b['controls']={'q1':dict(identity=ident,status='ok',response=response,response_sha256=sha(encoded(response)))}
    monkeypatch.setattr(delivery,'render',lambda *a:('',[]))
    run.freeze(tmp_path,b)
    run.execute(tmp_path,Fake())
    rows=cache.snapshot(tmp_path/'rows.json')
    assert len({r['answer_sha256'] for r in rows if r['id']=='q1'})==1

def test_fix1_transport_pin(tmp_path,monkeypatch):
    run.freeze(tmp_path,bundle())
    original=run.digest_file
    monkeypatch.setattr(run,'digest_file',lambda p: 'changed' if p.name=='models.py' else original(p))
    with pytest.raises(Incomplete):
        run.inputs(tmp_path)

def test_fix1_conservative_label(monkeypatch):
    entries=[dict(id='q',target=True,guard=False,dataset='lme-s',category='target')]
    rows=[dict(id='q',arm=a,judge=j,correct=True) for a in run.ARMS for j in run.JUDGES]
    monkeypatch.setattr(score,'table',lambda *a:dict(n=1,delta=0,p=1,ci95=[0,.01],paired95=[0,.06]))
    assert score.evaluate(entries,rows)['verdict']=='PARK-NO-SIGNAL'

def test_fix1_exhaustion_and_symmetric_exclusion(tmp_path,monkeypatch):
    delays=[]
    monkeypatch.setattr(cache.time,'sleep',delays.append)
    class Down(Fake):
        def call(self,*args,**kw):
            self.calls+=1
            raise ConnectionError('transport')
    backend=Down()
    c=cached(tmp_path,backend)
    c.deadline=time.time()+3600
    args=dict(role='reader',rubric='r',sources={},max_tokens=32)
    for _ in range(2):
        with pytest.raises(Incomplete):
            c.call(run.READER,[dict(role='user',content='synthetic')],[],**args)
    assert backend.calls==4 and delays==[60,180,600]
    entries=[dict(id=str(i),target=True,guard=False,dataset='lme-s',category='target',cluster=str(i)) for i in range(100)]
    rows=[dict(id=e['id'],arm=a,judge=j,correct=True) for e in entries for a in run.ARMS for j in run.JUDGES]
    result=score.evaluate(entries,rows,exclusions=[dict(id='0',reason='transient')])
    assert result['verdict']!='PARK-INCOMPLETE'
    assert all(t['overall']['n']==99 for t in result['judges'].values())
def test_10_mock_e2e_snapshots_warm_resume_and_corruption(tmp_path):
    b=bundle()
    backend=Fake()
    run.freeze(tmp_path,b)
    first=run.execute(tmp_path,backend)
    n=backend.calls
    assert first["verdict"].startswith("PARK-") and n == 5
    assert run.execute(tmp_path,backend) == first and backend.calls == n
    run.card(tmp_path)
    assert (tmp_path/"RESULT-MANIFEST.json").is_file()
    assert (tmp_path/"inputs.json").is_file() and (tmp_path/"report.json").is_file()
    with pytest.raises(Incomplete):
        run.freeze(tmp_path,b | {"product":"changed"})
    data=json.loads((tmp_path/"inputs.json").read_bytes())
    data["product"]="corrupt"
    (tmp_path/"inputs.json").write_text(json.dumps(data))
    with pytest.raises(Incomplete):
        run.execute(tmp_path,backend)
def test_11_threshold_edges_and_two_judge_conjunction():
    good=dict(target=dict(delta=.05,p=.001,ci95=[.001,.10]),overall=dict(delta=.02,p=.002,ci95=[.001,.04]),
              guard=dict(ci95=[-.019,.01],paired95=[-.019,.01]),categories={'native':dict(delta=-.02)})
    assert score.passes(good)
    for field,value in (('target',.049999),('overall',.019999)):
        bad=copy.deepcopy(good)
        bad[field]['delta']=value
        assert not score.passes(bad)
    bad=copy.deepcopy(good)
    bad['guard']['paired95'][0]=-.02
    assert not score.passes(bad)
    bad=copy.deepcopy(good)
    bad['categories']['native']['delta']=-.020001
    assert not score.passes(bad)
    entries=[dict(id=str(i),dataset='lme-s',cluster=str(i),category='target',target=True,guard=False) for i in range(100)]
    rows=[dict(id=e['id'],arm=a,judge=j,correct=True) for e in entries for a in run.ARMS for j in run.JUDGES]
    assert score.evaluate(entries,rows,exclusions=[str(i) for i in range(4)])['verdict']=='PARK-INCOMPLETE'
    rows[-1]['correct']=None
    assert score.evaluate(entries,rows)['verdict']!='PARK-INCOMPLETE'
    for ident in ('0','1'):
        next(r for r in rows if r['id']==ident and r['judge']==run.JUDGES[1] and r['arm']=='M8192')['correct']=None
    assert score.evaluate(entries,rows)['verdict']=='PARK-INCOMPLETE'
def test_12_cli_score_activation_missing_baseline_and_deadline(tmp_path,monkeypatch):
    b=bundle()
    bad=copy.deepcopy(b)
    bad['baselines'].pop('q1')
    with pytest.raises(Incomplete):
        run.freeze(tmp_path/'missing',bad)
    run.freeze(tmp_path/'run',b)
    run.execute(tmp_path/'run',Fake())
    monkeypatch.setattr('sys.argv',['derived','score','--product','unused','--source',str(tmp_path/'run'/'inputs.json'),'--run',str(tmp_path/'run')])
    assert run.main()==0 and (tmp_path/'run'/'RESULT-MANIFEST.json').exists()
    before=(tmp_path/'run'/'RESULT-MANIFEST.json').read_bytes()
    assert run.main()==0 and (tmp_path/'run'/'RESULT-MANIFEST.json').read_bytes()==before
    monkeypatch.setattr('sys.argv',['derived','answer','--product','unused','--source',str(tmp_path/'run'/'inputs.json'),'--run',str(tmp_path/'run')])
    assert run.main()==2
    incomplete=bundle() | dict(synthetic=False,w1_receipt=dict(status='INCOMPLETE'),w1_cache='missing')
    with pytest.raises(Incomplete):
        run.freeze(tmp_path/'incomplete-local25',incomplete)
    expired=bundle()
    expired['deadline']=time.time()-1
    run.freeze(tmp_path/'expired',expired)
    backend=Fake()
    assert run.execute(tmp_path/'expired',backend)['verdict']=='PARK-INCOMPLETE' and backend.calls==0

@pytest.mark.parametrize('bad',[['x'],{'k':1}])
def test_fix2_local_id(bad):
    w=window()
    got=extract.validate(wire(w,[item(),item(bad)]),w)
    assert len(got['items'])==1 and len(got['rejected'])==1

def test_fix2_quota_pause_resume(tmp_path):
    from bench.fmp import models
    class QuotaFake(Fake):
        def __init__(self):
            super().__init__()
            self.quota=models.Quota(tmp_path/'quota.json')
        def call(self,spec,*args,**kw):
            self.quota.check(models.parse_spec(spec)[0])
            if not self.calls:
                self.calls+=1
                self.quota.stop('codex')
                raise models.BudgetStopped()
            return super().call(spec,*args,**kw)
    run.freeze(tmp_path,bundle())
    backend=QuotaFake()
    assert run.execute(tmp_path,backend).get('status')=='PAUSED'
    assert not (tmp_path/'report.json').exists()
    assert not list((tmp_path/'responses').glob('*.excluded.json'))
    assert run.execute(tmp_path,backend)['verdict']!='PARK-INCOMPLETE'

@pytest.mark.parametrize('status',[429,500,503,529])
def test_fix2_real_backend_status(tmp_path,status):
    import subprocess
    from bench.fmp import models
    def runner(*args,**kw):
        return subprocess.CompletedProcess([],1,json.dumps(dict(type='result',is_error=True,
            api_error_status=status,result='Service Unavailable')),'')
    backend=models.Claude(tmp_path,models.Quota(tmp_path/'quota.json'),kind='glm',runner=runner)
    import os
    old=os.environ.get('GLM_API_KEY')
    os.environ['GLM_API_KEY']='synthetic'
    try:
        c=cached(tmp_path/'cache',backend)
        c.deadline=time.time()+3600
        with pytest.raises(cache.Paused if status==429 else cache.Excluded):
            c.call(run.JUDGES[1],[dict(role='user',content='synthetic')],[],role='judge',rubric='r',sources={},max_tokens=32)
    finally:
        backend.close()
        if old is None:
            os.environ.pop('GLM_API_KEY',None)
        else:
            os.environ['GLM_API_KEY']=old

def test_fix2_drift_parse_and_exclusion(tmp_path):
    b=bundle() | dict(repeat_ids=['q1'])
    a=b['baselines']['q1']
    ident=request_identity(run.READER,a,endpoint={},effort='medium',temperature=None,max_tokens=32000,
        tools=None,tool_choice='none',rubric='fmp-single',renderer={},sources={})
    response=dict(content='historical')
    b['controls']={'q1':dict(identity=ident,status='ok',response=response,response_sha256=sha(encoded(response)))}
    class Invalid(Fake):
        def call(self,*args,**kw):
            r=super().call(*args,**kw)
            if kw['max_tokens']==4096:
                r['content']='yes and no'
            return r
    assert not run.drift(b,cached(tmp_path/'invalid',Invalid()),tmp_path)
    assert not run.drift(b,cached(tmp_path/'invalid',Fake()),tmp_path)
    assert (tmp_path/'drift.json').exists()

def test_fix2_extraction_exclusion_interim(tmp_path):
    b=bundle()
    b['questions'][1]['sessions'][0]['turns'][0]['content']='I prefer coffee'
    run.freeze(tmp_path,b)
    h=extract.source_view(b['questions'][1])['haystack_hash']
    w=extract.windows(extract.source_view(b['questions'][0]))[0]
    got=extract.validate(wire(w,[item()]),w)
    cache.snapshot(tmp_path/'extraction.json',dict(items=got['items'],relations=[],ledger=[
        dict(source=w['haystack_hash'],state='succeeded',extracted_count=1),dict(source=h,state='excluded')]))
    run.execute(tmp_path,Fake())
    assert (tmp_path/'interim.json').exists()

@pytest.mark.parametrize('kind,fatal,prelude', [
    ('codex','stream disconnected','Reconnecting: 403 Forbidden'),
    ('codex','status 503','Reconnecting: 403 Forbidden'),
    ('codex','stream disconnected',''),
    ('codex',None,'Reconnecting: stream disconnected'),
    ('codex',None,'Reconnecting: 403 Forbidden'),
    ('codex',None,'usage limit reconnect'),
    ('codex','Forbidden','Reconnecting: status 503'),
    ('codex','quota reached','Reconnecting: 403 Forbidden'),
    ('glm','HTTP 429 rate-limit',''), ('glm','API error: status=429 rate_limit','stderr'),
])
def test_fix3_fatal_outcome(tmp_path,monkeypatch,kind,fatal,prelude):
    import subprocess
    from bench.fmp import models
    monkeypatch.setenv('GLM_API_KEY','synthetic')
    events=[dict(type='error',message=prelude)] if prelude else []
    if kind=='codex':
        events += ([dict(type='turn.failed',error=dict(message=fatal))] if fatal else [
            dict(type='item.completed',item=dict(type='agent_message',text='yes')),dict(type='turn.completed')])
    elif prelude!='stderr':
        events=[dict(type='result',is_error=True,api_error_status=429,result=fatal)]
    else:
        events=[]
    def runner(*a,**kw):
        return subprocess.CompletedProcess([],int(fatal is not None),'\n'.join(map(json.dumps,events)),
            fatal if prelude=='stderr' else '')
    quota=models.Quota(tmp_path/'quota.json')
    backend=(models.Codex(tmp_path,quota,runner=runner) if kind=='codex' else
             models.Claude(tmp_path,quota,kind='glm',runner=runner))
    c=cached(tmp_path/'cache',backend)
    c.deadline=time.time()+3600
    args=dict(role='judge',rubric='r',sources={},max_tokens=32)
    try:
        if fatal is None:
            assert c.call(run.JUDGES[0],[],[],**args)['content']=='yes'
        else:
            expected=cache.Paused if kind=='glm' or 'quota' in fatal else Incomplete if fatal=='Forbidden' else cache.Excluded
            with pytest.raises(expected):
                c.call(run.JUDGES[1 if kind=='glm' else 0],[],[],**args)
            assert len(list((tmp_path/'cache').glob('*.retry-*.json')))==(3+1 if expected is cache.Excluded else 0)
    finally:
        backend.close()

@pytest.mark.parametrize('bad_count',[1,2])
def test_fix3_drift_budget_stops_main(tmp_path,monkeypatch,bad_count):
    b=bundle() | dict(synthetic=False,repeat_ids=[],controls={})
    qs=[qid(f'q{i:03d}') for i in range(44)]
    for q in qs:
        q.question=f'Which drink {q.qid}?'
    b.update(questions=[asdict(q) for q in qs],panel=[dict(id=q.qid,dataset=q.dataset,cluster=q.qid,
        category=q.qtype,target=True,guard=False) for q in qs],baselines={q.qid:run.baseline_messages(q,'HIT') for q in qs})
    for q in qs:
        a=b['baselines'][q.qid]
        ident=request_identity(run.READER,a,endpoint={},effort='medium',temperature=None,max_tokens=32000,
            tools=None,tool_choice='none',rubric='fmp-single',renderer={},sources={})
        response=dict(content='historical')
        b['controls'][q.qid]=dict(identity=ident,status='ok',response=response,response_sha256=sha(encoded(response)))
        b['repeat_ids'].append(q.qid)
    class Invalid(Fake):
        def call(self,spec,messages,records,**kw):
            out=super().call(spec,messages,records,**kw)
            if spec==run.JUDGES[1] and 'historical' in json.dumps(messages) and any(
                    f'Which drink q{i:03d}?' in json.dumps(messages) for i in range(bad_count)):
                out['content']='yes and no'
            return out
    backend=Invalid()
    assert run.drift(b,cached(tmp_path/'responses',backend),tmp_path)==(bad_count==1)
    if bad_count==2:
        monkeypatch.setattr(run,'inputs',lambda root:b)
        result=run.execute(tmp_path,backend)
        assert result['verdict']=='PARK-INCOMPLETE' and cache.snapshot(tmp_path/'rows.json')==[]
        assert backend.calls==44*5+bad_count+1  # drift + parse replays + extraction; zero main sends
        assert (tmp_path/'report.json').exists()
        before=backend.calls
        assert run.execute(tmp_path,backend)==result and backend.calls==before

@pytest.mark.parametrize('module',['judges.py','prompts.py'])
def test_fix3_implementation_resume(tmp_path,monkeypatch,module):
    import shutil
    source=tmp_path/'modules'
    source.mkdir()
    for p in __import__('pathlib').Path(run.__file__).parent.glob('*.py'):
        shutil.copy2(p,source/p.name)
    monkeypatch.setattr(run,'__file__',str(source/'derived.py'))
    run.freeze(tmp_path/'run',bundle())
    assert run.inputs(tmp_path/'run')
    with (source/module).open('a') as stream:
        stream.write('\n# changed fixture\n')
    with pytest.raises(Incomplete,match=module):
        run.inputs(tmp_path/'run')
    run.freeze(tmp_path/'fresh',bundle())
    assert run.inputs(tmp_path/'fresh')

def test_fix3_population_exclusions():
    entries=[dict(id=str(i),target=True,guard=False,dataset='lme-s',category='target',cluster=str(i)) for i in range(100)]
    rows=[dict(id=e['id'],arm=a,judge=j,correct=True) for e in entries for a in run.ARMS for j in run.JUDGES]
    exclusions=[dict(id=f'drift{i}') for i in range(4)]
    assert score.evaluate(entries,rows,complete=False,exclusions=exclusions)['verdict']=='PARK-FUTILITY'
