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
    assert score.evaluate(entries,rows,exclusions=list(range(4)))['verdict']=='PARK-INCOMPLETE'
    rows[-1]['correct']=None
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
