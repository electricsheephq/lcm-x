"""Offline checks of the synthetic #2225 gate kit, never model calls."""
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter

from bench.instruments.cc_mcp_recall import kit as k

def test_corpus():
    assert len(k.C["facts"]) == 40 and len(k.C["superseded"]) == 4 and len(k.C["sessions"]) == 8
    assert Counter(f["class"] for f in k.C["facts"]) == dict(user_text=8, assistant=6, tool_input=6, tool_result_inline=8, tool_result_saved=4, pre_compact=4, subagent=2, resume_fork=2)
    sources = [p for s in k.C["sessions"] for p in s["turns"]] + [f.get("text", f.get("insert", {}).get("text", "")) for f in k.C["files"].values()]
    assert len({f["token"] for f in k.tokens()}) == 48
    for f in k.tokens():
        assert sum(s.count(f["token"]) for s in sources) == (0 if "derive" in f else 1)
        assert len(f["token"]) >= (5 if "derive" in f else 6)
        if "derive" in f:
            assert f["token"] == str(sum(f["derive"]["operands"]))
    for pair in k.C["superseded"]:
        assert int(pair["new_session"][1:]) > int(pair["old_session"][1:])
    assert Counter(q["kind"] for q in k.Q) == dict(positive=40, latest=4, negative=6)
    assert {q["fact"] for q in k.Q[:40]} == {f["id"] for f in k.C["facts"]}
    for q in k.Q:
        assert not re.search(r"S[1-8]|\d|[/\\]|\.(txt|log|env)", q["question"])
        assert all(f["token"] not in q["question"].upper() for f in k.tokens())
    for spec in k.C["files"].values():
        if "generate" in spec:
            text = k.log_text(spec)
            assert text == k.log_text(spec) and 790000 < len(text.encode()) < 820000
            assert [f["token"] for f in k.tokens() if f["token"] in text] == [spec["insert"]["text"].split()[-1]]
            assert text.count(spec["insert"]["text"].split()[-1]) == 1
    assert k.C["sessions"][6]["start"] == "resume:S1" and k.C["sessions"][7]["start"] == "fork:S2"
    assert "/compact" in k.C["sessions"][4]["turns"] and k.C["sessions"][4]["env"]

def test_judge():
    gold, other = k.Q[0]["gold"], k.Q[1]["gold"]
    assert k.judge('**' + gold.lower() + '**', k.Q[0])["correct"]
    assert not k.judge(gold + ' ' + other, k.Q[0])["correct"]
    assert not k.judge('X' + gold, k.Q[0])["correct"]
    assert k.judge('NOT FOUND', k.Q[-1])["correct"]
    assert k.judge('invented-seven', k.Q[-1])["fabricated"]
    assert not k.judge('', k.Q[0])["correct"]
    assert k.judge('', k.Q[-1]) == dict(correct=False, fabricated=False, verdict='error')
    assert k.judge(gold, k.Q[0], error=True)["verdict"] == 'miss'
    # Review round 1: hedges and invented values never pass; punctuation around a value does.
    assert not k.judge('NOT FOUND, but maybe ' + gold, k.Q[0])["correct"]
    assert not k.judge(gold + ' or ' + gold[:-1] + '9', k.Q[0])["correct"]
    assert not k.judge('NOT FOUND, maybe KESTREL-1234', k.Q[-1])["correct"]
    assert k.judge('NOT FOUND, maybe KESTREL-1234', k.Q[-1])["fabricated"]
    assert k.judge('Not found.', k.Q[-1])["correct"]
    assert k.judge('value=' + gold, k.Q[0])["correct"]
    assert k.judge(gold.replace('-', '\u2011'), k.Q[0])["correct"]
    numeric = next(q for q in k.Q if q["kind"] == "positive" and q["gold"].isdigit())
    assert k.judge('port: ' + numeric["gold"][:2] + ',' + numeric["gold"][2:], numeric)["correct"]
    pair = k.C['superseded'][0]
    assert [k.judge(a, k.Q[40])["verdict"] for a in (pair['new']['token'], pair['old']['token'], pair['new']['token'] + ' ' + pair['old']['token'], '')] == ['current', 'stale', 'both', 'miss']

def synthetic():
    manifest = {'sessions': {s['label']: {'session_id': 's1', 'audit': dict(rc=0, ok=True, verified=True)} for s in k.C['sessions']}}
    compacted = [f['id'] for f in k.C['facts'] if f['class'] == 'pre_compact']
    evidence = {s['label']: {'compact_boundaries': 1, 'compacted_facts': compacted, 'subagent_transcripts': 2, 'saved_outputs': 4} for s in k.C['sessions']}
    d = {'tokens': [{'id': f['id'], 'pass': True} for f in k.tokens()], 'evidence': evidence, 'plant_failed': []}
    t = {**copy.deepcopy(d), 'calls': [{'tool': 'lcm_grep', 'ms': 1}], 'empty': False}
    te = {'tokens': [r for r in copy.deepcopy(d['tokens']) if not r['id'].endswith('-old')], 'empty': True}
    a = {'answers': [{'id': q['id'], 'answer': q['gold'] or 'NOT FOUND', 'rc': 0, 'is_error': False, 'reader_not_connected': False, 'wall_s': .1, 'tool_calls': {'mcp__lcm-x__lcm_grep': 1}} for q in k.Q]}
    ae = {'answers': [{**r, 'answer': 'NOT FOUND'} for r in a['answers'][:40]]}
    return [manifest, d, t, te, a, None, ae]

def test_score():
    assert k.score(*synthetic())['verdict'] == 'PASS'
    latest = synthetic()
    latest[4]['answers'][40]['answer'] = 'NOT FOUND'
    assert k.score(*latest)['verdict'] == 'PASS'
    for index, verdict in [(1, 'STOP'), (2, 'FAIL'), (3, 'VOID')]:
        inputs = synthetic()
        inputs[index]['tokens'][0]['pass'] = False
        assert k.score(*inputs)['verdict'] == verdict
    inputs = synthetic()
    for r in inputs[4]['answers'][:5]:
        r['answer'] = 'NOT FOUND'
    result = k.score(*inputs)
    assert result['positives'] == [35] and result['second_run_required'] and result['verdict'] == 'SECOND RUN REQUIRED'
    inputs[5] = synthetic()[4]
    assert not k.score(*inputs)['bars']['A']
    for ms in ([1001] * 10, [1] * 99 + [5100]):
        inputs = synthetic()
        inputs[2]['calls'] = [{'tool': 'lcm_expand', 'ms': v} for v in ms]
        assert k.score(*inputs)['tripwire']
    inputs = synthetic()
    inputs[6]['answers'][0]['reader_not_connected'] = True
    assert k.score(*inputs)['verdict'] == 'VOID'
    # Bot review round 1 (#2517).
    # A corpus whose advertised shapes did not run (no compaction, no saved output) or whose value never reached its
    # session cannot pass.
    for label, shape in [(k.C['facts'][32]['session'], 'compact_boundaries'), (k.C['facts'][28]['session'], 'saved_outputs')]:
        inputs = synthetic()
        inputs[1]['evidence'][label][shape] = 0
        assert k.score(*inputs)['verdict'] == 'VOID'
    inputs = synthetic()
    inputs[1]['plant_failed'] = ['F01']
    assert k.score(*inputs)['verdict'] == 'VOID'
    # Bot review round 2: an empty-control receipt cannot stand in for the measured T arm; a saved-output fact that
    # sits in no saved output does not count as covered; T must measure the asset that captured.
    inputs = synthetic()
    inputs[2] = copy.deepcopy(inputs[3])
    inputs[2]['calls'] = [{'tool': 'lcm_grep', 'ms': 1}]
    assert k.score(*inputs)['verdict'] == 'VOID'
    inputs = synthetic()
    inputs[1]['tokens'][28]['in_saved_output'] = False
    assert k.score(*inputs)['verdict'] == 'VOID'
    inputs = synthetic()
    inputs[0]['bundle'], inputs[2]['bundle'] = {'core_commit': 'a'}, {'core_commit': 'b'}
    assert k.score(*inputs)['verdict'] == 'VOID'
    # A second run outside the 34-38 window changes nothing, even when it errors or misses.
    for broken in ({'is_error': True, 'rc': 1}, {'answer': 'NOT FOUND'}):
        inputs = synthetic()
        inputs[5] = synthetic()[4]
        inputs[5]['answers'][0].update(broken)
        assert k.score(*inputs)['verdict'] == 'PASS'
    # A T-pass/A-fail row needs a classification; a supplied one clears it, and so does a latency note.
    inputs = synthetic()
    inputs[4]['answers'][0]['answer'] = 'NOT FOUND'
    assert k.score(*inputs)['verdict'] == 'CLASSIFICATION REQUIRED'
    assert k.score(*inputs, classifications={'Q01': 'query wording'})['verdict'] == 'PASS'
    inputs = synthetic()
    inputs[2]['calls'] = [{'tool': 'lcm_grep', 'ms': 6000}]
    assert k.score(*inputs)['verdict'] == 'CLASSIFICATION REQUIRED'
    assert k.score(*inputs, classifications={'latency': 'shared-machine stall'})['verdict'] == 'PASS'

def test_tiny_store(tmp_path):
    cc, home, sid = k.package(), tmp_path / 'home', 's1'
    store_home = home / '.lcm-x/claude-code'
    k.empty_store(store_home)
    source = tmp_path / 's1.jsonl'
    content = 'The amber lantern freeze code is ' + k.C['facts'][0]['token']
    source.write_text(json.dumps(dict(type='user', uuid='u1', sessionId=sid, message=dict(role='user', content=content))) + '\n')
    store = cc.open_store(store_home / 'lcm.db', store_home)
    cc.drain(store, cc.Lineage(sid, None, source))
    cc.to_rollback_journal(store.connection)
    store.close()
    inputs = synthetic()
    inputs[0]['sessions']['S1']['session_id'] = 'wrong-session'
    assert not k.arm_d(home, inputs[0])['tokens'][0]['pass']
    inputs[0]['sessions']['S1']['session_id'] = sid
    d = k.arm_d(home, inputs[0])
    assert d['tokens'][0]['pass']
    assert d['tokens'][0]['matches'][0]['sha256'] == hashlib.sha256(content.encode()).hexdigest()
    t = k.arm_t(k.ROOT / 'adapters/claude-code', home, d)
    assert t['tokens'][0]['pass'] and t['calls']
    k.empty_store(tmp_path / 'empty/.lcm-x/claude-code')
    control = k.arm_t(k.ROOT / 'adapters/claude-code', tmp_path / 'empty', d, empty=True)
    assert len(control['tokens']) == 44 and all(r['pass'] for r in control['tokens'])
    inputs[1:3] = [d, t]
    for i, name in enumerate(('corpus-manifest', 'd', 't', 't-empty', 'a', 'a2', 'a-empty')):
        if inputs[i] is not None:
            k.write(tmp_path / (name + '.json'), inputs[i])
    command = [sys.executable, str(k.HERE / 'kit.py'), 'score', '--out', str(tmp_path / 'result')]
    command += [arg for name in ('corpus-manifest', 'd', 't', 't-empty', 'a', 'a-empty') for arg in ('--' + name, str(tmp_path / (name + '.json')))]
    subprocess.run(command, check=True)
    # One planted fact and no corpus files: the run did not exercise what it claims, so it is void, not a result.
    assert k.read(tmp_path / 'result/RESULT-MANIFEST.json')['verdict'] == 'VOID'

def test_arm_t_payload_hit(monkeypatch, tmp_path):
    # #2225: a fact held only in a saved tool output is found as a payload hit and must expand, by its ref, to the
    # payload D read; a payload with other bytes, or a hit from another session, does not count.
    fact = next(f for f in k.tokens() if f['class'] == 'tool_result_saved')
    payload = 'x' * 20000 + fact['token']
    d = {'tokens': [{'id': fact['id'], 'token': fact['token'], 'sessions': ['s4'], 'matches': [{'store_id': '9', 'session_id': 's4', 'role': 'tool', 'sha256': hashlib.sha256(payload.encode()).hexdigest(), 'externalized_ref': 'p1.json'}]}]}

    def run(served, session='s4'):
        class Fake:
            calls = []
            def __init__(self, *a):
                pass
            def rpc(self, *a):
                return {}
            def close(self):
                pass
            def call(self, name, args):
                if name == 'lcm_grep':
                    return {'results': [{'type': 'externalized', 'ref': 'p1.json', 'session_id': session, 'snippet': fact['token']}], 'partial': {'reasons': ['prefix']}}
                assert args == {'externalized_ref': 'p1.json', 'session_id': session, 'content_offset': 0, 'max_tokens': 50000}
                return {'content': served, 'has_more': False}
        monkeypatch.setattr(k, 'MCP', Fake)
        monkeypatch.setattr(k, 'server', lambda *a: None)
        return k.arm_t(tmp_path, tmp_path, d)['tokens'][0]

    row = run(payload)
    assert row['pass'] and row['via'] == ['payload'] and row['partial'] == {'reasons': ['prefix']}
    assert not run(payload + ' ')['pass']
    assert not run(payload, session='s5')['pass']


def test_shadow_review_round():
    # Codex shadow review of #2517. An invented alternative of any code length is a hedge.
    gold = k.Q[0]['gold']
    assert not k.judge(gold + ' or KESTREL-12345', k.Q[0])['correct']
    assert not k.judge(gold + ' or KESTREL-12', k.Q[0])['correct']
    # A compaction that came before the planted value does not cover it.
    inputs = synthetic()
    fact = next(f for f in k.C['facts'] if f['class'] == 'pre_compact')
    inputs[1]['evidence'][fact['session']]['compacted_facts'].remove(fact['id'])
    assert k.score(*inputs)['verdict'] == 'VOID'
    # A null or non-string classification never clears a gate.
    inputs = synthetic()
    inputs[4]['answers'][0]['answer'] = 'NOT FOUND'
    for note in (None, 0, '  '):
        assert k.score(*inputs, classifications={'Q01': note})['verdict'] == 'CLASSIFICATION REQUIRED'
    inputs = synthetic()
    inputs[2]['calls'] = [{'tool': 'lcm_grep', 'ms': 6000}]
    assert k.score(*inputs, classifications={'latency': None})['verdict'] == 'CLASSIFICATION REQUIRED'
    # Exactly one answer per question: a duplicate, even a correct one, leaves the run incomplete.
    inputs = synthetic()
    inputs[4]['answers'].append(dict(inputs[4]['answers'][-1]))
    assert k.score(*inputs)['verdict'] == 'INCOMPLETE'
    inputs = synthetic()
    inputs[6]['answers'].append(dict(inputs[6]['answers'][0]))
    assert k.score(*inputs)['verdict'] == 'VOID'


def test_compaction_must_follow_the_value(tmp_path, monkeypatch):
    fact = next(f for f in k.C['facts'] if f['class'] == 'pre_compact')
    transcript = tmp_path / 'main.jsonl'
    monkeypatch.setattr(k, 'session_files', lambda home, sid: [transcript])
    manifest = {'sessions': {fact['session']: {'session_id': 'x'}}}
    transcript.write_text('{"subtype": "compact_boundary"}\n' + fact['token'] + '\n')
    assert fact['id'] not in k.evidence(tmp_path, manifest)[fact['session']]['compacted_facts']
    transcript.write_text(fact['token'] + '\n{"subtype": "compact_boundary"}\n')
    assert fact['id'] in k.evidence(tmp_path, manifest)[fact['session']]['compacted_facts']
