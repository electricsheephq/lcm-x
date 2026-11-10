"""#1188 unit 1: read-only anchor windows and opt-in recall guidance."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

from hermes_lcm import schemas, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from tests import test_lcm_recall, test_recall_session_delivery

recall_engine = test_lcm_recall.recall_engine

GOLDEN = Path(__file__).with_name('fixtures') / 'recall_anchor_main_golden.json'


@pytest.fixture
def orchard(recall_engine, monkeypatch):
    e = recall_engine
    e._config.embeddings_enabled = False
    monkeypatch.setattr(tools.time, 'time', lambda: 1700000000.0)
    for sid in ('a', 'b', 'c'):
        e._store.append(sid, {'role': 'user', 'content': 'orchard café 🍎', 'timestamp': 1})
    return e


def outputs(e):
    return {**{detail: tools.lcm_recall({'query': 'orchard', 'detail': detail}, engine=e)
               for detail in ('snippets', 'answer_ready')},
            'load_session': tools.lcm_load_session({'session_id': 'a'}, engine=e)}


def test_config_and_default_output_bytes(orchard, monkeypatch):
    # This control is captured on b1768887 before product edits.
    assert outputs(orchard) == json.loads(GOLDEN.read_text())['outputs']
    assert LCMConfig().recall_guidance == 'v1'
    assert LCMConfig().recall_session_hints is False
    monkeypatch.setenv('LCM_RECALL_GUIDANCE', 'v2')
    monkeypatch.setenv('LCM_RECALL_SESSION_HINTS', 'true')
    cfg = LCMConfig.from_env()
    assert cfg.recall_guidance == 'v2' and cfg.recall_session_hints
    monkeypatch.setenv('LCM_RECALL_GUIDANCE', 'invalid')
    assert LCMConfig.from_env().recall_guidance == 'v1'


def test_anchor_window_clamps_ownership_and_cursor(orchard, monkeypatch):
    e = orchard
    ids = []
    for i in range(100):
        # load_session filters write time; host timestamps are stored as observed_at.
        monkeypatch.setattr(tools.time, 'time', lambda i=i: float(i))
        ids.append(e._store.append('window', {'role': 'user', 'content': f'row {i}', 'timestamp': i + 1000}))
    changes = e._store.connection.total_changes
    def load(**kw):
        return json.loads(tools.lcm_load_session({'session_id': 'window', **kw}, engine=e))
    def row_ids(result):
        return [r['store_id'] for r in result['messages']]
    page = load(anchor_store_id=ids[50])
    assert row_ids(page) == ids[44:63]
    assert page['has_more'] and page['next_cursor'] == ids[62]
    assert row_ids(load(after_store_id=page['next_cursor'])) == ids[63:]
    assert row_ids(load(anchor_store_id=ids[50], before=99, after=99)) == ids[10:91]
    assert row_ids(load(anchor_store_id=ids[50], before=-1, after=0)) == ids[50:51]
    assert row_ids(load(anchor_store_id=ids[50], before=2, after=2,
                        roles=['user'], time_from=50, time_to=51)) == ids[50:52]
    assert row_ids(load(anchor_store_id=ids[50], before=2, after=2, limit=2)) == ids[48:50]
    for bad in (1, 99999, True, 'bad', None):
        rejected = load(anchor_store_id=bad)
        assert 'anchor_store_id' in rejected['error'] and 'messages' not in rejected
    for key in ('before', 'after'):
        assert 'error' in load(anchor_store_id=ids[50], **{key: None})
    assert [r['store_id'] for r in e._store.load_session_window(
        'window', anchor_store_id=ids[50], before=99, after=99)] == ids[38:63]
    assert e._store.connection.total_changes == changes


@pytest.mark.parametrize('filters', [
    {'roles': ['assistant']}, {'roles': ['tool']},
    {'time_from': 12, 'time_to': 14},
    {'roles': ['user'], 'time_from': 12, 'time_to': 14},
])
def test_anchor_filters_match_session_page(orchard, monkeypatch, filters):
    e = orchard
    for i in range(10, 17):
        monkeypatch.setattr(tools.time, 'time', lambda i=i: float(i))
        e._store.append('filtered', {'role': 'user' if i % 2 else 'assistant',
                                   'content': f'row {i}', 'timestamp': i + 1000})
    rows = e._store.load_session_page('filtered')
    changes = e._store.connection.total_changes
    actual = json.loads(tools.lcm_load_session(
        {'session_id': 'filtered', 'anchor_store_id': rows[3]['store_id'],
         'before': 3, 'after': 3, **filters}, engine=e))
    expected = e._store.load_session_page('filtered', **filters)
    assert [r['store_id'] for r in actual['messages']] == [r['store_id'] for r in expected]
    assert e._store.connection.total_changes == changes


def test_terminal_anchor_has_no_cursor(orchard):
    result = json.loads(tools.lcm_load_session(
        {'session_id': 'a', 'anchor_store_id': 1}, engine=orchard))
    assert result['messages'] and result['has_more'] is False
    assert result['next_cursor'] is None


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('ranked_count', [1, 2])
def test_v2_later_child_counts_and_displaced_continuation(orchard, monkeypatch, strict, ranked_count):
    e = orchard
    e._config.recall_guidance = 'v2'
    e._config.recall_reference_strict = strict
    monkeypatch.setattr(tools.time, 'time', lambda: 1700000200)
    a = e._store.append('A', {'role': 'user', 'content': 'needle My hobby is pottery.', 'timestamp': 1700000000})
    b = e._store.append('B', {'role': 'user', 'content': 'needle I prefer cycling.', 'timestamp': 1700000000})
    c = e._store.append('C', {'role': 'user', 'content': 'My hobby is now painting.', 'timestamp': 1700000100})
    original = tools._lcm_recall_fts_arm
    def ranked(engine, query, **kw):
        hits, error = original(engine, query, **kw)
        if query == 'needle':
            by_id = {h['store_id']: h for h in hits}
            hits = [by_id[sid] for sid in [a, b][:ranked_count]]
        return hits, error
    monkeypatch.setattr(tools, '_lcm_recall_fts_arm', ranked)
    changes = e._store.connection.total_changes
    result = json.loads(tools.lcm_recall(
        {'query': 'needle', 'detail': 'answer_ready', 'limit': 2}, engine=e, later_statements=1))
    assert [h['store_id'] for h in result['hits']] == [a, c]
    assert result['provenance']['later_statements']['displaced'] == ranked_count - 1
    signal = result['provenance']['answer_ready']
    assert signal['more_available'] is (ranked_count == 2)
    assert signal['sessions_delivered'] == 2
    assert signal['sessions_in_pool'] == ranked_count + 1
    assert e._store.connection.total_changes == changes


@pytest.mark.parametrize('strict', [False, True])
def test_v2_delta_seen_tail_is_not_more_available(orchard, strict):
    e = orchard
    e._config.recall_guidance = 'v2'
    e._config.recall_reference_strict = strict
    args = {'query': 'orchard', 'detail': 'answer_ready', 'limit': 3, 'seen_refs': []}
    initial = json.loads(tools.lcm_recall(args, engine=e))
    # Keep only the first-ranked ref novel; the probe encounters a seen ref.
    seen = [h['exact_ref'] for h in initial['hits'][1:]]
    result = json.loads(tools.lcm_recall({**args, 'limit': 1, 'seen_refs': seen}, engine=e))
    assert result['delta']['novel_ref_count'] == 1
    assert result['provenance']['answer_ready']['more_available'] is False


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('count,limit,more', [(0, 8, False), (2, 2, False), (3, 2, True), (20, 10, True)])
def test_v2_continuation_and_hydration_budget(orchard, strict, count, limit, more):
    e = orchard
    for i in range(count):
        e._store.append(f'pool-{i}', {'role': 'user', 'content': 'pear orchard', 'timestamp': 1})
    e._config.recall_guidance = 'v2'
    e._config.recall_reference_strict = strict
    args = {'query': 'pear', 'detail': 'answer_ready', 'limit': limit}
    result = json.loads(tools.lcm_recall(args, engine=e))
    signal = result['provenance']['answer_ready']
    assert signal['sessions_delivered'] == len({h['session_id'] for h in result['hits']})
    assert signal['sessions_in_pool'] == count
    assert signal['more_available'] is more
    e._config.recall_guidance = 'v1'
    old = json.loads(tools.lcm_recall(args, engine=e))['provenance']['answer_ready']
    assert not {'sessions_delivered', 'sessions_in_pool', 'more_available'} & old.keys()


@pytest.mark.parametrize('kind', ['excerpt', 'summary', 'lead'])
def test_rollover_hint_uses_anchor_owner(orchard, monkeypatch, kind):
    e = orchard
    store_id = 1
    node_id = e._dag.add_node(SummaryNode(session_id='a', depth=0, summary='orchard',
        token_count=1, source_token_count=2, source_ids=[store_id], source_type='messages'))
    e._dag.reassign_session_nodes('a', 'rollover')
    e._config.recall_session_hints = True
    e._config.embeddings_enabled = True
    e._config.recall_reference_strict = kind != 'summary'
    from types import SimpleNamespace
    monkeypatch.setattr(tools, '_resolve_recall_provider', lambda *a, **k: SimpleNamespace(
        provider_id='mock', model_id='mock', last_usage_tokens=0))
    monkeypatch.setattr(tools, '_lcm_grep_embed_query', lambda *a, **k: [1.0])
    def summary(*args, **kw):
        if kind == 'summary':
            return [{'kind': 'summary', 'node_id': node_id, 'session_id': 'rollover',
                     'store_id': store_id, 'snippet': 'orchard', 'timestamp': 1}], 'full', 1, 1, []
        hits, leads = tools._lcm_recall_summary_source_hits(
            e, [(e._dag.get_node(node_id), 1.0)], current='session-cur',
            candidate_limit=kw['candidate_limit'], lead_limit=kw['lead_limit'], deadline=kw['deadline'])
        if kind == 'excerpt':
            hits[0]['session_id'] = 'rollover'
        return hits, 'full', 1, 1, leads
    monkeypatch.setattr(tools, '_lcm_recall_summary_arm', summary)
    result = json.loads(tools.lcm_recall({'query': 'orchard', 'detail': 'answer_ready',
                                       'include': 'summaries'}, engine=e))
    hit = result['provenance']['answer_ready']['summary_leads'][0] if kind == 'lead' else result['hits'][0]
    hint = ast.parse(hit['session_hint'], mode='eval').body
    args = {kw.arg: ast.literal_eval(kw.value) for kw in hint.keywords}
    assert args == {'session_id': 'a', 'anchor_store_id': store_id}
    assert store_id in [r['store_id'] for r in json.loads(tools.lcm_load_session(args, engine=e))['messages']]
    assert tools._lcm_recall_session_hint(e, {'store_id': 99999}) is None
    assert 'session_hint' not in outputs(e)['snippets']


def test_v2_response_cap_counts_delivered_sessions(orchard, monkeypatch):
    for sid in ('a', 'b', 'c'):
        orchard._store.append(sid, {'role': 'user', 'content': 'pear ' + 'x' * 2000})
    orchard._config.recall_guidance = 'v2'
    monkeypatch.setattr(tools, '_LCM_RECALL_RESPONSE_CHAR_CAP', 4700)
    raw = tools.lcm_recall({'query': 'pear', 'detail': 'answer_ready'}, engine=orchard)
    result = json.loads(raw)
    signal = result['provenance']['answer_ready']
    assert signal['sessions_delivered'] == len({h['session_id'] for h in result['hits']}) < 3
    assert signal['sessions_in_pool'] == 3 and signal['more_available']
    assert len(raw) <= 4700


def test_guidance_selector_is_import_free_and_preserves_inputs():
    spec = importlib.util.spec_from_file_location('standalone_1188', schemas.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = json.dumps([schemas.LCM_RECALL, schemas.LCM_LOAD_SESSION])
    selected = module.recall_guidance_schemas([schemas.LCM_RECALL, schemas.LCM_LOAD_SESSION], 'v2')
    assert 'min_sessions' not in json.dumps(selected)
    assert 'anchor_store_id' in json.dumps(selected)
    assert json.dumps([schemas.LCM_RECALL, schemas.LCM_LOAD_SESSION]) == before
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom))
                   for n in ast.walk(ast.parse(Path(schemas.__file__).read_text())))


internal_recall = test_recall_session_delivery.internal_recall

@pytest.mark.parametrize("consumer", ["pack", "proposal", "auto", "requirements", "adaptive"])
def test_internal_evidence_ignores_guidance_and_hints(internal_recall, monkeypatch, consumer):
    from types import SimpleNamespace
    from hermes_lcm import answer_contract, evidence_compiler, evidence_pack
    e, plugin = internal_recall
    args = {"query": "needle", "detail": "answer_ready"}
    # Exercise every product-owned retrieval callback with real ranked recall.
    def consume(*a, retrieve, **kw):
        return {"context": retrieve(args)}
    monkeypatch.setattr(evidence_pack, "build_evidence_pack", consume)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", consume)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", consume)
    monkeypatch.setattr(answer_contract, "compile_answer_contract",
                        lambda *a: SimpleNamespace(status="planned"))
    e._config.preanswer_evidence_enabled = True
    e._config.preanswer_evidence_mode = "requirements_v1"
    e._adaptive_retrieval = SimpleNamespace(search=lambda dispatch, **kw:
        consume(retrieve=lambda recall_args: dispatch("lcm_recall", recall_args)))
    def invoke():
        if consumer == "adaptive":
            return json.loads(tools.lcm_retrieve({"action": "search"}, engine=e))
        if consumer == "requirements":
            return plugin._pre_llm_context(e, {"user_message": "needle", "baseline_refs": []})
        if consumer == "pack":
            return tools.lcm_evidence_pack({}, engine=e)
        return json.loads(tools.lcm_compile_evidence(
            {"question": "needle", "mode": consumer}, engine=e))
    e._config.recall_delivery_unit = "hits"
    expected = invoke()
    assert len(json.loads(expected["context"])["hits"]) == 1
    e._config.recall_guidance = "v2"
    e._config.recall_session_hints = True
    assert invoke() == expected
    external = json.loads(e.handle_tool_call("lcm_recall", args))
    assert external["hits"][0]["session_hint"]
    assert "more_available" in external["provenance"]["answer_ready"]


def test_proactive_and_baseline_disable_guidance(orchard, monkeypatch):
    from types import SimpleNamespace
    from hermes_lcm.engine import LCMEngine
    e = orchard
    e._config.recall_guidance = 'v2'
    e._config.recall_session_hints = True
    e._config.proactive_recall_enabled = True
    e._config.embeddings_enabled = True
    captured = []
    monkeypatch.setattr(tools, 'lcm_recall', lambda args, **kw:
                        captured.append(kw) or '{"hits": []}')
    fake = SimpleNamespace(_config=e._config, _proactive_recall_eligible=lambda: True,
                           _newest_user_message_text=LCMEngine._newest_user_message_text,
                           _proactive_recall_skipped_count=0)
    LCMEngine._build_proactive_recall_message(
        fake, [{'role': 'user', 'content': 'orchard'}], 'user', set())
    spec = importlib.util.spec_from_file_location('hermes_lcm.anchor_hook',
                                                 Path(tools.__file__).with_name('__init__.py'))
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    e._disabled_tool_names = lambda: set()
    e.handle_tool_call = lambda *a, **kw: LCMEngine.handle_tool_call(e, *a, **kw)
    plugin._answer_ready_baseline(e, 'orchard', {})
    assert len(captured) == 2
    assert all(kw.get('session_hints') is False and kw.get('recall_guidance') == 'v1'
               for kw in captured)
