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
    # Keep only the first-ranked ref novel; every unshipped tail ref is seen.
    seen = [h['exact_ref'] for h in initial['hits'][1:]]
    result = json.loads(tools.lcm_recall({**args, 'limit': 1, 'seen_refs': seen}, engine=e))
    assert result['delta']['novel_ref_count'] == 1
    assert result['provenance']['answer_ready']['more_available'] is False


def test_v2_strict_delta_session_cap_reports_continuation(orchard):
    e = orchard
    e._config.recall_guidance = 'v2'
    e._config.recall_reference_strict = True
    ids = [e._store.append('cap', {'role': 'user', 'content': f'needle row {i}'})
           for i in range(7)]
    changes = e._store.connection.total_changes
    args = {'query': 'needle', 'detail': 'answer_ready', 'seen_refs': [], 'limit': 10}
    first = json.loads(tools.lcm_recall(args, engine=e))
    assert len(first['hits']) == 5
    assert first['provenance']['answer_ready']['more_available'] is True
    second = json.loads(tools.lcm_recall({**args, 'seen_refs': first['delta']['novel_refs']}, engine=e))
    assert len(second['hits']) == 2
    assert {h['store_id'] for h in first['hits'] + second['hits']} == set(ids)
    assert second['provenance']['answer_ready']['more_available'] is False
    assert e._store.connection.total_changes == changes


def test_v2_strict_delta_post_budget_rejection_reports_continuation(orchard, monkeypatch):
    e = orchard
    e._config.recall_guidance = 'v2'
    e._config.recall_reference_strict = True
    ids = [e._store.append(f'deferred-{i}', {'role': 'user', 'content': f'needle row {i}'})
           for i in range(26)]
    original = tools._lcm_recall_fts_arm
    def ranked(*a, **kw):
        hits, error = original(*a, **kw)
        hits.sort(key=lambda h: h['store_id'])
        # The stored row is valid, but this carried excerpt cannot be verified
        # once the selector has spent all 25 hydration slots.
        hits[-1].update(snippet='absent excerpt', _fts_snippet='absent excerpt')
        return hits, error
    monkeypatch.setattr(tools, '_lcm_recall_fts_arm', ranked)
    changes = e._store.connection.total_changes
    args = {'query': 'needle', 'detail': 'answer_ready', 'seen_refs': [], 'limit': 25}
    first = json.loads(tools.lcm_recall(args, engine=e))
    assert [h['store_id'] for h in first['hits']] == ids[:25]
    assert first['provenance']['answer_ready']['more_available'] is True
    second = json.loads(tools.lcm_recall({**args, 'seen_refs': first['delta']['novel_refs']}, engine=e))
    assert [h['store_id'] for h in second['hits']] == ids[25:]
    assert second['provenance']['answer_ready']['more_available'] is False
    assert e._store.connection.total_changes == changes


def test_v2_strict_delta_refill_keeps_session_blocked_tail(orchard, monkeypatch):
    e = orchard
    e._config.recall_guidance = 'v2'
    e._config.recall_reference_strict = True
    ids = [e._store.append(sid, {'role': 'user', 'content': f'needle row {i}'})
           for i, sid in enumerate(['A'] * 5 + ['B', 'A'])]
    original = tools._lcm_recall_fts_arm
    def ranked(*a, **kw):
        hits, error = original(*a, **kw)
        return sorted(hits, key=lambda h: h['store_id']), error
    monkeypatch.setattr(tools, '_lcm_recall_fts_arm', ranked)
    args = {'query': 'needle', 'detail': 'answer_ready', 'limit': 5,
            'seen_refs': [f'lcm:{ids[5]}:0-12']}
    first = json.loads(tools.lcm_recall(args, engine=e))
    assert [h['store_id'] for h in first['hits']] == ids[:5]
    assert first['provenance']['answer_ready']['more_available'] is True
    second = json.loads(tools.lcm_recall(
        {**args, 'seen_refs': args['seen_refs'] + first['delta']['novel_refs']}, engine=e))
    assert [h['store_id'] for h in second['hits']] == ids[6:]
    assert second['provenance']['answer_ready']['more_available'] is False


@pytest.mark.parametrize('strict', [False, True])
def test_v2_delta_hint_adds_no_hydration_or_store_reads(orchard, monkeypatch, strict):
    from hermes_lcm import recall_delivery
    e = orchard
    e._config.recall_reference_strict = strict
    for i in range(8):
        e._store.append(f'reads-{i}', {'role': 'user', 'content': f'needle row {i}'})
    counts = [0, 0]
    original, get_batch = recall_delivery.deliver_recall, e._store.get_batch
    def deliver(ordered, **kw):
        hydrate = kw['hydrate']
        def counted(*a, **options):
            counts[0] += 1
            return hydrate(*a, **options)
        kw['hydrate'] = counted
        return original(ordered, **kw)
    def batch(*a, **kw):
        counts[1] += 1
        return get_batch(*a, **kw)
    monkeypatch.setattr(recall_delivery, 'deliver_recall', deliver)
    monkeypatch.setattr(e._store, 'get_batch', batch)
    args = {'query': 'needle', 'detail': 'answer_ready', 'limit': 1, 'seen_refs': []}
    tools.lcm_recall(args, engine=e)
    baseline = counts[:]
    counts[:] = [0, 0]
    e._config.recall_guidance = 'v2'
    tools.lcm_recall(args, engine=e)
    assert counts == baseline


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


def test_generated_delta_chains_end_within_pool_size_plus_one(monkeypatch, record_property):
    """256 seeded pools, both strict modes and children on/off; <= pool size + 1 calls."""
    from copy import deepcopy
    from functools import partial
    import random
    from types import SimpleNamespace

    from hermes_lcm import recall_delivery
    from hermes_lcm.retrieval_core import _hit_identity

    rng = random.Random(1188)
    longest, chains, displaced = 0, 0, 0
    features = set()
    for pool_number in range(256):
        rows, nodes, children, pool = {}, {}, {}, []
        limit, session_cap = rng.randint(2, 9), rng.randint(1, 4)
        for index in range(rng.randint(10, 38)):
            sid = rng.choice(['A', 'A', 'A', 'B', 'C', 'D'])
            store_id = index + 1
            text = f'needle source {index}'
            row = dict(store_id=store_id, session_id=sid, role='user', content=text,
                       source='', timestamp=100, observed_at=100)
            hit = dict(kind='message_excerpt', store_id=store_id, session_id=sid,
                       role='user', snippet=text, content_offset=0, timestamp=100)
            if rng.random() < .15:
                node_id = 500 + index
                hit.update(kind='summary', node_id=node_id)
                nodes[node_id] = SimpleNamespace(summary=text)
                features.add('summary')
            elif rng.random() < .15:
                # Missing rows are permanently uncitable, including non-strict delta.
                features.add('uncitable')
            else:
                rows[store_id] = row
                if rng.random() < .25:
                    alias = 1000 + index
                    rows[alias] = {**row, 'store_id': alias}
                    hit['duplicate_store_ids'] = [alias]
                    features.add('alias')
                if rng.random() < .45:
                    child_id = 2000 + index
                    child = dict(kind='message_excerpt', store_id=child_id, session_id='child',
                                 role='user', snippet=f'needle later {index}', content_offset=0, timestamp=200)
                    rows[child_id] = dict(store_id=child_id, session_id='child', role='user',
                                          source='', content=child['snippet'], timestamp=200, observed_at=200)
                    children[text] = child
            pool.append(dict(hit=hit, ranks={0: index + 1}, rrf_score=1 / (index + 1),
                             _final_score=1 / (index + 1)))
        engine = SimpleNamespace(
            _store=SimpleNamespace(get_batch=lambda ids: {sid: rows[sid] for sid in ids if sid in rows}),
            _dag=SimpleNamespace(get_node=nodes.get), current_session_id='current')
        # In-memory read adapters; selection, hydration and later placement are production code.
        monkeypatch.setattr(tools, '_lcm_recall_fts_arm', lambda e, q, **kw:
                            ([deepcopy(children[q])] if q in children else [], None))
        monkeypatch.setattr(tools, '_lcm_recall_event_time_rows', lambda e, ids, **kw:
                            ({sid: rows[sid] for sid in ids if sid in rows}, None))
        monkeypatch.setattr(tools, 'duplicate_identity_rows', lambda e, entries, **kw: rows)
        window, dropped = recall_delivery._diverse_entries(pool, limit=limit, per_session_limit=session_cap)
        window_ids = {e['hit']['store_id'] for e in window}
        if dropped:
            features.add('session_cap')
        for strict in (False, True):
            for later in (False, True):
                seen, previous_empty = set(), False
                for call in range(1, len(pool) + 2):
                    ordered = deepcopy(pool)
                    # A replay can choose the other representative of the same collapsed group.
                    if call % 2 == 0:
                        for entry in ordered:
                            hit = entry['hit']
                            if hit.get('duplicate_store_ids'):
                                hit['store_id'], hit['duplicate_store_ids'] = hit['duplicate_store_ids'][0], [hit['store_id']]
                    place_later = partial(
                        tools._lcm_recall_later_statements, engine, identity_rows=rows, n=2,
                        query='needle', fts_arm_index=0, deadline=float('inf'), excluded=set(), allowed=None,
                    ) if later else None
                    result = json.loads(recall_delivery.deliver_recall(
                        ordered, query='needle', limit=limit, requested_limit=limit, scope_bias=0,
                        include='all', detail='answer_ready', reference_strict=strict,
                        delta_requested=True, seen_refs=seen, include_occurrence_time=False,
                        deadline=float('inf'), fts_arm_index=0, arm_order=['fts'], arm_weights=[1],
                        arm_hits={'fts': []}, coverage={}, served_fts_index='messages_fts',
                        rerank_status='off', rerank_scores=[], embedding_query_metrics=[],
                        degraded_reasons=[], timed_out=False, fts_anchor_info=None, summary_leads=[],
                        limit_cap=25, snippet_chars=320, response_char_cap=40000,
                        per_session_limit=session_cap, expanded_hit_limit=10, content_chars=4000,
                        strict_selector_factory=partial(tools._LcmRecallStrictSelector, engine=engine),
                        hydrate=partial(tools._lcm_recall_answer_ready_content, engine),
                        read_event_time=partial(tools._lcm_recall_event_time_rows, engine),
                        event_time_fields=tools._lcm_recall_event_time,
                        read_occurrence_source=lambda sid: ({}, rows.get(sid)), rerank_enabled=lambda: False,
                        guidance_v2=True, place_later=place_later,
                    ))
                    context = (pool_number, strict, later, call, limit, session_cap, pool, sorted(seen))
                    refs = {h['exact_ref'] for h in result['hits'] if h.get('exact_ref')}
                    more = result['provenance']['answer_ready']['more_available']
                    empty = more and not refs - seen
                    assert not (previous_empty and empty), context
                    previous_empty = empty
                    seen.update(refs)
                    displaced += result['provenance'].get('later_statements', {}).get('displaced', 0)
                    if not more:
                        break
                else:
                    pytest.fail(f'chain exceeded pool size + 1: {context}')
                longest = max(longest, call)
                chains += 1
                seen_ids = {int(ref.split(':')[1]) for ref in seen if ref.startswith('lcm:')}
                for entry in pool:
                    hit = entry['hit']
                    group = {hit['store_id'], *hit.get('duplicate_store_ids', [])}
                    # Summaries have no exact message ref in delta's novelty filter,
                    # regardless of strictness; missing rows likewise cannot ship.
                    undeliverable = (hit['kind'] == 'summary' or hit['store_id'] not in rows
                                     or not strict and hit['store_id'] not in window_ids)
                    assert undeliverable or group & seen_ids, (context, _hit_identity(hit))
    assert features == {'summary', 'uncitable', 'alias', 'session_cap'}
    assert displaced > 0
    record_property('generated_pools', 256)
    record_property('generated_chains', chains)
    record_property('longest_chain', longest)
    record_property('displaced_candidates', displaced)
