"""Round-2 REGRESSION cases fail on 1f43ad59; CONTROL cases preserve behavior."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import schemas, tools, evidence_pack, evidence_compiler, answer_contract
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / 'recall.db'))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    yield SimpleNamespace(_config=config, _store=store, _dag=dag,
                          current_session_id='current')
    dag.close()
    store.close()


def append(engine, session):
    return engine._store.append(session, {'role': 'user', 'content': 'orchard apples', 'timestamp': 1})


def recall(engine, **args):
    return json.loads(tools.lcm_recall({'query': 'orchard', **args}, engine=engine))


def ranked(sessions):
    return [{'hit': {'kind': 'message_excerpt', 'session_id': sid, 'store_id': i + 1},
             'rrf_score': 1 / (i + 1), 'ranks': {0: i + 1, 1: i + 1}}
            for i, sid in enumerate(sessions)]


def hook_module():
    spec = importlib.util.spec_from_file_location(
        'hermes_lcm._hook_1188_r2', Path(tools.__file__).with_name('__init__.py'))
    hook = importlib.util.module_from_spec(spec)
    hook.__package__ = 'hermes_lcm'
    spec.loader.exec_module(hook)
    return hook


@pytest.mark.parametrize('limit', [8, 10])
def test_REGRESSION_continuation_past_hydration(engine, limit):
    for i in range(20):
        append(engine, f'session-{i}')
    engine._config.recall_guidance = 'v2'
    result = recall(engine, detail='answer_ready', limit=limit)
    signal = result['provenance']['answer_ready']
    assert result['total_results'] == 8
    assert signal['sessions_in_pool'] == 20
    assert signal['more_available'] is True


def test_CONTROL_continuation_exhausted(engine):
    for i in range(8):
        append(engine, f'session-{i}')
    engine._config.recall_guidance = 'v2'
    result = recall(engine, detail='answer_ready', limit=8)
    assert result['total_results'] == 8
    assert result['provenance']['answer_ready']['more_available'] is False


OLD_SENTENCE = (
    'Use this after session_search or lcm_grep has identified a session_id '
    'that already exists in lcm.db.'
)
EXPECTED_V2_LOAD = (
    'Load an ordered raw-message transcript page for one explicit session_id from the plugin-local LCM database. '
    'This is enumeration, not search: it does not require a query, returns raw message content rather than snippets, '
    'and orders rows chronologically by store_id. Use this after lcm_recall, session_search or lcm_grep has identified '
    'a session_id that already exists in lcm.db; pass anchor_store_id to read the conversation around one hit. '
    'Output is bounded by limit, per-row content is bounded by max_content_chars, '
    'and row pagination uses after_store_id/next_cursor. '
    'It returns raw rows only; use lcm_describe/lcm_expand with both node_id and session_id for summary DAGs.'
)


def test_REGRESSION_v2_load_exact_description():
    assert schemas.recall_guidance_schemas('v2')['lcm_load_session']['description'] == EXPECTED_V2_LOAD


def test_CONTROL_v1_load_sentence():
    assert OLD_SENTENCE in schemas.recall_guidance_schemas('v1')['lcm_load_session']['description']


@pytest.mark.parametrize('kind', ['excerpt', 'summary', 'lead'])
def test_REGRESSION_rollover_hint_loads_anchor_row(engine, monkeypatch, kind):
    store_id = append(engine, 'old-session')
    node_id = engine._dag.add_node(SummaryNode(
        session_id='old-session', depth=0, summary='orchard', token_count=1,
        source_token_count=2, source_ids=[store_id], source_type='messages'))
    # This is the compression-rollover DAG operation; raw rows keep their owner.
    engine._dag.reassign_session_nodes('old-session', 'new-session')
    engine._config.recall_session_hints = True
    engine._config.embeddings_enabled = True
    engine._config.recall_reference_strict = kind != 'summary'
    monkeypatch.setattr(tools, '_resolve_recall_provider', lambda *a, **k: SimpleNamespace(
        provider_id='mock', model_id='mock', last_usage_tokens=0))
    monkeypatch.setattr(tools, '_lcm_grep_embed_query', lambda *a, **k: [1.0])

    def summary(*args, **kwargs):
        if kind == 'summary':
            return [{'kind': 'summary', 'node_id': node_id, 'session_id': 'new-session',
                     'store_id': store_id, 'snippet': 'orchard', 'timestamp': 1}], 'full', 1, 1, []
        hits, leads = tools._lcm_recall_summary_source_hits(
            engine, [(engine._dag.get_node(node_id), 1.0)], current='current',
            candidate_limit=kwargs['candidate_limit'], lead_limit=kwargs['lead_limit'],
            deadline=kwargs['deadline'])
        if kind == 'excerpt':
            # Even stale excerpt metadata must defer to the row's ownership.
            hits[0]['session_id'] = 'new-session'
        return hits, 'full', 1, 1, leads

    monkeypatch.setattr(tools, '_lcm_recall_summary_arm', summary)
    result = recall(engine, detail='answer_ready', include='summaries')
    hit = result['provenance']['answer_ready']['summary_leads'][0] if kind == 'lead' else result['hits'][0]
    hint = ast.parse(hit['session_hint'], mode='eval').body
    args = {kw.arg: ast.literal_eval(kw.value) for kw in hint.keywords}
    loaded = json.loads(tools.lcm_load_session(args, engine=engine))
    assert 'error' not in loaded, loaded
    assert args == {'session_id': 'old-session', 'anchor_store_id': store_id}
    assert store_id in [row['store_id'] for row in loaded['messages']]


def test_REGRESSION_hint_missing_row_omitted(engine):
    assert tools._lcm_recall_session_hint(engine, {
        'kind': 'message_excerpt', 'session_id': 'a', 'store_id': 999}) is None


def anchor_pipeline(engine, monkeypatch):
    original = ranked(['a'] * 9 + list('bcdef'))
    for entry in original:
        entry['hit'].update(snippet='orchard apples', timestamp=1)
    anchor = original[8]['hit']
    engine._config.embeddings_enabled = True
    monkeypatch.setattr(tools, '_resolve_recall_provider', lambda *a, **k: SimpleNamespace(
        provider_id='mock', model_id='mock', last_usage_tokens=0))
    monkeypatch.setattr(tools, '_lcm_grep_embed_query', lambda *a, **k: [1.0])
    monkeypatch.setattr(tools, '_lcm_recall_fts_arm', lambda *a, **k: ([anchor], None))
    monkeypatch.setattr(tools, '_lcm_recall_summary_arm', lambda *a, **k: (
        [entry['hit'] for entry in original], 'full', len(original), len(original), []))
    monkeypatch.setattr(tools, '_lcm_recall_chunk_arm', lambda *a, **k: ([], 'full', 0, 0))
    monkeypatch.setattr(tools, 'rrf_fuse', lambda *a, **k: [
        {**entry, 'hit': dict(entry['hit'])} for entry in original])
    return anchor['store_id']


def test_REGRESSION_floor_keeps_fts_anchor(engine, monkeypatch):
    anchor_id = anchor_pipeline(engine, monkeypatch)
    off = recall(engine, limit=8)
    assert off['hits'][7]['store_id'] == anchor_id
    engine._config.recall_min_sessions = 4
    on = recall(engine, limit=8)
    assert anchor_id in [hit['store_id'] for hit in on['hits']]
    assert on['provenance']['session_floor']['achieved'] == 4
    assert on['provenance']['fts_anchor'] == off['provenance']['fts_anchor']


def test_CONTROL_fts_anchor_provenance(engine, monkeypatch):
    anchor_pipeline(engine, monkeypatch)
    assert recall(engine, limit=8)['provenance']['fts_anchor'] == {
        'fired': True, 'position': 8, 'delivered': True}


def test_REGRESSION_null_floor_uses_env(engine, monkeypatch):
    monkeypatch.setenv('LCM_RECALL_MIN_SESSIONS', '4')
    engine._config.recall_min_sessions = LCMConfig.from_env().recall_min_sessions
    for sid in 'abcdef':
        append(engine, sid)
    assert recall(engine, min_sessions=None)['provenance']['session_floor']['requested'] == 4


def test_REGRESSION_default_schema_rc5_golden(engine):
    engine._disabled_tool_names = lambda: set()
    engine._omitted_tool_names = lambda: LCMEngine._omitted_tool_names(engine)
    golden = json.loads(Path(__file__).with_name('fixtures').joinpath(
        'issue_1188_rc5_default_schemas.json').read_text())
    assert json.dumps(LCMEngine.get_tool_schemas(engine), sort_keys=True) == json.dumps(golden, sort_keys=True)


def test_CONTROL_v2_properties():
    selected = schemas.recall_guidance_schemas('v2')
    floor = selected['lcm_recall']['parameters']['properties']['min_sessions']
    assert floor['type'] == 'integer'
    assert (floor['minimum'], floor['maximum']) == (0, 10)
    assert 'default 0' in floor['description']
    props = selected['lcm_load_session']['parameters']['properties']
    assert props['anchor_store_id']['type'] == 'integer'
    for name, default in [('before', 6), ('after', 12)]:
        assert props[name]['type'] == 'integer'
        assert props[name]['default'] == default
        assert '0..40' in props[name]['description']


def test_REGRESSION_plugin_registry_selected_variant(engine, monkeypatch):
    hook = hook_module()
    monkeypatch.setenv('LCM_RECALL_GUIDANCE', 'v2')
    monkeypatch.setattr(LCMConfig, 'from_env', classmethod(lambda cls: engine._config))
    engine._config.recall_guidance = 'v2'
    engine._omitted_tool_names = lambda: set()
    engine.apply_identity_migration = lambda notice: None
    monkeypatch.setattr('hermes_lcm.engine.LCMEngine', lambda **kwargs: engine)
    monkeypatch.setattr(hook, '_engine_took_slot', lambda *a: True)
    captured = []
    ctx = SimpleNamespace(context_engine_tool_handlers_receive_messages=True,
                          register_context_engine=lambda e: None,
                          register_tool=lambda **kwargs: captured.append(kwargs['schema']))
    hook.register(ctx)
    selected = {schema['name']: schema for schema in captured}
    assert selected['lcm_recall'] == schemas.recall_guidance_schemas('v2')['lcm_recall']
    assert selected['lcm_load_session'] == schemas.recall_guidance_schemas('v2')['lcm_load_session']


@pytest.mark.parametrize('path', ['pack', 'compile-proposal', 'compile-auto', 'requirements'])
def test_REGRESSION_internal_retrieve_keys_inert(engine, monkeypatch, path):
    for sid in 'abcdef':
        append(engine, sid)
    monkeypatch.setattr(tools.time, 'time', lambda: 1_700_000_000.0)
    # Exercise each real wrapper's callback with the real recall core, independently
    # of compiler decisions, so byte equality proves these calls isolate the levers.
    args = {'query': 'orchard', 'detail': 'answer_ready', 'limit': 3}
    monkeypatch.setattr(evidence_pack, 'build_evidence_pack',
                        lambda payload, *, engine, retrieve: retrieve(args))
    monkeypatch.setattr(evidence_compiler, 'compile_evidence',
                        lambda *a, retrieve, **k: {'recall': retrieve(args)})
    monkeypatch.setattr(evidence_compiler, 'compile_preanswer_evidence',
                        lambda *a, retrieve, **k: {'context': retrieve(args)})
    monkeypatch.setattr(answer_contract, 'compile_answer_contract',
                        lambda *a: SimpleNamespace(status='planned'))
    engine._disabled_tool_names = lambda: set()
    engine.handle_tool_call = lambda name, args, **kwargs: LCMEngine.handle_tool_call(
        engine, name, args, **kwargs)
    hook = hook_module()

    def run():
        if path == 'pack':
            return tools.lcm_evidence_pack({}, engine=engine)
        if path.startswith('compile'):
            return tools.lcm_compile_evidence({'question': 'orchard',
                'mode': 'auto' if path == 'compile-auto' else 'proposal'}, engine=engine)
        engine._config.preanswer_evidence_enabled = True
        engine._config.preanswer_evidence_mode = 'requirements_v1'
        return hook._pre_llm_context(engine, {'user_message': 'orchard', 'baseline_refs': []})

    for key in ('LCM_RECALL_MIN_SESSIONS', 'LCM_RECALL_SESSION_HINTS', 'LCM_RECALL_GUIDANCE'):
        monkeypatch.delenv(key, raising=False)
    engine._config = LCMConfig(database_path=engine._store.db_path)
    baseline = run()
    assert baseline
    monkeypatch.setenv('LCM_RECALL_MIN_SESSIONS', '4')
    monkeypatch.setenv('LCM_RECALL_SESSION_HINTS', 'true')
    monkeypatch.setenv('LCM_RECALL_GUIDANCE', 'v2')
    engine._config = LCMConfig.from_env()
    assert run() == baseline
