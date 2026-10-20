"""#1188 opt-in levers; names label REGRESSION (red on base) vs CONTROL."""
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import schemas, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / 'recall.db'))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    value = SimpleNamespace(_config=config, _store=store, _dag=dag,
                            current_session_id='current')
    yield value
    dag.close()
    store.close()


def append(engine, session='a', content='orchard apples', **extra):
    return engine._store.append(session, {'role': 'user', 'content': content,
                                          'timestamp': 1, **extra})


def recall(engine, **args):
    return json.loads(tools.lcm_recall({'query': 'orchard', **args}, engine=engine))


def ranked(sessions):
    return [{'hit': {'kind': 'message_excerpt', 'session_id': sid, 'store_id': i + 1},
             '_final_score': 1 / (i + 1), 'rrf_score': 1 / (i + 1)}
            for i, sid in enumerate(sessions)]


@pytest.mark.parametrize('requested,expected', [(4, 4), (10, 6)])
def test_REGRESSION_floor_rank_only(requested, expected):
    original = ranked(['a'] * 30 + ['b', 'c', 'd', 'e', 'f', None, ''])
    before = json.dumps(original)
    result, info = tools._lcm_recall_session_floor(original, 8, requested)
    assert len({e['hit']['session_id'] for e in result[:8] if e['hit']['session_id']}) == expected
    assert info == {'requested': requested, 'achieved': expected,
                    'promoted': expected - 1, 'delivered': None}
    assert json.dumps(original) == before
    assert sorted(result, key=lambda e: e['hit']['store_id']) == original
    assert [e['hit']['session_id'] for e in result[:8] if e['hit']['session_id'] != 'a'] == list('bcdef')[:expected - 1]


def test_REGRESSION_floor_preserves_existing_unique_sessions():
    result, info = tools._lcm_recall_session_floor(ranked(['a', 'a', None, 'b', 'c', 'd']), 4, 4)
    assert {e['hit']['session_id'] for e in result[:4]} == set('abcd')
    assert info['achieved'] == 4


def test_REGRESSION_floor_never_promotes_sessionless():
    original = ranked([None, '', 'a', 'a', None, '', 'b'])
    result, info = tools._lcm_recall_session_floor(original, 4, 10)
    assert info['achieved'] == 2
    assert info['promoted'] == 1
    assert [e['hit']['session_id'] for e in result[:4]] == [None, '', 'a', 'b']


@pytest.mark.parametrize('detail', ['snippets', 'answer_ready'])
def test_REGRESSION_floor_pipeline(engine, monkeypatch, detail):
    # Freeze the FTS arm in relevance order: a long session dominates it.
    ids = [append(engine) for _ in range(30)]
    ids += [append(engine, sid) for sid in 'bcdef']
    hits = [{'kind': 'message_excerpt', 'session_id': engine._store.get(i)['session_id'],
             'store_id': i, 'snippet': 'orchard apples', 'timestamp': 1,
             'expand_hint': f'lcm_expand(store_id={i}, content_offset=0)'} for i in ids]
    limits = []

    def arm(*args, candidate_limit, **kwargs):
        limits.append(candidate_limit)
        return hits[:candidate_limit], None

    monkeypatch.setattr(tools, '_lcm_recall_fts_arm', arm)
    engine._config.recall_min_sessions = 4
    result = recall(engine, detail=detail, limit=8)
    assert len({h['session_id'] for h in result['hits']}) >= 4
    assert limits == [100]
    info = result['provenance']['session_floor']
    assert info == {'requested': 4, 'achieved': 4, 'promoted': 3,
                    'delivered': len({h['session_id'] for h in result['hits']})}
    off = recall(engine, detail=detail, limit=8, min_sessions=0)
    assert limits[-1] == 50
    assert 'session_floor' not in off['provenance']
    scores = {h['store_id']: h['score'] for h in off['hits']}
    assert all(h['score'] == scores[h['store_id']] for h in result['hits'] if h['store_id'] in scores)


def test_CONTROL_default_bytes(engine, monkeypatch):
    # Store ingestion time is distinct from the supplied event timestamp.
    monkeypatch.setattr(tools.time, 'time', lambda: 1_700_000_000.0)
    append(engine, content='orchard café 🍎')
    append(engine, 'b', 'orchard second')
    outputs = {}
    for detail in ('snippets', 'answer_ready'):
        args = {'query': 'orchard', 'detail': detail}
        raw = tools.lcm_recall(args, engine=engine)
        assert raw == tools.lcm_recall({**args, 'min_sessions': 0}, engine=engine)
        payload = json.loads(raw)
        assert 'session_floor' not in payload['provenance']
        assert all('session_hint' not in h for h in payload['hits'])
        if detail == 'answer_ready':
            assert 'more_available' not in payload['provenance']['answer_ready']
        outputs[detail] = raw
    outputs['descriptions'] = [schemas.LCM_RECALL['description'], schemas.LCM_LOAD_SESSION['description']]
    if os.environ.get('RECALL_BYTE_RECEIPT'):
        Path(os.environ['RECALL_BYTE_RECEIPT']).write_text(json.dumps(outputs, ensure_ascii=False))


def test_REGRESSION_config_keys(monkeypatch):
    monkeypatch.setenv('LCM_RECALL_MIN_SESSIONS', '20')
    monkeypatch.setenv('LCM_RECALL_SESSION_HINTS', 'true')
    monkeypatch.setenv('LCM_RECALL_GUIDANCE', 'v2')
    config = LCMConfig.from_env()
    assert config.recall_min_sessions == 10
    assert config.recall_session_hints is True
    assert config.recall_guidance == 'v2'
    monkeypatch.setenv('LCM_RECALL_MIN_SESSIONS', '-3')
    monkeypatch.setenv('LCM_RECALL_GUIDANCE', 'unknown')
    config = LCMConfig.from_env()
    assert config.recall_min_sessions == 0
    assert config.recall_guidance == 'v1'


def test_REGRESSION_internal_consumers_opt_out(engine, monkeypatch):
    monkeypatch.setenv('LCM_RECALL_MIN_SESSIONS', '4')
    monkeypatch.setenv('LCM_RECALL_SESSION_HINTS', 'true')
    monkeypatch.setenv('LCM_RECALL_GUIDANCE', 'v2')
    config = LCMConfig.from_env()
    config.proactive_recall_enabled = True
    config.embeddings_enabled = True
    captured = []

    def capture(args, **kwargs):
        captured.append(kwargs)
        return '{"hits": []}'

    monkeypatch.setattr(tools, 'lcm_recall', capture)
    fake = SimpleNamespace(_config=config, _newest_user_message_text=LCMEngine._newest_user_message_text,
                           _proactive_recall_skipped_count=0)
    LCMEngine._build_proactive_recall_message(fake, [{'role': 'user', 'content': 'orchard'}], 'user', set())
    # Load the package hook just as the host does, without registering it.
    spec = importlib.util.spec_from_file_location('hermes_lcm._hook_1188', Path(tools.__file__).with_name('__init__.py'))
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    engine._config = config
    engine.handle_tool_call = lambda name, args, **kwargs: LCMEngine.handle_tool_call(engine, name, args, **kwargs)
    engine._disabled_tool_names = lambda: set()
    hook._answer_ready_baseline(engine, 'orchard', {})
    assert len(captured) == 2
    for options in captured:
        assert options.get('min_sessions') == 0
        assert options.get('session_hints') is False
        assert options.get('recall_guidance') == 'v1'


def test_REGRESSION_anchor_window_and_paging(engine):
    ids = [append(engine, content=f'orchard row {i}') for i in range(25)]
    result = json.loads(tools.lcm_load_session({'session_id': 'a', 'anchor_store_id': ids[8]}, engine=engine))
    assert [m['store_id'] for m in result['messages']] == ids[2:21]
    assert result['next_cursor'] == ids[20]
    assert result['has_more'] is True
    page = json.loads(tools.lcm_load_session({'session_id': 'a', 'after_store_id': result['next_cursor']}, engine=engine))
    assert [m['store_id'] for m in page['messages']] == ids[21:]


@pytest.mark.parametrize('anchor', [999, 'foreign'])
def test_REGRESSION_anchor_rejected(engine, anchor):
    if anchor == 'foreign':
        anchor = append(engine, 'b')
    result = json.loads(tools.lcm_load_session({'session_id': 'a', 'anchor_store_id': anchor}, engine=engine))
    assert 'anchor_store_id' in result['error']


def test_REGRESSION_anchor_clamps(engine):
    ids = [append(engine) for _ in range(100)]
    result = json.loads(tools.lcm_load_session({'session_id': 'a', 'anchor_store_id': ids[50],
                                              'before': 90, 'after': 90}, engine=engine))
    assert [m['store_id'] for m in result['messages']] == ids[10:91]
    result = json.loads(tools.lcm_load_session({'session_id': 'a', 'anchor_store_id': ids[50],
                                              'before': -2, 'after': 0}, engine=engine))
    assert [m['store_id'] for m in result['messages']] == [ids[50]]


def test_CONTROL_existing_window_clamp(engine):
    ids = [append(engine) for _ in range(50)]
    rows = engine._store.load_session_window('a', anchor_store_id=ids[25], before=99, after=99)
    assert [r['store_id'] for r in rows] == ids[13:38]


def test_REGRESSION_hints_and_summary_sources(engine, monkeypatch):
    sid = append(engine)
    node_id = engine._dag.add_node(SummaryNode(session_id='a', depth=0, summary='orchard',
        token_count=1, source_token_count=2, source_ids=[sid], source_type='messages'))
    engine._config.embeddings_enabled = True
    engine._config.recall_session_hints = True
    # Exercise the strict source hydration and lead creation, without an embedding service.
    def summary(*args, **kwargs):
        hits, leads = tools._lcm_recall_summary_source_hits(
            engine, [(engine._dag.get_node(node_id), 1.0)], current='current',
            candidate_limit=kwargs['candidate_limit'], lead_limit=kwargs['lead_limit'],
            deadline=kwargs['deadline'])
        return hits, 'full', 1, 1, leads

    # Use the existing summary source expansion rather than mock its return shape.
    monkeypatch.setattr(tools, '_resolve_recall_provider', lambda *a, **k: SimpleNamespace(
        provider_id='mock', model_id='mock', last_usage_tokens=0))
    monkeypatch.setattr(tools, '_lcm_grep_embed_query', lambda *a, **k: [1.0])
    monkeypatch.setattr(tools, '_lcm_recall_summary_arm', summary)
    result = recall(engine, detail='answer_ready', include='summaries')
    hint = f"lcm_load_session(session_id='a', anchor_store_id={sid})"
    assert result['hits'][0]['session_hint'] == hint
    lead = result['provenance']['answer_ready']['summary_leads'][0]
    assert lead['session_hint'] == hint
    assert lead['expand_hint'] == f"lcm_expand(node_id={node_id}, session_id='a')"
    assert engine._dag.source_message_ids(node_id, limit=1) == [sid]
    assert all('session_hint' not in h for h in recall(engine)['hits'])
    engine._config.recall_session_hints = False
    assert all('session_hint' not in h for h in recall(engine, detail='answer_ready')['hits'])


GUIDANCE = (
    "For questions that combine or compare facts from several conversations (totals, lists, changes over time, "
    "'how many', 'which ... first'), first check `provenance.answer_ready.sessions_delivered` and `more_available`. "
    "If more may exist, call again with `exclude_session_ids` set to the sessions you have read, or raise `min_sessions`. "
    "Read around a hit with its `session_hint`. Stop when every part of the question is supported by a cited span, "
    "or when a call returns no new sessions; then say what you could not find."
)


def test_REGRESSION_shared_guidance(engine):
    # Simulate the read-only MCP/bench path: schemas alone, no package/engine imports.
    spec = importlib.util.spec_from_file_location('standalone_schemas_1188', schemas.__file__)
    standalone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(ast.parse(Path(schemas.__file__).read_text())))
    v2 = standalone.recall_guidance_schemas('v2')
    engine._config.recall_guidance = 'v2'
    engine._omitted_tool_names = lambda: set()
    host = {s['name']: s for s in LCMEngine.get_tool_schemas(engine)}
    assert host['lcm_recall'] == v2['lcm_recall']
    assert host['lcm_load_session'] == v2['lcm_load_session']
    assert host['lcm_recall']['description'] == schemas.LCM_RECALL['description'] + ' ' + GUIDANCE
    assert 'Use this after lcm_recall, session_search or lcm_grep has identified a session_id that already exists in lcm.db; pass anchor_store_id to read the conversation around one hit.' in host['lcm_load_session']['description']


def test_REGRESSION_guidance_v1_identity():
    assert schemas.recall_guidance_schemas('v1') == {
        'lcm_recall': schemas.LCM_RECALL, 'lcm_load_session': schemas.LCM_LOAD_SESSION}


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('count,limit,more', [(2, 8, False), (2, 2, False), (3, 2, True), (8, 10, True)])
def test_REGRESSION_continuation(engine, strict, count, limit, more):
    for _ in range(count):
        append(engine)
    engine._config.recall_guidance = 'v2'
    engine._config.recall_reference_strict = strict
    result = recall(engine, detail='answer_ready', limit=limit)
    signal = result['provenance']['answer_ready']
    assert signal['sessions_delivered'] == signal['sessions_in_pool'] == 1
    assert signal['more_available'] is more
    assert result['total_results'] <= 5


def test_REGRESSION_continuation_after_response_cap(engine, monkeypatch):
    for sid in 'abc':
        append(engine, sid, 'orchard ' + 'x' * 2000)
    engine._config.recall_guidance = 'v2'
    engine._config.recall_min_sessions = 3
    monkeypatch.setattr(tools, '_LCM_RECALL_RESPONSE_CHAR_CAP', 4700)
    result = recall(engine, detail='answer_ready')
    signal = result['provenance']['answer_ready']
    assert signal['sessions_in_pool'] == 3
    assert signal['sessions_delivered'] == len({h['session_id'] for h in result['hits']}) < 3
    assert result['provenance']['session_floor']['delivered'] == signal['sessions_delivered']
    assert signal['more_available'] is True
    assert len(json.dumps(result, ensure_ascii=False)) <= 4700


@pytest.mark.parametrize('tail', ['empty', 'invalid', 'seen'])
def test_REGRESSION_continuation_exhausted_admissible_pool(engine, monkeypatch, tail):
    engine._config.recall_guidance = 'v2'
    args = {'detail': 'answer_ready', 'limit': 1}
    if tail != 'empty':
        store_id = append(engine)
        if tail == 'invalid':
            # A non-existent row in the tail is not usable continuation evidence.
            hits = [{'kind': 'message_excerpt', 'session_id': 'a', 'store_id': sid,
                     'snippet': 'orchard apples', 'timestamp': 1} for sid in (store_id, 999)]
            monkeypatch.setattr(tools, '_lcm_recall_fts_arm', lambda *a, **k: (hits, None))
        else:
            args['seen_refs'] = [f'lcm:{store_id}:0-14']
    result = recall(engine, **args)
    signal = result['provenance']['answer_ready']
    assert signal['more_available'] is False
    assert signal['sessions_in_pool'] == (0 if tail == 'empty' else 1)
    assert signal['sessions_delivered'] == (1 if tail == 'invalid' else 0)
