"""Round-two uncertainty gates and unchanged ordinary cutoffs."""
import json
import sqlite3
import time
from unittest.mock import patch

import pytest

from hermes_lcm import tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.harness import engine_context
from tests.test_issue_1779_carried_time import ABS, STAMP, REASON, compile_args, compute_args, plant, retrieve_finish
from tests.test_issue_1779_fix_round1 import proposal_args


def test_ordinary_proposal_cutoff_keeps_base_rejection(tmp_path):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = ['My project status is amber.']
        ids = plant(engine, texts)
        obj = json.loads(tools.lcm_compile_evidence(proposal_args(ids, texts, '2024-03-19'), engine=engine))
        assert 'exclusions' not in obj
        assert {x['reason_code'] for x in obj['rejections']} == {'claim_grounding_failed'}


@pytest.mark.parametrize('question,relative', [('What is the order of events this year?', '3 months ago'),
    ('Which event happened in March?', '3 weeks ago')])
def test_window_exclusion_blocks_auto_answer(tmp_path, question, relative):
    texts = ['The SculptureChallenge happened on 2024-03-19.', 'The Exhibition happened on 2024-03-17.',
        f'The Hackathon happened {relative}.']
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = plant(engine, texts)
        engine._store.add_message_relations([[(ids[2], 'carried_stamp', None, 0, STAMP)]])
        obj = json.loads(tools.lcm_compile_evidence(compile_args(ids, texts, question, '2024-03-22'), engine=engine))
        assert obj['reason_code'] == REASON
        assert obj['computation'] is None
        assert obj.get('direct_fact') is None
        assert any(x['exact_ref'].startswith(f'lcm:{ids[2]}:') and x['reason_code'] == 'event_outside_temporal_window'
            and x['time_uncertain'] == 'carried' for x in obj.get('exclusions', []))


@pytest.mark.parametrize('surface', ['auto', 'compute'])
def test_asof_carried_explicit_operand_is_not_certain(tmp_path, surface):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = plant(engine, ABS, True)
        raw = (tools.lcm_compute(compute_args(ids, ABS, question_date='2024-03-22'), engine=engine)
            if surface == 'compute' else tools.lcm_compile_evidence(compile_args(ids, ABS,
                question_date='2024-03-22'), engine=engine))
        obj = json.loads(raw)
        assert obj.get('reason', obj.get('reason_code')) == REASON
        assert obj.get('trace' if surface == 'compute' else 'computation') is None


@pytest.mark.parametrize('date_text', ['March 19, 2024', 'Mar 19, 2024', 'March 19 2024', 'Mar 19 2024', 'March 19th, 2024', '2024/03/19', '2024-3-19'])
def test_grounded_absolute_date_is_observation_independent(tmp_path, date_text):
    texts = [f'The PlankChallenge happened on {date_text}.', ABS[1]]
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        ids = plant(engine, texts, True)
        args = compute_args(ids, texts)
        args['operands'][0].pop('occurrence_time')
        args['operands'][0]['date'] = '2024-03-19'
        obj = json.loads(tools.lcm_compute(args, engine=engine))
        assert obj['status'] == 'computed', obj.get('reason')


@pytest.mark.parametrize('surface', ['compute', 'retrieve'])
def test_lookup_failure_is_json_fallback(tmp_path, surface):
    with patch('time.time', return_value=1800000000.0):
        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / 'lcm.db'), adaptive_retrieval_enabled=True))
        engine.on_session_start('session-a', conversation_id='conversation-a')
        try:
            ids = plant(engine, ABS, True, 'session-a')
            with patch.object(engine._store, 'get_carried_stamp_ids', side_effect=sqlite3.OperationalError('synthetic lookup failure')):
                obj = (retrieve_finish(engine, ids, ABS) if surface == 'retrieve' else
                    json.loads(tools.lcm_compute(compute_args(ids, ABS), engine=engine)))
            if surface == 'retrieve':
                assert obj['status'] == 'fallback'
                assert obj['query_view']['persistence']['status'] == 'skipped'
                obj = obj['computation']
            assert obj['status'] == 'fallback'
            assert obj['reason'] == REASON
        finally:
            engine.shutdown()


def test_finite_exclusions_are_bounded(tmp_path):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        text = 'I took a vacation to Kyoto 3 months ago.'
        ids = [engine._store.append('s', dict(role='user', content=text, timestamp=STAMP)) for _ in range(4096)]
        engine._store.add_message_relations([[(i, 'carried_stamp', None, 0, STAMP) for i in ids]])
        args = compile_args(ids[:1], [text], 'How many vacations did I take this year?', '2024-03-22')
        raw = tools.lcm_compile_evidence(args, engine=engine)
        obj = json.loads(raw)
        assert obj['reason_code'] == REASON
        assert len(obj['exclusions']) <= obj['budgets']['max_candidates']
        assert obj['exclusion_counts']['event_outside_temporal_window'] == 4096
        assert len(raw) < 32768


def test_recall_does_not_retain_private_connections(tmp_path):
    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        plant(engine, ABS, True, 's')
        initial = len(engine._store._carried_relation_connections)
        for _ in range(1000):
            obj = json.loads(tools.lcm_recall({'query': 'PlankChallenge', 'scope': 'all', 'limit': 2}, engine=engine, delivery_unit='hits'))
            assert obj['hits']
        assert len(engine._store._carried_relation_connections) == initial


@pytest.mark.parametrize('relative', ['3 weeks ago', '1 week ago', 'at some point'])
def test_targeted_retrieval_window_filter_preserves_uncertainty(tmp_path, relative):
    from hermes_lcm.answer_contract import compile_answer_contract
    from hermes_lcm.requirements_compiler import RequirementsBudgets, _base_result, _finish, _retrieve_sources

    with engine_context(tmp_path, embeddings_enabled=False) as engine:
        texts = [f'The Hackathon happened {relative}.']
        ids = plant(engine, texts, True)
        contract = compile_answer_contract('Which event happened in March?', '2024-03-22').contract
        limits = RequirementsBudgets(max_retrieval_calls=1)
        result = _base_result(contract=contract, limits=limits, baseline=[], baseline_input=[], started=time.perf_counter())
        sql = []
        engine._store.connection.set_trace_callback(lambda s: sql.append(s) if 'message_relations' in s else None)
        try:
            sources = _retrieve_sources(contract, retrieve=lambda _: {'hits': [dict(exact_ref=f'lcm:{ids[0]}:0-{len(texts[0])}')]},
                engine=engine, seen_refs=[], open_slots=contract.slots, capacity=12, limits=limits, result=result)
        finally:
            engine._store.connection.set_trace_callback(None)
        assert len(sql) <= 2
        if relative != '1 week ago':
            assert sources == []
            assert result['exclusions'][0]['reason_code'] == 'event_outside_temporal_window'
            assert result['exclusions'][0]['time_uncertain'] == 'carried'
            assert _finish(result, started=time.perf_counter())['reason_code'] == REASON
        else:
            from hermes_lcm.requirements_compiler import _candidate_inventory, _select_temporal_event
            candidates = _candidate_inventory(sources, contract, limit=48, engine=engine, result=result)
            assert _select_temporal_event(candidates, contract) == (None, REASON)
