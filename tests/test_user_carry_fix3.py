"""Carry round 3: multi-pass anchors and cumulative summarizer replay authority."""
# ruff: noqa: F811
from copy import deepcopy

import pytest

import hermes_lcm.engine as module
from tests.test_user_carry import factory, _seed, _text, HEADER  # noqa: F401
from tests.test_user_carry_review import _compress_setup, _tool_pair
from tests.test_user_carry_lineage import _more, _restart
from tests.test_issue_488_emission_proof import _real_hermes_merge


def _objective(engine, monkeypatch, pairs=6):
    _compress_setup(engine, monkeypatch, seed=False)
    rows = _seed(engine, ('ORIGIN: build.' + ' pad' * 120, 'CONSTRAINT: safe.',
                         'CORRECTION: allow.' + ' pad' * 120))
    rows += [{'role': 'user', 'content': 'OBJECTIVE: refactor X.'}]
    for i in range(pairs):
        rows += _tool_pair(i)
    engine.ingest(rows)
    return rows


@pytest.mark.parametrize('ctx,pairs', [(16000, 40), (32000, 6)])
def test_multi_leaf_anchor(factory, monkeypatch, ctx, pairs):
    engine = factory(budget=2048)
    rows = _objective(engine, monkeypatch, pairs)
    engine.context_length = ctx
    engine._config.leaf_chunk_tokens = 300
    engine._config.fresh_tail_count = 4
    for n in range(3):
        out = engine.compress(rows, force=True)
        text = _text(out)
        assert '[Current user objective preserved' in text
        assert HEADER in text and text.count('ORIGIN') == 1
        assert text.count('OBJECTIVE') == 1
        rows = _real_hermes_merge(deepcopy(out))
        for i in range(4):
            rows += _tool_pair(100 + n * 4 + i)
        engine.ingest(rows)


@pytest.mark.parametrize('cold', [False, True])
def test_carryless_anchor_recovers(factory, monkeypatch, cold):
    engine = factory(budget=2048)
    rows = _objective(engine, monkeypatch)
    select = module.user_carry.select
    monkeypatch.setattr(module.user_carry, 'select', lambda *a: '')
    out = engine.compress(rows, force=True)
    assert '[Current user objective preserved' in _text(out) and HEADER not in _text(out)
    proof = engine._durable_commit_proof_payload()
    assert proof.get('carry_anchor_store_id') == 7
    monkeypatch.setattr(module.user_carry, 'select', select)
    if cold:
        engine = _restart(engine, 2048)
    try:
        host = _real_hermes_merge(deepcopy(out))
        for i in range(6, 10):
            host += _tool_pair(i)
        engine.ingest(host)
        result = engine.compress(host, force=True)
        assert HEADER in _text(result) and _text(result).count('ORIGIN') == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize('budget', [0, 512])
@pytest.mark.parametrize('conversation', [None, 'conv'])
def test_superseded_packets_never_reach_summarizer(factory, monkeypatch, budget, conversation):
    engine = factory(budget=512, conversation=conversation)
    rows = _compress_setup(engine, monkeypatch)
    host = _real_hermes_merge(deepcopy(engine.compress(rows, force=True)))
    assert HEADER in _text(host)
    seen = []
    monkeypatch.setattr(module, 'summarize_with_escalation',
                        lambda **kw: (seen.append(kw['text']) or 'Bounded summary.', 1))
    for n in range(3):
        _more(host, n)
        engine.ingest(deepcopy(host))
        engine.compress(deepcopy(host), force=True)
    assert seen and all(HEADER not in text for text in seen)
    engine = _restart(engine, budget, conversation=conversation)
    try:
        seen.clear()
        _more(host, 4)
        engine.compress(deepcopy(host), force=True)
        assert seen and all(HEADER not in text for text in seen)
    finally:
        engine.shutdown()


def test_pending_anchor_cleared_for_next_conversation(factory, monkeypatch):
    engine = factory(budget=2048)
    rows = _objective(engine, monkeypatch)
    engine.compress(rows, force=True)
    assert engine._durable_commit_proof_payload().get('carry_anchor_store_id') == 7
    engine.on_session_start('T', platform='cli', conversation_id='other', context_length=128000)
    rows = _seed(engine, tuple(f'Other {i}: ' + 'delta ' * 200 for i in range(8)))
    out = engine.compress(rows, force=True)
    assert HEADER in _text(out) and '[Current user objective preserved' not in _text(out)
    assert engine._durable_commit_proof_payload().get('carry_anchor_store_id') is None


def test_off_without_history_skips_packet_recognition(factory, monkeypatch):
    engine = factory(budget=0)
    def unexpected(*args):
        pytest.fail('carry recognition without configuration or proof history')
    monkeypatch.setattr(engine, '_carry_occurrence_packet', unexpected)
    row = {'role': 'user', 'content': HEADER + ' quoted text'}
    assert engine._generated_context_carrier_remainder(row) is None
    assert not engine._is_replayed_context_scaffold_message(row)
