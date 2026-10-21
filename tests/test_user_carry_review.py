"""Portable regressions for C2's independently reproduced review findings."""
from copy import deepcopy

import pytest

from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from tests.test_user_carry import factory, _seed, _emit, _node, _stored, _text, HEADER  # noqa: F401
from tests.test_issue_488_emission_descriptors import _record
from tests.test_issue_488_emission_proof import _real_hermes_merge


def _compress_setup(engine, monkeypatch):
    engine._config.fresh_tail_count, engine._config.fresh_tail_max_tokens = 4, 0
    engine._config.leaf_chunk_tokens = 300
    monkeypatch.setattr('hermes_lcm.engine.summarize_with_escalation',
                        lambda **kw: ('Compact summary; user wording omitted. Expand for details: x', 1))
    return _seed(engine, tuple(f'Ask {i}: ' + 'alpha beta ' * 120 for i in range(8)))


@pytest.mark.parametrize('budget', [0, 384, 1024])
@pytest.mark.parametrize('disable', [False, True])
def test_superseded_replay(factory, monkeypatch, budget, disable):
    engine = factory(budget=budget)
    rows = _compress_setup(engine, monkeypatch)
    first = engine.compress(rows, force=True)
    assert (HEADER in _text(first)) == bool(budget)
    host = _real_hermes_merge(deepcopy(first))
    for i in range(3):
        host += [{'role': 'user', 'content': f'More {i}: ' + 'gamma ' * 300, 'timestamp': 300. + i * 2},
                 {'role': 'assistant', 'content': f'Ok more {i}', 'timestamp': 301. + i * 2}]
    engine.ingest(host)
    before = _stored(engine)
    if disable:
        engine._config.user_carry_tokens = 0
    engine.compress(host, force=True)
    assert engine._last_compression_status == 'compacted'
    genuine = [{'role': 'user', 'content': 'A truly new question.', 'timestamp': 400.},
               {'role': 'assistant', 'content': 'Reply new', 'timestamp': 401.}]
    engine.ingest(deepcopy(host) + genuine)
    assert _stored(engine) == before + [(len(before) + i + 1, m['role'], m['content']) for i, m in enumerate(genuine)]


@pytest.mark.parametrize('cold', [False, True])
@pytest.mark.parametrize('carry_only', [False, True])
def test_rotation_strips_only_carry(factory, cold, carry_only):
    results = []
    for budget in (0, 512):
        engine = factory(name=f'rotation-{budget}', budget=budget)
        rows = _seed(engine, ('Origin objective.', 'Older constraint.', 'Third request.', 'Latest request.'))
        if not carry_only:
            _node(engine)
        emitted = _emit(engine, rows)
        assert (HEADER in _text(emitted)) == bool(budget)
        _record(engine, rows, emitted)
        if cold:
            config, home = engine._config, engine._hermes_home
            engine.shutdown()
            engine = LCMEngine(config=config, hermes_home=home)
        try:
            engine.on_session_start('child', platform='cli', conversation_id='conv', context_length=128000,
                                    boundary_reason='compression', old_session_id='S')
            host = [*deepcopy(emitted), {'role': 'user', 'content': 'Genuine new turn.', 'timestamp': 500.}]
            engine.ingest(deepcopy(host))
            engine.ingest(deepcopy(host))
            results.append([(r, t) for _, r, t in _stored(engine)])
            assert all(HEADER not in t and 'Origin objective.' not in t for _, t in results[-1])
        finally:
            if cold:
                engine.shutdown()
    assert results[0] == results[1]
