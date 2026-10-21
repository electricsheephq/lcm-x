"""A lineage proof may supply only the objective currently rendered."""
# ruff: noqa: F811
from copy import deepcopy
import re

import pytest

from tests.test_user_carry import factory, _seed, _text, HEADER  # noqa: F401
from tests.test_user_carry_review import _compress_setup, _tool_pair
from tests.test_user_carry_lineage import _restart
from tests.test_issue_488_emission_proof import _real_hermes_merge


@pytest.mark.parametrize('cold', [False, True])
@pytest.mark.parametrize('mode', ['repeated', 'carry_off', 'matching'])
def test_lineage_anchor_matches_rendered_objective(factory, monkeypatch, cold, mode):
    engine = factory(budget=2048)
    _compress_setup(engine, monkeypatch, seed=False)
    repeated = 'Please continue with the plan.'
    rows = _seed(engine, ('ORIGIN: build.' + ' pad' * 120, repeated,
                         'CORRECTION: allow.' + ' pad' * 120))
    rows += [{'role': 'user', 'content': 'OBJECTIVE: refactor X.'}]
    for i in range(6):
        rows += _tool_pair(i)
    engine.ingest(rows)
    host = _real_hermes_merge(deepcopy(engine.compress(rows, force=True)))
    assert engine._durable_commit_proof_payload()['carry_anchor_store_id'] == 7
    engine.on_session_end('S', rows)
    engine.on_session_start('child', conversation_id='conv', boundary_reason='compression',
                            old_session_id='S', context_length=16000)
    engine.context_length = 16000
    engine._config.leaf_chunk_tokens, engine._config.fresh_tail_count = 300, 4
    objective = 'NEWOBJ: port module Y.' if mode == 'carry_off' else repeated
    host += [{'role': 'user', 'content': objective}]
    for i in range(6 if mode == 'carry_off' else 40):
        host += _tool_pair(60 + i)
    engine.ingest(deepcopy(host))
    child_id = engine._store._conn.execute(
        "SELECT store_id FROM messages WHERE session_id='child' AND content=?", (objective,)
    ).fetchone()[0]
    if mode == 'carry_off':
        engine._config.user_carry_tokens = 0
    out = engine.compress(deepcopy(host), force=True)
    assert HEADER not in _text(out)
    assert engine._durable_commit_proof_payload().get('carry_anchor_store_id') is None
    host = _real_hermes_merge(deepcopy(out))
    if cold:
        engine = _restart(engine, 2048, 'child')
    else:
        engine._config.user_carry_tokens = 2048
    try:
        if mode == 'matching':
            original = engine._durable_commit_proof_payload

            def matching_proof(session_id=None):
                proof = original(session_id)
                if session_id == 'S' and proof:
                    return {**proof, 'carry_anchor_store_id': child_id}
                return proof

            monkeypatch.setattr(engine, '_durable_commit_proof_payload', matching_proof)
        for i in range(4):
            host += _tool_pair(200 + i)
        engine.ingest(deepcopy(host))
        for n in range(2):
            out = engine.compress(deepcopy(host), force=True)
            text = _text(out)
            anchor = engine._durable_commit_proof_payload().get('carry_anchor_store_id')
            assert '[Current user objective preserved' in text
            if mode == 'matching':
                assert anchor == child_id
                carry = text[text.index(HEADER):]
                assert list(map(int, re.findall(r'\[Earlier user message, store (\d+)\]', carry))) == [1, 3, 5, 7]
                assert text.count('OBJECTIVE') == 1
            else:
                assert anchor is None
                assert HEADER not in text
            host = _real_hermes_merge(deepcopy(out))
            for i in range(4):
                host += _tool_pair(300 + n * 4 + i)
            engine.ingest(deepcopy(host))
    finally:
        engine.shutdown()
