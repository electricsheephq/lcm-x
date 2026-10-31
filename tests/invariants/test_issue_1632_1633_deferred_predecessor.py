"""I1/I2: pending-user repair and recreation retain v0.28.0's ordered identities."""
from copy import deepcopy

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1501_rotation_stamp import message
from tests.invariants.test_issue_1508_pending_predecessor import register


def identities(engine):
    return [(row['role'], row['content'], row['observed_at'], row['session_id'], row['conversation_id'])
            for row in engine._store.get_session_messages('parent')]


def expected(*rows):
    return [(row['role'], row['content'], row['timestamp'], 'parent', 'conv') for row in rows]


@pytest.mark.parametrize('anchor', ['0', '1'], ids=['anchor-off', 'anchor-on'])
def test_issue_1632_repeated_content_matches_ga(tmp_path, monkeypatch, anchor):
    """The earliest retained user can equal R while another pending user disappears."""
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', anchor)
    ctx = register(tmp_path, monkeypatch)
    e = ctx.engine
    try:
        e.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
        q, p1, p2, r = message('Q', 9., 'assistant'), message('X', 10.), message('Y', 20.), message('X', 30.)
        a = message('A', 31., 'assistant')
        e.ingest([q])
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                  conversation_history=[q, p1, p2, r])
        # Frozen floor repair retains P1's object/stamp and applies the latest user override.
        p1['content'] = r['content']
        retained = [q, p1, a]
        e.ingest(retained)
        e.ingest(retained)
        e.on_session_end('parent', retained)
        assert identities(e) == expected(q, message('X', 10.), a)
    finally:
        e.shutdown()


@pytest.mark.parametrize('anchor', ['0', '1'], ids=['anchor-off', 'anchor-on'])
def test_issue_1633_recreate_from_retained_list_matches_ga(tmp_path, monkeypatch, anchor):
    """Three users, repaired live list, cold agent, replay and next turn equal GA."""
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', anchor)
    ctx = register(tmp_path, monkeypatch)
    e = ctx.engine
    q, p1, p2, r = message('Q', 9., 'assistant'), message('P1', 10.), message('P2', 20.), message('R', 30.)
    a, n, b = message('A', 31., 'assistant'), message('N', 100.), message('B', 101., 'assistant')
    try:
        e.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
        e.ingest([q])
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                  conversation_history=[q, p1, p2, r])
        p1['content'] = r['content']
        retained = [q, p1, a]
        e.ingest(retained)
    finally:
        e.shutdown()
    with engine_context(tmp_path, window=128000, session_id='parent', conversation_id='conv') as cold:
        cold.ingest(deepcopy(retained))
        cold.ingest([*deepcopy(retained), n, b])
        cold.on_session_end('parent', [*deepcopy(retained), n, b])
        assert identities(cold) == expected(q, message('R', 10.), a, n, b)
