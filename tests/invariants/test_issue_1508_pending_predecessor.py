"""GA-compatible hook, replay and fault checks retained while #1508 is deferred."""
from copy import deepcopy
import sys
from types import ModuleType

import pytest

from tests.harness import engine_context
from tests.invariants.test_issue_1501_rotation_stamp import message
from tests.test_packaging_install import _load_plugin_entrypoint_module


def register(tmp_path, monkeypatch, *, whole=False):
    persistence = ModuleType('agent.session_persistence')
    def durable_user_row_content(agent, _msg, content, api_content):
        return (content if whole else agent._persist_user_message_override), api_content
    persistence.durable_user_row_content = durable_user_row_content
    monkeypatch.setitem(sys.modules, 'agent.session_persistence', persistence)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('LCM_DATABASE_PATH', str(tmp_path / 'lcm.db'))
    module = _load_plugin_entrypoint_module('hermes_lcm_pending_predecessor')

    class Context:
        def __init__(self):
            self.hooks = {}

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            self.hooks[name] = callback

    ctx = Context()
    module.register(ctx)
    ctx.module = module
    return ctx


@pytest.mark.parametrize('previous', [message('completed reply', 9., 'assistant'),
                                     message('Earlier turns have been compacted.\nsummary', 9.)],
                         ids=['completed-turn', 'scaffold'])
def test_premerge_capture_skips_other_predecessors(tmp_path, monkeypatch, previous):
    ctx = register(tmp_path, monkeypatch)
    try:
        ctx.engine.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
        monkeypatch.setattr(ctx.engine, 'ingest', lambda _rows: pytest.fail('not a pending user pair'))
        assert ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                         conversation_history=[deepcopy(previous), message('current', 20.)]) is None
    finally:
        ctx.engine.shutdown()


@pytest.mark.parametrize('history', [[None, message('current', 20.)], [message('pending', 10.), None]],
                         ids=['non-dict-predecessor', 'non-dict-current'])
def test_premerge_non_dict_history_preserves_context(tmp_path, monkeypatch, history):
    ctx = register(tmp_path, monkeypatch)
    try:
        e = ctx.engine
        e.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
        expected = object()
        monkeypatch.setattr(e, 'ingest', lambda _rows: pytest.fail('malformed history must not be ingested'))
        monkeypatch.setattr(ctx.module, '_pre_llm_context', lambda _engine, _payload: expected)
        assert ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                         conversation_history=history) is expected
        assert e._store.get_session_messages('parent') == []
    finally:
        ctx.engine.shutdown()


def test_same_session_marker_does_not_cross_conversation(tmp_path):
    q, p, r = message('earlier reply', 9., 'assistant'), message('previous', 10.), message('current', 10.)
    with engine_context(tmp_path, window=128000, session_id='parent', conversation_id='A') as e:
        e.ingest([deepcopy(q), deepcopy(p)])
        e.ingest([deepcopy(q), deepcopy(r)])
        stored = e._store.get_session_messages('parent')
        assert e._store.get_message_relations([stored[-1]['store_id']], 'carried_stamp')
    with engine_context(tmp_path, window=128000, session_id='parent', conversation_id='B') as e:
        e.ingest([message('unrelated request', 31.), message('unrelated reply', 32., 'assistant')])
        e.ingest([deepcopy(p), message('current', 20.)])
        rows = e._store.get_session_messages('parent')
        assert any(row['content'] == 'current' and row['conversation_id'] == 'B' and row['observed_at'] == 20.
                   for row in rows)


@pytest.mark.parametrize('shape', ['cancel-retry', 'r34', 'anchor-off-uid'])
def test_premerge_shared_slot_preserves_current_turn(tmp_path, monkeypatch, shape):
    """The pre-hook and host repair share the predecessor dict (I1/I2)."""
    if shape == 'anchor-off-uid':
        monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '0')
    whole = shape in {'r34', 'anchor-off-uid'}
    ctx = register(tmp_path, monkeypatch, whole=whole)
    e = ctx.engine
    try:
        e.on_session_start('s', platform='cli', context_length=128000, conversation_id='conv')
        q = message('reply T32', 9., 'assistant')
        p = message('[T33] pending predecessor', 10.)
        current = '[T34] current request'
        if shape == 'cancel-retry':
            current = p['content'] + '\n\nUser correction/guidance after interrupt: ' + p['content']
        if shape == 'anchor-off-uid':
            p['message_uid'] = 'pending-uid'
        r = message(current, 20.)
        e.ingest([q])
        history = [q, p, r]
        ctx.hooks['pre_llm_call'](session_id='s', conversation_id='conv', conversation_history=list(history))
        # v0.28.0 semantics: the pre-LLM hook stores nothing before the host repair.
        assert [row['content'] for row in e._store.get_session_messages('s')] == [q['content']]
        # Repair mutates P in place, drops R, then persistence applies the override.
        if whole:
            p['_merged_turn_prefix'] = p['content'] + '\n\n'
            p['content'] += '\n\n' + current
        else:
            p['content'] = current
        reply = message('reply T34', 21., 'assistant')
        e.ingest([q, p, reply])
        expected = [q['content'], p['content'] if whole else current, reply['content']]
        assert [row['content'] for row in e._store.get_session_messages('s')] == expected
        assert getattr(e, '_premerge_capture', None) is None
        # A second ingest must not create a surplus remainder.
        e.ingest([q, p, reply])
        assert [row['content'] for row in e._store.get_session_messages('s')] == expected
    finally:
        e.shutdown()


@pytest.mark.parametrize('steer', [True, False], ids=['steer', 'ordinary'])
def test_unmerged_predecessor_prematch_fault(tmp_path, monkeypatch, steer):
    """I1: a pre-match fault must not re-store an unchanged predecessor slot."""
    ctx = register(tmp_path, monkeypatch)
    e = ctx.engine
    try:
        e.on_session_start('s', platform='cli', context_length=128000, conversation_id='conv')
        q, p, r = message('Q', 9., 'assistant'), message('P', 10.), message('R', 20.)
        if steer:
            p['display_kind'] = 'steer'
        e.ingest([q])
        ctx.hooks['pre_llm_call'](session_id='s', conversation_id='conv', conversation_history=[q, p, r])
        if steer:
            assert getattr(e, '_premerge_capture', None) is None

        def boom(*_args, **_kwargs):
            raise RuntimeError('forced prematch failure')

        monkeypatch.setattr(e, '_identity_anchor_prematch', boom)
        e.ingest([q, p, r, message('A', 21., 'assistant')])
        rows = [(row['content'], row['observed_at']) for row in e._store.get_session_messages('s')]
        assert rows == [('Q', 9.), ('P', 10.), ('R', 20.), ('A', 21.)]
        assert getattr(e, '_premerge_capture', None) is None
    finally:
        e.shutdown()
