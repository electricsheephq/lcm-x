"""#2707 canonical reply control; requires the full-history differential job."""
import os

import pytest

if os.environ.get('LCM_DIFFERENTIAL') != '1':
    pytest.skip('opt in with LCM_DIFFERENTIAL=1', allow_module_level=True)

from tests.test_queued_user_rows import user


@pytest.mark.parametrize('shape', ['normal', 'steer', 'carrier', 'list', 'prefix', 'retry', 'reload', 'interrupt', 'midturn'])
def test_current_main_canonical_control(tmp_path, monkeypatch, shape):
    from pathlib import Path
    from tests.differential.paired import _load
    from tests.differential.session import _open
    from tests.differential.trees import materialize
    main = materialize('433a15fc65bb6dba4492fdfc2b2039dc1c48e6d4', tmp_path / 'base')
    # The control is the exact specified base, not an old release approximation.
    assert (main / 'store.py').exists()
    outputs = []
    for label, tree in [('main', main), ('candidate', Path(__file__).resolve().parents[2])]:
        module = _load(tree, '_pr2_' + label)
        ctx = _open(module, tmp_path / label, monkeypatch)
        e = ctx.engine
        try:
            run = [user('P'), user('R', 11.)]
            if shape in ('retry', 'reload', 'interrupt'):
                e.ingest([user('P', 1.)])
            if shape == 'steer':
                run[0]['display_kind'] = 'steer'
            elif shape == 'carrier':
                run[0]['content'] = 'CONTEXT SUMMARY\nEarlier turns.'
            elif shape == 'list':
                run[0]['content'] = []
            ctx.hooks['pre_llm_call'](turn_id='T', session_id='parent', conversation_id='conv',
                                    conversation_history=run, user_message='R')
            if shape == 'midturn':
                e._ingest_messages(run)
            if shape == 'prefix':
                run[0].update(content='P\n\nR', _merged_turn_prefix='P\n\n')
            else:
                run[0]['content'] = 'R'
            e.ingest([run[0], dict(role='assistant', content='answer', timestamp=12.)])
            outputs.append([{k: v for k, v in row.items() if k not in ('timestamp', 'ingested_at')}
                            for row in e._store.get_session_messages('parent')])
        finally:
            e.shutdown()
    assert outputs[0] == outputs[1], 'canonical row changed'
