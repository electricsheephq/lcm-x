"""Round-2 reviewer probes: durable emission lineage and objective row identity."""
# ruff: noqa: F811
from copy import deepcopy

import pytest

from hermes_lcm.engine import LCMEngine
from tests.test_user_carry import factory, _seed, _emit, _node, _stored, _text, HEADER  # noqa: F401
from tests.test_user_carry_review import _compress_setup, _tool_pair
from tests.test_issue_488_emission_descriptors import _record
from tests.test_issue_488_emission_proof import _real_hermes_merge


def _more(host, round=0):
    for i in range(3):
        host += [{'role': 'user', 'content': f'More {round}.{i}: ' + 'gamma ' * 300,
                  'timestamp': 300. + round * 20 + i * 2},
                 {'role': 'assistant', 'content': f'Ok {round}.{i}', 'timestamp': 301. + round * 20 + i * 2}]
    return host


def _restart(engine, budget, sid='S', conversation='conv', **kwargs):
    config, home = engine._config, engine._hermes_home
    engine.shutdown()
    config.user_carry_tokens = budget
    engine = LCMEngine(config=config, hermes_home=home)
    engine.on_session_start(sid, platform='cli', conversation_id=conversation, context_length=128000, **kwargs)
    return engine


GENUINE = [{'role': 'user', 'content': 'A truly new question.', 'timestamp': 500.},
           {'role': 'assistant', 'content': 'Reply new', 'timestamp': 501.}]


@pytest.mark.parametrize('restart_budget', [None, 0, 512])
@pytest.mark.parametrize('superseded', [1, 2, 3])
def test_superseded_emission_history(factory, monkeypatch, restart_budget, superseded):
    engine = factory(budget=512)
    rows = _compress_setup(engine, monkeypatch)
    host = _real_hermes_merge(deepcopy(engine.compress(rows, force=True)))
    assert HEADER in _text(host)
    for n in range(superseded):
        _more(host, n)
        engine.ingest(deepcopy(host))
        engine.compress(deepcopy(host), force=True)  # host does not adopt this output
        assert engine._last_compression_status == 'compacted'
    before = _stored(engine)
    if restart_budget is not None:
        engine = _restart(engine, restart_budget)
    try:
        engine.ingest(deepcopy(host) + GENUINE)
        after = _stored(engine)
        assert after == before + [(len(before) + i + 1, m['role'], m['content']) for i, m in enumerate(GENUINE)]
        assert all(HEADER not in text for _, _, text in after)
        if restart_budget == 0:
            # Recognition must survive another proof publication while disabled.
            engine.compress(deepcopy(host) + GENUINE, force=True)
            engine = _restart(engine, 0)
            engine.ingest(deepcopy(host) + GENUINE)
            assert all(HEADER not in text for _, _, text in _stored(engine))
    finally:
        engine.shutdown()


@pytest.mark.parametrize('cold', [False, True])
@pytest.mark.parametrize('child_budget', [0, 512])
@pytest.mark.parametrize('superseded', [False, True])
def test_rotation_and_child_restart(factory, monkeypatch, cold, child_budget, superseded):
    engine = factory(budget=512)
    rows = _compress_setup(engine, monkeypatch)
    host = _real_hermes_merge(deepcopy(engine.compress(rows, force=True)))
    if superseded:
        _more(host)
        engine.ingest(host)
        engine.compress(host, force=True)
    if cold:
        engine = _restart(engine, child_budget, 'child', boundary_reason='compression', old_session_id='S')
    else:
        engine._config.user_carry_tokens = child_budget
        engine.on_session_start('child', conversation_id='conv', boundary_reason='compression', old_session_id='S')
    try:
        engine.ingest(deepcopy(host) + GENUINE)
        assert all(HEADER not in text for _, _, text in _stored(engine))
        engine = _restart(engine, 0, 'child')
        engine.ingest(deepcopy(host) + GENUINE)
        assert all(HEADER not in text for _, _, text in _stored(engine))
    finally:
        engine.shutdown()


@pytest.mark.parametrize('cold', [False, True])
def test_bound_conversation_across_rotations(factory, cold):
    engine = factory(conversation=None)
    rows = _seed(engine, ('Origin objective wording.', 'Older constraint wording.', 'Third request.', 'Latest request.'))
    _node(engine)
    host = _emit(engine, rows)
    _record(engine, rows, host)
    engine.on_session_end('S', rows)
    try:
        for parent, child in [('S', 'child'), ('child', 'grandchild')]:
            if cold:
                engine = _restart(engine, 512, child, None, boundary_reason='compression', old_session_id=parent)
            else:
                engine.on_session_start(child, conversation_id=None, boundary_reason='compression', old_session_id=parent)
            host = deepcopy(host) + GENUINE
            engine.ingest(host)
            assert all(HEADER not in text for _, _, text in _stored(engine))
            out = _emit(engine, host, tail=host[-2:])
            assert HEADER in _text(out) and 'Origin objective wording.' in _text(out)
            _record(engine, host, out)
            host = out
    finally:
        engine.shutdown()


@pytest.mark.parametrize('budget', [512, 2048])
@pytest.mark.parametrize('duplicate', [False, True])
def test_repeated_objective_compactions(factory, monkeypatch, budget, duplicate):
    engine = factory(budget=budget)
    _compress_setup(engine, monkeypatch, seed=False)
    rows = _seed(engine, ('ORIGIN: build.' + ' pad' * 120, 'continue' if duplicate else 'go ahead please',
                         'CORRECTION: allow.' + ' pad' * 120))
    rows += [{'role': 'user', 'content': 'continue'}]
    for i in range(6):
        rows += _tool_pair(i)
    engine.ingest(rows)
    host = rows
    for n in range(3):
        out = engine.compress(host, force=True)
        text = _text(out)
        assert '[Current user objective preserved' in text
        assert HEADER in text and text.count('ORIGIN') == 1
        # Equal bytes are separate authored turns; exclude only the rendered objective row.
        assert text.count('continue') == (2 if duplicate else 1)
        host = _real_hermes_merge(deepcopy(out))
        for i in range(6 + n * 4, 10 + n * 4):
            host += _tool_pair(i)
        engine.ingest(host)


@pytest.mark.parametrize('scenario', ['paste_mid_rot', 'paste_start_rot', 'paste_mid_echo_rot',
                                     'paste_mid_echo_rotcold', 'plain_rot'])
def test_off_rotation_bytes(tmp_path, scenario):
    """Full reviewer driver output equals the pinned main capture, including stored summaries."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / 'tests/fixtures/user_carry_off_rotation.json').read_text())
    env = dict(os.environ)
    env.pop('LCM_USER_CARRY_TOKENS', None)
    # The driver runs as a script, so its sys.path starts at tests/; the host's `agent` package (CI stubs it at
    # the repo root) must still import.
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(root), env.get('PYTHONPATH')]))
    run = subprocess.run([sys.executable, '-B', str(root / 'tests/user_carry_off_driver.py'),
                          str(root), scenario, str(tmp_path)], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    data = json.loads(run.stdout.replace(str(tmp_path), '<WORK>'))
    actual = json.dumps(data, sort_keys=True, separators=(',', ':'))
    assert hashlib.sha256(actual.encode()).hexdigest() == expected['sha256'][scenario]


def test_disabled_restart_before_superseding(factory, monkeypatch):
    """test_more.py's restart precedes the compression whose output the host rejects."""
    engine = factory(budget=384)
    rows = _compress_setup(engine, monkeypatch)
    host = _more(_real_hermes_merge(deepcopy(engine.compress(rows, force=True))))
    engine.ingest(host)
    before = _stored(engine)
    engine = _restart(engine, 0)
    try:
        _compress_setup(engine, monkeypatch, seed=False)
        engine.compress(host, force=True)
        engine.ingest(deepcopy(host) + GENUINE)
        assert _stored(engine) == before + [(len(before) + i + 1, m['role'], m['content']) for i, m in enumerate(GENUINE)]
    finally:
        engine.shutdown()


@pytest.mark.parametrize('budget', [512, 2048])
def test_objective_first_appears_mid_turn(factory, monkeypatch, budget):
    """test_rr_e3.py: the objective moves from raw tail to anchor on the second compression."""
    engine = factory(budget=budget)
    _compress_setup(engine, monkeypatch, seed=False)
    rows = _seed(engine, ('ORIGIN: build.' + ' pad' * 120, 'CONSTRAINT: no unsafe.' + ' pad' * 120,
                         'CORRECTION: allow ffi.' + ' pad' * 120))
    host = rows + [{'role': 'user', 'content': 'OBJECTIVE: refactor X.'}] + _tool_pair(0)
    engine.ingest(host)
    for lo, hi in [(1, 6), (6, 10), (10, 10)]:
        out = engine.compress(host, force=True)
        text = _text(out)
        assert HEADER in text and text.count('ORIGIN') == 1 and text.count('OBJECTIVE') == 1
        host = _real_hermes_merge(deepcopy(out))
        for i in range(lo, hi):
            host += _tool_pair(i)
        engine.ingest(host)
