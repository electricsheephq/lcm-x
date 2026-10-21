"""Portable regressions for C2's independently reproduced review findings."""
# ruff: noqa: F811
from copy import deepcopy

import pytest

from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.test_user_carry import factory, _seed, _emit, _node, _stored, _text, HEADER  # noqa: F401
from tests.test_issue_488_emission_descriptors import _record
from tests.test_issue_488_emission_proof import _real_hermes_merge


def _compress_setup(engine, monkeypatch, seed=True):
    engine._config.fresh_tail_count, engine._config.fresh_tail_max_tokens = 4, 0
    engine._config.leaf_chunk_tokens = 300
    monkeypatch.setattr('hermes_lcm.engine.summarize_with_escalation',
                        lambda **kw: ('Compact summary; user wording omitted. Expand for details: x', 1))
    return _seed(engine, tuple(f'Ask {i}: ' + 'alpha beta ' * 120 for i in range(8))) if seed else []


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


def test_header_is_real_user_without_carry(factory):
    from hermes_lcm.user_carry import HEADER as full_header
    engine = factory(budget=0)
    rows = _seed(engine, (full_header + ' User-authored note.',))
    assert not engine._store.read_metadata_json('real_user_scaffold_store_id:1')
    # A pre-upgrade row has no synthetic-scaffold provenance.
    assert engine._durable_real_user_messages()[0]['store_id'] == 1
    engine._prepare_retained_user_anchor([{'role': 'system', 'content': 'System'}, *rows])
    assert engine._load_retained_user_anchor_row()['store_id'] == 1


@pytest.mark.parametrize('echo', [False, True])
def test_off_header_preserves_carrier_and_source_accounting(factory, echo):
    from hermes_lcm.user_carry import HEADER as full_header
    engine = factory(budget=0)
    rows = _seed(engine, ('Origin', 'Please explain ' + full_header, 'Latest'))
    node_id = _node(engine)
    if echo:
        engine._dag._conn.execute('UPDATE summary_nodes SET summary = ? WHERE node_id = ?', ('Echo: ' + full_header, node_id))
        engine._dag._conn.commit()
    emitted = _emit(engine, rows)
    assert len(emitted) == 4  # base folds the generated row into the first user
    assert emitted[0]['content'].endswith(rows[2]['content'])
    _record(engine, rows, emitted)
    occurrences, _ = engine._replay_occurrences(emitted)
    assert engine._raw_message_tokens(emitted[0], occurrences[0]) == count_message_tokens(emitted[0])
    count = engine._raw_messages_counter(emitted, dict(zip(map(id, emitted), occurrences)))
    assert count(emitted) == count_messages_tokens(emitted)
    assert engine._raw_backlog_tokens(emitted) == count_messages_tokens(engine._raw_backlog_messages(emitted))


def _tool_pair(i):
    return [{'role': 'assistant', 'content': 'Tool step', 'tool_calls': [
        {'id': f'c{i}', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': f'c{i}', 'content': 'Tool result ' * 150}]


@pytest.mark.parametrize('authored', [False, True])
def test_exact_objective_anchor_coexists(factory, monkeypatch, authored):
    engine = factory(budget=2048)
    _compress_setup(engine, monkeypatch, seed=False)
    if authored:
        engine = factory('authored', budget=2048)
        rows = _seed(engine, ('ORIGIN', 'MIDDLE', '[Current user objective preserved from compacted history]\nAUTHORED'))
        engine._pending_context_anchor_messages = rows
        text = _text(_emit(engine, rows, tail=rows[-1:]))
        assert text.count('AUTHORED') == 1 and text.count('MIDDLE') == 1
    else:
        rows = _seed(engine, ('ORIGIN ' + 'pad ' * 120, 'CONSTRAINT ' + 'pad ' * 120, 'CORRECTION ' + 'pad ' * 120))
        rows += [{'role': 'user', 'content': 'OBJECTIVE: refactor module X.'}]
        for i in range(6):
            rows += _tool_pair(i)
        engine.ingest(rows)
        text = _text(engine.compress(rows, force=True))
        assert text.count('OBJECTIVE: refactor module X.') == 1
        assert text.count('ORIGIN') == 1
    assert HEADER in text


def test_failed_publication_keeps_fitted_result(factory, monkeypatch):
    results = []
    for budget in (0, 512):
        engine = factory(f'proof-{budget}', budget=budget)
        _compress_setup(engine, monkeypatch, seed=False)
        engine._set_context_length(16000, source='test')
        rows = _seed(engine, tuple(f'User {i}: ' + 'work item ' * 450 for i in range(12)))
        monkeypatch.setattr(engine, '_persist_compress_commit_proof', lambda *a, **kw: None)
        out = engine.compress(rows, current_tokens=count_messages_tokens(rows), force=True)
        assert count_messages_tokens(out) <= engine.threshold_tokens
        assert HEADER not in _text(out)
        results.append([{k: v for k, v in r.items() if k != 'message_uid'} for r in out])
    assert results[0] == results[1]


def test_late_exception_clears_status(factory, monkeypatch):
    engine = factory()
    rows = _compress_setup(engine, monkeypatch)
    def boom(*a):
        raise RuntimeError('late failure')
    monkeypatch.setattr(engine, '_rekey_host_rewrite_watch', boom)
    try:
        out = engine.compress(rows, force=True)
    except RuntimeError:
        out = []
    assert HEADER not in _text(out)
    assert engine.get_status()['user_carry']['delivered_tokens'] == 0


def test_collision_filters_owned_rows(factory):
    b = factory('shared', sid='P', conversation='convB')
    a = factory('shared', sid='P', conversation='convA')
    _seed(b, ('BRAVO foreign',))
    source = _seed(a, ('ALPHA origin',))
    a.on_session_end('P', source)
    a.on_session_start('A2', platform='cli', conversation_id='convA', context_length=128000,
                       boundary_reason='compression', old_session_id='P')
    rows = _seed(a, ('ALPHA after rotation', 'ALPHA live'))
    text = _text(_emit(a, rows, tail=rows[-2:]))
    assert 'BRAVO' not in text
    assert 'ALPHA origin' in text


@pytest.mark.parametrize('unit', ['👨\u200d👩\u200d👧\u200d👦', 'a\u0301'])
def test_excerpt_grapheme_boundary(factory, unit):
    import re
    engine = factory(budget=300)
    rows = _seed(engine, (unit * 4000, 'Older', 'Live'))
    text = _text(_emit(engine, rows, tail=rows[-2:]))
    a, b = map(int, re.search(r'chars (\d+)-(\d+)', text).groups())
    assert a % len(unit) == b % len(unit) == 0


def test_packet_has_end_before_live_turn(factory):
    engine = factory()
    rows = _seed(engine)
    view = _emit(engine, rows)
    text = _text(view)
    assert '[End earlier user messages]' in text
    assert text.index('[End earlier user messages]') < text.index(rows[2]['content'])


def test_missing_result_does_not_suppress_carry(factory, monkeypatch):
    engine = factory(budget=2048)
    rows = _compress_setup(engine, monkeypatch)
    engine._config.leaf_chunk_tokens = 600
    out = engine.compress(rows, force=True)
    host = _real_hermes_merge(deepcopy(out)) + [
        {'role': 'user', 'content': 'Next A: ' + 'task ' * 300, 'timestamp': 300.},
        _tool_pair(1)[0], {'role': 'assistant', 'content': 'Latest reply', 'timestamp': 303.}]
    engine.ingest(host)
    out = engine.compress(host, force=True)
    host = _real_hermes_merge(deepcopy(out)) + [
        {'role': 'user', 'content': 'Short follow-up B.', 'timestamp': 310.},
        {'role': 'assistant', 'content': 'Reply B', 'timestamp': 311.}]
    engine.ingest(host)
    out = engine.compress(host, force=True)
    assert HEADER in _text(out)
    assert any(r.get('content') == engine._missing_tool_result_stub('c1')['content'] for r in out)


@pytest.mark.parametrize('budget', [0, 2048])
def test_bounded_trial_assemblies(factory, monkeypatch, budget):
    engine = factory(budget=budget)
    _compress_setup(engine, monkeypatch, seed=False)
    engine._config.leaf_chunk_tokens = 400
    rows = _seed(engine, tuple(f'Ask {i}: ' + 'task detail ' * 55 for i in range(300)))
    trials = []
    original = engine._assemble_committed_compaction_context
    def track(*a, **kw):
        trials.append(kw.get('persist', True))
        return original(*a, **kw)
    monkeypatch.setattr(engine, '_assemble_committed_compaction_context', track)
    engine._config.threshold_full_sweep_enabled = True
    engine.threshold_tokens = 3000
    engine.compress(rows, current_tokens=count_messages_tokens(rows))
    assert trials.count(False) <= 2


from tests.test_issue_671_stub_first_exit import make_engine, summaries  # noqa: E402,F401


def test_stub_first_keeps_fitted_result(make_engine, summaries, caplog):
    from tests.test_issue_671_stub_first_exit import exit_engine, text_view, tool_view, run, THRESHOLD
    engine = exit_engine(make_engine, threshold_full_sweep_enabled=False, user_carry_tokens=512)
    first = run(engine, text_view(), caplog)
    result = run(engine, tool_view(prefix=first[1:]), caplog)
    assert count_messages_tokens(result) <= THRESHOLD
