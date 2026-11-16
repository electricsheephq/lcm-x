"""I1/I2: seeded, ordered-row differential; only proven queued text may differ.

Random sequences have 1..6 ingests. Historical control seeds retain their exact
original traces (the watched-head eviction control necessarily exceeds six).
Set LCMX_DIFF_CANDIDATE_TREE to a git-archive tree for a positive control.
"""
import json
import os
import random
import sqlite3
from types import SimpleNamespace

import pytest

if os.environ.get("LCM_DIFFERENTIAL") != "1":
    pytest.skip("opt in with LCM_DIFFERENTIAL=1", allow_module_level=True)

from tests.differential import merged_user_turn_controls as base
from tests.differential import paired
from tests.differential.paired import versions  # noqa: F401
from tests.differential.session import FaultRun, Trace, assert_faults, audit_fault, fault_counts, freeze, label_missing
from tests.phase0_hygiene import product_env_keys


RANDOM_SEEDS = tuple(range(2707400, 2707720))
WIDE_RELOAD_SEEDS = tuple(range(2708000, 2708100))
RELOAD_SEEDS = tuple(range(2707800, 2707900)) + WIDE_RELOAD_SEEDS
CONTROL_SEEDS = {
    2707101: 'r1-A-alias-fallback',
    2707102: 'r1-B-fallback-replay',
    2707103: 'r1-C-fallback-evidence',
    2707201: 'r2-watched-eviction',
    2707202: 'r2-shifted-fallback',
    2707203: 'r2-new-uid-fallback',
    2707301: 'r3-shifted-no-fault',
    2707302: 'r3-S2-no-failure',
}


def _normalise(rows, preserved, reference_rows=()):
    return list(rows) if paired.CANDIDATE_CONTRACT == 'records-v1' else paired.normalise_rows(rows, preserved, reference_rows)


def _shifted_replay(tmp_path, monkeypatch, trees, record_property, capture):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    results, raw = {}, {}
    for name in ('main', 'candidate'):
        directory = tmp_path / name
        engine = base._open(trees[name], directory, monkeypatch).engine
        initial = [base._row('Q', 1., 'assistant'),
                   dict(base._row('P1\n\nR', 10.), message_uid='uid-A'),
                   base._row('A', 21., 'assistant'),
                   dict(base._row('P2\n\nR', 30.), message_uid='uid-B'),
                   base._row('B', 41., 'assistant')]
        try:
            engine.ingest(initial)
            base._seed_round3_aliases(engine)
        finally:
            engine.shutdown()
        ctx = base._open(trees[name], directory, monkeypatch)
        engine = ctx.engine
        try:
            live = [initial[0], dict(initial[1], content='R', timestamp=20.), initial[2],
                    dict(initial[3], content='R', timestamp=40.), initial[4]]
            engine.ingest(live)
            incoming = [base._row('summary', 1., 'assistant'), live[3], live[4]]
            if capture:
                incoming = base._capture(ctx, incoming)
            engine.ingest(incoming)
            raw[name] = base._identities(engine)
            results[name] = [_normalise(raw[name], {
                ('user', 'P2\n\nR2', 40., 'parent', 'conv'):
                ('user', 'R2', 50., 'parent', 'conv')}, raw.get('main', ()))] if capture else [raw[name]]
        finally:
            engine.shutdown()
    record_property('stored_rows', json.dumps(raw))
    base._compare_round2(results, record_property)
    assert len(raw['main']) == (8 if capture else 6)
    return raw


def _named_shifted(tmp_path, monkeypatch, differential_versions, record_property):
    _shifted_replay(tmp_path, monkeypatch, differential_versions, record_property, False)


def _named_capture(tmp_path, monkeypatch, differential_versions, record_property):
    _shifted_replay(tmp_path, monkeypatch, differential_versions, record_property, True)


def _historical(seed, tmp_path, monkeypatch, differential_versions, record_property):
    """Every finding from rounds 1..3 is an executable fixed seed, unchanged."""
    args = (tmp_path, monkeypatch, differential_versions, record_property)
    if seed == 2707101:
        base.test_round2_existing_alias_fallback_matches_main(*args, seed_alias=True)
    elif seed == 2707102:
        base.test_round2_fallback_then_again_matches_main(*args, failure='transient')
    elif seed == 2707103:
        base.test_round2_fallback_has_own_evidence_retry(*args)
    elif seed == 2707201:
        base.test_round3_watched_capture_stays_retired_after_eviction(*args)
    elif seed == 2707202:
        base.test_round3_rewritten_prefix_drops_false_fallback_marks(*args)
    elif seed == 2707203:
        base.test_round3_new_uid_keeps_host_time_through_fallback(*args)
    else:
        _shifted_replay(*args, capture=seed == 2707302)


def _plan(seed):
    rng = random.Random(seed)
    return {
        'ingests': rng.randint(1, 6),
        'aliases': rng.choice([False, True]),
        'cache': rng.choice(['cold', 'warm-newest', 'recreated']),
        'steps': [dict(prefix=rng.choice(['unchanged', 'whole-replace', 'prefix-keeping', 'summary-shift']),
                       uid=rng.choice(['stable', 'aligned-change', 'moved']),
                       capture=rng.choice(['none', 'proven', 'watched']),
                       fault=rng.choice(['none', 'busy', 'evidence', 'preservation-once', 'persistent']))
                  for _ in range(5)],
    }


def _capture_step(ctx, live, step, seed, turn, preserved):
    """Mutate retained host objects identically, independent of product state."""
    if step['capture'] == 'none':
        return live
    if hasattr(ctx.engine, '_paired_trace'):
        ctx.engine._paired_trace.capture_kind = step['capture']
    newest, earlier = f'R-{seed}-{turn}', f'P-{seed}-{turn}'
    stamp = float(100 + turn * 10)
    if step['capture'] == 'watched':
        # Host input must be identical for both products. Retain a host row
        # from the preceding ingest, regardless of either engine's watch state.
        index = next((i for i, row in enumerate(live) if row['role'] == 'user'), None)
        if index is None:
            return live
        head = live[index]
        ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message=newest,
                                  conversation_history=[*live[:index], head,
                                                        base._row(earlier, stamp + 5),
                                                        base._row(newest, stamp + 10)])
        head.update(content=newest, timestamp=stamp + 10)
        return live
    head, tail = base._row(earlier, stamp), base._row(newest, stamp + 10)
    ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message=newest,
                              conversation_history=[*live, head, tail])
    if step['prefix'] == 'prefix-keeping':
        head.update(content=earlier + '\n\n' + newest, _merged_turn_prefix=earlier + '\n\n')
    else:
        head.update(content=newest, timestamp=stamp + 10)
        preserved[('user', earlier + '\n\n' + newest, stamp, 'parent', 'conv')] = (
            'user', newest, stamp + 10, 'parent', 'conv')
    return [*live, head, base._row(f'reply-{seed}-{turn}', stamp + 11, 'assistant')]


def _faults(engine, name, kind, monkeypatch):
    calls = []
    if kind == 'none':
        return calls
    target = '_store_queued_user_rows' if kind in ('preservation-once', 'persistent') else '_identity_anchor_prematch'
    present = hasattr(engine, target)
    audit = audit_fault(engine, kind, present)
    if audit.stubbed or not present:
        return calls
    if kind in ('preservation-once', 'persistent'):
        from .record_contract import writer_fault
        engine._paired_trace.faults.remove(audit)
        writer_fault(engine, kind, monkeypatch, calls)
    elif kind in ('busy', 'evidence'):
        original = engine._identity_anchor_prematch

        def prematch(*args, **kwargs):
            # The fault belongs to the first ingest attempt, not to the first call: a lazy attempt that
            # reclassifies under the lock re-runs its classification inside the same attempt, as eager's one pass.
            attempt = engine.__dict__.get('_ingest_attempt_index')
            if attempt is None:
                pytest.fail("setup: _ingest_attempt_index missing")
            calls.append((kind, attempt))
            if attempt == calls[0][1]:
                if len(calls) == 1:
                    audit.called += 1
                    # raised counts faulted cases, not raises.
                    audit.raised += 1
                if kind == 'busy':
                    raise sqlite3.OperationalError('database is locked')
                raise RuntimeError('seeded evidence fault')
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, '_identity_anchor_prematch', prematch)
    return calls


def _run_sequence(module, name, directory, monkeypatch, seed, plan, references=()):
    ctx = base._open(module, directory, monkeypatch)
    engine = ctx.engine
    live = [base._row('Q', 1., 'assistant'),
            dict(base._row('P1\n\nR' if plan['aliases'] else 'R', 10.), message_uid='uid-A'),
            base._row('A', 21., 'assistant'),
            dict(base._row('P2\n\nR' if plan['aliases'] else 'R', 30.), message_uid='uid-B'),
            base._row('B', 41., 'assistant')]
    stages, preserved, faults, host_views, raw_stages = [], {}, [], [], []

    def ingest():
        host_views.append(json.loads(json.dumps(live)))
        engine.ingest(live)
        raw_stages.append(base._identities(engine))

    try:
        ingest()
        stages.append(base._identities(engine))
        if plan['aliases']:
            base._seed_round3_aliases(engine)
        # A cold reader loads hash-bound aliases without an ingest/cache proof.
        engine.shutdown()
        ctx = base._open(module, directory, monkeypatch)
        engine = ctx.engine
        if plan['aliases']:
            live = [live[0], dict(live[1], content='R', timestamp=20.), live[2],
                    dict(live[3], content='R', timestamp=40.), live[4]]
        for turn in range(plan['ingests'] - 1):
            # Warm-newest and recreated cases establish the newest-stamp view
            # before mutating it; the next ingest tests that cache or a new reader.
            if turn == 0 and plan['cache'] != 'cold':
                ingest()
                stages.append(base._identities(engine))
                if plan['cache'] == 'recreated':
                    engine.shutdown()
                    ctx = base._open(module, directory, monkeypatch)
                    engine = ctx.engine
                continue
            step = plan['steps'][turn]
            users = [i for i, row in enumerate(live) if row['role'] == 'user']
            if users and step['uid'] == 'aligned-change':
                index = users[0]
                live[index] = dict(live[index], message_uid=f'uid-new-{seed}-{turn}')
            elif len(users) > 1 and step['uid'] == 'moved':
                # Same UID/content moves to another slot; the prefix does not align.
                live = [base._row('summary', 1., 'assistant'), *live[users[-1]:]]
            if step['prefix'] == 'whole-replace':
                live = [base._row(f'replacement-{seed}-{turn}', 2., 'assistant')]
            elif step['prefix'] == 'summary-shift' and users:
                users = [i for i, row in enumerate(live) if row['role'] == 'user']
                live = [base._row('summary', 1., 'assistant'), *live[users[-1]:]] if users else live
            live = _capture_step(ctx, live, step, seed, turn, preserved)
            with monkeypatch.context() as patch:
                calls = _faults(engine, name, step['fault'], patch)
                ingest()
                faults.append(list(calls))
            stages.append(_normalise(base._identities(engine), preserved,
                                               references[len(stages)] if references else ()))
    finally:
        engine.shutdown()
    return stages, faults, host_views, raw_stages


def _seeded(seed, tmp_path, monkeypatch, differential_versions, record_property):
    monkeypatch.setenv('LCM_IDENTITY_ANCHOR', '1')
    monkeypatch.setenv('LCM_INGEST_LEDGER', 'shadow')
    plan = _plan(seed)
    record_property('seed_plan', json.dumps(dict(seed=seed, **plan)))
    results, inputs, raw = {}, {}, {}
    for name in ('main', 'candidate'):
        results[name], faults, inputs[name], raw[name] = _run_sequence(differential_versions[name], name, tmp_path / name,
                                              monkeypatch, seed, plan, raw.get('main', ()))
        record_property(name + '_faults', json.dumps(faults))
    record_property('host_views', json.dumps(inputs))
    record_property('stored_rows', json.dumps(raw))
    assert inputs['candidate'] == inputs['main'], 'Differential host inputs must be identical'
    base._compare_round2(results, record_property)


def _gateway_reload(transcript):
    # floor 345cd2b: gateway/run_turn.py:1921 reloads EACH turn;
    # gateway/session_transcript.py:496-510 requests repair_alternation=True;
    # hermes_state_messages.py:955-978,995-997 restores copies, strips text,
    # stamps _DB_PERSISTED_MARKER, then agent_runtime_helpers.py:523-552 joins
    # plain user rows into the first dict. Fixtures exclude steer/summary rows.
    history = []
    for row in transcript:
        row = dict(row, content=row['content'].strip(), _db_persisted=True)
        if history and history[-1]['role'] == row['role'] == 'user':
            old, new = history[-1]['content'], row['content']
            history[-1]['content'] = old + '\n\n' + new if old and new else old or new
        else:
            history.append(row)
    return history


def _reload_case(seed, directory, patch, trees, record):
    if seed in WIDE_RELOAD_SEEDS:
        return _wide_reload_case(seed, directory, patch, trees, record)
    rng = random.Random(seed)
    patch.setenv('LCM_IDENTITY_ANCHOR', str(rng.randrange(2)))
    # A durable pending-user transcript is the reload input. It is separate
    # from the overwritten live objects: repair never rewrites durable rows.
    pending = [base._row(f'P-{seed}-{i}', 10. + i) for i in range(rng.randint(1, 2))]
    pending.append(base._row(f'R-{seed}', 20.))
    for name in ('main', 'candidate'):
        ctx = base._open(trees[name], directory / name, patch)
        engine = ctx.engine
        q, reply = base._row('Q', 1., 'assistant'), base._row('A', 21., 'assistant')
        try:
            engine.ingest([q])
            live = [dict(row) for row in pending]
            ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                      user_message=live[-1]['content'], conversation_history=[q, *live])
            live[0].update(content=live[-1]['content'], timestamp=live[-1]['timestamp'])
            engine.ingest([q, live[0], reply])
            history = _gateway_reload([q, *pending, reply])
            if hasattr(engine, '_paired_trace'):
                trace = engine._paired_trace
                trace.reloads = [(trace.record_captures[-1]['ids'][0], history[1], '\n\n'.join(r['content'] for r in pending))]
            engine.ingest([*history, base._row(f'N-{seed}', 30.), base._row('B', 31., 'assistant')])
            joined = history[1]['content']
            rows = base._rows(engine)
            record(name + '_reload', json.dumps(dict(seed=seed,
                   stores_joined=any(row['role'] == 'user' and row['content'] == joined for row in rows),
                   predecessor_spans=sum(row['role'] == 'user' and row['content'] == joined
                                         for row in rows) * (len(pending) - 1))))
        finally:
            engine.shutdown()


def _wide_reload_plan(seed):
    rng = random.Random(seed)
    anchor, count = rng.randrange(2), rng.randint(1, 4)
    pending = [base._row(f'P-{seed}-{i}', 10. + i) for i in range(count)]
    blank, split = rng.choice([False, True]), rng.choice([False, True])
    if blank:
        pending[rng.randrange(count)]['content'] = ' \t '
    pending.append(base._row(f'R-{seed}', 20.))
    return anchor, pending, rng.randint(1, count) if split else None


def _wide_reload_case(seed, directory, patch, trees, record):
    anchor, pending, split = _wide_reload_plan(seed)
    patch.setenv('LCM_IDENTITY_ANCHOR', str(anchor))
    for name in ('main', 'candidate'):
        ctx = base._open(trees[name], directory / name, patch)
        engine = ctx.engine
        q, reply = base._row('Q', 1., 'assistant'), base._row('A', 21., 'assistant')
        try:
            engine.ingest([q])
            live = [dict(row) for row in pending]
            ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv',
                                    user_message=live[-1]['content'], conversation_history=[q, *live])
            live[0].update(content=live[-1]['content'], timestamp=live[-1]['timestamp'])
            engine.ingest([q, live[0], reply])
            # The durable transcript can place a reply between queued users.
            # The floor's reload strips each row and joins only consecutive users.
            durable = [dict(row) for row in pending]
            if split is not None:
                durable.insert(split, base._row('interleaved reply', 10. + split - .5, 'assistant'))
            history = _gateway_reload([q, *durable, reply])
            trace = engine._paired_trace
            trace.reloads = [(trace.record_captures[-1]['ids'][0], history[1],
                              '\n\n'.join(r['content'] for r in pending))]
            engine.ingest([*history, base._row(f'N-{seed}', 30.), base._row('B', 31., 'assistant')])
            record(name + '_reload', json.dumps(dict(seed=seed, pending=len(pending) - 1,
                   whitespace=any(not r['content'].strip() for r in pending), split=split)))
        finally:
            engine.shutdown()


CASES = [*RANDOM_SEEDS, *CONTROL_SEEDS, *RELOAD_SEEDS, 'shifted-replay', 'capture-without-fault']


def _execute_case(case, directory, versions, stub_faults=False, fake=None):  # noqa: F811
    if os.environ.get('LCMX_DIFF_FAKE_RECORDS') is not None and any(k in os.environ for k in ('CI', 'GITHUB_ACTIONS')):
        pytest.fail('LCMX_DIFF_FAKE_RECORDS is forbidden in CI')
    fake = os.environ.get('LCMX_DIFF_FAKE_RECORDS') == '1' if fake is None else fake
    if fake:
        versions = dict(versions, candidate=SimpleNamespace(register=versions['main'].register))
    traces, properties = {}, [('fake_candidate', 'fake_records.install')] if fake else []

    def record(name, value):
        properties.append((name, value))

    with pytest.MonkeyPatch.context() as patch:
        for key in product_env_keys(paired.ROOT):
            patch.delenv(key, raising=False)
        original = base._open

        def opened(module, path, monkeypatch):
            key = 'main' if module is versions['main'] else 'candidate'
            trace = traces.setdefault(key, Trace(stub_faults=stub_faults))
            trace.case = case
            if key == 'candidate':
                trace.baseline_stages = traces['main'].ingests
            return original(module, path, monkeypatch, trace=trace, fake=fake and key == 'candidate')

        patch.setattr(base, '_open', opened)
        if case == 'shifted-replay':
            _named_shifted(directory, patch, versions, record)
        elif case == 'capture-without-fault':
            _named_capture(directory, patch, versions, record)
        elif case in CONTROL_SEEDS:
            _historical(case, directory, patch, versions, record)
        elif case in RELOAD_SEEDS:
            _reload_case(case, directory, patch, versions, record)
        else:
            _seeded(case, directory, patch, versions, record)
    return properties, traces


def _label_violations(case, trace, stubbed):
    """Classify by host evidence and an actual fault-free counterfactual."""
    entries = []
    violations = trace.violations()
    clean = {(v['stage'], v['kind'], v['value']): v['count']
             for v in stubbed.violations()} if stubbed else {}
    for violation in violations:
        stage, kind, value, count = (violation[k] for k in ('stage', 'kind', 'value', 'count'))
        views = trace.ingests[stage - 1]['views']
        if kind == 'missing':
            if case in WIDE_RELOAD_SEEDS and views[violation['view']]['kind'] == 'hook':
                from .record_contract import declarations
                assert any(e['capture_view'] == violation['view'] and e['stage'] <= stage
                           and e['expected']['content'] == value
                           for e in declarations(case, trace.ingests, paired.BASELINE_REF)), violation
                label = '#2707 proven'
            else:
                label = ('#2707 reload' if case in RELOAD_SEEDS and views[violation['view']]['kind'] == 'ingest'
                         else label_missing(case, views, violation))
        elif case in WIDE_RELOAD_SEEDS:
            # A widened reload can append the already stored newest user/reply.
            # Pin this baseline shape separately from injected-fault duplicates.
            fields = dict(value)
            assert stage == 3 and not trace.faults and count == 1, violation
            assert value in trace.ingests[stage - 2]['rows'], violation
            assert (fields['role'], fields['content'], fields['observed_at']) in (
                ('assistant', 'A', 21.), ('user', f'R-{case}', 20.)), violation
            label = '#2707 reload duplicate'
        else:
            remaining = clean.get((stage, kind, value), 0)
            if remaining:
                fields = dict(value)
                uid_views = [{row['message_uid'] for row in view['rows']
                              if row['role'] == 'user' and row['content'] == fields['content']
                              and row.get('timestamp') == fields['observed_at'] and row.get('message_uid')}
                             for view in views]
                uid_views = [uids for uids in uid_views if uids]
                changed = any(a != b for a in uid_views for b in uid_views)
                assert fields['role'] == 'user' and changed, (case, violation, uid_views)
                assert remaining <= count, (case, violation, remaining)
                entries.append((stage, kind, value, None, remaining, '#2720 uid change'))
                count -= remaining
                if not count:
                    continue
            if count:
                first = next(v['stage'] for v in violations
                             if v['kind'] == 'duplicate' and v['value'] == value
                             and v['count'] > clean.get((v['stage'], 'duplicate', value), 0))
                faults = [f.kind for f in trace.faults
                          if f.stage <= first and f.raised and f.kind in ('busy', 'evidence')]
                assert faults, (case, violation, trace.faults)
                label = '#1815 fall-open, ' + faults[-1] + ' fault'
        entries.append((stage, kind, value, violation.get('view'), count, label))
    return entries


@pytest.mark.parametrize('case', [2707102, 2707202, 2707203])
def test_stubbed_control_with_preservation_seam(case, tmp_path, versions, monkeypatch):  # noqa: F811
    from .trees import check_host, PrerequisiteUnavailable, skip_or_fail
    try:
        check_host()
    except PrerequisiteUnavailable as exc:
        skip_or_fail(str(exc))
    monkeypatch.setattr(paired, 'CANDIDATE_CONTRACT', 'records-v1')
    _, traces = _execute_case(case, tmp_path, versions, stub_faults=True, fake=True)
    if case == 2707203:
        assert [dict(r)['observed_at'] for r in traces['candidate'].ingests[-1]['records']
                if dict(r)['content'] == 'P2'] == [40.]
    faults = [fault for fault in traces['candidate'].faults if fault.kind == 'preservation-once']
    assert faults and all(fault.present and fault.stubbed and not fault.called and not fault.raised for fault in faults)


@pytest.fixture(scope='module', autouse=True)
def fake_record_contract_switch():
    with pytest.MonkeyPatch.context() as patch:
        if 'LCMX_DIFF_FAKE_RECORDS' in os.environ and any(k in os.environ for k in ('CI', 'GITHUB_ACTIONS')):
            pytest.fail('LCMX_DIFF_FAKE_RECORDS is forbidden in CI')
        if os.environ.get('LCMX_DIFF_FAKE_RECORDS') == '1':
            patch.setattr(paired, 'CANDIDATE_CONTRACT', 'records-v1')
        yield


@pytest.fixture(scope='module')
def fault_run(request, record_testsuite_property):
    if paired.CANDIDATE_CONTRACT == 'records-v1' and 'LCMX_DIFF_BASELINE_REF' in os.environ:
        from .trees import skip_or_fail
        skip_or_fail('records-v1 requires pinned declarations; baseline override')
    run = FaultRun()
    selected = {item.callspec.params['sequence'] for item in request.session.items
                if item.module is request.module and hasattr(item, 'callspec')
                and 'sequence' in item.callspec.params}
    yield run
    result = run.finish(CASES, selected)
    try:
        record_testsuite_property('merged_user_turn_fault_floor', json.dumps(result))
        for item in request.session.items:
            if item.module is request.module and hasattr(item, 'callspec') and 'sequence' in item.callspec.params:
                paired.write_diagnostics(item.callspec.params['sequence'], item.originalname,
                                         [*item.user_properties, ('merged_user_turn_fault_floor', json.dumps(result))])
    finally:
        if paired.CANDIDATE_CONTRACT == 'records-v1':
            from tests.differential.record_contract import mandatory_floor
            mandatory_floor(result, run.cases)
        assert not result['errors'], result['errors']


@pytest.fixture(scope='class')
def sequence(request, tmp_path_factory, versions, fault_run):  # noqa: F811 - imported fixture
    case = request.param
    directory = tmp_path_factory.mktemp(f'merged-{case}')
    properties, traces = _execute_case(case, directory, versions)
    from tests.differential.merged_user_turns_extras import DECLARED
    from tests.differential.record_contract import declaration_stages, declarations
    if paired.resolve(os.environ.get('LCMX_DIFF_BASELINE_REF', paired.BASELINE_REF)) == paired.BASELINE_REF:
        assert declarations(case, declaration_stages(traces['main']), paired.BASELINE_REF) == DECLARED[case], \
            f'{case}: live baseline declarations differ'
    else:
        properties.append(('live_declarations', 'baseline override'))
    fault_run.record(case, traces)
    for name, trace in traces.items():
        counts = fault_counts(trace.faults)
        properties.append((name + '_fault_counts', json.dumps(counts)))
    stubbed = None
    if any(v['kind'] == 'duplicate' for v in traces['main'].violations()):
        # Rerun the identical host plan with all fault wrappers disabled.
        _, clean = _execute_case(case, directory / 'fault-free',
                                 dict(main=versions['main'], candidate=versions['candidate']),
                                 stub_faults=True, fake=False)
        stubbed = clean['main']
        properties.append(('fault_free_violations', json.dumps(stubbed.violations())))
    labels = _label_violations(case, traces['main'], stubbed)
    properties.append(('classified_baseline', json.dumps(labels)))
    return case, properties, traces


@pytest.mark.parametrize('sequence', CASES, indirect=True, scope='class')
class TestMergedUserTurns:
    def test_differential(self, sequence, record_property):
        case, properties, traces = sequence
        for name, value in properties:
            record_property(name, value)
        hosts = {name: [stage['views'] for stage in trace.ingests] for name, trace in traces.items()}
        record_property('all_host_views', json.dumps(hosts))
        record_property('all_stored_rows', json.dumps({
            name: [stage['rows'] for stage in trace.ingests] for name, trace in traces.items()}))
        assert hosts['main'] == hosts['candidate'], 'Differential host inputs must be identical'
        assert paired.CANDIDATE_CONTRACT in ('main', 'records-v1')
        for stage, (main, candidate) in enumerate(zip(traces['main'].ingests, traces['candidate'].ingests, strict=True), 1):
            assert main['canonical'] == candidate['canonical'], f'{case}: canonical row changed'
            assert not main['records'], f'{case}: main stored queued records'
            if paired.CANDIDATE_CONTRACT == 'main':
                assert not candidate['records'], f'{case}: main contract stored queued records'
            else:
                from tests.differential.record_contract import active, check_stage
                from tests.differential.merged_user_turns_extras import DECLARED
                check_stage(candidate, main, active(DECLARED[case], stage))
                from tests.differential.merged_user_turns_extras import FAULT_ABSENT
                missing = {(e['capture_view'], e['predecessor_ordinal'], e['expected']['content'])
                           for e in DECLARED[case] if e['stage'] <= stage and
                           (e['due_stage'] is None or e['due_stage'] > stage)}
                assert set(candidate['retrieval_missing']) == missing, f'{case}: retrieval-missing'
                assert {tuple(row[:3]) for row in FAULT_ABSENT[case]} == {
                    (e['capture_view'], e['predecessor_ordinal'], e['expected']['content'])
                    for e in DECLARED[case] if e['due_stage'] is None}
        assert_faults(case, traces)
        differences = [json.loads(value) for name, value in properties if name == 'differences']
        assert (differences or case in RELOAD_SEEDS) and not any(differences), f'{case}: {differences}'
        # The source scenarios retain their own exact substitutions. Historical
        # controls additionally check setup and warm ingests, which the source
        # compared only at selected checkpoints.
        if case in CONTROL_SEEDS or isinstance(case, str):
            stages = {name: [_normalise(stage['rows'], stage['preserved'],
                                                   traces['main'].ingests[index]['rows'])
                             for index, stage in enumerate(trace.ingests)] for name, trace in traces.items()}
            assert not paired.compare_stages(stages, record_property), case

    def test_absolute(self, sequence, record_property):
        from collections import Counter
        from .merged_user_turns_expected import EXPECTED

        case, properties, traces = sequence
        violations = {name: trace.violations() for name, trace in traces.items()}
        for name, value in properties:
            record_property(name, value)
        record_property('all_host_views', json.dumps({
            name: [s['views'] for s in trace.ingests] for name, trace in traces.items()}))
        record_property('all_stored_rows', json.dumps({
            name: [s['rows'] for s in trace.ingests] for name, trace in traces.items()}))
        record_property('absolute_violations', json.dumps(violations))
        record_property('binding_diagnostics', json.dumps({
            name: [s['binding_diagnostics'] for s in trace.ingests] for name, trace in traces.items()}))
        counters = {name: Counter((v['stage'], v['kind'], freeze(v['value']), v.get('view'))
                                 for v in rows for _ in range(v['count']))
                    for name, rows in violations.items()}
        classified = next(json.loads(value) for name, value in properties if name == 'classified_baseline')
        actual_pin = Counter((stage, kind, freeze(value), view, label)
                             for stage, kind, value, view, count, label in classified for _ in range(count))
        pinned = Counter((stage, kind, freeze(value), view, label) for stage, kind, value, view, count, label
                         in EXPECTED.get(case, ()) for _ in range(count))
        effective = paired.resolve(os.environ.get('LCMX_DIFF_BASELINE_REF', paired.BASELINE_REF))
        matches = actual_pin == pinned
        record_property('baseline_pin', json.dumps(dict(ref=effective, matches=matches,
                        skipped=None if effective == paired.BASELINE_REF else 'baseline override')))
        subset = not (counters['candidate'] - counters['main'])
        record_property('candidate_violation_subset', json.dumps(subset))
        assert_faults(case, traces)
        if effective == paired.BASELINE_REF:
            assert matches, (f'{case}: baseline pin mismatch at {paired.BASELINE_REF}; '
                             f'pins must be regenerated for it: {violations["main"]}')
        from tests.differential.merged_user_turns_extras import RETRIEVAL_MISSING
        if effective == paired.BASELINE_REF:
            assert [s['retrieval_missing'] for s in traces['main'].ingests] == RETRIEVAL_MISSING[case]['values']
        assert subset, f'{case}: added candidate violations: {counters["candidate"] - counters["main"]}'


@pytest.mark.parametrize('mutation,message', [
    ('correct', None), ('per-owner', None), ('retry', None),
    ('listing', 'canonical listing changed'), ('missing', 'declared record multiset'), ('extra', 'declared record multiset'),
    ('listing-content', 'canonical listing changed'),
    ('listing-extra', 'listed records differ from projection'),
    ('text', 'declared record multiset'), ('time', 'declared record multiset'), ('duplicate', 'declared record multiset'),
    ('hidden', 'retrieval-missing'), ('superseded', 'superseded without occurrence witness'),
    ('inert', 'declared fault preservation-once never fired'), ('skipped', 'floor skipped'),
    ('canonical', 'canonical row changed'),
    ('early', '2707102: declared record multiset'),
    ('unsuperseded', 'withheld locator missing'),  # T1-167: overlap rejects the visible duplicate first.
    ('error', 'lcm_load_session: injected tool error'), ('new-range-missing', 'declared record multiset'),
])
def test_fake_record_contract(mutation, message, tmp_path, versions, monkeypatch, record_property):  # noqa: F811
    from tests.differential import fake_records
    from tests.differential.merged_user_turns_extras import DECLARED
    from tests.differential.record_contract import active, check_stage, mandatory_floor
    new_range = mutation == 'new-range-missing'
    mutation = 'missing' if new_range else mutation
    if mutation == 'unsuperseded':
        monkeypatch.setattr(fake_records, 'withholding', lambda _engine: [])
    install = fake_records.install
    def modified(engine, entries, patch):
        tools = install(engine, entries, patch)
        writer = engine._store_queued_user_rows
        def write(rows):
            rows = [dict(row) for row in rows]
            if rows:
                if mutation == 'missing':
                    rows = rows[1:]
                if mutation in ('extra', 'duplicate'):
                    rows.append(dict(rows[0], ordinal=999) if mutation == 'extra' else rows[0])
                if mutation == 'text':
                    rows[0]['content'] = 'wrong text'
                if mutation == 'time':
                    rows[0]['observed_at'] = -1.
                if mutation == 'superseded':
                    rows[0]['superseded_by'] = rows[0]['owner_store_id']
            writer(rows)
            if mutation == 'duplicate' and rows:
                engine._store._conn.execute('INSERT INTO queued_user_rows SELECT * FROM queued_user_rows LIMIT 1')
            if mutation == 'canonical':
                engine._store._conn.execute("UPDATE messages SET content='changed'")
        patch.setattr(engine, '_store_queued_user_rows', write)
        if mutation in ('early', 'unsuperseded'):
            ingest = engine.ingest
            def defective(*args, **kwargs):
                result = ingest(*args, **kwargs)
                if mutation == 'early':
                    from .record_contract import expected_record
                    writer([expected_record(e) for e in entries
                            if e['stage'] <= len(engine._paired_trace.ingests) + 1])
                else:
                    engine._store._conn.execute('UPDATE queued_user_rows SET superseded_by=NULL')
                return result
            patch.setattr(engine, 'ingest', defective)
        if mutation == 'error':
            patch.setattr(tools, 'lcm_load_session',
                          lambda *_a, **_k: json.dumps(dict(error='injected tool error')))
        if mutation == 'hidden':
            patch.setattr(tools, 'lcm_grep', lambda *_a, **_k: json.dumps(dict(results=[])))
        if mutation in ('listing', 'listing-content', 'listing-extra'):
            load = tools.lcm_load_session
            def drop(args, **kwargs):
                reply = json.loads(load(args, **kwargs))
                if mutation == 'listing':
                    reply['messages'] = [r for r in reply['messages'] if r.get('store_id') != 1]
                elif mutation == 'listing-content':
                    for row in reply['messages']:
                        if row.get('store_id') == 1 and 'owner_store_id' not in row:
                            row['content'] = ''
                else:
                    for index, row in enumerate(reply['messages']):
                        if row.get('store_id') == 1 and 'owner_store_id' not in row:
                            from .session import queued_rows
                            records = queued_rows(engine)
                            if not records:
                                continue
                            reply['messages'].insert(index, dict(owner_store_id=1, ordinal=999,
                                role='user', content='invented queued message',
                                reference=fake_records.reference(dict(records[0]))))
                            break
                return json.dumps(reply)
            patch.setattr(tools, 'lcm_load_session', drop)
        return tools
    monkeypatch.setattr(fake_records, 'install', modified)
    monkeypatch.setattr(paired, 'CANDIDATE_CONTRACT', 'records-v1')
    monkeypatch.setattr(fake_records, 'WRITE_STYLE', mutation)
    def verify():
        if mutation in ('inert', 'skipped'):
            return mandatory_floor(dict(evaluated=mutation != 'skipped'), {})
        case = 2708002 if new_range else 2707480 if mutation in ('per-owner', 'retry') else 2707102 if mutation == 'early' else 2707802 if mutation == 'unsuperseded' else 2707800
        _, traces = _execute_case(case, tmp_path, versions, fake=True)
        if mutation in ('per-owner', 'retry'):
            assert any(f.called > 1 and f.raised == f.called for f in traces['candidate'].faults)
        for stage, (main, candidate) in enumerate(zip(traces['main'].ingests, traces['candidate'].ingests, strict=True), 1):
            try:
                check_stage(candidate, main, active(DECLARED[case], stage))
                missing = [(e['capture_view'], e['predecessor_ordinal'], e['expected']['content'])
                           for e in DECLARED[case] if e['stage'] <= stage and
                           (e['due_stage'] is None or e['due_stage'] > stage)]
                assert candidate['retrieval_missing'] == missing, 'retrieval-missing'
            except AssertionError as exc:
                raise AssertionError(f'{case}: {exc}') from exc
    if message:
        with pytest.raises(AssertionError, match=message) as caught:
            verify()
        record_property('mutation_failure', str(caught.value))
    else:
        verify()


@pytest.mark.parametrize('case', [2707102, 2707202, 2707203, 2707405, 2707407])
def test_record_fault_floor_and_frozen_retry(case, tmp_path, versions, monkeypatch):  # noqa: F811
    from tests.differential.record_contract import FAULT_CONTROLS
    monkeypatch.setattr(paired, 'CANDIDATE_CONTRACT', 'records-v1')
    _, traces = _execute_case(case, tmp_path, versions, fake=True)
    counts = fault_counts(traces['candidate'].faults)
    for kind in FAULT_CONTROLS[case]:
        assert counts[kind]['requested'] and not counts[kind]['inert'] and counts[kind]['called'] and counts[kind]['raised']
    assert all(a['canonical'] == b['canonical'] for a, b in zip(traces['main'].ingests, traces['candidate'].ingests, strict=True))
    if case == 2707102:
        from tests.differential.merged_user_turns_extras import DECLARED
        from tests.differential.record_contract import expected_record
        assert not traces['candidate'].ingests[1]['records']
        assert traces['candidate'].ingests[-1]['records'] == [freeze(expected_record(e)) for e in DECLARED[case]]


def test_record_declaration_completeness():
    from collections import Counter
    from tests.differential.merged_user_turns_expected import EXPECTED
    from tests.differential.merged_user_turns_extras import DISPOSITIONS
    from tests.differential.merged_user_turns_extras import FAULT_ABSENT
    assert Counter(r[3] for rows in FAULT_ABSENT.values() for r in rows) == {
        'injected persistent writer fault': 23, 'once writer fault at final ingest': 19}
    for case in DISPOSITIONS:
        assert Counter(tuple(row[:5]) for row in DISPOSITIONS[case]) == Counter(
            (stage, kind, value, view, count) for stage, kind, value, view, count, label in EXPECTED.get(case, ())
            if label == '#2707 proven')


def test_equal_text_supersession_needs_occurrence_witness():
    from tests.differential.merged_user_turns_extras import DECLARED
    from tests.differential.record_contract import check_stage, expected_record
    record = expected_record(DECLARED[2707800][0])
    canonical = [freeze(dict(record, store_id=record['owner_store_id'])), freeze(dict(record, store_id=999))]
    with pytest.raises(AssertionError, match='superseded without occurrence witness'):
        check_stage(dict(canonical=canonical, records=[freeze(dict(record, superseded_by=999))]),
                    dict(canonical=canonical), [DECLARED[2707800][0]])


def test_equal_text_supersession_accepts_occurrence_witness():
    from .merged_user_turns_extras import DECLARED
    from .record_contract import check_stage, expected_record
    entry = DECLARED[2707802][0]
    record = expected_record(entry)
    canonical = [freeze(dict(record, store_id=record['owner_store_id'])), freeze(dict(record, store_id=999))]
    check_stage(dict(canonical=canonical, records=[freeze(dict(record, superseded_by=999))]),
                dict(canonical=canonical, supersession_witness={record['owner_store_id']: 999}), [entry])


@pytest.mark.parametrize('mutation,message', [
    ('footer-page-2', 'withheld footer incomplete'),
    ('latch', 'latched visibility: withheld locator missing'),
])
def test_fake_withheld_pages_and_latch(mutation, message, tmp_path, versions, monkeypatch, record_property):  # noqa: F811
    from tests.differential import fake_records
    monkeypatch.setattr(paired, 'CANDIDATE_CONTRACT', 'records-v1')
    install, scan = fake_records.install, fake_records.withholding
    visible_before = []
    if mutation == 'latch':
        cached = []
        def latched(engine):
            from .session import queued_rows
            if queued_rows(engine) and not cached:
                cached.append(scan(engine))
                visible_before.append(not cached[0])
            return cached[0] if cached else []
        monkeypatch.setattr(fake_records, 'withholding', latched)
    else:
        footer_pages = []
        def modified(engine, entries, patch):
            tools = install(engine, entries, patch)
            load = tools.lcm_load_session
            def drop(args, **kwargs):
                reply = json.loads(load(args, **kwargs))
                footer = reply.get('queued_user_records_withheld')
                if footer:
                    footer_pages.append(args.get('after_store_id', 0))
                    if len(footer_pages) == 2:
                        footer['locators'] = footer['locators'][1:]
                        record_property('mutated_footer_page', json.dumps(dict(
                            case=2708002, footer_page=2, after_store_id=footer_pages[-1],
                            counts=footer['counts'], remaining=len(footer['locators']))))
                return json.dumps(reply)
            patch.setattr(tools, 'lcm_load_session', drop)
            return tools
        monkeypatch.setattr(fake_records, 'install', modified)
    with pytest.raises(AssertionError, match=message) as caught:
        try:
            _execute_case(2708002, tmp_path, versions, fake=True)
        except AssertionError as exc:
            if mutation == 'latch':
                assert visible_before == [True], 'record was not initially visible'
                raise AssertionError(f'latched visibility: {exc}') from exc
            raise
    record_property('mutation_failure', str(caught.value))
