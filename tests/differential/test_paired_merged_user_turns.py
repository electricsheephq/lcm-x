"""I1/I2: seeded, ordered-row differential; only proven queued text may differ.

Random sequences have 1..6 ingests. Historical control seeds retain their exact
original traces (the watched-head eviction control necessarily exceeds six).
Set LCMX_DIFF_CANDIDATE_TREE to a git-archive tree for a positive control.
"""
import json
import os
import random
import sqlite3

import pytest

if os.environ.get("LCM_DIFFERENTIAL") != "1":
    pytest.skip("opt in with LCM_DIFFERENTIAL=1", allow_module_level=True)

from tests.differential import merged_user_turn_controls as base
from tests.differential import paired
from tests.differential.paired import versions  # noqa: F401
from tests.differential.session import Trace, audit_fault, freeze
from tests.phase0_hygiene import product_env_keys


RANDOM_SEEDS = tuple(range(2707400, 2707720))
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
    return paired.normalise_rows(rows, preserved, reference_rows)


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
    target = '_store_merged_user_aliases' if kind in ('preservation-once', 'persistent') else '_identity_anchor_prematch'
    present = hasattr(engine, target)
    if audit_fault(engine, kind, calls, present) or not present:
        return calls
    if kind in ('preservation-once', 'persistent'):
        original = engine._store_merged_user_aliases

        def preserve(*args, **kwargs):
            calls.append('preservation')
            if kind == 'persistent' or len(calls) == 1:
                raise ValueError('seeded preservation failure')
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, '_store_merged_user_aliases', preserve)
    elif kind in ('busy', 'evidence'):
        original = engine._identity_anchor_prematch

        def prematch(*args, **kwargs):
            calls.append(kind)
            if len(calls) == 1:
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
            stages.append(paired.normalise_rows(base._identities(engine), preserved,
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


CASES = [*RANDOM_SEEDS, *CONTROL_SEEDS, 'shifted-replay', 'capture-without-fault']


def _execute_case(case, directory, versions, stub_faults=False):  # noqa: F811
    traces, properties = {}, []

    def record(name, value):
        properties.append((name, value))

    with pytest.MonkeyPatch.context() as patch:
        for key in product_env_keys(paired.ROOT):
            patch.delenv(key, raising=False)
        original = base._open

        def opened(module, path, monkeypatch):
            key = 'main' if module is versions['main'] else 'candidate'
            trace = traces.setdefault(key, Trace(stub_faults=stub_faults))
            return original(module, path, monkeypatch, trace=trace)

        patch.setattr(base, '_open', opened)
        if case == 'shifted-replay':
            _named_shifted(directory, patch, versions, record)
        elif case == 'capture-without-fault':
            _named_capture(directory, patch, versions, record)
        elif case in CONTROL_SEEDS:
            _historical(case, directory, patch, versions, record)
        else:
            _seeded(case, directory, patch, versions, record)
    return properties, traces


def _fault_counts(trace):
    result = {}
    for fault in trace.faults:
        item = result.setdefault(fault['kind'], dict(requested=0, fired=0, inert=0))
        item['requested'] += 1
        item['fired'] += len(fault['calls'])
        item['inert'] += not fault['present']
    return result


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
            ingest_shown = any(row['role'] == 'user' and row['content'] == value
                               for view in views if view['kind'] == 'ingest' for row in view['rows'])
            if not ingest_shown:
                capture = views[violation['view']]['capture']
                assert capture in ('proven', 'watched'), (case, violation, views)
                label = '#2707 ' + capture
            else:
                ingests = [view['rows'] for view in views if view['kind'] == 'ingest']
                assert any(len(b) < len(a) for a, b in zip(ingests, ingests[1:])), (case, violation, views)
                label = '#1525'
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
                entries.append((stage, kind, value, remaining, '#2720 uid change'))
                count -= remaining
                if not count:
                    continue
            if count:
                first = next(v['stage'] for v in violations
                             if v['kind'] == 'duplicate' and v['value'] == value
                             and v['count'] > clean.get((v['stage'], 'duplicate', value), 0))
                faults = [f['kind'] for f in trace.faults
                          if f['stage'] <= first and f['calls'] and f['kind'] in ('busy', 'evidence')]
                assert faults, (case, violation, trace.faults)
                label = '#1815 fall-open, ' + faults[-1] + ' fault'
        entries.append((stage, kind, value, count, label))
    return entries


@pytest.fixture(scope='class')
def sequence(request, tmp_path_factory, versions):  # noqa: F811 - imported fixture
    case = request.param
    directory = tmp_path_factory.mktemp(f'merged-{case}')
    properties, traces = _execute_case(case, directory, versions)
    for name, trace in traces.items():
        counts = _fault_counts(trace)
        properties.append((name + '_fault_counts', json.dumps(counts)))
    stubbed = None
    if any(v['kind'] == 'duplicate' for v in traces['main'].violations()):
        # Rerun the identical host plan with all fault wrappers disabled.
        _, clean = _execute_case(case, directory / 'fault-free',
                                 dict(main=versions['main'], candidate=versions['candidate']), stub_faults=True)
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
        unfired = [(name, kind, counts) for name, value in properties if name.endswith('_fault_counts')
                   for kind, counts in json.loads(value).items()
                   if counts['requested'] > counts['inert'] and not counts['fired']]
        assert not unfired, f'{case}: requested fault targets did not fire: {unfired}'
        differences = [json.loads(value) for name, value in properties if name == 'differences']
        assert differences and not any(differences), f'{case}: {differences}'
        # The source scenarios retain their own exact substitutions. Historical
        # controls additionally check setup and warm ingests, which the source
        # compared only at selected checkpoints.
        if case in CONTROL_SEEDS or isinstance(case, str):
            stages = {name: [paired.normalise_rows(stage['rows'], stage['preserved'],
                                                   traces['main'].ingests[index]['rows'])
                             for index, stage in enumerate(trace.ingests)] for name, trace in traces.items()}
            assert not paired.compare_stages(stages, record_property), case

    def test_absolute(self, sequence, record_property):
        from collections import Counter
        from .merged_user_turns_expected import BASELINE_REF, EXPECTED

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
        actual_pin = Counter((stage, kind, freeze(value), label)
                             for stage, kind, value, count, label in classified for _ in range(count))
        pinned = Counter((stage, kind, freeze(value), label) for stage, kind, value, count, label
                         in EXPECTED.get(case, ()) for _ in range(count))
        effective = os.environ.get('LCMX_DIFF_BASELINE_REF', BASELINE_REF)
        matches = actual_pin == pinned
        record_property('baseline_pin', json.dumps(dict(ref=effective, matches=matches,
                        skipped=None if effective == BASELINE_REF else 'baseline override')))
        subset = not (counters['candidate'] - counters['main'])
        record_property('candidate_violation_subset', json.dumps(subset))
        if effective == BASELINE_REF:
            assert matches, f'{case}: baseline pin mismatch: {violations["main"]}'
        assert subset, f'{case}: added candidate violations: {counters["candidate"] - counters["main"]}'
