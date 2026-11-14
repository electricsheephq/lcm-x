"""I1/I2: seeded, ordered-row differential; only proven queued text may differ.

Random sequences have 1..6 ingests. Historical control seeds retain their exact
original traces (the watched-head eviction control necessarily exceeds six).
Set LCMX_DIFF_CANDIDATE_TREE to a git-archive tree for a positive control.
"""
import json
import random
import sqlite3

import pytest

from tests.invariants import merged_user_turn_controls as base
from tests.differential import paired
from tests.differential.paired import versions  # noqa: F401
from tests.differential.session import AbsoluteViolation, Trace
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


def _normalise(rows, preserved):
    return paired.normalise_rows(rows, preserved)


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
                ('user', 'R2', 50., 'parent', 'conv')})] if capture else [raw[name]]
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
    if kind in ('preservation-once', 'persistent'):
        if name != 'candidate' or not hasattr(engine, '_store_merged_user_aliases'):
            return calls
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
            if not calls:
                calls.append(kind)
                if kind == 'busy':
                    raise sqlite3.OperationalError('database is locked')
                raise RuntimeError('seeded evidence fault')
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, '_identity_anchor_prematch', prematch)
    return calls


def _run_sequence(module, name, directory, monkeypatch, seed, plan):
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
            stages.append(_normalise(base._identities(engine), preserved))
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
                                              monkeypatch, seed, plan)
        record_property(name + '_faults', json.dumps(faults))
    record_property('host_views', json.dumps(inputs))
    record_property('stored_rows', json.dumps(raw))
    assert inputs['candidate'] == inputs['main'], 'Differential host inputs must be identical'
    base._compare_round2(results, record_property)


CASES = [*RANDOM_SEEDS, *CONTROL_SEEDS, 'shifted-replay', 'capture-without-fault']
# Exactly the pinned baseline's 191 absolute failures: 180 host-text losses
# and 53 duplicate-row cases (42 overlap). Both engines are still checked;
# only the candidate absolute assertion carries this strict expectation.
KNOWN_ABSOLUTE_FAILURES = frozenset({
    2707401, 2707402, 2707403, 2707404, 2707405, 2707407, 2707408, 2707409, 2707410,
    2707413, 2707416, 2707417, 2707418, 2707419, 2707420, 2707422, 2707424, 2707426,
    2707429, 2707431, 2707432, 2707434, 2707435, 2707438, 2707439, 2707441, 2707443,
    2707444, 2707447, 2707451, 2707452, 2707453, 2707454, 2707456, 2707460, 2707461,
    2707464, 2707465, 2707470, 2707471, 2707474, 2707476, 2707477, 2707479, 2707480,
    2707481, 2707482, 2707484, 2707485, 2707486, 2707487, 2707488, 2707490, 2707491,
    2707493, 2707494, 2707495, 2707496, 2707497, 2707499, 2707500, 2707503, 2707505,
    2707506, 2707507, 2707508, 2707510, 2707513, 2707516, 2707518, 2707520, 2707522,
    2707523, 2707524, 2707525, 2707526, 2707527, 2707528, 2707529, 2707531, 2707535,
    2707536, 2707539, 2707541, 2707542, 2707543, 2707544, 2707549, 2707551, 2707553,
    2707554, 2707557, 2707558, 2707559, 2707561, 2707562, 2707564, 2707565, 2707566,
    2707567, 2707569, 2707573, 2707574, 2707576, 2707577, 2707580, 2707581, 2707582,
    2707583, 2707584, 2707587, 2707589, 2707590, 2707591, 2707593, 2707594, 2707597,
    2707601, 2707604, 2707605, 2707606, 2707608, 2707609, 2707610, 2707611, 2707613,
    2707616, 2707617, 2707618, 2707622, 2707623, 2707625, 2707626, 2707627, 2707628,
    2707631, 2707632, 2707633, 2707634, 2707639, 2707640, 2707641, 2707642, 2707643,
    2707646, 2707647, 2707650, 2707651, 2707658, 2707659, 2707660, 2707661, 2707662,
    2707665, 2707669, 2707670, 2707671, 2707672, 2707673, 2707676, 2707681, 2707682,
    2707685, 2707686, 2707687, 2707691, 2707692, 2707694, 2707695, 2707697, 2707699,
    2707700, 2707701, 2707702, 2707703, 2707704, 2707705, 2707708, 2707709, 2707710,
    2707711, 2707712, 2707715, 2707716, 2707718, 2707719, 2707101, 2707102, 2707103,
    2707201, 2707203,
})


@pytest.fixture(scope='class')
def sequence(request, tmp_path_factory, versions):  # noqa: F811 - imported fixture
    case = request.param
    directory = tmp_path_factory.mktemp(f'merged-{case}')
    traces, properties = {}, []

    def record(name, value):
        properties.append((name, value))

    with pytest.MonkeyPatch.context() as patch:
        # Class fixtures run before the repository's per-test environment reset.
        for key in product_env_keys(paired.ROOT):
            patch.delenv(key, raising=False)
        original = base._open

        def opened(module, path, monkeypatch):
            key = 'main' if module is versions['main'] else 'candidate'
            trace = traces.setdefault(key, Trace())
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
    return case, properties, traces


@pytest.mark.parametrize('sequence', CASES, indirect=True, scope='class')
class TestMergedUserTurns:
    def test_differential(self, sequence, record_property):
        case, properties, traces = sequence
        for name, value in properties:
            record_property(name, value)
        hosts = {name: [stage['host'] for stage in trace.ingests] for name, trace in traces.items()}
        record_property('all_host_views', json.dumps(hosts))
        record_property('all_stored_rows', json.dumps({
            name: [stage['rows'] for stage in trace.ingests] for name, trace in traces.items()}))
        assert hosts['main'] == hosts['candidate'], 'Differential host inputs must be identical'
        differences = [json.loads(value) for name, value in properties if name == 'differences']
        assert differences and not any(differences), f'{case}: {differences}'
        # The source scenarios retain their own exact substitutions. Historical
        # controls additionally check setup and warm ingests, which the source
        # compared only at selected checkpoints.
        if case in CONTROL_SEEDS or isinstance(case, str):
            stages = {name: [paired.normalise_rows(stage['rows'], stage['preserved'])
                             for stage in trace.ingests] for name, trace in traces.items()}
            assert not paired.compare_stages(stages, record_property), case

    def test_absolute(self, sequence, request, record_property):
        case, properties, traces = sequence
        if case in KNOWN_ABSOLUTE_FAILURES:
            request.node.add_marker(pytest.mark.xfail(
                strict=True, raises=AbsoluteViolation,
                reason='#2707: pinned main loses captured host user text or repeats a stored row'))
        violations = {name: trace.violations() for name, trace in traces.items()}
        record_property('absolute_violations', json.dumps(violations))
        if violations['candidate']:
            raise AbsoluteViolation(f'{case}: {violations}')
