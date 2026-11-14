"""Isolated engine/session helpers and an absolute host-input row oracle.

Two ingest-shown rows stored as one joined row are not an absolute violation:
the content is kept. The differential half judges row structure against main;
#2707's design review decides that structure.

A loss can hide behind another view when same-text messages never share a
view; only the differential half can catch it. A host text may also match
across adjacent stored rows: content is kept, and the differential half judges
row structure. The store normalises blank source to 'unknown'; the absolute
loss check does not compare conversation_id. Neither can hide lost content
(the differential stored identity still includes conversation_id).
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import sys
from types import SimpleNamespace


def _row(text, stamp, role='user'):
    return {'role': role, 'content': text, 'timestamp': stamp}


def _rows(engine):
    return engine._store.get_session_messages('parent')


def freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def _bindings(engine):
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='host_uid_bindings' AND type='table'").fetchone():
        return []
    return list(conn.execute('SELECT store_id, uid, kind, lineage_key, binding_version, proof_kind, '
                             'first_check, disagree_seen FROM host_uid_bindings ORDER BY store_id, uid, kind'))


def _identities(engine):
    bindings = defaultdict(list)
    for sid, uid, kind, lineage, *_ in _bindings(engine):
        bindings[sid].append((uid, kind, lineage))
    senders = {sid: (sender, source) for sid, sender, source in engine._store._conn.execute(
        'SELECT store_id, sender_id, sender_source FROM messages')}
    ignored = {'store_id', 'timestamp', 'ingested_at', 'token_estimate'}
    return [freeze(dict({k: v for k, v in r.items() if k not in ignored},
                        bindings=tuple(sorted(bindings[r['store_id']])),
                        sender_id=senders[r['store_id']][0], sender_source=senders[r['store_id']][1])) for r in _rows(engine)]


def duplicate_rows(rows, views=()):
    """Full stored identity, with excess above the largest host-view count."""
    result = []
    for identity, count in Counter(rows).items():
        fields = dict(identity)
        allowed = max([1, *[sum(row['role'] == fields['role']
                                and row['content'] == fields['content']
                                and row.get('timestamp') == fields['observed_at']
                                for row in view['rows']) for view in views]])
        if count > allowed:
            result.append((identity, count - allowed))
    return sorted(result, key=repr)


def missing_messages(views, rows):
    """Consume whole segments independently in every snapshotted host view."""
    stream = [segment for row in rows if dict(row)['role'] == 'user'
              for segment in dict(row)['content'].split('\n\n')]
    missing = Counter()
    for index, view in enumerate(views):
        consumed = set()
        users = [row for row in view['rows'] if row['role'] == 'user']
        users.sort(key=lambda row: -int('\n\n' in row['content']))
        for row in users:
            parts = row['content'].split('\n\n')
            match = next((i for i in range(len(stream) - len(parts) + 1)
                          if stream[i:i + len(parts)] == parts
                          and not consumed.intersection(range(i, i + len(parts)))), None)
            if match is None:
                missing[(index, row['content'])] += 1
            else:
                consumed.update(range(match, match + len(parts)))
    return missing


@dataclass
class FaultAudit:
    kind: str
    present: bool
    stage: int
    stubbed: bool = False
    called: int = 0
    raised: int = 0


def audit_fault(engine, kind, present):
    """One registration per requested wrapper, including absent targets.

    Wrappers count condition-held entries separately from their actual raises.
    Unreached targets pass per case; a full primary run must reach every
    present kind and raise at least once. Fault-free reruns are excluded.
    """
    trace = getattr(engine, '_paired_trace', None)
    audit = FaultAudit(kind, present, len(trace.ingests) + 1 if trace else 0,
                       bool(trace and trace.stub_faults))
    if trace is not None:
        trace.faults.append(audit)
    return audit


def fault_counts(faults):
    result = {}
    for fault in faults:
        if fault.stubbed:
            continue
        item = result.setdefault(fault.kind, dict(requested=0, inert=0, unreached=0, called=0, raised=0))
        item['requested'] += 1
        item['inert'] += not fault.present
        item['unreached'] += fault.present and not fault.called
        item['called'] += fault.called
        item['raised'] += fault.raised
    return result


def assert_faults(case, traces):
    for tree, trace in traces.items():
        for fault in trace.faults:
            if not fault.stubbed and fault.present and fault.called and not fault.raised:
                raise AssertionError(f'{case}: {tree} {fault.kind} stage {fault.stage}: '
                                     'fault called but never raised')


@dataclass
class FaultRun:
    # Key by case rather than fixture order; randomly may recreate a case fixture.
    cases: dict = field(default_factory=dict)

    def record(self, case, traces):
        self.cases[case] = {tree: fault_counts(trace.faults) for tree, trace in traces.items()}

    def finish(self, expected, selected):
        totals = {}
        for trees in self.cases.values():
            for tree, kinds in trees.items():
                for kind, counts in kinds.items():
                    item = totals.setdefault(tree, {}).setdefault(kind, Counter())
                    item.update(counts)
        reason = None
        if set(selected) != set(expected):
            reason = f'partial case set selected ({len(set(selected))}/{len(set(expected))}); floor skipped'
        elif set(self.cases) != set(expected):
            reason = f'primary case set incomplete ({len(self.cases)}/{len(set(expected))}); floor skipped'
        errors = [] if reason else [f'{tree}: {kind} present but never raised across full case set'
                                   for tree, kinds in totals.items() for kind, counts in kinds.items()
                                   if counts['requested'] > counts['inert'] and not counts['raised']]
        return dict(evaluated=reason is None, reason=reason, errors=errors, totals=totals)


def _capture(ctx, history, earlier='P2', newest='R2', head_stamp=40., event_stamp=50.):
    head, tail = _row(earlier, head_stamp), _row(newest, event_stamp)
    ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message=newest,
                              conversation_history=[*history, head, tail])
    head.update(content=newest, timestamp=event_stamp)
    return [*history, head, _row('B', event_stamp + 1, 'assistant')]


@dataclass
class Trace:
    """Snapshot host messages before product mutation, across reader recreation."""
    views: list = field(default_factory=list)
    ingests: list = field(default_factory=list)
    host_objects: list = field(default_factory=list)
    captures: list = field(default_factory=list)
    preserved: dict = field(default_factory=dict)
    faults: list = field(default_factory=list)
    stub_faults: bool = False
    capture_kind: str = 'proven'

    def observe(self, messages, ingest=False, user_message=None):
        rows = deepcopy(messages)
        if isinstance(user_message, str) and not any(row['role'] == 'user' and row['content'] == user_message
                                                    for row in rows):
            rows.append(_row(user_message, None))
        self.views.append(dict(kind='ingest' if ingest else 'hook', rows=rows,
                               capture=None if ingest else self.capture_kind))

    def capture(self, messages, user_message=None):
        self.observe(messages, user_message=user_message)
        trailing = []
        for row in reversed(messages):
            if row['role'] != 'user':
                break
            trailing.insert(0, row)
        if len(trailing) < 2 or any(trailing[0] is old for old in self.host_objects):
            return
        head, newest = trailing[0], trailing[-1]
        self.captures.append((head, '\n\n'.join(row['content'] for row in trailing),
                              head['timestamp'], newest['content'], newest['timestamp']))

    def stored(self, engine, messages, originals):
        self.host_objects.extend(originals)
        for head, joined, stamp, newest, event_stamp in self.captures:
            if (any(head is row for row in originals) and head['content'] == newest
                    and head['timestamp'] == event_stamp):
                self.preserved[('user', joined, stamp, 'parent', 'conv')] = (
                    'user', newest, event_stamp, 'parent', 'conv')
        self.ingests.append(dict(host=deepcopy(messages), views=deepcopy(self.views),
                                 rows=_identities(engine), preserved=dict(self.preserved),
                                 binding_diagnostics=_bindings(engine)))

    def violations(self):
        result = []
        for index, stage in enumerate(self.ingests, 1):
            for (view, text), count in sorted(missing_messages(stage['views'], stage['rows']).items()):
                result.append(dict(stage=index, kind='missing', value=text, view=view, count=count))
            for identity, count in duplicate_rows(stage['rows'], stage['views']):
                result.append(dict(stage=index, kind='duplicate', value=identity, count=count))
        return result


def _open(module, directory, monkeypatch, trace=None):
    monkeypatch.setenv('HERMES_HOME', str(directory / 'home'))
    monkeypatch.setenv('LCM_DATABASE_PATH', str(directory / 'lcm.db'))
    monkeypatch.setenv('LCM_EMBEDDINGS_ENABLED', '0')
    monkeypatch.setenv('LCM_BACKGROUND_LEAF_PREP_ENABLED', '0')
    ctx = SimpleNamespace(hooks={})
    ctx.register_context_engine = lambda engine: setattr(ctx, 'engine', engine)
    ctx.register_hook = lambda name, callback: ctx.hooks.__setitem__(name, callback)
    module.register(ctx)
    engine = ctx.engine
    monkeypatch.setattr(sys.modules[engine.__module__], 'summarize_with_escalation',
                        lambda **_kwargs: ('Earlier turns.', 1))
    engine.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
    if trace is not None:
        engine._paired_trace = trace
        hook = ctx.hooks['pre_llm_call']

        def capture(*args, **kwargs):
            trace.capture(kwargs.get('conversation_history', []), kwargs.get('user_message'))
            return hook(*args, **kwargs)

        ingest = engine.ingest

        def observe(messages, *args, **kwargs):
            host = deepcopy(messages)
            trace.observe(host, ingest=True)
            result = ingest(messages, *args, **kwargs)
            trace.stored(engine, host, messages)
            return result

        ctx.hooks['pre_llm_call'] = capture
        monkeypatch.setattr(engine, 'ingest', observe)
    return ctx
