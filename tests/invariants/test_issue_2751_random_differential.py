"""#2751: seeded event histories, bounded by main and its fault-free run.

The fixture is generated from main 4f75302e in an isolated subprocess. Faults
only affect admission/append/drain; no cache or frontier state is patched.
"""
import copy
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from tests.invariants.test_issue_2742_warm_tail_refinal import _open, _turn
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _Probe

KINDS = ('ingest', 'restamp', 'append', 'busy_end', 'busy_once', 'end',
         'retry', 'sibling', 'busy_drain', 'cold_end')


def events_for_seed(seed):
    """Mix random walks with weighted restamp/end histories, including rollback."""
    rng = random.Random(seed)
    shape, uids = rng.choice(('user', 'turn')), rng.choice((True, False))
    events = [['ingest', shape, uids]]
    mode = rng.randrange(4)
    if mode == 0:
        # A transient second tail inside an attempt, followed by successful retry.
        kinds = ['restamp', 'busy_end', 'retry', 'restamp', 'busy_end',
                 'restamp', 'busy_drain', 'end']
    elif mode == 1:
        # A lone applied view followed by a committed two-tail chain.
        kinds = ['restamp', 'busy_end', rng.choice(('retry', 'sibling')),
                 'restamp', 'busy_end', 'restamp', 'busy_end', 'end']
    else:
        kinds = rng.choices(KINDS, weights=(1, 4, 2, 3, 2, 2, 2, 2, 3, 2), k=rng.randint(2, 8))
    roles = ['user'] if shape == 'user' else ['user', 'assistant', 'assistant', 'tool']
    for kind in kinds:
        if kind == 'restamp':
            choices = [i for i, role in enumerate(roles) if role in ('user', 'assistant')
                       and (i < 2 or i >= 4)]
            index = rng.choice(choices)
            events.append([kind, index, float(len(events) * 1000 + index + 1)])
        elif kind == 'append':
            events.append([kind])
            roles.extend(('user', 'assistant'))
        elif kind == 'busy_drain':
            events.append([kind, rng.choice((True, False)) if mode > 1 else True])
        else:
            events.append([kind])
    return events


def run_events(root, monkeypatch, events, *, faults=True):
    """Return rows before shutdown, errors, and observations at event boundaries."""
    root.mkdir()
    with monkeypatch.context() as patch:
        engine = _open(root, patch, None)
        probe = _Probe(patch)
        view, turn = [], 0
        chain, errors, boundaries, routes, pending_kinds = False, [], [], Counter(), []
        try:
            for position, event in enumerate(events):
                kind = event[0]
                before_ids = {id(tail) for tail in engine._pending_session_tails}
                before = len(before_ids)
                with patch.context() as injected:
                    try:
                        if kind == 'ingest':
                            if not view:
                                view = _turn(0, 'plain', uids=event[2])
                                if event[1] == 'user':
                                    view = view[:1]
                            engine.ingest(copy.deepcopy(view))
                        elif kind == 'restamp':
                            view[event[1]]['timestamp'] = event[2]
                        elif kind == 'append':
                            turn += 1
                            view.extend(_turn(turn, uids=events[0][2])[:2])
                        elif kind == 'retry':
                            engine._retry_pending_session_tails()
                        elif kind == 'sibling':
                            sibling = engine.__class__(config=engine._config, hermes_home=engine._hermes_home)
                            try:
                                sibling.on_session_start('parent', platform='cli', context_length=128000,
                                                         conversation_id='conv')
                            finally:
                                sibling.shutdown()
                        elif kind in ('busy_end', 'cold_end') and faults:
                            if kind == 'cold_end':
                                # Existing before-append helper also faults the read-only count read.
                                probe.busy_end(injected, engine, 'before-append', view)
                            else:
                                probe.busy_end(injected, engine, 'warm', view)
                        else:
                            append = engine._append_ingest_batch
                            hit = []

                            def busy_once(batch):
                                if not hit:
                                    hit.append(True)
                                    raise sqlite3.OperationalError('database is locked')
                                return append(batch)

                            if faults and (kind == 'busy_once' or kind == 'busy_drain' and event[1]):
                                injected.setattr(engine, '_append_ingest_batch', busy_once)
                            if faults and kind == 'busy_drain' and before:
                                first = engine._pending_session_tails[0]
                                match = engine._match_pending_tail

                                def busy_first(tail):
                                    if tail is first:
                                        raise sqlite3.OperationalError('database is locked')
                                    return match(tail)

                                injected.setattr(engine, '_match_pending_tail', busy_first)
                            engine.on_session_end('parent', copy.deepcopy(view))
                    except Exception as exc:
                        errors.append([position, type(exc).__name__, str(exc)])
                after = len(engine._pending_session_tails)
                boundaries.append(after)
                chain |= after >= 2
                pending_kinds.append([tail["kind"] for tail in engine._pending_session_tails])
                removed = before_ids - {id(tail) for tail in engine._pending_session_tails}
                if removed:
                    routes[kind] += len(removed)
            rows = sorted((r['content'], r['observed_at'])
                          for r in engine._store.get_session_messages('parent'))
            return dict(rows=rows, errors=errors, chain=chain, boundaries=boundaries, routes=dict(routes), pending_kinds=pending_kinds)
        finally:
            engine.shutdown()


def oracle(candidate, main, clean_candidate, clean_main):
    """Return every violation, counting each (content, observed_at) occurrence."""
    counts = [Counter(map(tuple, run['rows'])) for run in (candidate, main, clean_candidate, clean_main)]
    c, m, kc, km = counts
    failures = []
    if kc != km:
        failures.append('fault-free candidate differs from main')
    if any(error not in main['errors'] for error in candidate['errors']):
        failures.append('candidate-only exception')
    if candidate['chain']:
        if c != m:
            failures.append('committed chain differs from main')
    else:
        for row in c.keys() | m.keys() | km.keys():
            if not min(m[row], km[row]) <= c[row] <= max(m[row], km[row]):
                failures.append(f'single-tail row outside bounds: {row!r}')
    return failures


@pytest.mark.parametrize('seed', range(25))
def test_random_differential(tmp_path, monkeypatch, seed):
    fixture = Path(__file__).parents[1] / 'fixtures' / 'issue_2751_random_differential_main.json'
    recorded = json.loads(fixture.read_text())[str(seed)]
    events = events_for_seed(seed)
    assert events == recorded['events']
    candidate = run_events(tmp_path / 'candidate', monkeypatch, events)
    clean = run_events(tmp_path / 'clean', monkeypatch, events, faults=False)
    assert not oracle(candidate, recorded['M'], clean, recorded['K_m']), {
        'events': events, 'C': candidate, 'M': recorded['M'], 'K_m': recorded['K_m']}
