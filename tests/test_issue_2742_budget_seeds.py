"""Hand-written engine replays of decoded fork plans 2543384 and 2543487.

No fork code is imported or executed. Both plans run with a virtual instant
clock, and with a complete warm proof in the final drain spending 60 ms of its budget.
All original source occurrences and complete tool-call/result groups are checked.
"""
import copy
from contextlib import contextmanager

import pytest

from hermes_lcm import engine as engine_module, ingest_ledger, maintenance_scheduler
from hermes_lcm.engine import LCMEngine
from tests.invariants.test_issue_2742_warm_tail_refinal import _bind, _open, _row
from tests.invariants.test_issue_2550_busy_end_and_restamp_paths import _Probe, _turn
from tests.test_issue_2742_receipt_bridge import VirtualClock


def _turn_for_seed(seed, n, shape):
    rows = _turn(n, shape)
    for i, row in enumerate(rows):
        if row['content']:
            row['content'] = f'{seed}-parent-' + row['content']
        row['message_uid'] = f'{seed}-parent-{n}-{i}'
    return rows


def _clock(monkeypatch, expire, target_event):
    clock = VirtualClock(monkeypatch)
    match = LCMEngine._match_pending_tail
    clock.event = None
    clock.warm_pending = []
    clock.expired_admissions = []
    timeout = engine_module._temporary_sqlite_busy_timeout
    # Observe the expired admission itself, even if a concurrent writer defers it.
    # A later foreground append has a different budget and is not this assertion.
    @contextmanager
    def observe_admission(connections, milliseconds):
        if clock.advances and clock.event == target_event and not clock.expired_admissions:
            clock.expired_admissions.append(milliseconds)
        with timeout(connections, milliseconds):
            yield

    monkeypatch.setattr(engine_module, '_temporary_sqlite_busy_timeout', observe_admission)
    for name in ('_schedule_rollup_maintenance', '_schedule_embedding_maintenance'):
        monkeypatch.setattr(LCMEngine, name, lambda *args: None)

    def wrapped(engine, tail):
        result = match(engine, tail)
        if expire and not clock.advances and clock.event == target_event:
            clock.warm_pending.append(any(t['kind'] == 'warm' for t in engine._pending_session_tails))
            clock.expire()
        return result

    monkeypatch.setattr(LCMEngine, '_match_pending_tail', wrapped)
    return clock


def _end(engine, monkeypatch, probe, view, fault):
    if fault == 'none':
        engine.on_session_end('parent', copy.deepcopy(view))
    else:
        with monkeypatch.context() as patch:
            probe.busy_end(patch, engine, fault, view)


def _assert_sources(engine, original, finals, *, restamp=False):
    rows = engine._store.get_session_messages('parent')
    for message in original:
        stamp = message['timestamp']
        found = [r for r in rows if r['role'] == message['role'] and r['observed_at'] == stamp
                 and (bool(r.get('tool_calls')) if message.get('tool_calls') else r['content'] == message['content'])]
        assert len(found) == 1, (message, rows)
        if message.get('tool_calls'):
            digest = ingest_ledger.identity_digest(engine._store.ingest_ledger_salt(), message)
            receipts = engine._store._conn.execute(
                "SELECT identity_digest FROM occurrence_bindings WHERE store_id=? AND disposition='stored'",
                (found[0]['store_id'],)).fetchall()
            assert receipts == [(digest,)], (message, receipts)
    for text, stamp in finals:
        assert sum(r['content'] == text and r['observed_at'] == stamp for r in rows) == 1, rows
    if restamp:
        assert [r['observed_at'] for r in rows if r['content'] == original[0]['content']] == [1., 1001.], rows
    calls = [r for r in rows if r.get('tool_calls')]
    results = [r for r in rows if r['role'] == 'tool']
    assert len(calls) == len(results) == 1
    assert results[0]['tool_call_id'] == 'call-0'
    assert rows.index(calls[0]) + 1 == rows.index(results[0])
    assert len(rows) == len(original) + len(finals) + int(restamp), rows


@pytest.mark.parametrize('expire', [False, True], ids=['budget-intact', 'budget-expires'])
def test_budget_seed_2543384(tmp_path, monkeypatch, expire, record_property):
    engine = _open(tmp_path, monkeypatch, None)
    made = [engine]
    clock = _clock(monkeypatch, expire, 6)
    probe = _Probe(monkeypatch)
    try:
        # 0 ingest(data-uri-key), 1 sibling, 2 busy before-append(no final).
        live = _turn_for_seed(2543384, 0, 'data-uri-key')
        original = copy.deepcopy(live)
        engine.ingest(copy.deepcopy(live))
        sibling = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
        made.append(sibling)
        _bind(sibling)
        _end(engine, monkeypatch, probe, live, 'before-append')
        # 3 rebind to an alternate profile and back.
        home = engine._hermes_home
        alternate = tmp_path / 'profile-3'
        alternate.mkdir()
        _bind(engine, hermes_home=str(alternate))
        _bind(engine, hermes_home=home)
        finals = []
        # 4 busy before-append(final), 5 busy warm(final), 6 clean end(final).
        for index, fault in [(4, 'before-append'), (5, 'warm'), (6, 'none')]:
            clock.event = index
            text, stamp = f'2543384-final-{index}', float(1000 + index)
            finals.append((text, stamp))
            live.append(_row(text, stamp, 'assistant'))
            _end(engine, monkeypatch, probe, live, fault)
        # 7 drain(start), followed by each still-open owner's pending queue.
        _bind(engine)
        for owner in made:
            owner._retry_pending_session_tails()
        _assert_sources(engine, original, finals)
        assert all(not e._pending_session_tails for e in made)
        record_property("virtual_expiries", clock.advances)
        assert clock.advances == int(expire)
        if expire:
            assert clock.warm_pending == [True]
            assert clock.expired_admissions == [0]
    finally:
        for owner in reversed(made):
            owner.shutdown()


@pytest.mark.parametrize('expire', [False, True], ids=['budget-intact', 'budget-expires'])
def test_budget_seed_2543487(tmp_path, monkeypatch, expire, record_property):
    engine = _open(tmp_path, monkeypatch, None)
    made = [engine]
    clock = _clock(monkeypatch, expire, 10)
    probe = _Probe(monkeypatch)
    try:
        # 0 ingest(wrapped-base64), 1 ingest(turn1), 2 busy before-append(final).
        live = _turn_for_seed(2543487, 0, 'wrapped-base64')
        engine.ingest(copy.deepcopy(live))
        live += _turn_for_seed(2543487, 1, 'none')
        original = copy.deepcopy(live)
        engine.ingest(copy.deepcopy(live))
        finals = [('2543487-final-2', 1002.)]
        live.append(_row(*finals[0], role='assistant'))
        _end(engine, monkeypatch, probe, live, 'before-append')
        # 3 re-stamp q0, 4 shutdown/new engine, 5 exit, 6 clean end(no final).
        live[0]['timestamp'] += 1000
        engine.shutdown()
        engine = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
        made.append(engine)
        _bind(engine)
        maintenance_scheduler._flush_pending_tails_at_exit()
        _end(engine, monkeypatch, probe, live, 'none')
        # 7,8 busy before-append(finals), 9 busy warm(final), 10 drain(end).
        for index, fault in [(7, 'before-append'), (8, 'before-append'), (9, 'warm')]:
            text, stamp = f'2543487-final-{index}', float(1000 + index)
            finals.append((text, stamp))
            live.append(_row(text, stamp, 'assistant'))
            _end(engine, monkeypatch, probe, live, fault)
        clock.event = 10
        _end(engine, monkeypatch, probe, live, 'none')
        for owner in made:
            if not owner._stable_use_closed:
                owner._retry_pending_session_tails()
        _assert_sources(engine, original, finals, restamp=True)
        retained = [t for e in made for t in e._pending_session_tails]
        record_property("retained_cold_tails_before_drain", len(retained))
        assert all(t['kind'] == 'cold' for t in retained)
        # Retry the older cold view with a fresh budget, checking its rows again.
        clock.event = 11
        for owner in made:
            if not owner._stable_use_closed:
                owner._retry_pending_session_tails()
        record_property("retained_cold_tails_after_drain",
                        sum(len(owner._pending_session_tails) for owner in made))
        _assert_sources(engine, original, finals, restamp=True)
        record_property("virtual_expiries", clock.advances)
        assert clock.advances == int(expire)
        if expire:
            assert clock.warm_pending == [True]
            assert clock.expired_admissions == [0]
    finally:
        for owner in reversed(made):
            owner.shutdown()
