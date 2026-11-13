"""I1/I2: every reachable pre-admission SQL boundary serializes like eager.

Discover boundaries by executing the real scenario, rather than selecting helper
names or a preferred reclassify raise. A foreign writer must actually be able to
commit there. Boundaries inside an existing transaction cannot admit a foreign
commit. The corpus's held-writer faults likewise have no reachable commit window.
"""
import ast
import hashlib
import inspect
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_lcm.sqlite_util import TransactionOwner
from hermes_lcm.store import MessageStore
from tests.behaviour import ingest_trace_harness as harness

KEY = 'test-2293-class-foreign'


def reclassify_inventory():
    """Derive the raise sites and generation comparisons from product ASTs."""
    points = []
    for filename in ('store.py', 'sqlite_util.py'):
        tree = ast.parse((harness.ROOT / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and (
                    isinstance(node.exc.func, ast.Name) and node.exc.func.id == '_IngestReclassify'):
                points.append(f'{filename}:{node.lineno}:raise')
            if isinstance(node, ast.Compare) and any(
                    isinstance(child, ast.Attribute) and child.attr == '_ingest_write_generation'
                    for child in ast.walk(node)):
                points.append(f'{filename}:{node.lineno}:generation')
    return sorted(points)


def _foreign(db):
    other = sqlite3.connect(db, timeout=0)
    try:
        other.execute('INSERT INTO metadata(key, value) VALUES (?, ?)', (KEY, 'committed'))
        other.commit()
    finally:
        other.close()


def _writable(db):
    other = sqlite3.connect(db, timeout=0, isolation_level=None)
    try:
        other.execute('BEGIN IMMEDIATE')
        other.rollback()
        return True
    except sqlite3.OperationalError as error:
        assert error.sqlite_errorcode & 255 == sqlite3.SQLITE_BUSY
        return False
    finally:
        other.close()


def _run(case, root, *, lazy, target=None):
    payloads, points, fired, classifications = [], [], [], []
    with pytest.MonkeyPatch.context() as patch:
        import hermes_lcm.store as store_module
        patch.setattr(store_module.secrets, 'token_bytes', lambda n=32: bytes(range(n)))
        patch.setattr(MessageStore, '_ingest_prelock', lazy)
        fault = harness.sqlite_fault
        if case.name == 'recovery-stamp-rewrite':
            from copy import deepcopy
            from tests.invariants.test_issue_1505_recovery_stamp import lineage, message, recover

            def recovery_prepare(engine, case, root, engines):
                lineage(root)
                engine.on_session_start('parent', conversation_id='trace-conversation', context_length=128000)
                recovered, current = recover(engine, text="[T32] rebuild " + "x" * 4096)
                recovered[-1]['timestamp'] = 30.
                incoming = deepcopy(recovered)
                ingest = engine._ingest_messages
                calls = []

                def rewrite_on_followup(messages, *args, **kwargs):
                    calls.append(True)
                    if len(calls) == 2:
                        recovered[-1]['content'] = current['content']
                        messages[:] = [*recovered, message('reply', 31., 'assistant')]
                    return ingest(messages, *args, **kwargs)

                patch.setattr(engine, '_ingest_messages', rewrite_on_followup)
                return engine, incoming

            patch.setattr(harness, 'prepare', recovery_prepare)

        @contextmanager
        def around(engine, fault_name, events):
            with fault(engine, fault_name, events), pytest.MonkeyPatch.context() as instrument:
                store, conn = engine._store, engine._store._conn
                execute = harness.TraceConnection.execute
                executemany = harness.TraceConnection.executemany
                seam = MessageStore._ingest_prelock_before_begin
                ordinal = 0

                def boundary(connection, sql):
                    nonlocal ordinal
                    if connection is not conn or fired:
                        return
                    state = TransactionOwner.ingest_state_for(conn)
                    if state is None or state.phase != 'prelock' or conn.in_transaction:
                        return
                    # This is a physical concurrency precondition, not a scenario
                    # exclusion: no foreign commit can land under a held writer.
                    if not _writable(store.db_path):
                        return
                    frame = inspect.currentframe().f_back.f_back
                    point = dict(ordinal=ordinal, file=str(Path(frame.f_code.co_filename).relative_to(harness.ROOT)),
                                 line=frame.f_lineno, sql=sql)
                    points.append(point)
                    if target == ordinal:
                        _foreign(store.db_path)
                        fired.append(point)
                    ordinal += 1

                def observed(connection, sql, parameters=(), /):
                    boundary(connection, sql)
                    return execute(connection, sql, parameters)

                def observed_many(connection, sql, parameters, /):
                    boundary(connection, sql)
                    return executemany(connection, sql, parameters)

                def eager_begin(self):
                    if self is store and not fired:
                        _foreign(store.db_path)
                        fired.append('before BEGIN IMMEDIATE')
                    return seam(self)

                if lazy:
                    instrument.setattr(harness.TraceConnection, 'execute', observed)
                    instrument.setattr(harness.TraceConnection, 'executemany', observed_many)
                elif target is not None:
                    instrument.setattr(MessageStore, '_ingest_prelock_before_begin', eager_begin)
                before = store.ingest_prelock_counts['reclassified']
                yield
                reclassified = store.ingest_prelock_counts['reclassified'] - before
                if target is not None:
                    assert fired, (case.name, target, points)
                    classifications.append(reclassified)

        patch.setattr(harness, 'sqlite_fault', around)
        harness.observe(case, root, payloads=payloads)
    result = {}
    for label in ('state-after', 'search-after', 'cold-state'):
        value = json.loads(next(event[len(label) + 1:] for event in payloads if event.startswith(label + ' ')))
        if label != 'search-after':
            value['tables']['metadata'] = [row for row in value['tables']['metadata'] if row[0] != KEY]
        result[label] = value['tables'] if label == 'state-after' else value
    result['cold-state-sha256'] = hashlib.sha256(harness.encoded(result['cold-state']).encode()).hexdigest()
    return result, points, classifications


def _diff(eager, lazy, path=''):
    if type(eager) is not type(lazy):
        return dict(path=path, eager=eager, lazy=lazy)
    if isinstance(eager, dict):
        for key in sorted(eager.keys() | lazy.keys()):
            if key not in eager or key not in lazy:
                return dict(path=path + '/' + key, eager=eager.get(key), lazy=lazy.get(key))
            if eager[key] != lazy[key]:
                return _diff(eager[key], lazy[key], path + '/' + key)
    elif eager != lazy:
        return dict(path=path, eager=eager, lazy=lazy)
    return None


def _evidence(value):
    destination = os.environ.get('LCM_2293_CLASS_EVIDENCE')
    if destination:
        with Path(destination).open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True) + '\n')


@pytest.mark.parametrize('case', (*harness.SCENARIOS, harness.Scenario('recovery-stamp-rewrite', 'I1/I2')),
                         ids=lambda case: case.name)
def test_every_reclassify_window(case, tmp_path):
    _evidence(dict(scenario=case.name, inventory=reclassify_inventory()))
    _, points, _ = _run(case, tmp_path / 'discover', lazy=True)
    _evidence(dict(scenario=case.name, discovered=points))
    if not points:
        return  # no reachable reclassify: no admission, or foreign writer excluded by a held lock
    eager, _, _ = _run(case, tmp_path / 'eager', lazy=False, target=0)
    if case.name == 'recovery-stamp-rewrite':
        assert len(eager['cold-state']['tables']['messages']) == 5
    for point in points:
        lazy, observed, classifications = _run(case, tmp_path / f'lazy-{point["ordinal"]}', lazy=True, target=point['ordinal'])
        assert observed[-1] == point, (case.name, 'boundary discovery changed', point, observed[-1])
        difference = _diff(eager, lazy)
        row_diff = {label: {table: dict(eager=eager[label]['tables'][table], lazy=lazy[label]['tables'].get(table))
                           for table in eager[label]['tables']
                           if eager[label]['tables'][table] != lazy[label]['tables'].get(table)}
                    for label in ('cold-state',)} if difference else {}
        _evidence(dict(scenario=case.name, point=point, reclassifications=classifications,
                       difference=difference, row_diff=row_diff))
        assert difference is None, (case.name, point, difference)
