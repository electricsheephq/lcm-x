"""I1/I2/I3/I6: session-end canonical acceptance, pinned against main.

No preservation substitutions, records, product seam, or changed historical pins.
"""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

if os.environ.get('LCM_DIFFERENTIAL') != '1':
    pytest.skip('opt in with LCM_DIFFERENTIAL=1', allow_module_level=True)

from . import session_end_contract
from .paired import ROOT, _load
from .session import Trace
from .session_end_contract import (FULL_OUTPUT, Clock, Proof, _call, _history, _state_db, _write_output,
                                   fault, floor, judge, slow_seam, snapshot)
from .session_end_expected import BASELINE, CELLS, CONTROL_LOSSES, FAULTS, INTENDED_DIFFERENCES
from .trees import materialize, resolve

MODES = ('instant', 'busy-timeout', 'slow-cold')
CELL_CASES = ('wrapped-base64-cold-pairwise', 'wrapped-base64-cold-counted',
         'data-uri-key-cold-pairwise', 'data-uri-key-cold-counted',
         'changed-spillover-direct', 'changed-spillover-busy',
         'changed-hermes-results-direct', 'changed-hermes-results-busy')
assert set(CELLS) <= set(CELL_CASES)
CASES = (*CELL_CASES, 'plain-direct', 'plain-busy', 'append', 'commit', 'repeated-retry',
         'rollback', 'interrupt', 'bypass', 'warm-busy', 'shutdown-deadline', 'missing-payload',
         'restart-between-commits', 'detached', 'newer', 'ledger-off', 'before-ledger', 'failed-ledger-write',
         '2543384-before', '2543384-between', '2543384-after', '2543487-before', '2543487-between', '2543487-after')


@pytest.fixture(scope='module')
def end_versions(tmp_path_factory):
    if os.environ.get('LCMX_DIFF_SESSION_END_BASELINE') is not None and any(k in os.environ for k in ('CI', 'GITHUB_ACTIONS')):
        pytest.fail('LCMX_DIFF_SESSION_END_BASELINE is forbidden in CI')
    baseline = os.environ.get('LCMX_DIFF_SESSION_END_BASELINE', BASELINE)
    return {'main': _load(materialize(resolve(baseline), tmp_path_factory.mktemp('end-main')), '_end_main'),
            'candidate': _load(Path(os.environ.get('LCMX_DIFF_CANDIDATE_TREE', ROOT)), '_end_candidate')}


@pytest.fixture(scope='module')
def receipts(request):
    data = {'traces': [], 'routes': {}, 'stages': {}, 'differences': set()}
    yield data
    # A partial invocation cannot advertise that it exercised the mandatory floor.
    selected = {item.callspec.params['case'] for item in request.session.items
                if item.originalname == 'test_session_end' and 'case' in item.callspec.params}
    assert selected == set(CASES), 'floor skipped: partial session-end case set'
    assert len(data['traces']) == len(CASES) * len(MODES), 'floor skipped: incomplete primary set'
    floor(data['traces'])
    assert data['differences'] == set(INTENDED_DIFFERENCES), 'stale intended difference: unmatched entry'
    output = os.environ.get('LCM_DIFFERENTIAL_OUT')
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'g-routes.json').write_text(json.dumps(data['routes'], indent=2) + '\n')
        (directory / 'session-end-stages.json').write_text(json.dumps(data['stages'], indent=2) + '\n')


class Run:
    def __init__(self, module, root, patch, mode):
        self.root, self.patch, self.mode = root, patch, mode
        self.package = module.__name__
        self.clock, self.proof = Clock(mode), Proof()
        self.engines, self.paths, self.stages = [], set(), {}
        self.trace = Trace()
        self.check_proof = True
        self.clock.install(self.package, patch)
        self.mod = importlib.import_module(self.package + '.engine')
        self.config = importlib.import_module(self.package + '.config').LCMConfig
        self.scheduler = importlib.import_module(self.package + '.maintenance_scheduler')
        self.scheduler._PENDING_TAIL_ENGINES.clear()
        self.engine = self.open(root)

    def open(self, home):
        home.mkdir(parents=True, exist_ok=True)
        _state_db(home, [('S1', None, None), ('S2', None, None), ('S3', None, None)])
        config = self.config(database_path='', embeddings_enabled=False, temporal_rollups_enabled=False,
            large_output_externalization_path=str(self.root / 'payloads'))
        engine = self.mod.LCMEngine(config=config, hermes_home=str(home))
        self.engines.append(engine)
        self.patch.setattr(engine, '_schedule_rollup_maintenance', lambda *a, **k: None)
        self.patch.setattr(engine, '_schedule_embedding_maintenance', lambda *a, **k: None)
        engine._paired_trace = self.trace
        self.proof.install(engine, self.patch)
        self.paths.add(Path(engine._store.db_path))
        self.start(engine, 'S1')
        retry = engine._retry_pending_session_tails
        def observed_retry(*args, **kwargs):
            result = retry(*args, **kwargs)
            if hasattr(self, 'engine'):
                self.snap('retry-return')
            return result
        self.patch.setattr(engine, '_retry_pending_session_tails', observed_retry)
        return engine

    @staticmethod
    def start(engine, sid):
        engine.on_session_start(sid, platform='cli', context_length=200000, conversation_id='conv-' + sid)

    def snap(self, stage, special=None):
        if stage in self.stages:
            stage += '-' + str(sum(key.startswith(stage) for key in self.stages))
        self.stages[stage] = snapshot(self.engines, self.root, self.paths)
        if self.check_proof:
            self.proof.check(self.engine, stage, self.paths, special)

    def retry(self, stage, *, deadline=None):
        with self.patch.context() as patch:
            original = self.engine._retry_cold_tail
            def cold(*args, **kwargs):
                self.clock.cold()
                return original(*args, **kwargs)
            patch.setattr(self.engine, '_retry_cold_tail', cold)
            self.engine._retry_pending_session_tails(deadline=deadline)
        self.snap(stage)

    def end(self, view, kind=None, *, current=False):
        engine = self.engine
        target = {'append': (engine, '_append_off_current_session_end_suffix'),
                  'commit': (engine._store._conn, 'commit'),
                  'interrupt': (engine, '_ingest_messages_transaction'),
                  'rollback': (engine, '_append_ingest_batch'),
                  'warm-busy': (engine, '_append_ingest_batch'),
                  'bypass': (engine, '_append_off_current_session_end_suffix'),
                  'restart-between-commits': (engine, '_append_off_current_session_end_suffix')}
        with self.patch.context() as patch:
            if kind:
                owner, name = target.get(kind, (engine._store, 'get_range'))
                error = KeyboardInterrupt if kind in ('interrupt', 'restart-between-commits') else sqlite3.OperationalError
                if kind in ('rollback', 'commit'):
                    conn = engine._store._conn
                    class Proxy:
                        def __getattr__(self, key):
                            return getattr(conn, key)
                        def rollback(self):
                            conn.rollback()
                            if kind == 'rollback':
                                raise RuntimeError('rollback sentinel')
                        def commit(self):
                            return conn.commit()
                        def __enter__(self):
                            return self
                        def __exit__(self, typ, *_):
                            return self.rollback() if typ else self.commit()
                    proxy = Proxy()
                    patch.setattr(engine._store, '_conn', proxy)
                    if kind == 'commit':
                        owner = proxy
                audit = fault(engine, owner, name, kind, self.clock, patch, error=error,
                      after=kind in ('rollback', 'warm-busy', 'restart-between-commits'))
            try:
                engine.on_session_end('S1', deepcopy(view))
            except (sqlite3.Error, KeyboardInterrupt):
                assert kind in ('bypass', 'restart-between-commits')
        if kind:
            assert audit.present and audit.called and audit.raised, f'declared fault {kind} never entered/raised'
        self.snap('end-of-on_session_end')

    def close(self):
        for engine in reversed(self.engines):
            if not engine._stable_use_closed:
                engine.shutdown()
        self.scheduler._PENDING_TAIL_ENGINES.clear()


def scenario(run, case):
    engine, clock = run.engine, run.clock
    shape = case.rsplit('-', 2)[0]
    initial = _history('ok 1')
    if case.startswith(('wrapped-base64-', 'data-uri-key-')):
        shape = 'wrapped-base64' if case.startswith('wrapped-base64') else 'data-uri-key'
        initial[3] = dict(_call(shape), timestamp=4.)
    elif case.startswith('changed-'):
        location = 'spillover' if 'spillover' in case else 'hermes-results'
        path = run.root / ('cache/spillover/m1.txt' if location == 'spillover' else 'tmp/hermes-results/m1.txt')
        # Real payload stat evidence is deterministic in both isolated trees.
        protection = importlib.import_module(run.package + '.ingest_protection')
        recover = protection.recover_hermes_persisted_output_with_file_stat
        def stable(*args, **kwargs):
            found = recover(*args, **kwargs)
            return (found[0], dict(found[1], mtime_ns=1700001000000000000, ctime_ns=1700001000000000000)) if found else found
        run.patch.setattr(protection, 'recover_hermes_persisted_output_with_file_stat', stable)
        run.patch.setattr(protection.tempfile, 'gettempdir', lambda: str(run.root / 'tmp'))
        initial[4]['content'] = _write_output(path)
    if case == 'missing-payload':
        initial[4]['content'] = FULL_OUTPUT
        engine._config.large_output_externalization_threshold_chars = 200
    special = case if case in ('ledger-off', 'before-ledger', 'failed-ledger-write') else None
    run.proof.route = special
    with run.patch.context() as patch:
        if special in ('ledger-off', 'before-ledger'):
            patch.setenv('LCM_INGEST_LEDGER', 'off')
        elif special:
            ledger = importlib.import_module(run.package + '.ingest_ledger')
            fault(engine, ledger, 'insert_bindings', special, clock, patch, error=sqlite3.IntegrityError)
        engine.ingest(deepcopy(initial))
    run.proof.route = None
    assert len(engine._store.get_session_messages('S1')) == 10, 'seed must store every host row'
    run.snap('seed', special)
    if special:
        if case == 'before-ledger':
            run.patch.setenv('LCM_INGEST_LEDGER', 'shadow')
            run.start(engine, 'S1')
        engine.ingest([*deepcopy(initial), {'role': 'user', 'content': 'next canonical ingest', 'timestamp': 20.}])
        run.snap('next-ingest', special)
        return
    view = [*deepcopy(initial), {'role': 'assistant', 'content': 'FINAL', 'timestamp': 11.}]
    if case.startswith('changed-'):
        view[4]['content'] = _write_output(path, FULL_OUTPUT + 'NEW RESULT')
    current = case in ('rollback', 'interrupt', 'warm-busy')
    if not current:
        run.start(engine, 'S2')
    if case == 'bypass':
        engine._mark_lcm_bypass_lineage_session('S1', platform='cron')
    busy = (case != 'warm-busy' and case.endswith(('cold-pairwise', 'cold-counted', '-busy'))) or case in (
        'plain-busy', 'repeated-retry', 'shutdown-deadline', 'detached', 'newer') or case.startswith('2543')
    kind = 'busy-prefix' if busy else case if case in (*FAULTS, 'warm-busy') else None
    if case == 'missing-payload':
        for path in (run.root / 'payloads').rglob('*.json'):
            path.rename(path.with_suffix('.missing'))
        with run.patch.context() as patch:
            prefix = importlib.import_module(run.package + '.prefix_matching')
            fault(engine, prefix, '_marker_payload_metadata', case, clock, patch, error=FileNotFoundError)
            run.end(view)
    else:
        run.end(view, kind, current=current)
    run.snap('busy-refusal' if busy else 'direct-end')
    if case == 'restart-between-commits':
        # Crash boundary: S3's protected-batch COMMIT (prefix_matching.py:437)
        # has completed; lifecycle.finalize_session's COMMIT (engine.py:5366)
        # has not run. Close handles without shutdown/retry, then rebind.
        for helper in (engine._store, engine._dag, engine._lifecycle):
            helper.close()
        engine._stable_use_closed = True
        run.engine = engine = run.open(run.root)
        run.snap('restart-between-S3-and-lifecycle-commits')
        run.start(engine, 'S2')
    if case in CONTROL_LOSSES:
        assert len(engine._store.get_session_messages('S1')) == 10, CONTROL_LOSSES[case]
        assert not engine._pending_session_tails, CONTROL_LOSSES[case]
    if case.startswith('2543'):
        # Seeds 2543384/2543487: first mismatches, second is an ordinary S3 extension.
        engine._pending_session_tails[0]['view'][0]['content'] = 'edited prefix'
        run.start(engine, 'S3')
        engine.ingest(deepcopy(initial))
        run.start(engine, 'S2')
        with run.patch.context() as patch:
            fault(engine, engine._store, 'get_range', 'busy-prefix', clock, patch)
            engine.on_session_end('S3', deepcopy(view))
        assert len(engine._pending_session_tails) == 2
        budget = case.rsplit('-', 1)[1]
        deadline = clock.now + (-1 if budget == 'before' else 5.)
        original = engine._retry_cold_tail
        with run.patch.context() as patch:
            def step(*args, **kwargs):
                if budget == 'before':
                    # Main starts the first tail even with an expired deadline.
                    clock.now = deadline + 1
                result = original(*args, **kwargs)
                if budget == 'between':
                    clock.now = deadline + 1
                elif budget == 'after':
                    clock.now += 3.
                return result
            patch.setattr(engine, '_retry_cold_tail', step)
            engine._retry_pending_session_tails(deadline=deadline)
        run.snap('two-tail-' + budget)
        rows = engine._store.get_session_messages('S3')
        assert len(rows) == (11 if budget == 'after' else 10), (case, len(rows))
    if case == 'repeated-retry':
        for index in range(2):
            with run.patch.context() as patch:
                fault(engine, engine, '_retry_cold_tail', case, clock, patch)
                run.retry('failed-retry-' + str(index))
    if case == 'detached':
        engine._rebind_storage_for_home(str(run.root / 'other'))
        run.paths.add(Path(engine._store.db_path))
        assert engine._pending_session_tails and not engine._cold_tail_is_bound(engine._pending_session_tails[0])
        run.retry('detached-home-retry')
    if case == 'newer':
        # A newer same-session continuation must not reorder an older tail.
        with run.patch.context() as patch:
            fault(engine, engine, '_retry_cold_tail', 'repeated-retry', clock, patch)
            run.start(engine, 'S1')
            engine.ingest([*deepcopy(initial), {'role': 'user', 'content': 'newer', 'timestamp': 30.}])
        assert engine._pending_session_tails[0].get('went_on')
        run.snap('newer-continuation')
        run.retry('went-on-retry')
    if case == 'shutdown-deadline':
        with run.patch.context() as patch:
            fault(engine, engine, '_retry_cold_tail', case, clock, patch)
            attempt = engine._retry_cold_tail
            def expired(*args, **kwargs):
                clock.now = kwargs['deadline'] + 1.
                return attempt(*args, **kwargs)
            patch.setattr(engine, '_retry_cold_tail', expired)
            close = engine._store.close
            patch.setattr(engine._store, 'close', lambda: (run.snap('shutdown-deadline-before-close'), close())[-1])
            engine.shutdown()
        run.snap('shutdown-deadline-after-close')
        run.engine = engine = run.open(run.root)
    run.retry('retry-1')
    run.retry('retry-2')
    if case in CELL_CASES:
        rows = engine._store.get_session_messages('S1')
        if case in CELLS:
            if run.check_proof:  # Pin main's loss; judge compares the candidate at every stage.
                assert len(rows) == 10 and not any(r['content'] == 'FINAL' for r in rows), CELLS[case]
                if case.endswith('cold-counted'):
                    assert engine._cold_tail_absent_count(engine._pending_session_tails[0]) == 2, CELLS[case]
        else:
            assert sum(r['content'] == 'FINAL' for r in rows) == 1, f'{case}: FINAL must be stored exactly once'
    run.end(view)
    run.snap('repeated-end')
    # Next ordinary canonical ingest after every injected fault.
    run.start(engine, 'S2')
    engine.ingest([{'role': 'user', 'content': 'next canonical ingest', 'timestamp': 40.}])
    run.snap('next-ingest')
    run.scheduler._flush_pending_tails_at_exit()
    run.snap('at-exit-flush')
    close = engine._store.close
    with run.patch.context() as patch:
        def closing():
            run.snap('shutdown-before-close')
            close()
        patch.setattr(engine._store, 'close', closing)
        engine.shutdown()
    run.snap('shutdown-after-close')
    run.engine = run.open(run.root)
    run.engine.ingest(deepcopy(initial))
    run.snap('restart-rebind')


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('case', CASES)
def test_session_end(tmp_path, monkeypatch, end_versions, receipts, case, mode):
    results, traces = {}, {}
    for tree in ('main', 'candidate'):
        with monkeypatch.context() as patch:
            if case == 'before-ledger':
                patch.setenv('LCM_INGEST_LEDGER', 'off')
            run = Run(end_versions[tree], tmp_path / 'live', patch, mode)
            run.check_proof = tree == 'main'
            if tree == 'candidate' and os.environ.get('LCMX_DIFF_SLOW_SEAM'):
                slow_seam(os.environ['LCMX_DIFF_SLOW_SEAM'], run.clock, patch)
            try:
                scenario(run, case)
                results[tree], traces[tree] = run.stages, run.trace
                if tree == 'main':
                    receipts['routes'][case + '/' + mode] = run.proof.stages
            finally:
                run.close()
        (tmp_path / 'live').rename(tmp_path / ('saved-' + tree))
    assert results['main'].keys() == results['candidate'].keys(), 'stage set changed'
    for stage in results['main']:
        judge(results['main'][stage], results['candidate'][stage], stage,
              case=case, mode=mode, seen=receipts['differences'])
    receipts['traces'].append(traces)
    receipts['stages'][case + '/' + mode] = list(results['main'])


FAKES = {'content': ('direct-end', 'canonical'), 'drop': ('retry-1', 'canonical'),
         'duplicate': ('repeated-end', 'canonical'), 'pending-key': ('busy-refusal', 'pending'),
         'ledger-counter': ('direct-end', 'ledger'), 'lifecycle-row': ('direct-end', 'tables'),
         'payload-create': ('direct-end', 'payload'), 'payload-replace': ('direct-end', 'payload')}


def mutate(run, kind):
    engine, conn = run.engine, run.engine._store._conn
    if kind == 'content':
        conn.execute("UPDATE messages SET content='fake corruption' WHERE store_id=1")
    elif kind == 'drop':
        conn.execute('DELETE FROM messages WHERE store_id=(SELECT max(store_id) FROM messages)')
    elif kind == 'duplicate':
        columns = [r[1] for r in conn.execute('PRAGMA table_info(messages)') if r[1] != 'store_id']
        cols = ','.join(columns)
        conn.execute(f'INSERT INTO messages({cols}) SELECT {cols} FROM messages WHERE store_id=1')
    elif kind == 'pending-key':
        engine._pending_session_tails[0]['went_on'] = True  # Read at engine.py:12264.
    elif kind == 'ledger-counter':
        engine._ledger_count('ledger.tail.retry.deferred')
    elif kind == 'lifecycle-row':
        conn.execute('UPDATE lcm_lifecycle_state SET current_frontier_store_id=current_frontier_store_id+1')
    else:
        folder = run.root / 'payloads'
        folder.mkdir(exist_ok=True)
        path = next(folder.glob('*.json'), folder / 'fake.json') if kind == 'payload-replace' else folder / 'fake.json'
        path.write_text('{"content":"fake replacement"}')
    conn.commit()


@pytest.mark.parametrize('kind', ('unchanged', *FAKES))
def test_fake_candidates(tmp_path, monkeypatch, end_versions, kind, record_property):
    snapshots = {}
    stage, reason = FAKES.get(kind, ('direct-end', None))
    for tree in ('main', 'candidate'):
        with monkeypatch.context() as patch:
            run = Run(end_versions[tree], tmp_path / 'live', patch, 'instant')
            run.check_proof = tree == 'main'
            try:
                # The replacement fake starts with a real externalized payload.
                if kind == 'payload-replace':
                    engine = run.engine
                    engine._config.large_output_externalization_threshold_chars = 200
                    engine.ingest([{'role': 'tool', 'content': FULL_OUTPUT, 'tool_call_id': 'p'}])
                original = run.snap
                def snap(name, *args):
                    if tree == 'candidate' and name == stage and kind != 'unchanged':
                        mutate(run, kind)
                    original(name, *args)
                patch.setattr(run, 'snap', snap)
                # Stop just after the observed fake, so later ingest cannot mask it.
                run.engine.ingest(deepcopy(_history('ok')))
                run.start(run.engine, 'S2')
                view = [*_history('ok'), {'role': 'assistant', 'content': 'FINAL', 'timestamp': 11.}]
                run.end(view, 'busy-prefix' if kind in ('drop', 'pending-key') else None)
                run.snap('busy-refusal' if kind in ('drop', 'pending-key') else 'direct-end')
                if kind == 'drop':
                    run.retry(stage)
                elif kind == 'duplicate':
                    run.end(view)
                    run.snap(stage)
                snapshots[tree] = run.stages[stage]
            finally:
                run.close()
        (tmp_path / 'live').rename(tmp_path / ('saved-' + tree))
    if reason:
        with pytest.raises(AssertionError, match=reason + ' changed') as caught:
            judge(snapshots['main'], snapshots['candidate'], stage)
        record_property('failure', str(caught.value))
    else:
        judge(snapshots['main'], snapshots['candidate'], stage)


@pytest.mark.parametrize('key', ('canonical', 'tables'))
@pytest.mark.parametrize('kind, reason', (
    ('exempt-drop', 'lost'), ('exempt-duplicate', 'doubled'),
    ('exempt-new-once', None), ('exempt-new-twice', 'doubled'),
    ('exempt-reorder', 'reordered'), ('exempt-new-before', None)))
def test_fake_candidates_intended(monkeypatch, key, kind, reason, record_property):
    columns = ('store_id', 'session_id', 'role', 'content')
    rows = [(1, 'S1', 'user', 'A'), (2, 'S1', 'assistant', 'B'), (3, 'S1', 'user', 'C')]
    changed = rows[:-1] if kind == 'exempt-drop' else [*rows]
    extra = rows[0][1:] if kind == 'exempt-duplicate' else ('S1', 'assistant', 'FINAL')
    if kind != 'exempt-drop':
        changed.append((4, *extra))
    if kind == 'exempt-new-twice':
        changed.append((5, *extra))
    if kind == 'exempt-reorder':
        changed = [(i, *row[1:]) for i, row in enumerate(reversed(rows), 1)]
    elif kind == 'exempt-new-before':
        changed = [*rows, (0, *extra)]  # Input order differs from store_id order.
    changed = [(sid + 10, *values) for sid, *values in changed]  # store_id is exempt.
    main = {name: {} for name in ('canonical', 'schema', 'tables', 'payload', 'pending', 'ledger', 'lifecycle')}
    def observed(items):
        messages = (columns, items)
        return {'lcm.db': {'messages': messages} if key == 'tables' else messages}
    main[key] = observed(rows)
    candidate = dict(main, **{key: observed(changed)})
    monkeypatch.setattr(session_end_contract, 'INTENDED_DIFFERENCES',
                        [('cell', '*', 'retry-1', key, '#2543')])
    if reason:
        with pytest.raises(AssertionError, match=f'{key} {reason} stored row') as caught:
            judge(main, candidate, 'retry-1', case='cell', mode='instant')
        record_property('failure', str(caught.value))
    else:
        judge(main, candidate, 'retry-1', case='cell', mode='instant')


@pytest.mark.parametrize('table, remaining, reason', (
    ('message_relations', 0, 'tables lost rows in message_relations'),
    ('summaries', 1, 'tables lost rows in summaries'),
    ('ingest_views', 0, None), ('summaries', 3, None),
    ('messages_fts_data', 0, None), ('foo_data', 0, 'tables lost rows in foo_data'),
    ('lineage_marks', 1, None),
    ('notes_data', 0, 'tables lost rows in notes_data'),
    ('foo_data', 0, 'tables lost rows in foo_data'),
    ('v_content', 0, 'tables lost rows in v_content'),
    ('v_content', 0, None),
    ('v_content', 0, 'tables lost rows in v_content'),
    ('v_content', 0, None),
    ('notes2_data', 0, 'tables lost rows in notes2_data')),
    ids=('relations-drop-all', 'summaries-drop-one', 'views-prune', 'summaries-add',
         'fts-shadow-merge', 'lookalike-data-drop', 'lineage-merge', 'text-mention-lookalike',
         'nonfts-vt-lookalike', 'external-content-lookalike', 'own-content-shadow', 'contentless-lookalike',
         'own-content-column', 'embedded-ddl-lookalike'))
def test_fake_candidates_intended_tables(monkeypatch, table, remaining, reason, record_property, request):
    main = {name: {} for name in ('canonical', 'schema', 'tables', 'payload', 'pending', 'ledger', 'lifecycle')}
    schemas = {
        'text-mention-lookalike': ('notes', "CREATE TABLE notes (x TEXT DEFAULT 'USING fts5')"),
        'nonfts-vt-lookalike': ('foo', 'CREATE VIRTUAL TABLE foo USING rtree(id, x0, x1)'),
        'external-content-lookalike': ('v', "CREATE VIRTUAL TABLE v USING fts5(body, content='messages')"),
        'own-content-shadow': ('v', 'CREATE VIRTUAL TABLE v USING fts5(body)'),
        'contentless-lookalike': ('v', "CREATE VIRTUAL TABLE v USING fts5(body, content='')"),
        'own-content-column': ('v', 'CREATE VIRTUAL TABLE v USING fts5(content)'),
        'embedded-ddl-lookalike': ('notes2', "CREATE TABLE notes2 (x TEXT DEFAULT 'CREATE VIRTUAL TABLE x USING fts5')"),
    }
    if table in ('messages_fts_data', 'foo_data', 'lineage_marks') or request.node.callspec.id in schemas:
        name, sql = schemas.get(request.node.callspec.id,
                                ('messages_fts', 'CREATE VIRTUAL TABLE messages_fts USING fts5(content)'))
        main['schema'] = {'lcm.db': [('table', name, name, sql)]}
    rows = [(1, 'A'), (2, 'B'), (3, 'C')]
    main['tables'] = {'lcm.db': {table: (('id', 'text'), rows[:2])}}
    candidate = dict(main, tables={'lcm.db': {table: (('id', 'text'), rows[:remaining])}})
    monkeypatch.setattr(session_end_contract, 'INTENDED_DIFFERENCES',
                        [('cell', '*', 'retry-1', 'tables', '#2543')])
    if reason:
        with pytest.raises(AssertionError, match=reason) as caught:
            judge(main, candidate, 'retry-1', case='cell', mode='instant')
        record_property('failure', str(caught.value))
    else:
        judge(main, candidate, 'retry-1', case='cell', mode='instant')


def test_intended_tables_fts_segment_merge(tmp_path, monkeypatch, end_versions):
    package = end_versions['candidate'].__name__
    Clock().install(package, monkeypatch)
    store_type = importlib.import_module(package + '.store').MessageStore
    states = []
    for count in (15, 16):
        store = store_type(tmp_path / str(count) / 'lcm.db')
        try:
            for index in range(count):
                store.append('S1', {'role': 'user', 'content': f'message {index}', 'timestamp': index + 1.})
            schema, tables = session_end_contract.database(store.db_path)
        finally:
            store.close()
        state = {key: {} for key in ('canonical', 'schema', 'tables', 'payload', 'pending', 'ledger', 'lifecycle')}
        state.update(canonical={'lcm.db': tables['messages']}, schema={'lcm.db': schema}, tables={'lcm.db': tables})
        states.append(state)
    monkeypatch.setattr(session_end_contract, 'INTENDED_DIFFERENCES',
                        [('cell', '*', 'retry-1', key, '#2543') for key in ('canonical', 'tables')])
    assert len(states[1]['tables']['lcm.db']['messages_fts_data'][1]) < len(states[0]['tables']['lcm.db']['messages_fts_data'][1])
    judge(*states, 'retry-1', case='cell', mode='instant')


def _seam_probe(engine, canonical=False):
    if canonical:
        engine._store._conn.execute("UPDATE messages SET content='slow canonical work' WHERE store_id=1")
        engine._store._conn.commit()


@pytest.mark.parametrize('canonical', (False, True), ids=('slow-noop-passes', 'slow-canonical-fails'))
def test_slow_seam(tmp_path, monkeypatch, end_versions, canonical):
    snapshots = {}
    for tree in ('main', 'candidate'):
        with monkeypatch.context() as patch:
            run = Run(end_versions[tree], tmp_path / 'live', patch, 'instant')
            try:
                run.engine.ingest(deepcopy(_history('ok')))
                # A log-only seam after the retry's decision must remain harmless
                # even when it expires a still-active caller deadline.
                deadline = run.clock.now + 5.
                if tree == 'candidate':
                    slow_seam(__name__ + '._seam_probe', run.clock, patch)
                _seam_probe(run.engine, canonical=canonical and tree == 'candidate')
                assert tree == 'main' or run.clock.now > deadline
                snapshots[tree] = snapshot(run.engines, run.root, run.paths)
            finally:
                run.close()
        (tmp_path / 'live').rename(tmp_path / ('saved-' + tree))
    if canonical:
        with pytest.raises(AssertionError, match='canonical changed'):
            judge(snapshots['main'], snapshots['candidate'], 'slow-seam')
    else:
        judge(snapshots['main'], snapshots['candidate'], 'slow-seam')


@pytest.mark.parametrize('kind', ('never-entered', 'floor-skipped', 'candidate-inert'))
def test_fault_floor_controls(kind):
    from .session import FaultAudit, assert_faults
    if kind == 'floor-skipped':
        with pytest.raises(AssertionError, match='floor skipped'):
            floor([])
    else:
        pair = {tree: Trace(faults=[FaultAudit('busy-prefix', True, 1)]) for tree in ('main', 'candidate')}
        if kind == 'candidate-inert':
            pair['main'].faults[0].called = pair['main'].faults[0].raised = 1
            with pytest.raises(AssertionError, match='inert on candidate'):
                assert_faults('fake', pair)
        else:
            with pytest.raises(AssertionError, match='never entered/raised'):
                floor([pair], expected=('busy-prefix',))
