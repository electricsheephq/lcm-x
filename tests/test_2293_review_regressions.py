"""Round-3 witnesses: lossless replay and conservative hold timing (I1/I8)."""
from types import SimpleNamespace
import sqlite3

import pytest

from tests.test_2293_prelock import SESSION, _counts, _engine


def test_nonbusy_failed_begin_keeps_placeholder_cold_replay_lossless(tmp_path):
    counts = []
    for lazy in (False, True):
        root = tmp_path / str(lazy)
        root.mkdir()
        engine = _engine(root, prelock=lazy)
        engine._compiled_ignore_message_patterns = [SimpleNamespace(search=lambda text, **kw: 'IGNORE' in text)]
        base = [{'role': 'user', 'content': 'ordinary warm row', 'timestamp': 1000.0}]
        engine._ingest_messages(base)
        conn, denied = engine._store._conn, []
        def authorize(action, arg1, *_):
            if action == sqlite3.SQLITE_TRANSACTION and arg1 == 'BEGIN' and not denied and engine._store._ingest_is_prelock():
                denied.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        conn.set_authorizer(authorize)
        active = engine._ingest_messages(base + [{'role': 'assistant', 'content': 'IGNORE me', 'timestamp': 1001.0}])
        conn.set_authorizer(None)
        assert bool(denied) == lazy and 'message ignored' in str(active)
        poison = _counts(engine)
        engine._store.close()
        cold = _engine(root, prelock=lazy)
        cold._ingest_messages(active)
        counts.append(cold._store.get_session_count(SESSION))
        cold._store.close()
    assert counts == [1, 1], (counts, dict(poison))
    assert poison['poisoned'] == poison['reclassified'] == 1


@pytest.mark.parametrize('invalid', [False, True])
def test_hold_timer_survives_scope_rollback_and_invalid_implicit_begin(tmp_path, monkeypatch, invalid):
    from hermes_lcm import store as module
    store = module.MessageStore(tmp_path / 'hold.db')
    now = [10.0]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: now[0])
    with store.ingest_atomic(prelock=True) as timing:
        state = store._ingest_prelock_state
        state.clean_at = 10.0
        store._conn.execute("INSERT INTO metadata VALUES('witness', 'x')")
        now[0] = 20.0
        if invalid:
            state.poisoned = True
            with pytest.raises(module._IngestReclassify):
                store._sync_ingest_prelock()
            state.poisoned = state.reclassify_pending = False
        else:
            store._sync_ingest_prelock()
            assert store._rollback_uncovered_ingest_prelock()
            now[0] = 30.0
            store._acquire_ingest_prelock()
        now[0] = 31.0
    store.close()
    assert timing['hold_ms'] >= 21000
    assert timing.get('first_acquired_at') == 10.0


def test_swallowed_oracle_failure_still_fails_at_teardown(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = Path.cwd()
    (tmp_path / 'r3c_forced_oracle.py').write_text(
        'def pytest_configure(config):\n'
        ' from tests.prelock_instrument import PrelockRecorder\n'
        ' def fail(self): raise AssertionError("R3c forced oracle failure")\n'
        ' PrelockRecorder.check = fail\n')
    environment = dict(os.environ, PYTHONPATH=f"{root}:{tmp_path}:" + os.environ.get('PYTHONPATH', ''))
    assert environment['PYTHONPATH'].split(os.pathsep)[0] == str(root)
    targets = [f'tests/test_2293_prelock_probes.py::{name}[turns-new]' for name in (
        'test_p3_no_foreign_write_means_no_reclassify', 'test_p4_prelock_writes_stay_inside_the_attempt_transaction')]
    result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'r3c_forced_oracle', '-q', *targets,
                             f'--basetemp={tmp_path / "child"}', '-o', 'tmp_path_retention_policy=failed',
                             '-o', f'cache_dir={tmp_path / "cache"}'],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 1 and '2 errors' in result.stdout, result.stdout + result.stderr
    assert 'R3c forced oracle failure' in result.stdout


def test_engine_uses_first_acquisition_after_rollback(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import hermes_lcm.engine as module
    import hermes_lcm.store as store_module
    engine = _engine(tmp_path, prelock=True)
    now = [10.0]
    monkeypatch.setattr(store_module.time, 'perf_counter', lambda: now[0])
    monkeypatch.setattr(module, '_perf_counter', lambda: now[0])
    original = engine._store.ingest_atomic
    @contextmanager
    def witness(**kw):
        with original(**kw) as timing:
            state = engine._store._ingest_prelock_state
            state.clean_at = 10.0
            engine._store._conn.execute("INSERT INTO metadata VALUES('hold-witness', 'x')")
            engine._store._sync_ingest_prelock()
            now[0] = 20.0
            assert engine._store._rollback_uncovered_ingest_prelock()
            now[0] = 30.0
            engine._store._acquire_ingest_prelock()
            yield timing
            now[0] = 31.0
    monkeypatch.setattr(engine._store, 'ingest_atomic', witness)
    engine._ingest_messages([{'role': 'user', 'content': 'hold timing engine witness', 'timestamp': 1000.0}])
    assert engine._last_ingest_attribution['lock_hold_ms'] >= 21000
    engine._store.close()
