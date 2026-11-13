import sqlite3
import sys
import threading
from pathlib import Path

import pytest


@pytest.mark.parametrize('trace', [False, True])
@pytest.mark.parametrize('threaded', [False, True])
@pytest.mark.parametrize('profiled', [False, True])
@pytest.mark.parametrize('method', ['set_authorizer', 'set_trace_callback', 'set_progress_handler', 'create_function'])
def test_raw_store_callback_is_detected_without_path_resolution(tmp_path, monkeypatch, method, threaded, profiled, trace):
    from hermes_lcm.store import MessageStore
    from tests.message_store_callback_audit import CALLS, ROOT
    from tests.behaviour.ingest_trace_harness import TraceConnection
    import hermes_lcm.sqlite_session as session
    connect = sqlite3.connect
    if trace:
        monkeypatch.setattr(session.sqlite3, 'connect', lambda *a, **k: connect(*a, **dict(k, factory=TraceConnection)))
    store = MessageStore(tmp_path / 'audit-control.db')
    conn = store._conn
    assert type(conn) is (TraceConnection if trace else sqlite3.Connection)
    args = {'set_authorizer': (lambda *_: 0,), 'set_trace_callback': (lambda *_: None,),
            'set_progress_handler': (lambda: 0, 1), 'create_function': ('probe', 0, lambda: 0)}
    code = compile('conn.' + method + '(*args)', str(ROOT / 'store.py'), 'exec')
    start = len(CALLS)
    previous = sys.getprofile()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'resolve', lambda *_a, **_k: pytest.fail('audit resolved a path'))
            def register():
                if profiled:
                    sys.setprofile(lambda *_: None)
                exec(code, {'conn': conn, 'args': args[method]})
            if threaded:
                thread = threading.Thread(target=register)
                thread.start()
                thread.join(10)
                assert not thread.is_alive()
            else:
                register()
        assert CALLS[start:] == [(method, True, 'store.py', 1)]
    finally:
        sys.setprofile(previous)
        del CALLS[start:]
        store.close()


def test_audit_accepts_and_restores_an_active_cprofile(tmp_path):
    import subprocess
    code = """
import cProfile
import sys
from pathlib import Path
import pytest
import tests.conftest
from tests.message_store_callback_audit import install
from hermes_lcm.store import MessageStore
profiler = cProfile.Profile()
profiler.enable()
prior = sys.getprofile()
assert prior is None or (prior is profiler and not callable(profiler))
try:
    with pytest.MonkeyPatch.context() as patch:
        restore = install(patch)
        try:
            store = MessageStore(Path(sys.argv[1]))
            store.close()
        finally:
            restore()
        assert sys.getprofile() is prior
finally:
    profiler.disable()
"""
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'cprofile.db')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
