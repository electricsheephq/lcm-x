import sqlite3
import sys
import threading
from pathlib import Path

import pytest


@pytest.mark.parametrize('threaded', [False, True])
@pytest.mark.parametrize('profiled', [False, True])
@pytest.mark.parametrize('method', ['set_authorizer', 'set_trace_callback', 'set_progress_handler', 'create_function'])
def test_raw_store_callback_is_detected_without_path_resolution(tmp_path, monkeypatch, method, threaded, profiled):
    from hermes_lcm.store import MessageStore
    from tests.message_store_callback_audit import CALLS
    store = MessageStore(tmp_path / 'audit-control.db')
    conn = store._conn
    assert type(conn) is sqlite3.Connection
    args = {'set_authorizer': (lambda *_: 0,), 'set_trace_callback': (lambda *_: None,),
            'set_progress_handler': (lambda: 0, 1), 'create_function': ('probe', 0, lambda: 0)}
    code = compile('conn.' + method + '(*args)', str(Path.cwd() / 'store.py'), 'exec')
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
        assert len(CALLS) == start + 1 and CALLS[-1][:2] == (method, True)
    finally:
        sys.setprofile(previous)
        del CALLS[start:]
        store.close()
