"""D1: audit all MessageStore connection factories for the entire pytest suite."""
import inspect
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALLS = []


class CallbackAudit:
    _message_store_connection = False

    def _record_callback(self, method, callback):
        if not self._message_store_connection:
            return
        # The immediate caller outside this mixin is the registering code. A test
        # wrapper may call product code: walking every ancestor would exempt it.
        frame = inspect.currentframe().f_back.f_back
        path = Path(frame.f_code.co_filename).resolve()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            return
        if relative.parts[0] == "tests":
            return
        CALLS.append((method, callback is not None, str(relative), frame.f_lineno))

    def set_authorizer(self, authorizer, /):
        self._record_callback("set_authorizer", authorizer)
        return super().set_authorizer(authorizer)

    def set_trace_callback(self, callback, /):
        self._record_callback("set_trace_callback", callback)
        return super().set_trace_callback(callback)

    def set_progress_handler(self, callback, n, /):
        self._record_callback("set_progress_handler", callback)
        return super().set_progress_handler(callback, n)

    def create_function(self, name, narg, func, **kwargs):
        self._record_callback("create_function", func)
        return super().create_function(name, narg, func, **kwargs)


class AuditedConnection(CallbackAudit, sqlite3.Connection):
    pass


def install(monkeypatch):
    from dataclasses import replace
    from hermes_lcm import store
    connect, open_connection = sqlite3.connect, store.open_connection
    factories = {}

    def audited_connect(*args, **kwargs):
        factory = kwargs.get("factory", sqlite3.Connection)
        if isinstance(factory, type) and issubclass(factory, sqlite3.Connection):
            if not issubclass(factory, CallbackAudit):
                if factory not in factories:
                    factories[factory] = type("Audited" + factory.__name__, (CallbackAudit, factory), {})
                kwargs["factory"] = factories[factory]
        return connect(*args, **kwargs)

    def opened(policy):
        on_open = policy.on_open

        def mark(conn):
            conn._message_store_connection = True
            if on_open:
                on_open(conn)
        return open_connection(replace(policy, on_open=mark))

    monkeypatch.setattr(sqlite3, "connect", audited_connect)
    monkeypatch.setattr(store, "open_connection", opened)
