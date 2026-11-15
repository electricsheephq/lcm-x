"""D1: observe SQLite callback calls without replacing any connection type."""
import sys
import threading
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
CALLS = []
GAPS = []  # names of running threads the audit could not observe (Python 3.11 only)
METHODS = frozenset({'set_authorizer', 'set_trace_callback', 'set_progress_handler', 'create_function',
                     'create_aggregate', 'create_collation', 'create_window_function'})


def install(monkeypatch):
    from hermes_lcm import store
    open_connection = store.open_connection
    connections = {}  # strong references prevent connection-id reuse during this test
    setprofile, set_thread_profile = sys.setprofile, threading.setprofile
    all_threads = getattr(threading, 'setprofile_all_threads', None)  # Python 3.12+
    prior, thread_prior = sys.getprofile(), threading.getprofile()
    active = [True]

    def observed(frame, event, arg):
        if event == 'call' and frame.f_code.co_name in METHODS:
            method = frame.f_code.co_name
            if id(frame.f_locals.get('self')) not in connections:
                return
            caller = frame.f_back
        elif event == 'c_call' and getattr(arg, '__name__', '') in METHODS:
            method = arg.__name__
            connection = getattr(arg, '__self__', None)
            if id(connection) not in connections:
                return
            # A Python connection override was attributed at entry. Its eventual
            # C registration must not hide that caller or count it twice.
            parent = frame
            while parent is not None:
                if parent.f_code.co_name == method and parent.f_locals.get('self') is connection:
                    return
                parent = parent.f_back
            caller = frame
        else:
            return
        filename = caller.f_code.co_filename
        prefix = str(ROOT) + '/'
        if filename.startswith(prefix) and not filename.startswith(prefix + 'tests/'):
            CALLS.append((method, True, filename[len(prefix):], caller.f_lineno))

    def restore_profile(previous, setter):
        if previous is None or callable(previous):
            setter(previous)
        else:
            # cProfile's hook is an opaque Profile, not a Python callback.
            previous.enable()

    def compose(previous):
        if getattr(previous, '_d1_observer', False):
            return previous
        def profile(frame, event, arg):
            if not active[0]:
                restore_profile(previous, setprofile)
            else:
                observed(frame, event, arg)
            if callable(previous):
                previous(frame, event, arg)
        profile._d1_observer = True
        return profile

    def opened(policy):
        on_open = policy.on_open
        def mark(conn):
            first = not connections
            connections[id(conn)] = conn
            current, thread_hook = sys.getprofile(), threading.getprofile()
            if first and all_threads is not None:
                # Any running thread may register on this connection, not only the opener.
                # Another thread's own profiler cannot be read, so it is replaced, not chained.
                all_threads(compose(None))
            elif first:
                # Python 3.11 can profile only the calling thread and threads started later:
                # record the running threads this audit cannot see instead of passing silently.
                GAPS.extend(t.name for t in threading.enumerate() if t is not threading.current_thread())
            # Also cover connections opened by threads already alive at setup.
            if not getattr(current, '_d1_observer', False):
                setprofile(compose(current))
            set_thread_profile(compose(thread_hook))
            if on_open:
                on_open(conn)
        return open_connection(replace(policy, on_open=mark))

    monkeypatch.setattr(store, 'open_connection', opened)
    # Tests without a MessageStore need no profiler. Start at open, before on_open
    # can register a callback, including when that open happens on an existing thread.
    monkeypatch.setattr(sys, 'setprofile', lambda callback: setprofile(compose(callback) if connections else callback))
    monkeypatch.setattr(threading, 'setprofile', lambda callback: set_thread_profile(compose(callback) if connections else callback))
    def restore():
        active[0] = False
        restore_profile(prior, setprofile)
        restore_profile(thread_prior, set_thread_profile)
        connections.clear()
    return restore
