"""Ordering proofs for calls that must return before a blocker is released."""

import gc
import threading
import time
from contextlib import contextmanager

_frozen = []


@contextmanager
def earlier_heap_frozen():
    """Keep earlier tests' heap out of a timed section; yield a callable returning collector seconds since entry.

    A full collection walks every object the xdist worker still holds, so one landing in a timed sample cost
    about a second on CI (#2573). Freezing the heap on entry leaves later collections only the new objects.
    Entry does not collect, so it stays constant time; frozen garbage waits for the unfreeze. Time is read
    with perf_counter because some tests replace time.monotonic. Not nestable: an inner exit would unfreeze
    the outer section's heap.
    """
    if _frozen:
        raise RuntimeError("earlier_heap_frozen is already active")
    spent, started = [0.0], [None]

    def timer(phase, _info):
        if phase == "start":
            started[0] = time.perf_counter()
        elif started[0] is not None:
            spent[0] += time.perf_counter() - started[0]
            started[0] = None

    gc.freeze()
    _frozen.append(True)
    gc.callbacks.append(timer)
    try:
        yield lambda: spent[0]
    finally:
        gc.callbacks.remove(timer)
        gc.unfreeze()
        _frozen.clear()


@contextmanager
def deadline_block(monkeypatch, worker_name, budget, latency):
    """Hold a worker until its caller returns; fast cases expire a logical clock.

    Record the real join argument before simulating expiry. Other stages may
    finish with a generous hang guard so scheduling speed cannot fail PR CI.
    Latency cases use the real clock and the production join unchanged.
    """
    entered, release = threading.Event(), threading.Event()
    workers = []
    original_join = threading.Thread.join
    now = [0.0]
    with monkeypatch.context() as patch:
        if not latency:
            patch.setattr(time, "monotonic", lambda: now[0])

        def block():
            entered.set()
            assert release.wait(10), "deadline worker was never released"

        def join(worker, timeout=None):
            if worker.name == worker_name:
                workers.append(worker)
                assert timeout is not None and 0 < timeout <= budget, "join must use the remaining request budget"
                assert entered.wait(10), "deadline worker never reached its blocker"
                if not latency:
                    now[0] += budget + 1
                    return
            elif not latency and worker.name.startswith("lcm-"):
                return original_join(worker, timeout=10)
            return original_join(worker, timeout=timeout)

        patch.setattr(threading.Thread, "join", join)
        try:
            yield block
            assert workers, "the deadline join was not exercised"
        finally:
            release.set()
            for worker in workers:
                original_join(worker, timeout=10)
                assert not worker.is_alive(), "deadline worker did not exit after release"


def returns_while_blocked(call, release):
    results, errors = [], []

    def run():
        try:
            results.append(call())
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=run, daemon=True)
    caller.start()
    try:
        caller.join(timeout=10)
        assert not caller.is_alive(), "call waited for the blocker to be released"
        assert not release.is_set(), "blocker was released before the call returned"
        if errors:
            raise errors[0]
        return results[0]
    finally:
        release.set()
        caller.join(timeout=10)


class ContentionProbe:
    """Expose an actual failed lock acquisition instead of sleeping for a race."""

    def __init__(self, lock):
        self.lock = lock
        self.contended = threading.Event()
        self.attempts = []
        self.acquired = threading.Event()
        self.release_guard = None
        self.returned_before_release = False

    def acquire(self, blocking=True, timeout=-1):
        if self.lock.acquire(blocking=False):
            return True
        self.attempts.append((blocking, timeout))
        self.contended.set()
        if not blocking:
            return False
        acquired = self.lock.acquire(timeout=timeout)
        if self.release_guard is not None and not self.release_guard.is_set():
            self.returned_before_release = True
        self.acquired.set()
        return acquired

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()
