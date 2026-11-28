"""Session-end tails: crash before commit, after commit, and process-death resume (I1/I2)."""

from contextlib import contextmanager
import os
import sqlite3
import subprocess
import sys

import pytest

from hermes_lcm.engine import LCMEngine
from tests.invariants.test_p1e_decided_tails import contents, history, queue, replay
from tests.invariants.test_p1e_cold_tails import queue_cold
from tests.test_issue_2055_ingest_ledger_writes import count, engines as engines
from tests.test_host_uid_shadow import _m
from tests.p1_gate import expect_from_view, record, store_outcome


class Crash(BaseException):
    pass


class CrashOnRemove(list):
    def remove(self, tail):
        raise Crash("after COMMIT, before removal")


# Cover every between/route/stamp axis without a Cartesian gauntlet.
CELLS = [
    ("none", "next", True), ("partial", "next", False), ("full+new", "next", True),
    ("foreign", "off-current", False), ("other-session", "sibling", True), ("none", "shutdown", False),
    ("partial", "off-current", True), ("full+new", "sibling", False), ("foreign", "shutdown", True),
    ("other-session", "next", False),
]


@pytest.mark.parametrize("between,route,stamped", CELLS)
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
def test_decided_crash_cell(engines, monkeypatch, between, route, stamped, point):
    engine = engines()
    view = history(stamped)
    queue(engine, view)
    if between == "partial":
        replay(engine, view[2:3])
    elif between == "full+new":
        replay(engine, [*view[2:], _m("user", "new", 8. if stamped else None)])
    elif between in {"foreign", "other-session"}:
        replay(engine, view[2:], session="other" if between == "other-session" else "S",
               conversation="foreign")
    atomic = engine._store.ingest_atomic

    @contextmanager
    def interrupted(**kwargs):
        with atomic(**kwargs):
            yield
            raise Crash("before COMMIT")

    expected = ["A", "B", "C", "D"] + (["new"] if between == "full+new" else [])
    # A full replay completes by read-only pre-trim, so only the after-removal crash applies there.
    with monkeypatch.context() as patch:
        if point == "before_commit":
            patch.setattr(engine._store, "ingest_atomic", interrupted)
        else:
            patch.setattr(engine, "_pending_session_tails", CrashOnRemove(engine._pending_session_tails))
        if between == "full+new" and point == "before_commit":
            engine._retry_pending_session_tails()
        else:
            with pytest.raises(Crash):
                engine._retry_pending_session_tails()
    # A before-COMMIT crash leaves only the independently committed prefix. The pending tail survives.
    if point == "before_commit" and between != "full+new":
        own = [r["content"] for r in engine._store.get_session_messages("S") if r["conversation_id"] == "conv"]
        assert own == ["A", "B"] + (["C"] if between == "partial" else [])
    if route == "next":
        engine.on_session_start("next", platform="cli")
    elif route == "off-current":
        engine._session_id = "next"
        engine._retry_pending_session_tails()
    elif route == "sibling":
        sibling = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
        try:
            sibling.on_session_start("next", platform="cli")
        finally:
            sibling.shutdown()
    else:
        engine._retry_pending_session_tails(exiting=True)
    own = [r["content"] for r in engine._store.get_session_messages("S") if r["conversation_id"] == "conv"]
    assert own == expected  # Every extra row fails.
    assert not engine._pending_session_tails
    _gate_tail(engine, view, between, route, stamped, point, cold=False)


SUBPROCESS = r'''
from contextlib import contextmanager
import os, pathlib, sys
import pytest
import tests.conftest  # Register the checkout as hermes_lcm, as pytest does in the parent.
from hermes_lcm.store import MessageStore
from tests.harness import pin_char_counter
from tests.test_host_uid_shadow import _engine, _state_db
from tests.invariants.test_p1e_decided_tails import history, queue, replay
root, point, stamped = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "True"
patch = pytest.MonkeyPatch()
pin_char_counter(patch)
_state_db(root, [("S", None, None)])
store = MessageStore(root / "lcm.db")
store.ensure_ingest_ledger()
store.close()
engine = _engine(root)
queue(engine, history(stamped))
replay(engine, history(stamped)[2:3])
atomic = engine._store.ingest_atomic
@contextmanager
def interrupted(**kwargs):
    with atomic(**kwargs):
        yield
        os._exit(91)
class DieOnRemove(list):
    def remove(self, tail):
        os._exit(92)
if point == "before_commit":
    engine._store.ingest_atomic = interrupted
else:
    engine._pending_session_tails = DieOnRemove(engine._pending_session_tails)
engine._retry_pending_session_tails()
os._exit(0)
'''


@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
@pytest.mark.parametrize("stamped", [True, False])
def test_decided_subprocess_crash_resume(engines, tmp_path, point, stamped):
    root = tmp_path / "child"
    root.mkdir()
    env = {**os.environ, "LCM_INGEST_LEDGER": "shadow", "LCM_HOST_MESSAGE_UID": "off"}
    result = subprocess.run([sys.executable, "-c", SUBPROCESS, str(root), point, str(stamped)],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (91 if point == "before_commit" else 92), result.stderr
    # Fresh engine, same database, real host-view resume; the in-memory tail died with the child.
    from tests.test_host_uid_shadow import _engine
    fresh = _engine(root)
    try:
        before = contents(fresh)
        assert before == (["A", "B", "C"] if point == "before_commit" else ["A", "B", "C", "D"])
        fresh.ingest(history(stamped))
        assert contents(fresh) == ["A", "B", "C", "D"]
        _gate_tail(fresh, history(stamped), "none", "subprocess", stamped, point, cold=False)
    finally:
        fresh.shutdown()


def test_m9_window_is_counted(engines, caplog):
    engine = engines()
    queue(engine)
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine._retry_pending_session_tails(exiting=True)
        assert contents(engine) == ["A", "B"]
        assert count(engine, "tail.busy_at_end") == 1
        assert count(engine, "tail.retry.deferred") == 1
        assert count(engine, "tail.unflushed") == 2
        assert "unflushed=2" in caplog.text
    finally:
        locker.rollback()
        locker.close()


@pytest.mark.parametrize("between,route,stamped", CELLS)
@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
def test_cold_crash_cell(engines, monkeypatch, between, route, stamped, point):
    engine = engines()
    view = history(stamped)
    queue_cold(engine, monkeypatch, view)
    if between == "partial":
        replay(engine, view[2:3])
    elif between == "full+new":
        replay(engine, [*view[2:], _m("user", "new", 8. if stamped else None)])
    elif between in {"foreign", "other-session"}:
        replay(engine, view[2:], session="other" if between == "other-session" else "S",
               conversation="foreign")
    # Select the retry route before injecting the crash, as well as on recovery.
    if route in {"off-current", "shutdown"}:
        engine._session_id = "next"
    atomic = engine._store.ingest_atomic

    @contextmanager
    def interrupted(**kwargs):
        with atomic(**kwargs):
            yield
            raise Crash("before COMMIT")

    with monkeypatch.context() as patch:
        if point == "before_commit":
            patch.setattr(engine._store, "ingest_atomic", interrupted)
        else:
            patch.setattr(engine, "_pending_session_tails", CrashOnRemove(engine._pending_session_tails))
        # The current route reads that a full replay holds the whole view, so it opens no transaction.
        if between == "full+new" and route == "next" and point == "before_commit":
            engine._retry_pending_session_tails()
        else:
            with pytest.raises(Crash):
                engine._retry_pending_session_tails(sibling=route == "sibling")
    if point == "before_commit":
        own = [r["content"] for r in engine._store.get_session_messages("S") if r["conversation_id"] == "conv"]
        assert own == ["A", "B"] + (["C"] if between == "partial" else
                                      ["C", "D", "new"] if between == "full+new" else [])
    if route == "next":
        engine.on_session_start("next", platform="cli")
    elif route == "off-current":
        engine._session_id = "next"
        engine._retry_pending_session_tails()
    elif route == "sibling":
        sibling = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
        try:
            sibling.on_session_start("next", platform="cli")
        finally:
            sibling.shutdown()
    else:
        engine._retry_pending_session_tails(exiting=True)
    own = [r["content"] for r in engine._store.get_session_messages("S") if r["conversation_id"] == "conv"]
    assert own == ["A", "B", "C", "D"] + (["new"] if between == "full+new" else [])
    assert not engine._pending_session_tails
    _gate_tail(engine, view, between, route, stamped, point, cold=True)


@pytest.mark.parametrize("point", ["before_commit", "after_commit"])
@pytest.mark.parametrize("stamped", [True, False])
def test_cold_subprocess_crash_resume(engines, tmp_path, point, stamped):
    root = tmp_path / "child"
    root.mkdir()
    script = SUBPROCESS.replace(
        "queue(engine, history(stamped))",
        "from tests.invariants.test_p1e_cold_tails import queue_cold\nqueue_cold(engine, patch, history(stamped))")
    env = {**os.environ, "LCM_INGEST_LEDGER": "shadow", "LCM_HOST_MESSAGE_UID": "off"}
    result = subprocess.run([sys.executable, "-c", script, str(root), point, str(stamped)],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (91 if point == "before_commit" else 92), result.stderr
    from tests.test_host_uid_shadow import _engine
    fresh = _engine(root)
    try:
        assert contents(fresh) == (["A", "B", "C"] if point == "before_commit" else ["A", "B", "C", "D"])
        fresh.ingest(history(stamped))
        assert contents(fresh) == ["A", "B", "C", "D"]
        _gate_tail(fresh, history(stamped), "none", "subprocess", stamped, point, cold=True)
    finally:
        fresh.shutdown()


def _gate_tail(engine, view, between, route, stamped, point, *, cold):
    expected = expect_from_view(view, "conv")
    if between == "full+new":
        expected["conv"] += expect_from_view([_m("user", "new")], "conv")["conv"]
    if between in {"foreign", "other-session"}:
        expected.update(expect_from_view(view[2:], "foreign"))
    outcome = store_outcome(engine._store._conn, expected)
    record(dict(family="cold_tail" if cold else "decided_tail", between=between, route=route,
                stamped=stamped, point=point), outcome, engine)
    assert outcome["loss"] == outcome["silent"] == 0, outcome
