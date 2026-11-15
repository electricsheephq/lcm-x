"""#2738 stays open: a busy suffix append at an off-current end of a bypass-lineage session loses the suffix.

An off-current end whose store prefix is accepted appends the rows ingest never stored. For a session that also ran
bypassed once (bypass lineage), the end is not eligible for the off-current cold-tail queue, so a busy error from the
suffix append is re-raised: nothing is stored, queued or counted, and no later path (tail retry, another session's
end, rebind, shutdown) stores C and D. A real writer lock at the end loses them the same way. These strict xfails pin
that loss; a fix flips them. The controls show the same end with no fault appends C and D, and that a re-delivered
end or a resume that re-sends the full history stores them.
"""

import copy
import sqlite3

import pytest

from hermes_lcm.store import MessageStore
from tests.invariants.test_p1e_cold_tails import off_current_end
from tests.test_issue_2055_ingest_ledger_writes import engines as engines  # noqa: F401  (fixture)

_PIN = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#2738: a busy suffix append at a bypass-lineage off-current end neither stores nor queues the suffix",
)


def _require(condition, what):
    # Setup checks raise pytest's Failed, never AssertionError, so the strict xfails cannot absorb them.
    if not condition:
        pytest.fail(f"setup: {what}")


def contents(engine, sid="S1"):
    return [row["content"] for row in engine._store.get_session_messages(sid)]


def setup(engines):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    try:
        view = off_current_end(engine)  # S1 normal on cli/conv1 with A,B stored; S2 now current
    except AssertionError as exc:  # the shared helper's own setup check: never let a strict xfail absorb it
        pytest.fail(f"setup: off_current_end: {exc!r}")
    _require(contents(engine) == ["A", "B"], "S1 stored A and B before its end")
    engine._mark_lcm_bypass_lineage_session("S1", platform="cron")  # S1 also ran bypassed (stateless/ignored) once
    _require(engine._has_lcm_bypass_lineage_session("S1"), "S1 carries bypass lineage")
    return engine, view


def test_control_bypass_lineage_off_current_end_appends_suffix(engines):
    """What this end is meant to store: the prefix is accepted, so C,D are appended."""
    engine, view = setup(engines)
    engine.on_session_end("S1", copy.deepcopy(view))
    assert contents(engine) == ["A", "B", "C", "D"]


def busy_end(engine, view, monkeypatch):
    calls = []

    def busy(*args, **kwargs):
        calls.append(args)
        raise sqlite3.OperationalError("database is locked")

    raised = None
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_off_current_session_end_suffix", busy)
        try:
            engine.on_session_end("S1", copy.deepcopy(view))
        except Exception as exc:  # the host sees whatever escapes
            raised = exc
    _require(calls, "forcing failed: the suffix append was never reached")
    return raised


def accounted(engine, view):
    stored = contents(engine) == ["A", "B", "C", "D"]
    queued = any(t.get("session_id") == "S1" and t.get("view") == view for t in engine._pending_session_tails)
    counters = dict(getattr(engine, "_ingest_ledger_counters", {}))
    return stored, queued, counters


@_PIN
def test_busy_suffix_append_rows_stored_queued_or_counted(engines, monkeypatch):
    engine, view = setup(engines)
    raised = busy_end(engine, view, monkeypatch)
    stored, queued, counters = accounted(engine, view)
    assert stored or queued, (
        f"C,D lost at end: raised={raised!r} rows={contents(engine)} "
        f"pending={len(engine._pending_session_tails)} counters={counters}")


@_PIN
def test_recovery_retry_pending_tails(engines, monkeypatch):
    engine, view = setup(engines)
    busy_end(engine, view, monkeypatch)
    engine._retry_pending_session_tails()
    engine._retry_session_end_tails()
    assert contents(engine) == ["A", "B", "C", "D"], f"after tail retry: {contents(engine)}"


@_PIN
def test_recovery_later_end_of_other_session(engines, monkeypatch):
    engine, view = setup(engines)
    busy_end(engine, view, monkeypatch)
    engine.on_session_end("S2", [])
    assert contents(engine) == ["A", "B", "C", "D"], f"after S2 end: {contents(engine)}"


@_PIN
def test_recovery_rebind_S1_without_resend(engines, monkeypatch):
    engine, view = setup(engines)
    busy_end(engine, view, monkeypatch)
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    assert contents(engine) == ["A", "B", "C", "D"], f"after rebind: {contents(engine)}"


@_PIN
def test_recovery_shutdown_drain(engines, monkeypatch):
    engine, view = setup(engines)
    busy_end(engine, view, monkeypatch)
    engine.shutdown()  # a crash here fails the test instead of being absorbed by the strict xfail
    store = MessageStore(engine._store.db_path)
    try:
        rows = [r["content"] for r in store.get_session_messages("S1")]
    finally:
        store.close()
    assert rows == ["A", "B", "C", "D"], f"after shutdown drain: {rows}"


def test_host_replay_same_end_stores(engines, monkeypatch):
    """CONTROL: if the host re-delivered the identical end (Hermes does not normally), it stores."""
    engine, view = setup(engines)
    busy_end(engine, view, monkeypatch)
    engine.on_session_end("S1", copy.deepcopy(view))
    assert contents(engine) == ["A", "B", "C", "D"]


def test_resume_and_reingest_full_history_stores(engines, monkeypatch):
    """CONTROL: a resumed S1 whose host history still carries C,D re-sends them on ingest."""
    engine, view = setup(engines)
    busy_end(engine, view, monkeypatch)
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    engine.ingest(copy.deepcopy(view))
    assert contents(engine) == ["A", "B", "C", "D"], f"after resume+ingest: {contents(engine)}"


@_PIN
def test_real_writer_lock_at_end_rows_stored_queued_or_counted(engines, monkeypatch):
    """No injected fault: another connection holds the write lock during the end."""
    engine, view = setup(engines)
    reached = []
    real_append = engine._append_off_current_session_end_suffix

    def spy(*args, **kwargs):  # records the call and runs the real append
        reached.append(1)
        return real_append(*args, **kwargs)

    monkeypatch.setattr(engine, "_append_off_current_session_end_suffix", spy)
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    raised = None
    try:
        try:
            engine.on_session_end("S1", copy.deepcopy(view))
        except Exception as exc:
            raised = exc
    finally:
        locker.rollback()
        locker.close()
    _require(reached, "the real lock must hit the suffix append, not an earlier write")
    stored, queued, counters = accounted(engine, view)
    assert stored or queued, (
        f"C,D lost at end: raised={raised!r} rows={contents(engine)} "
        f"pending={len(engine._pending_session_tails)} counters={counters}")
