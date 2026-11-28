"""#2737 stays open: a KeyboardInterrupt during the session-end flush loses the session's final rows.

When the interrupt (for example a second Ctrl+C while the host exits) lands in the end flush's write transaction, the
transaction rolls back and the engine restores its snapshots, but the cold-tail queue only takes an ``Exception``, so
the final rows are not stored, not queued as a pending tail, and no counter moves; a warning says final messages may
be absent. A session that is never resumed keeps only its earlier rows, even after shutdown and a later session.
These strict xfails pin that loss; a fix flips them. The controls show that a resume re-sending the full history
stores the tail once, and that the same engine's next session start has nothing to retry.
"""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

_PIN = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#2737: an interrupt in the session-end flush neither stores, queues nor counts the final rows",
)


def _require(condition, what):
    # Setup checks raise pytest's Failed, never AssertionError, so the strict xfails cannot absorb them.
    if not condition:
        pytest.fail(f"setup: {what}")


@pytest.fixture
def engines(tmp_path):
    instances = []

    def create():
        instance = LCMEngine(config=LCMConfig(
            database_path=str(tmp_path / "interrupt.db"), temporal_rollups_enabled=False,
        ))
        instances.append(instance)
        return instance

    yield create
    for instance in reversed(instances):
        try:
            instance.shutdown()
        except Exception:
            pass


MESSAGES = [
    {"role": "user", "content": "stored question"},
    {"role": "assistant", "content": "stored answer"},
    {"role": "user", "content": "final question at exit"},
    {"role": "assistant", "content": "final answer at exit"},
]


def _contents(engine, session_id="S"):
    return [row["content"] for row in engine._store.get_session_messages(session_id)]


def _counters(engine):
    return dict(engine.__dict__.get("_ingest_ledger_counters", {}))


def _interrupted_end(engine, monkeypatch):
    engine.on_session_start("S", platform="cli")
    engine.ingest(MESSAGES[:2])
    _require(_contents(engine) == ["stored question", "stored answer"], "the first ingest stored two rows")
    before = _counters(engine)
    real = engine._ingest_messages_transaction
    calls = {"n": 0}

    def interrupted(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt()  # Ctrl+C lands inside the write transaction
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "_ingest_messages_transaction", interrupted)
    engine.on_session_end("S", MESSAGES)  # the engine swallows the interrupt (fail-open)
    monkeypatch.setattr(engine, "_ingest_messages_transaction", real)
    _require(calls["n"] == 1, "the fault must hit the session-end flush")
    after = _counters(engine)
    counted = {k: after[k] - before.get(k, 0) for k in after
               if after[k] > before.get(k, 0) and any(w in k for w in ("tail", "interrupt", "unflushed", "end"))}
    return counted


@_PIN
def test_interrupted_flush_rows_are_stored_queued_or_counted(engines, monkeypatch):
    """Every row of the end is stored, queued, or counted at exit."""
    engine = engines()
    counted = _interrupted_end(engine, monkeypatch)
    stored = _contents(engine) == [m["content"] for m in MESSAGES]
    queued = any(t.get("session_id") == "S" for t in engine._pending_session_tails)
    assert stored or queued or counted, (
        f"final rows neither stored ({_contents(engine)}), queued "
        f"({len(engine._pending_session_tails)} tails) nor counted ({_counters(engine)})")


@_PIN
def test_never_resumed_session_keeps_final_rows_after_shutdown(engines, monkeypatch):
    """A session that is never resumed: shutdown and a later different session must not lose the tail."""
    original = engines()
    _interrupted_end(original, monkeypatch)
    original.shutdown()
    nxt = engines()
    nxt.on_session_start("other-session", platform="cli")
    nxt.ingest([{"role": "user", "content": "other question"}])
    assert _contents(nxt) == [m["content"] for m in MESSAGES]


def test_control_same_session_resume_recovers_tail_exactly_once(engines, monkeypatch):
    """CONTROL: resuming S and ingesting its full history stores the tail once (delay, not loss)."""
    original = engines()
    _interrupted_end(original, monkeypatch)
    original.shutdown()
    resumed = engines()
    resumed.on_session_start("S", platform="cli")
    resumed.ingest(MESSAGES)
    resumed.ingest(MESSAGES)
    assert _contents(resumed) == [m["content"] for m in MESSAGES]


def test_control_next_start_keeps_stored_rows_once(engines, monkeypatch):
    """CONTROL: after the interrupted end, the same engine's next session start keeps the two stored rows once.

    A fix may add the final rows here; it must not drop or duplicate the rows already stored.
    """
    engine = engines()
    _interrupted_end(engine, monkeypatch)
    engine.on_session_start("S2", platform="cli")
    rows = _contents(engine, "S")
    assert rows[:2] == ["stored question", "stored answer"], rows
    assert len(rows) == len(set(rows)), rows
