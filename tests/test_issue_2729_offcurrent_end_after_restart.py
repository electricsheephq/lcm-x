"""#2729 stays open: after a restart or a profile rebind, an off-current session end stores none of its unstored rows.

When the host ends a session that is not the bound one, LCM-X matches the end to that session through lineage it keeps
in process memory, and stores the rows ingest never saw. A restart or a profile rebind clears that lineage, so the end
is ignored as stale (one warning) and the session's unstored rows, usually its final reply, are not stored until a
later resume and ingest of that session, which may never come. These strict xfails pin that loss for both shapes of
unstored rows: a final reply only the end view carries (the #2539 shape), and a turn whose post-turn ingest failed busy.
The no-cut cases are controls that store every row. A fix flips the xfails.

Busy post-turn ingest leaves unstored rows but does NOT queue a cold tail: that fallback requires session-end flushing.
"""

import copy
import logging
from contextlib import contextmanager
from pathlib import Path
import sqlite3

import pytest

from tests.test_host_uid_shadow import _engine
from tests.test_issue_2055_ingest_ledger_writes import engines as engines

_PIN = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#2729: after a restart or profile rebind an off-current session end is ignored as stale; its rows are lost",
)
_STALE = "LCM ignored unverified stale session-end callback for S2 while bound to S1"


def _require(condition, message):
    # A broken precondition is a change of the pinned shape, not the expected failure.
    if not condition:
        pytest.fail(message)


def _start(engine, sid, **kwargs):
    engine.on_session_start(
        sid, platform="cli", context_length=200_000,
        conversation_id=f"conv-{sid}", **kwargs,
    )


@pytest.mark.parametrize("cut", ["none", pytest.param("restart", marks=_PIN), pytest.param("profile-rebind", marks=_PIN)])
@pytest.mark.parametrize("shape", ["end-only-final", "busy-post-turn"])
def test_offcurrent_end_stores_unstored_rows(engines, monkeypatch, tmp_path, caplog, cut, shape):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    _start(engine, "S2")
    history = [
        {"role": "user", "content": "question", "timestamp": 1.0},
        {"role": "assistant", "content": "initial reply", "timestamp": 2.0},
    ]
    engine.ingest(copy.deepcopy(history))
    before = engine._store.get_session_messages("S2")
    _require([row["content"] for row in before] == [row["content"] for row in history], "S2's first turn was not stored")
    view = history + [{"role": "assistant", "content": "FINAL", "timestamp": 3.0}]

    if shape == "busy-post-turn":
        attempts = []

        @contextmanager
        def busy(**kwargs):
            attempts.append(kwargs)
            raise sqlite3.OperationalError("database is locked")
            yield  # pragma: no cover -- retain the transaction context-manager contract

        # Force both real ingest admission attempts to see SQLITE_BUSY. Do not
        # enable the session-end-only fallback or insert a synthetic cold tail.
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "ingest_atomic", busy)
            engine.ingest(copy.deepcopy(view))
        _require(len(attempts) == 2, f"expected 2 busy ingest attempts, saw {len(attempts)}")
        _require(engine._ingest_failure_count == 1, "the busy ingest was not counted as one failure")
        _require(not engine._session_end_flush_active, "a session-end flush was active during post-turn ingest")
        _require(not engine._pending_session_tails, "post-turn ingest unexpectedly queued a cold tail")
        _require(engine._store.get_session_messages("S2") == before, "the busy ingest stored rows")
    else:
        # The #2539 final-reply shape: ingest never receives this last row.
        _require(engine._ingest_failure_count == 0, "S2's first ingest failed")

    _start(engine, "S1")
    _require(engine._lcm_session_last_normal_conversation_id.get("S2") == "conv-S2", "S2's lineage was not recorded")
    _require(engine._store.get_session_messages("S2") == before, "binding S1 changed S2's stored rows")
    replacement = None
    try:
        if cut == "restart":
            home = Path(engine._hermes_home)
            engine.shutdown()
            replacement = _engine(home, session="S1", conversation="conv-S1")
            engine = replacement
        elif cut == "profile-rebind":
            # Exercise the real configured-database rebind route, which calls
            # _reset_profile_runtime_state while retaining this same store.
            home = engine._hermes_home
            other_home = tmp_path / "other-profile"
            other_home.mkdir()
            store = engine._store
            _start(engine, "S1", hermes_home=str(other_home))
            _start(engine, "S1", hermes_home=home)
            _require(engine._store is store, "the profile rebind did not return to the same store")

        _require(engine._session_id == "S1", "the engine is not bound to S1 before S2 ends")
        _require(engine._store.get_session_messages("S2") == before, "the cut changed S2's stored rows")
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            engine.on_session_end("S2", copy.deepcopy(view))
        stale = _STALE in caplog.text
        contents = [row["content"] for row in engine._store.get_session_messages("S2")]
        expected = [row["content"] for row in view]
        if cut == "none":
            _require(not stale, "the no-cut control took the stale-end route")
        elif contents != expected:
            # The pinned loss comes through the stale-end route; a loss by any other route is not #2729.
            _require(stale, f"S2 lost rows without the stale-end warning: stored={contents!r}")
        assert contents == expected, (
            f"shape={shape} cut={cut} stale_end={stale}; "
            f"stored={contents!r}; missing={[c for c in expected if c not in contents]!r}"
        )
        assert all(row["conversation_id"] == "conv-S2" for row in engine._store.get_session_messages("S2"))
        assert not engine._store.get_session_messages("S1")
    finally:
        if replacement is not None:
            replacement.shutdown()
