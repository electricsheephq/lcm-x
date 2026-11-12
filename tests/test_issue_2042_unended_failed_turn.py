"""#2042: a user row whose turn fails or is stopped, in a session where no later turn completes and no
session end fires, is never stored.

Host facts these tests model (Hermes v0.23.1):
- The terminal-failure paths (retry exhaustion, truncation end, interrupt before any assistant text) persist
  the accepted user row and return without ``finalize_turn``: no ``post_llm_call`` and no ``on_turn_complete``.
  ``agent/conversation_loop.py::_close_durable_failed_turn`` then appends a Hermes-authored assistant
  boundary (``display_kind='failed_turn'``) to the host transcript.
- An interrupt after streamed text reaches ``finalize_turn``: ``on_turn_complete(interrupted=True)`` fires,
  ``post_llm_call`` does not. On a first turn the engine owns no rows yet, so it defers (#1623).
- Desktop sessions never receive ``on_session_end`` in production: the backend exits and the next start
  reaps the session rows (``startup_orphan_reap`` / ``ws_orphan_reap``).

The strict-xfail tests assert at the latest point an LCM-X fix can act on a shipped host: after the reap,
when a new engine (a new process) binds the profile's next session. The controls pass on main today and
guard against a duplicate class once the fix lands.
"""
from __future__ import annotations

import copy
import sqlite3

import pytest

from hermes_lcm import maintenance_scheduler
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

# agent/turn_failure_copy.py FAILED_TURN_NOTICE (83 characters).
FAILED_TURN_NOTICE = "Your request was not processed. Send it again if you still want me to carry it out."
FAILED = "desktop-2042-failed"
NEXT = "desktop-2042-next"
XFAIL = pytest.mark.xfail(
    strict=True,
    reason="#2042: an unended Desktop session's failed or stopped first turn is never stored",
)


class HostTranscript:
    """The host state.db for one profile, plus the read handle Hermes passes to bind_session_state."""

    def __init__(self, home):
        self.path = home / "state.db"
        self.lists: dict[str, list[dict]] = {}
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT, "
                         "started_at REAL, ended_at REAL, end_reason TEXT, message_count INTEGER, "
                         "model_config TEXT)")

    def write(self, session_id, rows, *, started_at):
        self.lists[session_id] = copy.deepcopy(rows)
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?)",
                         (session_id, "desktop", None, started_at, None, None, len(rows), None))

    def reap(self, session_id, *, at):
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE sessions SET ended_at=?, end_reason='startup_orphan_reap' WHERE id=?",
                         (at, session_id))

    def get_messages_as_conversation(self, session_id, *, include_ancestors=True, include_inactive=False,
                                     repair_alternation=True, include_row_ids=False, include_compacted=False):
        return copy.deepcopy(self.lists.get(session_id, []))

    def get_compression_tip(self, session_id):
        return session_id


def _rows(*pairs, base=1_900_000_000):
    return [{"role": role, "content": text, "timestamp": base + i} for i, (role, text) in enumerate(pairs)]


def _new_engine(tmp_path, host, session_id):
    """One process's engine binding one Desktop session the way Hermes builds an agent."""
    home = tmp_path / "home"
    engine = LCMEngine(
        config=LCMConfig(database_path=str(home / "lcm.db"), embeddings_enabled=False,
                         temporal_rollups_enabled=False),
        hermes_home=str(home),
    )
    engine.bind_session_state(session_db=host, session_id=session_id)
    engine.on_session_start(session_id, hermes_home=str(home), platform="desktop", context_length=128000)
    return engine


def _quiesce():
    for value in vars(maintenance_scheduler).values():
        if isinstance(value, maintenance_scheduler._RollupMaintenanceScheduler):
            value.quiesce(timeout=10)


def _process_dies(engine):
    """Backend exit without a session end: storage closes, no on_session_end, no shutdown flush of turns."""
    _quiesce()
    engine._close_storage()


def _next_process_turn(tmp_path, host, rows):
    """The next process binds the profile's next Desktop session and completes one turn there."""
    host.write(NEXT, rows, started_at=1_900_000_300)
    engine = _new_engine(tmp_path, host, NEXT)
    engine.ingest(rows)  # post_llm_call of a completed turn
    _quiesce()
    return engine


def _contents(engine, session_id):
    return [(m["role"], m["content"]) for m in engine._store.get_session_messages(session_id)]


@pytest.fixture
def host(tmp_path):
    (tmp_path / "home").mkdir()
    return HostTranscript(tmp_path / "home")


@pytest.fixture
def engines():
    made = []
    yield made
    for engine in made:
        try:
            engine._close_storage()
        except Exception:
            pass


def _failed_first_turn(tmp_path, host, engines, text):
    engine = _new_engine(tmp_path, host, FAILED)
    engines.append(engine)
    # The host writes the accepted user row and the failed-turn boundary; the engine receives no hook.
    host.write(FAILED, _rows(("user", text), ("assistant", FAILED_TURN_NOTICE)), started_at=1_900_000_000)
    return engine


@XFAIL
def test_failed_first_turn_then_reap_user_row_is_stored(tmp_path, host, engines):
    _process_dies(_failed_first_turn(tmp_path, host, engines, "synthetic request 2042 failed"))
    host.reap(FAILED, at=1_900_000_200)
    nxt = _next_process_turn(tmp_path, host, _rows(("user", "unrelated next question"), ("assistant", "ok")))
    engines.append(nxt)
    assert _contents(nxt, FAILED) == [("user", "synthetic request 2042 failed"),
                                      ("assistant", FAILED_TURN_NOTICE)]


@XFAIL
def test_stopped_first_turn_before_text_then_reap_user_row_is_stored(tmp_path, host, engines):
    # Interrupt before any assistant text: the same early-return path and boundary as a failure.
    _process_dies(_failed_first_turn(tmp_path, host, engines, "synthetic request 2042 stopped early"))
    host.reap(FAILED, at=1_900_000_200)
    nxt = _next_process_turn(tmp_path, host, _rows(("user", "unrelated next question"), ("assistant", "ok")))
    engines.append(nxt)
    assert ("user", "synthetic request 2042 stopped early") in _contents(nxt, FAILED)


@pytest.mark.xfail(
    strict=True,
    reason="#2042 follow-up: a first turn stopped during a tool round (finalize path, no failed_turn boundary) is "
           "deferred by #1623 and never stored when no later turn completes",
)
def test_stopped_first_turn_during_tool_round_then_reap_user_row_is_stored(tmp_path, host, engines):
    # A stop after a tool ran reaches finalize_turn: on_turn_complete(interrupted=True) fires, post_llm_call does not,
    # and the host closes the tail with its own interrupted assistant row (no failed_turn boundary).
    engine = _new_engine(tmp_path, host, FAILED)
    engines.append(engine)
    rows = _rows(("user", "synthetic request 2042 stopped late"))
    rows += [
        {"role": "assistant", "content": "", "timestamp": 1_900_000_001, "tool_calls": [
            {"id": "call-2042", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": "synthetic tool result 2042", "tool_call_id": "call-2042",
         "timestamp": 1_900_000_002},
        {"role": "assistant", "content": "interrupted by user", "timestamp": 1_900_000_003},
    ]
    host.write(FAILED, rows, started_at=1_900_000_000)
    engine.on_turn_complete(rows, interrupted=True, failed=False)  # first turn: the engine owns no rows yet
    _process_dies(engine)
    host.reap(FAILED, at=1_900_000_200)
    nxt = _next_process_turn(tmp_path, host, _rows(("user", "unrelated next question"), ("assistant", "ok")))
    engines.append(nxt)
    assert ("user", "synthetic request 2042 stopped late") in _contents(nxt, FAILED)


def test_control_failed_turn_resent_in_new_session_is_stored_once_there(tmp_path, host, engines):
    _process_dies(_failed_first_turn(tmp_path, host, engines, "synthetic resent request 2042"))
    host.reap(FAILED, at=1_900_000_200)
    nxt = _next_process_turn(tmp_path, host, _rows(("user", "synthetic resent request 2042"), ("assistant", "ok")))
    engines.append(nxt)
    assert _contents(nxt, NEXT).count(("user", "synthetic resent request 2042")) == 1
    assert len(_contents(nxt, NEXT)) == 2


def test_control_failed_then_completed_turn_in_same_session_stores_each_row_once(tmp_path, host, engines):
    engine = _failed_first_turn(tmp_path, host, engines, "synthetic request 2042 retried")
    history = _rows(("user", "synthetic request 2042 retried"), ("assistant", FAILED_TURN_NOTICE),
                    ("user", "second try"), ("assistant", "done"))
    host.write(FAILED, history, started_at=1_900_000_000)
    engine.ingest(history)  # post_llm_call of the completed second turn carries the failed turn's rows
    _quiesce()
    assert _contents(engine, FAILED) == [(m["role"], m["content"]) for m in history]
