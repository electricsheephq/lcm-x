"""Unresolved review 5931574277: exercise the missing retained paths.

These tests deliberately assert the requested behavior, not the existing
retired writer's verdict. They are reproduction coverage, not a shipped fix.
"""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import pytest

from .test_issue_2152_lazy_storage import preflight  # noqa: F401
from .test_issue_2230_reload_gaps import durable, unload


def native_sessions(home, rows):
    with closing(sqlite3.connect(home / "state.db")) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, "
                   "parent_session_id TEXT, started_at REAL, ended_at REAL, end_reason TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT, "
                   "timestamp REAL, message_uid TEXT)")
        db.executemany("INSERT INTO sessions VALUES (?, ?, NULL, ?, ?, ?)", rows)


def bind(engine, session):
    engine.bind_session_state(SimpleNamespace(db_path=Path(engine._hermes_home) / "state.db"), session)


def pairs(home, session):
    return [(row["role"], row["content"]) for row in durable(home, session)]


def test_gap_new_session_end_keeps_last_completed_reply(preflight):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("A", platform="cli")
    original = [dict(role="user", content="A question", timestamp=1.0),
                dict(role="assistant", content="A answer", timestamp=2.0)]
    retained.ingest(deepcopy(original))
    now = time.time()
    native_sessions(preflight.home, [("A", "cli", now - 3, now - 1, "new_session")])
    unload(old)
    retained.on_session_reset()
    bind(retained, "B")
    first = [dict(role="user", content="B question 1", timestamp=5.0),
              dict(role="assistant", content="B answer 1", timestamp=6.0)]
    native_sessions(preflight.home, [("B", "cli", time.time(), None, None)])
    retained.on_turn_complete(deepcopy(first), interrupted=True)
    assert pairs(preflight.home, "B") == [(r["role"], r["content"]) for r in first]
    pending = first + [dict(role="user", content="B question 2", timestamp=7.0)]
    assert retained.should_compress_preflight(deepcopy(pending)) is False
    final = pending + [dict(role="assistant", content="B answer 2", timestamp=8.0)]
    retained.on_turn_complete(deepcopy(final), interrupted=False)
    retained.on_session_end("B", deepcopy(final))
    assert pairs(preflight.home, "B") == [(r["role"], r["content"]) for r in final]


@pytest.mark.parametrize("operation", ["preflight", "compress", "session-end"])
def test_gap_edited_history_with_unstamped_opening_uses_normal_ingest(preflight, operation):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("S", platform="cli")
    history = [dict(role="user", content="opening without timestamp"),
               dict(role="assistant", content="opening reply", timestamp=2.0),
               dict(role="user", content="old attempt", timestamp=3.0),
               dict(role="assistant", content="stopped reply", timestamp=4.0)]
    retained.ingest(deepcopy(history))
    unload(old)
    current = history[:2] + [dict(role="user", content="edited retry", timestamp=5.0)]
    if operation == "preflight":
        retained.should_compress_preflight(deepcopy(current))
    elif operation == "compress":
        retained.compress(deepcopy(current), force=True)
    else:
        retained.on_session_end("S", deepcopy(current))
    expected = [(r["role"], r["content"]) for r in history + current[-1:]]
    assert pairs(preflight.home, "S") == expected


@pytest.mark.parametrize("operation", ["preflight", "compress", "interrupted"])
def test_gap_exact_prefix_cannot_append_into_mixed_conversation(preflight, operation):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("S", platform="cli")
    history = [dict(role="user", content="opening", timestamp=1.0),
               dict(role="assistant", content="reply", timestamp=2.0)]
    retained.ingest(deepcopy(history))
    unload(old)
    with closing(sqlite3.connect(preflight.home / "lcm.db")) as db, db:
        # Add a row whose session matches but whose conversation does not.
        # Leave the original exact prefix unchanged to hit the fast path.
        db.execute("INSERT INTO messages (session_id, conversation_id, role, content, token_estimate, timestamp) "
                   "SELECT session_id, 'foreign', role, 'foreign lane row', token_estimate, timestamp "
                   "FROM messages ORDER BY store_id LIMIT 1")
    before = durable(preflight.home, "S")
    current = history + [dict(role="user", content="must not append", timestamp=3.0)]
    if operation == "preflight":
        retained.should_compress_preflight(deepcopy(current))
    elif operation == "compress":
        retained.compress(deepcopy(current), force=True)
    else:
        retained.on_turn_complete(deepcopy(current), interrupted=True)
    assert durable(preflight.home, "S") == before


@pytest.mark.parametrize("history", ["frontier", "debt", "maintenance"])
def test_empty_reset_proof_rejects_nonempty_lifecycle_history(preflight, history):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("A", platform="cli")
    changes = {
        "frontier": "last_finalized_frontier_store_id=7",
        "debt": "debt_kind='review-reproduction', debt_size_estimate=1",
        "maintenance": "last_maintenance_attempt_at=7.0",
    }
    with closing(sqlite3.connect(preflight.home / "lcm.db")) as db, db:
        db.execute("UPDATE lcm_lifecycle_state SET " + changes[history] + " WHERE conversation_id='A'")
    unload(old)
    retained.on_session_reset()
    bind(retained, "B")
    native_sessions(preflight.home, [("B", "cli", time.time(), None, None)])
    retained.should_compress_preflight([dict(role="user", content="unproven new lane", timestamp=2.0)])
    assert durable(preflight.home, "B") == []
