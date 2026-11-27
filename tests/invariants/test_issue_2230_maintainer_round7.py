"""I1/I2: preserve host-confirmed foreground views through unload and reload."""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import pytest

from tests.invariants.test_issue_2152_lazy_storage import preflight as preflight
from tests.invariants.test_issue_2230_reload_gaps import durable, unload


def native_sessions(home, rows, messages=()):
    home.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(home / "state.db")) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, "
                   "parent_session_id TEXT, started_at REAL, ended_at REAL, end_reason TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT, "
                   "timestamp REAL, message_uid TEXT, tool_call_id TEXT, tool_calls TEXT)")
        db.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, NULL, ?, ?, ?)", rows)
        db.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)", messages)


def bind(engine, session):
    engine.bind_session_state(SimpleNamespace(db_path=Path(engine._hermes_home) / "state.db"), session)


def pairs(home, session):
    return [(row["role"], row["content"]) for row in durable(home, session)]


@pytest.mark.parametrize("empty_a", [True, False])
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_gap_new_session_end_keeps_all_replies(preflight, empty_a, retry, closed):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("A", platform="cli")
    original = [dict(role="user", content="A question", timestamp=1.0),
                dict(role="assistant", content="A answer", timestamp=2.0)]
    if not empty_a:
        retained.ingest(deepcopy(original))
    now = time.time()
    native_sessions(preflight.home, [("A", "cli", now - 3, now - 1, "new_session")])
    unload(old)
    if not empty_a:
        retained.on_session_end("A", deepcopy(original))
    retained.on_session_reset()
    bind(retained, "B")
    first = [dict(role="user", content="B question 1", timestamp=5.0),
             dict(role="assistant", content="B answer 1", timestamp=6.0)]
    native_sessions(preflight.home, [("B", "cli", time.time(), None, None)])
    retained.should_compress_preflight(deepcopy(first[:1]))
    retained.on_turn_complete(deepcopy(first), interrupted=True)
    assert pairs(preflight.home, "B") == [(r["role"], r["content"]) for r in first]
    new_question = "B question 1" if retry else "B question 2"
    pending = ([] if retry else first) + [dict(role="user", content=new_question, timestamp=7.0)]
    assert retained.should_compress_preflight(deepcopy(pending)) is False
    final = pending + [dict(role="assistant", content="B answer 2", timestamp=8.0)]
    for _ in range(2):
        retained.on_turn_complete(deepcopy(final), interrupted=retry)
    if closed:
        with closing(sqlite3.connect(preflight.home / "state.db")) as db, db:
            db.execute("UPDATE sessions SET ended_at = ?, end_reason = 'cli_close' WHERE id = 'B'", (time.time(),))
            db.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('B', ?, ?, ?)", [
                (row["role"], row["content"], row["timestamp"]) for row in final
            ])
    retained.on_session_end("B", deepcopy(final))
    expected = first + final if retry else final
    assert pairs(preflight.home, "B") == [(r["role"], r["content"]) for r in expected]
    assert pairs(preflight.home, "A") == ([] if empty_a else [(r["role"], r["content"]) for r in original])


@pytest.mark.parametrize("operation", ["preflight", "compress", "session-end"])
@pytest.mark.parametrize("confirmed", [True, False])
def test_gap_rewound_history_uses_only_the_confirmed_host_view(preflight, operation, confirmed):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("S", platform="cli")
    history = [dict(role="user", content="original opening", timestamp=1.0),
               dict(role="assistant", content="opening reply", timestamp=2.0),
               dict(role="user", content="old attempt", timestamp=3.0),
               dict(role="assistant", content="stopped reply", timestamp=4.0)]
    retained.ingest(deepcopy(history))
    unload(old)
    current = [dict(role="user", content="new opening after /undo", timestamp=5.0)]
    native_sessions(preflight.home, [("S", "cli", 0.0, None, None)], [
        ("S" if confirmed else "another-session", "user", current[0]["content"], 5.0),
    ])
    if operation == "preflight":
        retained.should_compress_preflight(deepcopy(current))
    elif operation == "compress":
        supplied = deepcopy(current)
        assert retained.compress(supplied, force=True) is supplied
    else:
        retained.on_session_end("S", deepcopy(current))
    expected = [(r["role"], r["content"]) for r in history + current]
    assert pairs(preflight.home, "S") == expected


@pytest.mark.parametrize("operation", ["preflight", "compress", "interrupted"])
@pytest.mark.parametrize("reloaded", [False, True])
def test_retained_exact_prefix_cannot_append_into_mixed_conversation(preflight, operation, reloaded):
    prototype, old = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("S", platform="cli")
    history = [dict(role="user", content="opening", timestamp=1.0),
               dict(role="assistant", content="reply", timestamp=2.0)]
    retained.ingest(deepcopy(history))
    unload(old)
    with closing(sqlite3.connect(preflight.home / "lcm.db")) as db, db:
        db.execute("INSERT INTO messages (session_id, conversation_id, role, content, token_estimate, timestamp) "
                   "SELECT session_id, 'foreign', role, 'foreign lane row', token_estimate, timestamp "
                   "FROM messages ORDER BY store_id LIMIT 1")
    before_count = len(durable(preflight.home, "S"))
    if reloaded:
        _, new_ctx = preflight.register(supports_unload=True, manager=old._manager)
    current = history + [dict(role="user", content="appended row", timestamp=3.0)]
    if operation == "preflight":
        retained.should_compress_preflight(deepcopy(current))
        appended = 1
    elif operation == "compress":
        retained.compress(deepcopy(current), force=True)
        appended = 1
    else:
        current.append(dict(role="assistant", content="stopped reply", timestamp=4.0))
        retained.on_turn_complete(deepcopy(current), interrupted=True)
        appended = 2
    after = durable(preflight.home, "S")
    assert len(after) == before_count + appended
    assert [(r["role"], r["content"]) for r in after[before_count:]] == [
        (r["role"], r["content"]) for r in current[len(history):]
    ]
    if reloaded:
        unload(new_ctx)


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
    retained.should_compress_preflight([dict(role="user", content="first turn", timestamp=2.0)])
    # on_session_reset clears debt, so B binds; frontier/maintenance keep A bound.
    if history == "debt":
        assert pairs(preflight.home, "B") == [("user", "first turn")]
        assert pairs(preflight.home, "A") == []
    else:
        assert pairs(preflight.home, "B") == []
        assert pairs(preflight.home, "A") == [("user", "first turn")]


@pytest.mark.parametrize("switch", ["branch", "resume-current", "resume-finalized"])
@pytest.mark.parametrize("before_unload", [True, False])
def test_retained_cli_switch_writes_to_confirmed_target(preflight, switch, before_unload):
    prototype, context = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("A", platform="cli")
    original = [dict(role="user", content="old question", timestamp=1.0),
                dict(role="assistant", content="old reply", timestamp=2.0)]
    retained.ingest(deepcopy(original))
    branch = switch == "branch"
    if not branch:
        target = prototype.clone_for_agent()
        target.on_session_start("B", platform="cli", conversation_id="B-lineage")
        target.ingest(deepcopy(original))
        if switch == "resume-finalized":
            target.on_session_end("B", deepcopy(original))
    reason = "branched" if branch else "resumed_other"
    target_source = "cli" if branch else "telegram"
    native_sessions(preflight.home, [("A", "cli", 1.0, 3.0, reason),
                                     ("B", target_source, 2.0, None, None)])
    if branch:
        with closing(sqlite3.connect(preflight.home / "state.db")) as db, db:
            db.execute("UPDATE sessions SET parent_session_id = 'A' WHERE id = 'B'")
    if not before_unload:
        unload(context)
    retained.on_session_reset()
    bind(retained, "B")
    if before_unload:
        unload(context)
        _, context = preflight.register(supports_unload=True, manager=context._manager)
    next_q = dict(role="user", content="next question", timestamp=4.0)
    next_a = dict(role="assistant", content="stopped reply", timestamp=5.0)
    # Main does not rebind an unfinalized CLI session on reset+bind alone; preflight
    # appends only the suffix onto A without duplicating A's prefix into B.
    retained.should_compress_preflight(deepcopy(original + [next_q]))
    assert pairs(preflight.home, "A") == [(r["role"], r["content"]) for r in original + [next_q]]
    assert pairs(preflight.home, "B") == ([] if branch else [(r["role"], r["content"]) for r in original])
    if before_unload:
        # When post_llm_call carries context_compressor=retained under B, it
        # binds B and ingests without duplicating A's rows.
        final = original + [next_q, next_a]
        context.hooks["post_llm_call"](
            session_id="B", platform=target_source, conversation_history=deepcopy(final), context_compressor=retained,
        )
        assert pairs(preflight.home, "B") == [(r["role"], r["content"]) for r in final]
        unload(context)


def test_closed_target_flush_with_a_live_successor_for_previous_session(preflight):
    prototype, context = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("A", platform="cli")
    history = [dict(role="user", content="original", timestamp=1.0),
               dict(role="assistant", content="original reply", timestamp=2.0)]
    retained.ingest(deepcopy(history))
    unload(context)
    _, cur_ctx = preflight.register(supports_unload=True, manager=context._manager)
    retained.should_compress_preflight(deepcopy(history))
    assert retained._reload_successor is not None

    native_sessions(preflight.home, [("A", "cli", 1.0, 3.0, "branched"),
                                     ("B", "cli", 2.0, None, None)])
    with closing(sqlite3.connect(preflight.home / "state.db")) as db, db:
        db.execute("UPDATE sessions SET parent_session_id = 'A' WHERE id = 'B'")
    retained.on_session_reset()
    bind(retained, "B")
    final = history + [dict(role="user", content="branch question", timestamp=4.0),
                       dict(role="assistant", content="branch reply", timestamp=5.0)]
    with closing(sqlite3.connect(preflight.home / "state.db")) as db, db:
        db.execute("UPDATE sessions SET ended_at = ?, end_reason = 'cli_close' WHERE id = 'B'", (time.time(),))
        db.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('B', ?, ?, ?)", [
            (row["role"], row["content"], row["timestamp"]) for row in final
        ])
    retained.on_session_end("B", deepcopy(final))
    # Main keeps A bound when A was non-empty and unfinalized, so stale end('B') is ignored.
    assert pairs(preflight.home, "A") == [(row["role"], row["content"]) for row in history]
    assert pairs(preflight.home, "B") == []
    unload(cur_ctx)


def test_resume_hook_without_compressor_cannot_reopen_an_advanced_conversation(preflight):
    prototype, context = preflight.register(supports_unload=True)
    retained = prototype.clone_for_agent()
    retained.on_session_start("A", platform="cli")
    original = [dict(role="user", content="original", timestamp=1.0),
                dict(role="assistant", content="reply", timestamp=2.0)]
    retained.ingest(deepcopy(original))
    target = prototype.clone_for_agent()
    target.on_session_start("B", platform="cli", conversation_id="B-lineage")
    target.ingest(deepcopy(original))
    target.on_session_end("B", deepcopy(original))
    target._lifecycle.bind_session("C", conversation_id="B-lineage")
    assert target._lifecycle.get_by_conversation("B-lineage").current_session_id == "C"
    unload(context)
    native_sessions(preflight.home, [("A", "cli", 1.0, 3.0, "resumed_other"),
                                     ("B", "cli", 2.0, None, None)])
    retained.on_session_reset()
    bind(retained, "B")
    replacement, current = preflight.register(supports_unload=True, manager=context._manager)
    later = original + [dict(role="user", content="later question", timestamp=4.0),
                        dict(role="assistant", content="later reply", timestamp=5.0)]
    current.hooks["post_llm_call"](session_id="B", platform="cli", conversation_history=deepcopy(later))
    assert replacement._lifecycle.get_by_conversation("B-lineage").current_session_id == "C"
    assert pairs(preflight.home, "B") == [(row["role"], row["content"]) for row in later]
    unload(current)
