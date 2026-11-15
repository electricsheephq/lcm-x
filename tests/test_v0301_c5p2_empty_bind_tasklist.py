"""Two writers at an empty bind: default GA coverage and opt-in task-list loss."""
from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from collections import Counter

import pytest

from hermes_lcm import maintenance_scheduler
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.conftest import _wait_scheduler_idle


PREFIX = "[Your active task list was preserved across context compression]"
SID = "c5p2-two-writers"


class Host:
    """The empty host read handle/schema used by the #2042 Case fixture.

    Only engine B writes H, through ingest. No failed-turn catch-up is planted.
    """

    def __init__(self, home):
        self.path = home / "state.db"
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, "
                "parent_session_id TEXT, started_at REAL, ended_at REAL, "
                "end_reason TEXT, model_config TEXT, profile_name TEXT, "
                "session_key TEXT, transcript_revision INTEGER DEFAULT 0)"
            )
            conn.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT, role TEXT, display_kind TEXT, active INTEGER DEFAULT 1, "
                "timestamp REAL, payload TEXT, compacted INTEGER DEFAULT 0, "
                "_compressed_summary INTEGER DEFAULT 0)"
            )

    def get_messages_as_conversation(self, session_id, **kwargs):
        with sqlite3.connect(self.path) as conn:
            return [json.loads(row[0]) for row in conn.execute(
                "SELECT payload FROM messages WHERE session_id=? AND active=1 ORDER BY id",
                (session_id,),
            )]

    def get_compression_tip(self, session_id):
        return session_id


def _quiesce():
    for value in vars(maintenance_scheduler).values():
        if isinstance(value, maintenance_scheduler._RollupMaintenanceScheduler):
            if not _wait_scheduler_idle(value, 10):
                raise RuntimeError("setup/cleanup: maintenance did not finish")


def _fixture(variant):
    history = [
        {"role": "user", "content": "c5p2 H user 1: look up the fixture marker"},
        {"role": "assistant", "content": "c5p2 H assistant 1: checking marker", "tool_calls": [
            {"id": "c5p2-call-1", "type": "function", "function": {
                "name": "fixture_lookup", "arguments": '{"marker":"c5p2"}',
            }},
        ]},
        {"role": "tool", "content": "c5p2 H tool result: fixture marker found", "tool_call_id": "c5p2-call-1"},
        {"role": "assistant", "content": "c5p2 H assistant 1 final: marker found"},
        {"role": "user", "content": "c5p2 H user 2: retain the marker"},
        {"role": "assistant", "content": "c5p2 H assistant 2: marker retained"},
    ]
    todo = {"role": "user", "content": PREFIX + "\n- [ ] c5p2 retain this task verbatim"}
    reply = {"role": "assistant", "content": "c5p2 reply to the task-list turn"}
    plain = {"role": "user", "content": "c5p2 plain user: retain this next request"}
    plain_reply = {"role": "assistant", "content": "c5p2 reply to the plain turn"}
    if variant == "V2":
        history += [todo, reply]
        return history, history + [plain, plain_reply]
    if variant == "V3":
        return history, history + [todo, reply, plain, plain_reply]
    if variant == "control":
        return history, history + [plain, reply]
    return history, history + [todo, reply]


def _identity(row):
    return row["role"], hashlib.sha256(row["content"].encode("utf-8")).hexdigest()


def _state(engine):
    return {
        "cursor": engine._ingest_cursor,
        "needs_reconcile": engine._ingest_cursor_needs_reconcile,
        "transaction_open": engine._store.connection.in_transaction,
        "stored_rows": len(engine._store.get_session_messages(SID)),
        "last_reconciliation": copy.deepcopy(getattr(engine, "_last_ingest_reconciliation", {})),
    }


class TasklistUserRowMissing(AssertionError):
    """The opt-in defect omits exactly host index 6, the task-list user row."""


@pytest.mark.parametrize("flag_on,variant", [
    pytest.param(flag, variant, marks=pytest.mark.xfail(
        strict=True, raises=TasklistUserRowMissing,
        reason="#2042: empty-bind reconciliation skips exactly the task-list user row",
    ) if flag and variant in ("V1", "V3") else ())
    for flag in (False, True) for variant in ("V1", "V2", "V3", "control")
], ids=[f"{'on' if flag else 'default'}-{variant}"
        for flag in (False, True) for variant in ("V1", "V2", "V3", "control")])
def test_empty_bind_tasklist(tmp_path, flag_on, variant):
    history, host_rows = _fixture(variant)
    home = tmp_path / "home"
    home.mkdir()
    host = Host(home)
    config = LCMConfig(database_path=str(home / "lcm.db"), embeddings_enabled=False,
                       temporal_rollups_enabled=False,
                       **({"host_row_recovery_enabled": True} if flag_on else {}))
    engines = []
    try:
        def bind():
            engine = LCMEngine(config=copy.deepcopy(config), hermes_home=str(home))
            engines.append(engine)
            engine.bind_session_state(host, SID)
            engine.on_session_start(SID, hermes_home=str(home), platform="desktop", context_length=128000)
            _quiesce()
            return engine

        a = bind()
        a_empty = _state(a)
        if (a_empty["cursor"], a_empty["needs_reconcile"], a_empty["stored_rows"]) != (0, False, 0):
            raise RuntimeError(f"setup: A did not bind an empty store: {a_empty}")
        b = bind()
        separate_connections = a._store.connection is not b._store.connection
        if not separate_connections:
            raise RuntimeError("setup: writers share a connection")
        b.ingest(copy.deepcopy(history))
        _quiesce()
        a.ingest(copy.deepcopy(host_rows))
        _quiesce()
        a_after_ingest = _state(a)
        stored = a._store.get_session_messages(SID)
        counts = Counter(_identity(row) for row in stored)
        expected = Counter(_identity(row) for row in host_rows)
        missing = [i for i, row in enumerate(host_rows) if counts[_identity(row)] == 0]
        extras = sum((counts - expected).values())
        if not flag_on:
            assert not missing, {"missing": missing, "extra_copies": extras}
            assert extras <= {"V1": 6, "V2": 8, "V3": 6, "control": 6}[variant]
            assert a_after_ingest["last_reconciliation"] == {"action": "none", "reason": "not run"}, a_after_ingest
        elif variant in ("V1", "V3") and missing == [6]:
            raise TasklistUserRowMissing("missing exactly host index 6 (task-list user row)")
        else:
            assert not missing, {"missing": missing, "extra_copies": extras}
    finally:
        _quiesce()
        for engine in engines:
            engine._unregister_active_engine_binding()
            engine._close_storage()
