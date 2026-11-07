"""REPRO (expected RED): a turn that lands while a manual compress is in flight is stored twice, unlinked.

Host-view sequence (content-free evidence, ACP rotation 185a62de and TUI in-place c85ae0):
1. the host writes the user row T at the compress request (stamp t_req) and LCM ingests it;
2. compress() runs over the list ending in T; the assistant reply lands while it runs and is ingested;
3. the compaction commits (in place, or rotating into a child) and the host's compacted list carries
   T again with the commit-time stamp (Hermes' ``_insert_message_rows`` stamps ``now_ts`` when no
   active donor row matches), optionally with the same ``message_uid``;
4. the next turn is ingested.
LCM then holds T twice (t_req and t_commit) with no ``possible_replay_of`` link. The control keeps
T's stamp and stores T once. No fault is injected.
"""

import logging
import sqlite3
from copy import deepcopy
from typing import Any

import pytest

import hermes_lcm.engine as engine_module
from tests.harness import make_engine
from tests.test_compression_boundary import _stub_summarizer

T = "please list the three deploy steps again for the staging box"
REPLY = "steps: build, push, restart"
T_REQ, T_REPLY, T_COMMIT = 2000.146, 2003.167, 2155.192
Row = dict[str, Any]


def history(uids: bool) -> list[Row]:
    rows: list[Row] = []
    for i in range(14):
        rows.append({"role": "user", "content": f"[T{i:02d}] user turn {i}: " + "alpha beta gamma delta " * 40,
                     "timestamp": 1000.0 + 10 * i})
        rows.append({"role": "assistant", "content": f"reply to T{i:02d}: noted item {i}.",
                     "timestamp": 1000.0 + 10 * i + 1})
    if uids:
        for i, row in enumerate(rows):
            row["message_uid"] = f"{i + 1:032x}"
    return rows


def all_rows(engine: Any) -> list[Row]:
    cols = ("store_id", "session_id", "role", "content", "observed_at")
    return [dict(zip(cols, r)) for r in engine._store._conn.execute(
        "SELECT store_id, session_id, role, content, observed_at FROM messages ORDER BY store_id")]


def replay_links(engine: Any) -> set[tuple[int, int]]:
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone():
        return set()
    return {(a, b) for a, b in conn.execute(
        "SELECT store_id, related_store_id FROM message_relations WHERE kind='possible_replay_of'")}


def lineage(root) -> None:
    """Hermes' state.db for an ACP rotation: the child ``rotated`` of ``S``, ended by compression."""
    home = root / "home"
    home.mkdir(exist_ok=True)
    with sqlite3.connect(home / "state.db") as db:
        db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)",
                       [("S", None, "compression"), ("rotated", "S", None)])


@pytest.fixture
def build(tmp_path, monkeypatch):
    engines: list[Any] = []
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())

    def create(rotating: bool = False) -> Any:
        if rotating:
            lineage(tmp_path)
        engine = make_engine(tmp_path, window=200_000, session_id="S", conversation_id="conversation",
                             fresh_tail_count=6, leaf_chunk_tokens=400,
                             large_output_externalization_enabled=False,
                             large_output_active_replay_stubbing_enabled=False,
                             temporal_rollups_enabled=False, embeddings_enabled=False,
                             deferred_maintenance_enabled=False)
        engines.append(engine)
        return engine

    yield create
    for engine in engines:
        engine.shutdown()


def run(build, caplog, *, mode: str, uids: bool, restamp: bool) -> tuple[Any, list[Row]]:
    engine = build(rotating=mode == "rotating")
    hist = history(uids)
    engine.ingest(deepcopy(hist))
    user_t = {"role": "user", "content": T, "timestamp": T_REQ}
    reply_t = {"role": "assistant", "content": REPLY, "timestamp": T_REPLY}
    if uids:
        user_t["message_uid"], reply_t["message_uid"] = "a" * 32, "b" * 32
    # 1. the host writes T at the compress request; LCM ingests it
    before = hist + [user_t]
    engine.ingest(deepcopy(before))
    # 2. manual compress over that list; the reply lands while it runs
    output = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    engine.ingest(deepcopy(before + [reply_t]))
    # 3. the compaction commits (in place: same id; ACP: rotation into a child)
    sid = engine._session_id
    engine.on_session_end(sid, deepcopy(before))
    engine.on_session_start("rotated" if mode == "rotating" else sid, boundary_reason="compression",
                            old_session_id=sid, conversation_id=engine._conversation_id, platform="cli")
    view = deepcopy(output)
    carried = [row for row in view if row.get("role") == "user" and row.get("content") == T]
    assert len(carried) == 1, "compress output must carry the in-flight user row"
    if restamp:
        carried[0]["timestamp"] = T_COMMIT  # the host re-persists T at commit time
    view.append(dict(reply_t))
    # 4. the next turn
    view += [{"role": "assistant", "content": "steps again: build, push, restart", "timestamp": 2161.0},
             {"role": "user", "content": "thanks, next question", "timestamp": 2170.0}]
    caplog.clear()
    with caplog.at_level(logging.INFO):
        engine.ingest(deepcopy(view))
    for record in caplog.records:
        print("LOG", record.levelname, record.getMessage()[:200])
    rows = all_rows(engine)
    for row in rows[-12:]:
        print(row["store_id"], row["session_id"], row["role"], repr(str(row["content"])[:40]), row["observed_at"])
    return engine, rows


@pytest.mark.parametrize("uids", [False, True], ids=["no-uid", "same-uid"])
@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_inflight_turn_restamped_at_commit_is_not_a_silent_duplicate(build, caplog, mode, uids):
    engine, rows = run(build, caplog, mode=mode, uids=uids, restamp=True)
    copies = [row for row in rows if row["role"] == "user" and row["content"] == T]
    links = replay_links(engine)
    unlinked = [row for row in copies[1:] if (row["store_id"], copies[0]["store_id"]) not in links]
    assert not unlinked, (
        f"T stored {len(copies)}x with no possible_replay_of link: "
        f"{[(r['store_id'], r['session_id'], r['observed_at']) for r in copies]}")


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_control_kept_stamp_stores_once(build, caplog, mode):
    _engine, rows = run(build, caplog, mode=mode, uids=False, restamp=False)
    assert sum(row["role"] == "user" and row["content"] == T for row in rows) == 1
