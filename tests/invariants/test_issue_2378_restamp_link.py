"""A turn re-stamped while a manual compress is in flight is stored twice and linked.

Host-view sequence (content-free evidence, ACP rotation 185a62de and TUI in-place c85ae0):
1. the host writes the user row T at the compress request (stamp t_req) and LCM ingests it;
2. compress() runs over the list ending in T; the assistant reply lands while it runs and is ingested;
3. the compaction commits (in place, or rotating into a child) and the host's compacted list carries
   T again with the commit-time stamp (Hermes' ``_insert_message_rows`` stamps ``now_ts`` when no
   active donor row matches), optionally with the same ``message_uid``;
4. the next turn is ingested.
LCM holds T twice (t_req and t_commit) with a ``possible_replay_of`` link. The control keeps
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

pytestmark = pytest.mark.issue(2378)

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


def run(build, caplog, *, mode: str, uids: bool, restamp: bool,
        prior_copies: int = 1, before_ingest=None) -> tuple[Any, list[Row]]:
    engine = build(rotating=mode == "rotating")
    hist = history(uids)
    engine.ingest(deepcopy(hist))
    user_t = {"role": "user", "content": T, "timestamp": T_REQ}
    reply_t = {"role": "assistant", "content": REPLY, "timestamp": T_REPLY}
    if uids:
        user_t["message_uid"], reply_t["message_uid"] = "a" * 32, "b" * 32
    # 1. the host writes T at the compress request; LCM ingests it
    before = hist + ([{**user_t, "timestamp": T_REQ - 2, "message_uid": "c" * 32},
                      {"role": "assistant", "content": "earlier steps", "timestamp": T_REQ - 1}]
                     if prior_copies == 2 else []) + [user_t]
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
    assert len(carried) == prior_copies, "compress output must carry the in-flight user rows"
    if restamp:
        for idx, row in enumerate(carried):
            row["timestamp"] = T_COMMIT + idx  # the host re-persists T at commit time
    view.append(dict(reply_t))
    # 4. the next turn
    view += [{"role": "assistant", "content": "steps again: build, push, restart", "timestamp": 2161.0},
             {"role": "user", "content": "thanks, next question", "timestamp": 2170.0}]
    caplog.clear()
    if before_ingest is not None:
        before_ingest(engine, view)
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
    assert len(copies) == 2
    assert [row["observed_at"] for row in copies] == [T_REQ, T_COMMIT]
    assert links == {(copies[1]["store_id"], copies[0]["store_id"])}


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_control_kept_stamp_stores_once(build, caplog, mode):
    _engine, rows = run(build, caplog, mode=mode, uids=False, restamp=False)
    assert sum(row["role"] == "user" and row["content"] == T for row in rows) == 1
    assert replay_links(_engine) == set()


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_different_user_turn_is_never_linked(build, caplog, mode):
    different = "please cancel staging and deploy a different service"

    def replace(_engine, view):
        next(row for row in view if row.get("content") == T)["content"] = different

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=replace)
    assert sum(row["content"] == T for row in rows) == 1
    assert sum(row["content"] == different for row in rows) == 1
    assert replay_links(engine) == set()


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
@pytest.mark.parametrize("prior_copies", [1, 2])
def test_restamp_links_are_one_to_one(build, caplog, mode, prior_copies):
    def duplicate(_engine, view):
        if prior_copies == 1:
            idx = next(idx for idx, row in enumerate(view) if row.get("content") == T)
            # Retain the published prefix so its commit proof remains usable.
            view.append({**view[idx], "timestamp": T_COMMIT + 1})

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True,
                       prior_copies=prior_copies, before_ingest=duplicate)
    copies = [row for row in rows if row["content"] == T]
    assert len(copies) == prior_copies + 2
    links = replay_links(engine)
    assert links == {(copies[prior_copies + idx]["store_id"], copies[idx]["store_id"])
                     for idx in range(prior_copies)}
    assert len({target for _source, target in links}) == len(links)


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
def test_relation_write_failure_still_stores(build, caplog, monkeypatch, mode):
    attempted = []

    def inject(engine, _view):
        write = engine._store.add_message_relations

        def fail(groups):
            result = write(groups)
            if any(relation[1] == "possible_replay_of" for group in groups for relation in group):
                assert engine._store._conn.in_transaction
                assert engine._store._transaction_owner.owns_connection(engine._store._conn)
                attempted.append(True)
                raise RuntimeError("injected relation write failure after SQL")
            return result

        monkeypatch.setattr(engine._store, "add_message_relations", fail)

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=inject)
    assert attempted == [True]
    assert engine._ingest_failure_count == 0
    assert sum(row["content"] == T for row in rows) == 2
    assert replay_links(engine) == set()
    assert "re-stamped occurrence link write failed" in caplog.text


def test_rotation_without_snapshot_reuses_durable_recovery(build, caplog, monkeypatch):
    called = []

    def remove_snapshot(engine, _view):
        key = engine._replay_snapshot_metadata_key("host_occurrence_snapshot", "S")
        engine._store._conn.execute("DELETE FROM metadata WHERE key=?", (key,))
        engine._store._conn.commit()
        link = engine._identity_anchor_link_durable_recovery

        def spy(attempt, lineage):
            called.append(True)
            return link(attempt, lineage)

        monkeypatch.setattr(engine, "_identity_anchor_link_durable_recovery", spy)

    engine, rows = run(build, caplog, mode="rotating", uids=False, restamp=True,
                       before_ingest=remove_snapshot)
    assert called
    assert sum(row["content"] == T for row in rows) == 2
    assert engine._ingest_failure_count == 0


def test_cold_inplace_restamp_is_stored_and_linked(build, caplog):
    def rebind(engine, _view):
        engine.on_session_start("S", conversation_id=engine._conversation_id, platform="cli")

    engine, rows = run(build, caplog, mode="inplace", uids=False, restamp=True, before_ingest=rebind)
    copies = [row for row in rows if row["content"] == T]
    assert len(copies) == 2
    assert replay_links(engine) == {(copies[1]["store_id"], copies[0]["store_id"])}


@pytest.mark.parametrize("mode", ["inplace", "rotating"])
@pytest.mark.parametrize("target_count", [0, 2])
def test_missing_or_ambiguous_prior_target_stores_without_link(build, caplog, monkeypatch, mode, target_count):
    def change_targets(engine, _view):
        lookup = engine._store.find_rows_by_observed_at
        if target_count == 2:
            engine._store.append("S", {"role": "user", "content": T, "timestamp": T_REQ},
                                 conversation_id=engine._conversation_id)

        def find(conversation, sessions, stamps):
            rows = lookup(conversation, sessions, stamps)
            return [row for row in rows if target_count != 0 or row["observed_at"] != T_REQ]

        monkeypatch.setattr(engine._store, "find_rows_by_observed_at", find)

    engine, rows = run(build, caplog, mode=mode, uids=False, restamp=True, before_ingest=change_targets)
    assert sum(row["content"] == T for row in rows) == (3 if target_count == 2 else 2)
    assert replay_links(engine) == set()
    assert engine._ingest_failure_count == 0
