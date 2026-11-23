"""v0.30.2 can resume a store after hotfix assembly and recall (I1, I2, I11)."""

from contextlib import closing
from hashlib import sha256
import json
import shutil
import sqlite3

import pytest

from hermes_lcm import tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.compat.helpers import run_old
from tests.invariants.test_hotfix_stored_summary_head import NEW_TURN, _append_bound_carrier, _view
from tests.invariants.test_hotfix_stored_summary_head import _append as _append
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_issue_671_stub_first_exit import THRESHOLD, run, stop_line, summaries, text_view, tool_view  # noqa: F401


PINS = {"v0.30.2": "f77d6ed98c8f2e896851b31720876e4c9d10f7b4"}
BINDING = {"session_id": "hf2", "conversation_id": "hf2", "context_length": 200_000}
GREP = {"query": "Earlier turns", "limit": 20}
ROLLBACK_TURN = [
    {"role": "user", "content": "Please verify the opening balance.", "message_uid": "rollback-user"},
    {"role": "assistant", "content": "The opening balance matches.", "message_uid": "rollback-assistant"},
]
OLD_RESUME = r'''
import os
from hashlib import sha256
from unittest.mock import Mock

from hermes_lcm.config import host_message_uid_mode
from hermes_lcm.tools import lcm_grep

if mode == "on":
    os.environ["LCM_HOST_MESSAGE_UID"] = "on"
assert host_message_uid_mode() == ("on" if mode == "on" else "shadow")
inputs = json.loads(data)
engine = LCMEngine(config=LCMConfig(database_path=db, **inputs["settings"]), hermes_home=inputs["home"])
try:
    engine.on_session_start(**inputs["binding"])
    conn = engine._store.connection
    # Both runs create relations with the same clock, while preserving their full rows and group keys.
    relation_time = conn.execute("SELECT MAX(created_at) FROM message_relations").fetchone()[0] + 1
    store_mod.time = Mock(wraps=store_mod.time)
    store_mod.time.time.return_value = relation_time
    view = inputs["view"]
    engine.ingest(view)
    assembled = engine.compress(view, current_tokens=tokens.count_messages_tokens(view), force=True)
    hits = json.loads(lcm_grep(inputs["grep"], engine=engine))["results"]
    report = {
        "messages": [
            [store_id, role, sha256(content.encode()).hexdigest()]
            for store_id, role, content in conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id")
        ],
        "relations": conn.execute("SELECT * FROM message_relations ORDER BY relation_id").fetchall(),
        "assembled": [[row["role"], sha256(row.get("content", "").encode()).hexdigest()] for row in assembled],
        "grep_store_ids": [hit["store_id"] for hit in hits if "store_id" in hit],
        "integrity": [row[0] for row in conn.execute("PRAGMA integrity_check")],
        "compression_status": engine.last_compression_status,
    }
finally:
    engine.shutdown()
print(json.dumps(report))
'''


def _seed_copy(engine, caplog):
    """Reproduce the stored carrier with the same stub-first input as the invariant cell."""
    engine.on_session_start(**BINDING)
    engine.threshold_tokens = THRESHOLD
    seed = text_view()[1:]
    seed[0]["content"] = "Earlier turns. " + seed[0]["content"]
    first = run(engine, seed, caplog)
    head_input = {**first[0], "content": first[0]["content"] + "\n\n---\n\n" + first[1]["content"]}
    supplied = tool_view(prefix=[head_input, *first[2:]])[1:]
    returned = run(engine, supplied, caplog)
    assert stop_line(caplog).group(1) == "stub_first_exit"
    head = returned[0]
    assert len(returned) == len(supplied) and head["content"] != head_input["content"]
    remainder = engine._generated_context_carrier_remainder(head)
    assert remainder and any(
        row["content"] == remainder for row in engine._store.get_session_messages(engine._session_id)
    )
    flush = _host_flush()
    host_view = flush(returned + NEW_TURN)
    previous_max = engine._store.max_store_id()
    engine.ingest(host_view)
    assert all(
        row["content"] != head["content"]
        for row in engine._store.get_session_messages_after(engine._session_id, previous_max)
    )
    copy_id = _append_bound_carrier(engine, head)
    return head["content"], copy_id, flush(host_view + ROLLBACK_TURN)


def _snapshot(path):
    queries = {
        "messages": "SELECT * FROM messages ORDER BY store_id",
        "message_relations": "SELECT * FROM message_relations ORDER BY relation_id",
        "metadata": "SELECT key, value FROM metadata ORDER BY key",
        "summary_nodes": "SELECT * FROM summary_nodes ORDER BY node_id",
        "schema": "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name",
    }
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
        return {name: conn.execute(sql).fetchall() for name, sql in queries.items()}


@pytest.mark.usefixtures("summaries")
@pytest.mark.parametrize("uid_mode", ["unset", "on"])
def test_v0302_rollback_after_reading_known_head_copy(tmp_path, old_tree, host_root, monkeypatch, uid_mode, caplog):
    if uid_mode == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)

    old = old_tree("v0.30.2", PINS)
    original, twin = tmp_path / "original", tmp_path / "twin"
    original.mkdir()
    twin.mkdir()
    db = original / "lcm.db"
    home = str(tmp_path / "home")
    settings = dict(
        embeddings_enabled=False,
        temporal_rollups_enabled=False,
        fresh_tail_count=2,
        leaf_chunk_tokens=1_000,
        fresh_tail_pressure_yield_enabled=False,
        threshold_full_sweep_enabled=False,
        l3_truncate_tokens=1,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1_000_000,
        large_output_active_replay_stubbing_enabled=True,
    )
    seed = LCMEngine(config=LCMConfig(database_path=str(db), **settings), hermes_home=home)
    try:
        head, copy_id, host_view = _seed_copy(seed, caplog)
    finally:
        seed.shutdown()  # Checkpoints WAL as it closes the SQLite helpers.
    shutil.copy2(db, twin / db.name)
    for suffix in ("-wal", "-shm"):
        sidecar = db.with_name(db.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, twin / sidecar.name)

    settings["large_output_active_replay_stubbing_enabled"] = False
    hotfix = LCMEngine(config=LCMConfig(database_path=str(db), **settings), hermes_home=home)
    try:
        hotfix.on_session_start(**BINDING)
        tail = _view(hotfix)
        assert any(row["store_id"] == copy_id for row in tail)
        assembled = hotfix._assemble_context(None, tail, persist=False)
        assert assembled and all(row.get("store_id") != copy_id for row in assembled)
        hits = json.loads(tools.lcm_grep(GREP, engine=hotfix))["results"]
        assert any(hit.get("store_id") for hit in hits)
        # v0.30.3 leaves recall unchanged: lcm_grep may still return the copy (recall exclusion follows in v0.30.4).
    finally:
        hotfix.shutdown()

    before = _snapshot(twin / db.name)
    after = _snapshot(db)
    for table, rows in before.items():
        assert after[table] == rows, f"hotfix assembly/grep changed {table}"
    with closing(sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)) as conn:
        original_rows = [
            [store_id, role, sha256(content.encode()).hexdigest()]
            for store_id, role, content in conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id")
        ]

    data = json.dumps(dict(settings=settings, home=home, binding=BINDING, view=host_view, grep=GREP))
    touched = run_old(old, original, host_root, OLD_RESUME, mode=uid_mode, data=data)
    control = run_old(old, twin, host_root, OLD_RESUME, mode=uid_mode, data=data)
    assert touched == control
    assert touched["integrity"] == ["ok"]
    assert touched["compression_status"] == "compacted"
    assert copy_id in touched["grep_store_ids"]
    assert touched["messages"][:len(original_rows)] == original_rows
    added = [row[1:] for row in touched["messages"][len(original_rows):]]
    for row in ROLLBACK_TURN:
        assert added.count([row["role"], sha256(row["content"].encode()).hexdigest()]) == 1
    # v0.30.2's own bug (the one this hotfix prevents) stores the head once more on its next ingest; it does so
    # equally on the touched store and the untouched twin (asserted by touched == control above), so the hotfix
    # leaves nothing that changes v0.30.2's behaviour. Pin the count so a change in v0.30.2's shape is visible.
    assert added.count(["user", sha256(head.encode()).hexdigest()]) == 1
