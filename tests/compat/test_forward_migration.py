"""A previous-GA writer's real store survives upgrade, repeat open and rollback."""

import hashlib
import json
import sqlite3

import pytest

from tests.compat import codec_adapter
from tests.compat.helpers import FLEET_POLICY, config_for, run_old

# One source for both sides: the old writer's reader and the current reader hash the same fields.
DIGEST = r'''
import hashlib as _hashlib
def message_digest(rows):
    fields = [(r.get("role"), r.get("content"), r.get("tool_call_id"), r.get("tool_calls"), r.get("tool_name"),
               r.get("observed_at")) for r in rows]
    return _hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()
'''

WRITE_AND_READ = f"fleet_policy = {FLEET_POLICY!r}\n" + DIGEST + r'''
if mode == "write":
    options.update(fresh_tail_count=4, leaf_chunk_tokens=1000, large_output_externalization_enabled=True)
    if data == "fleet":
        # The managed fleet profile policy.
        options.update(fleet_policy)
from hermes_lcm import command, rollup_builder
from datetime import datetime, timezone
rollup_builder.summarize_with_escalation = lambda *args, **kwargs: ("Synthetic daily rollup.", 1)
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    if mode == "write":
        for index in range(3):
            session, conversation = f"session-{index}", f"conversation-{index}"
            engine.on_session_start(session, platform="cli", context_length=200_000,
                                    conversation_id=conversation)
            view = []
            for turn in range(12):
                view.extend([
                    {"role": "user", "content": [{"type": "text", "text": f"Question {index}/{turn}: " + "detail " * 800}],
                     "timestamp": 100.0 + 2 * turn},
                    {"role": "assistant", "content": f"Answer {index}/{turn}: " + "explanation " * 500,
                     "timestamp": 101.0 + 2 * turn},
                ])
            view.extend([
                {"role": "user", "content": "Run the large report.", "timestamp": 130.0},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "report", "type": "function", "function": {"name": "report", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "report", "name": "report", "tool_name": "report", "content": "report data " * 5000},
                {"role": "assistant", "content": "The report is complete."},
            ])
            engine.ingest(view)
            engine.compress(view, current_tokens=100_000, force=True)
            # The default old reader externalizes summarizer input, so expose
            # the large result outside the protected tail on a second pass.
            view.extend([{"role": "user", "content": f"Follow-up {n}", "timestamp": 140.0 + n}
                         for n in range(6)])
            engine.compress(view, current_tokens=100_000, force=True)
            assert engine._dag.get_session_nodes(session), "healthy fake summary must produce a DAG"
            if engine._config.temporal_rollups_enabled:
                assert engine.drain_rollup_maintenance(timeout=10)
                node = engine._dag.get_session_nodes(session)[0]
                day = datetime.fromtimestamp(node.earliest_at or node.created_at, timezone.utc).date()
                result = command.handle_lcm_command(f"rollups rebuild day {day}", engine)
                assert "status: complete" in result, result
        # Exercise the old survival fit separately from successful compaction;
        # all durable message writes still go through public ingest/compress.
        engine.on_session_start("survival", platform="cli", context_length=2000, conversation_id="survival-conv")
        view = [{"role": "user", "content": "retain " * 2000, "timestamp": 200.0},
                {"role": "assistant", "content": "answer " * 2000, "timestamp": 201.0}]
        engine.ingest(view)
''' + codec_adapter.OLD_FIT + r'''
        assert fitted != view
        first_id = engine._store.get_session_messages("session-0")[0]["store_id"]
        assert engine._store.add_message_relations([[(first_id, "alt_stamp", None, None, 777.0)]]) == 1
    sessions = [f"session-{index}" for index in range(3)] + ["survival"]
    rows = {session: engine._store.get_session_messages(session) for session in sessions}
    assert all(rows.values())
    nodes = {session: len(engine._dag.get_session_nodes(session)) for session in sessions[:3]}
    assert all(nodes.values())
    assert list((Path(home).parent / "payloads").glob("*")), "large output must be externalized"
    assert engine._store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(json.dumps({"counts": {session: len(value) for session, value in rows.items()}, "nodes": nodes,
                      "digests": {session: message_digest(value) for session, value in rows.items()}}))
finally:
    engine.shutdown()
'''


def snapshot(directory, before_columns=None):
    with sqlite3.connect(directory / "lcm.db") as conn:
        # FTS shadow storage is derived; retain the virtual tables themselves.
        shadows = {row[1] for row in conn.execute("PRAGMA table_list") if row[2] == "shadow"}
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                  if row[0] not in shadows]
        schema = {table: [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')] for table in tables}
        hashes = {}
        for table, columns in (before_columns or schema).items():
            assert set(columns) <= set(schema[table]), table
            names = ", ".join(f'"{column}"' for column in columns)
            rows = conn.execute(f'SELECT {names} FROM "{table}" ORDER BY rowid').fetchall()
            if table == "metadata" and before_columns is not None:
                # The upgrade adds this derived-index marker; every old key/value
                # must still hash identically, and no other addition is allowed.
                assert rows.count(("fts_stem_state", "backfilling")) == 1
                rows.remove(("fts_stem_state", "backfilling"))
            # SQLite BLOBs need an explicit lossless JSON representation.
            encoded = json.dumps(rows, default=lambda value: {"blob_hex": value.hex()}).encode()
            hashes[table] = (len(rows), hashlib.sha256(encoded).hexdigest())
        nodes = conn.execute("SELECT node_id, source_type, source_ids FROM summary_nodes ORDER BY rowid").fetchall()
        ids = {row[0] for row in conn.execute("SELECT store_id FROM messages")}
        node_ids = {row[0] for row in nodes}
        assert nodes and hashes["summary_node_provenance"][0]
        assert conn.execute("SELECT count(*) FROM message_relations WHERE kind='alt_stamp'").fetchone()[0] >= 1
        for _, source_type, source_ids in nodes:
            source_ids = json.loads(source_ids)
            assert source_ids and set(source_ids) <= (ids if source_type == "messages" else node_ids)
        payloads = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (directory / "payloads").rglob("*") if path.is_file()}
        return schema, hashes, payloads


@pytest.mark.parametrize("overlay", ["default", "fleet"])
def test_previous_ga_forward_migration(tmp_path, old_tree, host_root, monkeypatch, overlay):
    from hermes_lcm import db_bootstrap
    from hermes_lcm.engine import LCMEngine

    old = old_tree("v0.26.2")
    original = run_old(old, tmp_path, host_root, WRITE_AND_READ, mode="write", data=overlay)
    before = snapshot(tmp_path)
    if overlay == "fleet":
        assert before[1].get("lcm_rollups", (0, ""))[0] > 0
        assert before[1].get("lcm_rollup_sources", (0, ""))[0] > 0
    assert before[1]["messages"][0] == sum(original["counts"].values()) and before[-1]
    # Keep the asynchronous derived-index backfill out of the reopen comparison.
    monkeypatch.setattr(db_bootstrap, "dispatch_background_stem_backfill", lambda *args: None)
    namespace = {"json": json}
    exec(DIGEST, namespace)
    first_open = None
    for opening in range(2):
        engine = LCMEngine(config=config_for(tmp_path), hermes_home=str(tmp_path / "home"))
        try:
            for index in range(3):
                engine.on_session_start(f"session-{index}", platform="cli", context_length=200_000,
                                        conversation_id=f"conversation-{index}")
                assert len(engine._dag.get_session_nodes(f"session-{index}")) == original["nodes"][f"session-{index}"]
            assert engine._store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            # Rows on disk are not enough: every session must stay readable through the current reader.
            for session, count in original["counts"].items():
                current = engine._store.get_session_messages(session)
                assert len(current) == count, f"{session}: the current reader returns {len(current)} of {count} messages"
                assert namespace["message_digest"](current) == original["digests"][session], \
                    f"{session}: the current reader returns different message content or provenance"
        finally:
            engine.shutdown()
        after = snapshot(tmp_path, before[0])
        for table, digest in before[1].items():
            assert after[1][table] == digest, f"upgrade changed old columns/rows in {table}"
        assert after[2] == before[2], "upgrade changed externalized payloads"
        assert all(set(columns) <= set(after[0][table]) for table, columns in before[0].items())
        if opening:
            assert after == first_open, "second open must not change the persisted shape or data"
        first_open = after
    # #919's floor concerns replay of NEW short projections, not opening a store:
    # this candidate has emitted none. test_rollback_v0262_store defines this path.
    assert run_old(old, tmp_path, host_root, WRITE_AND_READ) == original
    assert snapshot(tmp_path, before[0])[1:] == before[1:]
