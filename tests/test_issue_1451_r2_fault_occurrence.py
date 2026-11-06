"""#1451 r2: calibrate occurrence retention after proof publication failure."""

import copy
import json
import sqlite3

from hermes_lcm import engine as engine_module
from hermes_lcm import host_uid_emit
from tests.test_host_uid_engine_uids import _compress, _host
from tests.test_host_uid_shadow import _engine, _state_db


def test_publication_failure_keeps_new_retained_occurrence(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "off")
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **kwargs: ("Synthetic summary preserving the task.", 1))
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 4
    real_write = engine._store.write_metadata_json
    fired = []

    def fail_proof(keys, *args, **kwargs):
        result = real_write(keys, *args, **kwargs)
        if not fired and any(key.startswith("compaction_commit_proof:") for key in keys):
            fired.append(True)
            raise sqlite3.OperationalError("injected r2 proof publication failure")
        return result

    monkeypatch.setattr(engine._store, "write_metadata_json", fail_proof)
    try:
        host = _host()
        output = _compress(engine, host)
        assert engine._last_compression_status == "compacted"
        assert fired == [True]
        copied = copy.deepcopy(output)
        retained_index = max(i for i, row in enumerate(copied) if row["role"] == "user")
        retained = copied[retained_index]
        old_uid, old_timestamp = retained["message_uid"], retained["timestamp"]
        retained.update(timestamp=9999.0, message_uid="r2-new-retained-occurrence")
        assert old_uid != retained["message_uid"] and old_timestamp != retained["timestamp"]
        before = len(engine._store.get_session_messages("S"))
        proof = engine._compress_commit_proof
        had_stamps = "output_metadata" in proof
        assert proof.get("durable") is False, "the injected failure must leave the proof non-durable"
        assert had_stamps and len(proof["output_metadata"]) == len(proof["output"])
        assert proof["output_metadata"][retained_index].get("message_uid") == old_uid, \
            "the per-row stamps survive the failed publication"
        cursor = engine._ingest_cursor
        engine.ingest(copied)
        rows = engine._store.get_session_messages("S")
        new_uid_rows = engine._store._conn.execute(
            "SELECT store_id FROM host_uid_bindings WHERE uid=? AND kind='canonical'",
            (retained["message_uid"],),
        ).fetchall()
        result = dict(before=before, after=len(rows), new_uid_rows=len(new_uid_rows),
                      fault_fired=True, in_memory_stamps=had_stamps, cursor=cursor,
                      output_length=len(output), anchor="off")
        print("R2_CALIBRATION " + json.dumps(result, sort_keys=True))
        assert len(rows) == before + 1, result
        assert len(new_uid_rows) == 1, result
    finally:
        engine.shutdown()


def test_malformed_non_durable_proof_is_consulted_once(tmp_path, monkeypatch):
    """#1451 item 4 (from #1904): the one-shot mark must land on the stored proof, not on the
    container-checked copy, so a later cursor equal to len(output) cannot arm the gate again."""
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "off")
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **kwargs: ("Synthetic summary preserving the task.", 1))
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 4
    real_write = engine._store.write_metadata_json
    fired = []

    def fail_proof(keys, *args, **kwargs):
        result = real_write(keys, *args, **kwargs)
        if not fired and any(key.startswith("compaction_commit_proof:") for key in keys):
            fired.append(True)
            raise sqlite3.OperationalError("injected item 4 proof publication failure")
        return result

    monkeypatch.setattr(engine._store, "write_metadata_json", fail_proof)
    real_gate = engine_module._hermes_salvage_protected_positions
    armed = []

    def count_gate(messages):
        armed.append(len(messages))
        return real_gate(messages)

    monkeypatch.setattr(engine_module, "_hermes_salvage_protected_positions", count_gate)
    try:
        output = _compress(engine, _host())
        assert fired == [True]
        stored = engine._compress_commit_proof
        assert stored.get("durable") is False
        stored["output_metadata"] = [None] * len(stored["output"])  # fails the container check
        assert engine._ingest_cursor == len(stored["output"])
        # A rewritten prefix row: neither the content proof nor a remap holds, so the gate keeps the
        # proof and asks for a reconcile (the only branch that leaves the proof in place).
        rewritten = copy.deepcopy(output)
        first_user = next(row for row in rewritten if row["role"] == "user")
        first_user["content"] = "Item 4 rewritten prefix row the proof never saw."
        engine.ingest(rewritten)
        assert len(armed) == 1, "the first ingest after the compaction must consult the proof"
        kept = engine._compress_commit_proof
        assert kept is not None and kept.get("output") == stored["output"], "the remap-failed branch keeps the proof"
        assert kept.get("consulted") is True
        # A later reconcile that leaves the cursor at len(output) must not re-arm the gate.
        engine._ingest_cursor = len(stored["output"])
        engine._ingest_cursor_needs_reconcile = False
        engine.ingest(rewritten)
        assert len(armed) == 1, armed
    finally:
        engine.shutdown()
