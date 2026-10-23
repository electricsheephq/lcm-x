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
        retained = next(row for row in reversed(copied) if row["role"] == "user")
        old_uid, old_timestamp = retained["message_uid"], retained["timestamp"]
        retained.update(timestamp=9999.0, message_uid="r2-new-retained-occurrence")
        assert old_uid != retained["message_uid"] and old_timestamp != retained["timestamp"]
        before = len(engine._store.get_session_messages("S"))
        proof = engine._compress_commit_proof
        had_stamps = "output_metadata" in proof
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
