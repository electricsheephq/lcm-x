"""The v0.28.1 reader ingests a candidate-written occurrence snapshot (I5)."""

import copy
import json

from tests.compat.helpers import open_engine, run_old
from tests.test_host_uid_engine_uids import _compress, _host


V0281 = "77b2b4f795fb6a4f84c55abf6d0441da4469971e"
OLD_INGEST = r'''
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    host = json.loads(data)
    before = engine._store.get_session_messages("S")
    proof = engine._durable_commit_proof_payload()
    assert isinstance(proof["output_metadata"], list)
    host.append({"role": "user", "content": "Synthetic row after v0.28.1 rollback.", "timestamp": 999.0})
    engine.ingest(host)
    first = engine._store.get_session_messages("S")
    engine.ingest(host)
    replay = engine._store.get_session_messages("S")
    columns = lambda rows: [(r["store_id"], r["session_id"], r["role"], r["content"]) for r in rows]
    assert columns(replay)[:len(before)] == columns(before)
    assert len(replay) == len(first)
    assert sum(r["content"] == host[-1]["content"] for r in replay) == 1
    assert engine._store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(json.dumps({"reader": "v0.28.1", "retained": len(before), "replay_added": 0, "integrity": "ok"}))
finally:
    engine.shutdown()
'''


def test_candidate_snapshot_opens_and_ingests_on_v0281(tmp_path, old_tree, host_root):
    old = old_tree("v0.28.1", {"v0.28.1": V0281})
    engine = open_engine(tmp_path)
    engine._config.fresh_tail_count = 4
    try:
        output = _compress(engine, _host())
        proof = engine._durable_commit_proof_payload()
        assert proof is not None and isinstance(proof.get("output_metadata"), list)
        snapshot = engine._store.read_metadata_json(engine._replay_snapshot_metadata_key("host_occurrence_snapshot"))
        assert snapshot["output_length"] == len(output)
    finally:
        engine.shutdown()
    result = run_old(old, tmp_path, host_root, OLD_INGEST, data=json.dumps(copy.deepcopy(output)))
    assert result["reader"] == "v0.28.1" and result["replay_added"] == 0 and result["integrity"] == "ok"
