"""Regression for rotation carry when the parent proof write fails."""

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _durable_commit_proof, _stub_summarizer, _turn


@pytest.mark.parametrize("fail_parent_write", [False, True], ids=["control", "parent-write-fails"])
def test_rotation_carry_after_parent_proof_write_failure(tmp_path, monkeypatch, fail_parent_write):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=6,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host = []
        for i in range(1, 13):
            host.extend(_turn(i))
            engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        pre = list(host)

        real_write = engine._store.write_metadata_json

        def failing_write(keys, serialized, **kwargs):
            if fail_parent_write and any("compaction_commit_proof" in key for key in keys):
                raise RuntimeError("simulated durable proof write failure")
            return real_write(keys, serialized, **kwargs)

        engine._store.write_metadata_json = failing_write
        compressed = engine.compress(list(host), force=True)
        engine._store.write_metadata_json = real_write
        assert engine._last_compression_status == "compacted"
        local_ranges = list(engine._compress_commit_proof["carry_ranges"])
        assert local_ranges
        assert (_durable_commit_proof(engine, "S0") is None) == fail_parent_write

        frontier = engine._last_compacted_store_id
        expected_ranges = engine._coalesce_compression_carry_ranges(
            (session_id, max(range_start, frontier), range_end)
            for session_id, range_start, range_end in local_ranges
            if range_end > frontier
        )
        carried_ids = {
            int(row[0])
            for row in engine._store._conn.execute(
                "SELECT store_id FROM messages WHERE session_id = 'S0' AND store_id > ?",
                (frontier,),
            )
        }
        engine.on_session_end("S0", pre)
        engine.on_session_start("S1", boundary_reason="compression", old_session_id="S0", platform="acp")
        child_proof = _durable_commit_proof(engine, "S1")
        assert child_proof is not None
        assert child_proof["carry_ranges"] == [list(carry_range) for carry_range in expected_ranges]
        assert engine._load_compression_carry_ranges() == expected_ranges

        engine._config.fresh_tail_count = 2
        engine.protect_last_n = 2
        child = list(compressed)
        child.extend(_turn(14))
        engine.ingest(child)
        assert engine._store.get_session_count("S1") == 2
        engine.compress(list(child), force=True)
        new_nodes = engine._dag.get_session_nodes("S1")

        assert engine._last_compression_status == "compacted"
        assert carried_ids <= {
            source_id for node in new_nodes for source_id in node.source_ids
        }
    finally:
        engine.shutdown()
