import pytest
import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

@pytest.mark.parametrize("include_system", [True, False])
def test_same_session_compression_boundary_survives_cold_resume(tmp_path, monkeypatch, include_system):
    config = LCMConfig(database_path=str(tmp_path / "same-session.db"),
        fresh_tail_count=80, leaf_chunk_tokens=1, incremental_max_depth=0)
    value = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    value.on_session_start("same", conversation_id="conversation", platform="cli", context_length=200000)
    messages = ([{"role": "system", "content": "system"}] if include_system else []) + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"ordinary turn {i}"}
        for i in range(120)
    ]
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("VISIBLE_BOUNDARY_SUMMARY", 1))
    active = value.compress(messages, force=True)
    assert value.last_compression_status == "compacted"
    assert len(active) > 64 and active[-80:] == messages[-80:]
    assert any(message.get("_compressed_summary") for message in active[:-80])
    assert not any(message.get("_compressed_summary") for message in active[-80:])
    before = value._store.get_session_count("same")
    counts = {"after_compress": before}
    # Hermes commit_memory_session forwards the ending history to on_session_end.
    value.on_session_end("same", messages)
    counts["after_end"] = value._store.get_session_count("same")
    value.on_session_start("same", old_session_id="same", boundary_reason="compression",
        conversation_id="conversation", platform="cli", context_length=200000)
    counts["after_boundary"] = value._store.get_session_count("same")
    value.shutdown()
    cold = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        cold.on_session_start("same", conversation_id="conversation", platform="cli", context_length=200000)
        cold.ingest(active)
        counts["after_cold"] = cold._store.get_session_count("same")
        print({"include_system": include_system, "counts": counts}, flush=True)
        assert counts == dict.fromkeys(counts, before), "lifecycle boundary duplicated history"
        fresh = [*active, {"role": "user", "content": "one new turn after cold resume"}]
        cold.ingest(fresh)
        assert cold._store.get_session_count("same") == before + 1
        cold.ingest(fresh)
        assert cold._store.get_session_count("same") == before + 1
    finally:
        cold.shutdown()

