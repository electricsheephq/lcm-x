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


@pytest.mark.parametrize("include_system", [False])
def test_current_post_turn_replay(tmp_path, monkeypatch, include_system):
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
        fresh.append({"role": "assistant", "content": "one actual reply"})
        cold.ingest(fresh)
        counts["after_turn"] = cold._store.get_session_count("same")
        digest = cold._replay_snapshot_digest(fresh, require_lcm_system_note=False)
        registered = digest in cold._load_compacted_active_replay_snapshot_digests()
        cold.on_session_end("same", fresh)
        counts["after_turn_end"] = cold._store.get_session_count("same")
        cold.shutdown()
        reopened = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
        try:
            reopened.on_session_start("same", conversation_id="conversation", platform="cli", context_length=200000)
            cursor = reopened._reconcile_ingest_cursor_from_store(fresh)
            reopened.ingest(fresh)
            counts["after_turn_cold"] = reopened._store.get_session_count("same")
            print({"counts": counts, "active_length": len(fresh), "registered": registered, "cursor": cursor}, flush=True)
            assert counts["after_turn"] == counts["after_turn_end"] == before + 2
            assert counts["after_turn_cold"] == before + 2
        finally:
            reopened.shutdown()
    finally:
        cold.shutdown()



def _renewal_engine(tmp_path, monkeypatch):
    config = LCMConfig(database_path=str(tmp_path / "renewal.db"), fresh_tail_count=4,
        leaf_chunk_tokens=1, incremental_max_depth=0)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("same", conversation_id="owned", platform="cli", context_length=200000)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("owned summary", 1))
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(12)]
    active = engine.compress(history, force=True)
    engine.ingest(active)
    assert engine._owned_exact_snapshot_prefix(active)
    return engine, active


@pytest.mark.parametrize("kind", ["text", "tools"])
def test_renewal_persists_fresh_repeated_occurrences(tmp_path, monkeypatch, kind):
    engine, active = _renewal_engine(tmp_path, monkeypatch)
    pair = ([{"role": "user", "content": "repeat"}, {"role": "assistant", "content": "repeat"}]
        if kind == "text" else [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "same-call", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "same-call", "content": "same result"}])
    try:
        before = engine._store.get_session_count("same")
        for expected in [2, 4]:
            active = [*active, *pair]
            engine.ingest(active)
            assert engine._store.get_session_count("same") == before + expected
            digest = engine._replay_snapshot_digest(active, require_lcm_system_note=False)
            assert digest in engine._load_compacted_active_replay_snapshot_digests()
        engine._schedule_ingest_cursor_reconciliation()
        engine.ingest(active)
        assert engine._store.get_session_count("same") == before + 4
    finally:
        engine.shutdown()


@pytest.mark.parametrize("mode", ["unregistered", "foreign", "skipped", "lossy"])
def test_renewal_declines_unproven_append(tmp_path, monkeypatch, mode):
    import re
    class _FakeTimeoutPattern:
        def __init__(self, pattern):
            self.pattern = pattern
            self._compiled = re.compile(pattern)
        def search(self, text, *, timeout=None):
            assert timeout is not None
            return self._compiled.search(text)
    engine, active = _renewal_engine(tmp_path, monkeypatch)
    try:
        suffix = [{"role": "user", "content": "new suffix"}]
        if mode == "unregistered":
            engine._store.write_metadata_json([engine._active_replay_snapshot_metadata_key()], '{}')
        elif mode == "foreign":
            engine.on_session_start("same", conversation_id="foreign", platform="cli", context_length=200000)
        elif mode == "skipped":
            engine._compiled_ignore_message_patterns = [_FakeTimeoutPattern("new suffix")]
        else:
            suffix[0]["content"] = "[LCM sensitive redaction: name=password_assignment; chars=8]"
        before = engine._store.get_session_count("same")
        extended = [*active, *suffix]
        engine.ingest(extended)
        if mode == "skipped":
            assert engine._store.get_session_count("same") == before
        digest = engine._replay_snapshot_digest(extended, require_lcm_system_note=False)
        assert digest not in engine._load_compacted_active_replay_snapshot_digests()
    finally:
        engine.shutdown()


def test_renewal_metadata_and_rows_rollback_together(tmp_path, monkeypatch):
    import sqlite3
    engine, active = _renewal_engine(tmp_path, monkeypatch)
    try:
        before = engine._store.get_session_count("same")
        proofs = engine._load_compacted_active_replay_snapshot_digests()
        original = engine._real_user_scaffold_metadata_rows
        def metadata(message, store_id):
            return [*original(message, store_id), ("test-original-scaffold-proof", "preserved")]
        monkeypatch.setattr(engine, "_real_user_scaffold_metadata_rows", metadata)
        engine._store._conn.execute("CREATE TRIGGER reject_renewal BEFORE UPDATE ON metadata BEGIN SELECT RAISE(ABORT, 'synthetic rollback'); END")
        with pytest.raises(sqlite3.IntegrityError):
            engine._ingest_messages([*active, {"role": "user", "content": "new suffix"}])
        assert engine._store.get_session_count("same") == before
        assert engine._load_compacted_active_replay_snapshot_digests() == proofs
        assert engine._store._conn.execute("SELECT value FROM metadata WHERE key='test-original-scaffold-proof'").fetchone() is None
    finally:
        engine.shutdown()
