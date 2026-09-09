"""Preserve already-proven legacy ownership before lifecycle pointers move."""
import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _legacy_engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "legacy.db"), fresh_tail_count=1,
                       leaf_chunk_tokens=1, incremental_max_depth=0)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    messages = [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"legacy synthetic turn {i}"} for i in range(5)]
    # Released append_batch omitted conversation_id; the old binding is the proof.
    engine._store.append_batch("A", messages)
    engine.on_session_start("A", conversation_id="owned", platform="cli", context_length=200000)
    return engine, messages


def test_legacy_summary_survives_two_rollovers_and_restart(tmp_path, monkeypatch):
    engine, messages = _legacy_engine(tmp_path)
    config = engine._config
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **kwargs: ("SYNTHETIC_LEGACY_SUMMARY", 1))
    try:
        engine.compress(messages, force=True)
        before = engine._store._conn.execute(
            "SELECT store_id,session_id,content FROM messages ORDER BY store_id").fetchall()
        assert len(before) == 5
        assert engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ''").fetchone()[0] == 5
        roots = {node.node_id for node in engine._owned_summary_roots()}
        assert roots
        for old, new in [("A", "B"), ("B", "C")]:
            engine.rollover_session(old, new, previous_messages=[], boundary_reason="compression",
                                    platform="cli", context_length=200000)
            assert {node.node_id for node in engine._owned_summary_roots()} == roots
        assert engine._store._conn.execute(
            "SELECT store_id,session_id,content FROM messages ORDER BY store_id").fetchall() == before
        assert engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = 'owned'").fetchone()[0] == 5
        engine.shutdown()
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
        engine.on_session_start("C", conversation_id="owned", platform="cli", context_length=200000)
        assert {node.node_id for node in engine._owned_summary_roots()} == roots
        assert any("SYNTHETIC_LEGACY_SUMMARY" in m.get("content", "")
                   for m in engine._assemble_context(None, []))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("conflict", ["binding", "explicit-owner"])
def test_ambiguous_legacy_owner_is_not_promoted(tmp_path, conflict):
    engine, _ = _legacy_engine(tmp_path)
    try:
        if conflict == "binding":
            engine._lifecycle.bind_session("A", conversation_id="foreign")
        else:
            engine._store.append_batch("A", [{"role": "user", "content": "foreign row"}],
                                       conversation_id="foreign")
        before = engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
        engine._lifecycle.finalize_session("owned", "A")
        assert engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall() == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("transition", ["bind", "finalize", "rollover"])
def test_owner_promotion_rolls_back_with_transition(tmp_path, transition):
    engine, _ = _legacy_engine(tmp_path)
    lifecycle = engine._lifecycle
    conn = lifecycle._conn
    before = lifecycle.get_by_conversation("owned")
    statements = []

    class FailingConnection:
        def __getattr__(self, name):
            return getattr(conn, name)
        def execute(self, sql, *args):
            normalized = " ".join(sql.split())
            statements.append(normalized)
            if normalized.startswith(("INSERT INTO lcm_lifecycle_state", "UPDATE lcm_lifecycle_state")):
                assert conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = 'owned'").fetchone()[0] == 5
                raise RuntimeError("synthetic transition failure")
            return conn.execute(sql, *args)

    lifecycle._conn = FailingConnection()
    try:
        with pytest.raises(RuntimeError, match="synthetic transition failure"):
            if transition == "bind":
                lifecycle.bind_session("B", conversation_id="owned")
            elif transition == "finalize":
                lifecycle.finalize_session("owned", "A")
            else:
                lifecycle.record_rollover("owned", old_session_id="A", new_session_id="B")
        assert statements[0] == "BEGIN IMMEDIATE"
        assert not conn.in_transaction
        assert lifecycle.get_by_conversation("owned") == before
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ''").fetchone()[0] == 5
    finally:
        lifecycle._conn = conn
        engine.shutdown()
