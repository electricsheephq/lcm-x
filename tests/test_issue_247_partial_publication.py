"""Retained legacy-root admission and partial-publication recovery."""
import pytest
import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def make_engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0),
        hermes_home=str(tmp_path / "home"))
    engine.on_session_start("producer", conversation_id="owned", platform="cli", context_length=200000)
    return engine


@pytest.mark.parametrize("summary,tokens", [("", 10), ("legacy invalid root", 0)])
def test_invalid_legacy_root_does_not_hide_valid_owned_root(tmp_path, summary, tokens):
    engine = make_engine(tmp_path)
    try:
        sources = engine._store.append_batch("producer", [
            {"role": "assistant", "content": "legacy turn"},
            {"role": "assistant", "content": "valid turn"}], conversation_id="owned")
        def add_node(text, token_count, source_ids, depth=0, source_type="messages"):
            node = SummaryNode(session_id="producer", depth=depth, summary=text,
                token_count=token_count, source_token_count=20, source_ids=source_ids,
                source_type=source_type, created_at=1, expand_hint="turns")
            engine._dag.add_node(node)
            return node
        invalid = add_node(summary, tokens, sources[:1])
        valid = add_node("VALID_RETAINED_SUMMARY", 10, sources[1:])
        add_node("invalid parent", 10, [invalid.node_id, valid.node_id], 1, "nodes")
        assert [node.node_id for node in engine._owned_summary_roots()] == [valid.node_id]
    finally:
        engine.shutdown()


def test_unmapped_second_pass_keeps_committed_summary_and_untouched_suffix(tmp_path, monkeypatch):
    engine = make_engine(tmp_path)
    engine._config.threshold_full_sweep_enabled = True
    engine.threshold_tokens = 1
    engine._lifecycle.bind_session("producer", conversation_id="other")
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
        lambda **kwargs: ("FIRST_COMMITTED_SUMMARY. Expand for details: first turn", 1))
    active = [{"role": "assistant", "content": "first owned turn"},
        {"role": "assistant", "content": "unmapped legacy occurrence"},
        {"role": "user", "content": "later owned request"},
        {"role": "assistant", "content": "later owned turn"},
        {"role": "user", "content": "untouched fresh tail"}]
    try:
        first = engine._store.append_batch("producer", active[:1], conversation_id="owned")
        engine._store.append_batch("producer", active[1:2])
        engine._store.append_batch("producer", active[2:], conversation_id="owned")
        original = [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages")]
        engine._ingest_cursor = len(active)
        engine._ingest_cursor_needs_reconcile = False
        result = engine.compress(active, current_tokens=100)
        nodes = engine._dag.get_session_nodes("producer")
        assert len(nodes) == 1 and nodes[0].source_ids == first
        assert engine._last_compacted_store_id == first[0]
        assert engine.last_compression_status == "error"
        assert "FIRST_COMMITTED_SUMMARY" in str(result)
        assert result[-4:] == active[1:]
        assert original == [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages")]
    finally:
        engine.shutdown()
