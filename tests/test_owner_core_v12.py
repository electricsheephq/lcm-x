"""Small logical-owner regression surface, with producer-local replay."""
import pytest
import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    value = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0), hermes_home=str(tmp_path / "home"))
    value.on_session_start("producer", conversation_id="owned", platform="cli", context_length=200000)
    yield value
    value.shutdown()


def history():
    return [{"role": "user", "content": "request one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "request two"},
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "retained tail"}]


def test_compaction_cannot_map_foreign_copies(engine, monkeypatch):
    active = history()
    foreign = engine._store.append_batch("producer", [*active, {"role": "assistant", "content": "other history"}], conversation_id="foreign")
    blank = engine._store.append_batch("producer", [{"role": "assistant", "content": "unknown blank"}], conversation_id="")
    original = [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id")]
    engine.ingest(active)
    own = [row["store_id"] for row in engine._store.get_session_messages("producer") if row["conversation_id"] == "owned"]
    assert len(own) == len(active)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("Owned summary", 1))
    result = engine.compress(active, force=True)
    assert engine.last_compression_status == "compacted"
    nodes = engine._dag.get_session_nodes("producer")
    assert {sid for node in nodes for sid in node.source_ids} == set(own[:-1])
    assert result[-1] == active[-1]
    assert [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages WHERE store_id <= ? ORDER BY store_id", (blank[-1],))] == original
    assert engine._lifecycle.get_by_conversation("owned").current_frontier_store_id == own[-2]


def test_foreign_full_history_cannot_prove_current_replay(engine):
    active = history()
    engine._store.append_batch("producer", active, conversation_id="foreign")
    engine._ingest_cursor_needs_reconcile = True
    engine.ingest(active)
    assert sum(row["conversation_id"] == "owned" for row in engine._store.get_session_messages("producer")) == len(active)


def test_explicit_owned_other_producer_is_publication_source(engine):
    message = {"role": "assistant", "content": "owned prior producer source"}
    source = engine._store.append_batch("earlier", [message], conversation_id="owned")[0]
    assert engine._get_store_id_map_for_messages([message]).get(id(message)) == source
    node = SummaryNode(session_id="producer", depth=0, summary="owned prefix", token_count=2,
        source_token_count=10, source_ids=[source], source_type="messages", created_at=1)
    engine._dag.add_node(node, before_commit=lambda conn, node_id: engine._lifecycle.stage_compaction_publication(
        conn, "owned", "producer", node_id, 0, [source]))
    assert engine._lifecycle.get_by_conversation("owned").current_frontier_store_id == source
    assert engine._store.get(source)["session_id"] == "earlier"


def test_unmapped_required_occurrence_preserves_input(engine, monkeypatch):
    active = history()
    engine.ingest(active)
    original = engine._get_store_ids_for_messages
    def omit_required(messages):
        return original(messages)[1:]
    monkeypatch.setattr(engine, "_get_store_ids_for_messages", omit_required)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("summary", 1))
    assert engine.compress(active, force=True) == active
    assert engine.last_compression_status == "error"
    assert not engine._dag.get_session_nodes("producer")
    assert engine._lifecycle.get_by_conversation("owned").current_frontier_store_id == 0


def test_assembly_selects_only_valid_owned_lineage(engine):
    for producer, owner, text in [("producer", "foreign", "FOREIGN"), ("earlier", "owned", "OWNED")]:
        source = engine._store.append_batch(producer, [{"role": "assistant", "content": text}], conversation_id=owner)[0]
        engine._dag.add_node(SummaryNode(session_id=producer, depth=0, summary=text,
            token_count=2, source_token_count=10, source_ids=[source], source_type="messages", created_at=1))
    assembled = str(engine._assemble_context(None, []))
    assert "OWNED" in assembled and "FOREIGN" not in assembled


def test_nonblank_legacy_owner_is_exact_across_history_and_roots(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("producer", conversation_id="owned", platform="cli", context_length=200000)
        source = engine._store.append_batch("producer", [{"role": "assistant", "content": "legacy source"}], conversation_id="owned")[0]
        # Simulate a legacy raw import; current append normalizes owner values.
        with engine._store._conn:
            engine._store._conn.execute("UPDATE messages SET conversation_id = ? WHERE store_id = ?", (" owned ", source))
        engine._dag.add_node(SummaryNode(session_id="producer", depth=0,
            summary="legacy explicit owner", token_count=3, source_token_count=10,
            source_ids=[source], source_type="messages", created_at=1))
        assert engine._owner_history() == []
        assert engine._owned_summary_roots() == []
    finally:
        engine.shutdown()
