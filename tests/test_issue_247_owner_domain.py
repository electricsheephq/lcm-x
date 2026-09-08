"""Logical ownership, rather than producer session, governs compaction."""
import pytest
import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.dag import SummaryNode
from hermes_lcm.lifecycle_state import unambiguous_legacy_session_ids
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError


@pytest.mark.parametrize("old_producer", ["active", "previous"])
def test_owned_history_compacts_among_foreign_producer_rows(tmp_path, monkeypatch, old_producer):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0),
        hermes_home=str(tmp_path / "home"))
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
        lambda **kw: ("Owned conversation. Expand for details: owned turns", 1))
    engine.on_session_start("active", conversation_id="owned", platform="cli", context_length=200000)
    active = [{"role": "user", "content": "first owned turn"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "owned-call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "content": "owned result", "tool_call_id": "owned-call"},
        {"role": "assistant", "content": "second owned turn"},
        {"role": "user", "content": "latest owned tail"}]
    try:
        foreign = engine._store.append_batch(old_producer,
            [{"role": "assistant", "content": "foreign turn"}], conversation_id="foreign")
        own = engine._store.append_batch(old_producer, active, conversation_id="owned")
        config = engine._config
        engine.shutdown()
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
        engine.on_session_start("active", conversation_id="owned", platform="cli", context_length=200000)
        before = [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id")]
        engine.ingest(active)
        after = [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id")]
        assert after == before, "restart ingestion duplicated retained owned occurrences"
        result = engine.compress(active, force=True)
        assert engine.last_compression_status == "compacted"
        nodes = engine._dag.get_session_nodes("active")
        assert {sid for node in nodes for sid in node.source_ids} == set(own[:-1])
        assert not set(foreign) & {sid for node in nodes for sid in node.source_ids}
        assert result[-1]["content"] == active[-1]["content"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("foreign_parent", [False, True])
def test_owned_prefix_root_is_retained_without_foreign_summary(tmp_path, monkeypatch, foreign_parent):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0),
        hermes_home=str(tmp_path / "home"))
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
        lambda **kw: ("New owned summary. Expand for details: turns", 1))
    engine.on_session_start("active", conversation_id="owned", platform="cli", context_length=200000)
    try:
        own_prefix = engine._store.append_batch("previous",
            [{"role": "assistant", "content": "covered own turn"}], conversation_id="owned")
        foreign = engine._store.append_batch("active",
            [{"role": "assistant", "content": "foreign turn"}], conversation_id="foreign")
        def node(sources, summary, depth=0, source_type="messages"):
            result = SummaryNode(session_id="active", depth=depth, summary=summary,
                token_count=10, source_token_count=20, source_ids=sources,
                source_type=source_type, created_at=1, expand_hint="turns")
            engine._dag.add_node(result)
            return result
        own_node = node(own_prefix, "OWNED_PREFIX_ROOT")
        foreign_node = node(foreign, "FOREIGN_ROOT_MUST_NOT_RENDER")
        if foreign_parent:
            node([own_node.node_id, foreign_node.node_id], "MIXED_ROOT_MUST_NOT_RENDER", 1, "nodes")
        active = [{"role": "assistant", "content": "new owned turn"},
                  {"role": "user", "content": "latest tail"}]
        own = engine._store.append_batch("active", active, conversation_id="owned")
        original = [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages")]
        engine._ingest_cursor = len(active)
        engine._ingest_cursor_needs_reconcile = False
        result = engine.compress(active, force=True)
        assert engine.last_compression_status == "compacted"
        rendered = str(result)
        assert "OWNED_PREFIX_ROOT" in rendered
        assert "FOREIGN_ROOT_MUST_NOT_RENDER" not in rendered
        assert "MIXED_ROOT_MUST_NOT_RENDER" not in rendered
        assert engine._last_compacted_store_id == own[0]
        assert original == [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages")]
        leaves = engine._dag.get_session_nodes("active", depth=0)
        assert sum(own_prefix[0] in leaf.source_ids for leaf in leaves) == 1
    finally:
        engine.shutdown()


def test_blank_ownership_requires_positive_unique_binding(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"))
    engine.on_session_start("active", conversation_id="owned", platform="cli")
    try:
        conn = engine._store._conn
        assert unambiguous_legacy_session_ids(conn, "owned", {"active", "never-bound"}) == {"active"}
        engine._lifecycle.bind_session("active", conversation_id="foreign")
        assert unambiguous_legacy_session_ids(conn, "owned", {"active"}) == set()
        blank = engine._store.append_batch("active", [{"role": "user", "content": "unknown owner"}])
        own = engine._store.append_batch("active", [{"role": "user", "content": "explicit owned"}], conversation_id="owned")
        mapped = engine._owner_history()
        assert [row["store_id"] for row in mapped] == own
        assert engine._store.get(blank[0]) is not None
    finally:
        engine.shutdown()


@pytest.mark.parametrize("threshold", [False, True])
def test_condensation_receives_only_owned_root_summaries(tmp_path, monkeypatch, threshold):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
        incremental_max_depth=1, condensation_fanin=2), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("active", conversation_id="owned", platform="cli")
    texts = []
    def summarize(**kwargs):
        texts.append(kwargs["text"])
        return "Owned combined summary. Expand for details: turns", 1
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    try:
        for owner in ("foreign", "owned", "owned"):
            sources = engine._store.append_batch("active",
                [{"role": "assistant", "content": owner}], conversation_id=owner)
            engine._dag.add_node(SummaryNode(session_id="active", depth=0,
                summary=owner, token_count=10, source_token_count=20,
                source_ids=sources, source_type="messages", created_at=sources[0], expand_hint=owner))
        if threshold:
            group = engine._select_threshold_sweep_condensation_group()
            assert len(group) == 2
            engine._condense_summary_nodes(group)
        else:
            assert engine._maybe_condense() == 1
        assert texts == ["owned\n\n---\n\nowned"]
        assert len(engine._summary_frontier_nodes()) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("selected_blank", [False, True])
def test_ambiguous_blank_rows_are_neither_claimed_nor_silently_consumed(tmp_path, monkeypatch, selected_blank):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0),
        hermes_home=str(tmp_path / "home"))
    engine.on_session_start("active", conversation_id="owned", platform="cli", context_length=200000)
    engine._lifecycle.bind_session("active", conversation_id="foreign")
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
        lambda **kw: ("Owned summary. Expand for details: turns", 1))
    try:
        blank_message = {"role": "assistant", "content": "unknown retained occurrence"}
        blank = engine._store.append_batch("active", [blank_message])
        active = [{"role": "assistant", "content": "explicit owned turn"},
                  {"role": "user", "content": "owned tail"}]
        own = engine._store.append_batch("active", active, conversation_id="owned")
        original = [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages")]
        if selected_blank:
            active.insert(0, blank_message)
        engine._ingest_cursor = len(active)
        engine._ingest_cursor_needs_reconcile = False
        result = engine.compress(active, force=True)
        if selected_blank:
            assert result == active
            assert engine.last_compression_status == "error"
            assert engine._last_compacted_store_id == 0
        else:
            assert engine.last_compression_status == "compacted"
            assert engine._last_compacted_store_id == own[0]
        assert original == [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages")]
        assert all(not set(blank) & set(node.source_ids) for node in engine._dag.get_session_nodes("active"))
        invalid = SummaryNode(session_id="active", depth=0, summary="unknown root",
            token_count=10, source_token_count=20, source_ids=blank,
            source_type="messages", created_at=1, expand_hint="unknown")
        engine._dag.add_node(invalid)
        with pytest.raises(LifecyclePublicationConflictError, match="retained summary lineage"):
            engine._lifecycle.stage_compaction_publication(engine._dag._conn,
                "owned", "active", invalid.node_id + 1, engine._last_compacted_store_id,
                own if selected_blank else own[1:], selected_retained_root_node_ids=[invalid.node_id])
    finally:
        engine.shutdown()
