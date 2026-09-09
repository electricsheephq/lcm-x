"""Focused ownership and visibility regressions for retained conversations."""
import pytest
import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import unambiguous_legacy_session_ids
from hermes_lcm.tokens import count_tokens


@pytest.fixture
def engine(tmp_path):
    value = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0),
        hermes_home=str(tmp_path / "home"))
    value.on_session_start("active", conversation_id="owned", platform="cli", context_length=200000)
    yield value
    value.shutdown()


def raw(engine, content, owner="owned", producer="active", role="assistant"):
    return engine._store.append_batch(producer, [{"role": role, "content": content}], conversation_id=owner)[0]


def node(engine, sources, text, producer="active", depth=0):
    value = SummaryNode(session_id=producer, depth=depth, summary=text,
        token_count=count_tokens(text), source_token_count=1000, source_ids=sources,
        source_type="nodes" if depth else "messages", created_at=1, expand_hint="turns")
    engine._dag.add_node(value)
    return value


def pair(call_id):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
        "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "same result"}]


def test_whitespace_legacy_row_cannot_be_skipped(engine):
    from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
    legacy = raw(engine, "unrepresented legacy", owner="")
    with engine._store._conn:
        engine._store._conn.execute("UPDATE messages SET conversation_id = ' ' WHERE store_id = ?", (legacy,))
    assert unambiguous_legacy_session_ids(engine._store._conn, "owned", {"active"}) == {"active"}
    fresh = raw(engine, "new represented row")
    candidate = SummaryNode(session_id="active", depth=0, summary="new leaf", token_count=2,
        source_token_count=10, source_ids=[fresh], source_type="messages", created_at=2)
    def publish(conn, node_id):
        engine._lifecycle.stage_compaction_publication(conn, "owned", "active", node_id,
            0, [fresh], [], {}, [])
    with pytest.raises(LifecyclePublicationConflictError):
        engine._dag.add_node(candidate, before_commit=publish)


def test_fresh_current_producer_duplicate_preserves_ambiguous_raw(engine, monkeypatch):
    old = raw(engine, "identical active occurrence", producer="previous")
    active = [{"role": "assistant", "content": "identical active occurrence"},
        {"role": "user", "content": "fresh retained request"}]
    engine.ingest(active)
    own = engine._store.get_session_messages("active")
    assert len(own) == 2
    fresh = own[0]["store_id"]
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("new leaf", 1))
    result = engine.compress(active, force=True)
    leaves = engine._dag.get_session_nodes("active")
    assert engine.last_compression_status == "error"
    assert result == active
    assert not leaves
    assert engine._last_compacted_store_id == 0
    assert engine._lifecycle.get_by_conversation("owned").current_frontier_store_id == 0
    assert fresh > old


def test_post_cursor_foreign_producer_pair_is_new(engine):
    prefix, repeated = pair("prefix"), pair("repeated")
    engine._store.append_batch("active", prefix, conversation_id="owned")
    engine._store.append_batch("previous", [*repeated, {"role":"assistant", "content":"old end"}], conversation_id="owned")
    active = [*prefix, {"role":"assistant", "content":"fresh interlude"},
        *repeated, {"role":"user", "content":"fresh request"}]
    assert engine._reconcile_ingest_cursor_from_store(active) == 2
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = True
    engine.ingest(active)
    assert engine._store.get_session_count("active") == 6


def test_publication_rechecks_concurrent_root_visibility(engine, monkeypatch):
    from hermes_lcm.dag import SummaryDAG
    old_sources = [raw(engine, "old source one"), raw(engine, "old source two")]
    old = [node(engine, [source], "oversized history " * 500) for source in old_sources]
    engine._lifecycle.advance_frontier("owned", "active", old_sources[-1])
    engine._last_compacted_store_id = old_sources[-1]
    engine._config.max_assembly_tokens = 180
    active = [{"role":"assistant", "content":"new raw source"}, {"role":"user", "content":"tail"}]
    engine.ingest(active)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("NEW_LEAF " * 40, 1))
    original = engine._dag.add_node
    concurrent = SummaryDAG(engine._config.database_path)
    def insert(value, **kwargs):
        if kwargs.get("before_commit"):
            parent = SummaryNode(session_id="active", depth=1, summary="parent " * 120,
                token_count=count_tokens("parent " * 120), source_token_count=2000,
                source_ids=[n.node_id for n in old], source_type="nodes", created_at=2)
            concurrent.add_node(parent)
        return original(value, **kwargs)
    monkeypatch.setattr(engine._dag, "add_node", insert)
    result = engine.compress(active, force=True)
    assert engine.last_compression_status != "compacted" or "NEW_LEAF" in str(result)


def test_all_ordered_duplicate_occurrences_map_unambiguously(engine):
    message = {"role": "assistant", "content": "same repeated content"}
    ids = engine._store.append_batch("active", [message, message.copy()], conversation_id="owned")
    active = [message.copy(), message.copy()]
    mapping = engine._get_store_id_map_for_messages(active)
    assert [mapping[id(value)] for value in active] == ids


def test_registered_current_tail_does_not_prove_old_owner_duplicate_extension(engine):
    old = [{"role": "user", "content": "current tail"},
        {"role": "assistant", "content": "repeated suffix"}]
    engine._store.append_batch("previous", old, conversation_id="owned")
    node(engine, [raw(engine, "covered history")], "owned summary")
    tail = [{"role": "user", "content": "current tail"}]
    engine._store.append_batch("active", tail, conversation_id="owned")
    snapshot = engine._assemble_context(None, tail)
    assert engine._load_compacted_active_replay_snapshot_digests()
    incoming = [*snapshot, old[1].copy()]
    assert engine._reconcile_ingest_cursor_from_store(incoming) < len(incoming)


@pytest.mark.parametrize("lineage", ["retained", "folded"])
def test_registered_occurrence_wins_over_uncovered_duplicate(engine, lineage):
    message = {"role": "user" if lineage == "retained" else "assistant", "content": "registered occurrence"}
    source = engine._store.append_batch("active", [message], conversation_id="owned")[0]
    if lineage == "retained":
        registered = engine._prepare_retained_user_anchor([{"role": "system", "content": "system"}, message])
        assert registered is not None and registered["store_id"] == source
    else:
        message = engine._prepend_generated_context_to_message(message, "[Recent Summary (d0, node 1)]\nsummary")
        assert engine._write_folded_tail_lineage(message, source)
    raw(engine, "registered occurrence", role="user" if lineage == "retained" else "assistant")
    assert engine._get_store_id_map_for_messages([message]) == {id(message): source}


def test_ordered_prefix_rules_out_older_producer_duplicate(engine):
    repeated = {"role": "assistant", "content": "same reply"}
    engine._store.append_batch("previous", [repeated], conversation_id="owned")
    active = [{"role": "user", "content": "distinct current request"}, repeated.copy()]
    current = engine._store.append_batch("active", active, conversation_id="owned")
    mapping = engine._get_store_id_map_for_messages(active)
    assert [mapping[id(message)] for message in active] == current
