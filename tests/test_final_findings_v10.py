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


def test_late_root_cannot_hide_admitted_new_leaf(engine, monkeypatch):
    from hermes_lcm.dag import SummaryDAG
    source = raw(engine, "already covered history")
    node(engine, [source], "oversized history " * 500)
    source = raw(engine, "independent late-admission history")
    engine._lifecycle.advance_frontier("owned", "active", source)
    engine._last_compacted_store_id = source
    engine._config.max_assembly_tokens = 220
    active = [{"role": "assistant", "content": "new raw source"}, {"role": "user", "content": "tail"}]
    engine.ingest(active)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("NEW_LEAF " * 12, 1))
    original = engine._dag.add_node
    concurrent = SummaryDAG(engine._config.database_path)
    inserted = []
    def insert(value, **kwargs):
        if kwargs.get("before_commit"):
            child = SummaryNode(session_id="active", depth=0, summary="late child " * 500,
                token_count=1000, source_token_count=2000, source_ids=[source], source_type="messages", created_at=2)
            concurrent.add_node(child)
            parent = SummaryNode(session_id="active", depth=1, summary="parent " * 150,
                token_count=count_tokens("parent " * 150), source_token_count=2000,
                source_ids=[child.node_id], source_type="nodes", created_at=2)
            concurrent.add_node(parent)
            inserted.append(parent.node_id)
        return original(value, **kwargs)
    monkeypatch.setattr(engine._dag, "add_node", insert)
    result = engine.compress(active, force=True)
    assert inserted, "new leaf must pass prospective admission before the interleaving"
    assert "NEW_LEAF" in str(result) or (
        engine._last_compacted_store_id == source and result[-2:] == active)


@pytest.mark.parametrize("refusal", ["lineage", "budget"])
def test_first_pass_refusal_keeps_preexisting_summary(engine, monkeypatch, refusal):
    covered = raw(engine, "covered history")
    node(engine, [covered], "PREEXISTING_SUMMARY")
    engine._lifecycle.advance_frontier("owned", "active", covered)
    engine._last_compacted_store_id = covered
    if refusal == "lineage":
        raw(engine, "ambiguous reply", producer="previous")
    else:
        engine._config.max_assembly_tokens = 200
        monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("oversized new summary " * 500, 1))
    raw_tail = [{"role": "assistant", "content": "ambiguous reply"},
        {"role": "user", "content": "fresh request"}]
    engine._store.append_batch("active", raw_tail, conversation_id="owned")
    active = engine._assemble_context(None, raw_tail)
    assert "PREEXISTING_SUMMARY" in str(active)
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False
    result = engine.compress(active, force=True)
    assert engine.last_compression_status == "error"
    assert result[-2:] == raw_tail
    assert engine._last_compacted_store_id == covered
    assert "PREEXISTING_SUMMARY" in str(result)


def test_first_pass_restore_does_not_resurrect_filtered_raw_or_duplicate_anchors(engine):
    node(engine, [raw(engine, "covered history")], "PREEXISTING_SUMMARY")
    system = {"role": "system", "content": "system"}
    tail = {"role": "user", "content": "remaining request"}
    source = engine._assemble_context(system, [tail])
    filtered = {"role": "assistant", "content": "filtered original raw"}
    source.insert(-1, filtered)
    working = [source[0], source[-1]]
    result = engine._preserve_rejected_compaction_context(working, source, 0)
    assert result == [source[0], source[1], tail]
    assert filtered not in result
    assert engine._preserve_rejected_compaction_context(result, source, 0) == result


def test_registered_duplicate_tail_extension_keeps_new_occurrence(engine, monkeypatch):
    messages = [{"role": "user", "content": "identical request"} for _ in range(2)]
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("covered first request", 1))
    snapshot = engine.compress(messages, force=True)
    assert engine.last_compression_status == "compacted"
    assert engine._store.get_session_count("active") == 2
    assert snapshot[-1] == messages[-1]
    config, home = engine._config, engine._hermes_home
    engine.shutdown()
    cold = LCMEngine(config=config, hermes_home=str(home))
    try:
        cold.on_session_start("active", conversation_id="owned", platform="cli", context_length=200000)
        cold.ingest([*snapshot, messages[-1].copy()])
        assert cold._store.get_session_count("active") == 3
    finally:
        cold.shutdown()


def test_blank_legacy_summary_survives_two_compression_rollovers(engine):
    source = raw(engine, "legacy retained history", owner="")
    leaf = node(engine, [source], "LEGACY_RETAINED_SUMMARY")
    assert leaf.node_id in {value.node_id for value in engine._owned_summary_roots()}
    for previous, current in [("active", "next"), ("next", "later")]:
        engine.on_session_start(current, old_session_id=previous, boundary_reason="compression",
            conversation_id="owned", platform="cli", context_length=200000)
        assert leaf.node_id in {value.node_id for value in engine._owned_summary_roots()}, current
    assert "LEGACY_RETAINED_SUMMARY" in str(engine._assemble_context(None, []))
    parent = node(engine, [leaf.node_id], "CONDENSED_LEGACY_SUMMARY", producer="later", depth=1)
    config, home = engine._config, engine._hermes_home
    engine.shutdown()
    cold = LCMEngine(config=config, hermes_home=str(home))
    try:
        cold.on_session_start("later", conversation_id="owned", platform="cli", context_length=200000)
        assert parent.node_id in {value.node_id for value in cold._owned_summary_roots()}
        assert source in {row["store_id"] for row in cold._owner_history()}
        assert cold._store.get(source)["session_id"] == "active"
        assert cold._store.get(source)["conversation_id"] == ""
    finally:
        cold.shutdown()


def test_new_producer_repeating_full_unregistered_history_is_persisted(engine):
    messages = [{"role": "user", "content": "same request"},
        {"role": "assistant", "content": "same answer"}]
    engine._store.append_batch("previous", messages, conversation_id="owned")
    assert engine._load_compacted_active_replay_snapshot_digests() == []
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = True
    engine.ingest(messages)
    assert engine._store.get_session_count("active") == 2


def test_tool_replay_preserves_current_producer_multiplicity(engine):
    repeated = pair("same-call")
    engine._store.append_batch("previous", [{"role": "user", "content": "older"}, *repeated], conversation_id="owned")
    engine._store.append_batch("active", repeated, conversation_id="owned")
    assert engine._reconcile_ingest_cursor_from_store([*repeated, *repeated]) == 2


def test_historical_blank_producer_cannot_be_claimed_by_foreign_binding(engine):
    raw(engine, "legacy owner evidence", owner="")
    for previous, current in [("active", "next"), ("next", "later")]:
        engine.on_session_start(current, old_session_id=previous, boundary_reason="compression",
            conversation_id="owned", platform="cli", context_length=200000)
    engine._lifecycle.bind_session("active", conversation_id="foreign")
    assert unambiguous_legacy_session_ids(engine._store._conn, "foreign", {"active"}) == set()


def test_new_binding_does_not_adopt_unbound_legacy_rows(engine):
    source = raw(engine, "unbound legacy", owner="", producer="forgotten")
    engine._lifecycle.bind_session("forgotten", conversation_id="owned")
    assert "forgotten" not in unambiguous_legacy_session_ids(engine._store._conn, "owned", {"forgotten"})
    assert source not in {row["store_id"] for row in engine._owner_history()}
