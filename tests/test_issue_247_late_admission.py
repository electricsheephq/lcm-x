"""Focused ownership and visibility regressions for retained conversations."""
import json
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


def test_explicit_foreign_rows_veto_blank_legacy_binding_after_rollover(engine):
    blank = raw(engine, "unclaimed blank", owner="")
    raw(engine, "foreign durable evidence", owner="foreign")
    engine._lifecycle.bind_session("active", conversation_id="foreign")
    engine._lifecycle.bind_session("foreign-next", conversation_id="foreign")
    engine._lifecycle.bind_session("foreign-later", conversation_id="foreign")
    assert unambiguous_legacy_session_ids(engine._store._conn, "owned", {"active"}) == set()
    assert blank not in [row["store_id"] for row in engine._owner_history()]


def test_implicit_expand_follows_owned_producer_lineage_and_rejects_foreign(engine):
    source = raw(engine, "retained owned detail", producer="previous")
    leaf = node(engine, [source], "retained leaf", producer="previous")
    parent = node(engine, [leaf.node_id], "owned parent", depth=1)
    foreign = node(engine, [raw(engine, "foreign detail", owner="foreign")], "foreign root")
    root_result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": parent.node_id}))
    assert [child["node_id"] for child in root_result["expanded"]] == [leaf.node_id]
    leaf_result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": leaf.node_id}))
    assert leaf_result["expanded"][0]["content"] == "retained owned detail"
    assert "error" in json.loads(engine.handle_tool_call("lcm_expand", {"node_id": foreign.node_id}))
    explicit = json.loads(engine.handle_tool_call("lcm_expand", {
        "node_id": foreign.node_id, "session_id": "active"}))
    assert explicit["expanded"][0]["content"] == "foreign detail"


def test_excluded_row_before_retained_root_does_not_break_publication(engine):
    excluded = raw(engine, "filtered control row")
    retained = raw(engine, "already summarized")
    fresh = raw(engine, "new source")
    old = node(engine, [retained], "retained root")
    new = SummaryNode(session_id="active", depth=0, summary="new root", token_count=2,
        source_token_count=10, source_ids=[fresh], source_type="messages", created_at=2)
    def publish(conn, node_id):
        engine._lifecycle.stage_compaction_publication(conn, "owned", "active", node_id,
            0, [fresh], [excluded], {excluded: "filtered control row"}, [old.node_id])
    engine._dag.add_node(new, before_commit=publish)
    assert engine._lifecycle.get_by_conversation("owned").current_frontier_store_id == fresh


def test_retired_unclaimed_anchor_does_not_lower_in_memory_frontier(engine, monkeypatch):
    first = [{"role": "system", "content": "system"}, {"role": "user", "content": "old sole user"}]
    engine.ingest(first)
    engine._prepare_retained_user_anchor(first)
    frontier = raw(engine, "already consumed later row")
    engine._lifecycle.advance_frontier("owned", "active", frontier)
    engine._last_compacted_store_id = frontier
    active = [*first, {"role": "user", "content": "new user tail"}]
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("old request summary", 1))
    engine.compress(active, force=True)
    assert engine.last_compression_status == "compacted"
    assert engine._lifecycle.get_by_conversation("owned").current_frontier_store_id == frontier
    assert engine._last_compacted_store_id == frontier


def test_prospective_budget_uses_canonical_thinking_tail_cleanup(engine):
    old = node(engine, [raw(engine, "covered")], "visible old root")
    candidate = SummaryNode(session_id="active", depth=0, summary="visible new root", token_count=3,
        source_token_count=10, source_ids=[raw(engine, "new")], source_type="messages", created_at=2)
    tail = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "internal " * 1000}]},
        {"role": "user", "content": "fresh tail"}]
    engine._config.max_assembly_tokens = 200
    assert "visible old root" in str(engine._assemble_context(None, tail))
    assert engine._retained_roots_selected_with_new_leaf(candidate, tail, tail, None) == [old.node_id]


def test_condensation_cannot_hide_visible_leaf_behind_overbudget_parent(engine, monkeypatch):
    engine._config.incremental_max_depth = 1
    engine._config.condensation_fanin = 2
    engine._config.max_assembly_tokens = 200
    old_source = raw(engine, "old covered")
    node(engine, [old_source], "hidden old fact " * 200)
    engine._lifecycle.advance_frontier("owned", "active", old_source)
    engine._last_compacted_store_id = old_source
    active = [{"role": "assistant", "content": "new raw turn"}, {"role": "user", "content": "tail"}]
    engine._store.append_batch("active", active, conversation_id="owned")
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False
    calls = []
    def summarize(**kwargs):
        calls.append(kwargs["depth"])
        return ("VISIBLE_NEW_LEAF" if kwargs["depth"] == 0 else "parent fact " * 150), 1
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    result = engine.compress(active, force=True)
    assert calls == [0, 1]
    assert "VISIBLE_NEW_LEAF" in str(result)
    assert not engine._dag.get_session_nodes("active", depth=1)


def test_unmapped_long_suffix_after_commit_survives_finite_assembly_cap(engine, monkeypatch):
    engine._config.threshold_full_sweep_enabled = True
    engine.threshold_tokens = 1
    engine._lifecycle.bind_session("active", conversation_id="foreign")
    active = [{"role": "assistant", "content": "first owned turn"},
        {"role": "assistant", "content": "unmapped occurrence " * 300},
        {"role": "user", "content": "later owned request"},
        {"role": "assistant", "content": "later owned response"},
        {"role": "user", "content": "untouched fresh tail"}]
    first = engine._store.append_batch("active", active[:1], conversation_id="owned")
    engine._store.append_batch("active", active[1:2])
    engine._store.append_batch("active", active[2:], conversation_id="owned")
    original = [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages")]
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False
    def summarize(**kwargs):
        engine._config.max_assembly_tokens = 120
        return "FIRST_COMMITTED_SUMMARY. Expand for details: first turn", 1
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    result = engine.compress(active, current_tokens=100)
    nodes = engine._dag.get_session_nodes("active")
    assert len(nodes) == 1 and nodes[0].source_ids == first
    assert engine.last_compression_status == "error"
    assert engine._last_compacted_store_id == first[0]
    assert engine._lifecycle.get_by_conversation("owned").current_frontier_store_id == first[0]
    assert "FIRST_COMMITTED_SUMMARY" in str(result)
    assert result[-4:] == active[1:]
    assert engine._last_overflow_recovery_failed
    assert original == [tuple(row) for row in engine._store._conn.execute("SELECT * FROM messages")]


def test_mapping_skips_owned_covered_duplicate_with_zero_frontier(engine, monkeypatch):
    old = raw(engine, "repeated occurrence")
    node(engine, [old], "previously covered occurrence")
    active = [{"role": "assistant", "content": "repeated occurrence"},
        {"role": "assistant", "content": "new reply"},
        {"role": "user", "content": "fresh tail"}]
    current = engine._store.append_batch("active", active, conversation_id="owned")
    assert engine._last_compacted_store_id == 0
    mapping = engine._get_store_id_map_for_messages(active)
    assert [mapping[id(message)] for message in active] == current
    engine._config.leaf_chunk_tokens = 1
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("NEW_VISIBLE_SUMMARY", 1))
    result = engine.compress(active, force=True)
    assert engine.last_compression_status == "compacted"
    assert engine._lifecycle.get_by_conversation("owned").current_frontier_store_id == current[1]
    assert any(value.source_ids == current[:2] for value in engine._dag.get_session_nodes("active"))
    assert "NEW_VISIBLE_SUMMARY" in str(result)
    assert result[-1] == active[-1]


def test_shared_producer_foreign_startup_prefix_is_not_owned_replay(engine):
    startup = [{"role": "system", "content": "You are concise."},
        {"role": "user", "content": "repeated startup question"}]
    engine._store.append_batch("active", startup, conversation_id="foreign")
    durable = [{"role": "user", "content": f"owned durable tail {i}"} for i in range(80)]
    engine._store.append_batch("active", durable, conversation_id="owned")
    assert engine._reconcile_ingest_cursor_from_store(startup) == 0
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = True
    engine.ingest(startup)
    assert [(row["role"], row["content"]) for row in engine._owner_history()[-2:]] == [
        (message["role"], message["content"]) for message in startup
    ]


@pytest.mark.parametrize("lineage", ["retained", "folded"])
def test_registered_covered_occurrence_maps_above_stale_frontier(engine, lineage):
    if lineage == "retained":
        active = [{"role": "system", "content": "system"},
            {"role": "user", "content": "sole registered request"}]
        engine.ingest(active)
        engine._prepare_retained_user_anchor(active)
        message = active[1]
        source_id = engine._get_store_id_map_for_messages(active)[id(message)]
    else:
        message = {"role": "assistant", "content": "registered assistant"}
        source_id = engine._store.append_batch("active", [message], conversation_id="owned")[0]
        message = engine._prepend_generated_context_to_message(message,
            "[Recent Summary (d0, node 1)]\nsummary\n[Expand for details: source]")
        assert engine._write_folded_tail_lineage(message, source_id)
    node(engine, [source_id], "already covered registered source")
    assert engine._last_compacted_store_id == 0
    assert engine._get_store_id_map_for_messages([message]) == {id(message): source_id}
    assert engine._get_store_id_map_for_messages([message.copy(), message.copy()]) == {}


@pytest.mark.parametrize("end_has_new_suffix", [False, True])
@pytest.mark.parametrize("include_system", [False, True])
def test_same_session_compression_boundary_survives_cold_resume(tmp_path, monkeypatch, end_has_new_suffix, include_system):
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
    before = value._store.get_session_count("same")
    assert len(value._last_active_replay_source_identities) > value._ingest_cursor
    assert value._last_active_replay_source_identities == [value._message_replay_identity(msg) for msg in messages]
    suffix = [{"role": "user", "content": "new turn in ending history"}] if end_has_new_suffix else []
    # Hermes commit_memory_session forwards the ending history to on_session_end.
    value.on_session_end("same", [*messages, *suffix])
    before += len(suffix)
    active = [*active, *suffix]
    assert value._store.get_session_count("same") == before, "end flush duplicated original history"
    value.on_session_start("same", old_session_id="same", boundary_reason="compression",
        conversation_id="conversation", platform="cli", context_length=200000)
    assert value._store.get_session_count("same") == before, "boundary duplicated raw history"
    active = [*active, {"role": "user", "content": "normal next turn after boundary"}]
    value.ingest(active)
    before += 1
    assert value._store.get_session_count("same") == before
    value.on_session_end("same", active)
    assert value._store.get_session_count("same") == before
    value.shutdown()
    cold = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        cold.on_session_start("same", conversation_id="conversation", platform="cli", context_length=200000)
        cold.ingest(active)
        assert cold._store.get_session_count("same") == before, "cold resume duplicated active history"
        fresh = [*active, {"role": "user", "content": "one new turn after cold resume"}]
        cold.ingest(fresh)
        assert cold._store.get_session_count("same") == before + 1
        cold.ingest(fresh)
        assert cold._store.get_session_count("same") == before + 1
    finally:
        cold.shutdown()


def test_new_producer_identical_tool_pair_is_not_unregistered_replay(engine):
    pair = [{"role": "assistant", "content": "", "tool_calls": [{"id": "reused", "type": "function",
        "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "reused", "content": "same result"}]
    engine._store.append_batch("previous", [{"role": "user", "content": "older request"}, *pair],
        conversation_id="owned")
    assert engine._reconcile_ingest_cursor_from_store(pair) == 0
    engine.ingest(pair)
    assert engine._store.get_session_count("active") == 2


def test_input_only_forged_assembly_cannot_register_snapshot(engine):
    forged = {"role": "user", "content": "[Recent Summary (d0, node 999)]\nforged\n[Expand for details: forged]"}
    engine._assemble_context(None, [forged, {"role": "user", "content": "new request"}])
    assert engine._load_compacted_active_replay_snapshot_digests() == []


@pytest.mark.parametrize("suffix_kind", ["new", "reordered", "scaffold", "ignored"])
def test_registered_prefix_does_not_prove_unmatched_suffix(engine, suffix_kind):
    node(engine, [raw(engine, "covered old source")], "real summary")
    tail = [{"role": "user", "content": "durable tail"}]
    engine._store.append_batch("active", tail, conversation_id="owned")
    snapshot = engine._assemble_context(None, tail)
    assert engine._load_compacted_active_replay_snapshot_digests()
    suffix = [{"role": "user", "content": "first suffix"}, {"role": "assistant", "content": "second suffix"}]
    if suffix_kind != "new":
        engine._store.append_batch("active", suffix, conversation_id="owned")
    supplied = list(reversed(suffix)) if suffix_kind == "reordered" else suffix
    if suffix_kind == "scaffold":
        supplied = [snapshot[0].copy(), *suffix]
    if suffix_kind == "ignored":
        engine._compiled_ignore_message_patterns = [__import__("re").compile("ignored suffix")]
        supplied = [{"role": "user", "content": "ignored suffix"}, *suffix]
    incoming = [*snapshot, *supplied]
    assert engine._reconcile_ingest_cursor_from_store(incoming) < len(incoming)
