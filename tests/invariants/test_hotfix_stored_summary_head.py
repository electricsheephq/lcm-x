"""A surplus summary head remains durable without entering model context (I1, I2, I11)."""

import json

import pytest

from hermes_lcm import host_uid_emit, ingest_ledger, tools
from hermes_lcm.tokens import count_message_tokens
from tests.invariants.test_head_copy_subset import _base_is_known_head_copy, head_copy_subset  # noqa: F401
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_issue_2854_no_summary_pass_unchanged import open_engine  # noqa: F401
from tests.test_issue_671_stub_first_exit import (
    THRESHOLD,
    TURN,
    exit_engine,
    make_engine,  # noqa: F401
    run,
    stop_line,
    summaries,  # noqa: F401
    text_view,
    tool_view,
)

pytestmark = pytest.mark.usefixtures("head_copy_subset")

RELATION = "generated_unproven"
NEW_TURN = [
    {"role": "user", "content": "Please compare the two accounts."},
    {"role": "assistant", "content": "Both accounts reconcile."},
]


@pytest.fixture(params=["default", "on"])
def uid_mode(request, monkeypatch):
    if request.param == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)


def _append(engine, row, relation=None, *, session=None):
    store_id, = engine._store.append_batch(
        session or engine._session_id, [row], conversation_id=engine._conversation_id,
    )
    if relation:
        engine._store.add_message_relations([[(store_id, relation, None, None, None)]])
    return store_id


def _append_bound_carrier(engine, row, *, source_id=None, session=None):
    """Seed a replay with a matching host stamp or the v0.30.2 ledger placement."""
    store = engine._store
    if source_id is not None:
        stamp = store.get(source_id)["observed_at"]
        assert stamp is not None
        return _append(engine, {**row, "timestamp": stamp}, RELATION, session=session)
    conn = store.connection
    sid = engine._session_id
    lineage = ingest_ledger.persisted_lineage(conn, sid, ingest_ledger.lineage_key(None, sid))
    remainder = engine._generated_context_carrier_remainder(row)
    own_id, = [r["store_id"] for r in store.get_session_messages(sid) if r["content"] == remainder]
    right = conn.execute(
        "SELECT min(b.ordinal) FROM occurrence_bindings b JOIN messages m USING (store_id) "
        "WHERE b.lineage=? AND m.session_id=? AND m.store_id>?", (lineage, sid, own_id),
    ).fetchone()[0]
    assert right is not None
    existing = ingest_ledger.lineage_ordinals(conn, lineage)
    left = max((ordinal for ordinal in existing if ordinal < right), default=None)
    ordinals, respace = ingest_ledger.plan_ordinals(existing, [ingest_ledger.Placement(left, right)])
    copy_id = _append(engine, row, RELATION)
    stored = store.get(copy_id)
    digest = ingest_ledger.identity_digest(store.ingest_ledger_salt(), stored)
    with store.atomic():
        ingest_ledger.apply_respace(conn, lineage, respace)
        ingest_ledger.insert_bindings(conn, lineage, [{
            "ordinal": ordinals[0], "store_id": copy_id, "identity_digest": digest,
            "stamp": ingest_ledger.stamp_text(stored["observed_at"]), "disposition": "stored",
            "view_seq": ingest_ledger.next_view_seq(conn),
        }])
    return copy_id


@pytest.fixture
def twins(uid_mode, make_engine, summaries, caplog, head_copy_subset):  # noqa: F811
    engines = []
    for with_copy in (False, True):
        engine = exit_engine(make_engine, threshold_full_sweep_enabled=False, l3_truncate_tokens=1)
        engine.on_session_start("hf2", conversation_id="hf2", context_length=200_000)
        engine.threshold_tokens = THRESHOLD
        seed = text_view()[1:]
        seed[0]["content"] = "Earlier turns. " + seed[0]["content"]
        first = run(engine, seed, caplog)
        head_input = {**first[0], "content": first[0]["content"] + "\n\n---\n\n" + first[1]["content"]}
        supplied = tool_view(prefix=[head_input, *first[2:]])[1:]
        returned = run(engine, supplied, caplog)
        assert stop_line(caplog).group(1) == "stub_first_exit"
        assert len(returned) == len(supplied) and returned[0]["content"] != head_input["content"]
        head = returned[0]
        remainder = engine._generated_context_carrier_remainder(head)
        assert remainder and any(
            r["content"] == remainder for r in engine._store.get_session_messages(engine._session_id)
        )
        old_top = engine._store.max_store_id()
        engine.ingest(_host_flush()(returned + NEW_TURN))
        assert not any(
            r["content"] == head["content"]
            for r in engine._store.get_session_messages_after(engine._session_id, old_top)
        )  # This synthetic warm ingest leaves the copy out. Seed the stored shape through the ingest writer.
        copy_id = _append_bound_carrier(engine, head) if with_copy else None
        if not with_copy:
            # Occupy the same global ID outside this conversation's session, so leaf IDs are directly comparable.
            _append(engine, NEW_TURN[0], session="unrelated")
        engine._config.large_output_active_replay_stubbing_enabled = False
        engines.append((engine, head, copy_id))
    return engines


def _view(engine):
    return [
        {**engine._store.to_openai_msg(row), "store_id": row["store_id"], "session_id": row["session_id"]}
        for row in engine._store.get_session_messages_after(engine._session_id, engine._last_compacted_store_id)
    ]


def _snapshot(engine):
    conn = engine._store.connection
    return (
        conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall(),
        conn.execute("SELECT * FROM message_relations ORDER BY relation_id").fetchall(),
    )


def test_assembly_matches_twin(twins):
    (clean, _, _), (copied, _, _) = twins
    assert copied._assemble_context(None, _view(copied), persist=False) == clean._assemble_context(
        None, _view(clean), persist=False
    )


def test_assembly_preserves_rows_and_relations(twins):
    _, (engine, head, copy_id) = twins
    before = _snapshot(engine)
    assert (copy_id, "user", head["content"]) in before[0]
    engine._assemble_context(None, _view(engine), persist=False)
    assert _snapshot(engine) == before


def test_consecutive_stored_carrier_copies_are_both_known(twins):
    from hermes_lcm.head_copy import is_known_head_copy

    _, (engine, head, first_id) = twins
    second_id = _append_bound_carrier(engine, head)
    assert second_id == first_id + 1
    for store_id in (first_id, second_id):
        assert is_known_head_copy(engine._store.connection, engine._store.get(store_id))


@pytest.mark.parametrize("position", ["leaf", "tail"])
def test_compaction_matches_twin(twins, summaries, caplog, position):  # noqa: F811
    results = []
    for engine, _, copy_id in twins:
        if position == "leaf":
            _append(engine, {"role": "user", "content": "One more unrelated question."})
            _append(engine, {"role": "assistant", "content": "One more reply."})
        before = _snapshot(engine)
        view = _view(engine)
        # These rows came from the store and are already durable. The host's loaded list is the input to compress.
        engine._ingest_cursor = len(view)
        engine._ingest_cursor_needs_reconcile = False
        engine._compress_commit_proof = None
        start = len(summaries)
        nodes_before = {node.node_id for node in engine._dag.get_session_nodes(engine._session_id)}
        result = run(engine, view, caplog, force=True)
        leaves = [
            node.source_ids for node in engine._dag.get_session_nodes(engine._session_id)
            if node.source_type == "messages" and node.node_id not in nodes_before
        ]
        assert engine.last_compression_status == "compacted"
        assert "lacks raw store lineage" not in engine.last_compression_noop_reason
        assert _snapshot(engine)[0] == before[0]
        assert copy_id is None or all(copy_id not in ids for ids in leaves)
        results.append((leaves, summaries[start:], result))
    assert results[0] == results[1]
    assert results[0][0] and results[0][1]


def test_search_preserves_original_hits_and_summary(twins):
    _, (engine, _, copy_id) = twins
    hits = json.loads(tools.lcm_grep({"query": "Earlier turns", "limit": 20}, engine=engine))["results"]
    assert all(hit.get("store_id") != copy_id for hit in hits)
    assert any(hit.get("store_id") for hit in hits)
    assert any(hit.get("node_id") for hit in hits)
    expanded = json.loads(tools.lcm_expand({"store_id": copy_id}, engine=engine))
    assert "error" in expanded
    loaded = json.loads(tools.lcm_load_session({"session_id": engine._session_id, "limit": 100}, engine=engine))
    assert all(row["store_id"] != copy_id for row in loaded["messages"])


@pytest.mark.usefixtures("uid_mode")
def test_budget_trimmed_tail_carrier_copy_is_excluded(make_engine, summaries, caplog):  # noqa: F811
    engine = exit_engine(make_engine, threshold_full_sweep_enabled=False)
    run(engine, text_view()[1:], caplog)
    oversized = {"role": "user", "content": "Earlier uncovered request." + TURN * 10}
    carried = {"role": "user", "content": "Please check the pending invoice.", "timestamp": 101}
    cap = 512
    assert count_message_tokens(oversized) > cap
    oversized_id = _append(engine, oversized)
    carried_id = _append(engine, carried)
    _append(engine, {"role": "assistant", "content": "The invoice is pending."})
    for row in NEW_TURN:
        _append(engine, row)

    emitted = engine._assemble_context(None, _view(engine), assembly_cap_override=cap)
    head = emitted[0]
    assert engine._generated_context_carrier_remainder(head) == carried["content"]
    assert all(row["content"] != oversized["content"] for row in emitted)
    _, node_ids = engine._verified_lcm_summary_prefix(head["content"])
    assert engine._dag.coverage_end(node_ids) < oversized_id < carried_id
    assert any(candidate["kind"] == "carrier" and candidate["retained_source"] == {"store_id": carried_id}
               for candidate in engine._pending_emission_candidates)

    copy_id = _append_bound_carrier(engine, head, source_id=carried_id)
    _append(engine, {"role": "assistant", "content": "The pending invoice was checked."})
    for row in NEW_TURN:
        _append(engine, row)
    view = _view(engine)
    control = [row for row in view if row["store_id"] != copy_id]
    assert len(control) == len(view) - 1
    prompt_visible = engine._assemble_context(None, view, THRESHOLD, persist=False) != engine._assemble_context(
        None, control, THRESHOLD, persist=False,
    )
    expanded = json.loads(tools.lcm_expand({"store_id": copy_id}, engine=engine))
    loaded = json.loads(tools.lcm_load_session({"session_id": engine._session_id, "limit": 100}, engine=engine))
    recall_ids = {row["store_id"] for row in loaded["messages"]}
    assert carried_id in recall_ids

    engine._ingest_cursor = len(view)
    engine._ingest_cursor_needs_reconcile = False
    engine._compress_commit_proof = None
    nodes_before = {node.node_id for node in engine._dag.get_session_nodes(engine._session_id)}
    run(engine, view, caplog, force=True)
    sources = {source for node in engine._dag.get_session_nodes(engine._session_id)
               if node.source_type == "messages" and node.node_id not in nodes_before for source in node.source_ids}
    assert carried_id in sources
    assert engine._store.get(copy_id)["content"] == head["content"]
    visibility = {
        "assembly": prompt_visible,
        "compaction": copy_id in sources,
        "expand": "error" not in expanded,
        "session recall": copy_id in recall_ids,
    }
    assert not any(visibility.values()), visibility


class _QuotedTurnHidden(AssertionError):
    """A separately identified turn was excluded as a copy."""


def _assert_distinct_head_uid(engine, emitted, quote):
    store = engine._store
    salt = store.ingest_ledger_salt()
    heads = store.connection.execute(
        "SELECT e.output_index, e.emission_id, e.descriptor_digest, v.entries "
        "FROM ledger_emissions e LEFT JOIN ingest_views v "
        "ON v.lineage=e.lineage AND v.view_seq=e.view_seq AND v.kind='emission' "
        "WHERE e.kind IN ('summary', 'carrier')"
    ).fetchall()
    assert any(index == 0 for index, *_ in heads)
    quote_uid = quote["message_uid"]
    for index, emission_id, digest, entries in heads:
        entry = ingest_ledger.decode_entries(entries)[index]
        head = emitted[index]
        assert entry.is_descriptor and entry.ref == emission_id
        assert entry.identity_digest == digest == ingest_ledger.identity_digest(salt, head)
        uid, absorbed = head["message_uid"], head.get("_absorbed_message_uids", [])
        assert entry.uidset_digest == ingest_ledger.uidset_digest(salt, uid, absorbed)
        assert entry.uidset_digest != ingest_ledger.uidset_digest(salt, quote_uid, [])
        assert quote_uid != uid and quote_uid not in absorbed


@pytest.mark.parametrize(("quote_mode", "repeat_request"), [
    pytest.param("default", False, id="default-quote-only"),
    pytest.param("default", True, id="default-quote-and-carrier-remainder"),
    pytest.param("on", True, id="on-quote-and-carrier-remainder"),
    pytest.param("on", False, id="on-quote-only", marks=pytest.mark.xfail(
        strict=True, raises=_QuotedTurnHidden,
        reason="UID-on refinement: keep a separately stamped quote with a UID distinct from every emitted head",
    )),
])
def test_separately_stamped_summary_quote(open_engine, summaries, caplog, monkeypatch,  # noqa: F811
                                          quote_mode, repeat_request):
    """A verbatim head keeps the legacy handling and a separately stamped remainder stays visible."""
    if quote_mode == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    engine = exit_engine(open_engine)
    assert engine._uid_recognition_enabled() == (quote_mode == "on")
    flush = _host_flush()
    seed = flush([dict(row, message_uid=f"seed-{index}")
                  for index, row in enumerate([*text_view()[1:-2], *NEW_TURN])])
    emitted = run(engine, seed, caplog)
    head = emitted[0]["content"]
    end, node_ids = engine._verified_lcm_summary_prefix(head)
    assert emitted[0]["role"] == "user" and end == len(head)
    assert engine._last_compress_leaves[1] > 0

    earlier = engine._store.get_session_messages(engine._session_id)
    covered_through = engine._dag.coverage_end(node_ids) or 0
    request = next(row["content"] for row in earlier if row["role"] == "user" and row["store_id"] > covered_through)
    content = f"{head}\n\n{request}" if repeat_request else head
    live = flush(emitted)
    quote, = flush([{"role": "user", "content": content, "message_uid": "separately-stamped-quote"}])
    assert quote["timestamp"] > live[0]["timestamp"]
    if quote_mode == "on":
        _assert_distinct_head_uid(engine, emitted, quote)
    engine.note_turn_complete()
    engine.ingest([*live, quote])
    stored = engine._store.get_session_messages_after(engine._session_id, earlier[-1]["store_id"])
    row, = [row for row in stored if row["observed_at"] == quote["timestamp"]]
    assert (row["role"], row["content"]) == ("user", content)
    quote_id = row["store_id"]

    view = _view(engine)
    assert any(row["store_id"] == quote_id for row in view)
    control = [row for row in view if row["store_id"] != quote_id]
    prompt_visible = engine._assemble_context(None, view, persist=False) != engine._assemble_context(
        None, control, persist=False,
    )
    expanded = json.loads(tools.lcm_expand({"store_id": quote_id}, engine=engine))
    loaded = json.loads(tools.lcm_load_session({"session_id": engine._session_id, "limit": 100}, engine=engine))
    recall_visible = ("error" not in expanded, any(row["store_id"] == quote_id for row in loaded["messages"]))

    followers = flush([
        {"role": "assistant", "content": "I read the quote."},
        {"role": "user", "content": "Please check the next accounts." + TURN * 10},
        {"role": "assistant", "content": "These accounts also reconcile."},
        {"role": "user", "content": "Please file the comparison."},
        {"role": "assistant", "content": "The comparison is filed."},
    ])
    engine.ingest([*live, quote, *followers])
    view = _view(engine)
    assert any(row["store_id"] == quote_id for row in view)
    reply_id, = [row["store_id"] for row in view if row["content"] == followers[0]["content"]]
    # Present the durable rows as a loaded host view, including the separately stored quote.
    engine._ingest_cursor = len(view)
    engine._ingest_cursor_needs_reconcile = False
    engine._compress_commit_proof = None
    nodes_before = {node.node_id for node in engine._dag.get_session_nodes(engine._session_id)}
    run(engine, view, caplog, force=True)
    sources = {source for node in engine._dag.get_session_nodes(engine._session_id)
               if node.source_type == "messages" and node.node_id not in nodes_before for source in node.source_ids}
    assert reply_id in sources
    assert engine._store.get(quote_id) == row
    visible = (prompt_visible, quote_id in sources, *recall_visible)
    assert _base_is_known_head_copy(engine._store.connection, row)
    if repeat_request:
        assert visible == (True, True, True, True)
    elif quote_mode == "default":
        assert visible == (False, False, False, False)
    elif not all(visible):
        raise _QuotedTurnHidden(f"Quote visibility (prompt, compaction, expand, session recall): {visible}")


def _summary_node(engine, source_ids, *, depth=0):
    from hermes_lcm.dag import SummaryNode

    node = SummaryNode(
        session_id=engine._session_id, depth=depth, summary="Earlier turns.", expand_hint="turns",
        source_type="nodes" if depth else "messages", source_ids=source_ids, token_count=3, source_token_count=10,
    )
    engine._dag.add_node(node)
    return node


def _head_with_remainder(remainder, *nodes):
    from hermes_lcm.engine import _summary_part_text

    prefix = "\n\n---\n\n".join(_summary_part_text(node) for node in nodes)
    return {"role": "user", "content": prefix + "\n\n" + remainder}


@pytest.mark.usefixtures("uid_mode")
def test_carried_old_repeated_turn_is_not_a_known_copy(make_engine):  # noqa: F811
    from hermes_lcm.head_copy import is_known_head_copy

    engine = make_engine()
    earlier = _append(engine, {"role": "user", "content": "continue"})
    head = _head_with_remainder("continue", _summary_node(engine, [earlier]))
    _append(engine, {"role": "user", "content": "Stop at the next checkpoint."})
    store_id = _append(engine, head, RELATION)
    stored = engine._store.get(store_id)
    assert not is_known_head_copy(engine._store.connection, stored)
    hits = json.loads(tools.lcm_grep({"query": "continue"}, engine=engine))["results"]
    assert any(hit.get("store_id") == store_id for hit in hits)


@pytest.mark.parametrize(("legacy_id", "recognised"), [(" conv ", True), (" \t ", True), (" other ", False)])
def test_carried_head_copy_with_legacy_conversation_id(make_engine, legacy_id, recognised):  # noqa: F811
    from hermes_lcm.head_copy import is_known_head_copy

    engine = make_engine()
    engine.on_session_start("legacy", conversation_id="conv", context_length=200_000)
    source = _append(engine, {"role": "assistant", "content": "Earlier reply."})
    earlier = _append(engine, {"role": "user", "content": "continue", "timestamp": 101})
    head = _head_with_remainder("continue", _summary_node(engine, [source]))
    store_id = _append_bound_carrier(engine, head, source_id=earlier)
    conn = engine._store.connection
    with engine._store.atomic():
        conn.execute("UPDATE messages SET conversation_id=? WHERE store_id=?", (legacy_id, earlier))
    assert is_known_head_copy(conn, engine._store.get(store_id)) == recognised


@pytest.mark.usefixtures("uid_mode")
def test_carried_head_copy_uses_highest_source_across_the_verified_prefix(make_engine):  # noqa: F811
    from hermes_lcm.head_copy import is_known_head_copy

    engine = make_engine()
    first = _append(engine, {"role": "user", "content": "First covered turn."})
    later = _append(engine, {"role": "user", "content": "Last covered turn."})
    leaf = _summary_node(engine, [first, later])
    condensed = _summary_node(engine, [leaf.node_id], depth=1)
    older = _summary_node(engine, [first])
    earlier = _append(engine, {"role": "user", "content": "First uncovered turn.", "timestamp": 101})
    for nodes in ((condensed, older), (older, condensed)):
        head = _head_with_remainder("First uncovered turn.", *nodes)
        store_id = _append_bound_carrier(engine, head, source_id=earlier)
        assert is_known_head_copy(engine._store.connection, engine._store.get(store_id))


@pytest.mark.usefixtures("uid_mode")
def test_carried_head_copy_with_no_source_coverage_uses_first_user_turn(make_engine):  # noqa: F811
    from hermes_lcm.head_copy import is_known_head_copy

    engine = make_engine()
    _append(engine, {"role": "user", "content": "First turn."})
    _append(engine, {"role": "user", "content": "Second turn."})
    node = _summary_node(engine, [])
    conn = engine._store.connection
    for remainder, recognised in (("First turn.", True), ("Second turn.", False)):
        store_id = _append(engine, _head_with_remainder(remainder, node), RELATION)
        stored = engine._store.get(store_id)
        assert _base_is_known_head_copy(conn, stored) == recognised
        assert not is_known_head_copy(conn, stored)


@pytest.mark.usefixtures("uid_mode")
def test_rotated_head_copy_can_prove_parent_nodes_and_remainder(make_engine):  # noqa: F811
    from hermes_lcm.head_copy import is_known_head_copy

    engine = make_engine()
    parent = engine._session_id
    source = _append(engine, {"role": "assistant", "content": "Earlier reply."})
    node = _summary_node(engine, [source])
    first = _append(engine, {"role": "user", "content": "First parent turn.", "timestamp": 101})
    second = _append(engine, {"role": "user", "content": "Second parent turn.", "timestamp": 102})
    head = _head_with_remainder("First parent turn.", node)
    engine._record_rotation_predecessor("child", parent)
    store_id = _append_bound_carrier(engine, head, source_id=first, session="child")
    conn = engine._store.connection
    assert is_known_head_copy(conn, engine._store.get(store_id))
    store_id = _append_bound_carrier(engine, head, source_id=first, session="unrelated")
    assert not is_known_head_copy(conn, engine._store.get(store_id))
    _append(engine, {"role": "user", "content": "Stop at the next checkpoint."}, session="child")
    later_head = _head_with_remainder("Second parent turn.", node)
    store_id = _append_bound_carrier(engine, later_head, source_id=second, session="child")
    assert is_known_head_copy(conn, engine._store.get(store_id))


def test_host_prefilter_skips_store_lookup_for_ordinary_turn(make_engine, monkeypatch):  # noqa: F811
    engine = make_engine()
    messages = [{"role": "user", "session_id": engine._session_id, "store_id": 1, "content": "continue"}]

    def unexpected_lookup(store_id):
        pytest.fail(f"Unexpected store read for ordinary user row {store_id}")

    monkeypatch.setattr(engine._store, "get", unexpected_lookup)
    assert engine._without_known_head_copies(messages) is messages


@pytest.mark.parametrize("shape", ["unverified", "unique_remainder", "no_relation"])
def test_uncertain_rows_are_still_assembled_and_searchable(twins, shape):
    (engine, head, _), _ = twins
    if shape == "unverified":
        content = head["content"].replace("Earlier turns", "Quoted turns", 1)
    elif shape == "unique_remainder":
        end = engine._verified_lcm_summary_prefix_end(head["content"])
        content = head["content"][:end] + "\n\nOne uniquely stored instruction."
    else:
        content = head["content"]
    row = {"role": "user", "content": content}
    store_id = _append(engine, row, None if shape == "no_relation" else RELATION)
    assembled = engine._assemble_context(None, _view(engine), persist=False)
    assert any(r["content"] == content for r in assembled)
    hits = json.loads(tools.lcm_grep({"query": "turns", "limit": 20}, engine=engine))["results"]
    assert any(hit.get("store_id") == store_id for hit in hits)
