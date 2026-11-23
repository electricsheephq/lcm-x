"""I1, I2, I11: a stored summary carrier must preserve a new user occurrence."""

from copy import deepcopy
from dataclasses import replace
import json
from statistics import median
from time import perf_counter, thread_time

import pytest

from hermes_lcm import host_uid_emit, ingest_ledger
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.head_copy import is_known_head_copy
from hermes_lcm.store import MessageStore
from tests.invariants.test_hotfix_host_shaped_passes import uid_mode  # noqa: F401
from tests.invariants.test_hotfix_stored_summary_head import (
    NEW_TURN, _append, _view as _engine_store_view,
)
from tests.invariants.test_issue_3006_post_refusal import _host_flush
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_1116_stub_first_exit_commit_proof import Host, committed_session, host_for  # noqa: F401
from tests.test_issue_671_stub_first_exit import (
    THRESHOLD, exit_engine, make_engine, run, summaries, text_view, tool_view,  # noqa: F401
)


RELATION = "generated_unproven"
REMAINDERS = ("continue", "yes", "Approved", "go ahead")


def _append_user(host, content, *, stamped=True):
    top = host.engine._store.max_store_id()
    host.clock += 30
    row = {"role": "user", "content": content}
    if stamped:
        row["timestamp"] = host.clock
    host.live.append(row)
    host.engine.ingest(host.live)
    stored, = host.engine._store.get_session_messages_after(host.sid, top)
    assert stored["role"] == "user" and stored["content"] == content
    return stored["store_id"]


def _head_sources(engine, head):
    end, node_ids = engine._verified_lcm_summary_prefix(head)
    assert end == len(head)
    messages = set()
    while node_ids:
        node = engine._dag.get_node(node_ids.pop())
        if node.source_type == "nodes":
            node_ids.extend(node.source_ids)
        else:
            assert node.source_type == "messages"
            messages.update(node.source_ids)
    return messages


def _leaf_sources(host):
    return {
        store_id
        for node in host.engine._dag.get_session_nodes(host.sid)
        if node.source_type == "messages"
        for store_id in node.source_ids
    }


def _view(host):
    engine = host.engine
    return [
        {**engine._store.to_openai_msg(row), "store_id": row["store_id"], "session_id": row["session_id"]}
        for row in engine._store.get_session_messages_after(host.sid, engine._last_compacted_store_id)
    ]


def _assembled_ids(host):
    return {row.get("store_id") for row in host.engine._assemble_context(None, _view(host), persist=False)}


def _compact_store_view(host):
    engine = host.engine
    host.turn(tool_words=1_900)
    view = _view(host)
    engine._ingest_cursor = len(view)
    engine._ingest_cursor_needs_reconcile = False
    engine._compress_commit_proof = None
    return engine.compress(view, current_tokens=engine.threshold_tokens + 5_000, force=True)


@pytest.fixture(params=["unset", "on"])
def session(request, host_for, monkeypatch):  # noqa: F811
    if request.param == "on":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "on")
    else:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    host_config = Host._config
    monkeypatch.setattr(Host, "_config", lambda host: replace(
        host_config(host), embeddings_enabled=False, temporal_rollups_enabled=False
    ))
    host = host_for("carrier")
    home = host.tmp_path / "home"
    home.mkdir(exist_ok=True)
    _state_db(home, [(host.sid, None, None)])
    assert host.engine._uid_recognition_enabled() == (request.param == "on")
    return host


def _commit_head(host, remainder):
    host.turn()
    earlier_id = _append_user(host, remainder)
    host.clock += 0.5
    host.live.append({"role": "assistant", "content": "Earlier reply.", "timestamp": host.clock})
    host.engine.ingest(host.live)
    committed_session(host, "carrier")
    emitted = host.live[0]["content"]
    end = host.engine._verified_lcm_summary_prefix_end(emitted)
    assert end
    head = emitted[:end]
    assert earlier_id in _head_sources(host.engine, head)
    return head, earlier_id


def _assert_carrier(host, carrier_id, head, remainder):
    store = host.engine._store
    row = store.get(carrier_id)
    assert row["content"] == head + "\n\n" + remainder
    assert store.get_message_relations([carrier_id], RELATION)
    return row


@pytest.mark.parametrize("remainder", REMAINDERS)
@pytest.mark.parametrize("merged", [True, False], ids=["merged", "own-row"])
def test_new_message_survives_assembly_and_compaction(session, remainder, merged):
    host = session
    head, earlier_id = _commit_head(host, remainder)
    target = _append_user(host, head + "\n\n" + remainder if merged else remainder)
    assert target > earlier_id
    if merged:
        _assert_carrier(host, target, head, remainder)

    assert target in _assembled_ids(host)
    frontier = host.frontier()
    output = _compact_store_view(host)
    assert host.frontier() > frontier
    assert target in _leaf_sources(host) or (
        target > host.frontier() and target in _assembled_ids(host)
        and any(row.get("store_id") == target for row in output)
    )


def _replay_carrier(host, head, own_row, *, stamp):
    # Present the list again, replacing the separate row with the host's merge.
    # Ingest can already recognize this replay. In that case seed the stored
    # representation that rc1's read-time predicate is responsible for.
    carrier = {"role": "user", "content": head + "\n\n" + own_row["content"]}
    if stamp is not None:
        carrier["timestamp"] = stamp
    top = host.engine._store.max_store_id()
    host.live[-1] = carrier
    engine = host.engine
    engine.ingest(deepcopy(host.live))
    added = [
        row for row in engine._store.get_session_messages_after(host.sid, top)
        if row["role"] == "user" and row["content"] == carrier["content"]
    ]
    if added:
        stored, = added
        target = stored["store_id"]
    else:
        target, = engine._store.append_batch(host.sid, [carrier], conversation_id=engine._conversation_id)
        engine._store.add_message_relations([[(target, RELATION, None, None, None)]])
    return target


@pytest.mark.parametrize("later_covered", [False, True], ids=["fresh", "later-covered"])
def test_replay_twin_stays_a_known_head_copy(session, later_covered):
    host = session
    remainder = "continue"
    head, earlier_id = _commit_head(host, remainder)
    own_id = _append_user(host, remainder)
    engine = host.engine
    own_row = engine._store.get(own_id)
    old_host_view = deepcopy(host.live)
    assert {earlier_id, own_id} & _head_sources(engine, head) == {earlier_id}
    if later_covered:
        for _ in range(4):
            if own_id in _leaf_sources(host):
                break
            _compact_store_view(host)
        assert own_id in _leaf_sources(host)
        assert own_id not in _head_sources(engine, head)
        host.live = old_host_view

    assert own_row["observed_at"] is not None
    carrier_id = _replay_carrier(host, head, own_row, stamp=own_row["observed_at"])
    carrier = _assert_carrier(host, carrier_id, head, remainder)
    assert carrier["observed_at"] == own_row["observed_at"]
    assert is_known_head_copy(engine._store.connection, carrier)
    assert carrier_id not in _assembled_ids(host)


@pytest.mark.parametrize("stamped", [True, False], ids=["distinct-stamps", "unstamped"])
def test_fresh_repeat_with_an_uncovered_twin_stays_visible(session, stamped):
    host = session
    remainder = "continue"
    head, _ = _commit_head(host, remainder)
    own_id = _append_user(host, remainder, stamped=stamped)
    target = _append_user(host, head + "\n\n" + remainder, stamped=stamped)
    assert own_id not in _head_sources(host.engine, head)
    _assert_carrier(host, target, head, remainder)
    assert target in _assembled_ids(host)


def test_unstamped_replay_without_emission_evidence_stays_visible(session):
    host = session
    remainder = "continue"
    head, _ = _commit_head(host, remainder)
    own_id = _append_user(host, remainder, stamped=False)
    own_row = host.engine._store.get(own_id)
    carrier_id = _replay_carrier(host, head, own_row, stamp=None)
    carrier = _assert_carrier(host, carrier_id, head, remainder)
    assert own_row["observed_at"] is carrier["observed_at"] is None
    assert not is_known_head_copy(host.engine._store.connection, carrier)
    assert carrier_id in _assembled_ids(host)


@pytest.mark.parametrize("stamped", [False, True], ids=["unstamped", "flush-stamped"])
def test_flushed_emission_matches_clean_twin(stamped, uid_mode, make_engine, summaries, caplog):  # noqa: F811
    # The review probe seeds the stored v0.30.2 shape after exercising host flush.
    twins = []
    for with_copy in (False, True):
        engine = exit_engine(make_engine, threshold_full_sweep_enabled=False, l3_truncate_tokens=1)
        engine.on_session_start("hf2", conversation_id="hf2", context_length=200_000)
        assert engine._uid_recognition_enabled() == (uid_mode == "on")
        engine.threshold_tokens = THRESHOLD
        seed = text_view()[1:]
        seed[0]["content"] = "Earlier turns. " + seed[0]["content"]
        first = run(engine, _host_flush()(seed), caplog)
        head_input = {**first[0], "content": first[0]["content"] + "\n\n---\n\n" + first[1]["content"]}
        returned = run(engine, tool_view(prefix=[head_input, *first[2:]])[1:], caplog)
        head = returned[0]
        flush = _host_flush()
        engine.ingest(flush(returned + NEW_TURN))
        copy_row = flush([head])[0] if stamped else head
        copy_id = _append(engine, copy_row, RELATION) if with_copy else None
        if not with_copy:
            _append(engine, NEW_TURN[0], session="unrelated")
        engine._config.large_output_active_replay_stubbing_enabled = False
        twins.append((engine, head, copy_id))

    (clean, _, _), (copied, head, copy_id) = twins
    store = copied._store
    stored = store.get(copy_id)
    remainder = copied._generated_context_carrier_remainder(head)
    own_row, = [r for r in store.get_session_messages(copied._session_id) if r["content"] == remainder]
    assert own_row["observed_at"] is not None
    assert (stored["observed_at"] is not None) == stamped
    assert stored["observed_at"] != own_row["observed_at"]
    assert is_known_head_copy(store.connection, stored)
    assembled = copied._assemble_context(None, _engine_store_view(copied), persist=False)
    assert copy_id not in {row.get("store_id") for row in assembled}
    assert assembled == clean._assemble_context(None, _engine_store_view(clean), persist=False)

    _append(copied, {"role": "user", "content": "One more unrelated question."})
    _append(copied, {"role": "assistant", "content": "One more reply."})
    view = _engine_store_view(copied)
    copied._ingest_cursor = len(view)
    copied._ingest_cursor_needs_reconcile = False
    copied._compress_commit_proof = None
    before = {n.node_id for n in copied._dag.get_session_nodes(copied._session_id)}
    start = len(summaries)
    run(copied, view, caplog, force=True)
    leaves = [n for n in copied._dag.get_session_nodes(copied._session_id)
              if n.source_type == "messages" and n.node_id not in before]
    assert copied.last_compression_status == "compacted" and leaves and summaries[start:]
    assert all(copy_id not in n.source_ids for n in leaves)
    assert all(head["content"][:200] not in call for call in summaries[start:])


@pytest.fixture
def guard_store(tmp_path):
    store = MessageStore(tmp_path / "store.db")
    dag = SummaryDAG(store.db_path)
    try:
        yield store
    finally:
        dag.close()
        store.close()


def _add_sources(conn, source_type, source_ids):
    return conn.execute(
        "INSERT INTO summary_nodes(session_id, summary, expand_hint, created_at, source_type, source_ids) "
        "VALUES ('guard', 'Proof.', 'details', 1, ?, ?)", (source_type, json.dumps(source_ids)),
    ).lastrowid


def _append_guard_user(store, content, timestamp):
    store_id, = store.append_batch("guard", [{"role": "user", "content": content, "timestamp": timestamp}])
    return store_id


def _guard_carrier(store, node_id, remainder="continue"):
    content = f"[Recent Summary (d0, node {node_id})]\nProof.\n[Expand for details: details]\n\n{remainder}"
    store_id = _append_guard_user(store, content, 30)
    store.add_message_relations([[(store_id, RELATION, None, None, None)]])
    return store.get(store_id)


def _record_guard_emission(store, row, *, kind="carrier", state="adopted", bind_before=False):
    store.ensure_ingest_ledger()
    conn = store.connection
    lineage = "provisional:guard"
    digest = ingest_ledger.identity_digest(store.ingest_ledger_salt(), row)
    with store.atomic():
        if bind_before:
            ingest_ledger.insert_bindings(conn, lineage, [{
                "ordinal": 1, "store_id": row["store_id"], "identity_digest": digest,
                "stamp": ingest_ledger.stamp_text(row["observed_at"]), "disposition": "stored",
                "view_seq": ingest_ledger.next_view_seq(conn),
            }])
        seq = ingest_ledger.next_view_seq(conn)
        ingest_ledger.insert_emissions(conn, lineage, seq, [
            {"output_index": 0, "descriptor_digest": digest, "kind": kind},
        ])
        ingest_ledger.set_emission_state(conn, lineage, seq, state)


@pytest.mark.parametrize("case", ["covered", "empty-sources", "multiple", "missing-node"])
def test_unproven_coverage_keeps_carrier(guard_store, case):
    store = guard_store
    conn = store.connection
    first = _append_guard_user(store, "continue", 30)
    unrelated = _append_guard_user(store, "different message", 20)
    if case == "multiple":
        _append_guard_user(store, "continue", 30)
    with store.atomic():
        sources = {"covered": [first], "empty-sources": []}.get(case, [unrelated])
        node = _add_sources(conn, "messages", sources)
        if case == "missing-node":
            node = _add_sources(conn, "nodes", [node, 999_999])
    assert not is_known_head_copy(conn, _guard_carrier(store, node))


@pytest.mark.parametrize("case", ["fresh-binding", "ambiguous", "wrong-kind", "refused", "no-record", "no-ledger"])
def test_unproven_emission_keeps_stamped_carrier(guard_store, case):
    store = guard_store
    other = _append_guard_user(store, "different message", 20)
    _append_guard_user(store, "continue", 25)
    if case == "ambiguous":
        _append_guard_user(store, "continue", 26)
    with store.atomic():
        node = _add_sources(store.connection, "messages", [other])
    row = _guard_carrier(store, node)
    if case == "no-record":
        store.ensure_ingest_ledger()
    elif case != "no-ledger":
        _record_guard_emission(
            store, row, kind="summary" if case == "wrong-kind" else "carrier",
            state="refused" if case == "refused" else "adopted", bind_before=case == "fresh-binding",
        )
    assert not is_known_head_copy(store.connection, row)


def test_zero_match_cost_stays_near_rc1(guard_store):
    store = guard_store
    conn = store.connection
    covered = 100_000
    with store.atomic():
        conn.execute(
            "WITH RECURSIVE ids(id) AS (VALUES (1) UNION ALL SELECT id+1 FROM ids WHERE id<?) "
            "INSERT INTO messages(session_id, role, content, timestamp) "
            "SELECT 'guard', 'user', 'covered text', 1 FROM ids", (covered,),
        )
        node = _add_sources(conn, "messages", list(range(1, covered + 1)))
    row = _guard_carrier(store, node)

    def rc1_remainder_query():
        return conn.execute(
            "SELECT 1 FROM messages WHERE session_id=? AND store_id<? AND role='user' "
            "AND COALESCE(conversation_id, '') IN ('', ?) AND content=? LIMIT 1",
            ("guard", row["store_id"], "", "continue"),
        ).fetchone()

    queries = []
    conn.set_trace_callback(queries.append)
    try:
        assert not is_known_head_copy(conn, row)
    finally:
        conn.set_trace_callback(None)
    assert not any("WITH RECURSIVE sources" in query for query in queries)

    def measure(call):
        wall_start = perf_counter()
        cpu_start = thread_time()
        assert not call()
        return (thread_time() - cpu_start) * 1_000, (perf_counter() - wall_start) * 1_000

    rc1, guard = zip(*[(measure(rc1_remainder_query), measure(lambda: is_known_head_copy(conn, row)))
                       for _ in range(21)])
    rc1_cpu, rc1_wall = map(median, zip(*rc1))
    guard_cpu, guard_wall = map(median, zip(*guard))
    print(f"{covered:,} covered rows, 21 calls per arm. Median thread CPU: rc1 SQL {rc1_cpu:.3f} ms, "
          f"guard predicate {guard_cpu:.3f} ms, delta {guard_cpu - rc1_cpu:+.3f} ms. "
          f"Median elapsed: rc1 SQL {rc1_wall:.3f} ms, guard {guard_wall:.3f} ms.")
    assert guard_cpu <= rc1_cpu + 3
