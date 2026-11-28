"""#1250 R2: binding conflicts, competing engines and recorded compression rotations."""

from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
import sqlite3
import sys

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.aux_session import _DELEGATE_SOURCE
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.tokens import count_messages_tokens
from tests.invariants.test_issue_1250_publication_reown import (
    EARLY, LATER, SESSION, _add_turn, _bind_gateway, _digests, _publish, _rows,
    open_engine,  # noqa: F401
)
from tests.test_issue_671_stub_first_exit import payload_of, tool_content, tool_pair

S = "S"
C = "S2"
G = "agent:main:local:dm:G"
pytestmark = pytest.mark.issue(1250)

SETTINGS = dict(fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0,
                temporal_rollups_enabled=False, embeddings_enabled=False)


def turns(label, n=2):
    return [
        {"role": role, "content": f"{label} {i} {role}: a distinct synthetic message."}
        for i in range(n) for role in ("user", "assistant")
    ]


def digests(rows):
    return Counter((row["role"], sha256(row["content"].encode()).hexdigest()) for row in rows)


@pytest.fixture
def pair(tmp_path, monkeypatch):
    calls = []

    def summarize(**kwargs):
        calls.append(None)
        return "Synthetic summary.", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    with ExitStack() as stack:
        def build(session, conversation=None, **kwargs):
            engine = LCMEngine(
                config=LCMConfig(database_path=str(tmp_path / "lcm.db"), **SETTINGS),
                hermes_home=str(tmp_path / "hermes"),
            )
            stack.callback(engine.shutdown)
            bind(engine, session, conversation, **kwargs)
            return engine
        yield build, calls


def bind(engine, session, conversation=None, **kwargs):
    engine.on_session_start(session, conversation_id=conversation, platform="gui",
                            context_length=200_000, **kwargs)


def rows(engine, sessions):
    return sorted((r for s in sessions for r in engine._store.get_session_messages(s)),
                  key=lambda r: r["store_id"])


def step(label, engine, calls, host, sessions=(S,), expected=None):
    count = len(calls)
    nodes = len(engine._dag.get_session_nodes(engine._session_id))
    before = digests(rows(engine, sessions))
    view = list(engine.compress(list(host)))
    stored = rows(engine, sessions)
    counts = digests(stored)
    print(json.dumps(dict(step=label, status=engine.last_compression_status,
                          calls=len(calls) - count, new_leaves=len(engine._dag.get_session_nodes(engine._session_id)) - nodes,
                          owner_pairs=[(r["session_id"], r["conversation_id"]) for r in stored],
                          before=[(*key, value) for key, value in sorted(before.items())],
                          digests=[(*key, value) for key, value in sorted(counts.items())],
                          duplicates=sum((counts - expected).values()) if expected is not None else None,
                          missing=sum((expected - counts).values()) if expected is not None else None)))
    assert counts == expected
    return view


def test_w5_a(pair):
    build, calls = pair
    writer = build(S)
    early = turns("early")
    writer.ingest(list(early))
    gateway = build(S, G)
    later = turns("gateway")
    gateway.ingest(early + later)
    expected = digests(early + later)
    step("G1", gateway, calls, early + later, expected=expected)

    host = list(early)
    for i in range(1, 4):
        added = turns(f"writer:{i}", 1)
        host += added
        expected += digests(added)
        host = step(f"S{i}", writer, calls, host, expected=expected)


def test_w5_b(pair):
    build, calls = pair
    first = build(S, G)
    early = turns("early")
    first.ingest(list(early))
    second = build(S, G)
    second.ingest(list(early))
    expected = digests(early)
    step("G1", first, calls, early, expected=expected)
    assert first.last_compression_status == "compacted"
    host = list(early)
    for i in range(1, 4):
        added = turns(f"second:{i}", 1)
        host += added
        expected += digests(added)
        host = step(f"G2.{i}", second, calls, host, expected=expected)


def rotate_and_bind(engine, *, store_child_turn=True, prior=None):
    early = turns("early")
    engine.ingest(list(early))
    host = list(engine.compress(list(early)))
    assert engine.last_compression_status == "compacted"
    engine.on_session_end(S, list(early))
    bind(engine, C, boundary_reason="compression", old_session_id=S)
    assert engine._conversation_id == (prior or S)
    assert engine._store.read_metadata_json(f"rotation_predecessor_session:{C}") == S
    child_turn = [dict(row, timestamp=200 + index) for index, row in enumerate(turns("child", 1))] if store_child_turn else []
    if child_turn:
        host += child_turn
        engine.ingest(list(host))
    bind(engine, C, G)
    later = turns("gateway")
    return host + later, early + child_turn + later


@pytest.mark.parametrize("store_child_turn", [True, False], ids=["child-rows", "empty-child"])
def test_rotation_publishes_without_copying_carried_rows(pair, store_child_turn):
    build, calls = pair
    engine = build(S)
    host, expected = rotate_and_bind(engine, store_child_turn=store_child_turn)
    before = rows(engine, (S, C))
    step("gateway", engine, calls, host, (S, C), expected=digests(expected))
    assert engine.last_compression_status == "compacted"
    after = rows(engine, (S, C))
    assert [(r["session_id"], r["conversation_id"]) for r in after] == (
        [(S, S)] * 3 + [(S, G)] + [(C, G)] * (len(expected) - 4)
    )
    assert [dict(old, conversation_id=new["conversation_id"]) for old, new in zip(before, after)] == after[:len(before)]
    old = engine._lifecycle.get_by_conversation(S)
    assert old.current_session_id is None and old.last_finalized_session_id == C
    assert old.last_finalized_frontier_store_id == 3


@pytest.mark.parametrize("full_view", [False, True], ids=["child-only", "carried-fresh-tail"])
def test_rotated_session_window_backs_off_after_gateway_reownership(pair, tmp_path, full_view):
    build, calls = pair
    (tmp_path / "hermes").mkdir(exist_ok=True)
    with sqlite3.connect(tmp_path / "hermes" / "state.db") as db:
        db.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT INTO sessions VALUES (?, ?, ?)", [(S, None, "compression"), (C, S, None)])
    writer = build(S)
    early = [dict(row, timestamp=100 + i) for i, row in enumerate(turns("early"))]
    writer.ingest(early)
    parent_view = writer.compress(early)
    assert writer.last_compression_status == "compacted"
    writer.on_session_end(S, early)
    bind(writer, C, boundary_reason="compression", old_session_id=S)
    assert writer._conversation_id == S
    child = turns("child")
    writer_view = parent_view + child
    writer.ingest(writer_view)
    carried = rows(writer, (S,))[-1]
    if full_view:
        assert (S, carried["store_id"] - 1, carried["store_id"]) in writer._load_compression_carry_ranges()
    child_ids = {row["store_id"] for row in rows(writer, (C,)) if row["conversation_id"] == S}
    assert len(child_ids) == len(child)

    gateway = build(C, G)
    gateway_view = writer_view + turns("gateway")
    gateway.ingest(gateway_view)
    gateway.compress(gateway_view, force=True)
    assert gateway.last_compression_status == "compacted"
    before = rows(gateway, (S, C))
    assert child_ids <= {row["store_id"] for row in before
                         if row["session_id"] == C and row["conversation_id"] == G}
    if full_view:
        # G's commit superseded C's rotation proof. Register the same proven
        # carry through the ancestor-anchor path to exercise a mixed-session leaf.
        writer._register_identity_anchor_carry([carried])

    count = len(calls)
    fallback = writer.compress(writer_view if full_view else child, force=True)
    assert writer.get_status()["last_compression_failure_reason"] == "session_owned_by_other_window"
    if full_view:
        assert carried["store_id"] in writer._session_owner_backoff[3]
        # The carried row alone must keep the back-off armed on a shorter view.
        conflict = writer._other_window_publication_backoff(parent_view)
        assert conflict is not None
    assert len(calls) == count + 1
    assert writer.compress(fallback, force=True) == fallback
    assert writer.get_status()["last_compression_failure_reason"] == "session_owned_by_other_window"
    assert len(calls) == count + 1
    assert rows(writer, (S, C)) == before


def test_failed_provisional_rebind_proof_preserves_identity_anchors(pair, monkeypatch):
    build, _ = pair
    engine = build(C, G)
    anchors = [(S, 3, 7)]
    monkeypatch.setattr(engine, "_identity_anchor_carry_ranges", lambda: anchors)
    monkeypatch.setattr(engine, "_durable_commit_proof_payload", lambda *_: None)
    engine._store.write_metadata_json([f"rotation_predecessor_session:{C}"], json.dumps(S))

    def unavailable(_predecessors):
        raise RuntimeError("injected provisional proof failure")

    monkeypatch.setattr(engine, "_provisional_rebind_proof", unavailable, raising=False)
    assert engine._load_compression_carry_ranges() == anchors


def stage(engine, covered, *, carried_ranges=None):
    session = engine._session_id
    return engine._dag.add_node(
        SummaryNode(session_id=session, summary="Synthetic summary.", source_ids=covered),
        before_commit=lambda conn, node_id: engine._lifecycle.stage_compaction_publication(
            conn, G, session, node_id, 0, covered, carried_ranges=carried_ranges,
        ),
    )


@pytest.mark.parametrize("binding", ["missing", "other-session"])
def test_unbound_publisher_refuses_before_reownership(pair, binding):
    build, _ = pair
    engine = build(S)
    engine.ingest(turns("provisional"))
    if binding == "other-session":
        build("unrelated", G)
    before = rows(engine, (S,))
    # A SQLite callback records even an UPDATE that a subsequent error rolls back.
    conn = engine._dag._conn
    moved = []
    conn.create_function("record_reownership", 1, moved.append)
    conn.execute(
        "CREATE TEMP TRIGGER record_reownership AFTER UPDATE OF conversation_id ON messages "
        "BEGIN SELECT record_reownership(OLD.store_id); END"
    )
    with pytest.raises(LifecyclePublicationConflictError, match="^Compaction publication source ownership changed$"):
        stage(engine, [row["store_id"] for row in before])
    assert moved == [] and rows(engine, (S,)) == before
    assert not engine._dag.get_session_nodes(S)


def test_reownership_row_count_mismatch_rolls_back(pair):
    build, _ = pair
    engine = build(S)
    engine.ingest(turns("provisional", 1))
    bind(engine, S, G)
    before = rows(engine, (S,))
    engine._dag._conn.execute(
        "CREATE TEMP TRIGGER skip_reownership BEFORE UPDATE OF conversation_id ON messages "
        f"WHEN OLD.store_id = {before[-1]['store_id']} BEGIN SELECT RAISE(IGNORE); END"
    )
    with pytest.raises(LifecyclePublicationConflictError, match="^Compaction publication re-owned row count changed$"):
        stage(engine, [row["store_id"] for row in before])
    assert rows(engine, (S,)) == before
    assert not engine._dag.get_session_nodes(S)
    assert engine._lifecycle.get_by_conversation(G).current_frontier_store_id == 0
    assert engine._lifecycle.get_by_conversation(S).current_session_id == S


@pytest.mark.parametrize("prior", ["gateway", "delegate", "unrecorded"])
def test_rotation_keeps_foreign_owner(pair, prior):
    build, _ = pair
    first_gateway = "agent:main:local:dm:G1"
    if prior == "unrecorded":
        engine = build(C, G)
        row_id = engine._store.append(C, turns("unrecorded")[0], conversation_id=S)
        carry = []
    else:
        engine = build(S, first_gateway if prior == "gateway" else None)
        rotate_and_bind(engine, prior=first_gateway if prior == "gateway" else None)
        if prior == "gateway":
            row_id = engine._store.get_session_messages(C)[0]["store_id"]
            carry = []
        else:
            [row_id] = engine._append_off_current_session_end_suffix(
                S, turns("delegate")[0:1], source=_DELEGATE_SOURCE, conversation_id=S,
            )
            carry = [(S, row_id - 1, row_id)]
    before = rows(engine, (S, C))
    with pytest.raises(LifecyclePublicationConflictError, match="^Compaction publication source ownership changed$"):
        stage(engine, [row_id], carried_ranges=carry)
    assert rows(engine, (S, C)) == before


@pytest.mark.parametrize("ratio", [0.0, 0.0002])
def test_backoff_takes_staged_stub_first_exit_under_depth_target(open_engine, monkeypatch, ratio):  # noqa: F811
    # A depth target the stub trial misses stages the free cuts; the back-off must not drop them.
    writer = open_engine(
        large_output_externalization_threshold_chars=1_000_000,
        leaf_chunk_tokens=250,
        summary_prefix_target_tokens=100,
        depth_target_ratio=ratio,
    )
    gateway = open_engine()
    writer.on_session_start(SESSION, platform="gui", context_length=200_000)
    writer.ingest(list(EARLY))
    _bind_gateway(gateway)
    gateway.ingest(EARLY + LATER)
    first = _publish(gateway, EARLY + LATER)
    _publish(gateway, _add_turn(gateway, first, 0))

    leaf_calls = []
    summarize_leaf = writer._summarize_leaf_chunk_with_rescue

    def count_leaf(*args, **kwargs):
        leaf_calls.append(None)
        return summarize_leaf(*args, **kwargs)

    monkeypatch.setattr(writer, "_summarize_leaf_chunk_with_rescue", count_leaf)
    writer.threshold_tokens = 3_500
    host = list(EARLY) + [{"role": "user", "content": "Read the tool fixture."}]
    host += tool_pair("threshold-tool", payload_of(6_000, "threshold"))
    host = writer.compress(host, current_tokens=count_messages_tokens(host))
    assert writer.get_status()["last_compression_failure_reason"] == "session_owned_by_other_window"
    assert writer._session_owner_backoff is not None
    first_pass_leaf_calls = len(leaf_calls)

    host += [{"role": "user", "content": "The tool output is now outside the fresh tail."}]
    tokens = count_messages_tokens(host)
    expected = _digests(_rows(writer)) + _digests(host[-1:])
    out = writer.compress(host, current_tokens=tokens)
    assert writer.last_compression_status != "error"
    assert writer._last_threshold_full_sweep["stop_reason"] == "stub_first_exit"
    assert count_messages_tokens(out) < tokens
    assert tool_content(out, "threshold-tool").startswith("[Externalized tool output:")
    assert writer._session_owner_backoff is not None
    assert len(leaf_calls) == first_pass_leaf_calls
    assert _digests(_rows(writer)) == expected


def _backed_off_depth_target_writer(open_one):
    writer = open_one(
        large_output_externalization_threshold_chars=1_000_000,
        leaf_chunk_tokens=250,
        summary_prefix_target_tokens=100,
        depth_target_ratio=0.0002,
    )
    gateway = open_one()
    writer.on_session_start(SESSION, platform="gui", context_length=200_000)
    writer.ingest(list(EARLY))
    _bind_gateway(gateway)
    gateway.ingest(EARLY + LATER)
    first = _publish(gateway, EARLY + LATER)
    _publish(gateway, _add_turn(gateway, first, 0))
    writer.threshold_tokens = 3_500
    # The system message carries the LCM note, so the stubbed list has a non-empty replay digest.
    host = [{"role": "system", "content": "You are a careful agent."}] + list(EARLY)
    host += [{"role": "user", "content": "Read the tool fixture."}]
    host += tool_pair("threshold-tool", payload_of(6_000, "threshold"))
    host = writer.compress(host, current_tokens=count_messages_tokens(host))
    assert writer.get_status()["last_compression_failure_reason"] == "session_owned_by_other_window"
    host += [{"role": "user", "content": "The tool output is now outside the fresh tail."}]
    finalized = []
    finalize = writer._finalize_stub_first_exit

    def record_finalize(*args, **kwargs):
        finalized.append(finalize(*args, **kwargs))
        return finalized[-1]

    writer._finalize_stub_first_exit = record_finalize
    return writer, host, finalized


@pytest.mark.parametrize("error", [RuntimeError("frontier read failed"),
                                   sqlite3.OperationalError("database is locked")],
                         ids=["runtime", "locked"])
def test_backoff_stub_first_fallback_keeps_replay_proof_consistent(open_engine, monkeypatch, error):  # noqa: F811
    # The fallback commits its exit (replay digest, marker) before reading the summary frontier for status.
    writer, host, finalized = _backed_off_depth_target_writer(open_engine)
    frontier = writer._summary_frontier_tokens

    def failing_frontier():
        if sys._getframe(1).f_code.co_name == "_use_stub_first_fallback":
            raise error
        return frontier()

    monkeypatch.setattr(writer, "_summary_frontier_tokens", failing_frontier)
    before = set(writer._load_compacted_active_replay_snapshot_digests())
    expected = _digests(_rows(writer)) + _digests(host[-1:])
    out = writer.compress(host, current_tokens=count_messages_tokens(host))
    assert finalized and finalized[-1] is not None
    stubbed = finalized[-1]
    assert writer._compacted_active_replay_snapshot_digest(stubbed)
    new = set(writer._load_compacted_active_replay_snapshot_digests()) - before
    took_exit = (out == stubbed and writer._compacted_active_replay_snapshot_digest(out) in new
                 and writer._last_stub_first_exit is not None and writer.last_compression_status != "error")
    failed_open = not new and writer._last_stub_first_exit is None and out == host
    assert took_exit or failed_open, (writer.last_compression_status, len(new), writer._last_stub_first_exit)
    assert writer._session_owner_backoff is not None
    assert _digests(_rows(writer)) == expected


def test_backoff_stub_first_fallback_writes_the_returned_replay_proof(open_engine):  # noqa: F811
    writer, host, finalized = _backed_off_depth_target_writer(open_engine)
    before = set(writer._load_compacted_active_replay_snapshot_digests())
    out = writer.compress(host, current_tokens=count_messages_tokens(host))
    assert finalized and finalized[-1] is not None and out == finalized[-1]
    digest = writer._compacted_active_replay_snapshot_digest(out)
    assert digest and digest in set(writer._load_compacted_active_replay_snapshot_digests()) - before
    assert writer._last_threshold_full_sweep["stop_reason"] == "stub_first_exit"
    assert writer._last_threshold_full_sweep["summary_prefix_tokens_after"] == writer._summary_frontier_tokens()
