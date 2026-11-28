"""#1250 R2: binding conflicts, competing engines and recorded compression rotations."""

from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.aux_session import _DELEGATE_SOURCE
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError

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
    child_turn = turns("child", 1) if store_child_turn else []
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
