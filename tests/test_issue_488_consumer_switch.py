"""#488 PR-B: consumers read the emission projection, never summary shape.

Occurrence-level checks for the consumer switch: an unproven, ambiguous or
out-of-place summary-shaped row keeps FULL identity (store direction), and a
proven emitted occurrence keeps its projected remainder.
"""

from __future__ import annotations

import json
import time

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import (
    _config,
    _host_merge_consecutive_users,
    _stub_summarizer,
    _summary_carrier_fixture,
    _turn,
)
from tests.test_issue_488_emission_descriptors import _record, _scoped_proof
from tests.test_issue_488_emission_proof import (
    _phase1_compacted_engine,
    _stored_user_rows,
    _summary_block,
)

X = "ISSUE-488-X authored remainder"


def _engine_with_verified_block(tmp_path):
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    node = engine._dag.add_node(
        SummaryNode(
            session_id="S0",
            depth=0,
            summary="Verified summary.",
            token_count=3,
            source_token_count=5,
            source_ids=[],
            source_type="messages",
            created_at=time.time(),
            earliest_at=time.time(),
            latest_at=time.time(),
            expand_hint="verified",
        )
    )
    node_id = getattr(node, "node_id", node)
    block = f"[Recent Summary (d0, node {node_id})]\nVerified summary.\n[Expand for details: verified]"
    assert engine._verified_lcm_summary_prefix_end(block) == len(block)
    return engine, block


def test_authored_summary_shaped_sole_prompt_keeps_its_retained_anchor(tmp_path):
    engine, block = _engine_with_verified_block(tmp_path)
    system = {"role": "system", "content": "You are concise."}
    authored = {"role": "user", "content": block + "\n\n" + X}
    try:
        engine.ingest([system, authored])
        stored = engine._store._conn.execute(
            "SELECT store_id, content FROM messages WHERE role = 'user'"
        ).fetchall()
        assert [content for _store_id, content in stored] == [authored["content"]]
        # The stored row keeps full identity, and the live anchor is compared in full.
        row = engine._prepare_retained_user_anchor([system, dict(authored)])
        assert row is not None and int(row["store_id"]) == stored[0][0]
    finally:
        engine.shutdown()


def test_replay_cache_never_serves_a_carrier_view_for_its_remainder(tmp_path):
    engine, block = _engine_with_verified_block(tmp_path)
    head = {"role": "user", "content": "first"}
    authored = [head, {"role": "user", "content": block + "\n\n" + X}]
    plain = [head, {"role": "user", "content": X}]
    try:
        engine._remember_active_replay_messages(authored, [dict(m) for m in authored])
        assert engine._cached_active_replay_messages([dict(m) for m in authored]) is not None
        assert engine._cached_active_replay_messages(plain) is None
        assert not engine._is_cached_active_replay_message_at_index(1, plain[1])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("emitted_row", ["present", "gone"])
def test_authored_summary_prefix_past_its_emission_is_stored_whole(tmp_path, monkeypatch, emitted_row):
    """AMENDMENT 2: S+Y matching a standalone summary descriptor past the emitted
    occurrence's position is authored: full identity, stored whole, never a skip."""
    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=1)
    block = _summary_block(engine, compressed)
    assert compressed[0]["content"] == block
    authored = {"role": "user", "content": block + "\n\n" + X}
    host = [dict(m) for m in compressed] + [_turn(13)[1], authored]
    if emitted_row == "gone":
        host = host[1:]
    try:
        projection, identities = engine._occurrence_replay_identities(host, engine._active_emission_proof())
        # Gone: the projection alone would admit the authored row (N6). Present: the emitted row
        # takes the descriptor and stays a proven scaffold.
        assert projection.entries[-1].generated_span == (block if emitted_row == "gone" else None)
        assert (identities[0] is None) == (emitted_row == "present")
        assert identities[-1][1] == authored["content"]
        engine.ingest(host)
        stored = [content for _store_id, content in _stored_user_rows(engine)]
        assert stored.count(authored["content"]) == 1
        assert stored.count(X) == 0
    finally:
        engine.shutdown()


def test_identical_authored_summary_after_the_emitted_one_is_content(tmp_path, monkeypatch):
    """AMENDMENT 2: a sole emitted S gone and an identical authored S later (multiplicity
    one) is content, not a proven scaffold."""
    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=1)
    block = _summary_block(engine, compressed)
    host = [dict(m) for m in compressed[1:]] + [_turn(13)[1], {"role": "user", "content": block}]
    try:
        projection, identities = engine._occurrence_replay_identities(host, engine._active_emission_proof())
        assert projection.entries[-1].generated_span == block
        assert identities[-1] is not None and identities[-1][1] == block
    finally:
        engine.shutdown()


@pytest.mark.parametrize("source", ["cached", "durable"])
def test_declined_empty_suffix_descriptor_keeps_full_identity(tmp_path, source):
    """AMENDMENT 3: an empty-suffix descriptor the projection declines (ambiguous
    multiplicity) leaves the row as content in the walks and the mapper, cached and
    after a durable reload; no span is re-derived from the descriptor."""
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        engine._assemble_context({"role": "system", "content": "system"}, tail, include_lcm_note=False)
        candidate = next(item for item in engine._pending_emission_candidates if item["kind"] == "summary")
        span = candidate["span"]
        _record(engine, [compacted, *tail], [candidate["row"], {"role": "user", "content": span}])
        if source == "durable":
            engine._last_emission_descriptors = None
            assert engine._active_emission_proof()["emissions"]
        remaining = [{"role": "user", "content": span}]
        full = engine._message_replay_identity(remaining[0], strip_carrier=False)
        _projection, identities = engine._occurrence_replay_identities(remaining, engine._active_emission_proof())
        assert identities == [full]
        assert engine._effective_replay_identities(remaining) == [full]
        stored_id = engine._store.append("S0", {"role": "user", "content": span})
        assert engine._get_store_id_map_for_messages(remaining) == {id(remaining[0]): stored_id}
    finally:
        engine.shutdown()


@pytest.mark.parametrize("retained", [{"store_id": 7}], ids=["retained"])
@pytest.mark.parametrize("kind", ["carrier", "summary"], ids=["carrier", "folded"])
@pytest.mark.parametrize(
    "suffix,remainder",
    [("retained", "retained"), ("authored instead", None), ("retained\n\nnew turn", "retained\n\nnew turn")],
    ids=["matched", "unmatched", "prefix-extension"],
)
def test_carrier_remainder_only_when_its_suffix_matched(tmp_path, kind, retained, suffix, remainder):
    """AMENDMENTS 1, 4, 5: a carrier or folded carrier yields its remainder only when its
    recorded suffix matched (else full identity); bytes past the recorded suffix stay in
    the identity, so they never match the retained row (store direction)."""
    engine, _block = _engine_with_verified_block(tmp_path)
    span = "GENERATED\n\n"
    row = {"role": "user", "content": span + suffix}
    try:
        messages = [{"role": "assistant", "content": "earlier"}, row]  # past output index 0: still bound
        _projection, identities = engine._occurrence_replay_identities(
            messages, _scoped_proof(kind, span, "retained", retained_source=retained)
        )
        assert identities[1][1] == (row["content"] if remainder is None else remainder)
    finally:
        engine.shutdown()


def _merged_composite_session(tmp_path, monkeypatch, *, tail, mode, objective=""):
    """Compact once, then the host merges a NEW user row into the emitted head (#499)."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())

    def make():
        return LCMEngine(config=_config(tmp_path, fresh_tail_count=tail), hermes_home=str(tmp_path / "home"))

    engine, host, child = make(), [], "S0" if mode == "inplace" else "S1"
    engine.on_session_start("S0", platform="acp", context_length=200_000)

    def add(message):
        host.append(dict(message))
        host[:] = _host_merge_consecutive_users(host)
        engine.ingest(host)

    for i in range(1, 13):
        add(_turn(i)[0])
        add(_turn(i)[1])
    add({"role": "user", "content": _turn(13)[0]["content"] + objective})
    pre = list(host)
    out = engine.compress(list(host), force=True)
    assert engine._last_compression_status == "compacted"
    engine.on_session_end("S0", pre)
    engine.on_session_start(child, boundary_reason="compression", old_session_id="S0", platform="acp")
    host[:] = [dict(m) for m in out]
    add({"role": "user", "content": "NEW-0 typed after the compaction"})
    add({"role": "assistant", "content": "reply to NEW-0"})
    return engine, host, make, child, add


def _rows(engine):
    return engine._store._conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()


@pytest.mark.parametrize("mode", ["inplace", "rotation"])
def test_host_merged_composite_maps_whole_and_compacts_again(tmp_path, monkeypatch, mode):
    """#499 tail 0: the composite is stored whole, so the mapper maps it whole (no source gap)."""
    engine, host, _make, _child, add = _merged_composite_session(tmp_path, monkeypatch, tail=0, mode=mode)
    try:
        for i in range(30, 34):
            add(_turn(i)[0])
            add(_turn(i)[1])
        add(_turn(34)[0])
        assert id(host[0]) in engine._get_store_id_map_for_messages(list(host))
        engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "tail,mode,objective",
    [(1, "rotation", ""), (0, "inplace", "\nimage data:image/png;base64," + "iVBORw0KGgoAAAANSUhEUgAA" * 400)],
    ids=["carrier-extension", "objective-payload"],
)
def test_host_merged_composite_is_not_stored_again_after_restart(tmp_path, monkeypatch, tail, mode, objective):
    """#499 after a restart: the store-tail walk and the durable proof walk match the composite
    in its stored (whole) form, so the replayed rows are not persisted twice."""
    engine, host, make, child, _add = _merged_composite_session(
        tmp_path, monkeypatch, tail=tail, mode=mode, objective=objective
    )
    before = _rows(engine)
    engine.shutdown()
    engine = make()
    try:
        engine.on_session_start(child, platform="acp", context_length=200_000)
        engine.ingest(host)
        assert _rows(engine) == before
    finally:
        engine.shutdown()

# --- fix round 1 (gpt-6-astra review of 5e015fee): each test reproduces a reviewer counterexample.


@pytest.mark.parametrize("bad", ["bad", {"index": "0"}], ids=["non-mapping", "non-int-index"])
def test_malformed_output_occurrence_declines_the_descriptor(tmp_path, monkeypatch, bad):
    """F7: a persisted v4 descriptor with a malformed output_occurrence is declined (full
    identity for its row); reconciliation never raises on it."""
    engine, _pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=1)
    block = _summary_block(engine, compressed)
    try:
        key = "compaction_commit_proof:S0"
        payload = engine._store.read_metadata_json(key)
        assert payload["emissions"][0]["output_occurrence"]["index"] == 0
        payload["emissions"][0]["output_occurrence"] = bad
        engine._store.write_metadata_json([key], json.dumps(payload, sort_keys=True))
        engine._last_emission_descriptors = None
        host = [dict(m) for m in compressed]
        projection, identities = engine._occurrence_replay_identities(host, engine._active_emission_proof())
        assert projection.entries[0].generated_span is None
        assert identities[0] is not None and identities[0][1] == block
        engine.ingest(host + [_turn(13)[1]])
    finally:
        engine.shutdown()
