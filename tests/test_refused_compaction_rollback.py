import json
import logging

import pytest

import hermes_lcm.engine as lcm_engine_module
from bench.instruments.gate_kit.audit import carrier_remainder, verified_summary_head_store_ids
from hermes_lcm import host_uid_emit, ingest_ledger
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _config, _drain_session_end_maintenance, _stub_summarizer, _turn
from tests.test_issue_1116_stub_first_exit_commit_proof import (
    committed_session,
    grow_over_threshold,
    host_for,  # noqa: F401
    stub_first_result,
)


@pytest.fixture
def committed_host(tmp_path, monkeypatch, request):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    config = _config(tmp_path, fresh_tail_count=getattr(request, "param", 6))
    home = str(tmp_path / "home")
    binding = dict(platform="acp", context_length=200_000, conversation_id="refusal-test")
    engine = LCMEngine(config=config, hermes_home=home)
    engine.on_session_start("S0", **binding)
    try:
        pre = [row for i in range(1, 13) for row in _turn(i)]
        # Two visible assistant replies become one in the emitted tail. Neither
        # the ordinary stored-tail matcher nor #524 can replay it as raw rows.
        pre.extend([{"role": "assistant", "content": "One more reply to turn 12."}, _turn(13)[0]])
        compressed = engine.compress(pre, force=True)
        _drain_session_end_maintenance(engine)
        assert engine._verified_lcm_summary_prefix_end(compressed[0]["content"])
        assert engine._last_compress_leaves[1] == 1
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", **binding)
        kept = list(compressed) + [_turn(13)[1]]
        for i in range(14, 21):
            kept.extend(_turn(i)[:1] if i == 20 else _turn(i))
        yield engine, kept, binding
    finally:
        engine.shutdown()


def _compaction_records(engine):
    return dict(engine._store._conn.execute(
        "SELECT key,value FROM metadata WHERE key IN ('compaction_commit_proof:S0','host_occurrence_snapshot:S0')"))


def _rewriting_pass(engine, kept):
    refused = engine.compress(kept, force=True)
    assert engine._last_compression_status == "compacted"
    assert engine._last_compress_leaves[1] == 1
    assert refused[0]["content"] != kept[0]["content"]
    return refused


def _assert_only_reply(engine, before, head, reply):
    rows = engine._store.get_session_messages("S0", limit=10_000)
    duplicates = [(row["role"], len(row["content"])) for row in rows if row["content"] == head]
    reason = engine._last_ingest_reconciliation.get("reason")
    assert (len(rows) - before, duplicates) == (1, []), (len(rows) - before, duplicates, reason)
    assert rows[-1]["content"] == reply["content"]
    assert reason != "persisted ambiguous delta"


@pytest.mark.parametrize("restart", [False, True], ids=["same-engine", "restart"])
def test_refused_compaction_keeps_the_ingested_input(committed_host, restart):
    engine, kept, binding = committed_host
    prior = _compaction_records(engine)
    _rewriting_pass(engine, kept)
    assert engine._compress_commit_reason == "ready"
    assert _compaction_records(engine) != prior
    engine.record_rejected_compaction()

    before = engine._store.get_session_count("S0")
    reply = _turn(20)[1]
    if restart:
        config, home = engine._config, str(engine._hermes_home)
        engine.shutdown()
        resumed = LCMEngine(config=config, hermes_home=home)
        try:
            resumed.on_session_start("S0", **binding)
            resumed.ingest(kept + [reply])
            _assert_only_reply(resumed, before, kept[0]["content"], reply)
            assert _compaction_records(resumed) == prior
        finally:
            resumed.shutdown()
    else:
        engine.compress(kept + [reply], current_tokens=10_000)
        _assert_only_reply(engine, before, kept[0]["content"], reply)
        assert _compaction_records(engine) == prior


def test_refused_cleanup_only_pass_keeps_the_ingested_input(committed_host):
    engine, kept, _binding = committed_host
    kept.insert(-1, {"role": "assistant", "content": "Another visible reply to turn 19."})
    prior = _compaction_records(engine)
    descriptors = engine._last_emission_descriptors
    engine._compaction_request.below_threshold_cleanup_only = True
    output = engine.compress(kept, current_tokens=10_000)
    assert engine._last_compression_status == "sanitized"
    assert engine._compress_commit_reason == "ready"
    assert len(output) < len(kept)
    assert _compaction_records(engine) == prior
    assert engine._last_emission_descriptors is not descriptors
    engine.record_rejected_compaction()

    resumed_cursor, resumed_descriptors = engine._ingest_cursor, engine._last_emission_descriptors
    before = engine._store.get_session_count("S0")
    reply = _turn(20)[1]
    engine.compress(kept + [reply], current_tokens=10_000)
    _assert_only_reply(engine, before, kept[0]["content"], reply)
    assert resumed_cursor == len(kept)
    assert resumed_descriptors is descriptors


def test_refused_stub_first_pass_leaves_the_accepted_proof(host_for, caplog):  # noqa: F811
    host = host_for("carrier")
    committed_session(host, "carrier")
    grow_over_threshold(host)
    engine = host.engine
    proof_key = engine._replay_snapshot_metadata_key("compaction_commit_proof")
    prior = engine._store.read_metadata_json(proof_key)
    descriptors = engine._last_emission_descriptors
    with caplog.at_level(logging.INFO):
        output, kept = stub_first_result(host, caplog)
    assert output != kept and engine._compress_commit_proof["stub_first"]
    assert engine._store.read_metadata_json(proof_key) == prior
    engine.record_rejected_compaction()
    assert engine._compress_commit_proof is None
    assert engine._last_emission_descriptors is descriptors
    resumed = engine._ingest_cursor
    before = host.stored_rows()
    reply = {"role": "assistant", "content": "A new reply after the refused stub-first pass."}
    engine.ingest(kept + [reply])
    assert host.stored_rows() == before + 1
    assert resumed == len(kept)
    assert engine._store.read_metadata_json(proof_key) == prior


@pytest.mark.parametrize("reason", ["reconcile_pending", "cursor_mismatch", "construction_error"])
def test_refused_pass_without_ready_proof_keeps_ingested_rows(committed_host, monkeypatch, reason):
    engine, kept, _binding = committed_host
    original = engine._record_compress_commit_proof

    def record(messages, result):
        if reason == "reconcile_pending":
            engine._ingest_cursor_needs_reconcile = True
        elif reason == "cursor_mismatch":
            engine._ingest_cursor -= 1
        return original(messages, result)

    def fail(*_args):
        raise ValueError("injected proof construction failure")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_record_compress_commit_proof", record)
        if reason == "construction_error":
            patch.setattr(engine, "_prepare_compress_commit_proof", fail)
        _rewriting_pass(engine, kept)
        assert engine._compress_commit_reason == reason
        assert engine._compress_commit_proof is None
    engine.record_rejected_compaction()

    resumed_cursor = (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile)
    before = engine._store.get_session_count("S0")
    reply = _turn(20)[1]
    engine.compress(kept + [reply], current_tokens=10_000)
    _assert_only_reply(engine, before, kept[0]["content"], reply)
    assert resumed_cursor == (len(kept), False)


def test_refused_identity_noop_preserves_prior_descriptors(committed_host, monkeypatch):
    engine, kept, _binding = committed_host
    engine.ingest(kept)
    descriptors, reason = engine._last_emission_descriptors, engine._compress_commit_reason
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_compress_impl", lambda messages, **_kwargs: messages)
        assert engine.compress(kept, force=True) == kept
        assert engine._compress_commit_reason == "identity_noop"
    engine.record_rejected_compaction()
    assert (engine._last_emission_descriptors, engine._compress_commit_reason) == (descriptors, reason)
    before = engine._store.get_session_count("S0")
    reply = _turn(20)[1]
    engine.compress(kept + [reply], current_tokens=10_000)
    _assert_only_reply(engine, before, kept[0]["content"], reply)


def test_only_the_latest_compress_snapshot_can_be_restored(committed_host):
    engine, kept, _binding = committed_host
    first = _rewriting_pass(engine, kept)
    first_records = _compaction_records(engine)
    kept = first + [_turn(20)[1]] + [row for i in range(21, 28) for row in _turn(i)][:-1]
    _rewriting_pass(engine, kept)
    assert _compaction_records(engine) != first_records
    engine.record_rejected_compaction()
    engine.record_rejected_compaction()
    assert _compaction_records(engine) == first_records
    before = engine._store.get_session_count("S0")
    reply = _turn(27)[1]
    engine.compress(kept + [reply], current_tokens=10_000)
    _assert_only_reply(engine, before, kept[0]["content"], reply)


def test_refusal_cannot_replace_a_newer_durable_proof(committed_host):
    engine, kept, _binding = committed_host
    _rewriting_pass(engine, kept)
    key = "compaction_commit_proof:S0"
    newer = json.loads(_compaction_records(engine)[key])
    newer["created_at"] += 1
    engine._store.write_metadata_json([key], json.dumps(newer, sort_keys=True))
    expected = _compaction_records(engine)
    engine.record_rejected_compaction()
    assert _compaction_records(engine) == expected
    assert engine._ingest_cursor == len(kept)


def test_refusal_restores_absence_of_a_durable_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    try:
        view = [row for i in range(1, 13) for row in _turn(i)] + [_turn(13)[0]]
        assert _compaction_records(engine) == {}
        engine.compress(view, force=True)
        assert _compaction_records(engine)
        engine.record_rejected_compaction()
        assert _compaction_records(engine) == {}
        before, reply = engine._store.get_session_count("S0"), _turn(13)[1]
        engine.ingest(view + [reply])
        _assert_only_reply(engine, before, None, reply)
    finally:
        engine.shutdown()


def test_refusal_restores_output_keyed_records(committed_host, monkeypatch):
    engine, kept, _binding = committed_host
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda *_a, **_kw: ("test:S0", None))
    engine.ingest(kept)
    old_note = {"role": "user", "content": "Prior recovery note."}
    recovery = [{"kind": "recovery", "output_occurrence": {"index": 0}}]
    engine._identity_anchor_record_recovery([old_note], recovery, kept)
    receipt = engine._identity_anchor_recovery
    rekey = engine._rekey_host_rewrite_watch
    captured = {}

    def record_output(messages, result, **kwargs):
        captured.update({sid: tuple(id(row) for row in objects)
                         for sid, (_, _, objects) in engine._host_rewrite_watch.items()})
        rekey(messages, result, **kwargs)
        engine._identity_anchor_record_recovery(result, recovery, messages)

    def watch_ids():
        return {sid: tuple(id(row) for row in objects)
                for sid, (_, _, objects) in engine._host_rewrite_watch.items()}

    monkeypatch.setattr(engine, "_rekey_host_rewrite_watch", record_output)
    published = _rewriting_pass(engine, kept)
    uid = published[0]["message_uid"]
    assert uid in engine._host_uid_engine_uids([uid])
    assert watch_ids() != captured
    assert receipt["blocked"]
    nodes = [node.node_id for node in engine._dag.get_session_nodes("S0")]
    frontier = engine._lifecycle.get_by_session("S0").current_frontier_store_id
    engine.record_rejected_compaction()
    assert uid not in engine._host_uid_engine_uids([uid])
    assert watch_ids() == captured
    assert engine._identity_anchor_recovery is receipt and not receipt.get("blocked")
    assert [node.node_id for node in engine._dag.get_session_nodes("S0")] == nodes
    assert engine._lifecycle.get_by_session("S0").current_frontier_store_id == frontier


def test_legacy_session_end_consumes_the_proof_before_refusal(committed_host):
    engine, kept, _binding = committed_host
    _rewriting_pass(engine, kept)
    _drain_session_end_maintenance(engine)
    engine.on_session_end("S0", kept)
    proof = engine._compress_commit_proof
    assert proof["end_consumed"]
    saved = (_compaction_records(engine), engine._last_emission_descriptors, engine._ingest_cursor)
    engine.record_rejected_compaction()
    assert engine._compress_commit_proof is proof
    assert (_compaction_records(engine), engine._last_emission_descriptors, engine._ingest_cursor) == saved
    before = engine._store.get_session_count("S0")
    reply = _turn(20)[1]
    engine.ingest(kept + [reply])
    _assert_only_reply(engine, before, kept[0]["content"], reply)


@pytest.mark.parametrize("signal", [False, True], ids=["control", "refused"])
def test_different_kept_view_preserves_the_existing_reconcile_result(committed_host, signal):
    engine, kept, _binding = committed_host
    _rewriting_pass(engine, kept)
    if signal:
        engine.record_rejected_compaction()
    before = engine._store.get_session_count("S0")
    adopted = [{"role": "user", "content": "New request from the view the host adopted."}, _turn(20)[1]]
    engine.ingest(adopted)
    rows = engine._store.get_session_messages("S0", limit=10_000)
    assert [row["content"] for row in rows[before:]] == [row["content"] for row in adopted]
    assert engine._last_ingest_reconciliation["reason"] == "persisted ambiguous delta"


def test_ledger_uses_the_last_adopted_view_after_refusal(committed_host):
    engine, kept, _binding = committed_host
    store = engine._store._conn
    adopted_handle = engine._emission_handle
    assert adopted_handle is not None
    _rewriting_pass(engine, kept)
    refused_handle = engine._emission_handle
    assert refused_handle.view_seq > adopted_handle.view_seq
    engine.record_rejected_compaction()

    assert ingest_ledger.emission_candidate(store, refused_handle.lineage) == (None, "refused")
    attempt = engine._prepare_ingest_ledger(kept, {"ledger_digest_ms": 0.0})
    _, counts = ingest_ledger.cold_cursor_candidate(
        store, session_id="S0", attempt=attempt, ignored=[False] * len(kept))
    assert counts["emission_withheld.refused"] == 0
    view, reason = ingest_ledger.emission_candidate(store, refused_handle.lineage, adopted_fallback=True)
    assert reason == "usable"
    assert (view["view_seq"], view["state"]) == (adopted_handle.view_seq, "adopted")


@pytest.mark.parametrize("committed_host", [5], indirect=True)
def test_repeated_refusals_store_no_verified_summary_head(committed_host):
    engine, kept, _binding = committed_host
    head = kept[0]["content"]
    assert engine._verified_lcm_summary_prefix_end(head) == len(head)
    reply = _turn(20)[1]
    _rewriting_pass(engine, kept)
    engine.record_rejected_compaction()
    kept.append(reply)
    for i in range(21, 24):
        kept.extend(_turn(i))
        engine.compress(kept, current_tokens=10_000)
        engine.record_rejected_compaction()
    rows = engine._store.get_session_messages("S0", limit=10_000)
    nodes = {node.node_id: (node.session_id, node.depth, node.summary, node.expand_hint)
             for node in engine._dag.get_session_nodes("S0")}

    def strip(content, session):
        return carrier_remainder(content, engine._LCM_SUMMARY_PART_HEADER_RE, nodes, lambda sid: sid == session)

    assert strip(head, "S0") == ""
    assert verified_summary_head_store_ids(
        (tuple(row[field] for field in ("store_id", "session_id", "role", "content")) for row in rows), strip) == []
