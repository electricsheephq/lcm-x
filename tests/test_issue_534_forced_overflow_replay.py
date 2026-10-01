"""#534: a forced-overflow recovery's returned list is durable provenance (exact prefix digest).

Cold restarts re-index the exact recovery list instead of re-storing generated rows.
Helper-seam boundary cases are controls, not real-host warm-boundary reproductions.
Placeholder cases cover the helper seam only. Provenance never skips a row for its text.
"""
from collections import Counter
from copy import deepcopy

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.engine import _OVERFLOW_RECOVERY_OVERCAP_NOTE, count_messages_tokens
from hermes_lcm.reconcile import _COMPACTION_COMMIT_PROOF_METADATA_PREFIX
from tests.test_issue_524_gen2_replay import CID, SID, _compress, _engine, _transcript

CAP = 120
BIG = "oversized assistant/tool chatter " * 400
CHILD = "issue-534-child"
SYSTEM = {"role": "system", "content": "You are an agent."}


@pytest.fixture
def provider(monkeypatch):
    def stub(*_a, **_k):  # a summary larger than CAP, so the recovery reaches its fallback rows
        return "Stub summary " + "sumword " * 600 + "\nExpand for details about: stub", 1

    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", stub)
    return stub


def _host(kind, prefix=()):
    tail = {
        "note": [
            {"role": "user", "content": "OLDER_ASK fits"},
            {"role": "assistant", "content": BIG},
            {"role": "user", "content": "NEWEST_ASK over cap " * 300},
            {"role": "tool", "tool_call_id": "orphan", "content": "status"},
        ],
        "placeholder": [
            {"role": "tool", "tool_call_id": "orphan-a", "content": "a"},
            {"role": "tool", "tool_call_id": "orphan-b", "content": "b"},
        ],
    }[kind]
    return [dict(SYSTEM), *prefix, *tail]


def _recover(engine, host):
    """compress()'s forced-overflow no-leaf branch (compaction.py), on an ingested host list."""
    engine._ingest_messages(host)
    anchors = engine._leading_anchor_count(host)
    out = engine._assemble_overflow_recovery_context(
        host[0] if anchors else None, host[anchors:], assembly_cap_override=CAP,
        **({"retained_user_message": host[1]} if anchors == 2 else {}),
    )
    return deepcopy(engine._finalize_forced_overflow_result(host, out, assembly_cap_override=CAP))


def _generated(out):
    rows = [m["content"] for m in out if m["content"].startswith("[LCM overflow recovery]")]
    assert rows, out
    return rows


def _counts(engine, *sessions):
    return Counter(r["content"] for s in sessions for r in engine._store.get_session_messages(s))


TURN = [{"role": "assistant", "content": "A1 reply"}, {"role": "user", "content": "U2 ask"}]


@pytest.mark.parametrize("prior_compaction", [False, True], ids=["no-proof", "v4-proof"])
@pytest.mark.parametrize("kind", ["note", "placeholder"])
def test_cold_restart_reindexes_the_recovery_output(tmp_path, provider, kind, prior_compaction):
    engine = _engine(tmp_path)
    prefix = ()
    if prior_compaction:
        adopted = _compress(engine, _transcript())
        engine._ingest_messages(adopted)
        assert (engine._active_emission_proof() or {}).get("version") == 4
        if kind == "note":  # the host continues the adopted list
            prefix = [m for m in adopted if m.get("role") != "system"]
    out = _recover(engine, _host(kind, prefix))
    before = _counts(engine, SID)
    engine.shutdown()

    replay = _engine(tmp_path)
    try:
        replay._ingest_messages(out + TURN)
        after = _counts(replay, SID)
        assert replay._last_ingest_reconciliation["cursor"] == len(out)
    finally:
        replay.shutdown()

    assert all(after[text] == 0 for text in _generated(out))
    assert after - before == Counter({"A1 reply": 1, "U2 ask": 1})


@pytest.mark.parametrize("restart", [False, True], ids=["warm", "cold"])
@pytest.mark.parametrize("child", [CHILD, SID], ids=["rotate", "in-place"])
@pytest.mark.parametrize("kind", ["note", "placeholder"])
def test_compression_boundary_after_recovery_stores_only_new_turns(tmp_path, provider, kind, child, restart):
    engine = _engine(tmp_path)
    host = _host(kind)
    out = _recover(engine, host)
    before = _counts(engine, SID, CHILD)
    engine.on_session_end(SID, out)
    engine.on_session_start(child, boundary_reason="compression", old_session_id=SID,
                            platform="cli", conversation_id=CID, context_length=200_000)
    if restart:
        engine.shutdown()
        engine = _engine(tmp_path, session_id=child)
    try:
        engine._ingest_messages(out + TURN)
        after = _counts(engine, SID, CHILD)
    finally:
        engine.shutdown()

    assert all(after[text] == 0 for text in _generated(out))
    assert after - before == Counter({"A1 reply": 1, "U2 ask": 1})


def test_later_leaf_compaction_boundary_still_carries_its_commit_proof(tmp_path, provider):
    """The recovery digest never steers a boundary: a later compaction keeps its proof carry."""
    engine = _engine(tmp_path)
    out = _recover(engine, _host("note"))
    host = out + _transcript()
    engine._ingest_messages(host)
    compacted = deepcopy(_compress(engine, host))
    assert engine._last_compression_status == "compacted"
    engine.on_session_end(SID, host)
    engine.on_session_start(CHILD, boundary_reason="compression", old_session_id=SID,
                            platform="cli", conversation_id=CID, context_length=200_000)
    try:
        assert engine._compress_commit_proof is not None
        assert (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile) == (len(compacted), False)
        engine._ingest_messages(compacted + TURN)
        assert [r["content"] for r in engine._store.get_session_messages(CHILD)] == ["A1 reply", "U2 ask"]
    finally:
        engine.shutdown()


def test_a_user_row_with_the_note_bytes_after_the_recovery_is_stored(tmp_path, provider):
    """Provenance, not text: the same bytes typed by the user after the emitted list are new."""
    engine = _engine(tmp_path)
    out = _recover(engine, _host("note"))
    note = _generated(out)[0]
    engine.shutdown()
    replay = _engine(tmp_path)
    try:
        replay._ingest_messages(out + [{"role": "assistant", "content": "A1 reply"}, {"role": "user", "content": note}])
        assert _counts(replay, SID)[note] == 1
    finally:
        replay.shutdown()


def test_an_unemitted_generated_row_is_still_stored(tmp_path, provider):
    """No digest, no skip: the #533 duplicate-not-loss direction holds for rows LCM never returned."""
    engine = _engine(tmp_path)
    engine._ingest_messages([dict(SYSTEM), {"role": "user", "content": "First"}, {"role": "assistant", "content": "Reply"}])
    engine.shutdown()
    note = _OVERFLOW_RECOVERY_OVERCAP_NOTE.format(tokens=count_messages_tokens([{"role": "user", "content": "x"}]), cap=CAP)
    replay = _engine(tmp_path)
    try:
        replay._ingest_messages([dict(SYSTEM), {"role": "user", "content": "First"}, {"role": "assistant", "content": "Reply"},
                                 {"role": "user", "content": note}])
        assert _counts(replay, SID)[note] == 1
    finally:
        replay.shutdown()


def test_residual_turns_ingested_after_the_recovery_are_duplicated_on_restart(tmp_path, provider):
    """Pinned residual (duplicate direction, follow-up): the exact-prefix term stops at the emitted list,
    so turns this process already stored after it are stored again on a cold restart."""
    engine = _engine(tmp_path)
    out = _recover(engine, _host("note"))
    engine._ingest_messages(out + TURN)
    engine.shutdown()
    replay = _engine(tmp_path)
    try:
        replay._ingest_messages(out + TURN + [{"role": "assistant", "content": "A2 reply"}])
        counts = _counts(replay, SID)
    finally:
        replay.shutdown()
    assert all(counts[text] == 0 for text in _generated(out))
    assert (counts["A1 reply"], counts["U2 ask"], counts["A2 reply"]) == (2, 2, 1)


# The verification probes below drive compress() and the real boundary hooks.
def _real_engine(tmp_path, session_id=SID):
    return _engine(tmp_path, session_id=session_id, max_assembly_tokens=CAP)


def _real_recover(engine, host):
    engine.ingest(deepcopy(host))
    out = deepcopy(engine.compress(deepcopy(host), current_tokens=count_messages_tokens(host)))
    assert engine._last_compression_status == "overflow_recovery"
    return out


@pytest.mark.parametrize("boundary", ["none", "in-place", "rotate"])
def test_real_compress_cold_restart_reindexes_recovery(tmp_path, provider, boundary):
    engine = _real_engine(tmp_path)
    host = _host("note")
    out = _real_recover(engine, host)
    before = _counts(engine, SID, CHILD)
    session_id = CHILD if boundary == "rotate" else SID
    if boundary != "none":
        engine.on_session_end(SID, deepcopy(host))
        engine.on_session_start(
            session_id, boundary_reason="compression", old_session_id=SID,
            platform="cli", conversation_id=CID, context_length=200_000,
        )
        if boundary == "rotate":
            assert engine._store.get_session_count(CHILD) == 0
    engine.shutdown()

    replay = _real_engine(tmp_path, session_id=session_id)
    try:
        replay.ingest(deepcopy(out + TURN))
        after = _counts(replay, SID, CHILD)
        assert replay._last_ingest_reconciliation["cursor"] == len(out)
        assert all(after[text] == 0 for text in _generated(out))
        assert after - before == Counter({"A1 reply": 1, "U2 ask": 1})
    finally:
        replay.shutdown()


def test_real_warm_turn_then_cold_restart_pins_duplicate_residual(tmp_path, provider):
    engine = _real_engine(tmp_path)
    out = _real_recover(engine, _host("note"))
    live = out + TURN
    engine.ingest(deepcopy(live))
    engine.shutdown()
    replay = _real_engine(tmp_path)
    try:
        replay.ingest(deepcopy(live + [
            {"role": "assistant", "content": "A2 reply"},
            {"role": "user", "content": "U3 ask"},
        ]))
        counts = _counts(replay, SID)
        assert all(counts[text] == 0 for text in _generated(out))
        assert (counts["A1 reply"], counts["U2 ask"]) == (2, 2)
        assert (counts["A2 reply"], counts["U3 ask"]) == (1, 1)
    finally:
        replay.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["warm", "cold"])
def test_reset_same_session_preserves_new_continue_row(tmp_path, provider, restart):
    engine = _real_engine(tmp_path)
    host = [dict(SYSTEM), {"role": "user", "content": "continue"},
            {"role": "assistant", "content": BIG}]
    out = _real_recover(engine, host)
    assert out == [dict(SYSTEM), {"role": "user", "content": "continue"}]
    assert _counts(engine, SID)["continue"] == 1
    engine.on_session_reset()
    for prefix in (_COMPACTION_COMMIT_PROOF_METADATA_PREFIX,
                   "forced_overflow_replay_snapshot_digests"):
        assert engine._store.read_metadata_json(engine._replay_snapshot_metadata_key(prefix)) is None
    if restart:
        engine.shutdown()
        engine = _real_engine(tmp_path)
    else:
        engine.on_session_start(SID, platform="cli", conversation_id=CID, context_length=200_000)
    try:
        engine.ingest([dict(SYSTEM), {"role": "user", "content": "continue"},
                       {"role": "assistant", "content": "NEW reply"}])
        counts = _counts(engine, SID)
        assert counts["continue"] == 2
        assert counts["NEW reply"] == 1
        assert engine._last_ingest_reconciliation["cursor"] == 0
    finally:
        engine.shutdown()


def test_real_in_place_boundary_warm_turn_then_cold_restart(tmp_path, provider):
    engine = _real_engine(tmp_path)
    host = _host("note")
    out = _real_recover(engine, host)
    engine.on_session_end(SID, deepcopy(host))
    engine.on_session_start(
        SID, boundary_reason="compression", old_session_id=SID,
        platform="cli", conversation_id=CID, context_length=200_000,
    )
    live = out + TURN
    engine.ingest(deepcopy(live))
    engine.shutdown()
    replay = _real_engine(tmp_path)
    try:
        replay.ingest(deepcopy(live + [{"role": "assistant", "content": "A2 reply"}]))
        counts = _counts(replay, SID)
        assert all(counts[text] == 0 for text in _generated(out))
        assert (counts["A1 reply"], counts["U2 ask"], counts["A2 reply"]) == (2, 2, 1)
    finally:
        replay.shutdown()


def test_host_rejects_recovery_original_list_is_not_skipped(tmp_path, provider):
    engine = _real_engine(tmp_path)
    host = _host("note")
    out = _real_recover(engine, host)
    before = _counts(engine, SID)
    engine.shutdown()
    replay = _real_engine(tmp_path)
    try:
        replay.ingest(deepcopy(host + [{"role": "assistant", "content": "A1 reply"}]))
        after = _counts(replay, SID)
        assert after - before == Counter({"A1 reply": 1})
        assert replay._last_ingest_reconciliation["cursor"] == len(host)
        assert all(after[text] == 0 for text in _generated(out))
    finally:
        replay.shutdown()


def test_two_real_recoveries_then_cold_restart(tmp_path, provider):
    engine = _real_engine(tmp_path)
    out = _real_recover(engine, _host("note"))
    host = out + [
        {"role": "assistant", "content": BIG + " v2"},
        {"role": "user", "content": "SECOND over cap " * 300},
    ]
    out2 = _real_recover(engine, host)
    before = _counts(engine, SID)
    engine.shutdown()
    replay = _real_engine(tmp_path)
    try:
        replay.ingest(deepcopy(out2 + [{"role": "assistant", "content": "A1 reply"}]))
        after = _counts(replay, SID)
        assert after - before == Counter({"A1 reply": 1})
        assert replay._last_ingest_reconciliation["cursor"] == len(out2)
    finally:
        replay.shutdown()
