"""Synthetic, offline origin-reference acceptance and preservation controls."""
import copy
import hashlib
import json
import socket
from pathlib import Path

import pytest

import hermes_lcm.engine as em
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig

ORIGIN = "  Synthetic initial request \t\nUnicode café 日本語\n"
LATEST = "Cancel the initial task. Perform the synthetic replacement instead."
HISTORICAL = "[Session-start user request (historical reference). Later user messages take precedence; do not resume cancelled or replaced work.]"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(em, "summarize_with_escalation", lambda **kw: (
        "Synthetic history without request bytes.\nExpand for details about: fixture", 1))
    def forbidden(*args, **kwargs):
        raise AssertionError("NETWORK_FORBIDDEN")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    instance = em.LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "db.sqlite"), fresh_tail_count=4,
        leaf_chunk_tokens=1, large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False, temporal_rollups_enabled=False,
        embeddings_enabled=False, proactive_recall_enabled=False, background_leaf_prep_enabled=False,
    ), hermes_home=str(tmp_path / "home"))
    instance.on_session_start("S", platform="cli", context_length=200000)
    yield instance
    instance.shutdown()
    tokens._count_tokens_cached.cache_clear()


def rounds(start=0, count=8):
    rows = []
    for i in range(start, start + count):
        rows.extend([
            {"role": "assistant", "content": "Synthetic read", "tool_calls": [{
                "id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "Synthetic data " * 100},
        ])
    return rows


def messages(with_system=False):
    return ([{"role": "system", "content": "Synthetic host policy"}] if with_system else []) + [
        {"role": "user", "content": ORIGIN},
        {"role": "assistant", "content": "Synthetic interim reply"},
        {"role": "user", "content": LATEST}, *rounds(),
    ]


def assert_origin(result, latest=LATEST):
    carriers = [r for r in result if r.get("role") in {"user", "assistant"}]
    count = sum(str(r.get("content", "")).encode().count(ORIGIN.encode()) for r in carriers)
    assert count == 1, "whole origin byte count must be one"
    assert any(HISTORICAL + "\n" + ORIGIN in str(r.get("content", "")) for r in carriers), "historical carrier missing"
    assert any(latest in str(r.get("content", "")) for r in result), "latest instruction missing"


@pytest.mark.parametrize("with_system", [False, True])
def test_origin_survives_first_compaction(engine, with_system):
    source = messages(with_system)
    original = copy.deepcopy(source)
    result = engine.compress(source, force=True)
    assert_origin(result)
    stored = engine._durable_real_user_messages()
    assert [r["content"] == s for r, s in zip(stored, (ORIGIN, LATEST))] == [True, True]
    assert source == original


def test_origin_survives_reassembly(engine):
    first = engine.compress(messages(), force=True)
    second = engine.compress([*first, *rounds(8)], force=True)
    assert_origin(second)
    users = engine._durable_real_user_messages(stop_after=100)
    assert len(users) == 2, "generated context must not become a user occurrence"


def test_origin_survives_proven_rotation_resume(engine):
    first = engine.compress(messages(), force=True)
    engine.on_session_start("S2", platform="cli", context_length=200000,
                            boundary_reason="compression", old_session_id="S")
    config, home = engine._config, engine._hermes_home
    engine.shutdown()
    reopened = em.LCMEngine(config=config, hermes_home=home)
    try:
        reopened.on_session_start("S2", platform="cli", context_length=200000)
        result = reopened.compress([*first, *rounds(8)], force=True)
        assert_origin(result)
    finally:
        reopened.shutdown()


def test_replay_merged_suffix_and_literal_header(engine):
    first = engine.compress(messages(), force=True)
    suffix = "Synthetic genuine new request."
    merged = [{**first[0], "content": first[0]["content"] + "\n\n" + suffix}, *first[1:]]
    second = engine.compress([*merged, *rounds(8)], force=True)
    assert_origin(second, suffix)
    users = engine._durable_real_user_messages(stop_after=100)
    assert sum(r.get("content") == suffix for r in users) == 1
    literal = {"role": "user", "content": HISTORICAL + "\nA literal user quote."}
    engine.compress([*second, literal, *rounds(16)], force=True)
    users = engine._durable_real_user_messages(stop_after=100)
    assert sum(r.get("content") == literal["content"] for r in users) == 1


def test_origin_budget_trial_and_incomplete_lineage(engine):
    first = engine.compress(messages(), force=True)
    before = engine._store._conn.serialize()
    pending = copy.deepcopy(engine._pending_emission_candidates)
    result = engine._assemble_context(None, first[-4:], persist=False)
    assert any(ORIGIN in str(r.get("content", "")) for r in result)
    assert engine._store._conn.serialize() == before
    assert engine._pending_emission_candidates == pending
    cap = tokens.count_messages_tokens(first[-4:]) + 25
    result = engine._assemble_context(None, first[-4:], assembly_cap_override=cap, persist=False)
    assert not any(HISTORICAL in str(r.get("content", "")) for r in result)
    engine._store.write_metadata_json([engine._rotation_predecessor_metadata_key("S")], json.dumps("missing"))
    assert engine._origin_reference_part([], []) is None


def test_new_conversation_does_not_inherit_origin(engine):
    engine.compress(messages(), force=True)
    engine.on_session_start("new", platform="cli", context_length=200000)
    result = engine.compress([{"role": "user", "content": "Synthetic new task"}, *rounds(8)], force=True)
    assert not any(ORIGIN in str(r.get("content", "")) for r in result)


def test_equal_text_is_a_distinct_occurrence(engine):
    first = engine.compress(messages(), force=True)
    result = engine.compress([*first, {"role": "user", "content": ORIGIN}, *rounds(8)], force=True)
    users = engine._durable_real_user_messages(stop_after=100)
    assert sum(r.get("content") == ORIGIN for r in users) == 2
    assert any(HISTORICAL in str(r.get("content", "")) for r in result)


def test_missing_rotation_predecessor_omits_instead_of_guessing(engine):
    first = engine.compress(messages(), force=True)
    engine.on_session_start("S2", platform="cli", context_length=200000,
                            boundary_reason="compression", old_session_id="S")
    engine.compress([*first, {"role": "user", "content": "Synthetic child task"}, *rounds(8)], force=True)
    engine._store.write_metadata_json([engine._rotation_predecessor_metadata_key("S2")], "null")
    assert engine._origin_reference_part([], []) is None


def test_reset_starts_new_origin(engine):
    engine.compress(messages(), force=True)
    engine.on_session_reset()
    engine.on_session_start("reset", platform="cli", context_length=200000)
    source = messages()
    source[0] = {"role": "user", "content": "Synthetic post-reset origin"}
    result = engine.compress(source, force=True)
    assert not any(ORIGIN in str(r.get("content", "")) for r in result)
    assert any(HISTORICAL + "\nSynthetic post-reset origin" in str(r.get("content", "")) for r in result)


def test_no_emission_proof_omits_origin_reference(engine):
    first = engine.compress(messages(), force=True)
    engine._last_emission_descriptors = None
    engine._store.write_metadata_json([engine._replay_snapshot_metadata_key("compaction_commit_proof")], "null")
    assert engine._origin_reference_part([], first) is None


def test_unreadable_ancestor_does_not_select_a_child_origin(engine, monkeypatch):
    first = engine.compress(messages(), force=True)
    engine.on_session_start("S2", platform="cli", context_length=200000,
                            boundary_reason="compression", old_session_id="S")
    engine.ingest([*first, {"role": "user", "content": "Synthetic child one"},
                   {"role": "user", "content": "Synthetic child two"}])
    original = engine._store.load_session_page
    def unavailable(session, **kwargs):
        if session == "S":
            raise OSError("synthetic ancestor unavailable")
        return original(session, **kwargs)
    monkeypatch.setattr(engine._store, "load_session_page", unavailable)
    assert engine._origin_reference_part([], []) is None


def assembly_fixture_output(case, root, patch):
    """Capture every field of returned rows and finalized descriptors, with seeded IDs."""
    from tests.behaviour import assembly_trace_harness as h
    from hermes_lcm.emission_projection import _finalize_emission_descriptors
    capture = {}
    original = em.LCMEngine._assemble_context
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        canonical = h.Canonical(root)
        capture.update(canonical.value({
            "messages": result,
            "descriptors": _finalize_emission_descriptors(
                result, self._pending_emission_candidates or [], self._emission_binding()),
        }))
        return result
    with patch.context() as scoped:
        scoped.setattr(em.LCMEngine, "_assemble_context", wrapped)
        h.observe(case, root)
    return capture


from tests.behaviour.assembly_trace_harness import SCENARIOS  # noqa: E402


@pytest.mark.parametrize("case", SCENARIOS, ids=lambda case: case.name)
def test_existing_assembly_fixture_identity(case, tmp_path, monkeypatch):
    expected = json.loads((Path(__file__).parent / "fixtures" / "issue_1592_assembly_identity.json").read_text())
    actual = assembly_fixture_output(case, tmp_path / case.name, monkeypatch)
    if case.name == "compress-scaffold":
        # This sole fixture compacts O out of the raw tail: its new reference is
        # the intentional change, rather than a no-op identity precondition.
        from tests.behaviour.assembly_trace_harness import PAD
        assert sum(("Turn 0" + PAD) in str(m.get("content", "")) for m in actual["messages"]) == 1
        assert any(HISTORICAL in str(m.get("content", "")) for m in actual["messages"])
        return
    # Full serialized rows and descriptors are compared; no field is removed.
    expected_digest = hashlib.sha256(json.dumps(expected[case.name], sort_keys=True).encode()).hexdigest()
    actual_digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    assert actual_digest == expected_digest, "complete output/descriptor identity changed"
