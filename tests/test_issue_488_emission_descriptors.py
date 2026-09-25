from __future__ import annotations

import hashlib

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _project_emitted_occurrences
from tests.test_compression_boundary import _summary_carrier_fixture, _turn
from tests.test_issue_488_emission_proof import (
    OBJECTIVE,
    _phase1_compacted_engine,
    _real_hermes_merge,
)


def _record(engine, before, returned):
    engine._ingest_cursor = len(returned)
    engine._ingest_cursor_needs_reconcile = False
    engine._last_compression_status = "compacted"
    engine._record_compress_commit_proof(before, returned)
    proof = engine._compress_commit_proof
    assert proof is not None
    return proof


def _carry_forward_through_real_ingest(engine, monkeypatch, first, second):
    engine.ingest(first)
    assert engine._compress_commit_proof is None
    host = [dict(message) for message in second]
    host.append(_turn(99)[1])
    engine.ingest(host)

    def compact_again(_messages, **_kwargs):
        engine._last_compression_status = "compacted"
        engine._ingest_cursor = len(second)
        engine._pending_emission_candidates = []
        return second

    monkeypatch.setattr(engine, "_compress_impl", compact_again)
    returned = engine.compress(host, force=True)
    assert engine._last_compression_status == "compacted"
    assert engine._compress_commit_proof is not None
    return returned, engine._compress_commit_proof


def _assert_exact_descriptors(returned, proof):
    descriptors = proof["emissions"]
    assert descriptors
    assert len(descriptors) == len({d["output_occurrence"]["index"] for d in descriptors})
    for descriptor in descriptors:
        content = returned[descriptor["output_occurrence"]["index"]]["content"]
        raw = content.encode("utf-8")
        span = raw[: descriptor["generated_span_bytes"]]
        assert hashlib.sha256(span).hexdigest() == descriptor["generated_span_sha256"]


def _scoped_proof(kind, span, suffix, *, output=None, retained_source=None):
    scope = {
        "hermes_home": "home",
        "session_id": "session",
        "conversation_id": "conversation",
        "reset_epoch": None,
    }
    normalized_suffix = suffix.strip().encode("utf-8")
    identity = ("user", span + suffix, "", "[]", "")
    return {
        "version": 4,
        **scope,
        "output": output if output is not None else [identity],
        "emissions": [{
            "kind": kind,
            "role": "user",
            "same_prefix_ordinal": 0,
            "output_occurrence": {"index": 0, "same_identity_ordinal": 0},
            "generated_span_sha256": hashlib.sha256(span.encode("utf-8")).hexdigest(),
            "generated_span_bytes": len(span.encode("utf-8")),
            "suffix_sha256": hashlib.sha256(normalized_suffix).hexdigest(),
            "suffix_length": len(normalized_suffix),
            "retained_source": retained_source,
            "scope": scope,
        }],
    }


def test_descriptor_exactness_for_carrier_and_system_layouts(tmp_path):
    engine, compacted, tail, tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        carrier_output = engine._assemble_context(None, tail)
        carrier_proof = _record(engine, [compacted, *tail], carrier_output)
        _assert_exact_descriptors(carrier_output, carrier_proof)
        carrier = carrier_proof["emissions"]
        assert [item["kind"] for item in carrier] == ["carrier"]
        assert carrier[0]["generated_span_bytes"] == len(
            carrier_output[0]["content"][: -len(tail[0]["content"])].encode("utf-8")
        )
        assert carrier[0]["retained_source"] == {"store_id": tail_ids[0]}

        system_output = engine._assemble_context(
            {"role": "system", "content": "system"}, tail, include_lcm_note=False
        )
        system_proof = _record(engine, [compacted, *tail], system_output)
        _assert_exact_descriptors(system_output, system_proof)
        assert [item["kind"] for item in system_proof["emissions"]] == ["summary"]
    finally:
        engine.shutdown()


def test_descriptor_exactness_for_objective_and_cap_trimmed_output(tmp_path):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        engine._pending_context_anchor_messages = [
            {"role": "user", "content": "objective outside the selected tail"},
            {"role": "assistant", "content": "work"},
        ]
        objective_output = engine._assemble_context(
            {"role": "system", "content": "system"}, tail, include_lcm_note=False
        )
        objective_proof = _record(engine, [compacted, *tail], objective_output)
        _assert_exact_descriptors(objective_output, objective_proof)
        assert objective_proof["emissions"][0]["kind"] == "objective"
        assert OBJECTIVE in objective_output[1]["content"]

        trimmed_output = engine._assemble_context(
            {"role": "system", "content": "system"},
            tail,
            include_lcm_note=False,
            assembly_cap_override=80,
        )
        trimmed_proof = _record(engine, [compacted, *tail], trimmed_output)
        _assert_exact_descriptors(trimmed_output, trimmed_proof)
    finally:
        engine._pending_context_anchor_messages = None
        engine.shutdown()


def test_projection_is_deterministic_copy_safe_and_merge_safe(tmp_path):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        returned = engine._assemble_context(None, tail)
        proof = _record(engine, [compacted, *tail], returned)
        first = _project_emitted_occurrences(returned, proof=proof)
        copied = _project_emitted_occurrences([dict(row) for row in returned], proof=proof)
        assert first == copied
        assert first == _project_emitted_occurrences(returned, proof=proof)

        merged_rows = _real_hermes_merge(
            [dict(returned[0]), {"role": "user", "content": "host-appended suffix"}]
        )
        merged = _project_emitted_occurrences(merged_rows, proof=proof)
        assert merged.entries[0].kind == "carrier"
        assert merged.entries[0].retained_source == first.entries[0].retained_source
        assert merged.entries[0].effective_identity[1].endswith("host-appended suffix")
    finally:
        engine.shutdown()


def test_projection_is_monotonic_one_to_one_and_slice_preserves_entries(tmp_path):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        returned = engine._assemble_context(None, tail)
        proof = _record(engine, [compacted, *tail], returned)
        authored = dict(returned[0])
        rows = [*returned, authored]
        projection = _project_emitted_occurrences(rows, proof=proof)
        assert projection.entries[0].kind == "carrier"
        assert projection.entries[-1].kind is None
        assert projection.entries[-1].effective_identity == projection.entries[-1].full_identity
        assert projection.slice(1, 2).entries == projection.entries[1:3]
        assert projection.slice(1).offset == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("version", [2, 3])
def test_legacy_proof_authorizes_no_emission_projection(version):
    proof = {
        "version": version,
        "emissions": [{
            "kind": "carrier",
            "generated_span_sha256": hashlib.sha256(b"summary\n\n").hexdigest(),
            "generated_span_bytes": len(b"summary\n\n"),
        }],
    }
    projection = _project_emitted_occurrences(
        [{"role": "user", "content": "summary\n\nauthored"}], proof=proof
    )
    assert all(entry.generated_span is None for entry in projection.entries)
    assert all(entry.effective_identity == entry.full_identity for entry in projection.entries)


@pytest.mark.parametrize("missing_field", ["role", "same_prefix_ordinal"])
def test_v4_descriptor_without_occurrence_provenance_is_ignored(missing_field):
    scope = {
        "hermes_home": "home",
        "session_id": "session",
        "conversation_id": "conversation",
        "reset_epoch": None,
    }
    descriptor = {
        "kind": "carrier",
        "role": "user",
        "same_prefix_ordinal": 0,
        "output_occurrence": {"index": 0, "same_identity_ordinal": 0},
        "generated_span_sha256": hashlib.sha256(b"summary\n\n").hexdigest(),
        "generated_span_bytes": len(b"summary\n\n"),
        "scope": scope,
    }
    del descriptor[missing_field]
    projection = _project_emitted_occurrences(
        [{"role": "user", "content": "summary\n\nauthored"}],
        proof={"version": 4, **scope, "emissions": [descriptor]},
    )
    assert projection.entries[0].generated_span is None
    assert projection.entries[0].effective_identity == projection.entries[0].full_identity


def test_emission_descriptor_carries_forward_to_the_next_proof(tmp_path):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        first = engine._assemble_context(None, tail)
        first_proof = _record(engine, [compacted, *tail], first)
        original = first_proof["emissions"][0]

        engine._pending_emission_candidates = []
        second = [first[0], *first[2:]]
        second_proof = _record(engine, first, second)
        carried = second_proof["emissions"][0]
        assert carried["generated_span_sha256"] == original["generated_span_sha256"]
        assert carried["generated_span_bytes"] == original["generated_span_bytes"]
        assert carried["retained_source"] == original["retained_source"]
    finally:
        engine.shutdown()


def test_compress_preserves_live_placeholder_provenance(tmp_path, monkeypatch):
    engine, _compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    generated = {"role": "assistant", "content": "ignored generated placeholder"}
    try:
        engine._last_active_replay_messages = [dict(generated)]
        engine._generated_ignored_active_replay_placeholder_message_ids = {id(generated)}
        before = set(engine._generated_ignored_active_replay_placeholder_message_ids)
        monkeypatch.setattr(engine, "_compress_impl", lambda messages, **_kwargs: messages)

        returned = engine.compress([generated], force=True)

        assert returned[0] is generated
        assert engine._generated_ignored_active_replay_placeholder_message_ids == before
    finally:
        engine.shutdown()


def test_real_ingest_keeps_emission_descriptor_for_next_compress(tmp_path, monkeypatch):
    engine, _pre, first = _phase1_compacted_engine(tmp_path, monkeypatch, tail=7)
    try:
        original = engine._compress_commit_proof["emissions"][0]
        original_content = first[original["output_occurrence"]["index"]]["content"]
        original_span = original_content.encode("utf-8")[: original["generated_span_bytes"]]
        engine.ingest(first)
        assert engine._compress_commit_proof is None

        host = [dict(message) for message in first]
        host.append(_turn(13)[1])
        engine.ingest(host)
        host.extend(_turn(14))
        engine.ingest(host)
        surviving = [first[0], *first[2:], *host[len(first):]]

        def compact_again(_messages, **_kwargs):
            engine._last_compression_status = "compacted"
            engine._ingest_cursor = len(surviving)
            engine._pending_emission_candidates = []
            return surviving

        monkeypatch.setattr(engine, "_compress_impl", compact_again)
        second = engine.compress(host, force=True)
        assert engine._last_compression_status == "compacted"
        assert any(
            isinstance(message.get("content"), str)
            and message["content"].encode("utf-8").startswith(original_span)
            for message in second
        )
        assert any(
            descriptor["generated_span_sha256"] == original["generated_span_sha256"]
            and descriptor["generated_span_bytes"] == original["generated_span_bytes"]
            and descriptor["retained_source"] == original["retained_source"]
            for descriptor in engine._compress_commit_proof["emissions"]
        ), second
    finally:
        engine.shutdown()


def test_carry_forward_keeps_generated_user_when_system_has_same_span(tmp_path, monkeypatch):
    engine, compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        generated = engine._assemble_context(None, [])
        span = generated[0]["content"]
        first = engine._assemble_context(
            {"role": "system", "content": span}, [], include_lcm_note=False
        )
        first_proof = _record(engine, [compacted], first)
        assert first_proof["emissions"][0]["output_occurrence"]["index"] == 1

        second, second_proof = _carry_forward_through_real_ingest(
            engine, monkeypatch, first, [dict(message) for message in first]
        )
        projection = _project_emitted_occurrences(second, proof=second_proof)

        assert second_proof["emissions"][0]["output_occurrence"]["index"] == 1
        assert second_proof["emissions"][0]["role"] == "user"
        assert projection.entries[0].generated_span is None
        assert projection.entries[1].generated_span == span
    finally:
        engine.shutdown()


def test_carry_forward_keeps_rewritten_generated_user_after_authored_prefix(
    tmp_path, monkeypatch
):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        generated = engine._assemble_context(None, tail)
        span = engine._pending_emission_candidates[0]["span"]
        first = [{"role": "user", "content": span}, *generated]
        first_proof = _record(engine, [compacted, *tail], first)
        assert first_proof["emissions"][0]["output_occurrence"]["index"] == 1

        rewritten = [dict(message) for message in first]
        original_suffix = generated[0]["content"][len(span):]
        rewritten[1]["content"] = span + original_suffix + "\n\nnew host turn"
        second, second_proof = _carry_forward_through_real_ingest(
            engine, monkeypatch, first, rewritten
        )
        projection = _project_emitted_occurrences(second, proof=second_proof)

        assert second_proof["emissions"][0]["output_occurrence"]["index"] == 1
        assert second_proof["emissions"][0]["same_prefix_ordinal"] == 1
        assert projection.entries[0].generated_span is None
        assert projection.entries[1].effective_identity[1] == original_suffix + "\n\nnew host turn"
    finally:
        engine.shutdown()


def test_carry_forward_binds_identical_same_role_generated_row(
    tmp_path, monkeypatch
):
    engine, compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        generated = engine._assemble_context(None, [])
        assert len(generated) == 1 and generated[0]["role"] == "user"
        span = generated[0]["content"]
        first = [{"role": "user", "content": span}, *generated]
        first_proof = _record(engine, [compacted], first)

        assert first_proof["emissions"][0]["output_occurrence"]["index"] == 1
        assert first_proof["emissions"][0]["same_prefix_ordinal"] == 1

        rewritten = [dict(message) for message in first]
        rewritten[1]["content"] = span + "\n\nnew host turn"
        second, second_proof = _carry_forward_through_real_ingest(
            engine, monkeypatch, first, rewritten
        )
        projection = _project_emitted_occurrences(second, proof=second_proof)

        assert second_proof["emissions"][0]["output_occurrence"]["index"] == 1
        assert second_proof["emissions"][0]["same_prefix_ordinal"] == 1
        assert projection.entries[0].generated_span is None
        assert projection.entries[0].effective_identity == projection.entries[0].full_identity
        assert projection.entries[1].effective_identity[1] == "\n\nnew host turn"
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "tool_fields",
    [
        {"tool_calls": [{"id": "call-1"}]},
        {"tool_call_id": "call-1"},
    ],
)
def test_descriptor_ordinal_excludes_tool_bearing_rows(
    tmp_path, monkeypatch, tool_fields
):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        assembled = engine._assemble_context(None, tail)
        generated = assembled[0]
        span = engine._pending_emission_candidates[0]["span"]
        first = [
            {"role": "user", "content": span + "tool", **tool_fields},
            generated,
            {"role": "user", "content": span + "authored"},
        ]
        first_proof = _record(engine, [compacted, *tail], first)
        first_projection = _project_emitted_occurrences(first, proof=first_proof)

        assert first_proof["emissions"][0]["output_occurrence"]["index"] == 1
        assert first_proof["emissions"][0]["same_prefix_ordinal"] == 0
        assert first_projection.entries[0].generated_span is None
        assert first_projection.entries[1].generated_span == span
        assert first_projection.entries[2].generated_span is None

        second, second_proof = _carry_forward_through_real_ingest(
            engine, monkeypatch, first, [dict(message) for message in first]
        )
        second_projection = _project_emitted_occurrences(second, proof=second_proof)

        assert second_proof["emissions"][0]["output_occurrence"]["index"] == 1
        assert second_projection.entries[1].generated_span == span
        assert second_projection.entries[2].generated_span is None
    finally:
        engine.shutdown()


@pytest.mark.parametrize("duplicate", [False, True])
def test_finalizer_copy_fallback_requires_unique_identity(tmp_path, duplicate):
    engine, compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        generated = engine._assemble_context(None, [])
        copied = dict(generated[0])
        returned = [copied, dict(copied)] if duplicate else [copied]
        proof = _record(engine, [compacted], returned)

        if duplicate:
            assert proof["emissions"] == []
        else:
            assert proof["emissions"][0]["output_occurrence"]["index"] == 0
    finally:
        engine.shutdown()


def test_finalizer_prefers_constructed_row_after_in_place_rewrite(tmp_path):
    engine, compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        returned = engine._assemble_context(None, [])
        span = returned[0]["content"]
        returned[0]["content"] = span + "\n\nhost suffix"
        proof = _record(engine, [compacted], returned)

        assert proof["emissions"][0]["output_occurrence"]["index"] == 0
        assert _project_emitted_occurrences(returned, proof=proof).entries[0].generated_span == span
    finally:
        engine.shutdown()


def test_carry_forward_merged_same_role_collision_fails_closed(tmp_path, monkeypatch):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        generated = engine._assemble_context(None, tail)
        span = engine._pending_emission_candidates[0]["span"]
        first = [{"role": "user", "content": span}, *generated]
        first_proof = _record(engine, [compacted, *tail], first)
        assert first_proof["emissions"][0]["output_occurrence"]["index"] == 1

        merged = _real_hermes_merge(first)
        second, second_proof = _carry_forward_through_real_ingest(
            engine, monkeypatch, first, merged
        )

        assert len(second) < len(first)
        assert second_proof["emissions"] == []
        assert all(
            entry.generated_span is None
            for entry in _project_emitted_occurrences(second, proof=second_proof).entries
        )
    finally:
        engine.shutdown()


@pytest.mark.parametrize("authored_first", [True, False])
def test_finalizer_binds_identical_bytes_to_generated_role(tmp_path, authored_first):
    engine, compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        generated = engine._assemble_context(None, [])
        assert len(generated) == 1 and generated[0]["role"] == "user"
        content = generated[0]["content"]
        authored = {"role": "system", "content": content}
        if authored_first:
            returned = engine._assemble_context(authored, [], include_lcm_note=False)
        else:
            returned = [*generated, authored]
        proof = _record(engine, [compacted], returned)
        generated_index = 1 if authored_first else 0

        assert len(proof["emissions"]) == 1
        assert proof["emissions"][0]["output_occurrence"]["index"] == generated_index
        assert returned[generated_index]["role"] == "user"
    finally:
        engine.shutdown()


def test_rotation_transfers_emissions_and_reset_clears_them(tmp_path, monkeypatch):
    engine, pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=7)
    try:
        assert engine._compress_commit_proof["emissions"]
        assert engine._last_emission_descriptors["emissions"]
        engine.on_session_end("S0", pre)
        engine.on_session_start(
            "S1", boundary_reason="compression", old_session_id="S0", platform="acp"
        )
        proof = engine._compress_commit_proof
        assert proof["emissions"]
        assert {item["scope"]["session_id"] for item in proof["emissions"]} == {"S1"}
        assert {
            item["scope"]["session_id"]
            for item in engine._last_emission_descriptors["emissions"]
        } == {"S1"}
        assert engine._durable_commit_proof_payload()["emissions"]

        engine.on_session_reset()
        assert engine._compress_commit_proof is None
        assert engine._last_emission_descriptors is None
        assert engine._durable_commit_proof_payload() is None
    finally:
        engine.shutdown()


def test_projection_rejects_same_prefix_authored_replacement():
    span = "GENERATED\n\n"
    proof = _scoped_proof(
        "carrier",
        span,
        "old retained",
        retained_source={"store_id": 7},
    )

    projection = _project_emitted_occurrences(
        [{"role": "user", "content": span + "authored"}],
        proof=proof,
    )

    assert projection.entries[0].generated_span is None
    assert projection.entries[0].retained_source is None
    assert projection.entries[0].effective_identity == projection.entries[0].full_identity


def test_projection_accepts_hermes_append_after_recorded_suffix():
    span = "GENERATED\n\n"
    proof = _scoped_proof(
        "carrier",
        span,
        "retained",
        retained_source={"store_id": 7},
    )

    projection = _project_emitted_occurrences(
        [{"role": "user", "content": span + "retained\n\nnew turn"}],
        proof=proof,
    )

    entry = projection.entries[0]
    assert entry.generated_span == span
    assert entry.effective_identity[1] == "retained\n\nnew turn"
    assert entry.retained_source == {"store_id": 7}


def test_projection_normalizes_emission_suffix_edge_whitespace():
    span = "GENERATED\n\n"
    proof = _scoped_proof("carrier", span, "retained\n")

    projection = _project_emitted_occurrences(
        [{"role": "user", "content": span + "retained"}],
        proof=proof,
    )

    assert projection.entries[0].generated_span == span
    assert projection.entries[0].effective_identity[1] == "retained"


def test_empty_suffix_summary_requires_unambiguous_occurrence(tmp_path):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        engine._assemble_context(
            {"role": "system", "content": "system"}, tail, include_lcm_note=False
        )
        candidate = next(
            item for item in engine._pending_emission_candidates if item["kind"] == "summary"
        )
        span = candidate["span"]
        first = [candidate["row"], {"role": "user", "content": span}]
        proof = _record(engine, [compacted, *tail], first)

        appended = _project_emitted_occurrences(
            [{"role": "user", "content": span + "\n\nnew turn"}], proof=proof
        )
        replacement = _project_emitted_occurrences(
            [{"role": "user", "content": span}], proof=proof
        )

        assert proof["emissions"][0]["suffix_length"] == 0
        assert appended.entries[0].generated_span == span
        assert appended.entries[0].effective_identity[1] == "\n\nnew turn"
        assert replacement.entries[0].generated_span is None
        assert replacement.entries[0].effective_identity == replacement.entries[0].full_identity
    finally:
        engine.shutdown()


def test_conversation_rebind_drops_cached_emission_descriptors(tmp_path):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        first = engine._assemble_context(None, tail)
        old_proof = _record(engine, [compacted, *tail], first)
        old_content = first[old_proof["emissions"][0]["output_occurrence"]["index"]]["content"]

        engine.on_session_start(
            "S0",
            platform="acp",
            conversation_id="new-conversation",
            context_length=200_000,
        )
        assert engine._last_emission_descriptors is None

        engine._pending_emission_candidates = []
        generated_looking = {"role": "user", "content": old_content}
        new_proof = _record(
            engine,
            [{"role": "system", "content": "new conversation"}, generated_looking],
            [generated_looking],
        )
        assert new_proof["emissions"] == []
    finally:
        engine.shutdown()


def test_placeholder_keepalive_tracks_only_current_snapshot(tmp_path):
    engine, _compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        class IgnoreRowPattern:
            pattern = "^ignored row"

            @staticmethod
            def search(text, timeout=None):
                return text.startswith("ignored row")

        engine._compiled_ignore_message_patterns = [IgnoreRowPattern()]
        messages = []
        for turn in range(200):
            messages.append({"role": "user", "content": f"ignored row {turn}"})
            engine._ingest_messages(messages)

        current = engine._last_active_replay_messages
        current_placeholder_count = sum(
            id(message) in engine._generated_ignored_active_replay_placeholder_message_ids
            for message in current
        )
        assert current_placeholder_count == len(messages)
        assert len(engine._generated_ignored_active_replay_placeholder_messages) <= (
            2 * current_placeholder_count
        )
    finally:
        engine.shutdown()


def test_cache_hit_placeholder_retention_stays_bounded_to_live_replay(tmp_path):
    engine, _compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        generated = {"role": "user", "content": "ignored generated placeholder"}
        engine._generated_ignored_active_replay_placeholder_message_ids.add(id(generated))
        engine._remember_active_replay_messages([generated], [generated])
        engine._ingest_cursor = 1

        for _ in range(200):
            returned = engine._ingest_messages([generated])

        assert len(engine._generated_ignored_active_replay_placeholder_messages) == 2
        assert engine._generated_ignored_active_replay_placeholder_message_ids == {
            id(generated),
            id(returned[0]),
        }
    finally:
        engine.shutdown()


def test_remember_keeps_live_original_generated_provenance_until_replaced(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "provenance.db")))
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = engine._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        literal = {"role": "user", "content": placeholder}
        generated = {"role": "user", "content": placeholder}
        literal_store_id = engine._store.append("S0", literal)
        assert literal_store_id == 1
        engine._remember_generated_ignored_placeholder_hash(digest)
        engine._generated_ignored_active_replay_placeholder_message_ids.add(id(generated))
        active = [literal, generated]

        before = engine._get_store_id_map_for_messages(active)
        assert before.get(id(literal)) == literal_store_id
        assert id(generated) not in before

        returned = engine._remember_active_replay_messages(active, active)
        after = engine._get_store_id_map_for_messages(returned)
        assert after.get(id(literal)) == literal_store_id
        assert id(generated) not in after
        old_generated_ids = set(
            engine._generated_ignored_active_replay_placeholder_message_ids
        )

        replacement = {"role": "user", "content": placeholder}
        engine._generated_ignored_active_replay_placeholder_message_ids.add(id(replacement))
        engine._remember_active_replay_messages([replacement], [replacement])
        assert old_generated_ids.isdisjoint(
            engine._generated_ignored_active_replay_placeholder_message_ids
        )
        assert len(engine._generated_ignored_active_replay_placeholder_messages) == 2
    finally:
        engine.shutdown()


def test_append_after_remember_keeps_live_original_generated_provenance(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "append.db")))
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        placeholder = engine._ignored_active_replay_placeholder("api_key=sk-ignore...cdef")
        digest = engine._active_replay_placeholder_digest(placeholder)
        assert digest is not None
        literal = {"role": "user", "content": placeholder}
        generated = {"role": "user", "content": placeholder}
        assert engine._store.append("S0", literal) == 1
        engine._remember_generated_ignored_placeholder_hash(digest)
        engine._generated_ignored_active_replay_placeholder_message_ids.add(id(generated))
        active = [literal, generated]
        engine._remember_active_replay_messages(active, active)
        engine._ingest_cursor = len(active)

        active.append({"role": "assistant", "content": "appended assistant reply"})
        engine.ingest(active)

        mapping = engine._get_store_id_map_for_messages(active)
        assert [mapping.get(id(message)) for message in active] == [1, None, 2]
    finally:
        engine.shutdown()


def test_projection_retained_source_is_immutable_and_detached(tmp_path):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        returned = engine._assemble_context(None, tail)
        proof = _record(engine, [compacted, *tail], returned)
        projection = _project_emitted_occurrences(returned, proof=proof)
        entry = next(item for item in projection.entries if item.retained_source is not None)
        descriptor = next(item for item in proof["emissions"] if item["retained_source"] is not None)
        original_store_id = descriptor["retained_source"]["store_id"]

        with pytest.raises(TypeError):
            entry.retained_source["store_id"] = original_store_id + 1

        assert descriptor["retained_source"]["store_id"] == original_store_id
    finally:
        engine.shutdown()
