from __future__ import annotations

import hashlib

import pytest

from hermes_lcm.reconcile import _project_emitted_occurrences
from tests.test_compression_boundary import _summary_carrier_fixture
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


def _assert_exact_descriptors(returned, proof):
    descriptors = proof["emissions"]
    assert descriptors
    assert len(descriptors) == len({d["output_occurrence"]["index"] for d in descriptors})
    for descriptor in descriptors:
        content = returned[descriptor["output_occurrence"]["index"]]["content"]
        raw = content.encode("utf-8")
        span = raw[: descriptor["generated_span_bytes"]]
        assert hashlib.sha256(span).hexdigest() == descriptor["generated_span_sha256"]


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


def test_rotation_transfers_emissions_and_reset_clears_them(tmp_path, monkeypatch):
    engine, pre, compressed = _phase1_compacted_engine(tmp_path, monkeypatch, tail=7)
    try:
        assert engine._compress_commit_proof["emissions"]
        engine.on_session_end("S0", pre)
        engine.on_session_start(
            "S1", boundary_reason="compression", old_session_id="S0", platform="acp"
        )
        proof = engine._compress_commit_proof
        assert proof["emissions"]
        assert {item["scope"]["session_id"] for item in proof["emissions"]} == {"S1"}
        assert engine._durable_commit_proof_payload()["emissions"]

        engine.on_session_reset()
        assert engine._compress_commit_proof is None
        assert engine._durable_commit_proof_payload() is None
    finally:
        engine.shutdown()
