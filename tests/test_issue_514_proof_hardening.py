"""#514: prior-proof consumption never raises on malformed nested descriptor
shapes, a bad prior proof costs only its carry-forward, and the writer, the
durable reader and the rebind check share one emission binding."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time

from hermes_lcm.reconcile import (
    _COMPACTION_COMMIT_PROOF_METADATA_PREFIX,
    _COMPACTION_COMMIT_PROOF_VERSION,
    _commit_proof_identity_digest,
)
from tests.test_compression_boundary import _summary_carrier_fixture
from tests.test_issue_488_emission_descriptors import _record


def _descriptor(scope, output_occurrence):
    span = b"summary\n\n"
    return {
        "kind": "carrier",
        "role": "user",
        "same_prefix_ordinal": 0,
        "output_occurrence": output_occurrence,
        "generated_span_sha256": hashlib.sha256(span).hexdigest(),
        "generated_span_bytes": len(span),
        "suffix_sha256": hashlib.sha256(b"authored").hexdigest(),
        "suffix_length": len(b"authored"),
        "retained_source": None,
        "scope": scope,
    }


def test_durable_payload_drops_malformed_descriptor_shapes(tmp_path):
    engine, _compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        scope = {
            "hermes_home": str(engine._hermes_home or ""),
            "session_id": engine._session_id,
            "conversation_id": engine._conversation_id,
            "reset_epoch": state.last_reset_at if state is not None else None,
        }
        well_formed = _descriptor(scope, {"index": 0, "same_identity_ordinal": 0, "of": 2})
        emissions = [
            well_formed,
            _descriptor(scope, [0]),
            _descriptor(scope, "x"),
            _descriptor(scope, {"index": "1", "same_identity_ordinal": 0}),
            _descriptor(scope, {"index": 1, "same_identity_ordinal": 0, "of": 1}),
            _descriptor("bad", {"index": 1, "same_identity_ordinal": 0}),
        ]
        payload = {
            "version": _COMPACTION_COMMIT_PROOF_VERSION,
            **scope,
            "created_at": time.time(),
            "effective_sha256": [],
            "carry_ranges": [],
            "emissions": emissions,
        }
        engine._store.write_metadata_json(
            [engine._replay_snapshot_metadata_key(_COMPACTION_COMMIT_PROOF_METADATA_PREFIX)],
            json.dumps(payload, sort_keys=True),
        )

        loaded = engine._durable_commit_proof_payload()
        assert loaded is not None
        assert loaded["emissions"] == [well_formed]
        assert {key: loaded[key] for key in scope} == scope  # nothing else in the payload changes
    finally:
        engine.shutdown()


def test_bad_prior_proof_costs_only_the_carry_forward(tmp_path, monkeypatch):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        first = engine._assemble_context(None, tail)
        _record(engine, [compacted, *tail], first)
        assert engine._last_emission_descriptors["emissions"]

        def explode(*_args, **_kwargs):
            raise RuntimeError("malformed prior proof")

        monkeypatch.setattr("hermes_lcm.compaction._project_emitted_occurrences", explode)
        engine._pending_emission_candidates = []
        second = [first[0], *first[2:]]
        engine._ingest_cursor = len(second)
        engine._ingest_cursor_needs_reconcile = False
        engine._last_compression_status = "compacted"
        engine._record_compress_commit_proof(first, second)

        proof = engine._compress_commit_proof
        assert proof is not None
        assert proof["emissions"] == []  # fresh emissions only: nothing pending, nothing carried
        assert proof["published"] is True
        durable = engine._durable_commit_proof_payload()
        assert durable is not None and durable["emissions"] == []
        assert durable["effective_sha256"] == [
            _commit_proof_identity_digest(identity) for identity in proof["output_effective"]
        ]
    finally:
        engine.shutdown()


def test_rebind_check_uses_the_writer_binding_without_a_conversation_id(tmp_path, monkeypatch):
    # Not reachable on a supported path: every bound start leaves a non-empty
    # conversation id (bind_session falls back to the session id). Pinned as a
    # unit test of the shared binding across the writer, reader and rebind check.
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        real_state = engine._lifecycle.get_by_session(engine._session_id)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_conversation_id", "")
            patch.setattr(
                engine._lifecycle,
                "get_by_session",
                lambda _sid: dataclasses.replace(real_state, last_reset_at=123.0),
            )
            first = engine._assemble_context(None, tail)
            proof = _record(engine, [compacted, *tail], first)
            assert proof["reset_epoch"] == 123.0
            # A start that leaves the binding in place (an early-return path).
            patch.setattr(engine, "_on_session_start_unlocked", lambda *_a, **_k: None)
            engine.on_session_start("S0", platform="acp")

            assert engine._compress_commit_proof is proof
            assert engine._last_emission_descriptors is not None
            binding = engine._emission_binding()
            assert binding == {key: proof[key] for key in binding}
            assert engine._emission_proof_matches_binding(engine._last_emission_descriptors, binding)
    finally:
        engine.shutdown()
