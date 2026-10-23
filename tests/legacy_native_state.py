"""Historical v0.24.x writer fixture for unchanged persisted-state reader tests.

Never installed by the plugin. These routines prepare old proofs/digests so
reader coverage does not depend on a runtime writer that was removed in O2.
"""
import copy
import logging
from typing import Any, Dict, List
from hermes_lcm.message_analysis import _matched_tool_call_ids
from hermes_lcm.message_content import text_content_for_pattern_matching
from hermes_lcm.tokens import count_messages_tokens
from hermes_lcm.reconcile import (
    _COMPACTION_COMMIT_PROOF_VERSION, _commit_proof_identity_digest,
    _finalize_emission_descriptors, _has_lossy_redacted_identity,
    _project_emitted_occurrences, _emission_identity, _proof_user_identity,
)

logger = logging.getLogger(__name__)

def legacy_adoption(
    self,
    messages: List[Dict[str, Any]],
    *,
    current_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    """Let the host archive a native summary without claiming LCM coverage.

    Only an opted-in, cancellation-fenced Hermes invocation may use this
    recovery. Its existing transaction owns archive/publish; this helper
    writes no DAG, lifecycle, frontier, or host-session state. In particular
    it must not use the bypass helper's deterministic tail-trimming path.
    """
    if not self._config.native_recovery:
        return messages
    self._last_compress_aborted = True
    self._last_compression_status = "error"
    self._last_compression_noop_reason = "native recovery did not produce a usable summary"
    self._last_summary_error = self._last_compression_noop_reason
    self._last_native_recovery_rejection = ""

    def _reject(reason: str, **details: Any) -> List[Dict[str, Any]]:
        self._last_native_recovery_rejection = reason
        self._last_compression_noop_reason = reason
        logger.warning(
            "Native recovery rejected; retaining context "
            "(reason=%s, input_rows=%d, recovered_rows=%d, protected_rows=%d, details=%s)",
            reason, len(messages), int(details.pop("recovered_rows", 0)),
            int(details.pop("protected_rows", 0)), details or None,
        )
        return messages

    cancelled = getattr(self, "_compression_cancelled_check", None)
    if not callable(cancelled):
        return _reject("binding_changed")
    if cancelled():
        return _reject("cancelled")
    try:
        import agent.context_compressor as context_compressor

        ContextCompressor = context_compressor.ContextCompressor

        fresh_tail = self._fresh_tail_boundary(messages)
        protected_tail = messages[fresh_tail.start:]
        prefix = messages[:fresh_tail.start]
        tail_key = getattr(context_compressor, "_COMPACTION_TAIL_MARKER", None)
        persisted_key = getattr(context_compressor, "_DB_PERSISTED_MARKER", None)
        split = bool(tail_key and persisted_key and protected_tail)
        if split and len(prefix) <= self.protect_first_n + 4:
            return _reject("prefix_too_short", protected_rows=len(protected_tail), prefix_rows=len(prefix))

        # Fresh per attempt: a timed-out predecessor must never share its
        # native compressor's mutable summary/cooldown state with a retry.
        native = ContextCompressor(
            model=self.model,
            provider=self.provider,
            api_mode=self.api_mode,
            base_url=self.base_url,
            api_key=self.api_key,
            config_context_length=self.context_length,
            threshold_percent=self.threshold_percent,
            protect_first_n=self.protect_first_n,
            protect_last_n=3 if split else fresh_tail.count,
            summary_model_override=self._config.summary_model or None,
            abort_on_summary_failure=True,
            quiet_mode=True,
        )
        if split or fresh_tail.tokens > 0:
            native.tail_token_budget = 1 if split else fresh_tail.tokens
        native.on_session_start(self._session_id)
        native._compression_cancelled_check = cancelled
        derive_focus = getattr(ContextCompressor, "_derive_auto_focus_topic", None)
        focus_topic = derive_focus(messages) if callable(derive_focus) else None
        synthetic_user = getattr(
            ContextCompressor, "_is_synthetic_compression_user_turn", None
        )
        suffix_has_real_user = False
        for message in protected_tail:
            is_user = message.get("role") == "user"
            has_text = bool(
                (text_content_for_pattern_matching(message.get("content")) or "").strip()
            )
            is_synthetic = callable(synthetic_user) and synthetic_user(message)
            if is_user and has_text and not is_synthetic:
                suffix_has_real_user = True
                break
        if split and suffix_has_real_user and hasattr(native, "_find_inflight_user_task"):
            native._find_inflight_user_task = lambda _messages: None
        native_kwargs = {
            "current_tokens": (
                current_tokens
                if current_tokens is not None and current_tokens > 0
                else count_messages_tokens(messages)
            ),
            "force": True,
        }
        if split:
            native_kwargs["focus_topic"] = focus_topic
        recovered_head = native.compress(
            copy.deepcopy(prefix if split else messages),
            **native_kwargs,
        )
        if split:
            def _carry(message):
                carried = copy.deepcopy(message)
                carried.pop(persisted_key, None)
                carried[tail_key] = True
                return carried
            recovered = recovered_head + [_carry(message) for message in protected_tail]
        else:
            recovered = recovered_head

        def _native_source_rows(rows):
            return [
                projected for row in rows
                if (projected := ContextCompressor._strip_context_summary_handoff_message(row)) is not None
            ]

        recovered_source_rows = _native_source_rows(recovered)
        protected_source_rows = _native_source_rows(protected_tail)
        recovered_tail = (
            recovered_source_rows[-len(protected_source_rows):]
            if protected_source_rows
            else []
        )
        recovered_tokens = count_messages_tokens(recovered)
        input_tokens = count_messages_tokens(messages)
        recovered_identities = [
            self._message_replay_identity(message) for message in recovered_tail
        ]
        protected_identities = [
            self._message_replay_identity(message) for message in protected_source_rows
        ]
        reason = next((name for name, failed in (
            ("cancelled", cancelled()),
            ("binding_changed", getattr(self, "_compression_cancelled_check", None) is not cancelled),
            ("native_aborted", native._last_compress_aborted),
            ("no_progress", not getattr(native, "_last_compression_made_progress", False)),
            ("digest_unavailable", any("[digest unavailable for segment " in str(m.get("content", "")) for m in recovered)),
            ("fallback_used", getattr(native, "_last_summary_fallback_used", False)),
            ("empty", not recovered_head),
            ("not_smaller", recovered_tokens >= input_tokens),
            ("suffix_changed", recovered_identities != protected_identities),
        ) if failed), "")
        if reason:
            details = {"recovered_rows": len(recovered), "protected_rows": len(protected_source_rows)}
            if reason == "native_aborted":
                details["failure_class"] = (getattr(native, "_last_compression_telemetry", {}) or {}).get("failure_class")
            elif reason == "not_smaller":
                details.update(input_tokens=input_tokens, recovered_tokens=recovered_tokens)
            elif reason == "suffix_changed":
                changed_offsets = [index for index, pair in enumerate(zip(recovered_identities, protected_identities)) if pair[0] != pair[1]]
                details["changed_rows"] = len(changed_offsets) + abs(len(recovered_identities) - len(protected_identities))
                details["first_changed_offset"] = changed_offsets[0] if changed_offsets else min(len(recovered_identities), len(protected_identities))
            return _reject(reason, **details)
    except Exception as exc:
        self._last_native_recovery_rejection = "exception"
        self._last_compression_noop_reason = "exception"
        logger.warning(
            "Native recovery failed; retaining context (exception_type=%s)",
            type(exc).__name__,
        )
        return messages
    self._last_compression_status = "host_native"
    self._last_summary_error = None
    self._last_compression_noop_reason = ""
    self.compression_count += 1
    self._last_compress_aborted = False
    # The helper returns before the host's archive transaction commits.
    self._remember_native_recovery_replay_snapshot_digest(
        self._native_recovery_replay_snapshot_digest(recovered)
    )
    self._ingest_cursor = len(recovered)
    self._ingest_cursor_needs_reconcile = False
    self._last_native_summary_index = next((i for i, row in enumerate(recovered) if ContextCompressor._strip_context_summary_handoff_message(row) != row), None)
    logger.info(
        "Native recovery returned a summary for the host archive transaction "
        "(input_rows=%d, recovered_rows=%d, cursor=%d); LCM source history and frontier are unchanged.",
        len(messages), len(recovered), self._ingest_cursor,
    )
    return recovered

def legacy_commit_proof(self, messages, result) -> None:
    """Remember the exact host input of this compress() call (process-local).

    Hermes commits a compaction by calling ``on_session_end(sid, <this input>)``
    before it adopts ``result``. Every row of that input was ingested here,
    and ``_ingest_cursor`` now indexes ``result``, so the session-end hook skips
    the re-ingest; it still finalizes the session with its own frontier (#483).
    """
    try:
        self._compress_commit_proof = None
        if (
            not self._session_id
            or not isinstance(result, list)
            or self._bypasses_lcm_context_management()
            or self._ingest_cursor_needs_reconcile
            or self._ingest_cursor != len(result)
        ):
            return
        emission_binding = self._emission_binding()
        prior_proof = getattr(self, "_last_emission_descriptors", None)
        if not self._emission_proof_matches_binding(prior_proof, emission_binding):
            prior_proof = None
        if not prior_proof:
            try:
                durable_proof = self._durable_commit_proof_payload()
            except Exception:  # malformed durable history never blocks a fresh proof
                durable_proof = None
            prior_proof = durable_proof if durable_proof and durable_proof.get("emissions") else None
        emissions = _finalize_emission_descriptors(
            result, getattr(self, "_pending_emission_candidates", ()), emission_binding
        )
        if prior_proof:
            fresh_count = len(emissions)
            try:
                prior_projection = _project_emitted_occurrences(messages, proof=prior_proof)
                result_identities = [_emission_identity(message) for message in result]
                for entry in prior_projection.entries:
                    if entry.generated_span is None or result_identities.count(entry.full_identity) != 1:
                        continue
                    carried = _finalize_emission_descriptors(result, [{
                        "kind": entry.kind,
                        "span": entry.generated_span,
                        "retained_source": dict(entry.retained_source) if entry.retained_source is not None else None,
                        "full_identity": entry.full_identity,
                    }], emission_binding)
                    if carried and all(
                        item["output_occurrence"]["index"] != carried[0]["output_occurrence"]["index"]
                        for item in emissions
                    ):
                        emissions.extend(carried)
            except Exception as exc:  # a bad prior proof costs its carry-forward, never the fresh proof (#514)
                logger.warning("LCM prior-proof carry-forward skipped: %r", exc)
                del emissions[fresh_count:]
        emissions.sort(key=lambda item: item["output_occurrence"]["index"])
        # The output's rows map through THIS proof's emissions (A3), never a DAG-shaped strip (F1).
        occurrences, _v4 = self._replay_occurrences(result, {"version": 4, **emission_binding, "emissions": emissions})
        store_ids = self._get_store_id_map_for_messages(result, occurrences)
        carried_rows = self._store.get_batch(sorted({store_ids[id(m)] for m in result if id(m) in store_ids}))
        proof = {
            "version": _COMPACTION_COMMIT_PROOF_VERSION,
            **emission_binding,
            "session_id": self._session_id,
            "conversation_id": self._conversation_id,
            # Full identities: the multiplicity witness (F4); output_effective is the projection.
            "input": [self._proof_replay_identity(m, strip_carrier=False) for m in messages],
            "output": [self._proof_replay_identity(m, strip_carrier=False) for m in result],
            "end_consumed": False,
            "carry_ranges": self._coalesce_compression_carry_ranges(
                (str(row["session_id"]), store_id - 1, store_id)
                for store_id, row in sorted(carried_rows.items())
                if row.get("session_id")
            ),
            "emissions": emissions,
        }
        for item in emissions:  # multiplicity witness for carried-forward proofs, which omit "output"
            item["output_multiplicity"] = proof["output"].count(proof["output"][item["output_occurrence"]["index"]])
        if proof["output"] == proof["input"]:
            # No-progress compress: Hermes has nothing to commit, and an
            # end call with this list must stay a real session end.
            return
        _projection, output_identities = self._occurrence_replay_identities(
            result, {**proof, **emission_binding}
        )
        proof["output_effective"] = [
            _proof_user_identity(identity) for identity in output_identities if identity is not None
        ]
        # v0.24.0's own digests (b9ad016e: stripped identities, scaffold-filtered), which its
        # reader recomputes after a rollback (#517); the durable record carries both sets.
        proof["output_sha256_v3"] = [_commit_proof_identity_digest(self._proof_replay_identity(m)) for m in result]
        proof["effective_sha256_v3"] = [
            digest for m, digest in zip(result, proof["output_sha256_v3"])
            if not self._is_replayed_context_scaffold_message(m)
        ]
        proof["native"] = self._last_compression_status == "host_native"
        if proof["native"]:
            matched_tool_ids = _matched_tool_call_ids(result)
            proof["droppable"] = []
            proof["skip_landing"] = []
            for identity in proof["output_effective"]:
                is_tool_result = identity[0] == "tool"
                has_tool_id = bool(identity[2])
                is_orphan = identity[2] not in matched_tool_ids
                is_lossless = not _has_lossy_redacted_identity(identity)
                proof["droppable"].append(
                    is_tool_result and has_tool_id and is_orphan and is_lossless
                )
                proof["skip_landing"].append(not identity[2] and not identity[3])
            summary_index = getattr(self, "_last_native_summary_index", None)
            proof["native_summary_index"] = sum(
                identity is not None for identity in output_identities[:summary_index]
            ) if summary_index is not None else None
        proof["published"] = self._last_compression_status == "compacted"
        self._last_emission_descriptors = {
            "version": _COMPACTION_COMMIT_PROOF_VERSION,
            **emission_binding,
            "emissions": copy.deepcopy(emissions),
        }
        self._compress_commit_proof = proof
        if proof["published"] or proof["native"]:
            self._persist_compress_commit_proof(proof)
    except Exception:
        self._compress_commit_proof = None

def install_legacy_writer(monkeypatch, engine_class):
    """Fixture setup only: emit a historical snapshot and proof for reader tests."""
    def old_impl(self, messages, current_tokens=None, **kwargs):
        working = self._ingest_messages(messages)
        self._prepare_retained_user_anchor(working)
        return legacy_adoption(self, working, current_tokens=current_tokens)

    original_compress = engine_class.compress

    def old_compress(self, *args, **kwargs):
        result = original_compress(self, *args, **kwargs)
        if self._last_compression_status == "host_native" and self._compress_commit_proof is None:
            self._ingest_cursor = 0
            self._ingest_cursor_needs_reconcile = True
        return result

    monkeypatch.setattr(engine_class, "compress", old_compress)
    monkeypatch.setattr(engine_class, "_compress_impl", old_impl)
    monkeypatch.setattr(engine_class, "_record_compress_commit_proof", legacy_commit_proof)
