"""Leaf-compaction pipeline for the LCM engine (WS5 Seam 6).

The ``CompactionMixin`` holds the compaction gate + pipeline: ``should_compress``
/ ``should_compress_preflight`` (public), the leaf-candidate and chunk-selection
helpers, and the main ``compress`` entry point. These methods were lifted
verbatim out of ``LCMEngine`` and continue to run bound to the engine instance
(``self`` is the ``LCMEngine``), so they read and write the engine's runtime
state (``_ingest_cursor``, ``_store``, ``_dag``, ``_lifecycle``, status/telemetry
fields, per-turn caches) and call back into engine helpers (ingest,
reconciliation, placeholder-ledger, the summarize-with-rescue step, assembly,
lifecycle) through normal attribute lookup. ``LCMEngine`` mixes this in ahead of
``ContextEngine`` so the mixin's ``compress`` / ``should_compress`` /
``should_compress_preflight`` override the ContextEngine protocol defaults.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from .dag import SummaryNode
from .lifecycle_state import LifecycleBindingChangedError, LifecyclePublicationConflictError
from .message_analysis import _matched_tool_call_ids
from .message_content import text_content_for_pattern_matching
from .reconcile import (
    _COMPACTION_COMMIT_PROOF_METADATA_PREFIX,
    _COMPACTION_COMMIT_PROOF_VERSION,
    _COMPACTION_COMMIT_PROOF_WIRE_VERSION,
    _commit_proof_identity_digest,
    _emission_identity,
    _finalize_emission_descriptors,
    _has_lossy_redacted_identity,
    _project_emitted_occurrences,
    _proof_user_identity,
)
from .sanitize import _contains_sensitive_redaction
from .sqlite_util import _is_sqlite_locked_error
from .tokens import count_message_tokens, count_messages_tokens, count_tokens

_UNPROVEN_FILTER_EXCLUSION = object()

logger = logging.getLogger(__name__)

_THRESHOLD_FULL_SWEEP_MAX_PASSES = 12
_THRESHOLD_FULL_SWEEP_MAX_SECONDS = 120.0


class CompactionMixin:
    def _maybe_reclassify_late_auxiliary_before_compaction_write(self) -> None:
        maybe_reclassify = getattr(
            self,
            "_maybe_reclassify_current_session_as_auxiliary_before_message_ingest",
            None,
        )
        if callable(maybe_reclassify):
            maybe_reclassify()

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._bypasses_lcm_context_management():
            if self._compression_boundary_cooldown_active():
                return False
            if prompt_tokens is not None:
                tokens = prompt_tokens
            else:
                auxiliary_session_id = self._thread_context_session_id()
                if auxiliary_session_id:
                    tokens = self._current_auxiliary_prompt_tokens(auxiliary_session_id)
                else:
                    tokens = self.last_prompt_tokens
            if self._should_force_overflow_recovery(observed_tokens=tokens):
                return True
            if self.threshold_tokens <= 0:
                return False
            return tokens >= self.threshold_tokens
        if self._compression_boundary_cooldown_active():
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages):
        """Pre-flight check — also ingests messages into the store."""
        with self._fresh_tail_pressure_yield_invocation():
            return self._should_compress_preflight_impl(messages)

    def _should_compress_preflight_impl(self, messages):
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        self._native_recovery_preflight_cleanup_only = False
        self._maybe_reclassify_late_auxiliary_before_compaction_write()
        if self._bypasses_lcm_context_management():
            # Bypassed traffic observes nothing about the pressured session's
            # tail: it must neither extend nor reset the blocked streak.
            self._pressure_yield_invocation_verdict = "neutral"
            self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            rough = count_messages_tokens(messages)
            if self._compression_boundary_cooldown_active():
                return False
            if self._should_force_overflow_recovery(observed_tokens=rough, messages=messages):
                return True
            return self.threshold_tokens > 0 and rough >= self.threshold_tokens
        rough = count_messages_tokens(messages)
        if self.threshold_tokens > 0 and rough < self.threshold_tokens:
            self._note_fresh_tail_pressure_relieved()
        pre_ingest_placeholder_ambiguous_noop = False
        pre_ingest_noop_reason = ""
        pre_ingest_placeholder_cleanup_requested = False
        if (
            self.threshold_tokens > 0
            and rough >= self.threshold_tokens
            and not self._compiled_ignore_message_patterns
            and any(
                self._is_ignored_active_replay_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                for msg in messages
            )
        ):
            # Yield-aware (observed_tokens): a persisted-placeholder session
            # whose only blocker is the fresh tail must not park in the
            # ambiguous-noop state while compress() would engage the pressure
            # yield and make progress.
            eligible, reason = self._leaf_compaction_candidate_status(
                messages,
                allow_partial_leaf=self._config.threshold_full_sweep_enabled,
                observed_tokens=rough,
            )
            pre_ingest_placeholder_cleanup_requested = bool(
                not eligible and self._pressure_yield_tail_token_limit > 0
            )
            pre_ingest_placeholder_ambiguous_noop = not eligible
            pre_ingest_noop_reason = reason
        replay_messages = None
        if self._session_id and messages:
            try:
                replay_messages = self._ingest_messages(messages)
                self._prepare_retained_user_anchor(replay_messages)
                self._record_ingest_success()
            except Exception as e:
                # Fail closed for NORMAL threshold compaction: the store did not
                # accept this turn, so do not compact against a store missing the
                # latest messages - that could rebuild active context without
                # them. But still honor emergency overflow recovery, whose whole
                # job is to keep the prompt under the provider limit; it converges
                # via deterministic L3 truncation without needing the store write.
                self._record_ingest_failure("preflight", e)
                if self._should_force_overflow_recovery(observed_tokens=rough):
                    return True
                return False
        if replay_messages is not None and replay_messages != messages:
            replay_rough = count_messages_tokens(replay_messages)
            cleanup_requested = self._replay_diff_requests_ingest_cleanup(
                messages,
                replay_messages,
            )
            force_overflow_requested = self._should_force_overflow_recovery(
                observed_tokens=rough,
                messages=messages,
            ) or self._should_force_overflow_recovery(
                observed_tokens=replay_rough,
                messages=replay_messages,
            )
            if cleanup_requested:
                self._native_recovery_preflight_cleanup_only = bool(
                    self._config.native_recovery
                    and not force_overflow_requested
                    and (
                        self.threshold_tokens <= 0
                        or max(rough, replay_rough) < self.threshold_tokens
                    )
                )
                if (
                    not force_overflow_requested
                    and self._compression_boundary_cooldown_active()
                ):
                    self._preflight_cleanup_only_due_to_boundary_cooldown = True
                return self._mark_preflight_compression_requested()
            if force_overflow_requested:
                return self._mark_preflight_compression_requested()
            # A boundary skip cools down summary-producing leaf/condensation
            # work. It must not prevent the host from adopting a replay cleanup
            # that ingest has already made durable (for example a live tool
            # result stub); those returns above are deterministic and add no
            # summarizer spend.
            if self._compression_boundary_cooldown_active():
                return False
            if (
                self._config.native_recovery
                and self.threshold_tokens > 0
                and replay_rough >= self.threshold_tokens
            ):
                return self._mark_preflight_compression_requested(
                    depends_on_pressure_yield=self._pressure_yield_preflight_candidate,
                )
            if pre_ingest_placeholder_cleanup_requested:
                return self._mark_preflight_compression_requested(
                    depends_on_pressure_yield=True,
                )
            if pre_ingest_placeholder_ambiguous_noop:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = pre_ingest_noop_reason
                logger.info("LCM preflight compression no-op: %s", pre_ingest_noop_reason)
                return False
            eligible, reason = self._leaf_compaction_candidate_status(
                replay_messages,
                allow_partial_leaf=bool(
                    self._config.threshold_full_sweep_enabled
                    and self.threshold_tokens > 0
                    and replay_rough >= self.threshold_tokens
                ),
                observed_tokens=replay_rough,
            )
            if eligible:
                if self.threshold_tokens > 0 and replay_rough >= self.threshold_tokens:
                    return self._mark_preflight_compression_requested(
                        depends_on_pressure_yield=self._pressure_yield_preflight_candidate,
                    )
                self._refresh_raw_backlog_debt(
                    replay_messages,
                    observed_tokens=replay_rough,
                )
                if self._critical_budget_pressure_reached(
                    observed_tokens=replay_rough,
                    messages=replay_messages,
                ):
                    return self._mark_preflight_compression_requested(
                        depends_on_pressure_yield=self._pressure_yield_preflight_candidate,
                    )
                return False
            if self._has_ignored_backlog_outside_fresh_tail(replay_messages):
                return self._mark_preflight_compression_requested()
            if self.threshold_tokens > 0 and replay_rough >= self.threshold_tokens:
                if self._should_run_deferred_maintenance(replay_messages, observed_tokens=replay_rough):
                    return self._mark_preflight_compression_requested()
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = reason
                logger.info("LCM preflight compression no-op: %s", reason)
                return False
            self._refresh_raw_backlog_debt(replay_messages, observed_tokens=replay_rough)
            # A disabled critical-pressure ratio intentionally leaves this debt
            # deferred until threshold, overflow, cleanup, ignored-backlog, or
            # another explicit required trigger makes preflight synchronous.
            if self._critical_budget_pressure_reached(
                observed_tokens=replay_rough,
                messages=replay_messages,
            ) and self._should_run_deferred_maintenance(
                replay_messages,
                observed_tokens=replay_rough,
            ):
                return self._mark_preflight_compression_requested()
            return False
        if self._compression_boundary_cooldown_active():
            return False
        if self._should_force_overflow_recovery(observed_tokens=rough):
            return self._mark_preflight_compression_requested()
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            if self._config.native_recovery:
                return self._mark_preflight_compression_requested(
                    depends_on_pressure_yield=self._pressure_yield_preflight_candidate,
                )
            if pre_ingest_placeholder_cleanup_requested:
                return self._mark_preflight_compression_requested(
                    depends_on_pressure_yield=True,
                )
            if pre_ingest_placeholder_ambiguous_noop:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = pre_ingest_noop_reason
                logger.info("LCM preflight compression no-op: %s", pre_ingest_noop_reason)
                return False
            eligible, reason = self._leaf_compaction_candidate_status(
                messages,
                allow_partial_leaf=self._config.threshold_full_sweep_enabled,
                observed_tokens=rough,
            )
            if eligible:
                return self._mark_preflight_compression_requested(
                    depends_on_pressure_yield=self._pressure_yield_preflight_candidate,
                )
            if self._has_ignored_backlog_outside_fresh_tail(messages):
                return self._mark_preflight_compression_requested()
            if self._should_run_deferred_maintenance(messages, observed_tokens=rough):
                return self._mark_preflight_compression_requested()
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            logger.info("LCM preflight compression no-op: %s", reason)
            return False
        self._refresh_raw_backlog_debt(messages, observed_tokens=rough)
        # With critical pressure disabled, routine debt remains recorded until
        # another required preflight trigger is reached.
        if self._critical_budget_pressure_reached(
            observed_tokens=rough,
            messages=messages,
        ) and self._should_run_deferred_maintenance(
            messages,
            observed_tokens=rough,
        ):
            return self._mark_preflight_compression_requested()
        return False

    def _replay_diff_requests_ingest_cleanup(
        self,
        original_messages: List[Dict[str, Any]],
        replay_messages: List[Dict[str, Any]],
    ) -> bool:
        if len(original_messages) != len(replay_messages):
            return True
        for original_msg, replay_msg in zip(original_messages, replay_messages):
            original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
            replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
            if original_text != replay_text:
                if replay_text.startswith("[Externalized LCM ingest payload:"):
                    return True
                if replay_text.startswith("[Externalized payload: kind=raw_payload;"):
                    return True
                if replay_text.startswith("[Externalized tool output:"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: assistant output quarantined;"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: message ignored;"):
                    return True
                if "[LCM sensitive redaction:" in replay_text:
                    return True
            if original_msg.get("content") != replay_msg.get("content") and _contains_sensitive_redaction(
                replay_msg.get("content")
            ):
                return True
            if original_msg.get("tool_calls") != replay_msg.get("tool_calls") and _contains_sensitive_redaction(
                replay_msg.get("tool_calls")
            ):
                return True
        return False

    def _has_ignored_backlog_outside_fresh_tail(self, messages: List[Dict[str, Any]]) -> bool:
        if not self._compiled_ignore_message_patterns or not messages:
            return False
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False
        previous_store_id_map = self._current_compress_store_ids_by_message_id
        occurrences, v4 = self._replay_occurrences(messages)  # the complete list, sliced (A2, F5)
        self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(
            messages[leading_anchor_count:fresh_tail_start],
            occurrences[leading_anchor_count:fresh_tail_start] if v4 else None,
        )
        try:
            return any(
                self._matches_ignore_message_patterns(msg)
                or self._mapped_stored_row_matches_ignore_message_patterns(msg)
                for msg in messages[leading_anchor_count:fresh_tail_start]
            )
        finally:
            self._current_compress_store_ids_by_message_id = previous_store_id_map

    def _leaf_compaction_candidate_status(
        self,
        messages: List[Dict[str, Any]],
        *,
        force_overflow: bool = False,
        allow_partial_leaf: bool = False,
        observed_tokens: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Return whether a normal leaf compaction pass can actually run.

        The host asks ``should_compress_preflight`` before it emits user-visible
        compression status. A session can be over the global context threshold
        while all pressure sits in the protected fresh tail, or while the raw
        backlog outside that tail is still smaller than the configured leaf
        chunk. In that case ``compress()`` would immediately no-op, so preflight
        should not advertise a compaction attempt yet.

        When ``observed_tokens`` reports host-observed over-threshold pressure
        and the protected tail itself is why no pass can run, the fresh-tail
        pressure yield re-resolves the tail with a derived token bound and the
        check runs once more (see ``_maybe_engage_fresh_tail_pressure_yield``;
        the yield arms only once that pressure is sustained).
        """
        eligible, reason, filtered_eligible_tokens = self._leaf_compaction_candidate_status_once(
            messages,
            force_overflow=force_overflow,
            allow_partial_leaf=allow_partial_leaf,
        )
        if eligible:
            return eligible, reason
        if reason in (
            "no eligible raw backlog outside fresh tail",
            "raw backlog outside fresh tail is below leaf chunk threshold",
        ) and self._maybe_engage_fresh_tail_pressure_yield(
            messages,
            observed_tokens,
            eligible_tokens=filtered_eligible_tokens,
        ):
            eligible, reason, _ = self._leaf_compaction_candidate_status_once(
                messages,
                force_overflow=force_overflow,
                allow_partial_leaf=allow_partial_leaf,
            )
            if eligible:
                self._pressure_yield_preflight_candidate = True
        return eligible, reason

    def _leaf_compaction_candidate_status_once(
        self,
        messages: List[Dict[str, Any]],
        *,
        force_overflow: bool = False,
        allow_partial_leaf: bool = False,
    ) -> tuple[bool, str, int]:
        """Single candidate-status pass.

        Returns ``(eligible, reason, eligible_tokens)`` where
        ``eligible_tokens`` is the token count of the FILTERED raw backlog
        outside the resolved tail — the same view ``compress()`` operates on —
        so the pressure-yield gate judges tail blockage from the numbers the
        compaction pass would actually see.
        """
        if not messages:
            return False, "empty message list", 0
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False, "no eligible raw backlog outside fresh tail", 0

        candidate_raw = messages[leading_anchor_count:fresh_tail_start]
        if not candidate_raw:
            return False, "no eligible raw backlog outside fresh tail", 0
        generated_placeholder_hashes = self._load_generated_ignored_placeholder_hashes()
        if self._compiled_ignore_message_patterns or generated_placeholder_hashes:
            previous_store_id_map = self._current_compress_store_ids_by_message_id
            occurrences, v4 = self._replay_occurrences(messages)  # the complete list, sliced (A2, F5)
            self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(
                candidate_raw, occurrences[leading_anchor_count:fresh_tail_start] if v4 else None
            )
            try:
                filtered_candidate_raw: list[Dict[str, Any]] = []
                for msg in candidate_raw:
                    content_text = text_content_for_pattern_matching(msg.get("content")) or ""
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    generated_volatile_placeholder = (
                        self._is_volatile_ignored_quarantine_placeholder(msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in generated_placeholder_hashes
                    )
                    if (
                        self._matches_ignore_message_patterns(msg)
                        or self._mapped_stored_row_matches_ignore_message_patterns(msg)
                        or self._is_ignored_active_replay_placeholder(msg, content_text)
                        or generated_volatile_placeholder
                    ):
                        continue
                    filtered_candidate_raw.append(msg)
            finally:
                self._current_compress_store_ids_by_message_id = previous_store_id_map
            candidate_raw = filtered_candidate_raw
            if not candidate_raw:
                return False, "no eligible raw backlog outside fresh tail", 0

        if force_overflow:
            return True, "forced overflow recovery", 0

        raw_tokens_outside_tail = count_messages_tokens(candidate_raw)
        if allow_partial_leaf:
            return True, "eligible partial threshold-sweep leaf", raw_tokens_outside_tail
        if self._config.dynamic_leaf_chunk_enabled:
            working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
        else:
            working_leaf_chunk_tokens = self._config.leaf_chunk_tokens
            ctx_cap = self._context_aware_leaf_cap()
            if ctx_cap is not None and working_leaf_chunk_tokens > ctx_cap:
                working_leaf_chunk_tokens = ctx_cap
        if (
            raw_tokens_outside_tail < working_leaf_chunk_tokens
            and self._pressure_yield_tail_token_limit <= 0
        ):
            return (
                False,
                "raw backlog outside fresh tail is below leaf chunk threshold",
                raw_tokens_outside_tail,
            )
        return True, "eligible raw backlog outside fresh tail", raw_tokens_outside_tail

    def _context_aware_leaf_cap(self) -> int | None:
        """Return a context-proportional cap for leaf chunk sizing.

        When context_length is known and large enough to matter (> 50K),
        leaf chunks should never exceed ~40% of the model window —
        otherwise the fresh tail alone can consume the entire context
        and compression becomes impossible.
        Returns None when context_length is unknown or too small to
        warrant clamping (test fixtures, tiny models).
        """
        ctx = getattr(self, "context_length", 0) or 0
        if ctx < 50_000:
            return None
        return max(1, int(ctx * 0.4))

    def _working_leaf_chunk_tokens(self, raw_tokens_outside_tail: int) -> int:
        base = max(1, self._config.leaf_chunk_tokens)
        ctx_cap = self._context_aware_leaf_cap()
        if ctx_cap is not None and base > ctx_cap:
            base = ctx_cap
        if not self._config.dynamic_leaf_chunk_enabled:
            return base
        ceiling = max(base, self._config.dynamic_leaf_chunk_max)
        if ctx_cap is not None and ceiling > ctx_cap:
            ceiling = ctx_cap
        working = base
        while working < ceiling and raw_tokens_outside_tail > working * 2:
            working = min(ceiling, working * 2)
        return working

    def _select_oldest_leaf_chunk(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_leaf_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        selected: list[Dict[str, Any]] = []
        used = 0
        for msg in candidate_raw:
            msg_tokens = count_message_tokens(msg)
            if used + msg_tokens > working_leaf_chunk_tokens and selected:
                break
            selected.append(msg)
            used += msg_tokens
        return selected

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None,
                 focus_topic: Optional[str] = None,
                 force: bool = False) -> List[Dict[str, Any]]:
        """Run compaction and leave a terminal public status on every failure."""
        try:
            self._pending_emission_candidates = []
            self._compress_occurrences = None
            with self._fresh_tail_pressure_yield_invocation():
                result = self._compress_impl(
                    messages,
                    current_tokens=current_tokens,
                    focus_topic=focus_topic,
                    force=force,
                )
            self._compress_occurrences = None
            if (
                isinstance(result, list)
                and result is not messages
                and self._last_compression_status != "error"
                and len(result) == len(messages)
                and all(
                    self._public_compression_row(left)
                    == self._public_compression_row(right)
                    for left, right in zip(result, messages)
                )
            ):
                result = messages
            self._record_compress_commit_proof(messages, result)
            logger.debug("LCM compaction emission descriptor count=%d",
                         len((self._compress_commit_proof or {}).get("emissions") or ()))
            proof_missing = self._last_compression_status == "host_native" and self._compress_commit_proof is None
            if proof_missing:
                self._ingest_cursor = 0
                self._ingest_cursor_needs_reconcile = True
            self._rekey_host_rewrite_watch(messages, result)
            return result
        except BaseException:
            self._compress_occurrences = None
            self._last_compression_status = "error"
            self._last_compression_noop_reason = ""
            raise

    @staticmethod
    def _public_compression_row(message: Any) -> Any:
        if not isinstance(message, dict):
            return message
        return {key: value for key, value in message.items()
                if key != "timestamp" and not str(key).startswith("_")}

    def _record_compress_commit_proof(self, messages, result) -> None:
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

    def _persist_compress_commit_proof(self, proof) -> None:
        """Durable twin of the process-local proof: lets a restarted/resumed
        process re-index the host's post-compaction list without guessing.

        Written for a published LCM compaction or adopted native recovery;
        ``last_store_id`` marks where later rows begin.
        """
        try:
            tail = self._store.get_session_tail(self._session_id, limit=1)
            effective = [_commit_proof_identity_digest(identity) for identity in proof["output_effective"]]
            full = [_commit_proof_identity_digest(identity) for identity in proof["output"]]
            payload = {
                "version": _COMPACTION_COMMIT_PROOF_WIRE_VERSION,
                "descriptor_version": _COMPACTION_COMMIT_PROOF_VERSION,
                # Scope: the Hermes home that wrote it (a configured shared
                # database_path serves several homes) and its creation time,
                # so a proof older than a lifecycle reset is ignored.
                "hermes_home": str(self._hermes_home or ""),
                "session_id": proof.get("session_id") or "",
                "conversation_id": proof.get("conversation_id") or "",
                "reset_epoch": proof.get("reset_epoch"),
                "created_at": time.time(),
                # effective_sha256/scaffold_sha256 are what v0.24.0 compares; the *_v4 twins are
                # this reader's projection digests, swapped in for a bound record (#517).
                "effective_sha256": proof.get("effective_sha256_v3", effective),
                "effective_sha256_v4": effective,
                "last_store_id": int(tail[-1]["store_id"]) if tail else 0,
                "native": bool(proof.get("native")),
                "droppable": list(proof.get("droppable") or []),
                "skip_landing": list(proof.get("skip_landing") or []),
                "native_summary_index": proof.get("native_summary_index"),
                "carry_ranges": [
                    list(item)
                    for item in self._coalesce_compression_carry_ranges(
                        proof.get("carry_ranges") or []
                    )
                ],
                "emissions": copy.deepcopy(proof.get("emissions") or []),
            }
            if not payload["effective_sha256"]:
                # Scaffold-only output: bind the proof to the emitted rows (#484 item 11l).
                payload["scaffold_sha256"] = proof.get("output_sha256_v3", full)
            if not effective:
                payload["scaffold_sha256_v4"] = full
            self._store.write_metadata_json(
                [self._replay_snapshot_metadata_key(_COMPACTION_COMMIT_PROOF_METADATA_PREFIX)],
                json.dumps(payload, sort_keys=True),
                skip_unchanged=True,
            )
        except Exception:
            logger.debug("LCM durable compaction-commit proof write failed", exc_info=True)

    def _fail_open_after_publication_failure(
        self,
        active_context: List[Dict[str, Any]],
        exc: BaseException,
        *,
        compress_started: float,
        threshold_full_sweep_active: bool,
        recovery_assembly_cap: int | None,
        leaf_passes: int,
        condensation_passes: int = 0,
        context_is_assembled: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return a replay-safe active view after publication cannot finish."""
        self._store.rollback_pending_write()
        fallback = active_context
        if recovery_assembly_cap is not None and not context_is_assembled:
            leading_anchor_count = self._leading_anchor_count(active_context)
            fallback = self._assemble_overflow_recovery_context(
                active_context[0] if leading_anchor_count else None,
                active_context[leading_anchor_count:],
                assembly_cap_override=recovery_assembly_cap,
                **(
                    {"retained_user_message": active_context[1]}
                    if leading_anchor_count == 2
                    else {}
                ),
            )
        self._last_compression_status = "error"
        binding_changed = isinstance(exc, LifecycleBindingChangedError)
        if binding_changed:
            failure_reason, noop_reason = (
                "lifecycle_binding_changed",
                "summary publication lost its lifecycle binding"
            )
        elif isinstance(exc, LifecyclePublicationConflictError):
            failure_reason, noop_reason = (
                "publication_invariant_conflict",
                "summary publication could not prove contiguous source coverage"
            )
        else:
            failure_reason, noop_reason = (
                "sqlite_publication_locked",
                "summary publication blocked by SQLite lock"
            )
        self._last_compression_noop_reason = noop_reason
        self._ingest_cursor = len(fallback)
        self._ingest_cursor_needs_reconcile = False
        self._last_compaction_duration_ms = (
            time.perf_counter() - compress_started
        ) * 1000.0
        if recovery_assembly_cap is not None:
            self._last_overflow_recovery_failed = (
                count_messages_tokens(fallback) > recovery_assembly_cap
            )
        if threshold_full_sweep_active:
            self._last_threshold_full_sweep = {
                **self._last_threshold_full_sweep,
                "status": "error",
                "leaf_passes": leaf_passes,
                "condensation_passes": condensation_passes,
                "total_passes": leaf_passes + condensation_passes,
                "duration_ms": round(self._last_compaction_duration_ms, 3),
                "stop_reason": failure_reason,
            }
        logger.warning(
            "LCM summary publication could not finish; preserving replay-safe "
            "context (reason=%s, code=%s, name=%s)",
            failure_reason,
            getattr(exc, "sqlite_errorcode", None),
            getattr(exc, "sqlite_errorname", None),
        )
        return fallback

    def _compress_native_recovery(
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
        self._remember_native_recovery_replay_snapshot(recovered)
        self._ingest_cursor = len(recovered)
        self._ingest_cursor_needs_reconcile = False
        self._last_native_summary_index = next((i for i, row in enumerate(recovered) if ContextCompressor._strip_context_summary_handoff_message(row) != row), None)
        logger.info(
            "Native recovery returned a summary for the host archive transaction "
            "(input_rows=%d, recovered_rows=%d, cursor=%d); LCM source history and frontier are unchanged.",
            len(messages), len(recovered), self._ingest_cursor,
        )
        return recovered

    def _assemble_committed_compaction_context(
        self,
        working_messages: List[Dict[str, Any]],
        anchor_source_messages: List[Dict[str, Any]],
        recovery_assembly_cap: int | None,
    ) -> List[Dict[str, Any]]:
        """Assemble and register replay proof for every committed leaf."""
        leading_anchor_count = self._leading_anchor_count(working_messages)
        anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
        self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
        try:
            return self._assemble_context(
                working_messages[0] if leading_anchor_count else None,
                working_messages[leading_anchor_count:],
                assembly_cap_override=recovery_assembly_cap,
                **(
                    {"retained_user_message": working_messages[1]}
                    if leading_anchor_count == 2
                    else {}
                ),
            )
        finally:
            self._pending_context_anchor_messages = None

    def _stored_publication_filter_exclusions(
        self,
        expected_frontier: int,
        covered_end: int,
        already_proven_store_ids: List[int],
        initial_proofs: Dict[int, Any],
        carried_ranges: List[tuple[str, int, int]] | None = None,
    ) -> Dict[int, Any]:
        proven = {int(store_id) for store_id in already_proven_store_ids}
        proofs = dict(initial_proofs)
        scan_ranges = [(self._session_id, expected_frontier, covered_end)]
        scan_ranges.extend(
            (source, max(expected_frontier, start), min(covered_end, end))
            for source, start, end in (carried_ranges or [])
        )
        for source_session_id, range_start, range_end in scan_ranges:
            after_store_id = range_start
            while after_store_id < range_end:
                rows = self._store.get_session_messages_after(
                    source_session_id,
                    after_store_id=after_store_id,
                )
                if not rows:
                    break
                for row in rows:
                    store_id = int(row.get("store_id") or 0)
                    if store_id > range_end:
                        break
                    if (
                        store_id not in proven
                        and store_id not in proofs
                        and self._matches_ignore_message_patterns(row, stored_row=True)
                    ):
                        proofs[store_id] = row.get("content")
                after_store_id = int(rows[-1].get("store_id") or after_store_id)
        return proofs

    def _committed_replay_drops(
        self,
        working: List[Dict[str, Any]],
        start: int,
    ) -> tuple[list[int], Optional[dict[int, int]], int, int]:
        """(#457) Positions of the replayed run at ``start`` that committed lineage accounts
        for, the store-id map of ``working[start:]`` when computed, the token cost of the
        covering leaves, and how many run rows stay. The run must be exactly this session's
        durable rows ending at the lifecycle frontier F and followed by F+1 (or nothing); a
        rotation child's run behind a verified summary head is its lineage (C, F] (#526).
        A row is consumed only when a leaf's ``source_ids`` hold it, or when it is an
        assistant/tool reply the publication folded in after the first lineage row (a run
        trailing the lineage must hold only such replies). Every other row stays in place:
        a user row outside lineage is never consumed. Any failed check consumes nothing:
        today's behaviour. Leaf nodes are deleted only by a session reset or purge, which
        can only shrink the lineage and so keep more rows."""
        state = self._lifecycle.get_by_conversation(self._conversation_id)
        frontier = int(getattr(state, "current_frontier_store_id", 0) or 0)
        if (
            state is None
            or not self._session_id
            or str(state.current_session_id or "") != self._session_id
            or frontier <= 0
            or int(self._last_compacted_store_id or 0) != frontier
        ):
            return [], None, 0, 0
        ids = self._get_store_id_map_for_messages(working[start:])
        # #524: LCM scaffold heading a superseded emission drops with the run it heads.
        scaffold, (start, carrier) = start, self._replay_head(working, start, frontier)
        end = next(
            (i for i in range(start, len(working)) if ids.get(id(working[i]), 0) > frontier),
            len(working),
        )
        rows = self._store.get_session_rows_through(self._session_id, frontier, end - start)
        if (carrier or start > scaffold) and len(rows) != end - start:  # #526: a rotation child's run is its lineage
            rows = self._head_lineage_rows(working[start if carrier else start - 1], frontier, end - start) or rows
        after = self._store.get_session_messages_after(self._session_id, frontier, limit=1)
        if (
            not rows
            or len(rows) != end - start
            or int(rows[-1]["store_id"]) != frontier
            or [int(row["store_id"]) for row in after]
            != ([ids[id(working[end])]] if end < len(working) else [])
        ):
            return [], ids, 0, 0
        self._load_host_rewrite_overrides(rows)
        if not all(
            self._replay_row_admits(message, row, carrier=carrier and not offset)
            for offset, (message, row) in enumerate(zip(working[start:end], rows))
        ):
            return [], ids, 0, 0
        leaf_sources = self._dag.get_leaf_sources_through(self._session_id, frontier)
        lineage = {store_id for _node, _tokens, store_id in leaf_sources}
        replies = {"assistant", "tool"}
        low, high = (min(lineage), max(lineage)) if lineage else (0, 0)
        if not lineage or any(
            int(row["store_id"]) > high and str(row.get("role") or "") not in replies
            for row in rows
        ):
            return [], ids, 0, 0
        dropped = {
            start + offset: int(row["store_id"])
            for offset, row in enumerate(rows)
            if int(row["store_id"]) in lineage
            or (int(row["store_id"]) > low and str(row.get("role") or "") in replies)
        }
        dropped_ids = set(dropped.values())
        covering = {node: tokens for node, tokens, store_id in leaf_sources if store_id in dropped_ids}
        return [*range(scaffold, start), *sorted(dropped)], ids, sum(covering.values()), len(rows) - len(dropped)

    def _compress_impl(self, messages: List[Dict[str, Any]],
                       current_tokens: int = None,
                       focus_topic: Optional[str] = None,
                       force: bool = False) -> List[Dict[str, Any]]:
        """Main compaction entry point.

        1. Ingest any new messages into the store
        2. Identify messages outside the fresh tail
        3. Summarize them into DAG leaf nodes
        4. Check if condensation is needed
        5. Assemble new active context: summaries + fresh tail
        """
        # Preflight handoffs are one-shot instructions for this invocation.
        # Consume them before every early return so a later unrelated turn can
        # never inherit stale cleanup-only state.
        native_cleanup_only_requested = bool(
            self._native_recovery_preflight_cleanup_only
        )
        boundary_cleanup_only_requested = bool(
            self._preflight_cleanup_only_due_to_boundary_cooldown
        )
        self._native_recovery_preflight_cleanup_only = False
        self._preflight_cleanup_only_due_to_boundary_cooldown = False

        if not messages:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "empty message list"
            return messages

        self._last_compression_status = "running"
        self._last_compression_noop_reason = ""
        _compress_started = time.perf_counter()

        self._maybe_reclassify_late_auxiliary_before_compaction_write()
        if self._bypasses_lcm_context_management():
            # Bypassed traffic observes nothing about the pressured session's
            # tail: it must neither extend nor reset the blocked streak.
            self._pressure_yield_invocation_verdict = "neutral"
            bypass_current_tokens = current_tokens
            if bypass_current_tokens is None or bypass_current_tokens <= 0:
                auxiliary_session_id = self._thread_context_session_id()
                if auxiliary_session_id:
                    auxiliary_prompt_tokens = self._current_auxiliary_prompt_tokens(
                        auxiliary_session_id
                    )
                    if auxiliary_prompt_tokens > 0:
                        bypass_current_tokens = auxiliary_prompt_tokens
            return self._compress_lcm_bypassed_session(
                messages,
                current_tokens=bypass_current_tokens,
                focus_topic=focus_topic,
                force=force,
            )

        # ``current_tokens`` is optional in the ContextEngine contract. After a
        # yield-aware preflight, use the current active messages as the
        # pressure observation when the host calls ``compress(messages)`` so
        # the follow-up invocation can re-arm the advertised bounded yield.
        observed_prompt_tokens = (
            current_tokens
            if current_tokens is not None
            else count_messages_tokens(messages)
        )
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        # NOTE: deliberately do NOT clear the spend guard on force_overflow.
        # force_overflow is automatic (set every turn the prompt exceeds the
        # assembly cap), which is exactly the sustained-over-cap state a runaway
        # compaction loop produces - clearing it per turn would defeat the guard
        # in the case it exists for. A tripped guard still converges the
        # emergency via deterministic L3 truncation (no LLM spend).
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        # Step 1: Ingest new messages into the immutable store. Work from a
        # replay-safe view so quarantined assistant loops do not enter summaries
        # or provider context after the durable row has been written.
        working_messages = self._ingest_messages(messages)
        # #488: project the complete admitted list ONCE; subset consumers read a row's occurrence
        # here. Only a row at exactly one position is registered (F6): a copy or an aliased row
        # is unproven (full identity).
        occurrences, v4 = self._replay_occurrences(working_messages)
        positions = Counter(id(m) for m in working_messages)
        self._compress_occurrences = {
            id(m): occurrence for m, occurrence in zip(working_messages, occurrences) if positions[id(m)] == 1
        } if v4 else None
        self._prepare_retained_user_anchor(working_messages)
        native_cleanup_only = bool(
            self._config.native_recovery
            and native_cleanup_only_requested
            and not force
            and not force_overflow
            and (
                self.threshold_tokens <= 0
                or observed_prompt_tokens < self.threshold_tokens
            )
        )
        ingest_cleanup_changed_active_context = working_messages != messages
        cleanup_only_requested = bool(
            (
                boundary_cleanup_only_requested
                or native_cleanup_only
            )
            and not force_overflow
        )
        if cleanup_only_requested:
            sanitized_messages = self._sanitize_active_context_messages(
                working_messages,
                insert_missing_tool_stubs=False,
            )
            self._refresh_raw_backlog_debt(
                sanitized_messages,
                observed_tokens=observed_prompt_tokens,
            )
            self._ingest_cursor = len(sanitized_messages)
            self._last_compression_status = "sanitized"
            self._last_compression_noop_reason = ""
            if native_cleanup_only:
                self._last_compress_aborted = False
            self._note_fresh_tail_pressure_relieved()
            self._write_generated_ignored_placeholder_hash_counts(
                self._generated_placeholder_digest_budget_for_active_replay(
                    sanitized_messages
                )
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                self._generated_placeholder_digest_ordinals_for_active_replay(
                    sanitized_messages
                )
            )
            return sanitized_messages
        if self._config.native_recovery:
            # Even a rejected summary must retain ingest's replay protections.
            # The helper preserves its error/aborted status and returns the
            # sanitized working input when native recovery cannot finish.
            return self._compress_native_recovery(
                working_messages,
                current_tokens=observed_prompt_tokens,
            )
        anchor_source_messages = list(working_messages)
        pressure_messages = messages if len(messages) == len(working_messages) else working_messages
        leaf_compacted_this_turn = False
        dropped_replayed_scaffold_messages = False
        resumed_prefix, resumed_ahead = False, 0
        leaf_passes = 0
        estimated_active_tokens = (
            observed_prompt_tokens
            if observed_prompt_tokens is not None and observed_prompt_tokens > 0
            else count_messages_tokens(messages)
        )
        threshold_full_sweep_active = bool(
            self._config.threshold_full_sweep_enabled
            and not force_overflow
            and self.threshold_tokens > 0
            and estimated_active_tokens >= self.threshold_tokens
        )
        sweep_deadline = time.monotonic() + _THRESHOLD_FULL_SWEEP_MAX_SECONDS
        configured_sweep_target = int(self._config.summary_prefix_target_tokens)
        sweep_target_tokens = max(
            1,
            configured_sweep_target
            if configured_sweep_target > 0
            else int(self._config.leaf_chunk_tokens),
        )
        sweep_summary_prefix_before = (
            self._summary_frontier_tokens() if threshold_full_sweep_active else 0
        )
        if threshold_full_sweep_active:
            self._last_threshold_full_sweep = {
                "status": "running",
                "leaf_passes": 0,
                "condensation_passes": 0,
                "total_passes": 0,
                "duration_ms": 0.0,
                "tokens_before": estimated_active_tokens,
                "tokens_after": estimated_active_tokens,
                "summary_prefix_tokens_before": sweep_summary_prefix_before,
                "summary_prefix_tokens_after": sweep_summary_prefix_before,
                "summary_prefix_target_tokens": sweep_target_tokens,
                "stop_reason": "",
                "budget_exhausted": False,
            }
        critical_budget_pressure = self._critical_budget_pressure_reached(
            observed_tokens=observed_prompt_tokens,
            messages=working_messages,
        )
        deferred_maintenance_active = (
            not force_overflow
            and not threshold_full_sweep_active
            and self._should_run_deferred_maintenance(
                working_messages,
                observed_tokens=observed_prompt_tokens,
            )
        )
        if deferred_maintenance_active:
            self._lifecycle.record_maintenance_attempt(self._conversation_id)
        base_max_leaf_passes = 4 if self._config.dynamic_leaf_chunk_enabled else 1
        max_leaf_passes = base_max_leaf_passes
        if threshold_full_sweep_active:
            max_leaf_passes = _THRESHOLD_FULL_SWEEP_MAX_PASSES
        if deferred_maintenance_active:
            max_leaf_passes = max(1, self._config.deferred_maintenance_max_passes)

        explicit_focus_topic = focus_topic is not None

        noop_reason = "no eligible raw backlog outside fresh tail"
        sweep_stop_reason = ""
        sweep_raw_drained = False
        dependent_reply_message_ids: set[int] = set()
        preexisting_dependent_reply_records = self._load_generated_ignored_dependent_reply_records()

        while leaf_passes < max_leaf_passes:
            if threshold_full_sweep_active and time.monotonic() >= sweep_deadline:
                sweep_stop_reason = "time_budget_exhausted"
                break
            fresh_tail_start = self._fresh_tail_start(pressure_messages)

            # Keep only a real system prompt anchored. Gateway sessions may
            # pass only conversation messages, so index 0 can be an old user
            # turn; that must remain eligible for compaction instead of being
            # replayed forever as fresh-looking intent.
            leading_anchor_count = self._leading_anchor_count(working_messages)
            publication_excluded_store_ids = self._get_store_ids_for_messages(
                working_messages[:leading_anchor_count]
            )
            filter_exclusion_proofs: Dict[int, Any] = {}
            if fresh_tail_start <= leading_anchor_count:
                # Also reached with threshold_full_sweep_active: a sweep whose
                # "drained" raw prefix is really a tail covering the whole
                # session must yield like any other blocked pass, or the sweep
                # condenses nothing and the deadlock survives in sweep mode.
                if self._maybe_engage_fresh_tail_pressure_yield(
                    pressure_messages,
                    observed_prompt_tokens,
                    eligible_tokens=0,
                ):
                    continue
                noop_reason = "no eligible raw backlog outside fresh tail"
                if threshold_full_sweep_active:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                break

            candidate_start = leading_anchor_count
            while candidate_start < fresh_tail_start and (
                self._is_replayed_context_scaffold_message(working_messages[candidate_start])
                if self._compress_occurrences is None  # no v4 proof in force: the rc4 contract
                else self._compress_occurrences.get(id(working_messages[candidate_start]), (None, ()))[1] is None
            ):
                candidate_start += 1
            # #457: a retry replays rows a cancelled attempt already committed; drop
            # those with the scaffold, keeping any row committed lineage does not hold.
            # Reuse the map when the pass maps this same list.
            drops, premapped_store_ids, kept = set(range(leading_anchor_count, candidate_start)), None, 0
            if leaf_passes == 0 and not resumed_prefix:
                resumed, store_ids, summary_tokens, kept = self._committed_replay_drops(
                    working_messages, candidate_start
                )
                resumed_prefix = bool(resumed)
                resumed_ahead = resumed[0] - candidate_start if resumed else 0  # kept rows before lineage
                premapped_store_ids = None if drops or resumed else store_ids
                if resumed:  # the committed summary replaces the dropped rows in the estimate
                    resumed_tokens = count_messages_tokens([working_messages[index] for index in resumed])
                    estimated_active_tokens = max(0, estimated_active_tokens - resumed_tokens + summary_tokens)
                drops.update(resumed)
            if drops:
                publication_excluded_store_ids.extend(
                    self._get_store_ids_for_messages(
                        working_messages[leading_anchor_count:candidate_start]
                    )
                )
                dropped_replayed_scaffold_messages = True
                working_messages = [message for index, message in enumerate(working_messages) if index not in drops]
                pressure_messages = [message for index, message in enumerate(pressure_messages) if index not in drops]
                candidate_start = leading_anchor_count
                fresh_tail_start = self._fresh_tail_start(pressure_messages)
                # A kept row at or below F has no raw store lineage for a new leaf: return it
                # raw after the committed summary instead of a pass that cannot publish.
                if fresh_tail_start <= leading_anchor_count or (resumed_prefix and kept):
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            if candidate_start < fresh_tail_start:
                self._current_compress_store_ids_by_message_id = (
                    premapped_store_ids
                    if premapped_store_ids is not None
                    else self._get_store_id_map_for_messages(working_messages[leading_anchor_count:])
                )
                compactable_pairs = list(
                    zip(
                        working_messages[candidate_start:fresh_tail_start],
                        pressure_messages[candidate_start:fresh_tail_start],
                    )
                )
                kept_working: list[Dict[str, Any]] = []
                kept_pressure: list[Dict[str, Any]] = []
                dropped_ignored_backlog = False
                drop_dependent_reply = False
                for working_msg, pressure_msg in compactable_pairs:
                    role = str(working_msg.get("role") or "")
                    content_text = text_content_for_pattern_matching(working_msg.get("content")) or ""
                    generated_dependent_reply = self._is_generated_ignored_dependent_reply(
                        working_msg,
                        content_text,
                    )
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    generated_volatile_placeholder = (
                        self._is_volatile_ignored_quarantine_placeholder(working_msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in self._load_generated_ignored_placeholder_hashes()
                    )
                    configured_filter_match = (
                        self._matches_ignore_message_patterns(working_msg)
                        or self._matches_ignore_message_patterns(pressure_msg)
                        or self._mapped_stored_row_matches_ignore_message_patterns(working_msg)
                    )
                    if (
                        configured_filter_match
                        or self._is_ignored_active_replay_placeholder(working_msg, content_text)
                        or generated_volatile_placeholder
                    ):
                        mapped_store_id = (
                            self._current_compress_store_ids_by_message_id.get(
                                id(working_msg)
                            )
                        )
                        if mapped_store_id is not None:
                            store_id = int(mapped_store_id)
                            publication_excluded_store_ids.append(store_id)
                            stored = self._store.get(store_id)
                            if configured_filter_match:
                                filter_exclusion_proofs[store_id] = (
                                    stored.get("content")
                                    if stored is not None
                                    and self._matches_ignore_message_patterns(
                                        stored,
                                        stored_row=True,
                                    )
                                    else _UNPROVEN_FILTER_EXCLUSION
                                )
                        dropped_ignored_backlog = True
                        if role in {"user", "system", "tool", "assistant"}:
                            drop_dependent_reply = True
                        continue
                    if generated_dependent_reply:
                        dependent_reply_message_ids.add(id(working_msg))
                        if role in {"assistant", "tool"}:
                            drop_dependent_reply = True
                    if drop_dependent_reply and role in {"assistant", "tool"}:
                        dependent_reply_message_ids.add(id(working_msg))
                        self._remember_generated_ignored_dependent_reply(working_msg, content_text)
                    if role in {"user", "system"}:
                        drop_dependent_reply = False
                    kept_working.append(working_msg)
                    kept_pressure.append(pressure_msg)
                drop_dependent_reply_into_tail = drop_dependent_reply
                if dropped_ignored_backlog:
                    dropped_replayed_scaffold_messages = True
                    working_messages = (
                        working_messages[:candidate_start]
                        + kept_working
                        + working_messages[fresh_tail_start:]
                    )
                    pressure_messages = (
                        pressure_messages[:candidate_start]
                        + kept_pressure
                        + pressure_messages[fresh_tail_start:]
                    )
                    fresh_tail_start = self._fresh_tail_start(pressure_messages)
                if drop_dependent_reply_into_tail:
                    tail_scan_start = max(fresh_tail_start, leading_anchor_count)
                    pending_tail_dependents: list[tuple[Dict[str, Any], str]] = []
                    saw_tail_boundary = False
                    for tail_msg in working_messages[tail_scan_start:]:
                        if not isinstance(tail_msg, dict):
                            continue
                        tail_role = str(tail_msg.get("role") or "")
                        if tail_role in {"user", "system"}:
                            saw_tail_boundary = True
                            break
                        if tail_role in {"assistant", "tool"}:
                            tail_text = text_content_for_pattern_matching(tail_msg.get("content")) or ""
                            self._remember_generated_ignored_dependent_reply(tail_msg, tail_text)
                            pending_tail_dependents.append((tail_msg, tail_text))
                    if saw_tail_boundary or leading_anchor_count > 0 or kept_working:
                        for tail_msg, _tail_text in pending_tail_dependents:
                            dependent_reply_message_ids.add(id(tail_msg))
                if dropped_ignored_backlog and fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            # Auto-derive focus topic from the post-filter compaction view when
            # not explicitly provided.  The derived focus is summarizer-visible,
            # so it must follow the same ignored-message filtering as the leaf
            # chunk itself.
            if not explicit_focus_topic:
                focus_topic = self._derive_auto_focus_topic(working_messages)

            candidate_raw = working_messages[leading_anchor_count:fresh_tail_start]
            if not candidate_raw:
                if self._maybe_engage_fresh_tail_pressure_yield(
                    pressure_messages,
                    observed_prompt_tokens,
                    eligible_tokens=0,
                ):
                    continue
                noop_reason = "no eligible raw backlog outside fresh tail"
                if threshold_full_sweep_active:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                break

            pressure_candidate_raw = pressure_messages[leading_anchor_count:fresh_tail_start]
            raw_tokens_outside_tail = count_messages_tokens(pressure_candidate_raw)
            if threshold_full_sweep_active:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(
                    raw_tokens_outside_tail
                )
                to_compact = self._select_oldest_leaf_chunk(
                    candidate_raw,
                    working_leaf_chunk_tokens,
                )
            elif self._config.dynamic_leaf_chunk_enabled:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
                # An armed pressure yield waives the leaf-chunk minimum: the
                # whole point of the yield is progress, and the freed backlog
                # can legitimately be smaller than one configured chunk.
                if (
                    raw_tokens_outside_tail < working_leaf_chunk_tokens
                    and not force_overflow
                    and self._pressure_yield_tail_token_limit <= 0
                ):
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        if self._maybe_engage_fresh_tail_pressure_yield(
                            pressure_messages,
                            observed_prompt_tokens,
                            eligible_tokens=raw_tokens_outside_tail,
                        ):
                            continue
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                if force_overflow:
                    to_compact = candidate_raw
                else:
                    to_compact = self._select_oldest_leaf_chunk(candidate_raw, working_leaf_chunk_tokens)
            else:
                if (
                    raw_tokens_outside_tail < self._config.leaf_chunk_tokens
                    and not force_overflow
                    and self._pressure_yield_tail_token_limit <= 0
                ):
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        if self._maybe_engage_fresh_tail_pressure_yield(
                            pressure_messages,
                            observed_prompt_tokens,
                            eligible_tokens=raw_tokens_outside_tail,
                        ):
                            continue
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                if force_overflow:
                    to_compact = candidate_raw
                elif self._pressure_yield_tail_token_limit > 0:
                    to_compact = self._select_oldest_leaf_chunk(
                        candidate_raw,
                        max(1, int(self._config.leaf_chunk_tokens)),
                    )
                else:
                    to_compact = candidate_raw

            if not to_compact:
                noop_reason = "no eligible leaf chunk selected"
                break

            selected_raw_chunk = to_compact
            sources = {}
            summary_input_chunk = []
            for message in selected_raw_chunk:
                if id(message) in dependent_reply_message_ids:
                    continue
                remainder = self._generated_context_carrier_remainder(message)
                if remainder is not None:
                    original = message
                    message = {**message, "content": remainder}
                    sources[id(message)] = original
                summary_input_chunk.append(message)
            if not summary_input_chunk:
                compacted_chunk = selected_raw_chunk
                source_tokens = count_messages_tokens(selected_raw_chunk)
                summary_text = (
                    "Filtered replies derived from ignored messages.\n"
                    "[Expand for details: ignored-dependent reply]"
                )
                _level = 0
                _rescue_attempts = 0
            else:
                # Pre-compaction extraction: best-effort, never blocks compaction.
                # Use the same dependency-filtered view as summarization so ignored
                # turns cannot leak through derived assistant/tool replies.
                if self._config.extraction_enabled:
                    extraction_timeout = None
                    if threshold_full_sweep_active:
                        extraction_timeout = max(0.001, sweep_deadline - time.monotonic())
                    self._run_pre_compaction_extraction(
                        summary_input_chunk,
                        timeout_seconds=extraction_timeout,
                    )
                if bool(
                    getattr(
                        self._config,
                        "assertion_extraction_enabled",
                        False,
                    )
                ):
                    self._schedule_pre_compaction_assertions(summary_input_chunk)

                try:
                    summary_kwargs: dict[str, Any] = {"focus_topic": focus_topic}
                    if threshold_full_sweep_active:
                        summary_kwargs["deadline"] = sweep_deadline
                    (
                        compacted_chunk,
                        source_tokens,
                        summary_text,
                        _level,
                        _rescue_attempts,
                    ) = self._summarize_leaf_chunk_with_rescue(
                        summary_input_chunk,
                        **summary_kwargs,
                    )
                except Exception as exc:
                    if threshold_full_sweep_active and leaf_compacted_this_turn:
                        sweep_stop_reason = "leaf_summary_error"
                        logger.warning(
                            "LCM threshold full sweep stopped after %d persisted leaf pass(es): %s",
                            leaf_passes,
                            exc,
                        )
                        break
                    raise
            compacted_chunk = [sources.get(id(message), message) for message in compacted_chunk]
            compacted_summary_ids = {id(message) for message in compacted_chunk}
            compacted_positions = [
                idx for idx, message in enumerate(selected_raw_chunk) if id(message) in compacted_summary_ids
            ]
            last_compacted_raw_pos = max(compacted_positions) if compacted_positions else len(compacted_chunk) - 1
            last_consumed_raw_pos = last_compacted_raw_pos
            while (
                last_consumed_raw_pos + 1 < len(selected_raw_chunk)
                and id(selected_raw_chunk[last_consumed_raw_pos + 1]) in dependent_reply_message_ids
            ):
                last_consumed_raw_pos += 1
            source_lookup_chunk = selected_raw_chunk[: last_consumed_raw_pos + 1]
            selected_raw_len = len(source_lookup_chunk)
            remaining_messages = working_messages[leading_anchor_count + selected_raw_len:]
            source_tokens = count_messages_tokens(source_lookup_chunk)

            source_lineage_chunk = [
                message for message in source_lookup_chunk if id(message) not in dependent_reply_message_ids
            ]
            source_store_ids = self._get_store_ids_for_messages(source_lineage_chunk)
            source_store_ids = sorted(dict.fromkeys(source_store_ids))
            consumed_store_ids = self._get_store_ids_for_messages(source_lookup_chunk)
            consumed_store_ids = sorted(dict.fromkeys(consumed_store_ids))
            earliest_at, latest_at = self._store.get_time_bounds(source_store_ids)
            summary_tokens = count_tokens(summary_text)

            node = SummaryNode(
                session_id=self._session_id,
                depth=0,
                summary=summary_text,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                source_ids=source_store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            published_frontier = (
                max(consumed_store_ids) if consumed_store_ids else 0
            )
            publication_state = self._lifecycle.get_by_conversation(
                self._conversation_id
            )
            expected_frontier = int(
                getattr(publication_state, "current_frontier_store_id", 0)
            )
            carried_ranges = self._load_compression_carry_ranges()
            filter_exclusion_proofs = self._stored_publication_filter_exclusions(
                expected_frontier,
                published_frontier,
                consumed_store_ids,
                filter_exclusion_proofs,
                carried_ranges,
            )
            publication_excluded_store_ids.extend(filter_exclusion_proofs)
            # The frontier consumes every durable row removed from the active
            # prefix, including trailing dependent replies. Summary lineage
            # excludes those replies; their durable ledger drives replay cleanup.
            try:
                before_commit = None
                if self._session_id and self._conversation_id:
                    publication_session_id = self._session_id
                    publication_conversation_id = self._conversation_id
                    def stage_frontier(conn, node_id) -> None:
                        self._lifecycle.stage_compaction_publication(
                            conn,
                            publication_conversation_id,
                            publication_session_id,
                            node_id,
                            expected_frontier,
                            consumed_store_ids,
                            publication_excluded_store_ids,
                            filter_exclusion_proofs,
                            carried_ranges,
                        )
                    before_commit = stage_frontier
                self._dag.add_node(node, before_commit=before_commit)
            except Exception as exc:
                if (
                    not _is_sqlite_locked_error(exc)
                    and not isinstance(exc, LifecycleBindingChangedError)
                    and not isinstance(exc, LifecyclePublicationConflictError)
                ):
                    raise
                fallback = working_messages
                context_is_assembled = False
                if leaf_passes or dropped_replayed_scaffold_messages:
                    fallback = self._assemble_committed_compaction_context(
                        working_messages,
                        anchor_source_messages,
                        recovery_assembly_cap,
                    )
                    context_is_assembled = True
                return self._fail_open_after_publication_failure(
                    fallback,
                    exc,
                    compress_started=_compress_started,
                    threshold_full_sweep_active=threshold_full_sweep_active,
                    recovery_assembly_cap=recovery_assembly_cap,
                    leaf_passes=leaf_passes,
                    context_is_assembled=context_is_assembled,
                )
            self._last_compacted_store_id = published_frontier
            self._invalidate_rollups_for_published_node(node)

            pressure_remaining_messages = pressure_messages[leading_anchor_count + selected_raw_len:]
            working_messages = working_messages[:leading_anchor_count] + remaining_messages
            pressure_messages = pressure_messages[:leading_anchor_count] + pressure_remaining_messages
            leaf_compacted_this_turn = True
            leaf_passes += 1
            estimated_active_tokens = max(0, estimated_active_tokens - source_tokens + summary_tokens)
            if (
                getattr(self._config, "large_output_transcript_gc_enabled", False)
                and source_store_ids
            ):
                committed_context = self._assemble_committed_compaction_context(
                    working_messages,
                    anchor_source_messages,
                    recovery_assembly_cap,
                )
                try:
                    self._maybe_gc_compacted_tool_results(
                        compacted_chunk,
                        source_store_ids,
                    )
                except Exception as exc:
                    if not _is_sqlite_locked_error(exc):
                        raise
                    return self._fail_open_after_publication_failure(
                        committed_context,
                        exc,
                        compress_started=_compress_started,
                        threshold_full_sweep_active=threshold_full_sweep_active,
                        recovery_assembly_cap=recovery_assembly_cap,
                        leaf_passes=leaf_passes,
                        context_is_assembled=True,
                    )

            if threshold_full_sweep_active:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                if not remaining_raw:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                    break
                continue

            if not self._config.dynamic_leaf_chunk_enabled:
                break

            if not force_overflow:
                if (not deferred_maintenance_active) and self.threshold_tokens > 0 and estimated_active_tokens < self.threshold_tokens:
                    break
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                if not remaining_raw:
                    break
                pressure_remaining_raw = pressure_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                remaining_raw_tokens = count_messages_tokens(pressure_remaining_raw)
                remaining_threshold = self._working_leaf_chunk_tokens(remaining_raw_tokens)
                if remaining_raw_tokens < remaining_threshold:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        break

        if (
            threshold_full_sweep_active
            and not sweep_raw_drained
            and not sweep_stop_reason
            and leaf_passes >= max_leaf_passes
        ):
            sweep_stop_reason = "pass_budget_exhausted"

        if not leaf_compacted_this_turn:
            self._refresh_raw_backlog_debt(
                working_messages,
                observed_tokens=observed_prompt_tokens,
            )
            if force_overflow and len(messages) >= 1:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                compressed = self._assemble_overflow_recovery_context(
                    working_messages[0] if leading_anchor_count else None,
                    working_messages[leading_anchor_count:],
                    assembly_cap_override=recovery_assembly_cap,
                    **(
                        {"retained_user_message": working_messages[1]}
                        if leading_anchor_count == 2
                        else {}
                    ),
                )
                return self._finalize_forced_overflow_result(
                    working_messages,
                    compressed,
                    assembly_cap_override=recovery_assembly_cap,
                    ingest_cleanup_changed_active_context=ingest_cleanup_changed_active_context,
                )
            active_context_messages = self._drop_preexisting_generated_ignored_dependent_eof_replies(
                working_messages,
                preexisting_dependent_reply_records,
            )
            if dropped_replayed_scaffold_messages:
                leading_anchor_count = self._leading_anchor_count(active_context_messages)
                if resumed_ahead == 1 and leading_anchor_count == 1 and active_context_messages[1].get("role") == "user":
                    leading_anchor_count = 2  # #457: attempt 1's retained anchor, kept, goes back ahead of its summary
                anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
                self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
                try:
                    sanitized_messages = self._assemble_context(
                        active_context_messages[0] if leading_anchor_count else None,
                        active_context_messages[leading_anchor_count:],
                        assembly_cap_override=recovery_assembly_cap,
                        **(
                            {"retained_user_message": active_context_messages[1]}
                            if leading_anchor_count == 2
                            else {}
                        ),
                    )
                finally:
                    self._pending_context_anchor_messages = None
            else:
                sanitized_messages = self._sanitize_active_context_messages(
                    active_context_messages,
                    insert_missing_tool_stubs=False,
                )
            if sanitized_messages != working_messages or ingest_cleanup_changed_active_context:
                # _ingest_messages() already advanced the cursor to the original
                # active-context length. If the host continues from a sanitized
                # or reassembled context, keeping the old cursor could make the
                # next appended messages look already ingested. This applies to
                # content-only cleanup as well as dropped-message cleanup.
                self._ingest_cursor = len(sanitized_messages)
                # A resumed committed prefix returns that compaction's output.
                self._last_compression_status = "compacted" if resumed_prefix else "sanitized"
                self._last_compression_noop_reason = ""
                self._note_fresh_tail_pressure_relieved()
            else:
                if dropped_replayed_scaffold_messages:
                    # The active context changed even though no new leaf node was
                    # written. Keep the cursor aligned with the returned context
                    # so the next appended turn is ingested instead of skipped.
                    self._ingest_cursor = len(sanitized_messages)
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = noop_reason
                logger.info("LCM compression no-op: %s", noop_reason)
            if threshold_full_sweep_active:
                duration_ms = (time.perf_counter() - _compress_started) * 1000.0
                self._last_threshold_full_sweep = {
                    **self._last_threshold_full_sweep,
                    "status": "noop",
                    "duration_ms": round(duration_ms, 3),
                    "stop_reason": sweep_stop_reason or noop_reason,
                    "budget_exhausted": sweep_stop_reason
                    in {"pass_budget_exhausted", "time_budget_exhausted"},
                }
            self._write_generated_ignored_placeholder_hash_counts(
                self._generated_placeholder_digest_budget_for_active_replay(sanitized_messages)
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                self._generated_placeholder_digest_ordinals_for_active_replay(sanitized_messages)
            )
            return sanitized_messages

        # Step 6: Check if condensation is needed. A threshold full sweep only
        # condenses after the eligible raw prefix has been drained, and shares
        # the same total pass/deadline budget as its leaf work.
        pre_condensation_context = self._assemble_committed_compaction_context(
            working_messages,
            anchor_source_messages,
            recovery_assembly_cap,
        )
        condensation_passes = 0
        try:
            if threshold_full_sweep_active:
                if sweep_raw_drained:
                    remaining_passes = max(
                        0,
                        _THRESHOLD_FULL_SWEEP_MAX_PASSES - leaf_passes,
                    )
                    condensation_passes, sweep_stop_reason = (
                        self._run_threshold_sweep_condensation(
                            target_tokens=sweep_target_tokens,
                            pass_budget=remaining_passes,
                            deadline=sweep_deadline,
                            focus_topic=focus_topic,
                        )
                    )
            else:
                condensation_passes = self._maybe_condense(
                    focus_topic=focus_topic,
                    leaf_compacted_this_turn=True,
                    force_overflow=force_overflow,
                    critical_budget_pressure=critical_budget_pressure,
                )
        except Exception as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            return self._fail_open_after_publication_failure(
                pre_condensation_context,
                exc,
                compress_started=_compress_started,
                threshold_full_sweep_active=threshold_full_sweep_active,
                recovery_assembly_cap=recovery_assembly_cap,
                leaf_passes=leaf_passes,
                condensation_passes=int(
                    getattr(exc, "lcm_completed_condensation_passes", 0)
                ),
                context_is_assembled=True,
            )

        # Step 7: Assemble new active context
        self._refresh_raw_backlog_debt(
            working_messages,
            observed_tokens=observed_prompt_tokens,
        )
        compressed = pre_condensation_context
        if condensation_passes:
            compressed = self._assemble_committed_compaction_context(
                working_messages,
                anchor_source_messages,
                recovery_assembly_cap,
            )
        self.compression_count += 1
        self._last_compaction_duration_ms = (time.perf_counter() - _compress_started) * 1000.0
        logger.info(
            "LCM leaf compaction finished in %.1fms", self._last_compaction_duration_ms
        )
        self._last_compression_status = "compacted"
        self._last_compression_noop_reason = ""
        self._note_fresh_tail_pressure_relieved()
        if recovery_assembly_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > recovery_assembly_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d after compaction; returning best-effort context (%d tokens)",
                    recovery_assembly_cap,
                    count_messages_tokens(compressed),
                )
        # Reset cursor to the length of the compressed context so that
        # only messages appended *after* this point get ingested next time.
        self._ingest_cursor = len(compressed)
        self._ingest_cursor_needs_reconcile = False

        logger.info(
            "LCM compaction #%d: %d messages → %d (%d leaf pass%s, %d→%d tokens, %d DAG nodes%s)",
            self.compression_count,
            len(messages),
            len(compressed),
            leaf_passes,
            "es" if leaf_passes != 1 else "",
            count_messages_tokens(messages),
            count_messages_tokens(compressed),
            len(self._dag.get_session_nodes(self._session_id)),
            ", forced overflow recovery" if force_overflow else "",
        )

        # ── Active-context cleanup / tool-pair guardrail (same as _assemble_context) ──
        # compress() output is consumed directly by the main loop in some
        # edge cases (e.g. forced overflow recovery bypassing _assemble_context).
        compressed = self._sanitize_active_context_messages(compressed)
        if threshold_full_sweep_active:
            total_passes = leaf_passes + condensation_passes
            duration_ms = (time.perf_counter() - _compress_started) * 1000.0
            final_stop_reason = sweep_stop_reason or "raw_prefix_drained"
            partial_stop_reasons = {
                "pass_budget_exhausted",
                "time_budget_exhausted",
                "leaf_summary_error",
                "condensation_error",
                "condensation_no_progress",
                "no_same_depth_condensation_group",
            }
            self._last_threshold_full_sweep = {
                "status": "partial" if final_stop_reason in partial_stop_reasons else "completed",
                "leaf_passes": leaf_passes,
                "condensation_passes": condensation_passes,
                "total_passes": total_passes,
                "duration_ms": round(duration_ms, 3),
                "tokens_before": self._last_threshold_full_sweep["tokens_before"],
                "tokens_after": count_messages_tokens(compressed),
                "summary_prefix_tokens_before": sweep_summary_prefix_before,
                "summary_prefix_tokens_after": self._summary_frontier_tokens(),
                "summary_prefix_target_tokens": sweep_target_tokens,
                "stop_reason": final_stop_reason,
                "budget_exhausted": final_stop_reason
                in {"pass_budget_exhausted", "time_budget_exhausted"},
            }
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(compressed)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(compressed)
        )
        record_successful_compaction = getattr(
            self,
            "_record_successful_compaction_telemetry",
            None,
        )
        if callable(record_successful_compaction):
            record_successful_compaction()

        return compressed
