"""Session-scoped runtime-state resets.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``ResetStateMixin``
(WS5 seam). The methods clear the session-scoped counters, compaction
progress, and per-turn placeholder-boundary bookkeeping when a session is
reset or rolled over. State stays on the engine (accessed via ``self``);
mixing this in leaves every call site and ``self._*`` reference unchanged.
"""

import uuid


class ResetStateMixin:
    def _reset_session_counters(self) -> None:
        """Reset session-scoped counters and token tracking.

        Safe to call on boundary skip because it does not affect compaction progress.
        """
        self._hung_summary_strike = None  # #1096: conversation changes also reset counters only
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self._last_gate_tokens = 0  # #954: a rebind never inherits the previous binding's gate observation
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self._compaction_telemetry_counter_epoch = uuid.uuid4().hex
        self._context_probed = False
        self._context_probe_persistable = False
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        self._last_stub_first_exit = None  # #671: the record describes the latest compaction of this binding
        self._last_survival_fit = None  # #1162: status must not expose the previous binding's fit
        self._last_boundary_skip_time = 0
        self._compaction_telemetry_counter_rebaseline_pending = True
        self._compaction_telemetry_turn_reset_pending = False

    def _reset_compaction_progress(self) -> None:
        """Reset process-local compaction markers for a fresh/unproven session."""
        self._frontier_holder().reset_progress()
        self._last_ingest_reconciliation = {"action": "none", "reason": "not run"}

    def _reset_session_scoped_runtime_state(self) -> None:
        """Reset all session-scoped runtime state.

        Calls both _reset_session_counters and _reset_compaction_progress.
        Proven carry-over paths must restore/advance state from the verified
        source lifecycle after rebinding.
        """
        self._reset_session_counters()
        self._reset_compaction_progress()
        # A commit proof is valid only through the compression handoff of the
        # session that produced it (#483); any other reset drops it.
        self._frontier_holder().reset_proofs()
        self._retired_durable_proof = None  # #671: a stale durable proof retired by reconcile, by created_at
        self._clear_fresh_tail_pressure_yield_state()
        self._generated_ignored_active_replay_placeholder_hashes = set()
        self._generated_ignored_active_replay_placeholder_message_ids = set()
        self._generated_ignored_active_replay_placeholder_messages = {}
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}
        # #651: a hold armed by the previous binding never holds the next one. #618: nor does the #608 hold.
        self._no_progress_hold, self._no_progress_candidate = None, False
        self._zero_leaf_streaks = {}  # #1162: nor does its zero-leaf streak (a kept conversation id included)
        self._objective_only_noop = False
        self._sweep_budget_hold_until, self._sweep_budget_hold_conversation = 0.0, ""
        self._hung_summary_strike, self._sweep_budget_hold_turn_paced = None, False
        # #677: a preflight request of the previous binding never shapes the next one's compress().
        self._compaction_request.below_threshold_cleanup_only = self._compaction_request.automatic_request = False
