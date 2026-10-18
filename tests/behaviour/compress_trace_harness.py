"""Golden observations of real compaction; regenerate with this module's CLI.

Only the already-validated provider boundary, clock and offline token estimator
are replaced. SQL tracing starts after scenario setup and includes ingestion,
publication, rollback and the next assembly. Consecutive identical callbacks
use a repeat count, preserving their order and multiplicity. SQL shapes are
defined once at first use; subsequent records reference that shape and hash
the full canonical bound statement.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# The plugin is directory-loaded by Hermes, not installed as a Python package.
# Reuse the suite's registration when invoked with python -m outside pytest.
import tests.conftest  # noqa: F401
from tests.behaviour.frontier_observer import observed_attribute
from tests.phase0_hygiene import product_env_keys
from tests.harness import engine_context, pin_char_counter

compaction: Any = importlib.import_module("hermes_lcm.compaction")
escalation: Any = importlib.import_module("hermes_lcm.escalation")
dag: Any = importlib.import_module("hermes_lcm.dag")
tokens: Any = importlib.import_module("hermes_lcm.tokens")
leaf_prep: Any = importlib.import_module("hermes_lcm.leaf_prep")
harness: Any = importlib.import_module("tests.harness")
GOLDEN = Path(__file__).parents[1] / "fixtures" / "compress_trace.json"
Trace = list[str]
PAD = " alpha beta gamma delta epsilon" * 40


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    mode: str = "healthy"
    shape: str = "turns"
    turns: int = 8
    sweep: bool = False
    force: bool = True
    current_tokens: int | None = 3000
    overrides: tuple[tuple[str, Any], ...] = ()
    request: str = ""
    seeded_depths: tuple[int, ...] = ()
    retry: bool = False
    focus: str | None = None
    setup: str = ""
    expiry_checkpoint: int = 0
    window: int = 200000


SCENARIOS = (
    Scenario("resume-commit-wide-tail", shape="no-system", setup="resume-wide-tail", mode="open-circuit",
             overrides=(("fresh_tail_count", 100),)),
    Scenario("persisted-volatile-no-pattern", shape="volatile", setup="stored-volatile"),
    Scenario("ignored-unknown-role", shape="ignored-unknown", setup="stored", overrides=(
        ("ignore_message_patterns", ["IGNORE THIS ROW"]),)),
    Scenario("dependent-nonreply-barrier", setup="nonreply-ledger"),
    Scenario("ignored-tail-unknown-role", shape="tail-unknown", setup="stored", overrides=(
        ("fresh_tail_count", 3), ("ignore_message_patterns", ["IGNORE THIS ROW"]))),
    Scenario("ignored-tail-nondict-barrier", shape="tail-nondict", overrides=(
        ("fresh_tail_count", 3), ("ignore_message_patterns", ["IGNORE THIS ROW"]))),
    Scenario("ignored-tail-unanchored-eof", shape="unanchored-eof", setup="stored", overrides=(
        ("fresh_tail_count", 2), ("ignore_message_patterns", ["IGNORE THIS ROW"]))),
    Scenario("empty", shape="empty"),
    Scenario("fresh-tail-protected", turns=1, force=False),
    Scenario("below-leaf-minimum", overrides=(("leaf_chunk_tokens", 20000),)),
    Scenario("leaf-pass"),
    Scenario("leaf-no-host-token-count", current_tokens=None),
    Scenario("leaf-automatic-no-sweep", force=False),
    Scenario("dynamic-leaf-passes", overrides=(("dynamic_leaf_chunk_enabled", True),
                                               ("dynamic_leaf_chunk_max", 800))),
    Scenario("dynamic-below-minimum", overrides=(("dynamic_leaf_chunk_enabled", True),
                                                ("leaf_chunk_tokens", 20000))),
    Scenario("condensation-levels", seeded_depths=(0, 0, 1),
             overrides=(("condensation_fanin", 2),)),
    Scenario("sweep-drain-condense", sweep=True, force=False),
    Scenario("sweep-pre-leaf-condensation", sweep=True, force=False, seeded_depths=(0, 0, 0),
             overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("escalation-level-two", mode="reject-first"),
    Scenario("escalation-level-three", mode="outage", overrides=(("survival_fit", False),)),
    Scenario("escalation-rejected-keeps-backlog", mode="outage"),
    Scenario("context-length-rescue", mode="context-refusal"),
    Scenario("budget-exhausted-before-leaf", mode="timeout", sweep=True, force=False),
    Scenario("budget-partial-pass", mode="slow", sweep=True, force=False),
    Scenario("provider-aborted", mode="cancel-first"),
    Scenario("resume-after-interruption", mode="cancel-second", sweep=True, retry=True),
    Scenario("resume-unadopted-commit", retry=True),
    Scenario("tool-pair-groups", shape="tools"),
    Scenario("objective-anchor", shape="objective", overrides=(("fresh_tail_count", 8),)),
    Scenario("identity-anchor", shape="identity"),
    Scenario("externalized-payloads", shape="tools", overrides=(
        ("large_output_externalization_enabled", True),
        ("large_output_externalization_threshold_chars", 100),
        ("large_output_active_replay_stubbing_enabled", True),
        ("large_output_active_replay_stub_threshold_tokens", 100))),
    Scenario("lifecycle-publication-conflict", mode="conflict"),
    Scenario("child-session-bypass", shape="child", force=False),
    Scenario("child-session-forced", shape="child"),
    *(Scenario(f"{kind}-bypass-{label}", setup="bypass-no-aux", force=False,
               current_tokens=count, request="all", retry=True,
               overrides=((f"{pattern}_session_patterns", ["trace-bypass"]),
                          ("context_threshold", 0.9)))
      for kind, pattern in (("ignored", "ignore"), ("stateless", "stateless"))
      for label, count in (("missing", None), ("zero", 0), ("negative", -1))),
    Scenario("boundary-cleanup-only", request="cleanup_only_due_to_boundary_cooldown"),
    Scenario("below-threshold-cleanup-only", force=False, current_tokens=1,
             request="below_threshold_cleanup_only"),
    Scenario("hold-fit-only", force=False, request="hold_fit_only_requested"),
    Scenario("ignored-message-dependents", shape="ignored", overrides=(
        ("ignore_message_patterns", ["IGNORE THIS ROW"]),)),
    Scenario("forced-overflow", current_tokens=250000,
             overrides=(("max_assembly_tokens", 2000),)),
    Scenario("hidden-owned-backlog", shape="hidden"),
    Scenario("route-unavailable", mode="open-circuit"),
    Scenario("route-unavailable-full-tail", mode="open-circuit", turns=1),
    Scenario("route-unavailable-adopt-commit", mode="open-after-first", retry=True),
    Scenario("sweep-tail-drained", turns=1, sweep=True, force=False),
    Scenario("fresh-tail-yield", force=False, overrides=(("fresh_tail_count", 100),
             ("fresh_tail_pressure_yield_min_observations", 1))),
    Scenario("dynamic-fresh-tail-yield", force=False, overrides=(("fresh_tail_count", 100),
             ("fresh_tail_pressure_yield_min_observations", 1), ("dynamic_leaf_chunk_enabled", True))),
    Scenario("sweep-fresh-tail-yield", force=False, sweep=True, overrides=(("fresh_tail_count", 100),
             ("fresh_tail_pressure_yield_min_observations", 1))),
    Scenario("dynamic-threshold-reached", current_tokens=250, overrides=(("dynamic_leaf_chunk_enabled", True),)),
    Scenario("dynamic-raw-drained", turns=3, overrides=(("dynamic_leaf_chunk_enabled", True),)),
    Scenario("sweep-budget-entry", mode="spent", sweep=True, force=False),
    Scenario("sweep-pass-budget", turns=40, sweep=True, overrides=(("leaf_chunk_tokens", 1),)),
    Scenario("pre-leaf-condensation-rejected", mode="outage", sweep=True,
             seeded_depths=(0, 0, 0), overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("post-leaf-condensation-rejected", mode="outage-after-leaves", sweep=True,
             overrides=(("summary_prefix_target_tokens", 20),)),
    Scenario("post-leaf-publication-conflict", mode="conflict-second", sweep=True),
    Scenario("condensation-publication-locked", mode="lock-condensation", seeded_depths=(0, 0, 1),
             overrides=(("condensation_fanin", 2),)),
    Scenario("pre-leaf-publication-locked", mode="lock-condensation", sweep=True,
             seeded_depths=(0, 0, 0), overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("deferred-maintenance", setup="debt", overrides=(("deferred_maintenance_enabled", True),)),
    Scenario("deferred-critical-small-backlog", setup="debt", current_tokens=190000, overrides=(
        ("deferred_maintenance_enabled", True), ("critical_budget_pressure_ratio", 0.5),
        ("leaf_chunk_tokens", 20000), ("dynamic_leaf_chunk_enabled", True))),
    Scenario("identity-anchor-disabled", setup="no-identity"),
    Scenario("explicit-focus", focus="Preserve source coverage."),
    Scenario("no-system-prompt", shape="no-system"),
    Scenario("retained-tail-occurrence", shape="retained", overrides=(("fresh_tail_count", 1),
             ("leaf_chunk_tokens", 1)), retry=True),
    Scenario("replayed-summary-scaffold", setup="replay", retry=True),
    Scenario("stub-first-exit", shape="tools", force=False, current_tokens=12001, overrides=(
        ("context_threshold", 0.06), ("large_output_externalization_enabled", True),
        ("large_output_active_replay_stubbing_enabled", True),
        ("large_output_externalization_threshold_chars", 100),
        ("large_output_active_replay_stub_threshold_tokens", 100))),
    Scenario("legacy-filtered-backlog", shape="legacy-filter", setup="stored", overrides=(
        ("ignore_message_patterns", ["IGNORE THIS ROW"]),)),
    Scenario("legacy-filtered-tail-dependents", shape="legacy-tail", setup="stored", overrides=(
        ("fresh_tail_count", 3), ("ignore_message_patterns", ["IGNORE THIS ROW"]))),
    Scenario("legacy-filtered-eof-dependents", shape="legacy-eof", setup="stored", overrides=(
        ("ignore_message_patterns", ["IGNORE THIS ROW"]),)),
    Scenario("externalized-transcript-gc", shape="tools", overrides=(
        ("dynamic_leaf_chunk_max", 20000), ("large_output_externalization_enabled", True),
        ("large_output_externalization_threshold_chars", 100), ("large_output_transcript_gc_enabled", True))),
    Scenario("transcript-gc-locked", shape="tools", mode="lock-gc", overrides=(
        ("dynamic_leaf_chunk_max", 20000), ("large_output_externalization_enabled", True),
        ("large_output_externalization_threshold_chars", 100), ("large_output_transcript_gc_enabled", True))),
    Scenario("default-sweep-target", sweep=True, overrides=(("summary_prefix_target_tokens", 0),)),
    Scenario("child-without-usage", shape="child", current_tokens=None),
    Scenario("child-with-cached-usage", shape="child", current_tokens=None, setup="child-usage"),
    Scenario("generated-dependent-replies", setup="dependents"),
    Scenario("dependent-only-leaf", shape="dependent-only", setup="dependents", overrides=(("leaf_chunk_tokens", 1),)),
    Scenario("ignored-dependent-tail-eof", shape="dependent-eof", setup="stored", overrides=(
        ("fresh_tail_count", 2), ("ignore_message_patterns", ["IGNORE THIS ROW"]))),
    Scenario("dynamic-below-minimum-no-fit", force=False, overrides=(
        ("dynamic_leaf_chunk_enabled", True), ("leaf_chunk_tokens", 20000), ("survival_fit", False))),
    Scenario("single-below-minimum-no-fit", force=False, overrides=(
        ("leaf_chunk_tokens", 20000), ("survival_fit", False))),
    *[Scenario(f"deadline-checkpoint-{point}", sweep=True, force=False, expiry_checkpoint=point)
      for point in (4, 6, 8, 10)],
    Scenario("deadline-hidden-backlog-read", shape="hidden", sweep=True, force=False, expiry_checkpoint=6),
    Scenario("unknown-window-default-target", window=0, sweep=True,
             overrides=(("summary_prefix_target_tokens", 0),)),
    Scenario("pre-leaf-frontier-read-error", mode="frontier-read-error", sweep=True, force=False,
             seeded_depths=(0, 0, 0), overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("unthresholded-default-target", overrides=(("context_threshold", 0.0),
             ("summary_prefix_target_tokens", 0))),
    Scenario("sweep-stub-first-exit", shape="tools", sweep=True, force=False, current_tokens=12001, overrides=(
        ("context_threshold", 0.06), ("large_output_externalization_enabled", True),
        ("large_output_active_replay_stubbing_enabled", True),
        ("large_output_externalization_threshold_chars", 100),
        ("large_output_active_replay_stub_threshold_tokens", 100))),
    Scenario("condensation-spends-soft-budget", mode="very-slow", sweep=True, force=False,
             seeded_depths=(0, 0, 0), overrides=(("summary_prefix_target_tokens", 100),)),
    Scenario("hung-primary-leaves-fallback", mode="timeout", sweep=True, force=False,
             overrides=(("summary_fallback_models", ["trace-fallback"]),)),
    Scenario("dynamic-forced-overflow", current_tokens=250000, overrides=(
        ("max_assembly_tokens", 2000), ("dynamic_leaf_chunk_enabled", True))),
    Scenario("dynamic-small-backlog-yields", force=False, overrides=(
        ("dynamic_leaf_chunk_enabled", True), ("leaf_chunk_tokens", 20000),
        ("survival_fit", False), ("fresh_tail_pressure_yield_min_observations", 1))),
    Scenario("single-small-backlog-yields", force=False, overrides=(
        ("leaf_chunk_tokens", 20000), ("survival_fit", False), ("fresh_tail_pressure_yield_min_observations", 1))),
    Scenario("objective-cap-too-small", shape="objective", overrides=(
        ("fresh_tail_count", 8), ("max_assembly_tokens", 10))),
    Scenario("objective-without-cap", shape="objective", overrides=(
        ("fresh_tail_count", 8), ("max_assembly_tokens", 0))),
    Scenario("optional-extraction-host-unavailable", overrides=(
        ("extraction_enabled", True), ("dynamic_leaf_chunk_enabled", True),
        ("assertion_extraction_enabled", True))),
    Scenario("unattributed-rotated-history", setup="rotated", retry=True,
             overrides=(("fresh_tail_pressure_yield_enabled", False),)),
    Scenario("prepared-leaf-promotion", setup="prepared", sweep=True, force=False,
             current_tokens=150000, overrides=(("background_leaf_prep_enabled", True),
             ("background_leaf_prep_max_calls", 64), ("context_threshold", 0.75),
             ("max_assembly_tokens", 200000),
             ("large_output_externalization_path", ""))),
    Scenario("prepared-leaf-route-rejected", setup="prepared-route-changed", force=False,
             current_tokens=150000, overrides=(("background_leaf_prep_enabled", True),
             ("background_leaf_prep_max_calls", 64), ("context_threshold", 0.75),
             ("max_assembly_tokens", 200000),
             ("large_output_externalization_path", ""))),
)


# Frozen before phase extraction: use code qualnames, never a bare function
# name that could also belong to an unrelated service or a nested callback.
CLOCK_CALLERS = frozenset(
    ("hermes_lcm.compaction", "CompactionMixin." + name)
    for name in (
        "_compress_impl",
        "_begin_compression",
        "_ingest_compression_view",
        "_compress_impl.<locals>.sweep_step_done",
        "_prepare_compression_sweep",
        "_condense_before_leaves",
        "_admit_leaf_and_adopt_replay",
        "_filter_leaf_backlog",
        "_select_leaf_candidate",
        "_prepare_leaf_input",
        "_summarize_selected_leaf",
        "_build_leaf_publication",
        "_publish_selected_leaf",
        "_advance_after_leaf",
        "_settle_leaf_stop",
        "_finish_without_leaf",
        "_condense_after_leaves",
        "_finish_with_leaf",
        "sweep_step_done",
        "warn_rejected",
    )
)


@dataclasses.dataclass
class Clock:
    now: float = 1000.0
    nonce: int = 0
    expiry_checkpoint: int = 0
    checkpoints: int = 0

    def monotonic(self) -> float:
        if self.expiry_checkpoint:
            frame = inspect.currentframe()
            assert frame is not None and frame.f_back is not None
            caller = frame.f_back
            if (caller.f_globals.get("__name__"), caller.f_code.co_qualname) in CLOCK_CALLERS:
                self.checkpoints += 1
                if self.checkpoints == self.expiry_checkpoint:
                    self.now += 121.0
            del frame
        return self.now

    def wall(self) -> float:
        return 1700000000.0 + self.now

    def time_ns(self) -> int:
        self.nonce += 1
        return 1700000000000000000 + self.nonce


class HostInterrupted(BaseException):
    """The host's cancellation is outside the provider's Exception recovery."""


class LiteralPattern:
    """The configured pattern interface without the optional regex dependency."""

    pattern = "IGNORE THIS ROW"

    def search(self, text: str, timeout: float | None = None) -> bool:
        return self.pattern in text


# Deliberately frozen: extracting compaction phases must not change observation.
# The original 123 names include callable fingerprints. Four existing helper
# state fields complete the original transitive inventory. The two hung-route
# fields added on the R3 train and the leaf preparation/refusal state are also
# read/written by compaction. Update deliberately
# when changing this list, never derive the runtime observer from source.
OBSERVED_ATTRIBUTES = (
    "_active_replay_placeholder_digest",
    "_assemble_committed_compaction_context",
    "_assemble_context",
    "_assemble_overflow_recovery_context",
    "_build_preserved_objective_summary_part",
    "_bypasses_lcm_context_management",
    "_committed_replay_drops",
    "_compaction_request",
    "_compress_dropped_ignored_rows",
    "_compress_forced_overflow",
    "_compress_hidden_rows_consumed",
    "_compress_host_rows_consumed",
    "_compress_lcm_bypassed_session",
    "_compress_occurrences",
    "_compress_run",
    "_config",
    "_context_aware_leaf_cap",
    "_conversation_id",
    "_critical_budget_pressure_reached",
    "_current_auxiliary_prompt_tokens",
    "_dag",
    "_derive_auto_focus_topic",
    "_drop_preexisting_generated_ignored_dependent_eof_replies",
    "_effective_assembly_token_cap",
    "_extract_expand_hint",
    "_fail_open_after_publication_failure",
    "_finalize_forced_overflow_result",
    "_fit_can_rescue",
    "_foreground_budget",
    "_fresh_tail_start",
    "_frontier",
    "_generated_context_carrier_remainder",
    "_generated_placeholder_digest_budget_for_active_replay",
    "_generated_placeholder_digest_ordinals_for_active_replay",
    "_get_store_id_map_for_messages",
    "_get_store_ids_for_messages",
    "_hidden_backlog_label",
    "_hidden_backlog_status",
    "_hung_summary_route_has_fallback",
    "_hung_summary_strike",
    "_identity_anchor_extend_chunk",
    "_identity_anchor_summary_input",
    "_ingest_cursor",
    "_ingest_cursor_needs_reconcile",
    "_ingest_messages",
    "_invalidate_rollups_for_published_node",
    "_is_generated_ignored_dependent_reply",
    "_is_ignored_active_replay_placeholder",
    "_is_replayed_context_scaffold_message",
    "_is_volatile_ignored_quarantine_placeholder",
    "_kept_reply_tokens",
    "_last_compacted_store_id",
    "_last_compaction_duration_ms",
    "_last_compression_noop_reason",
    "_last_compression_status",
    "_last_condensation_suppressed_reason",
    "_last_hidden_backlog",
    "_last_leaf_content_filter_error",
    "_last_leaf_level_3_verbatim",
    "_last_leaf_summary_model",
    "_last_overflow_recovery_failed",
    "_last_stub_first_exit",
    "_last_threshold_full_sweep",
    "_latest_user_context_anchor",
    "_leading_anchor_count",
    "_leaf_rescue_admit",
    "_lifecycle",
    "_load_compression_carry_ranges",
    "_load_generated_ignored_dependent_reply_records",
    "_load_generated_ignored_placeholder_hashes",
    "_mapped_stored_row_matches_ignore_message_patterns",
    "_matches_ignore_message_patterns",
    "_maybe_condense",
    "_maybe_engage_fresh_tail_pressure_yield",
    "_maybe_gc_compacted_tool_results",
    "_maybe_reclassify_late_auxiliary_before_compaction_write",
    "_new_foreground_budget",
    "_no_progress_candidate",
    "_no_progress_hold",
    "_no_progress_hold_blocks",
    "_note_fresh_tail_pressure_relieved",
    "_objective_only_noop",
    "_overflow_recovery_assembly_cap",
    "_pending_context_anchor_messages",
    "_pending_emission_candidates",
    "_prepare_retained_user_anchor",
    "_pressure_yield_invocation_verdict",
    "_pressure_yield_tail_token_limit",
    "_primary_summary_route",
    "_rebind_after_unadopted_compaction_commit",
    "_record_successful_compaction_telemetry",
    "_refresh_raw_backlog_debt",
    "_remember_generated_ignored_dependent_reply",
    "_replay_occurrences",
    "_run_pre_compaction_extraction",
    "_run_threshold_sweep_condensation",
    "_sanitize_active_context_messages",
    "_schedule_pre_compaction_assertions",
    "_select_oldest_leaf_chunk",
    "_serialize_messages_with_clip",
    "_session_id",
    "_should_force_overflow_recovery",
    "_should_run_deferred_maintenance",
    "_start_sweep_budget_hold",
    "_store",
    "_store_complete_backlog",
    "_store_complete_blocked",
    "_store_complete_copy_exclusion_proofs",
    "_store_complete_cut",
    "_store_complete_excluded",
    "_stored_publication_filter_exclusions",
    "_stub_first_exit",
    "_stub_first_exit_now",
    "_summarize_leaf_chunk_with_rescue",
    "_summary_frontier_tokens",
    "_summary_route_available",
    "_summary_route_seconds_left",
    "_summary_route_stop_applies",
    "_survival_fit_reason",
    "_survival_host_overhead",
    "_survival_pass_observed",
    "_sweep_budget_hold_applies",
    "_sweep_budget_hold_turn_paced",
    "_sweep_budget_hold_until",
    "_take_leaf_summary_model",
    "_thread_context_session_id",
    "_unattributed_leaf_warned",
    "_working_leaf_chunk_tokens",
    "_write_generated_ignored_placeholder_hash_counts",
    "_write_generated_ignored_placeholder_hash_ordinals",
    "compression_count",
    "context_length",
    "threshold_tokens",
)


class Canonical:
    """Preserve semantic values; name only generated IDs and process addresses."""

    def __init__(self, root: Path) -> None:
        self.root = str(root)
        self.ids: dict[str, str] = {}
        self.excluded: dict[str, str] = {}

    def text(self, value: str) -> str:
        value = value.replace(self.root, "<sandbox>")

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            return self.ids.setdefault(raw, f"<generated-{len(self.ids) + 1}>")

        return re.sub(r"(?<![a-zA-Z0-9])[0-9a-f]{32}(?![a-zA-Z0-9])|"
                      r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", replace, value)

    def value(self, value: Any, path: str = "") -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (str, Path)):
            return self.text(str(value))
        if isinstance(value, bytes):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
        if callable(value):
            return {"callable": value.__qualname__}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: self.value(getattr(value, f.name), f"{path}.{f.name}")
                    for f in dataclasses.fields(value)}
        if isinstance(value, (escalation.ForegroundBudget, escalation.ForegroundEstimates)):
            return self.value(vars(value), path)
        if isinstance(value, dict):
            # These keys are Python object identities, not durable store IDs.
            identity_keys = path.endswith(("store_ids_by_message_id", "_compress_occurrences"))
            return {(f"<message-{i}>" if identity_keys else str(k)): self.value(v, f"{path}.{k}")
                    for i, (k, v) in enumerate(value.items(), 1)}
        if isinstance(value, (list, tuple)):
            return [self.value(v, path) for v in value]
        if isinstance(value, (set, frozenset)):
            return sorted((self.value(v, path) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
        self.excluded[path] = type(value).__name__ + ": no stable value (connection/lock/service)"
        return {"excluded_type": type(value).__name__}


TRACE_GUIDANCE = (
    "Regenerate from repo root: python -m tests.behaviour.compress_trace_harness --regenerate\n"
    "Attribute every changed event in the PR description to its commit or behaviour change.\n"
    "See docs/process/turn-path-probe-packet.md#golden-traces."
)


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(text: str) -> str:
    return f"{hashlib.sha256(text.encode()).hexdigest()[:12]}/{len(text)}"


def compact(value: Any) -> str:
    text = encoded(value)
    return text if len(text) <= 80 else f"sha12/len={fingerprint(text)}"


def sql_event(connection: str, sql: str, shapes: dict[tuple[str, str], int], events: Trace) -> str:
    if sql in {"BEGIN IMMEDIATE", "BEGIN", "COMMIT", "ROLLBACK"}:
        return "sql " + sql
    # SQLite-internal FTS callbacks are filtered before normalization.
    # LCM callbacks contain expanded values, which must not be embedded in the
    # golden. Keep every column and expression in the shape.
    shape = re.sub(
        r"'(?:''|[^'])*'|(?<![\w])\d+(?:\.\d+)?(?![\w])", "?", sql)
    key = (connection, shape)
    if key not in shapes:
        shapes[key] = len(shapes) + 1
        events.append(f"sql shape={shapes[key]} conn={connection} {shape}")
    reference = f"sql #{shapes[key]}"
    return reference if shape == sql else f"{reference} sha12/len={fingerprint(sql)}"


def append_sql(events: Trace, event: str) -> None:
    if events:
        previous = events[-1]
        base, separator, count = previous.rpartition(" x")
        if previous == event:
            events[-1] = event + " x2"
            return
        if separator and count.isdigit() and base == event:
            events[-1] = event + f" x{int(count) + 1}"
            return
    events.append(event)


def transcript(case: Scenario) -> list[dict[str, Any]]:
    if case.shape == "empty":
        return []
    rows: list[dict[str, Any]] = [{"role": "system", "content": "Preserve the user's work."}]
    for i in range(case.turns):
        rows.extend([{"role": "user", "content": f"Turn {i}: inspect the source.{PAD}", "timestamp": 10.0 * (i + 1)},
                     {"role": "assistant", "content": f"Turn {i}: recorded evidence.{PAD}"}])
    if case.shape in {"tools", "objective"}:
        if case.shape == "objective":
            rows = [{"role": "user", "content": "Check the named files and preserve every source."}]
        rows.extend(harness.sequential_pairs(4, 1600)[1:])
    if case.shape == "identity":
        rows[1]["content"] = "My preferred name is Reference. Preserve this identity. " + PAD
    if case.shape == "ignored":
        rows[3]["content"] = "IGNORE THIS ROW " + PAD
    if case.shape.startswith("legacy-"):
        index = 3 if case.shape == "legacy-filter" else len(rows) - (5 if case.shape == "legacy-tail" else 3)
        rows[index]["content"] = "IGNORE THIS ROW " + PAD
    if case.shape in {"dependent-only", "dependent-eof"}:
        rows = [{"role": "system", "content": "Preserve the user's work."},
                {"role": "user", "content": "IGNORE THIS ROW " + PAD},
                {"role": "assistant", "content": "Dependent reply. " + PAD},
                {"role": "assistant", "content": "Dependent continuation. " + PAD}]
        if case.shape == "dependent-only":
            rows.extend([{"role": "user", "content": "A new request."},
                         {"role": "assistant", "content": "A fresh reply."}])
    if case.shape in {"no-system", "retained"}:
        rows = rows[1:]
    if case.setup == "resume-wide-tail":
        for row in rows:
            row.pop("timestamp", None)
    if case.shape == "volatile":
        rows[2]["content"] = (
            "[LCM active replay placeholder: assistant output quarantined; "
            "kind=quarantined_assistant_output; reason=fixture; "
            "scope=ignored_message_pattern; field=content; chars=8; bytes=8; "
            "sha256=0123456789abcdef]")
    if case.shape == "ignored-unknown":
        rows[3] = {"role": "unknown", "content": "IGNORE THIS ROW " + PAD}
    if case.shape in {"tail-unknown", "tail-nondict"}:
        rows[-4]["content"] = "IGNORE THIS ROW " + PAD
        rows[-2] = {"role": "unknown", "content": "Unclassified tail row."}
        if case.shape == "tail-nondict":
            rows[-2] = None
    if case.shape == "unanchored-eof":
        rows = [{"role": "user", "content": "IGNORE THIS ROW " + PAD},
                {"role": "assistant", "content": "Dependent reply. " + PAD},
                {"role": "assistant", "content": "Dependent continuation. " + PAD}]
    return rows


class Provider:
    def __init__(self, case: Scenario, clock: Clock, events: Trace, canonical: Canonical, engine: Any) -> None:
        self.case, self.clock, self.events, self.canonical, self.engine = case, clock, events, canonical, engine
        self.calls = 0

    def __call__(self, prompt: str, max_tokens: int, model: str = "", timeout: float | None = None,
                 reasoning_effort: str = "") -> str | None:
        self.calls += 1
        # Prompt headings distinguish L1/L2 and condensation depth; no internal spy.
        level = 2 if "bullet" in prompt.lower().split("\n\n", 1)[0] else 1
        self.events.append(f"call {self.calls} model={model} level={level} budget={max_tokens} "
                           f"timeout={timeout} in={fingerprint(self.canonical.text(prompt))}")
        mode = self.case.mode
        if mode == "cancel-first" or (mode == "cancel-second" and self.calls == 2):
            raise HostInterrupted()
        if mode == "timeout":
            self.clock.now += float(timeout or 120)
            raise TimeoutError("provider timed out")
        if mode in {"slow", "very-slow"}:
            self.clock.now += 29.0 if mode == "slow" else 61.0
        if (mode == "conflict" and self.calls == 1) or (mode == "conflict-second" and self.calls == 2):
            # A competing owner changes a selected row's ownership before
            # publication. Real lifecycle coverage validation must reject it.
            self.engine._store._conn.execute(
                "UPDATE messages SET conversation_id = 'other-conversation' WHERE store_id = ?",
                (2 if self.calls == 1 else 6,))
            self.engine._store._conn.commit()
        if mode == "lock-condensation" and ((self.case.sweep and self.calls == 1)
                                            or (not self.case.sweep and self.calls == 2)):
            self.engine._dag._conn.execute("""CREATE TEMP TRIGGER lock_summary BEFORE INSERT ON summary_nodes
                BEGIN SELECT RAISE(ABORT, 'database is locked'); END""")
        if mode == "lock-gc" and self.calls == 1:
            self.engine._store._conn.execute("""CREATE TEMP TRIGGER lock_gc BEFORE UPDATE OF content ON messages
                BEGIN SELECT RAISE(ABORT, 'database is locked'); END""")
        if mode == "context-refusal" and self.calls <= 2:
            raise RuntimeError("maximum context length exceeded")
        if mode == "outage" or (mode == "outage-after-leaves" and self.calls >= 8):
            return None
        if mode == "reject-first" and self.calls == 1:
            return "not shorter " * 4000
        return f"Earlier turns, call {self.calls}.\nExpand for details about: turns"


def install_clock(patch: pytest.MonkeyPatch, clock: Clock) -> None:
    replacement = SimpleNamespace(**{name: getattr(time, name) for name in dir(time) if not name.startswith("__")})
    replacement.monotonic, replacement.perf_counter, replacement.time = clock.monotonic, clock.monotonic, clock.wall
    replacement.time_ns = clock.time_ns
    replacement.gmtime = lambda: time.gmtime(1700000000)
    for name in ("compaction", "engine", "escalation", "store", "dag", "lifecycle_state",
                 "survival_fit", "externalize"):
        module = importlib.import_module(f"hermes_lcm.{name}")
        if hasattr(module, "time"):
            patch.setattr(module, "time", replacement)


def connections(engine: Any) -> list[tuple[str, sqlite3.Connection]]:
    found: list[tuple[str, sqlite3.Connection]] = []
    seen: set[int] = set()
    for name, service in sorted(vars(engine).items()):
        conn = getattr(service, "_conn", None)
        if isinstance(conn, sqlite3.Connection) and id(conn) not in seen:
            found.append((name, conn))
            seen.add(id(conn))
    return found


@contextmanager
def capture_logs(events: Trace, canonical: Canonical) -> Iterator[None]:
    """Observe plugin records in order, without formatting run-dependent args."""
    class TraceHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            events.append(f"log {record.name} {record.levelname} " + encoded(canonical.text(str(record.msg))))

    handler = TraceHandler()
    loggers = [logging.getLogger(name) for name in sorted(logging.Logger.manager.loggerDict)
               if name == "hermes_lcm" or name.startswith("hermes_lcm.")]
    previous = [(logger, logger.level, logger.propagate) for logger in loggers]
    try:
        for logger, _, _ in previous:
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            logger.propagate = False  # one event even when a plugin child has a parent logger
        yield
    finally:
        for logger, level, propagate in previous:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate
        handler.close()


def prepared_state(engine: Any, canonical: Canonical) -> Any:
    """Read the ready cache without ready_chain()'s stale-chain eviction.

    The worker's private SQLite connection is read-only and closed before
    compression. There is no SQL ready-chain table: promotion/rejection writes
    this in-memory cache. Snapshot it directly under its own lock instead.
    """
    prep = leaf_prep._LEAF_PREP
    key = (str(engine._store.db_path), engine._conversation_id)
    with prep._condition:
        return canonical.value({"chain": list(prep._cache.get(key, [])),
                                "counters": dict(prep.counters)})


def state(engine: Any, canonical: Canonical) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in OBSERVED_ATTRIBUTES:
        value = observed_attribute(engine, name)
        if name == "_store":
            value = engine._store.get_session_messages(engine._session_id, limit=10000)
        elif name == "_dag":
            value = engine._dag.get_session_nodes(engine._session_id, limit=10000)
        elif name == "_lifecycle":
            value = engine._lifecycle.get_by_conversation(engine._conversation_id)
        result[name] = canonical.value(value, name)
    return result


def observe(case: Scenario, root: Path) -> Trace:
    root.mkdir(parents=True, exist_ok=True)
    events: Trace = []
    clock, canonical = Clock(expiry_checkpoint=case.expiry_checkpoint), Canonical(root)
    with pytest.MonkeyPatch.context() as patch:
        # Shared harness helpers are deliberately untyped; keep the boundary Any.
        pin: Any = pin_char_counter
        pin(patch)
        install_clock(patch, clock)
        patch.setattr(leaf_prep, "_LEAF_PREP", leaf_prep._LeafPrep())
        patch.setenv("HERMES_HOME", str(root / "home"))
        if case.setup == "no-identity":
            patch.setenv("LCM_IDENTITY_ANCHOR", "false")
        settings = dict(fresh_tail_count=2, fresh_tail_max_tokens=0, leaf_chunk_tokens=800,
                        dynamic_leaf_chunk_max=1600, context_threshold=0.001,
                        threshold_full_sweep_enabled=case.sweep, max_assembly_tokens=100000,
                        summary_model="trace-model", l3_truncate_tokens=2,
                        temporal_rollups_enabled=False, embeddings_enabled=False,
                        large_output_externalization_enabled=False,
                        large_output_active_replay_stubbing_enabled=False,
                        large_output_externalization_path=str(root / "home" / "payloads"),
                        summary_prefix_target_tokens=1000)
        settings.update(dict(case.overrides))
        context: Any = engine_context
        with context(root, window=case.window, conversation_id="trace-conversation", **settings) as engine:
            view = transcript(case)
            if settings.get("ignore_message_patterns"):
                engine._compiled_ignore_message_patterns = [LiteralPattern()]
            if case.setup == "resume-wide-tail":
                engine.ingest(view)
                engine._dag.add_node(dag.SummaryNode(
                    session_id="S", depth=0, summary="Committed prefix. Expand for details about: turns",
                    token_count=20, source_token_count=1200, source_ids=[1, 2, 3, 4],
                    source_type="messages", created_at=100.0))
                engine._lifecycle.advance_frontier("trace-conversation", "S", 4)
                engine._last_compacted_store_id = 4
            if case.setup == "bypass-no-aux":
                engine.ingest(view)
                engine.on_session_start("trace-bypass", context_length=200000, platform="cli")
                assert engine._bypasses_lcm_context_management()
                assert engine._thread_context_session_id() == ""
            if case.setup in {"stored", "stored-volatile"}:
                engine._store.append_batch("S", view, conversation_id="trace-conversation")
                engine._ingest_cursor = len(view)
            if case.setup == "stored-volatile":
                engine._store.write_metadata_json(
                    engine._ignored_placeholder_metadata_keys(), json.dumps(["0123456789abcdef"]))
                assert not engine._compiled_ignore_message_patterns
            if case.setup == "nonreply-ledger":
                engine.ingest(view)
                engine._compress_run.store_ids_by_message_id = engine._get_store_id_map_for_messages(view)
                user = view[1]
                store_id = engine._compress_run.store_ids_by_message_id[id(user)]
                digest = hashlib.sha256(f"S\0{store_id}".encode()).hexdigest()[:16]
                engine._store.write_metadata_json(engine._ignored_dependent_reply_metadata_keys(),
                                                  json.dumps([{"store": digest}]))
                assert engine._ignored_dependent_reply_store_fingerprint(user) is None
                assert engine._ignored_dependent_reply_content_fingerprint(user, user["content"]) is None
                assert not engine._is_generated_ignored_dependent_reply(user, user["content"])
                events.append("non-reply role: both fingerprints absent despite persisted ledger; generated=False")
            if case.shape == "child":
                engine._mark_thread_context_stateless("trace-child")
                if case.setup == "child-usage":
                    engine.update_from_response({"prompt_tokens": 1000})
            if case.setup == "dependents":
                engine.ingest(view)
                engine._compress_run.store_ids_by_message_id = engine._get_store_id_map_for_messages(view)
                for row in view:
                    if row.get("role") == "assistant":
                        engine._remember_generated_ignored_dependent_reply(row, row["content"])
            for i, depth in enumerate(case.seeded_depths):
                engine._dag.add_node(dag.SummaryNode(
                    session_id="S", depth=depth, summary=f"Seed {i}. " + PAD,
                    token_count=500, source_token_count=1000, source_ids=[],
                    source_type="messages" if depth == 0 else "summaries", created_at=100.0 + i))
            if case.shape == "hidden":
                engine.ingest(view)
                view = view[-2:]
            if case.shape == "retained":
                engine.ingest(view)
                view = [*view[2:], dict(view[0]), dict(view[1])]
                engine._ingest_cursor = len(view)
            if case.setup == "debt":
                engine._lifecycle.record_debt("trace-conversation", kind="raw_backlog", size_estimate=4000)
            if case.mode == "open-circuit":
                for _ in range(3):
                    engine._summary_circuit_breaker.record_failure("trace-model")
            request_fields = ("cleanup_only_due_to_boundary_cooldown", "below_threshold_cleanup_only",
                              "automatic_request", "hold_fit_only_requested")
            if case.request == "all":
                for name in request_fields:
                    setattr(engine._compaction_request, name, True)
            elif case.request:
                setattr(engine._compaction_request, case.request, True)
            provider = Provider(case, clock, events, canonical, engine)
            patch.setattr(escalation, "_call_llm_for_summary", provider)
            if case.setup == "rotated":
                engine.ingest(view)
                assert engine.rotate_active_session(apply=True)["ok"]
            if case.setup.startswith("prepared"):
                engine.ingest(view)
                engine.last_prompt_tokens = case.current_tokens
                engine._prepare_leaves_after_turn(view)
                assert leaf_prep._LEAF_PREP.wait_idle(10), "preparation worker did not finish"
                assert leaf_prep._LEAF_PREP.status()["cached_leaves"] > 0
                # Setup calls are outside the trace. Foreground provider numbering starts at one.
                events.clear()
                provider.calls = 0
                if case.setup == "prepared-route-changed":
                    engine._config.summary_model = "changed-trace-model"
            initial = state(engine, canonical)
            conns = connections(engine)
            shapes: dict[tuple[str, str], int] = {}
            if case.setup == "bypass-no-aux":
                parent_rows = harness.stored_rows(engine, "S")
                parent_binding = (engine._session_id, engine._conversation_id,
                                  engine._ingest_cursor, engine._last_compacted_store_id)

            frontier_read_denied = False

            def deny_frontier_read(action: int, table: str | None, column: str | None,
                                   database: str | None, trigger: str | None) -> int:
                nonlocal frontier_read_denied
                if action == sqlite3.SQLITE_READ and table == "summary_nodes" and not frontier_read_denied:
                    frontier_read_denied = True
                    events.append("db deny post-publication frontier read")
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            def trace_sql(connection: str, sql: str) -> None:
                if (case.mode == "frontier-read-error" and connection == "_dag"
                        and sql.strip() == "COMMIT" and provider.calls == 1 and not frontier_read_denied):
                    # Publication errors are converted to condensation_error by
                    # the sweep. Its post-publication frontier read propagates.
                    engine._dag._conn.set_authorizer(deny_frontier_read)
                # SQLite-internal FTS5 statements vary by SQLite build.
                # Filter raw SQL before normalization can erase quoted schema names.
                if sql.lstrip().startswith("--") or re.search(r"(['\"])(main|temp)\1\s*\.", sql):
                    return
                normalized = canonical.text(sql.strip())
                if re.match(r"(?:--\s*)?(?:INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE)\b",
                            normalized, re.IGNORECASE):
                    # Normalize whitespace outside literals only: literal whitespace is data.
                    pieces = re.split(r"('(?:''|[^'])*')", normalized)
                    normalized = "".join(piece if i % 2 else " ".join(piece.split())
                                         for i, piece in enumerate(pieces))
                    append_sql(events, sql_event(connection, normalized, shapes, events))

            for name, conn in conns:
                conn.set_trace_callback(partial(trace_sql, name))
            with capture_logs(events, canonical):
                try:
                    for attempt in range(2 if case.retry else 1):
                        events.append(f"compress attempt={attempt + 1}")
                        if case.setup == "bypass-no-aux":
                            before_storage = [list(conn.iterdump()) for _, conn in conns]
                        events.append("leaf_prep before " + compact(prepared_state(engine, canonical)))
                        returned = view
                        # The public wrapper owns these scopes. Invoke the target
                        # directly so its return/state precede survival-fit changes.
                        engine._foreground_budget = engine._new_foreground_budget()
                        if case.mode == "spent":
                            clock.now += 121
                        if case.mode == "open-after-first" and attempt:
                            for _ in range(3):
                                engine._summary_circuit_breaker.record_failure("trace-model")
                        if case.setup == "resume-wide-tail":
                            view = [dict(row) for row in view]
                            engine._config.fresh_tail_count = 100
                            before_calls = provider.calls
                            before_rows = harness.stored_rows(engine, "S")
                            before_nodes = engine._dag.get_session_nodes("S", limit=10000)
                        propagated_frontier_error = False
                        try:
                            with engine._fresh_tail_pressure_yield_invocation():
                                returned = engine._compress_impl(view, current_tokens=case.current_tokens,
                                                                 force=case.force, focus_topic=case.focus)
                            events.append("return " + compact(canonical.value(returned)))
                        except BaseException as exc:
                            if not isinstance(exc, (Exception, HostInterrupted)):
                                raise
                            events.append(f"raise {type(exc).__name__} message={compact(canonical.text(str(exc)))}")
                            if case.shape == "tail-nondict":
                                assert isinstance(exc, AttributeError)
                                frames = []
                                traceback = exc.__traceback__
                                while traceback is not None:
                                    frames.append(traceback.tb_frame.f_code.co_name)
                                    traceback = traceback.tb_next
                                assert "_begin_compression" in frames
                                assert "count_message_tokens" in frames
                                events.append("non-dict tail: token counting rejects input before ingestion and leaf phases")
                            if case.mode == "frontier-read-error":
                                assert isinstance(exc, sqlite3.DatabaseError)
                                assert not compaction._is_sqlite_locked_error(exc)
                                propagated_frontier_error = True
                        finally:
                            if case.mode == "frontier-read-error":
                                engine._dag._conn.set_authorizer(None)
                        if case.setup == "resume-wide-tail":
                            assert provider.calls == before_calls
                            assert harness.stored_rows(engine, "S") == before_rows
                            assert engine._dag.get_session_nodes("S", limit=10000) == before_nodes
                            assert len(returned) < len(view)
                            assert all(any(row["content"] in str(out.get("content", ""))
                                           and row["role"] == out.get("role") for out in returned)
                                       for row in view[int(engine._last_compacted_store_id):])
                            events.append("resumed prefix: no new leaf; durable rows unchanged; remaining tail survives")
                        if case.mode == "frontier-read-error":
                            assert frontier_read_denied and propagated_frontier_error
                            assert engine._foreground_budget.leaf_reserve == 0.0
                            assert engine._foreground_budget.progress == "condensation"
                            assert provider.calls == 1
                            events.append("pre-leaf error propagated; leaf_reserve=0.0; committed condensation retained")
                        if case.name == "unknown-window-default-target":
                            assert engine.threshold_tokens == 0
                            assert not engine._foreground_budget.sweep_active
                            assert engine._last_threshold_full_sweep["status"] == "never_run"
                            events.append("unknown window: threshold=0; sweep inactive; target has no gated consumer")
                        if case.setup == "bypass-no-aux":
                            assert not any(getattr(engine._compaction_request, name) for name in request_fields)
                            assert before_storage == [list(conn.iterdump()) for _, conn in conns]
                            assert parent_rows == harness.stored_rows(engine, "S")
                            assert parent_binding == (engine._session_id, engine._conversation_id,
                                                      engine._ingest_cursor, engine._last_compacted_store_id)
                            events.append("bypass no-aux: request consumed; parent storage and binding unchanged")
                        events.append("leaf_prep after " + compact(prepared_state(engine, canonical)))
                        # This host contract exposes _assemble_context rather than a public assemble().
                        leading = engine._leading_anchor_count(returned)
                        try:
                            assembled = engine._assemble_context(returned[0] if leading else None, returned[leading:],
                                **({"retained_user_message": returned[1]} if leading == 2 else {}))
                            events.append("assemble")
                            events.extend(f"message {row.get('role')} "
                                          f"{fingerprint(canonical.text(str(row.get('content', ''))))}"
                                          for row in assembled)
                        except Exception as exc:
                            events.append(f"assemble raise {type(exc).__name__}")
                        if case.setup == "replay":
                            view = returned
                    final = state(engine, canonical)
                    for name, value in final.items():
                        if value != initial[name]:
                            events.append(f"attr {name}={compact(value)}")
                    events.append(f"attrs count={len(final)} sha256={hashlib.sha256(encoded(final).encode()).hexdigest()}")
                    for name, reason in sorted(canonical.excluded.items()):
                        events.append(f"excluded {name}={reason}")
                finally:
                    for _, conn in conns:
                        conn.set_trace_callback(None)
                    if case.shape == "child":
                        engine._clear_thread_context_stateless("trace-child")
    return events


def first_difference(expected: Trace, actual: Trace) -> str | None:
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else "<end of trace>"
        right = actual[index] if index < len(actual) else "<end of trace>"
        if left != right:
            return f"[{index}]: expected {left!r}, got {right!r}"
    return None


def main() -> None:
    environment = dict(os.environ)
    try:
        for key in product_env_keys(Path(__file__).resolve().parents[2]):
            os.environ.pop(key, None)
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--regenerate", action="store_true")
        args = parser.parse_args()
        if not args.regenerate:
            parser.error("pass --regenerate to intentionally replace the golden fixture")
        with tempfile.TemporaryDirectory(prefix="compress-trace-") as directory:
            observed = {case.name: observe(case, Path(directory) / case.name) for case in SCENARIOS}
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(observed, sort_keys=True, indent=0) + "\n")
        print(f"Wrote {len(observed)} scenarios to {GOLDEN.name}")
    finally:
        os.environ.clear()
        os.environ.update(environment)


if __name__ == "__main__":
    main()
