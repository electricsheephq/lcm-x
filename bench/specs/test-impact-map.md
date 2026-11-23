# LCM-X test-impact map

This is LCM-X's map under rule 3 of the ElectricSheep release and escalation standard (v1.1): tests follow the diff.
Each row below names a **mechanism**: the code paths behind one behaviour, and the suites that judge that
behaviour. A release, or a scheduled run, runs only the suites whose inputs changed. It records every suite it
skipped, with the reason.

`bench/specs/test-impact-map.json` is the machine-readable copy. The tables below are the same data;
`tests/test_suite_impact.py` fails when the doc and the JSON disagree.

## How to use it

```
python3 scripts/release/suite_impact.py <base> <head> [--input NAME ...]
```

The script reads `git diff --no-renames --name-only <base>..<head>` and prints one line per suite:

- `RUN <suite>: <reason>`: a mechanism, own path or other input of the suite changed. The reason names it.
- `SKIP <suite>: suite <name> skipped: nothing it measures changed`. Copy these lines into the release notes.
  For `release-yml` and `plugin-payload` a SKIP means only that their inputs did not change: GitHub still runs
  both at every `v*` tag. Write them as "ran at the tag; inputs unchanged".
- `MANUAL <suite>: ...`: the suite is not decided by the map. It is never printed as skipped.
- `UNMAPPED <path>: ...`: no row covers a changed path. The script fails closed and every mapped suite runs.

Pass `--input NAME` once for each non-path input that changed (see "Other inputs"). For example, pass
`--input hermes-host` when the host build under test moved. Exit codes: 0 normally, 2 on a bad map or an unknown
input name, 1 on a usage or git error.

## Rules

1. **Run only the suites whose mechanisms changed.** A suite runs when a changed path matches one of its
   mechanisms' paths, one of its own paths, or one of its path sets, or when one of its other inputs changed.
   Every other mapped suite is skipped. The release notes carry each skip as
   `suite X skipped: nothing it measures changed`, printed by the script.
   **Precedence until the release spec adopts this map.** The release readiness spec still decides the release
   rows. A row it requires on every release candidate (for example the L1 checks) runs even when the script prints
   SKIP for it. That SKIP line then records that the row's inputs did not change; it does not permit leaving the
   row out. The release manager amends the readiness spec separately.
2. **Fail closed.** A changed path that no row covers is UNMAPPED, and every mapped suite runs. A change to a
   path in `fail_closed_paths` (a shared hub, or a product module that no mechanism or suite row lists yet) also
   runs every mapped suite. Renames are read as a delete plus an add (`--no-renames`). The old path hits its row, and a new path
   that no glob covers is UNMAPPED until the map grows a row for it.
3. **Manual suites are not decided here.** Their product reach is not mapped (the unmapped list below). Run them
   by hand when someone judges they are needed. Never record them as "skipped: nothing it measures changed".
4. **Scheduled runs are map-gated too.** For each suite, a nightly or weekly run diffs the last commit on which
   that suite actually ran against the current head. A suite skipped on earlier heads, by the map or by a skip
   claim (rule 8), therefore runs at the first scheduled run after its inputs changed. It passes `--input hermes-host` when the Hermes host version under test changed, and
   `--input provider-route` when a provider route or model it calls changed. A scheduled suite with nothing
   changed is skipped and logged the same way. A scheduled-run regression gets its own fix and never gates an
   unrelated hotfix.
5. **The expensive boards** (BEAM, LongMemEval, LoCoMo and the other shared-reader boards, and AML) run
   when a recall or summary mechanism changes (M06, M07, M12, M13); before any claim of a board gain
   (`--input board-claim`); once a month for drift (`--input monthly-drift`); and when their own harness or a
   provider route they call changes (`bench/aml/**`, `--input provider-route`). Boards are targets, not release
   gates.
6. **Pull-request CI and tag workflows keep their own triggers.** Branch protection runs the required CI checks
   on every pull request, and GitHub runs `release.yml` and `plugin-payload.yml` at every `v*` tag (not at
   `claude-code-v*` tags), whatever this map says. The map decides the release-time, scheduled and by-hand runs. It records
   whether a CI suite's inputs changed, so a release note can say so.
7. **Catch log.** Each suite keeps a catch log: the last time its result changed a release decision (date, pull
   request or tag, and what changed). A heavy suite (any suite that is not a pull-request CI check) with no catch in a quarter is proposed for
   retirement, and the
   owner decides. The column starts empty: no catch is recorded before this map.
8. **Interim skip claim, until the symbol tier lands.** The script matches files, so it can mark a suite RUN
   when the changed functions are outside that suite's mechanisms. The release manager may skip such a suite only
   by writing, in the release note, the changed functions and why none of them is in, or called by, that suite's
   mechanism. The second reviewer checks that claim against the diff. A claim is allowed only when every RUN
   reason the script printed for that suite is a mechanism match through a shared file. It is never allowed when
   the suite's own path or another input triggered it, when an UNMAPPED or fail-closed path changed, for a MANUAL
   suite, or for a careful-lane mechanism test that the change's own lane requires. The backstop is the next
   scheduled run of that suite (rule 4 diffs from the last commit it actually ran on). A regression found there
   gets its own fix and never gates an unrelated hotfix.
9. **Change the map with the code.** A pull request that adds a module, moves a behaviour or adds a suite updates
   the JSON and these tables in the same change. The agreement test fails if they drift. The script fails
   closed on paths the map has not learned yet.

**What the map does not prove.** The rows come from what each suite's bars judge, plus the files each suite or
workflow reads. Call reach has not been traced. So the map does not prove that every suite's inputs are fully
listed. File-level matching can only over-run: `engine.py` and `compaction.py` sit in almost every mechanism,
so a change to them runs nearly every suite. The `Key symbols` column is for a reader and for rule 8's skip
claims; the script does not read it yet. A symbol tier is planned as a separate change: it maps changed hunks to
their enclosing functions and closes each mechanism's symbols over the in-repo call graph.

## Other inputs

Non-path inputs that a suite reads. The script cannot see them in a diff, so the caller passes them.

<!-- INPUTS TABLE -->
| Input | Changes when |
|---|---|
| `hermes-host` | The Hermes host version or host build under test changes (a host pin, the supported floor, or upstream main). |
| `provider-route` | A summary, reader, judge or embedding provider route or model that the suite calls changes. |
| `baseline` | The comparison arm changes: the previous GA tree, a control tree, or a stored baseline snapshot. |
| `instrument-pin` | The pinned instrument commit that the release rows run from moves. |
| `board-claim` | A release or post is about to claim a board gain. |
| `monthly-drift` | The monthly drift run for the boards. |

## Path sets and lists

`@name` in a suite's paths expands to the path set. A glob reached through a path set runs its suite but does not
map a path: a new root module is matched by `@product` and still prints as UNMAPPED. Globs: `**/` matches zero
or more whole directories, any other `**` matches anything including `/`, and `*` and `?` never match `/`.

- `fail_closed_paths` are shared hubs and product modules that no mechanism lists yet. A hub is a module
  imported by 10 or more root modules but listed in fewer than half of the mechanisms (`tokens.py`, `store.py`,
  `db_bootstrap.py`, `sqlite_util.py`, `sqlite_session.py`, `reconcile.py`, `externalize.py`,
  `ingest_protection.py`, `embedding_provider.py`, `escalation.py`), plus the content helpers
  (`message_content.py`, `message_analysis.py`, `sanitize.py`, `session_patterns.py`). Recount the importers when a
  module's reach grows. The map files and this script are fail-closed too: a change to the selector is itself
  unverified, so everything runs. A change to any of them runs every
  mapped suite. Each one should move into a mechanism row when someone maps it.
- `ignored_paths` feed no suite through a literal glob. The review sweep and CI shards read every change,
  and the host suite and nightly latency read everything tests may read (`@tests_read`).

<!-- LISTS TABLE -->
| List | Globs |
|---|---|
| `@any` | `**` |
| `@python` | `**/*.py` |
| `@pytest` | `tests/**`, `pytest.ini` |
| `@tests_read` | `*`, `.agents/**`, `.github/*`, `.github/actions/**`, `.github/workflows/**`, `.github/ISSUE_TEMPLATE/**`, `access_context/**`, `access_policy/**`, `adapters/**`, `bench/**`, `benchmarking/**`, `benchmarks/**`, `companions/**`, `docs/**`, `release/**`, `scripts/**`, `skills/**`, `teams/**`, `tests/**` |
| `@product` | `*.py`, `access_context/**`, `access_policy/**`, `adapters/**`, `companions/**`, `skills/**`, `teams/**`, `plugin.yaml`, `requirements-semantic.txt` |
| `@turn_path` | `__init__.py`, `assembly_plan.py`, `aux_session.py`, `bypass.py`, `codex_routing.py`, `compaction.py`, `compaction_policy.py`, `compaction_request.py`, `compress_run.py`, `config.py`, `dag.py`, `db_bootstrap.py`, `diagnostics.py`, `emission_projection.py`, `engine.py`, `engine_registry.py`, `escalation.py`, `expand_tools.py`, `externalize.py`, `extraction.py`, `frontier_state.py`, `fresh_tail.py`, `host_rewrite.py`, `host_row_recovery.py`, `host_uid.py`, `host_uid_emit.py`, `identity_anchor.py`, `ingest_content.py`, `ingest_integrity.py`, `ingest_ledger.py`, `ingest_pem_model.py`, `ingest_pem_scan.py`, `ingest_persisted.py`, `ingest_policy.py`, `ingest_protection.py`, `leaf_prep.py`, `lifecycle_state.py`, `maintenance_scheduler.py`, `message_analysis.py`, `message_content.py`, `message_patterns.py`, `model_routing.py`, `placeholder_ledger.py`, `plugin_identity.py`, `prefix_matching.py`, `recognition_matching.py`, `reconcile.py`, `reset_state.py`, `retrieval_core.py`, `rollup_builder.py`, `rollup_periods.py`, `rollup_store.py`, `sender_capture.py`, `sanitize.py`, `schemas.py`, `search_query.py`, `session_patterns.py`, `sqlite_util.py`, `store.py`, `store_complete.py`, `summary_attempt.py`, `summary_input_clip.py`, `summary_outcome.py`, `survival_codec.py`, `survival_fit.py`, `tokens.py`, `tool_pairs.py`, `tool_registry.py`, `tools.py`, `unended_catchup.py` |
| `fail_closed_paths` | `message_content.py`, `message_analysis.py`, `sanitize.py`, `session_patterns.py`, `tokens.py`, `db_bootstrap.py`, `store.py`, `ingest_protection.py`, `externalize.py`, `sqlite_session.py`, `sqlite_util.py`, `reconcile.py`, `embedding_provider.py`, `escalation.py`, `answer_contract.py`, `assertion_*.py`, `cloud_*.py`, `codex_routing.py`, `command_auth.py`, `command_router.py`, `command_status.py`, `cpu_capacity.py`, `engine_registry.py`, `entity_*.py`, `evidence_compiler.py`, `evidence_pack.py`, `expansion_evidence.py`, `inactive_record.py`, `ingest_classifier.py`, `maintenance.py`, `mcp_reader.py`, `model_routing.py`, `occurrence_time.py`, `presets.py`, `query_view_store.py`, `readonly_store.py`, `reasoning.py`, `recorder.py`, `recorder_read.py`, `requirements_compiler.py`, `scope_storage.py`, `selective_compiler.py`, `sender_capture.py`, `trajectory_store.py`, `access_context/**`, `access_policy/**`, `teams/**`, `skills/**`, `bench/specs/test-impact-map.json`, `bench/specs/test-impact-map.md`, `scripts/release/suite_impact.py` |
| `ignored_paths` | `.agents/**`, `.github/ISSUE_TEMPLATE/**`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/CODEOWNERS`, `.github/dependabot.yml`, `.gitignore` |

## Mechanisms

<!-- MECHANISMS TABLE -->
| Mechanism | Name | Behaviour | Paths | Key symbols | Measured by |
|---|---|---|---|---|---|
| `M01` | Host-row recognition | Recognises host rows by replay identity, host UID or identity anchor on bind, cold start and replay, so each handed row is stored once: no duplicates, no wrong-session rows, cursor in the right place. | `reconcile.py`, `host_uid.py`, `identity_anchor.py`, `engine.py`, `ingest_ledger.py`, `host_rewrite.py`, `host_uid_emit.py`, `prefix_matching.py`, `placeholder_ledger.py`, `config.py`, `ingest_persisted.py`, `emission_projection.py`, `recognition_matching.py`, `message_patterns.py` | `ReconcileMixin`, `HostUidShadowMixin`, `IdentityAnchorMixin`, `bind_session_state`, `_on_session_start_unlocked`, `_ingest_messages_transaction`, `_build_ingest_replay`, `_ingest_filter_rows`, `_ledger_adopt`, `_write_ingest_ledger`, `_ledger_cold_cursor`, `plan_ordinals`, `cold_cursor_candidate`, `prove`, `host_rewrite_match`, `PrefixMatchingMixin`, `PlaceholderLedgerMixin`, `host_message_uid_mode`, `ingest_ledger_mode`, `identity_anchor_enabled` | `c1a`, `respin`, `c1b`, `probes`, `probe-matrix`, `phase-c`, `d5b`, `gate-copy`, `gate-walks`, `gate-timing`, `gate-reversibility`, `gate-crossing`, `gate-live`, `ci-lint`, `nightly-matrix`, `nightly-compare`, `real-host`, `long-soak`, `invariants` |
| `M02` | Session lifecycle and recovery | Session start, end, reset, rotation and rollover keep lineage and store every tail; cold or off-current tails are retried; an interrupted turn's rows are recovered from an abandoned compaction child. | `engine.py`, `host_row_recovery.py`, `unended_catchup.py`, `reset_state.py`, `lifecycle_state.py`, `store.py`, `maintenance_scheduler.py` | `on_session_end`, `_on_session_end_unlocked`, `queue_off_current_busy_tail`, `_retry_off_current_fallback`, `_finalize_foreground_session_end_unlocked`, `on_session_reset`, `_record_rotation_predecessor`, `rollover_session`, `rotate_active_session`, `_queue_cold_session_tail`, `_retry_cold_tail`, `_retry_pending_session_tails`, `_shutdown_unlocked`, `HostRowRecoveryMixin`, `UnendedCatchupMixin`, `ResetStateMixin`, `bind_session`, `finalize_session`, `record_rollover`, `record_session_lineage` | `c1a`, `respin`, `c1b`, `probe-matrix`, `d5b`, `ci-session-cost`, `nightly-matrix`, `nightly-compare` |
| `M03` | Compaction trigger and cadence | Decides when a compaction pass runs: thresholds, preflight, leaf-candidate status, and the cooldown, sweep-budget and no-progress holds that block automatic passes. | `compaction.py`, `compaction_policy.py`, `engine.py`, `tokens.py`, `compaction_request.py`, `config.py` | `should_compress`, `should_compress_preflight`, `_leaf_compaction_candidate_status`, `_context_aware_leaf_cap`, `resolve_threshold`, `threshold_tokens`, `_effective_threshold_tokens`, `_compression_boundary_cooldown_active`, `_sweep_budget_hold_active`, `_no_progress_hold_active`, `_automatic_compression_blocked`, `count_tokens`, `count_messages_tokens`, `CompactionRequest` | `m3`, `d7`, `gate-walks`, `track-s` |
| `M04` | Compaction progress and frontier | A pass selects a leaf, publishes it and advances the frontier, and reports progress to the host; a refused or rejected pass rolls back cleanly; a no-progress hold arms only when it should. | `compaction.py`, `frontier_state.py`, `lifecycle_state.py`, `engine.py`, `compress_run.py`, `fresh_tail.py` | `compress`, `_compress_owned`, `_compress_impl`, `_select_leaf_candidate`, `_publish_selected_leaf`, `_advance_after_leaf`, `_finish_without_leaf`, `_finish_with_leaf`, `_made_progress`, `_reports_progress`, `stage_frontier`, `FrontierState`, `advance_frontier`, `_start_no_progress_hold`, `record_rejected_compaction`, `_restore_refused_compaction`, `_persist_frontier_marker`, `CompressRun` | `c1a`, `respin`, `c1b`, `m3`, `phase-c`, `d7`, `gate-copy`, `gate-walks`, `gate-timing`, `gate-reversibility`, `gate-crossing`, `gate-live`, `ci-lint`, `nightly-matrix`, `nightly-compare`, `long-soak`, `track-s` |
| `M05` | Nothing lost across compaction | Every row leaving the returned list is covered by a byte-equal or fuller stored copy under a node in the head; hidden backlog and host-pruned output are accounted for; the commit proof matches the stored rows. | `store_complete.py`, `compaction.py`, `identity_anchor.py`, `leaf_prep.py`, `engine.py`, `fresh_tail.py`, `head_copy.py` | `StoreCompleteMixin`, `_record_compress_commit_proof`, `_rekey_durable_commit_proof`, `_leaf_consumption`, `_committed_replay_drops`, `_match_refused_assistant_replay`, `_filter_leaf_backlog`, `_identity_anchor_audit`, `excluded_from_summary_input`, `coverage_digest`, `_prepare_compress_commit_proof` | `c1a`, `respin`, `probe-matrix`, `phase-c`, `d5b`, `d7`, `gate-copy`, `gate-walks`, `gate-live`, `nightly-matrix`, `nightly-compare`, `invariants`, `track-s` |
| `M06` | Summary generation route | Produces leaf summaries through the configured summary route, with rescue chunks, contract checks, circuit breaker, spend guard and background leaf preparation. | `engine.py`, `escalation.py`, `summary_attempt.py`, `summary_outcome.py`, `leaf_prep.py`, `compaction.py`, `summary_input_clip.py`, `extraction.py`, `config.py`, `fresh_tail.py`, `head_copy.py` | `summarize_leaf_chunk_with_rescue`, `_summary_route_available`, `_prepare_leaves_after_turn`, `SummaryCircuitBreaker`, `SummarySpendGuard`, `_check_summary_contract`, `_call_llm_for_summary`, `SummaryAttempt`, `classify_summary_outcome`, `_LeafPrep`, `_summarize_selected_leaf` | `m3`, `d7`, `gate-live`, `track-s`, `boards`, `aml` |
| `M07` | Condensation, level-3 rollups and publication | Condenses summary nodes into higher depths, keeps level-3 nodes untruncated, builds temporal rollups, and refuses publications that conflict with lifecycle coverage. | `engine.py`, `compaction.py`, `dag.py`, `level3_repair.py`, `rollup_builder.py`, `rollup_store.py`, `lifecycle_state.py`, `head_copy.py` | `_maybe_condense`, `_condense_summary_nodes`, `_run_threshold_sweep_condensation`, `_condense_before_leaves`, `_condense_after_leaves`, `SummaryDAG`, `CondensationPublicationRejected`, `scan_level3_fragments`, `build_day`, `RollupStore`, `stage_compaction_publication` | `phase-c`, `gate-walks`, `gate-crossing`, `gate-live`, `boards`, `aml` |
| `M08` | Survival fit | Fits the assembled list under the host's window. Judged as zero survival-fit events and zero rows dropped by a fit. | `survival_fit.py`, `survival_codec.py`, `engine.py`, `assembly_plan.py`, `config.py` | `SurvivalFitMixin`, `_survival_fit`, `_plan_survival_fit`, `_survival_cut`, `_survival_projection`, `_apply_survival_fit`, `project_message`, `projection_mark_candidates`, `_resolve_survival_prefix_budget`, `default_prefix_budget` | `c1a`, `respin`, `m3`, `d7`, `gate-copy`, `gate-walks`, `gate-timing`, `gate-crossing`, `gate-live`, `nightly-matrix`, `nightly-compare`, `real-host`, `invariants`, `track-s` |
| `M09` | Returned-list assembly and tool pairs | Builds the returned list from the head, carrier and fresh tail with no split tool groups, no orphan results and no unneeded repair stubs, including overflow recovery and the no-leaf return. | `engine.py`, `tool_pairs.py`, `fresh_tail.py`, `assembly_plan.py`, `compaction.py`, `head_copy.py` | `_assemble_context`, `_select_tail_forced_recovery`, `_assemble_overflow_recovery_context`, `_sanitize_tool_pairs`, `_get_session_fresh_tail`, `repair_tool_pairs`, `resolve_fresh_tail_boundary`, `select_prefix`, `select_tail`, `_assemble_committed_compaction_context`, `_stub_first_exit` | `gate-copy`, `gate-walks`, `gate-live`, `ci-lint` |
| `M10` | Payload protection | Externalizes large tool output to payload files with placeholders, redacts sensitive text and quarantines suspicious assistant output on ingest. | `externalize.py`, `ingest_protection.py`, `engine.py`, `store.py`, `tools.py`, `ingest_policy.py`, `ingest_content.py`, `ingest_integrity.py`, `ingest_pem_model.py`, `ingest_pem_scan.py`, `config.py` | `_write_externalized_payload`, `_build_externalized_placeholder`, `protect_embedding_text`, `redact_sensitive_text`, `protect_message_for_ingest`, `quarantine_suspicious_assistant_message`, `_protect_ingest_rows`, `_maybe_stub_active_tool_result`, `gc_externalized_tool_result`, `_get_externalized_payload` | `c2`, `c4`, `probes`, `d1-identity`, `fmp` |
| `M11` | Auxiliary and delegated child sessions | Classifies auxiliary and subagent sessions and stores a delegated child's saved results once per handed row. | `aux_session.py`, `__init__.py`, `engine.py`, `compaction.py` | `AuxiliarySessionMixin`, `DelegateChildMixin`, `record_subagent_start`, `_on_live_delegate_child_list`, `_store_delegate_child`, `_maybe_reclassify_late_auxiliary_before_compaction_write` | `probes` |
| `M12` | Recall search and delivery | lcm_recall and lcm_grep find, fuse and deliver stored history under the response and deadline caps. | `tools.py`, `recall_delivery.py`, `retrieval_core.py`, `search_query.py`, `store.py`, `dag.py`, `engine.py`, `__init__.py`, `guidance.py`, `config.py`, `retrieval_context.py`, `recall_session.py`, `adaptive_retrieval.py`, `selective_recall.py`, `rollup_periods.py`, `ingest_policy.py`, `ingest_protection.py` | `lcm_recall`, `lcm_grep`, `_lcm_grep_full_text`, `deliver_recall`, `rrf_fuse`, `fuse_candidates`, `sanitize_fts5_query`, `build_recall_or_query`, `MessageStore.search`, `_build_proactive_recall_message`, `_pre_llm_context`, `get_recall_policy` | `c2`, `c4`, `m2`, `phase-c`, `floor-1586`, `d4-d8`, `d1-flips`, `boards`, `aml`, `fmp` |
| `M13` | Embedding provider and vectors | Embeds nodes and chunks through the supported providers, publishes vectors under a profile identity, runs incremental catch-up, and serves semantic or hybrid KNN. | `embedding_provider.py`, `vector_store.py`, `embedding_maintenance.py`, `retrieval_core.py`, `tools.py`, `engine.py`, `companions/**`, `chunking.py`, `schema_families.py`, `config.py`, `ingest_policy.py`, `command.py`, `ingest_protection.py` | `VoyageProvider`, `OllamaProvider`, `OpenAICompatibleProvider`, `FastembedProvider`, `VectorStore`, `run_incremental_embedding_pass`, `run_knn`, `run_chunk_knn`, `_lcm_grep_semantic`, `_lcm_grep_hybrid`, `_schedule_embedding_maintenance`, `_embedding_backfill_summary_run`, `_chunk_backfill_run` | `c2`, `c3`, `m2`, `phase-c`, `gate-live`, `floor-1586`, `d4-d8`, `d1-flips`, `boards`, `aml`, `fmp` |
| `M14` | Store schema, indexes and migrations | Creates, upgrades and verifies the store schema, its tables, columns and indexes, and refuses newer schemas. | `db_bootstrap.py`, `store.py`, `dag.py`, `vector_store.py`, `lifecycle_state.py`, `rollup_store.py`, `command.py`, `docs/store-contract.md` | `SCHEMA_VERSION`, `refuse_schema_version_too_new`, `classify_version_mismatch`, `set_schema_version`, `MessageStore._init_db`, `_ensure_identity_anchor_schema`, `_ensure_ingest_ledger_schema`, `_ensure_host_uid_schema`, `_ensure_embedding_schema` | `c1b`, `c3`, `probes`, `d5b`, `gate-timing`, `gate-reversibility`, `ci-lint` |
| `M15` | Write lock and busy database | Holds the SQLite write lock briefly, retries busy or snapshot errors and times lock waits; transaction scope inside the ingest and compaction commits. | `sqlite_util.py`, `db_bootstrap.py`, `store.py`, `sqlite_session.py`, `engine.py`, `config.py` | `_run_sqlite_write_with_snapshot_retry`, `_sqlite_savepoint`, `_TimedLock`, `configure_connection`, `SQLITE_BUSY_TIMEOUT_MS`, `ingest_atomic`, `_timed_atomic`, `atomic`, `ConnectionPolicy`, `TransactionOwner`, `_record_ingest_lock`, `_raise_ingest_busy`, `_exclusive_lifecycle`, `_ingest_messages_transaction`, `find_session_rows_by_content` | `probes`, `busy-bar`, `gate-live` |
| `M16` | Turn latency and pass time | Keeps foreground compaction inside its soft and hard budget, records compaction telemetry, and keeps per-turn ingest and hook cost low. New work on the ingest or compress path counts. | `escalation.py`, `engine.py`, `compaction.py`, `store.py`, `leaf_prep.py`, `tokens.py`, `config.py` | `ForegroundEstimates`, `ForegroundBudget`, `_foreground_call_budget`, `_record_turn_compaction_telemetry`, `_new_foreground_budget`, `increment_compaction_telemetry`, `start_encoder_load` | `m3`, `phase-c`, `d5b`, `d7`, `gate-walks`, `gate-timing`, `gate-crossing`, `gate-live`, `ci-session-cost`, `nightly-latency`, `track-s` |
| `M17` | Tool registry, schemas and handlers | Registers every lcm_* tool with its schema and dormancy rule, dispatches calls, and each handler meets its postconditions in local and cloud postures. | `tool_registry.py`, `schemas.py`, `engine.py`, `__init__.py`, `expand_tools.py`, `inspect_tools.py`, `recent.py`, `memory_tools.py`, `evidence_tools.py`, `expand_query_tools.py`, `tools.py`, `plugin.yaml` | `ToolSpec`, `TOOL_REGISTRY`, `recall_guidance_schemas`, `get_tool_schemas`, `handle_tool_call`, `register`, `_make_wrapped_handler`, `lcm_load_session`, `lcm_status` | `c2` |
| `M18` | Hermes host integration | Plugs into Hermes hosts from the supported floor through upstream main: hook registration, capability probes, the host-fallback compressor and Hermes config reads. | `plugin.yaml`, `__init__.py`, `compaction.py`, `engine.py`, `bypass.py`, `plugin_identity.py`, `config.py` | `requires_hermes`, `provides_hooks`, `_host_forwards_registered_tool_messages`, `register`, `_on_pre_llm_call`, `_on_post_llm_call`, `_bind_and_ingest`, `_host_has_post_tool_reanchor`, `BypassMixin`, `load_hermes_config`, `_hermes_compression_threshold_with_source` | `c1a`, `respin`, `phase-c`, `d5b`, `nightly-matrix`, `nightly-compare`, `real-host`, `long-soak` |
| `M19` | Install and doctor | The plugin and its local-embeddings companion install from the published payload, and lcm_doctor reports no failed checks and only the allowed warnings. | `command_doctor.py`, `tools.py`, `diagnostics.py`, `command.py`, `host_uid.py`, `store.py`, `__init__.py`, `plugin.yaml`, `requirements-semantic.txt`, `companions/**`, `runtime_identity.py` | `DoctorFacts`, `collect_doctor`, `survival_fit_observation`, `render_doctor`, `lcm_doctor`, `fresh_sqlite_check`, `host_uid_doctor_lines`, `records_doctor` | `c2`, `phase-c`, `d5b`, `plugin-payload` |
| `M20` | Release notes and docs | Release notes, changelog fragments and docs text. No runtime behaviour. | `.github/release-notes/**`, `changelog.d/**`, `**/*.md`, `docs/**` | — | — |

## Suites

`Kind` is `mapped` (decided by this map) or `manual` (not decided by it). `Runs` says where the suite sits today.

<!-- SUITES TABLE -->
| Suite | Name | Kind | Mechanisms | Own paths | Other inputs | Runs | Catch log |
|---|---|---|---|---|---|---|---|
| `c1a` | L1 C1a reliability matrix, retained-store pair. Exports the whole plugin tree, so any product file is an input. | mapped | `M01`, `M02`, `M04`, `M05`, `M08`, `M18` | `@product`, `bench/instruments/reliability/**`, `.github/workflows/reliability-nightly.yml`, `.github/actions/**` | `hermes-host`, `baseline`, `instrument-pin` | Release L1, after the tag | — |
| `respin` | Pre-tag respin reliability matrix (final head and previous GA). Same instrument and inputs as C1a. | mapped | `M01`, `M02`, `M04`, `M05`, `M08`, `M18` | `@product`, `bench/instruments/reliability/**`, `.github/workflows/reliability-nightly.yml`, `.github/actions/**` | `hermes-host`, `baseline`, `instrument-pin` | Release, before the tag | — |
| `c1b` | L1 C1b crash-and-restart cells. | mapped | `M01`, `M02`, `M04`, `M14` | — | `hermes-host`, `instrument-pin` | Release L1 | — |
| `c2` | L1 C2 tool matrix, local and cloud postures. | mapped | `M17`, `M12`, `M13`, `M19`, `M10` | — | `provider-route`, `hermes-host`, `instrument-pin` | Release L1 | — |
| `c3` | L1 C3 upgrade and rollback roundtrip. | mapped | `M14`, `M13` | — | `baseline`, `instrument-pin` | Release L1 | — |
| `c4` | L1 C4 LongMemEval D1, embeddings off. | mapped | `M12`, `M10` | `scripts/lcm_longmemeval.py`, `benchmarking/longmemeval.py`, `bench/tools/v1m-rebank/**`, `bench/tools/d8_guard.py`, `benchmarking/__init__.py`, `benchmarking/d1_cache.py`, `benchmarking/standalone.py` | `baseline`, `instrument-pin` | Release L1 | — |
| `c5` | L1 C5 review sweep of the release diff. Reviews the diff itself, docs and notes included, so any change is an input. | mapped | — | `@any` | `provider-route` | Release L1 | — |
| `probes` | L1 paired probes (stored rows against handed rows). | mapped | `M01`, `M10`, `M11`, `M14`, `M15` | `tests/test_issue_2482_interrupted_child.py`, `tests/test_issue_1931_delegate_children.py`, `tests/invariants/test_issue_2539_offcurrent_marker_end.py`, `tests/test_f3_spillover_redacted_preview.py`, `tests/test_issue_10_persisted_output_spillover.py` | `baseline`, `instrument-pin` | Release L1 | — |
| `m2` | L1 M2 LongMemEval-S mini with the reader. | mapped | `M12`, `M13` | `benchmarks/qa-harness/**`, `benchmarking/longmemeval.py`, `benchmarking/__init__.py`, `benchmarking/d1_cache.py`, `benchmarking/standalone.py` | `provider-route`, `baseline`, `instrument-pin` | Release L1 | — |
| `m3` | L1 M3 Track S, one seed. | mapped | `M03`, `M04`, `M06`, `M08`, `M16` | — | `provider-route`, `instrument-pin`, `baseline` | Release L1 | — |
| `probe-matrix` | L2 probe matrix (stored-row identity against the previous GA). | mapped | `M01`, `M02`, `M05` | `ingest_classifier.py` | `instrument-pin`, `baseline` | Release L2 | — |
| `host-suite` | L2 host-suite pair (whole tests/ on pinned host Pythons). | mapped | — | `@product`, `@tests_read`, `.github/release-notes/**`, `tests/**`, `.github/workflows/host-suite.yml`, `.github/actions/**`, `bench/instruments/reliability/ci.py`, `bench/instruments/reliability/hosts.ci.json` | `hermes-host` | Release L2 | — |
| `phase-c` | L2 soaks and install cells. | mapped | `M01`, `M04`, `M05`, `M07`, `M12`, `M13`, `M16`, `M18`, `M19` | `companions/**`, `requirements-semantic.txt`, `docs/embeddings-setup.md`, `plugin.yaml` | `hermes-host`, `provider-route`, `instrument-pin` | Release L2 | — |
| `d5b` | L2 older-host cells. | mapped | `M01`, `M02`, `M05`, `M14`, `M16`, `M18`, `M19` | — | `hermes-host`, `provider-route`, `instrument-pin` | Release L2 | — |
| `d7` | L2 Track S, three seeds. | mapped | `M03`, `M04`, `M05`, `M06`, `M08`, `M16` | — | `provider-route`, `instrument-pin`, `baseline` | Release L2 | — |
| `busy-bar` | L2 busy-bar (busy events and write-lock hold times). | mapped | `M15` | — | `instrument-pin`, `baseline` | Release L2 | — |
| `gate-copy` | Gate cells: copy checks. | mapped | `M01`, `M04`, `M05`, `M08`, `M09` | `bench/instruments/gate_kit/**` | `baseline`, `instrument-pin` | Release gate sheet | — |
| `gate-walks` | Gate cells: walks. | mapped | `M01`, `M03`, `M04`, `M05`, `M07`, `M08`, `M09`, `M16` | `bench/instruments/gate_kit/**` | `provider-route`, `baseline`, `instrument-pin` | Release gate sheet | — |
| `gate-timing` | Gate cells: timing (index build, cold ingest, cold starts). | mapped | `M14`, `M16`, `M01`, `M04`, `M08` | — | `baseline`, `instrument-pin` | Release gate sheet | — |
| `gate-reversibility` | Gate cells: reversibility and refusal replay. | mapped | `M14`, `M04`, `M01` | `bench/instruments/gate_kit/**` | `baseline`, `instrument-pin` | Release gate sheet | — |
| `gate-crossing` | Gate cells: crossing rehearsal. | mapped | `M01`, `M04`, `M07`, `M08`, `M16` | `bench/instruments/gate_kit/**` | `instrument-pin` | Release gate sheet | — |
| `gate-live` | Gate cells: live stand-in and field read. Measures the tagged install. | mapped | `M01`, `M04`, `M05`, `M06`, `M07`, `M08`, `M09`, `M13`, `M15`, `M16` | `@product` | `hermes-host`, `provider-route` | Release, after the tag | — |
| `ci-tests` | CI test shards (required). Tests read files across the repo, one release-notes file included, so any change is an input. Branch protection runs it on every pull request whatever the map says. | mapped | — | `@any`, `tests/**`, `scripts/**`, `.github/workflows/ci.yml`, `.github/actions/**`, `pytest.ini`, `.github/tests-ran-allowlist.txt` | — | Every pull request | — |
| `ci-lint` | CI lint, golden outcomes and store contract (required). | mapped | `M01`, `M04`, `M09`, `M14` | `@python`, `.github/workflows/**`, `.github/actionlint.yaml`, `ruff.toml`, `tests/fixtures/*_trace.json` | — | Every pull request | — |
| `ci-bench` | CI bench-tests (instrument self-tests). Instrument tests call real product handlers, so the whole product tree is an input. | mapped | — | `@product`, `bench/**`, `scripts/dev/ci_step.sh`, `scripts/dev/cloud_setup.sh`, `.github/workflows/ci.yml`, `@pytest` | — | Every pull request | — |
| `ci-session-cost` | CI session-end attribution cost (advisory). Its measurement program imports tests/conftest.py, which registers the product modules. | mapped | `M16`, `M02` | `@product`, `tests/fixtures/session_end_attribution_cost/**`, `tests/conftest.py`, `.github/workflows/ci.yml`, `@pytest` | — | Every pull request | — |
| `differential` | Differential against older GA trees. Same file filter as the workflow's own relevance step. | mapped | — | `@turn_path`, `tests/differential/**`, `.github/workflows/differential.yml`, `.github/actions/**` | — | Pull request and nightly | — |
| `nightly-matrix` | Reliability nightly matrix. | mapped | `M01`, `M02`, `M04`, `M05`, `M08`, `M18` | `@product`, `bench/instruments/reliability/**`, `.github/workflows/reliability-nightly.yml`, `.github/actions/**` | `hermes-host` | Nightly | — |
| `nightly-compare` | Reliability nightly compare, alarm and publish. Scores the matrix tables, so it has the matrix's inputs. | mapped | `M01`, `M02`, `M04`, `M05`, `M08`, `M18` | `@product`, `bench/instruments/reliability/**`, `.github/workflows/reliability-nightly.yml`, `.github/actions/**` | `hermes-host`, `baseline` | Nightly | — |
| `nightly-latency` | Reliability nightly latency tests. Any test file can carry the latency marker, and tests read files across the repo. | mapped | `M16` | `@tests_read`, `tests/**`, `pytest.ini`, `.github/actions/**` | — | Nightly | — |
| `real-host` | Real host regression files (advisory). Loads the whole plugin under a real Hermes. | mapped | `M18`, `M01`, `M08` | `@product`, `tests/test_real_*.py`, `tests/test_issue_524_gen2_replay.py`, `tests/test_issue_559_tool_group_boundary.py`, `tests/test_issue_631_not_a_pm_member.py`, `tests/test_issue_1306_cli_new.py`, `tests/invariants/test_issue_1438_cli_new.py`, `tests/test_session_end_attribution_host.py`, `tests/test_host_stub_parity.py`, `tests/test_catalog_scanner_on.py`, `tests/test_issue_1186_host_dependencies.py`, `tests/conftest.py`, `.github/workflows/real-host.yml`, `bench/instruments/reliability/**`, `@pytest` | `hermes-host` | Every pull request | — |
| `long-soak` | Opt-in long real-host soak. No workflow sets the switch; run it by hand when the map says RUN. | mapped | `M18`, `M01`, `M04` | `@product`, `tests/test_real_turn_loop_acp_override.py`, `@pytest` | `hermes-host` | By hand | — |
| `invariants` | Invariants nightly (property tests). Runs every property test under tests/invariants/, and those import product modules freely. | mapped | `M01`, `M05`, `M08` | `@product`, `tests/invariants/**`, `tests/conftest.py`, `.github/workflows/invariants-nightly.yml`, `.github/actions/**`, `@pytest` | — | Nightly | — |
| `floor-1586` | Local catch-up floor surrogate and covered-store stage. | mapped | `M13`, `M12` | `bench/instruments/local_catchup/**`, `.github/workflows/lcmx-1586-floor.yml`, `.github/workflows/lcmx-1586-covered-store.yml` | `provider-route` | By dispatch | — |
| `d1-identity` | D1 shard identity. | mapped | `M10` | `benchmarking/d1_cache.py`, `benchmarking/longmemeval.py`, `scripts/lcm_d1_local.py`, `bench/tools/d1-shard-check/**`, `.github/workflows/d1-shard-identity.yml`, `benchmarking/__init__.py`, `benchmarking/standalone.py` | — | Pull request, path filter | — |
| `cc-smoke` | Claude Code release smoke. The unit bundles the root core files. | mapped | — | `@product`, `claude_code_audit.py`, `claude_code_capture.py`, `claude_code_store.py`, `adapters/claude-code/**`, `scripts/build_claude_code_unit.py`, `scripts/claude_code_*`, `scripts/lcm_x_capture.py`, `scripts/lcm_x_mcp.py`, `.github/workflows/claude-code-release.yml` | — | claude-code tags only | — |
| `d4-d8` | D4/D8 judged QA rows. | mapped | `M12`, `M13` | `benchmarks/qa-harness/**`, `bench/tools/d4_health_rerun.py`, `bench/tools/d8_guard.py`, `scripts/lcm_longmemeval.py`, `benchmarking/longmemeval.py`, `benchmarking/__init__.py`, `benchmarking/d1_cache.py`, `benchmarking/standalone.py` | `provider-route`, `baseline`, `instrument-pin` | Release, first rc | — |
| `d1-flips` | D1 recall flips (full row). | mapped | `M12`, `M13` | `scripts/lcm_d1_local.py`, `benchmarking/d1_cache.py`, `benchmarking/longmemeval.py`, `scripts/lcm_longmemeval.py`, `bench/tools/v1m-rebank/**`, `benchmarking/__init__.py`, `benchmarking/standalone.py` | `baseline`, `provider-route`, `instrument-pin` | Release rc | — |
| `track-s` | D2/D3/D7 Track S rows. | mapped | `M03`, `M04`, `M05`, `M06`, `M08`, `M16` | — | `provider-route`, `instrument-pin`, `baseline` | Release rc | — |
| `boards` | Board rows (BEAM, LongMemEval, LoCoMo and the other shared-reader boards). Boards are targets, not release gates. | mapped | `M06`, `M07`, `M12`, `M13` | `benchmarks/qa-harness/**`, `benchmarking/longmemeval.py`, `benchmarking/__init__.py`, `benchmarking/d1_cache.py`, `benchmarking/standalone.py` | `provider-route`, `board-claim`, `monthly-drift` | On a recall or summary change, before a claimed gain, monthly | — |
| `aml` | AML board cycle. Boards are targets, not release gates. | mapped | `M06`, `M07`, `M12`, `M13` | `bench/aml/**` | `provider-route`, `board-claim`, `monthly-drift` | On a recall or summary change, before a claimed gain, monthly | — |
| `fmp` | FMP frontier-reader screen and dev loop. Not a gate. | mapped | `M10`, `M12`, `M13` | `bench/fmp/**`, `benchmarking/longmemeval.py`, `benchmarking/__init__.py`, `benchmarking/d1_cache.py`, `benchmarking/standalone.py` | `provider-route` | By hand | — |
| `other-instruments` | Other instruments (MCP recall, scale, occurrence fuzz, clone, probe drivers). Product reach is not mapped. | manual | — | `bench/instruments/cc_mcp_recall/**`, `bench/instruments/scale389/**`, `bench/instruments/occurrence_fuzz/**`, `bench/instruments/d5_clone/**`, `bench/instruments/compaction_probe/**` | — | By hand | — |
| `replay-stress` | Deterministic replay and stress harnesses. Calls compress, so it reaches compaction, but its product reach is not mapped. | manual | — | `benchmarking/**`, `benchmarks/**`, `scripts/lcm_benchmark.py`, `scripts/lcm_stress_check.py`, `scripts/lcm_steady_state_bench.py`, `scripts/benchmark_threshold_full_sweep.py` | — | By hand | — |
| `historical` | Historical rows (recorded results only). Retired: recorded results only. | manual | — | — | — | Never | — |
| `release-yml` | release.yml (needs the tag's release notes). | mapped | — | `.github/release-notes/**`, `.github/workflows/release.yml` | — | Every v* tag | — |
| `plugin-payload` | plugin-payload.yml (catalog payload publish). | mapped | `M19` | `@product`, `scripts/build_plugin_payload.py`, `scripts/payload/**`, `release/**`, `LICENSE`, `.github/workflows/plugin-payload.yml` | — | Every v* tag | — |
| `maintainability` | Maintainability report (advisory). | mapped | — | `@python`, `scripts/maintainability/**`, `.github/workflows/maintainability-report.yml` | — | Every pull request | — |

## Unmapped suites (manual)

- **Other instruments** (MCP recall, scale, occurrence fuzz, clone, and the compaction-probe drivers outside Track
  S). Product reach is not mapped.
- **Deterministic replay and stress**, including the h3 and h5 replay harnesses and the metered state-embedding
  backfill. It calls compress, so it reaches compaction, but which mechanisms it judges is not mapped.
- **Historical rows** (AMA-Bench, LME-V1 and V2): recorded results only. They never run.

The issue-intake workflow runs on issue events, not diffs, so it has no row.

## Known limits

- Rows name mechanisms by what each suite's bars judge. Which product paths an external runner or bridge
  reaches (the boards, AML, the reader rows) is unverified beyond the recall tool, so those rows list the recall
  and summary mechanisms only.
- The release rows run from a pinned instrument commit, so every release row takes `--input instrument-pin`. An
  edit under `bench/instruments/` on the candidate tree changes those rows only when the pin moves. The exceptions
  are C1a, the respin, the nightly rows and the gate-kit rows, which also read the tree's own harness.
- Rows that run test files (CI shards, CI bench-tests, host suite, invariants, latency, session cost, real host,
  long soak) take the whole product tree, because tests import product modules freely, and the shared test tree
  and pytest configuration (`@pytest`). Rows that list the shared LongMemEval helper also list its in-package
  imports. Board and reader rows take the in-repo QA
  harness that bridges their external runners. Every embedding-backed row takes `provider-route`.
- The differential row copies the workflow's own turn-path list. If the workflow's list changes, update
  `@turn_path` with it.
