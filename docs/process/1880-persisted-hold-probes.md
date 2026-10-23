# #1880 B persisted no-progress hold probe packet

Scope: automatic unchanged no-progress suppression across engine rebuilds using one existing metadata-table key per session. Refs #1880; #1162 is context only and remains deferred.

Baseline: `10407e12e33c1d7b09fe0e05141153555aec5d62`. Product candidate: `970e47f9b5ee33cd6cca2c71a27d777683eb412e`. Evidence files are under `docs/process/evidence/1880-persisted-hold/`.

Only ordinary completed `no_progress` persists. A positive completed-terminal classification excludes unfinished budget, route, rejection, summary failure and error stops; the separate sweep hold remains process-local. A record hit never rewrites metadata or extends expiry. Invalid, unreadable or future-dated metadata fails open with one generic metadata-only warning. A changed fingerprint is logically invalidated in the same call without a gate write. Successful publication clears only the applicable scoped record.

The digest is computed during existing successful ingest; admission reuses that proof and bounded indexed state. Cold threshold queries without replay proof fail open. Rebuilt preflight/direct compression ingest before the shared held decision; no repeated attribution or summary work is admitted on a matching proof. The resolved store identity is cached against exact store-object and db-path equality, while metadata and current indexed state are read on every gate. Leaf target ratio/minimum/maximum, L2 budget ratio and summary prompt version are included in the settings digest. Current model/provider/API-mode scalar route identity is included; opaque or URL-shaped routes fail open without reading credentials or base URLs.

Manual force, provider recovery, forced overflow, survival ceiling, replay-changing below-threshold cleanup, nonordinary no-progress reasons, and sweep holds retain their own policies. The part-A cleanup block is byte-identical to BASE.

## 1. Consumer enumeration

Raw inventories: base-consumers.txt is the complete narrow BASE hold producer/consumer search; candidate-consumers.txt (refreshed after settings/cost correction) includes call sites of every new helper plus ingest/reset consumers in production and tests. Root's host-consumers.txt pins NousResearch/hermes-agent at `d923a5eb487a17c0a68801bba9ef0f56c81040a8` through GitHub API; installed host/profile sources were not read.

| BASE surface | Candidate effect / proof |
| --- | --- |
| compaction.should_compress | Existing `_sweep_budget_hold_applies` reaches the durable-aware predicate. RED1 threshold proves rebuilt current-ingest rejection, cold unknown fail-open, host block and reason. |
| compaction.should_compress_preflight | Existing shared hold check after ingest sees the durable proof; RED1 preflight prevents repeated candidate scan. Replay-cleanup branch around BASE 316–341 is byte-identical. |
| compaction.compress | Local nonordinary pre-ingest behavior is preserved. Ordinary direct backstop checks after current ingest; RED1 direct prevents sweep. Pending ordinary hold is armed only after successful proof bookkeeping and no error status. |
| compaction._begin_compression | Records this invocation's pressure observation in `_last_gate_tokens` before the fingerprint decision. RED1/3 and force/recovery cases exercise the call path. |
| compaction._ingest_compression_view | Current proof permits ordinary held return before attribution/sweep. Current-ingest replay changes mismatch and run normally. RED1 direct and RED3 changed states cover it. |
| engine._start_no_progress_hold | Only ordinary reason calls the persistence helper. Hidden/objective/host refusals remain local; existing reason-specific tests remain remote-CI obligations. |
| engine._no_progress_hold_active / status | None/ordinary tuple consult scoped metadata; nonordinary reasons keep existing monotonic semantics. RED1/2/4 cover durable status, fixed expiry, corrupt/expired/future records. tools.py operator status still consumes the same status schema. |
| engine._automatic_compression_blocked / _no_progress_hold_blocks / _sweep_budget_hold_applies / _compression_block_reason | Existing common applicability and reason path; RED1 covers positive block and stable cooldown reason; RED4 covers recovery bypass and ceiling. |
| engine._record_ingest_failure / _ingest_messages | Clears current replay proof on ingest start/failure. New RED4 unreadable/failed-ingest case proves fail-open and source/record preservation. All existing production/test ingest consumers remain in candidate-consumers.txt. |
| engine._remember_active_replay_messages / _cached_active_replay_messages | Both successful cache construction and cache-hit paths refresh digest during ingest. No admission-time replay hash. RED1/2/3/4 pass through these paths. |
| engine._observe_no_progress_replay | Called only by the two existing successful replay bookkeeping paths above; RED3 replay case also injects a synthetic changed active replay to prove same-max-id invalidation. |
| engine._no_progress_fingerprint | Called only by persistence and the durable reader, never as an independent producer. Missing inputs return unknown. |
| engine._persisted_no_progress_hold_active | Called by shared active predicate and arming's no-renew check; RED1/2/4 cover it. |
| engine._persist_no_progress_hold | Called only for ordinary `_start_no_progress_hold`; no metadata record for other reasons. |
| engine._clear_persisted_no_progress_hold | Called after committed leaf in `_advance_after_leaf` and committed condensation in `_run_threshold_sweep_condensation`; RED3 all-tail publication covers the leaf path. Existing #909 condensation coverage is a remote obligation. |
| engine._no_progress_metadata_warning | Called by failing digest, fingerprint, metadata reader/writer and clear helper; RED4 unreadable proves a bounded generic diagnostic. |
| reset_state._reset_session_scoped_runtime_state | Local proof reset only; no metadata erase. New RED4 session-key isolation; existing behavior/test_rebind_equivalence remains remote obligation. |
| engine.clone_for_agent / construction | No hold copied; each binding loads from its own store after ingest. RED1 reconstruction covers durable effect without changing clone. |
| engine.note_turn_complete | Ordinary reason stays armed, same as BASE; RED2 covers reconstruction plus completed-turn notification. |
| engine.record_rejected_compaction | Still supplies nonordinary reasons, no new persistence. Existing #597/#665 host-refusal tests cover unchanged path; no new test needed to alter its policy. |
| engine._hold_fit_only_applies | Sweep-budget-only semantics retained; no no-progress fit-only policy added. Existing rc4 ceiling and #618 tests remain remote obligations. |
| benchmarking/replay.py / steady_state.py | Existing preflight consumers inherit the same decision; no benchmark implementation change or new caller-specific policy. Root owns deterministic cost evaluation. |
| Host automatic gates | Pinned host uses type-based `_automatic_compression_blocked` and `_compression_block_reason`; no API change. Synthetic RED1 invokes the same hook. |
| Host threshold/preflight/idle entrypoints | All continue through should_compress/preflight/compress. Pinned host inventory covers turn_context_compaction, turn_preflight and clone creation. No host source changed. |
| Manual/forced/provider-recovery callers | Existing explicit exemptions retained, with direct force and recovery call-spy proof plus ceiling predicate proof. Full supported host shapes remain remote-CI evidence. |


## 2. Previous-GA differential row

Required remote `Differential` workflow: default previous-GA baseline v0.28.0 at `709099b3a397a7893a7abae8802ebe7bf605f79e`; native default config, exact shapes, generated seed 1227 (60 public cases), 1,000 projection cases, and the pinned historical v0.26.2 g0/overflow cells. No accepted delta was added. Remote results and oracle counts are recorded in the PR Evidence block after completion.

## 3. Sibling shapes

Focused cells: rebuilt threshold/preflight/direct; cold unknown versus ingest-known; retained blocker immediately after valid frontier; +1/+300/+599/+601 expiry with turn completion; changed history/settings/summary/active replay/discrete pressure; all-tail complete tool group becoming eligible; distinct session and store; malformed/unknown/future/expired/unreadable metadata; failed ingest/publication; manual/recovery/ceiling exemptions; no metadata writes on hits; exact source bytes and frontier preserved. Target-only changes cover leaf target ratio/min/max and store binding cache invalidation.

Existing #651 file: `23 passed in 2.52s`, including real pinned synthetic host refusal adapters, below-threshold cleanup, manual/forced/recovery exemptions, survival ceiling, auxiliary isolation and session rebind. Rows/pressure invalidation assertions were updated from the obsolete TTL-only expectation, with a preceding failing run retained. Wider summary-only condensation, #909 replay cleanup, #665 auxiliary, golden traces, and rebind-equivalence coverage are remote-CI obligations.

## 4. Invariants

| Invariant | Sensitive proof / boundary |
| --- | --- |
| I1 exactly-once; I2 lossless | RED1/3/4 compare source rows and frontier; retained blocker never skipped |
| I3 fit; I4 objective | Recovery/ceiling and all-tail/new-tail fixture; supported fit/objective matrix in remote differential |
| I5 rollback; I6 migration | v0.28.1 reader opens candidate-written synthetic store; schema, rows, unknown metadata preserved |
| I7 summary outcomes | Failed ingest/publication cannot arm a new ordinary record; full route matrix in CI |
| I8/I9 tool groups | New eligible complete tool group preserves all original source ids/bytes; existing boundary code unchanged |
| I10 codec; I13 trial purity | Codec/trial source unchanged; remote differential required |
| I11 progress | Rebuilt-engine backstops and fixed expiry regressions; no sliding extension |
| I12 child isolation | Distinct session/store and existing #651 auxiliary call preserve foreground record |
| I14 atomic publication | DAG/lifecycle transactions unchanged; stale record cleared only after committed progress |

## 5. Risk probes

1. Arm a retained-blocker ordinary no-op, reconstruct with same binding, ingest, then query threshold/preflight/direct/host gate: all held, fixed expiry, zero scan/call/write, identical source rows/frontier. Cold query without replay proof must fail open.
2. Keep raw max id unchanged but alter active occurrence order/cleanup bytes or summary generation: old proof must not suppress same-turn work. Check useful summary-only condensation separately.
3. Append a complete tool group and enough tail to release old work: legitimate coverage advances without moving past unresolved retained sources.
4. Corrupt JSON, version, finite time, expiry interval or read access: fail open, one metadata-only diagnostic, no payload/identifier leak. No deadline sliding under clock skew.
5. Exercise manual force, provider recovery, forced overflow and exact survival ceiling: none may be suppressed. Rebind/auxiliary calls must not borrow or erase another session record.
6. Inspect replay/settings/current indexed state on every admission. Unknown state fails open; metadata hits do not write. Query plans and paired samples are in the cost card.

## Schema and compatibility

Metadata-only: no table, index, DDL, migration, message/DAG format, source row, stored provenance or coverage changes. v0.28.1 at `fafc21b977731da04bdae374cee36288c104adeb` reads a candidate-written synthetic store and preserves its unknown key; see `compatibility.txt` and `compat_probe.py`. This narrow reader proof does not claim every historical rollback shape.

## Proof boundary

Testing improved: rebuilt-engine and fingerprint-invalidation regressions for the compaction hold.

Synthetic proof only. This does not establish live group latency, host prompt/tool refresh latency, a #1162 repair, merge, release, runtime safety or customer readiness.

## Consolidated review correction

The existing first-map #608 budget-stop fixture now asserts no durable ordinary record while its process-local sweep hold stays active. Five required settings/route-only mutations invalidate an unchanged record. These six checks were RED at846a1b9b (`6 failed in 0.87s`); the same checks plus the three rebuilt consumer cells pass after repair (`9 passed in 0.87s`). The same reviewers inspect only this relevant delta, preserving their prior unaffected coverage. The final steady cost recheck and card are in `cost-review-delta/`; the initial `cost/` remains historical.
