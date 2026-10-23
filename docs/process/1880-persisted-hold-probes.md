# #1880 B persisted no-progress hold probe packet

Scope: automatic unchanged no-progress suppression across engine rebuilds using one existing metadata-table key per session. Refs #1880; #1162 is context only and remains deferred.

Baseline: `10407e12e33c1d7b09fe0e05141153555aec5d62`. Admission product: `778609b68bc39ef4f216728ccb2952d36b53a8b1`; the final status/repair delta is recorded below. Evidence files are under `docs/process/evidence/1880-persisted-hold/`.

Only ordinary completed `no_progress` persists. Legacy unfinished/budget and no-lineage refusal pacing remains process-local; it cannot certify a durable proof. A positive completed-terminal classification excludes unfinished budget, route, rejection, summary failure and error stops; the separate sweep hold remains process-local. A record hit never rewrites metadata or extends expiry. Invalid, unreadable or future-dated metadata fails open with one generic metadata-only warning. A changed fingerprint is logically invalidated in the same call without a gate write. Successful publication clears only the applicable scoped record.

The digest is computed during existing successful ingest; admission reuses that proof and bounded indexed state. Cold threshold queries without replay proof fail open. Rebuilt preflight/direct compression ingest before the shared held decision; no repeated attribution or summary work is admitted on a matching proof. The resolved store identity is cached against exact store-object and db-path equality, while metadata and current indexed state are read on every gate. Leaf target ratio/minimum/maximum, L2 budget ratio, summary prompt version, stub-first exit target, proactive-recall enable/budget and embedding availability are included in the settings digest. Exact request tokens accompany the pressure bucket and discrete classes so within-bucket guard transitions invalidate. Current model/provider/API-mode scalar route identity is included; opaque or URL-shaped routes cannot supply durable proof without reading credentials or base URLs; a completed ordinary result retains BASE process-local pacing.

Manual force, provider recovery, forced overflow, survival ceiling, replay-changing below-threshold cleanup, nonordinary no-progress reasons, and sweep holds retain their own policies. Part A merged as8d1475ae3e0415aef69e7d3ee1eee8c8f1394832 and was incorporated with a merge commit; its cleanup changes are preserved.

## 1. Consumer enumeration

Raw inventories: base-consumers.txt is the complete narrow BASE hold producer/consumer search; candidate-consumers.txt (refreshed after settings/cost correction) includes call sites of every new helper plus ingest/reset consumers in production and tests. Root's host-consumers.txt pins NousResearch/hermes-agent at `d923a5eb487a17c0a68801bba9ef0f56c81040a8` through GitHub API; installed host/profile sources were not read.

| BASE surface | Candidate effect / proof |
| --- | --- |
| compaction.should_compress | Existing `_sweep_budget_hold_applies` reaches the durable-aware predicate. RED1 threshold proves rebuilt current-ingest rejection, cold unknown fail-open, host block and reason. |
| compaction.should_compress_preflight | Existing shared hold check after ingest sees the durable proof; RED1 preflight prevents repeated candidate scan. The separately landed part-A replay-cleanup branch is preserved. |
| compaction.compress | Local nonordinary pre-ingest behavior is preserved. Ordinary direct backstop checks after current ingest; RED1 direct prevents sweep. Pending ordinary hold is armed only after successful proof bookkeeping and no error status. |
| compaction._begin_compression | Records this invocation's dedicated fingerprint pressure observation without mutating the legacy runtime gate field. RED1/3 and force/recovery cases exercise the call path. |
| compaction._ingest_compression_view | Current proof permits ordinary held return before attribution/sweep. Current-ingest replay changes mismatch and run normally. RED1 direct and RED3 changed states cover it. |
| engine._start_no_progress_hold | Only a positively completed ordinary reason calls the persistence helper with explicit durable intent. Hidden/objective/host refusals remain local; existing reason-specific tests remain remote-CI obligations. |
| engine._no_progress_hold_active / status | No tuple or a durable-proof tuple consults scoped metadata; a local-only ordinary/refusal tuple keeps legacy monotonic pacing. A durable tuple never overrides fingerprint mismatch or metadata failure. RED1/2/4 cover durable status, fixed expiry, corrupt/expired/future records. tools.py operator status still consumes the same status schema. |
| engine._automatic_compression_blocked / _no_progress_hold_blocks / _sweep_budget_hold_applies / _compression_block_reason | Existing common applicability and reason path; RED1 covers positive block and stable cooldown reason; RED4 covers recovery bypass and ceiling. |
| engine._record_ingest_failure / _ingest_messages | Clears current replay proof on ingest start/failure. New RED4 unreadable/failed-ingest case proves fail-open and source/record preservation; a recovered successful direct ingest clears prior failure bookkeeping. All existing production/test ingest consumers remain in candidate-consumers.txt. |
| engine._remember_active_replay_messages / _cached_active_replay_messages | Both successful cache construction and cache-hit paths refresh digest during ingest. No admission-time replay hash. RED1/2/3/4 pass through these paths. |
| engine._observe_no_progress_replay | Called only by the two existing successful replay bookkeeping paths above; RED3 replay case also injects a synthetic changed active replay to prove same-max-id invalidation. |
| engine._no_progress_fingerprint | Called only by persistence and the durable reader, never as an independent producer. Missing inputs return unknown. |
| engine._persisted_no_progress_hold_active | Called by shared active predicate and arming's no-renew check; RED1/2/4 cover it. |
| engine._persist_no_progress_hold | Called only for ordinary `_start_no_progress_hold`; no metadata record for other reasons. |
| engine._clear_persisted_no_progress_hold | Called after committed leaf in `_advance_after_leaf` and committed condensation in `_run_threshold_sweep_condensation`; The r1 committed-leaf and committed-condensation tests assert scoped metadata clearing; each fails with its publication clear removed. RED3 proves eligibility and coverage advancement, not the metadata clear. |
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

## Consolidated integration correction

Historical integration candidate8fc7b6a5 used221 changed lines versus BASE; H2 below supersedes this source identity. Source attribution refusal is not a completed ordinary eligibility/publication proof: no-lineage refusals retain legacy local pacing and never write a durable record. The known retained-occurrence blocker after a completed ownership scan remains durable, including rebuilt threshold/preflight/direct consumers. This conservative distinction repairs prior-GA cadence without removing fingerprint components or adding a no-work classifier.

Behavioral selection:49passed in16.47s. No-lineage durable-absence assertion was RED before repair, then the targeted no-lineage/rebuilt/corrupt/settings/environment selection15passed in2.55s. Two existing small-window probes match saved GA cadence and coverage; full remote Differential remains required and no bounds/accepted deltas changed. Trace recovery224passed in37.67s; all162 compress goldens passed in the preceding trace run. Two local --repo subprocess bootstrap failures are environment-only; required remote CI owns their proof.

New ingest observer state is retained. Sandbox-dependent opaque digest equality and address-keyed cache identities are canonicalized; actual replay/public bytes, state transitions and trace events remain observed. The fixture-size cap increases only from3,350,000 to3,400,000bytes for stable3,371,173bytes after added state (original3,345,646bytes). Assembly keeps the added metadata read; unrelated local embedding-query removals were restored. The historical integration steady protocol/card/raw re-score are in cost-integration-delta/; H2 cost remains pending and prior results remain unchanged.

Current feedback: direct-ingest recovery, effective stub/recall settings, exact-pressure guards and cache projection are fixed with targeted checks. The UID/identity binding concern received one bounded existing-fixture check; rawmax/replay already changed, so no current stale-match failure was verified and no speculative generation subsystem was added.

## H2 semantic observation isolation

Product81229ac490067e2514a29601c0a09f771ae42de0 uses243 B product lines versus merged main8d1475ae,288 total versus original BASE, below450. The dedicated pressure observation fails open for unknown/nonpositive inputs and leaves the legacy runtime pressure field unchanged. Fingerprinting includes the configured yield-ready class, so reconstructed pre-yield counts0/1/2 match at shipped minimum3 while crossing3 invalidates. Ingest records its starting monotonic failure generation; replay proof and success bookkeeping require no new current-ingest failure. A swallowed audited-prefix failure remains unknown until a genuinely clean later ingest.

Embedding availability is an explicit effective assembly input. Current host-held stub replacements supply their actual filtered public bytes without extra public-row projections. Trace observation retains copy-return owners before prepare/cache/error replacement can free and recycle their addresses; existing aliases and public values remain observed.

Eight targeted cells were RED before product edits, then12 targeted cells passed. Seventeen preservation cells passed alongside three old golden failures before regeneration. All419 affected-trace checks passed after one regeneration per affected harness. The ingest fixture is3,388,310bytes under its unchanged3,400,000 cap. Assembly changes only its legacy gate observation3000 to0; unrelated environment-dependent vector-query removals were excluded. Commands and observed lines are in h2-red-green.txt.

Both independent targeted delta3 reviewers PASS this H2 product/test scope, preserving prior unaffected coverage. The previous-GA Differential and four pinned real-host workflows retain their assertions and bounds. The existing #1353 invariant expectation is updated to retain a current swallowed failure, then explicitly prove the first clean retry clears it and restores replay proof; all row/identity/idempotence assertions remain.

H2 hosted cost failed two nonmatching-record cells; its samples and failed card are retained in cost-h2-failed-hosted/. Profiling identified repeated synchronous auxiliary-scope discovery. H3 passes the already checked scope through private calls, never caches that verdict, and retains current metadata/indexed fingerprint reads on each admission. Auxiliary default helpers remain guarded.

## Final admission cost and host oracle

Final hosted measured source441c338ff49d3d1bd64a864a9191ce2723c509bb, admission product778609b68bc39ef4f216728ccb2952d36b53a8b1: all four UNHELD cells pass the original paired200 cost bar. HELD warm and reconstructed gates make zero model calls, full-history scans or metadata writes. Exact p50/p95, protocol, raw timings and re-score inputs are in cost-final-hosted/RESULT.md and RESULT-MANIFEST.json. Genuine current ingest and pressure setup occur outside timing; no failure generation is assigned manually. One hosted run; no cross-run/cross-machine speedup claim. The local disk floor was respected by using existing hosted CI. Prior cost cards retain their named source identities.

The permanent injected child-publication-conflict cell now checks every content-changing host override for current-user identity, full raw prompt before, exact clean override after, and no summary carrier. It checks full raw prompt storage for all80 turns, positive fault activation and all inherited fault-return/rotation assertions. Ordinary whitespace, uniqueness and identity assertions remain unchanged. The pinned host explicitly restores API-only current-user variants. This fault-specific oracle does not claim duplicate or identity repair under permanent injected failure. The exact runtime-file tests-ran exception links the existing real-host workflow; all four prepared host refs remain required. Independent H4 acceptance passed sourceabb4ec20; the named pinned reference-host case passed in run39916037724/job120289728706 on sourcec6a31be8. All-four-host final execution remains a separate gate.

Final correction restores observation of legacy local holds in auxiliary status without relaxing admission guards. Successful in-place level3 summary repair invalidates affected sessions' existing persisted hold keys atomically with repair; unrelated/rolled-back session proofs remain. This covers summary-content mutation without an admission scan, new key or schema. Exact commands/results and final hosted gates are recorded in final-proof.txt and the PR Evidence block; this packet alone makes no full-suite/readiness claim.

## Final repair/merge preservation delta

Actual in-place repair session ownership is validated with current text under the existing transaction before any summary or hold mutation. Supported session transfer during summarization now aborts repair; moved ownership, original summaries/raw rows and both session records remain. The new callback regression was RED remotely (one failure among33repair cells), then all33repair cells passed on the correction. Normal repair invalidation/refusal/concurrent-parent/post-delete rollback assertions remain intact. The targeted preservation reviewer verified the prior blocker fixed; prior acceptance coverage remains.

Current mainca98ba6a was merged because its conflict prevented PR CI. Source merged cleanly. Exactly four sequence ingest traces overlapped: existing full202scenario observation plus fullfixture hash verified their composition, with198cases unchanged and original size/assertion bounds retained. See final-proof.txt for source IDs/commands. B-owned product is276 changed lines against pinnedmain; originalBASE comparison includes alreadylanded upstream work and is disclosed separately.

Review disposition: direct-ingest recovery, effective settings/route/exactpressure, legacy budget/refusal pacing, current-ingest failure/cache, pressure readiness, embedding availability, in-place summaryrepair and stale ownership are fixed now. Temporary manual CI guards removed and public local_raw metadata removed. The earlier attribution-repair hypothesis did not demonstrate current stale-match impact in its bounded existing-fixture check; no speculative generation subsystem was added. Threads were neither replied to nor resolved under this task's authority.

## FIX ROUND r1

The completed-verdict fingerprint is captured in `_finish_without_leaf` and must equal the current fingerprint before persistence. A changed or unknown fingerprint and a failed metadata write retain exactly the BASE 600-second monotonic local hold, without a durable record; rebuilt engines remain open. Committed progress clears corrupt JSON through the existing metadata-only null write, while an absent key causes no write.

The focused r1 regressions cover route-with-`@` and metadata-write failure, rebuilt-engine noninheritance, local expiry/direct backstop, same-session publication between verdict and persist, corrupt/valid/absent leaf clears, and condensation clear. Both clear-removal mutants fail the intended record assertion. The leaf fixture forbids condensation so one clear cannot mask another. [Proof and exact observed results](evidence/1880-persisted-hold/r1/proof.txt), [affected test files](evidence/1880-persisted-hold/r1/affected-files.txt), and [unchanged admission call graph](evidence/1880-persisted-hold/r1/admission-call-graph.json).

The compress golden inventory adds `_no_progress_proof_fingerprint`, preserving equality and transitions through stable digest labels. Its 162 scenario changes contain only the new snapshot observation and full-state count/digest; existing event and message observations remain. No new cost run is required: all admission implementations remain byte-identical. The runtime-only test allowlist now correctly describes execution by the non-required real-host.yml.

Testing improved: publication-clear tests with mutant proof; persistence-failure and concurrent-publication regression tests.
