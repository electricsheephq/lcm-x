# Changelog

This file is the repository-root version history. LCM-X currently publishes
version tags but does not have a destination GitHub Release object for RC2.

## Unreleased

(nothing yet)

## v0.23.0 — 2026-08-21

The correctness batch: 17 product PRs, each adversarially reviewed with RED/GREEN receipts;
score-sensitive changes carry architect verdicts (#252) and boundary rows in
`bench/BASELINE-LEDGER.md`.

- Replay-proof/data-integrity: #177 (occurrence-bounded replay proofs + tool-name identity;
  one-time snapshot-digest re-persist on upgrade), #203 (OOB proofs bound to unique row
  identity; ID-less rows fail closed), #259 (fork-guard removed per architect veto; drop
  contract fenced).
- Retrieval: #173 (FTS cannot starve semantic recall; hardened preflight), #183
  (cross-session summary DAG expansion), #184 (session exclusion filters), #273 (crashed
  search discloses, never reports 'ok'), #172 (configurable Voyage reranker).
- Reliability/ops: #261 (FTS bootstrap race; integrity state publishes only with
  proof-at-publish-time), #179 (doctor payload-ref provenance), #168 (durable compaction
  totals in status), #198 (separate summary/expansion reasoning controls; warn-and-ignore
  config posture).
- Teams candidate (NOT enabled): #286 — AccessContextV1 contract package, pure additive.
- Tests/docs: #254 #257 (salvaged regression coverage), #262 (gpt-5.4 retirement note).
- Governance: #241 — the main ruleset is satisfiable again (phantom CodeQL contexts dropped,
  last-push-approval deadlock removed); --admin is an exception, not the path.

Known gaps are listed in `.github/release-notes/v0.23.0.md` (notably #90 under
investigation, #244/#288/#265/#260 accepted follow-ups, Teams slices 2-5 in v0.24.0).

## v0.22.0 - 2026-08-19

- Rename the project-facing documentation to **LCM-X — Lossless Context Memory
  eXtension** while preserving the compatibility identifiers `hermes-lcm`
  (plugin/skill/install path) and `lcm` (runtime engine).
- Point current install, CI, contribution, and tag links at
  `electricsheephq/lcm-x`; retain upstream links only as labeled provenance.
- Document the RC2 memory-evaluation evidence separately from the unmerged LCM
  Teams, RC2 reconciliation, and Codex/whitepaper candidates.

## v0.21.0-rc2 - 2026-08-05

### Changed

- #492 corrects the optional `tiktoken` trajectory-state chunking path to
  preserve UTF-8 character boundaries while keeping each decoded chunk within
  its token budget. If the budget cannot contain one complete Unicode
  character, the path fails explicitly instead of emitting replacement
  characters.

### Evaluation baseline included in RC2

- RC2 contains the deterministic LongMemEval retrieval harness, the committed
  500-question FastEmbed result, and the vendored judged-QA adapter described
  in [`benchmarks/METHODOLOGY.md`](benchmarks/METHODOLOGY.md). These evaluation
  surfaces landed before RC2; RC2 includes them rather than introducing all of
  them in the RC2-only delta.
- The full judged-QA result and recommended Voyage retrieval run remain pending.
  Retrieval metrics are configuration-specific evidence, not a release,
  runtime-safety, or customer-readiness claim.

## v0.21.0-rc1 - 2026-08-03

### Highlights

- Add the trajectory/experience-memory subsystem and the opt-in assertion,
  evidence, query-view, and adaptive-retrieval surfaces delivered by the
  consolidated wave-1 merge (#436).
- Keep the core SQLite schema at version 5. New feature stores use additive,
  named migrations in the same profile database, while disabled/default-off
  installs do not create optional assertion, query-view, or embedding tables.
- Improve large-store and startup behavior with bounded vector/metadata work,
  lock-contention retry during WAL conversion, and deferred temporal-rollup
  maintenance (#361, #440, #446, #447).

### Changed

- #436 adds the consolidated trajectory/experience-memory, retrieval,
  exact-evidence, citable-delivery, privacy, scale, and release-validation wave.
  Its committed benchmark results are directional evidence for the documented
  harness and corpus, not universal provider or workload guarantees.
- #361 retries WAL conversion when connection setup meets lock contention.
- #440 moves temporal-rollup maintenance off the session-start critical path;
  bounded background work is eventual and `lcm_recent` retains its fallback.
- #446 and #447 batch large fixture setup for embedding/vector metadata release
  coverage without changing runtime behavior.

### Upgrade notes

- Back up `lcm.db`, update the plugin checkout, restart Hermes, send one normal
  message, then verify `plugin_version: 0.21.0-rc1` and the expected database
  path with `lcm_status`. The core schema remains version 5.
- No manual core migration or embedding backfill is required from v0.20.0.
- Query/evidence tool schemas are exposed after upgrade, but assertion
  extraction, assertion storage, query-view storage, pre-answer evidence, and
  adaptive retrieval remain opt-in. Review provider/privacy boundaries before
  enabling model- or embedding-backed paths.

- Added nested-default-JSON-bounded, tool-extracted `lcm_expand_query` evidence provenance so successful and degraded answers retain synthesis-context identities, occurrences, paths, and excerpts while explicitly distinguishing locator coverage from unverified replay, semantic entailment, and caller authorization.

## v0.20.0 - 2026-07-23

Release focus: Lossless-Claw parity plus the merged cross-session recall and temporal retrieval stack.

- Completed the five selected Lossless-Claw parity behaviors: recoverable active-replay stubs for large externalized tool results; token-bounded fresh tails that preserve the newest message and complete tool-call/result groups; dry-run-first historical tool-output backfill with guarded rollback; bounded active-session externalized-payload search with strict ownership and recoverability checks; and bounded atomic threshold full sweeps with one final active-context publication. (#380, #381, #382, #413)
- Shipped the merged #413 recall and temporal surface: `lcm_recall`, `lcm_recent`, and `lcm_load_session`; semantic and hybrid retrieval over summaries and message chunks; temporal rollups with bounded fallback; optional proactive recall; and the corresponding benchmark and reproduction documentation.
- Release boundary: stock installs keep large-output externalization, active-replay stubbing, embeddings, temporal rollups, proactive recall, and threshold full sweeps disabled by default. Payload search requires explicit `content_scope`; historical backfill remains an operator-invoked, dry-run-first command. Committed benchmark results are directional evidence under their documented model and harness, not a universal provider-parity claim. This release does not include the later work tracked in #423, #434, or #436.

## v0.19.0 - 2026-07-07

Release focus: data-safety hardening, operator diagnostics, import tooling, benchmarking, and the WS5 engine decomposition.

- Hardened lossless storage and replay boundaries: GC tombstones preserve surrounding text, ingest failures surface in status/doctor, ignored-message drops are counted, persisted Hermes tool outputs and redacted durable retries replay losslessly, and auxiliary bypass/session fallback edge cases are covered. (#298, #308, #310, #312, #313)
- Strengthened storage and downgrade safety with serialized lifecycle/DAG writes, monotonic frontiers, path-contained externalized payloads, ReDoS-safe redaction, wrapped-base64 handling, a summary spend guard, and a schema-too-new open guard. (#300, #301, #302)
- Added operator and migration surfaces: read-only `lcm_inspect`, JSONL session export import, compression no-op status, compaction telemetry, benchmark-backed preset validation, and steady-state hot-path benchmarks. (#295, #303, #306, #307, #309, #320)
- Added CI-backed ruff linting and release/validation-friendly tooling updates, including follow-up JSONL import hardening and metadata JSON access through `MessageStore`. (#314, #315, #316)
- Began and documented the behaviour-preserving WS5 decomposition of the ~9k-line `engine.py`: stateful method clusters became `*Mixin` classes (`compaction.py`, `reconcile.py`, `aux_session.py`, `placeholder_ledger.py`) mixed back into `LCMEngine`, and pure/helper groups became plain modules (`engine_registry.py`, `codex_routing.py`, `sqlite_util.py`, `runtime_identity.py`, `message_analysis.py`). (#323, #324, #325, #326, #327, #328, #329, #330, #331, #332, #333, #334, #335, #336, #337, #338, #339)

## v0.18.1 - 2026-06-30

Release focus: compaction privacy, clone/hook integrity, doctor signal accuracy, and model-context safety.

- Excluded ignored backlog and stripped injected context before compaction, preventing ignored or synthetic context from entering LCM summaries. (#283, #282)
- Preserved Discord lane metadata, active LCM clone resolution, and context metadata through cloned engines and post hooks. (#292, #293, #289)
- Hardened runtime identity, raw tool call integrity refs, payload integrity checks, and doctor path/lifecycle diagnostics. (#281, #278, #279, #291, #273, #280)
- Updated Codex OAuth effective context window safety defaults. (#274, #276)
- Completed focus-topic demotion behavior and preserved raw session ownership across compression rollover. (#268, #269)
- Refreshed operator docs, community-health files, and release-validation guidance. (#272)

## v0.18.0 - 2026-06-18

Release focus: retrieval depth, durability, status provenance, and long-session correctness.

- Added recursive evidence support for `lcm_expand_query`, improving synthesized answers from expanded LCM context. (#266)
- Hardened externalized payload durability. (#265)
- Avoided duplicate ingest protection work on hot paths. (#262)
- Aggregated DAG status stats for cheaper health surfaces. (#264)
- Preserved source lineage after long sessions. (#263)
- Surfaced LCM config provenance in runtime status. (#261)
- Fixed per-turn ingest for WebUI sessions and batch timestamp deduplication. (#260)

## v0.17.0 - 2026-06-14

Release focus: automatic focus-topic derivation and lifecycle hygiene.

- Added auto-derived focus topics during compression.
- Added empty lifecycle-row garbage collection to prevent unbounded accumulation. (#256)
- Improved runtime context indicators.

## v0.16.x - 2026-06

Release focus: engine isolation, WAL durability, database-path clarity, and startup cost control.

- Isolated LCM engine state per agent. (#247)
- Preferred bound sessions on sibling chains when the host has zero DAG.
- Tuned compaction defaults and clarified context-threshold ownership. (#245)
- Clarified `LCM_DATABASE_PATH` override behavior. (#249)
- Hardened WAL durability and graceful-close checkpoints. (#237)
- Throttled startup FTS integrity checks to reduce launch time. (#236)

## Links

- Version tags: https://github.com/electricsheephq/lcm-x/tags
- Release workflow: [`.github/workflows/release.yml`](.github/workflows/release.yml)
- Validation expectations: [`CONTRIBUTING.md`](CONTRIBUTING.md)
