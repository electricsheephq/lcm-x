# PROGRAM HISTORY — archived plan-of-record eras (append-only)

The maintainers run LCM-X from a living plan of record that is kept outside the repository (the
convention `bench/PROGRAM-PLAN-v5.md` states: the human-readable plan lives outside the repo; the repo
documents are authoritative for gates). When a plan era is superseded, its verbatim text is archived
privately and this file receives one curated entry: what happened, what was decided, which pull
requests, issues, tags and findings carry the evidence, and which rules the era produced. Entries are
append-only. Issue and pull-request numbers refer to `electricsheephq/lcm-x`; `F<n>` refers to
`bench/FINDING-F<n>-*.md`; tags are this repository's git tags. The referenced artifact is the record; this
file only points at it.

---

## Entry 1 — 2026-08-19 → 2026-09-05: plan of record v21 → v96

**Scope.** The period from the migration of the benchmark program into this repository (v0.22.0) through
the v0.23.2 stable release, its soak, the co-maintainer review queue, and the parked V1-M re-bank. The
successor plan (v97, 2026-09-05) restarted the weekly release cadence and opened the Teams and
benchmark lanes described in #323.

### 2026-08-19/20 — consolidation (plan v21)

- Roles fixed: an architect + release-manager orchestrator with a standing autonomous grant; owner gates
  = public naming/publication content, net-new spend categories, release cuts, live customer changes.
- Verdict channel for lane dossiers: #252 (architect verdict batch 1: #172 LAND, #173 LAND-AFTER-REBASELINE
  as part of the joint FTS program with F58, #194 and others).
- Flagship retrieval row banked: FINDING-F53 (LongMemEval V1-M, r@10 95.6 / ndcg@10 0.861 / r@1 50.0,
  A/A′ 0.00pt spread). The rerank family closed without adoption (F55–F58); the L2 tier stayed parked (F52
  §5 correction, #287).

### 2026-08-20/21 — compaction pilot, host migration, LoCoMo arms

- N9 compaction pilot → FINDING-F59: seven arms; headless `codex exec` never compacts (paginated long
  context, 0 compactions at 553,914 in-context tokens); LCM single-session retention 100% at roughly one
  third of the tokens; the restart-bridge and compressor controls 86.7%. Amendments 9–14 (#302, #310,
  #312, #313, #315) recorded the corrections an adversarial review round found (soul-prompt confound,
  recall riding the host memory surface, `publication_invariant_conflict` warnings on every ACP arm).
  Product gaps filed: #301 (ACP toolset not disableable), #314 (compaction fires once, at any threshold).
- Host migration for the retired model: #309 (fixes #263, the wrong 372,000 context cap; first pinning
  test), customer guidance drafted for the owner.
- LoCoMo C1 (FTS-ON) → FINDING-F60: FAIL inside the noise floor (54.03 / 54.38 vs the 54.6 declared
  configuration, F48); #306. LoCoMo C2 (speaker attribution, B3-A) registered in #316 → FINDING-F61:
  aggregate 67.42 / 67.72 vs 54.6 (+12.8pt), adversarial 62.3 vs 32.7, both arms pass; #381. The
  productization epic became #379 with #317 → #324 as the dependency order.

### 2026-08-21 → 25 — the interim lane

- While the orchestrator was away, a second lane shipped v0.23.0 (08-22) and v0.23.1 (08-23, privacy
  release: #332 / #333 / #338), changed the V1-M instrument (#352, receipt "land after re-baseline"),
  installed the exact-head AI-review gate as a required check (#349, #358), opened the quality roadmap
  #323 with #317–#321 and #324, and froze the retrieval lane on an owner gate (#353, PR #354).

### 2026-08-26 — reconciliation and the merge gate

- Durable-state audit of every open issue, PR, milestone and document; milestones for v0.23.2 and the
  B3-A flagship; architecture issues #375 (Phase-B runner), #376 (Phase-C runner), #377 (v0.23.2
  tracker), #378 (F61 registration), #379 (B3-A epic), #380 (F53 re-bank); #346 amended on record.
- Security: #365 (PEM private-key body could reach a cloud embedding provider when an assignment prefix
  preceded the block) → #366, closed after 25 review rounds with a declared threat model and a
  marker-independent fail-closed validator.
- Instrument integrity: #367 → #370 (privacy-policy errors on the recall path fail loud instead of
  degrading to full-text). Ledger reconciliation: #368. Rerank payload protection: #371, fixed inside
  #374.
- Owner ruling recorded in #374: lossless is the brand. Durable sensitive-pattern redaction is opt-in
  forever; cloud-embedding privacy became an independent flag (`LCM_EMBEDDING_PRIVACY_ENABLED`) that
  touches only the provider-bound copy.
- Release process: `bench/specs/RELEASE-READINESS-V1.md` (#373) — every release with product code is
  rc-first through a three-phase live gauntlet (A: all-tools matrix on a fresh clone; B: multi-agent
  P0/P1 sweep of the previous-GA→rc diff; C: 30+ turn live session soak).
- The required "AI review exact-head" check was reverse-engineered and adopted as the program's merge
  discipline: two receipts per exact head (acceptance + an independent adversarial review, reviewer ≠
  author), zero unresolved review threads, receipts re-issued whenever the head or base moves. The
  gate's peer-reset behaviour was fixed in #362 (13 review rounds; the dispatch payload now carries a
  required dispatch id). Gate right-sizing follow-ups: #369, #392.

### 2026-08-26/27 — the v0.23.2 train

- rc1 (Phase A caught #383, a truncated-PEM leak → #384, seven review rounds) → rc2 (Phase A caught
  #389, the over-block regression of that fix → #391) → rc3 (#388 release identity, #393 notes
  hygiene). A five-auditor read-only audit of the train ran before the rc3 merge (#391 records the
  audit; the count of 25 confirmed findings, 3 of them P0, comes from the maintainers' session record); all
  fixed or dispositioned on record.
- rc3 gauntlet: Phase A PASS (30/30 matrix rows, planted-secret battery); Phase B zero release-blocking
  findings after the owner's severity reclassification (severity = impact × exposure; the residual
  redaction precision class became the long-term backlog issue #394); Phase C PASS with a
  pre-existing finding (#247, proven pre-existing by an identical soak against v0.23.1).
- GA v0.23.2 on 2026-08-27 (#395 notes, #396 contributor credits + CHANGELOG coverage). The GA tree is
  byte-identical to rc3 plus the notes file.
- The owner then held the release for a one-week soak; no field reports arrived (maintainers' support
  record; the hold itself is public on #411).

### 2026-09-04/05 — hold lift, co-maintainer queue, the parked re-bank

- Co-maintainer Tosko4 opened #397–#410 on 2026-08-31. Verdicts posted 09-04; #404 (cached FastEmbed
  warmup), #407 (telemetry test deflake) and #408 (linear Teams backfill) merged; the rest await
  revision under the posted findings. #155 (session-end prefix matching extracted to a mixin) merged.
  #411 ("receipt producer outage") was diagnosed as the gate's base-push reset plus the review hold —
  the dispatcher works; the fix path is #369.
- Ledger rows for the security/privacy train: #412. Gate fix merged: #362.
- F53 re-bank registration: #413 (23 rounds; four confirm rounds failed on the way and are disclosed
  in the PR record; follow-up instrument items in #415). Execution then PARKED at the pre-spend gate:
  under the shipped privacy posture the transform changes 617 occurrences / 93 units across 258 of 500
  questions and refuses 3 corpus texts (17 questions) → FINDING-F62 (#416; five confirm rounds failed
  before r9 passed; instrument follow-ups K1–K11 on #415). #380 became an owner fork (leave / precision
  fix / opt-out variant); the owner ruled on 2026-09-05 (cloud-copy privacy transform opt-in by default
  → the raw-path re-bank becomes the re-bank of record, registered against the post-flip main; the ruling
  is recorded in the maintainers' plan of record and lands on #380 with the privacy-flip train, the next
  minor release).
- 2026-09-05: plan of record v97 — weekly rc-first cadence with GA on Wednesdays, take-over of the
  co-maintainer queue, the privacy-default change as a versioned train, the Teams enterprise lane
  (evaOS-first, shared-store architecture) and the benchmark lane (raw-path re-bank, the V1-S reader
  row truth correction, the V1-M reader row). Tracked from #323.

### Rules this era produced (each with the incident that taught it)

1. **Recompute at claim boundaries** — a labelled tuple carried a scramble that thirty seconds of
   recomputation caught (F46/F48 category labels; F62 §4.2 occurrences vs units vs questions).
2. **Append-only corrections** — findings, ledgers and run sheets are never rewritten; a dated line
   or amendment supersedes (F59 Amendments 9–14; RUN-SHEET-V1M-REBANK Amendment 1).
3. **Registration before spend, A/A′ per configuration** — every paid run has a merged run sheet with
   pre-declared bands and a noise-floor pair (F53, F59, F60, F61; the re-bank parked exactly because
   the gate ran before spend).
4. **Reviewer ≠ author, across the model boundary** — the author's own tests missed what an independent
   reviewer found on #366 (25 rounds), #384 (7), #362 (13), #413 (23).
5. **rc-first gauntlet** — the release candidate is the testing vehicle; GA is the last green rc plus
   notes only (rc1 and rc2 of v0.23.2 each caught a real defect the suite did not).
6. **Receipts bind exact heads; never merge on stale receipts** — a merge that preceded its gate result
   is disclosed on #364/#369; `merge_gated` became the only merge path.
7. **Severity = impact × exposure** — an impact-only rubric over-invested a day in a minority path
   (owner reclassification during the rc3 gauntlet; #394).
8. **Occurrences vs units vs questions are three different numbers** — five confirm rounds on #416
   each corrected a sentence that conflated them.
9. **A procedure sentence is a capability claim** — a run sheet registered a transform-off knob that did
   not exist (#413 r11); verify the knob before registering the step.
10. **Snapshot outputs before any re-run; never mutate a worktree another lane owns** — both measured
    losses (a completed run's scores wiped; a codex round corrupted by a stash in its worktree).
11. **Audit the run-path premise, not only the numbers** — two passing audits re-derived parity figures
    while a wrapper's attribute forwarding sent cached runs down an uncached path (#413 r4).

### Archive pointer

The verbatim orchestrator plan-of-record history for this era (2,628 lines, 266,983 bytes, sha256
`7e1630ce968f9d940956b35cacdb3f5bf7c293422775e557d9d976f5d2412ac8`) is archived privately by the
maintainers on 2026-09-05. Locator: file `PLAN-OF-RECORD-HISTORY-2026-08-19..2026-09-05.md` with its
`manifest.json` (both hash-anchored) in the maintainers' private session-notes store; a mirror gist, when
created, is recorded in the maintainers' plan of record. It contains operational detail and local paths
and is not published — request it from a maintainer; this entry is the public record. Anyone holding the
archive can verify it against the hash above.
