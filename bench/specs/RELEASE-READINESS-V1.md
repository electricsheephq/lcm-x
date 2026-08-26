# RELEASE READINESS V1 — the rc-first GA gate (owner-ratified 2026-08-26)

Every release that touches product code (anything outside bench/, docs/, tests/) is **rc-first**:
no GA tag without a release candidate that has passed this gauntlet live. Point releases are not
exempt — a "small" diff on a hot path (ingest, recall, privacy) is exactly the profile that
needs a live soak. Docs/bench-only releases may skip to GA with a note in the release notes.

## Pipeline

1. **Merge** the release PRs through the two-lane exact-head receipt gate (unchanged).
2. **Tag `vX.Y.Z-rc1`** (hyphen ⇒ release.yml publishes it as a PRERELEASE) with curated notes
   at `.github/release-notes/vX.Y.Z-rc1.md`.
3. **Run the gauntlet** (phases A-C below) against the rc tag. Every phase produces a receipt
   file under the session-notes artifacts dir; the GA cut references all three.
4. **Fix-and-respin**: any P0/P1 finding → fix PR (through the gate) → `-rc2` → re-run the
   affected phases (full re-run if the fix touches ingest/recall/privacy).
5. **GA tag `vX.Y.Z`** only when A+B+C receipts are green for the exact rc tree that GA is cut
   from (rcN tree == GA tree, tag difference only). GA notes = rc notes + gauntlet summary.

## Phase A — Live all-tools matrix (the "clone" test)

A fresh, isolated hermes+LCM clone (fresh HOME-style env, fresh DB, HERMES_LCM_REPO = worktree
at the rc tag) — never a developer working tree. Two configurations, both driven LIVE:

- **cloud posture**: real cloud embedding provider (small corpus — spend is cents), sensitive
  patterns enabled (the only configuration production permits for cloud).
- **local posture**: fastembed/local provider, patterns off.

Matrix: EVERY registered `lcm_*` tool (enumerate from the tool registry at run time — currently
lcm_recall, lcm_status, lcm_doctor, lcm_expand, lcm_expand_query, lcm_describe, lcm_grep,
lcm_inspect, lcm_recent, lcm_retrieve, lcm_load_session, lcm_query_state, lcm_compute,
lcm_compile_evidence, lcm_evidence_pack — the runner FAILS if it finds a registered tool with no
matrix row, so new tools cannot ship untested) × both postures. Each row asserts a real
post-condition (hits returned, status fields present, doctor clean), never just "no exception".

Privacy battery (cloud posture): ingest planted secrets (the #365 shape battery: PEM complete /
truncated / encrypted-armor / serialized / prefixed; passwords; api keys) through the real
ingest path, then verify LIVE: durable store redacted, embedding dispatches validated (no
residual), placeholders canonical, privacy revision registered, recall returns placeholders.
Loud-fail battery: cloud provider + patterns off ⇒ lcm_recall raises; proactive counter
increments; status exposes privacy_policy_errors; assembly never breaks.

## Phase B — P0/P1 adversarial sweep (ultracode)

Multi-agent workflow over the FULL release diff (previous GA tag → rc tag), not per-PR deltas:
dimensions ≥ {correctness, security/privacy, performance/DoS, API contract, upgrade/migration,
concurrency} → independent finders → 2-vote adversarial verification → verdicts. The
upgrade/migration dimension is mandatory: open a previous-GA-created DB (old vectors,
placeholders, revisions) under the rc tree and exercise re-embed/migration paths. Cross-model
rule applies: finders and verifiers must not all share the author's model family.
Findings: P0/P1 verified ⇒ respin. P2 ⇒ tracked issue with disposition before GA.

## Phase C — Live-session soak

A scripted multi-turn session battery over `hermes acp` (the measured headless single-session
transport) against the clone: ingest-heavy turns, recall probes, compaction crossing at least
one threshold, doctor at close. Green = zero unexpected errors in engine logs, zero
publication-invariant conflicts (#247-class), recall probes hit, doctor clean. Minimum 30 turns.

## Receipts

Each phase writes `PHASE-{A,B,C}-RECEIPT.md`: rc tag + tree sha, exact commands, matrix results
(per-row pass/fail), findings + dispositions, and the claim class per the gate-closeout
discipline (a phase receipt claims what it measured, never "customer ready"). The GA release
notes link all three.
