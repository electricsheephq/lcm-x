# RELEASE READINESS V1 — the rc-first GA gate (owner-ratified 2026-08-26)

Every release that touches product code (anything outside bench/, docs/, tests/, and
.github/release-notes/) is **rc-first**:
no GA tag without a release candidate that has passed this gauntlet live. Point releases are not
exempt — a "small" diff on a hot path (ingest, recall, privacy) is exactly the profile that
needs a live soak. Docs/bench-only releases may skip to GA with a note in the release notes.

## Pipeline

1. **Merge** the release PRs through `land-pr` (exact-head CI, resolved threads, recorded
   review evidence).
2. **Tag `vX.Y.Z-rc1`** (hyphen ⇒ release.yml publishes it as a PRERELEASE) with curated notes
   at `.github/release-notes/vX.Y.Z-rc1.md`.
3. **Run the gauntlet** (phases A-C below) against the rc tag. Every phase produces a receipt
   file under the session-notes artifacts dir; the GA cut references all three.
4. **Fix-and-respin**: any P0/P1 finding → fix PR (through the gate) → `-rc2` → re-run.
   Carry-forward rule (every receipt binds an exact tree, so carrying needs proof): **Phase B
   re-runs on every respin** (it is diff-scoped by definition). Phases A and C may carry a prior
   rc's receipt ONLY when the rcN→rcN+1 diff touches nothing outside bench/, docs/, tests/, and
   .github/release-notes/ — AND nothing under bench/instruments/release_gauntlet/ (a changed
   gauntlet invalidates its own receipts) — the carried receipt is referenced WITH the diff-scope proof
   (`git diff rcN..rcN+1 --name-only`) recorded beside it. Any product-code delta re-runs all
   affected phases (ingest/recall/privacy deltas re-run everything).
5. **GA tag `vX.Y.Z`** only when A+B+C receipts are green for the passing rc tree. Because
   release.yml reads curated notes from the tagged tree, the GA commit may differ from the rc
   tree by EXACTLY the release-notes addition and nothing else — verified mechanically:
   `git diff --name-status rcN..GA` must show ONLY `A` (added) entries under `.github/release-notes/` — modifying or deleting existing notes is not the exception. GA notes =
   rc notes + gauntlet summary + receipt links.

## Phase A — Live all-tools matrix (the "clone" test)

A fresh, isolated hermes+LCM clone (fresh HOME-style env, fresh DB, HERMES_LCM_REPO = worktree
at the rc tag) — never a developer working tree. Two configurations, both driven LIVE:

- **cloud-default posture**: real cloud embedding provider (small corpus — spend is cents),
  durable sensitive patterns off (lossless store) and embedding privacy auto-on.
- **local posture**: fastembed/local provider, patterns off.

Matrix: EVERY registered `lcm_*` tool (enumerate from the tool registry at run time — currently
lcm_recall, lcm_status, lcm_doctor, lcm_expand, lcm_expand_query, lcm_describe, lcm_grep,
lcm_inspect, lcm_recent, lcm_retrieve, lcm_load_session, lcm_query_state, lcm_compute,
lcm_compile_evidence, lcm_evidence_pack — the runner FAILS if it finds a registered tool with no
matrix row, so new tools cannot ship untested) × both postures. Each row asserts a real
post-condition (hits returned, status fields present, doctor clean), never just "no exception".

Privacy batteries (behind the cloud key gate): the planted-secret battery proves the shipped
default keeps every planted secret raw in durable rows and recall while provider dispatches are
transformed, canonical, residual-free, and revision-validated. A fail-closed refusal is a valid
no-leak outcome for the chunk corpus: the chunk splitter can cut a dense planted fixture
mid-key, and the residual backstops then refuse that dispatch (report `stop_reason:
privacy_refused`, `privacy_blocked >= 1`) rather than ship it — the battery accepts exactly
that refusal shape (any other error still fails) and the raw-secret sweep covers everything
that did dispatch. Opt-out proves `privacy:off`
preserves byte-identical provider input. Durable-redaction preserves those same redaction and
placeholder checks as an opt-in posture. Misconfiguration uses an invalid pattern catalog to
prove lcm_recall raises, the proactive counter increments, status exposes privacy_policy_errors,
and assembly never breaks; its negative control proves the shipped default recall succeeds.

## Phase B — P0/P1 adversarial sweep (ultracode)

Multi-agent workflow over the FULL release diff (previous GA tag → rc tag), not per-PR deltas:
dimensions ≥ {correctness, security/privacy, performance/DoS, API contract, upgrade/migration,
concurrency} → independent finders → 2-vote adversarial verification → verdicts. The
upgrade/migration dimension is mandatory: open a previous-GA-created DB (old vectors,
placeholders, revisions) under the rc tree and exercise re-embed/migration paths. Cross-model
rule applies: finders and verifiers must not all share the author's model family.
Findings: P0/P1 verified ⇒ respin. P2 ⇒ tracked issue with disposition before GA.

A hands-on lane runs beside the sweep: the release's own fault cells and the regression suites
on each supported host build, at the rc tag. The release manager adjudicates the sweep and the
hands-on results together in `PHASE-B-RECEIPT.md`.

**Differential rule.** A verified P0/P1 shape that the previous GA tree also shows, where the
candidate is no worse than the base in every compared cell, is pre-existing, not a regression of
the candidate. It is established by a base-versus-candidate probe whose pass rule is written
down before it runs (same host pin, fixture and configuration; counts per cell on both trees).
A cell compares the affected rows by identity (role, turn and content hash), not only by count:
the candidate is no worse only when its affected rows are the base's rows or a subset of them.
A pre-existing shape is recorded as KNOWN with a tracked issue and does not force a respin. A
cell where the candidate is worse than the base follows the P0/P1 rule above. A shape the base does not show follows the P0/P1 rule above; in
particular the candidate respins when, against the base, it adds loss, duplicate rows, a wedge,
a session reset, or a request over the model window. The probe, its rule and both counts go
into the Phase B receipt.

## Phase C — Live-session soak

A scripted multi-turn session battery over `hermes acp` (the measured headless single-session
transport) against the clone: ingest-heavy turns, recall probes, compaction crossing at least
one threshold, doctor at close. Minimum 30 turns. Green =

- zero unexpected errors in engine logs. A provider failure the engine handles as designed (a
  rejected summary result logged with its reason, after which the compaction commits or stops)
  is not unexpected; the receipt lists each one;
- zero truncated level 3 nodes in the store at close, however a compaction ended (a verbatim
  level 3 node, whose summary is its own serialized source, is allowed);
- zero NEW publication-invariant conflicts (#247-class): when the count is not zero, the same
  soak runs on the previous GA tree on the same host, and the candidate fails only on a
  conflict kind the base does not show or a higher count. Both counts go into the receipt;
- zero recall-probe LOSS. A probe whose planting turn's raw row, or the literal in it, is no
  longer stored is `LOSS`, even when a summary or the answer still carries the literal. A probe
  with that row stored is `exact` when the answer carries the literal and `recoverable` when it
  does not (readers sometimes treat a planted decision as an injected claim). Only LOSS fails;
  the receipt reports all three counts. An `exact` count below the previous GA receipt's for the
  same battery is a finding with a tracked issue before GA;
- doctor clean.

## Receipts

Each phase writes `PHASE-{A,B,C}-RECEIPT.md`: rc tag + tree sha, exact commands, matrix results
(per-row pass/fail), findings + dispositions, and the claim class per the gate-closeout
discipline (a phase receipt claims what it measured, never "customer ready"). The receipts are
published as assets of the GA release, redacted (no local paths, host names or people), with a
`SHA256SUMS` file, and the GA release notes link each asset.
