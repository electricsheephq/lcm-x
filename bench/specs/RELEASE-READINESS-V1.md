# RELEASE READINESS V1 — the rc-first GA gate (owner-ratified 2026-08-26)

Every release that touches product code (anything outside bench/, docs/, tests/, and
.github/release-notes/) is **rc-first**:
no GA tag without a release candidate that has passed this gauntlet live. Point releases are not
exempt — a "small" diff on a hot path (ingest, recall, privacy) is exactly the profile that
needs a live soak. Docs/bench-only releases may skip to GA with a note in the release notes.

## Pipeline

1. **Merge** the release PRs through the two-lane exact-head receipt gate (unchanged).
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
   (`git diff rcN..rcN+1 --name-only`) recorded beside it. A respin whose rcN→rcN+1 delta is
   confined to those same paths re-runs Phase B over that delta only and composes the result with
   the carried Phase B receipt (the composed receipt records both). **This specification is never
   carry-safe:** when the delta changes an acceptance criterion in this file, every carried receipt
   for the affected phase is re-evaluated against the new criterion from the evidence it recorded,
   and the carry record states that re-evaluation; if the recorded evidence cannot be evaluated under
   the new criterion, the phase re-runs. Any product-code delta re-runs all affected phases
   (ingest/recall/privacy deltas re-run everything).
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

## Phase C — Live-session soak

A scripted multi-turn session battery over `hermes acp` (the measured headless single-session
transport) against the clone: ingest-heavy turns, recall probes, compaction crossing at least
one threshold, doctor at close. Green = zero unexpected errors in engine logs, zero NEW
publication-invariant conflicts (#247-class), recall probes hit, doctor clean. Minimum 30 turns.
**Differential rule for #247-class conflicts** (applied since the v0.23.2 train; codified
2026-09-05, #427): a non-zero conflict count passes ONLY when the same soak, on the same host
build, against the previous GA tree reproduces the same conflicts — matched by identity, never by
aggregate count alone. A conflict's identity is its kind, its publication point and message where
the host exposes them, and — always available, because the soak is scripted and deterministic in
its turn structure — its position: the soak turn index and compaction round at which it fired,
plus the error code/name the engine logs. Two runs match only when the multiset of identities is
equal; the receipt records both identity lists, the host pin and the differential run, and states
which identity fields the host surface exposed (the engine currently collapses every publication
conflict to the `publication_invariant_conflict` label without its source — tracked as a logging
follow-up, so today identity = kind + position + code/name). A conflict that does not reproduce on
the previous GA — by identity — is NEW and fails the phase. While #247 is open the count is
expected to be non-zero, so the differential run is mandatory whenever it is.

## Receipts

Each phase writes `PHASE-{A,B,C}-RECEIPT.md`: rc tag + tree sha, exact commands, matrix results
(per-row pass/fail), findings + dispositions, and the claim class per the gate-closeout
discipline (a phase receipt claims what it measured, never "customer ready"). The GA release
notes link all three. Receipts are published verbatim (local paths redacted) as comments on the
release train tracker issue; that comment URL is the link of record and is what the GA notes link,
and the GA notes (or the carry record they link) print the sha256 of each published receipt body
so an edited or deleted comment is detectable (codified 2026-09-05, #427).
