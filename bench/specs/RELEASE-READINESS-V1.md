# RELEASE READINESS V1 — the rc-first GA gate (owner-ratified 2026-08-26)

Every release that touches product code (anything outside bench/, docs/, tests/, and
.github/release-notes/) is **rc-first**:
no GA tag without a release candidate that has passed this gauntlet live. Point releases are not
exempt — a "small" diff on a hot path (ingest, recall, privacy) is exactly the profile that
needs a live soak. Docs/bench-only releases may skip to GA with a note in the release notes.

Release dates are ceilings, not schedules: a release ships as soon as its gate passes; scope is never added to fill time; a failed invariant is never moved.

## Release trains (from v0.29.2)

A release is a train. One release candidate carries every fix that is merged and ready when
its rc1 is cut, and the gauntlet runs once for the whole train, not once per fix.

- **Patch trains** are cut from the previous GA commit on the maintenance branch
  `release/vX.Y.x`, never from main once main holds the next minor's work. Each fix merges on
  main first, unless main does not have the defect (for example, its code was rewritten); that
  fix lands on the maintenance branch only, with the reason recorded. A twin whose
  `git patch-id` equals its main PR's links that PR's review evidence; any other twin gets its
  own review. The release manager cuts rc1 when the queued fixes are worth a gauntlet. A ready
  fix waits at most 14 days for its train's rc1; that is a ceiling, not a schedule.
- **Hotfixes:** only a P0 or P1 ships outside a train, in its own rc. That rc still runs the
  smoke stage and the rows its diff selects. It may carry other ready fixes only when they add
  no row to its selection. A hotfix GA that ships after a train's cut and before the train's GA makes
  the train respin onto that GA, which becomes the train's previous GA for the selector.
- **After the cut,** a train's tree changes only by fix-and-respin (Pipeline step 4). A fix
  that misses the cut waits for the next train.
- **Minor releases** are trains cut from main at their milestone's gate.

## Pipeline

1. **Merge** the release PRs through `land-pr` (exact-head CI, resolved threads, recorded
   review evidence).
2. **Tag `vX.Y.Z-rc1`** (hyphen ⇒ release.yml publishes it as a PRERELEASE) with curated notes
   at `.github/release-notes/vX.Y.Z-rc1.md`.
3. **Run the gauntlet** (phases A-D below) against the rc tag. Every phase produces a receipt
   file under the run artifacts directory; the GA cut references all of them. Phase D's full
   tier runs at the rc1 of every release, major, minor or patch (later rcs: its carry-forward rule).
   From v0.31.0, and for patch releases from v0.29.1, the Tiers section below selects the required rows; the minor
   releases v0.29.0 and v0.30.0 run all phases. From v0.29.2 the smoke stage (L1, below) runs first, and no L2 row starts until it passes.
4. **Fix-and-respin**: any P0/P1 finding → fix PR (through the gate) → `-rc2` → re-run.
   Carry-forward rule (every receipt binds an exact tree, so carrying needs proof): **Phase B
   re-runs on every respin** (it is diff-scoped by definition). Phases A and C may carry a prior
   rc's receipt ONLY when the rcN→rcN+1 diff touches nothing outside bench/, docs/, tests/, and
   .github/release-notes/ — AND nothing under bench/instruments/release_gauntlet/ (a changed
   gauntlet invalidates its own receipts) — the carried receipt is referenced WITH the diff-scope proof
   (`git diff rcN..rcN+1 --name-only`) recorded beside it. Any product-code delta re-runs all
   affected phases (ingest/recall/privacy deltas re-run everything; under the Tiers section, every selected row).
5. **GA tag `vX.Y.Z`** only when A+B+C+D receipts are green for the passing rc tree (under the Tiers section,
   the receipts for the selected rows). Because
   release.yml reads curated notes from the tagged tree, the GA commit may differ from the rc
   tree by EXACTLY the release-notes addition and nothing else — verified mechanically:
   `git diff --name-status rcN..GA` must show ONLY `A` (added) entries under `.github/release-notes/` — modifying or deleting existing notes is not the exception. GA notes =
   rc notes + gauntlet summary + receipt links.

## Tiers (from v0.31.0; patch releases from v0.29.1)

The tiers select which rows run; selected rows keep their existing pass, validity and
carry-forward rules. From v0.31.0, and for patch releases from v0.29.1, this cadence supersedes the every-rc/every-rc1 labels
and the cadence lines under Carry-forward below, and the full-tier cadence in Pipeline step 3. Phase receipts account for the selected
rows and identify rows assigned to M; an unrun selected gating row is never green.

- **R, every release candidate:** Phase B, the CI-required checks, the Phase C live soak
  and the reference-agent window (D5), and the upgrade/rollback roundtrip.
- **T, per release:** the rows behind that release's own gate, named in its milestone
  and `ROADMAP.md`. For a patch release, T is the rows the promotion map selects for its
  diff, plus any row its milestone names. For v0.29.1, D1 (in the shipped LOCAL default) and D4/D8
  are also T rows (D8 reads D4's runs), because the v0.29.0-only decisions below commit them to v0.29.1. D2 runs in
  v0.29.1 only when this selection selects it, and the D2 validity addition those decisions name
  applies to that run.
- **M, weekly on main:** every other row, run on the main tip once a week, with receipts kept.
  Before v0.31.0 there is no weekly M run: a patch release's unselected rows are not run, and
  its release notes list them as not run.

**Promotion map.** Ingest, storage or retry changes promote Phase A, D1 and D4/D8 into
the rc; compaction or summariser changes promote D2/D3 and D7; recall or delivery changes
promote D1 and D4/D8; host-integration changes promote Phase C cells. The release manager
records the promotion decision per rc with the diff-scope proof
(`git diff <prevGA>..rcN --name-only`). From v0.29.2 the row selector below applies the map.

**Row selector (from v0.29.2).** A script applies the promotion map, so the selection does
not depend on judgment:
- **Diff:** `git diff --name-only <previous GA>..<rc tree>`.
- **Path classes:** each changed path takes the classes of the first matching entry in the
  committed class table. Paths under `bench/`, `docs/`, `tests/` and `.github/release-notes/`
  take none. Any other path that matches no entry takes every class, so a gap in the table
  fails closed.
- **Imports:** a changed product module also takes the classes of each product module that
  imports it, directly or through a chain of imports that does not pass through a hub. The
  plugin entry point and the engine are hubs, and a change to a hub takes every memory class.
  So a helper that the store imports counts as a storage change. A module the package loads by
  name (through importlib or a registration table) counts as imported by the module that loads
  it. A product module the selector cannot place in the import graph takes every memory class.
- **Hub closure:** a changed module that a hub reaches, directly or through any chain, and
  whose classes still select no Phase D row, takes every memory class, even when a table entry
  matched it. The exception is a module the table names as exempt, one module at a time,
  with the reason it cannot change ingest, storage, recall, compaction or delivery.
- **Rows:** ingest, storage and retry select Phase A, D1 and D4/D8. Compaction and summariser
  select D2/D3 and D7. Recall and delivery select D1 and D4/D8. Host integration selects the
  Phase C cells. Other classes (for example the CLI, doctor or installer) select no Phase D
  row when the hub closure below does not apply to the module (no hub reaches it, or the table
  exempts it), so a train that touches only such modules runs L1 and the R rows.
- **Record:** the selector prints each changed path's classes with the reason (the matching
  table entry or the import chain) and the selected rows. The output is committed in the
  train's kit at rc1 and at every respin, and quoted in the receipts. The release manager may
  add rows, with a reason. Removing a selected row needs a release-manager decision in the
  table below.
- **Review:** changes to the class table or the selector need an independent review, like
  code. A train records the instrument commit it ran the selector from. Until the selector
  merges, the release manager applies these rules by hand and records the same output.

**Conditional promotions.** When D5 uses its integrity-only path (b), or the patch-only
integrity-only alternative in the release-manager decisions below, D7 and D5-B are promoted
into that rc: those paths gate on latency receipts bound to the same rc, which a weekly M
receipt cannot supply.

**Blockers follow selection.** A row's GA blocker applies when the row is selected for the
rc (R, the release's T rows, a promotion or the full tier). In particular, D2's rule that an
unrun D2 is not green (the #1824 implementation boundary) blocks GA only in a release that
selects D2; otherwise D2 runs in M and its receipt is recorded there.

Tiering must still catch a replay that keeps row counts but loses timestamps or source
attribution, so recall returns the wrong historical state. The ingest→D1/D4 promotion
covers this defect class.

**Full tier:** all phases and rows, at least every third minor and at every default flip.
v0.30.0 runs the full tier. Each rc records elapsed hours and machine-exclusive hours
separately.

## Smoke stage — L1 (from v0.29.2)

Testing runs in four levels:

| Level | When | What | Effect |
|---|---|---|---|
| L0 | every PR | the CI-required checks and the differential tests | as now |
| L1 | every rc, first | the smoke stage below; target 2 h elapsed or less | can stop the rc; its full rows give their receipts |
| L2 | every rc, after L1 passes | the rows the Tiers section selects, with their existing bars | the GA receipts |
| L3 | at its cadence | the full tier and the weekly M rows | as now |

No L2 row starts until L1 passes. In v0.29.x the release-changing defects came from the
cheaper layers, not the long rows. The reliability matrix found rc1's rotation defect, and the
Phase B review sweep and the soak found the others. With L1 first, a broken rc fails in hours
instead of after a day of long rows.

**L1 checks,** run on every rc whatever the selection. Each one that is itself a row gives
that row's receipt when it runs on the rc tree with the row's full bars, and is not run again
in L2; a respin that may carry a receipt (Pipeline step 4) carries it instead. Phase B's
receipt is completed in L2 by the rest of its hands-on lane.
- the reliability matrix and the hands-on lane's fault cells, on one supported host build
  (the other host builds run in L2);
- Phase A, both postures;
- the upgrade/rollback roundtrip;
- D1 OFF in full (the fast tier: deterministic, about 10 minutes);
- the Phase B review sweep, alongside the checks above.

**Mini rows,** only for rows the selection puts in L2:
- **D1 LOCAL:** a fixed 25-question subset of the D1 set, stratified by question type, drawn
  once and listed in the kit. The rc runs it and is compared with the previous GA's stored
  per-question outcomes. Any per-question `lcm_recall` flip that is not attributed to a named
  commit stops the rc. So do timeouts and fallbacks.
- **D4:** a fixed 30-question sample of LongMemEval-S at the shipped default, stratified by
  question type and drawn once, compared with the previous GA's stored per-question outcomes.
  A hard failure stops the rc: a crash, a timeout, a missing or unreadable result, or an
  INVALID verdict under D4's own validity rules. So do net losses above the bound declared from
  the previous GA's A/A′ runs; until a bound is declared, it is 3.
- **Compaction (D2/D3, D7):** one D7 seed on the rc, judged on D7's bars 1, 2, 3, 5 and 6.
  Facts kept is recorded, not judged, because one seed is too noisy for bar 4. A missed bar
  re-runs on a second seed, and the rc stops only if both seeds miss; a hard failure stops it at
  once. A rate-limited or service-invalid run re-runs once under D7's
  and D2's rules and is not, by itself, a product defect.

The kit records each mini row's measured time. A mini row that would push L1 past its target
starts alongside L2 instead, and its failure still stops the rc.

**Rules:**
- L1 bars are written into the train's kit before rc1 is cut and are not changed afterwards.
- A mini row is a tripwire. It gives no receipt and no credit: passing it never stands in
  for its L2 row, and no L2 verdict depends on it.
- An L1 failure stops the rc. A product defect goes through fix-and-respin. An instrument
  defect is fixed, and L1 re-runs on the same rc.
- Each rc records L1's elapsed hours beside the elapsed and machine-exclusive hours the Tiers
  section asks for.
- Checks that judge latency keep the machine to themselves, as elsewhere.

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
mid-key, and the residual backstops then withhold that chunk while dispatching the protected
remainder (report `status: partial`, `privacy_blocked >= 1`, `selected >= 1`) — the battery
accepts that partial shape with a real dispatch (any error still fails), and the raw-secret sweep covers everything
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
A cell compares the affected rows by identity (role, turn and content hash; an assistant row
that carries tool calls adds each call's id, name and arguments; inside one fixture session,
and a multi-session fixture adds the session and, for a tool result, its tool-call id), not
only by count:
the candidate is no worse only when its affected rows are the base's rows or a subset of them.
A pre-existing shape is recorded as KNOWN with a tracked issue and does not force a respin. A
cell where the candidate is worse than the base follows the P0/P1 rule above. A shape the base does not show follows the P0/P1 rule above; in
particular the candidate respins when, against the base, it adds loss, a wedge, a session reset,
a request over the model window, reordering, or a row in the wrong session. A duplicate also
respins on any path without an injected fault. One exception: a duplicate-only shape reachable
only under an injected crash or kill may ship as a disclosed known issue at P2, with a tracked
fix planned for the next patch release, only when all of these conditions hold:

- Only rows written by the turn in flight at the crash are duplicated. Each row appears at
  most twice. The count of duplicated rows and the number of appearances of each go into the receipt.
- A crash–recover–crash probe adds no further duplicates. Its second injected crash lands in the same session during another in-flight turn, then the session recovers.
- Message identity stays unambiguous. A tool call and its results are duplicated only together,
  as a complete group. The context assembled after recovery has no orphaned or doubled tool result.
- Every base row remains present, in order, in its session.
- A reviewer from a different model than the author of the shape's classification verifies
  these four conditions. The Phase B receipt names that reviewer.

If any condition fails, the normal P0/P1 rule applies and the rc respins. The probe, its rule
and the base and candidate counts go into the Phase B receipt.

## Phase C — Live-session soak

A scripted multi-turn session battery over `hermes acp` (the measured headless single-session
transport) against the clone: ingest-heavy turns, recall probes, compaction crossing at least
one threshold, doctor at close. Minimum 30 turns. Green =

- at least one compaction committed during the soak. A soak in which every compaction stopped
  (on a handled provider failure or otherwise) proves little: it is inconclusive and re-runs;
- lossless, as the Phase C scorer (`bench/instruments/reliability/scorers/`) checks it; its
  code is the exact rule. Sent prompts and result records must have equal counts; each recorded
  `input_sha256` must match the aligned prompt's unstripped UTF-8 bytes. A mismatch is INCONCLUSIVE.
  The first non-empty transcript item must identify one owning `conversation_id`, or the bar is
  INCONCLUSIVE. Only that conversation's rows, across all its sessions (including rotation children),
  may satisfy the transcript; foreign conversations are counted separately, and their transcript-key
  rows fail as surplus. Other foreign keys stay report-only. Non-empty user and
  assistant rows are compared as a multiset: user edges are stripped as ACP does, assistant bytes
  are exact; Hermes strips assistant content on store, so drivers must also strip `raw_answer`.
  Missing copies, surplus copies and unexpected stored rows fail. Phase C alone adopts the release
  multiset-v2 split predicate (r2, sha256 8943a6a7…) without its own-turn prompt rule; the store-order
  check below fails the borrowed-turn case that rule guarded. A missing, unique assistant item may match exactly
  one run of 2–8 consecutive non-empty assistant rows in one session, joined with a space or directly
  under v2's NFC, CRLF and whitespace-run normalization. Tool rows may intervene; every user row,
  even empty, across any owned session is a turn boundary. No fragment key may be a transcript item; no row may serve two answers;
  extra copies of used fragments fail. Whole-row absence, uniqueness and fragment ownership use
  v2-normalized keys for this split exception only. The transcript must also be a subsequence of
  store-id-ordered rows (accepted fragments represent one answer); extras may intervene but retain
  their multiset checks. The first out-of-order transcript index and role are reported as FAIL.
  Rows empty after stripping are excluded; tool-result rows, tool-call ids, names and arguments
  are not part of this bar (#710);
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

## Phase D — Release-over-release quality (the release scorecard, #898 / #664)

Phases A-C prove that a candidate does not break. Phase D asks whether its compactions and recall
got worse for the agent than the previous GA's, on the same material, seeds, question sets, reader,
judge and summariser. Every row names its configuration (`OFF`, `LOCAL`, `VOYAGE-TRACKER`; #898). A row
that did not run says `UNRUN` or `UNAVAILABLE`; it is never substituted. A gating row that is `UNRUN`,
`UNAVAILABLE`, `INCOMPLETE` or `INCONCLUSIVE` is not green. From the release that
flips the shipped default to `LOCAL` (#1186), `LOCAL` gates D1, D4 and D8, with
`OFF` recorded beside it. Construct the default arm from each measured tree, and
record its effective provider/model (including the provider-scoped default), not
just its empty configured model. A LOCAL quality row that silently uses full text
is unhealthy, even when that fallback is correct runtime behavior. Its own LOCAL
A/A′ prerequisite is mandatory; an OFF placebo cannot qualify LOCAL.

### Maximum-configuration gate (from v0.30.0)

**Activation:** from v0.30.0 rc1, every score-sensitive change named in the release's
gate (today #1607, #1589 and #1590) also runs on the cloud-embedding (Voyage)
arm of the #1963 ablation. This promotes that recorded arm to a gate under D6.
**Prerequisite:** the D8 scorer must first classify the cloud arm's recall health, using D1's
`LOCAL` definition applied to the cloud provider's semantic arms (today it accepts only `OFF`
and `LOCAL`, so a cloud arm reads as missing and the comparison is `INCOMPLETE`). Until that
scorer support merges with its tests (#1995), the arm stays recorded in D6 and the release
receipt says so; the gate starts at the first rc cut after it merges.
**Pass rule:** it must pass D8's existing non-inferiority rule there: in each named
category, the lower 95% bound of the paired candidate-minus-comparison difference is
strictly above −2 points, using D8's same cluster bootstrap and seed 1607. D4's QA
rule and the arm's own valid D4 A/A′ placebo in the same session are required, as are D8's per-category
placebo, recall-health and degraded-rate rules. A non-significant loss is not
non-inferior. A win on the default configuration alone is recorded as an experiment,
not adopted. Agentic delivery (#1188) is outside D8's single-shot scope, so its
cloud-embedding arm stays recorded in D6 until this spec gives agentic delivery a pass rule.
Latency and cost claims for a default continue to use RFM-v1.

### Reference-machine measurement fixture — RFM-v1 (#1186)

This is the single fixture for every latency or cost row that claims the
[VISION reference machine](../../VISION.md): Apple M3 or newer, 16 GB RAM.
It applies to reference-machine claims throughout this spec; field measurements
keep their declared host and are not relabelled as reference-machine results.
Pin `RFM-v1`, the spec commit SHA, hardware/OS, provider and model revisions,
backend, effective configuration and all limits in each measurement receipt.

- **CPU:** at most 8 CPU threads available to the measured workload, including
  child processes. Record the affinity/quota or harness enforcement and observed
  concurrency; thread environment variables alone do not establish a CPU quota.
- **Math libraries:** launch with `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `VECLIB_MAXIMUM_THREADS=1`, `OMP_NUM_THREADS=1` and `NUMEXPR_NUM_THREADS=1`.
  Record the loaded BLAS implementation and effective settings. ONNX Runtime
  uses one intra-op thread for bge-small and two for other FastEmbed models, as
  shipped; record its effective intra/inter-op settings and execution provider.
  Keep aggregate workload concurrency within the 8-thread envelope.
- **Apple GPU:** use and record the supported Apple GPU execution path when the
  selected provider exposes one, including backend/version and actual device.
  A CPU-only provider (including the shipped FastEmbed path) stays on CPU; record
  GPU support as unavailable, never silently substitute a different provider.
- **Model and vectors:** budget approximately 4 GB combined for the model and
  resident vector working set, within the 16 GB machine. Record model/cache
  bytes, resident vector bytes, process peak memory and corpus size separately;
  a cache-size measurement alone is not resident-memory proof. Disclose any
  over-budget result rather than claiming it meets this fixture.
- **Larger hosts:** enforce the same CPU envelope and model/vector budget, and
  a 16 GB memory limit for the measured workload; record the mechanism and its
  verification, chip/GPU differences and machine load. If a limit cannot be
  enforced, label the result a larger-host diagnostic, not RFM-v1 evidence.
  Run latency cells in isolation except for the explicit overlap rule below.

The fixture changes no pass thresholds or producers in other rows. A changed
fixture requires new comparable measurements; it cannot restamp old results.

| gate | workload | producing unit | denominator | pass rule |
|---|---|---|---|---|
| D1 recall flips (fast tier, every rc) | LongMemEval-M retrieval, in the configuration each candidate tree ships by default, chosen per candidate. When that default is `LOCAL` (#1186), `OFF` is recorded beside it. The receipt records the default read from that candidate tree. An RC that ships `LOCAL` by default is accepted only after that row has its own A/A′ in the same session with 0 discordant rows (D6). QA rows use D4's placebo rule instead | the recall runner + the per-question identity projection (`bench/tools/v1m-rebank/`) | 500 questions, per-question `lcm_recall`; every question stays in the denominator | Two conditions, both required. (1) Flips against the previous GA, re-scored under the current instrument: the row passes with 0 flips; if there are flips, it passes only when every flip is attributed to a named commit and classified intended, net R@10 ≥ previous GA − 0.005, and no category is down more than 0.02. A defect-class loss blocks either way. (2) Every executed recall's health matches its configuration's healthy status (abstention rows run no recall, carry no health and are exempt): no timeout, and the full-text arm's coverage is `ok`. In `OFF`, `degraded: true` from the absent semantic arm with `coverage.fts: ok` is the expected healthy status (RUN-SHEET-ROW1-FULLTEXT); any other degraded reason is unhealthy. In `LOCAL` (#1612), healthy means all of the following. `provenance.coverage` has a key for every semantic arm the request schedules (the default `include`, which D1 and D4 use, schedules both `summary` and `chunk`), and every semantic key is `full` or `full_approx` (the binary prescreen reaches the whole corpus). Every `full_approx` key has its matching disclosure in `degraded_reason` (`<arm> arm coverage full_approx: …`, so `degraded` is true), and every degraded reason is the disclosure of a scheduled semantic arm whose recorded coverage is `full_approx`. `degraded` is true exactly when at least one scheduled semantic arm's coverage is `full_approx`, so `degraded: true` with every semantic key `full` and no disclosure is unhealthy, and so is a disclosure for an arm whose coverage is `full`. Unhealthy: `bounded`, `none` or `disabled` coverage; a scheduled semantic arm without a key; a `full_approx` key without its disclosure; a `full_approx` disclosure for an arm whose recorded coverage is not `full_approx`; or any other reason (provider unavailable or preparing, a stale embedding identity, a failed query embedding). A candidate whose semantic arm silently falls back to full text is therefore never healthy. Read coverage, not `provenance.arms_run`, which lists only the arms that returned hits. D4 and D8 read recall health with these same definitions. Health is checked on its own because the identity projection carries only category, abstention and metrics. An unhealthy question is re-run once; still unhealthy, it counts as a loss. |
| D2 facts kept (full tier) | Track S v3 material, fixed seeds 1–3 × 9 independent replays per seed per arm, checkpoints 176 and 304 (54 planned replays) | `bench/instruments/compaction_probe/track_s/` (producer/scorer support for this design is a separate implementation gate; see the implementation boundary below and #1824; #1620 (D7 two-run fallback) does not cover it) | the run, stratified by seed; a run's kept share is kept / scored facts. Facts sharing compactions are correlated and are not independent experimental units | Calibrated non-inferiority (option B): at cp-304, PASS iff the one-sided 90% stratified pooled Student-t lower confidence bound of candidate-minus-comparison mean run share is strictly above −0.05. Completeness, validity, reader-placebo and prospective A/A prerequisites below also apply. INCONCLUSIVE is not green. The bare “mean ≥ previous GA” bar, absolute spread veto and spread-triggered D7 substitution are retired prospectively (#1538, #1554); D7's own six bars are unchanged. |
| D3 Track S absolutes | the D2 runs | the same scorer | per run | 0 admission-proven losses (a planted fact missing from the store at a checkpoint). Level-3 nodes follow Phase C's rule. |
| D4 QA rows (full tier) | LongMemEval-S (500) and LoCoMo (1,986), through the pinned benchmark bridge, in the configuration each candidate tree ships by default, chosen per candidate. When that default is `LOCAL` (#1186), `OFF` is recorded beside it. The receipt records the default read from that candidate tree. An RC that ships `LOCAL` by default is accepted only after that row has its own A/A′ in the same session satisfying D4's placebo rule (D6) | the QA runner + its fail-closed scorer | per-question pairs, candidate vs previous GA | Blocks when the paired exact McNemar gives p < 0.05 with a net loss. **Gain gates:** a gain claim requires a positive paired improvement (exact McNemar p < 0.05 with a net gain, and a valid D4 A/A′ placebo) plus the issue's own absolute bar. An unchanged answer set never passes a gain gate. BEAM rows with n ≤ 40 per ability are tracking-only unless they meet this predicate. For a fractional per-question score (BEAM's `nugget_score`), McNemar does not apply: the paired test is a two-sided sign-flip permutation test on the per-question candidate-minus-comparison score differences (10,000 resamples, seed 1607), and a gain needs p < 0.05 with a positive mean difference; its A/A′ placebo is valid only when the same test on A versus A′ gives p ≥ 0.05. The A/A′ placebo is the candidate's rerun: the 100-question subset for LongMemEval-S, the full set for LoCoMo; whenever D8 runs, LongMemEval-S A′ covers all 500 questions as D8 requires. The placebo is accepted only when at most 5% of A/A′ pairs are discordant AND the A/A′ pair does not itself meet the blocking predicate (exact McNemar p < 0.05 with a net change in either direction). Otherwise the row re-runs once (A and A′, fresh stores) and the re-run replaces it; a second rejected placebo makes the row `INCONCLUSIVE`. Failed, missing and judge-fallback questions count against the candidate. From v0.29.0 rc1 (whose D4 runs record each question's recall health, #1609), recall health follows D1's condition (2), with D1's definitions for the row's configuration: every executed recall must be healthy (abstention rows are exempt). An unhealthy question is re-run once. The re-run replaces the first attempt, both for scoring and for the recall-health record D8 reads: a healthy re-run is scored normally, and a question still unhealthy after its re-run counts against the candidate. A row in which more than 1% of its questions remain unhealthy after their re-run is `INCONCLUSIVE`, which is not green, and the receipt names the unhealthy reasons. A bounded or missing semantic coverage, a stale embedding identity or a silent full-text fallback therefore never earns credit in D4, and cannot leave a D4 row green when it is widespread. |
| D5 latency | the gauntlet cells; the reference-agent acceptance window on the exact rc tree (normally ≥ 24 h; integrity-only option below; not Phase C's scripted battery) | the field-window counter | per compaction event; at least 10 events unless the integrity-only path below is met | Gauntlet cells: max ≤ 145 s, with p50/p90 recorded. Soak (from v0.27.0-rc3, #1123/#1128): max ≤ 120 s and p90 at or below one baseline p90, over at least 10 compaction events. Event times are LCM-X-owned (from v0.27.0-rc5, #1155): a session LCM-X bypasses hands compaction to the host's built-in compressor, the stop line books that time as `host=`, and the soak measures `elapsed − host`; host time is reported (count, p50, max, sources), never gated, and never added to a per-message wait. The rc4 soak failed its max bar on host time (124.5 s) and stands as recorded. The baseline is the previous GA's p90 on the same summary route (provider, served model, effort): its own D5 soak on that route when it has one, else the managed fleet's field p90 for that release and route; ≤ 60 s when neither exists. The soak's pre-declaration names the baseline (release, route, source, window, event count, p90) before the window starts, and it is not changed afterwards; plus the per-message wait bars on the same window: at most 25 compactions per 100 user messages after the first threshold crossing, mean wait ≤ 15 s per message and at most 25% of messages waiting. The fixed 60 s p90 bar it replaces failed the previous GA itself on the slow route the managed fleet uses (v0.26.2: p90 86.9 s). Percentiles are nearest-rank: p90 is the ⌈0.9·n⌉-th smallest event time (the rule the Track S scorer uses), and the receipt names the method. If the organic window has fewer than 10 events, either path may be used: (a) a disclosed synthetic session on the reference agent supplies the rest (the v0.26.0 pattern), and the receipt says how many events each source gave; or (b) an integrity-only window with engine registration on the reference agent, zero LCM-frame tracebacks and publication-invariant lines, a clean doctor, completion of every organic turn, 0 lost rows, 0 duplicate rows and 0 LCM-X tracebacks, lasting at least 24 h for a major or minor release and at least 8 h for a patch release. Path (b) also requires at least one completed organic turn and at least one organic threshold crossing whose compaction was published, both inside the window on the exact rc tree. If the window ends without both, it extends in 8 h steps, to at most 72 h in total. If both have still not happened, D5 is `INCONCLUSIVE`, which is not green. On path (b), latency gates on Phase C and the other gauntlet cells (max ≤ 145 s, as above), and on D7's latency bars. The receipt names the path and records the organic turn count and the published-compaction count. D5 is `INCONCLUSIVE` when neither path meets all its requirements, including the activity requirements for path (b) (#1554). Track S event times are recorded but not judged: they are bound by the summariser lane's latency, so Track S runs off the provider's peak hours. |
| D6 recorded, not gating | the non-default configuration (`LOCAL` or `OFF`, chosen per candidate) and `VOYAGE-TRACKER` recall and QA rows; the continuity suite (#659); the external arms (Codex native, lossless-claw, Hermes built-in, Claude Code) | as each row's kit | per row | Recorded in the receipt. The candidate's shipped default gates under D1 and D4. Any other row becomes a gate only through a change to this spec that names its pass rule (D1's for a recall row, D4's for a QA row) and the release it starts at. A newly gated row must first have its own A/A′ in the same session: 0 discordant rows for a recall row, and D4's placebo rule for a QA row. The continuity suite's per-class bars are declared the same way, from its v0.27.0 recorded run, before v0.28.0 rc1 (#659). External arms run on demand as H1 evidence. |
| D7 per-message wait (replay cell, every rc1; from v0.28.0, #1128) | Track S v3 material, seeds 1–3, one run each, on the managed fleet's summary route (provider, served model, effort; v0.27.0: gpt-6.1-sol, default tier, effort low) with the fleet policy arm (`LCMX-fleet-rc3`: the fleet policy on the product's leaf-chunk, leaf-target and timeout defaults) and `--turn-end-notify`, so turn-paced holds end at each user-turn end as on a live host | `bench/instruments/compaction_probe/track_s/decision/run_pmw_cell.sh` (`s2/run_s_lcmx.py` → `decision/score_decision.py` → `decision/pmw_read.py`) | per run, the user turns from that run's first threshold crossing to its stop row. A compaction counts when its own wall is ≥ 1 s. A turn's wait is the sum of all its compaction walls, sub-second ones included, and the turn waits when that sum is ≥ 1 s; so two 0.6 s compactions in one turn make a waiting turn and no counted compaction. Compactions per 100 turns pool the counted compactions and the turns of all runs. Mean wait (over every post-crossing turn), waited share and cp-304 facts are the mean of the per-run figures. Wall p90 pools every counted compaction | Six bars, all required: (1) at most 25 compactions per 100 post-crossing user turns; (2) mean wait ≤ 15 s per turn; (3) at most 25% of turns waiting; (4) facts kept at cp-304, mean over the runs, at least the previous GA's D7 mean minus 0.10 (v0.28.0: 0.872 − 0.10 = 0.772), declared to the runner as `FACTS_FLOOR`, which it requires; (5) 0 rows dropped by survival fits; (6) compaction wall p90 ≤ 120 s (nearest-rank). `pmw_read.py` exits 0 on PASS, 1 on FAIL and 2 on INVALID. INVALID covers: a missing run; a run that did not finish, or that a rate limit stopped; a missing cp-304 score; an unreadable input; or a run whose recorded config differs from the cell's declared leaf chunk, leaf target maximum or summary timeout. INVALID is not green. cp-176, tokens freed per wait, call failures and survival-fit counts are reported only. The Phase 1 target (p90 per-turn wait ≤ 5 s) is recorded, not gated, until invisible compaction ships. A rate limit or usage-limit marker in a run stops the cell, and it re-runs; it is never scored. A cell run with another arm or seed set, or with an injected summariser fault (`FAULT`, cell 1096-H), is recorded as non-gating in `cell.json`, and the receipt records the kit commit. v0.27.0 baseline (rc8): 20.0 per 100, mean 8.1 s, share 20%, cp-304 0.872, 0 dropped, wall p90 45.8 s. |
| D8 delivery guards (recorded from v0.29.0 rc1; gates from the #1186 default-flip release, or v0.30.0 rc1 if earlier; #1607, #1591) | the D4 runs (LongMemEval-S 500 and LoCoMo 1,986, in D4's configuration with `OFF` recorded beside `LOCAL` when LOCAL is the shipped default, through D4's single-shot benchmark bridge), read per category: LongMemEval abstention (the fixed set of questions whose `question_id` ends in `_abs`), LongMemEval `knowledge-update`, and LoCoMo adversarial (the scorer's category 5). This row covers single-shot delivery only. Agentic delivery (the model driving `lcm_recall`, `lcm_grep` and `lcm_expand`) is guarded at its own change's gate (#1188, under the frontier-model protocol, #1189) and is recorded in D6 until this spec gives it a pass rule | the D4 QA runner and scorer, read per category; each question's recall health as the QA results record it (#1609) | per-question pairs within each named category, candidate vs previous GA. The degraded rate is the share of all the run's questions whose recorded recall health is unhealthy under D1's definition (in `OFF`, `degraded: true` from the absent semantic arm with `coverage.fts: ok` is healthy) or timed out; failed questions count as degraded. A run whose records lack recall health makes the row `INCOMPLETE` | **Pass rule:** non-inferiority. In each named category, the lower 95% bound of the paired difference must be above −2 points (cluster bootstrap by conversation for LoCoMo, by question for LongMemEval). **Interval:** a percentile bootstrap with 10,000 resamples of clusters with replacement and seed 1607; the bound is the 2.5th percentile of the paired difference in points, and a bound at exactly −2.00 fails. The D4 scorer implements it with #1609. The degraded rate may be at most the previous GA's rate plus 1 point. **Placebo:** D4's A/A′ rerun, read per category. In a release that records or gates D8, D4's LongMemEval-S A′ runs on all 500 questions, because the 100-question subset is too small per category; the receipt records each category's count. The row is valid only when each category's A/A′ difference has a 95% interval inside ±2 points. Otherwise it re-runs once with D4, and a second failure makes the row `INCONCLUSIVE`, which is not green. **Delivery under test:** D4 and D8 run the bridge with the candidate tree's default delivery. From the release that ships #1607's token budget, the bridge passes no hit cap smaller than the product default and does not truncate the response; that bridge change ships with #1607. **Configuration, re-runs and carry-forward** follow D4 exactly. D8's recall configuration is D4's, chosen from each candidate tree's shipped default, with D4's A/A′ prerequisite. D8 re-runs or carries forward whenever D4 does. **Activation:** recorded at v0.29.0 rc1, with its own A/A′ in the same session (D6). Gates in the release that flips the shipped default to LOCAL (#1186), even if that is v0.29.0; otherwise gates from v0.30.0 rc1. It uses whichever configuration D4 then ships, with OFF recorded alongside LOCAL. **v0.29.0 delivery changes** are judged before they merge, at their own gate: the acceptance in the change's issue, with these margins on its held-out split, and the receipt posted on that issue. That acceptance also defines the per-change absolute bars and their producers (knowledge-update, HaluMem memory boundary, recall latency; #1607). They are not part of this row. |
| LongMemEval-S single-session-preference no-loss (#1186; recorded at the default-flip release, gating from the next minor) | all `single-session-preference` questions in LongMemEval-S, candidate vs previous GA, each in its own shipped configuration | the D4 QA runner and fail-closed scorer, read per category; D4's LongMemEval-S A/A′, using the full-500 A′ whenever D8 runs | per-question pairs over every preference question; record the complete category count, candidate wins, candidate losses and category A/A′ discordant pairs | PASS iff **candidate wins − candidate losses ≥ −min(2, max(1, category A/A′ discordant pairs))**. A win is candidate-correct/previous-GA-wrong; a loss is the converse. Failed, missing and judge-fallback questions count against the candidate under D4. D4's health, placebo, comparability and carry-forward rules apply; LOCAL must satisfy its own A/A′ prerequisite, with OFF recorded alongside. Record the raw verdict in the default-flip release; this row becomes a gate at the following minor release. |

### D2 calibrated facts gate (#1538 / #1554)

These rules replace the D2 mean/spread rulings from #1617 for future comparisons; they do
not re-score or rewrite any past raw verdict. The comparison arm is the previous GA.

- **Design and pins.** Fixed seeds 1–3, 9 independent replays per seed per arm: exactly
  54 planned replays, with both cp-176 and cp-304 complete. Interleave comparison and
  candidate in one session, instrument and route. Pin material/fact identities, builds,
  instrument, summary provider/model/effort/tier, reader, policy and retry behavior before
  starting. Changed instruments or routes need a comparability ruling; a shared metric name
  is insufficient. Do not pool D7's different summary route into D2's variance estimate.
- **Statistic.** At cp-304, take the mean over seeds of candidate mean run share minus
  comparison mean run share. Estimate pooled variance after centering each arm×seed group;
  SE = pooled SD × sqrt(2/(seeds×runs)), degrees of freedom = 2×seeds×(runs−1).
  Order: a summariser timeout or failure shown to be caused or made more likely by the candidate (by a named candidate commit or a reproduction recorded in the receipt) is FAIL; otherwise an incomplete comparison, one over the replacement budget, or one past the elapsed stop is INCONCLUSIVE; otherwise a tripped cp-176 placebo is INCONCLUSIVE; otherwise FAIL unless the one-sided 90% lower bound is above −0.05; otherwise, in a gated comparison, a candidate arm with at least 2 more service-invalid attempts than the comparison arm (every attempt counted, replacements included) is INCONCLUSIVE; otherwise PASS.
- **Reader placebo.** On the full fixed run set at cp-176, two-sided stratified pooled-t
  p<0.05 AND absolute mean difference≥0.05 makes D2 INCONCLUSIVE. No second look or
  result-dependent continuation is allowed. The old exact 2+2 enumeration is not this test.
- **Diagnostics only.** Report the corresponding one-sided 90% upper bound and whether it
  is below 0 (positive loss evidence), the seed-stratified run-level permutation p, the
  per-occurrence McNemar and each arm's per-seed spreads alongside the verdict. Spread is
  included in the variance estimate, with no absolute veto. A single replay outside the
  instrument's validity checks is invalid, not a low score. Admission loss still fails D3.
- **Completeness and replacements.** Every planned replay stays in the completeness
  denominator. A failed, partial, rate-limited, missing or truncated replay never earns a
  score. From v0.29.1's D2 runs, the prospective A/A included, a replay with a summariser call
  attempt that timed out or failed (retries and fallback attempts included) is service-invalid: it
  never earns a D2 score and is an invalid replay for the replacement rule that follows; D3's
  admission check still applies to it. The receipt lists each service-invalid replay (arm, seed,
  replay index, the summariser events and the rows its survival fit dropped); those rows are
  reported, not scored, and the product behaviour is tracked in #2004. Service invalidation never
  waives a release-caused defect: the verdict order above makes a candidate-caused summariser
  failure FAIL and a candidate-arm excess of service-invalid attempts INCONCLUSIVE. Keep both attempts; allow at most one replacement per invalid replay that is not rate-limited (a rate-limit
   marker triggers the stop below) under
  unchanged pins, and at most 2 replacement attempts across the whole comparison.
  Never rerun a valid low score. If the fixed set cannot complete within budget, return
  INCONCLUSIVE; INCONCLUSIVE is not green.
- **Stop budget.** Exactly 54 planned replays; at most 2 concurrent, one per arm. Reserve
  6 hours ELAPSED including setup and scoring: wall clock, not aggregate replay time.
  At 6 hours elapsed (setup and scoring included) the comparison stops. Any replay or scoring not finished by then makes D2 INCONCLUSIVE. A stop on a usage or rate-limit marker before the fixed set is complete is also INCONCLUSIVE.
  Stop on the first real usage or rate-limit marker; do not probe repeatedly or use a
  replacement to bypass that stop. No result-dependent continuation and no second look.
  An INCONCLUSIVE D2 may be re-run once, as a new full 54-replay comparison under a new reservation; the earlier raw verdict stays in the receipt. A FAIL is never re-run.
- **Prospective A/A.** Before this rule first gates a release, run ONE comparison of the
  current GA against itself with the same design (54 replays). Record A/A acceptance and
  the cp-176 placebo. Until the producer and scorer (#1824) exist and, for option B, the prospective A/A has passed, D2 is UNRUN and not green: no release reaches GA on D2. Option A, used only on a recorded funding refusal, needs #1824 but not the A/A. A failed A/A is reported and the release manager records a new ruling before D2 can be green.
  Arm-to-slot assignment is counterbalanced: within each seed, odd replays run the candidate label in concurrency slot 1 and even replays in slot 2 (the same in the A/A and in every gated comparison). The A/A passes only if the one-sided 90% lower bound is above −0.05 AND the point estimate is below +0.05; a material difference in either direction fails the A/A.
- **Claim.** A complete passing comparison supports: “non-inferior by the predeclared
  5-point margin on these fixed materials and routes at one-sided 90% confidence.” It
  does not establish improvement, no loss at all, a five-point loss positively detected,
  unseen-seed performance, other summariser routes, all release gates or customer readiness.

Calibration uses 24 complete cp-304 observations from builds
`709099b3a397a7893a7abae8802ebe7bf605f79e` and
`54e8d21d891271d84623e89087f87dd0eb617271`, instrument
`1e5b89f5f387d0e3984fe4bf0b7effba4aec1743`; see the [comparison receipt][rc3-d].
The independent Gaussian replay model retains historical batch variation and fixed seed
strata; it is a planning approximation, not a distribution-free or empirical coverage guarantee.

| Calibration quantity | Value / interpretation |
|---|---|
| Pooled within-build/seed SD | 0.067729 |
| One-sided 95% upper SD | 0.093771 |
| A/A planning target / modeled acceptance | at least 90% / 92.04% |
| Non-acceptance target / modeled rate at a true 5-point loss | at least 80% / 90.00%; non-acceptance does not prove loss |
| A/A acceptance at the upper-SD estimate | 74.56% |
| Serial mean / two-arm mean replay-hours | 5.54 / 2.77 |
| Two-arm p90 planning allowance, replay-hours | 3.14; excludes setup, scoring and replacements; not a percentile of total elapsed time |

Rates are conditional on complete valid runs; incomplete comparisons are INCONCLUSIVE and lower realised acceptance.
The +0.05 point-estimate cap changes the modeled A/A acceptance by under 0.5 points (SE ≈ 0.018 at the pooled SD).

The 90% A/A target is a **point-estimate planning target**; at the upper-SD estimate it is
74.56%. Only the prospective A/A supplies a measured check. Historical spread-triggered
replacement selection and the small fixed-material sample do not establish stable tails or
provider throughput. Reserve capacity before execution; zero marginal charge proves no capacity.

**Option-A contingency.** Only when the release manager records that the 54-replay reservation
cannot be funded: fixed 3 seeds × 4 replays per arm (24 total), using the two-sided seed-stratified
run-level permutation regression screen, p<0.05 AND loss≥0.05 at cp-304. Label it a
**coarse regression screen, never non-inferiority**. Keep the same complete-run, cp-176
placebo, replacement, concurrency and 6-hour elapsed stop rules. The permutation producer
must support all within-seed 4+4 allocations, not reuse the 2+2 enumerator. This contingency
has no new power claim and is not equivalent to option B.
Raw verdict: FAIL when p<0.05 AND loss ≥ 0.05 at cp-304; otherwise PASS (a coarse regression screen, never non-inferiority). The completeness, placebo, budget and elapsed INCONCLUSIVE rules apply. The prospective A/A prerequisite does not apply to option A. Option A also needs #1824 (the current reader supports only two runs per seed).

**Implementation boundary.** This is a release-spec rule only. Nine-replay production,
completeness accounting, pooled-t/placebo scoring and generalized permutation diagnostics
need separate producer/scorer support. The existing #1620 D7 producer work is separate;
a two-run producer or the existing 2+2 D2 analyzer cannot certify this nine-replay design.
Until the producer and scorer (#1824) exist and, for option B, the prospective A/A has passed, D2 is UNRUN and not green: no release reaches GA on D2. Option A, used only on a recorded funding refusal, needs #1824 but not the A/A. A failed A/A is reported and the release manager records a new ruling before D2 can be green.

**Comparability.**
- A candidate row is judged only against a previous-GA row with the same:
  - material or dataset sha, seeds, and fact or question ids;
  - reader and judge (model and effort);
  - summariser, for D2 and D3 (provider, served model, effort and route);
  - scoring path.
- A scorer change re-scores the stored runs of BOTH trees under the current scorer before the comparison.
- A harness change keeps the previous row comparable only when the receipt shows that it cannot change any scored row or denominator of the previous run. Examples: it only adds diagnostic text, or it only changes a failure path the previous run never took (failed, missing and fallback counts all 0). Any other harness change re-runs the previous GA row under the changed harness. That includes a retry that fills answers and a changed exclusion.

**Carry-forward.** Phase D evidence binds the exact candidate tree and the exact Phase D kit: scorer, material or dataset, reader, judge, summariser, benchmark bridge and harness.
- D1 runs on every rc, patch rcs included, in that candidate tree's shipped default configuration.
- D2-D4 and D7 run at the rc1 of every release: major, minor or patch. D8 runs at the rc1 of every release from v0.29.0 on (its activation row above); v0.28.x releases do not run it. D7 follows D2-D4's carry-forward rule below, with its kit (the Track S kit, `run_pmw_cell.sh`, `pmw_read.py`) counted as Phase D kit. D8 reads D4's runs and follows D4.
- A later rc of the same release may carry D2-D4, D7 and D8 forward only when all of these hold:
  - its diff from the measured rc touches nothing outside `docs/`, `tests/`, `.github/release-notes/`, `CHANGELOG.md` and the parts of `bench/` that are not in the Phase D kit;
  - the Phase D kit is unchanged;
  - the candidate ships the same default recall configuration as the measured rc, and that row has met its A/A′ prerequisite.

  The receipt records that diff (`git diff --name-only`). Any other change, including a default recall configuration change, invalidates prior carry-forward and re-runs D2-D4, D7 and D8 on the new rc, with the new configuration's own A/A′ prerequisite. The preference no-loss row follows D4 and is recorded or gated according to its activation rule.
- D5's soak is the reference-agent acceptance window on the rc tree, with the duration and path defined in D5, not Phase C's scripted battery. A later rc of the same release may carry it forward under the same diff-scope rule as D2-D4 (no product change and an unchanged field counter), with the diff in the receipt; otherwise a new window runs on the new rc.

**Wall clock.** The target is ≤ 60 min per rc for D1. D2 and D4 are hours-long. Until v0.29.2 they run in parallel with Phases A-C; from v0.29.2 they start after the smoke stage passes and run in parallel with the other L2 rows:
- D2: a 6-hour ELAPSED reservation including setup and scoring, at most two replays in parallel (one per arm), under the fixed stop rules above;
- D4: about 3-5 h per configuration. Arms may run in parallel only within one concurrency envelope: the same machine, provider lane and tier, and worker count, with interleaved or simultaneous starts. A stopped parallel arm resumes only while its peer is still running in the same envelope. Otherwise both arms re-run one after the other in isolation. If they cannot share that envelope, the arms run one after the other in isolation. In serial runs, an arm stopped by a transport error may resume under the same run id and pins. Models, prompts, trees, harness, scoring and pins must stay identical. The receipt logs the resumed question ids; no question is scored twice. The receipt records the concurrency envelope and machine load. Rows that judge latency still get the machine to themselves (#1554), except under the overlap row in the release-manager decisions.

## Release-manager decisions — raw verdict and disposition (#1554)

Every gate-result row records its **RAW verdict** (PASS / FAIL / INCONCLUSIVE / UNRUN / UNAVAILABLE / INCOMPLETE), with the meanings the Phase D contract already gives them; only PASS is green. Never relabel UNRUN/UNAVAILABLE/INCOMPLETE as INCONCLUSIVE. It records any underlying producer status and a **separate disposition column** naming the release manager's
decision and exact rule reference (or “none” when there is no disposition). A disposition without a named rule reference is invalid and never changes whether the row is green. A disposition never rewrites the raw verdict.
A non-green raw row is never presented as an unqualified gate PASS. In particular, there is
no blanket “mean passed, spread waived” disposition. Past receipts and release decisions stand
as recorded; these prospective rules grant no retroactive PASS.

The rows not marked "v0.29.0 only" or "v0.29.1 only" are adopted as of 2026-10-22 and apply from the next release candidate.
The evidence below binds candidate `54e8d21d891271d84623e89087f87dd0eb617271` versus
comparison `709099b3a397a7893a7abae8802ebe7bf605f79e`; it is rationale, not proof that a
future candidate meets the proposed conditions. #1617's D4 scheduling/resume and injected-crash
duplicate rules remain in force; D2 is superseded by the calibrated rule above. The seven rows marked "v0.29.0 only" were adopted on 2026-10-28. They bind `v0.29.0-rc2` (`4b94600853e7f9e9a99a28bfe3275187417e9dc5`) against v0.28.1 (`77b2b4f795fb6a4f84c55abf6d0441da4469971e`), apply to v0.29.0 alone, and grant no PASS. The two rows marked "v0.29.1 only" were adopted on 2026-10-28 (Phase C) and 2026-10-29 (D5-B). They bind `v0.29.1-rc2` (`0eb6e8736ac0361341b3f29ebfb8e30e5b4dabc5`) against v0.29.0 (`c13dea3783bdbd1aab52d83dff5d009bfade0bd2`), apply to v0.29.1 alone, and grant no PASS.

| Deviation | Proposed rule | Exact limit | Evidence link | Status |
|---|---|---|---|---|
| D5 reference-agent window without published organic compaction | Adopt a patch-only integrity-only alternative to the activity clause; otherwise require the disclosed synthetic-session top-up. Retain exact-rc registration, doctor and organic-turn checks and path (b)'s zero LCM-frame traceback and publication-invariant checks. | At least 8 h, 0 lost, 0 duplicate rows, 0 LCM-X tracebacks (host-caught traces follow the lifecycle/trace row), every organic turn complete and at least one completed organic turn. Name a passing latency receipt per controlled surface: Phase C and D7 (path (b)'s existing bars) and the controlled clone cell on the reference agent's effective settings (D5-B; max ≤ 145 s), each bound to the same rc and declared settings. D2 wait reads are recorded, not judged. No waiver for existing duplicate or traceback shapes under this alternative; major/minor duration and activity bars remain unchanged. | [window][rc3-window], [per-surface latency][rc3-d] | ADOPTED (release manager, 2026-10-22) |
| Synthetic top-up cannot call the reference provider in the selected host mode | Adopt a disclosed test-profile top-up using the reference LCM-X settings and a provider compatible with that host mode, if the integrity-only alternative is not met. | Run on the reference agent's host and exact rc with its effective LCM-X settings; pin the replacement route and pre-declare D5's baseline for that route before the window; name organic and synthetic event counts separately and meet D5's existing event bars. Latency is judged per route, never pooled: organic events against the reference route's baseline, synthetic events against the pre-declared replacement-route baseline. A session making no model calls or publishing no compaction supplies no compaction credit. It is not organic activity or field latency. | [window][rc3-window], [notes][rc3-notes] | ADOPTED (release manager, 2026-10-22) |
| The controlled clone cell on the reference agent's settings (D5-B) uses a sanitized settings snapshot instead of a live profile read | Adopt only as controlled-clone evidence. | Name the missing live-read item and the snapshot digest, provenance and effective compression/LCM settings that replace it; pin rc, host and route and disclose it in notes. No profile-content clone, field-latency claim or silent settings substitution. | [D5-B][rc3-d] | ADOPTED (release manager, 2026-10-22) |
| D5-B reader misses externalized doctor output | Adopt a kit-gap disposition only with the complete saved doctor result as replacement evidence. | Name the missing read, replacement result, exact cell/tree and kit #432 item; all doctor checks must be accounted for. Missing output stays INCONCLUSIVE. Any warning requires its own matched previous-GA classification; any failed check blocks. Disclose in notes. Available until the named kit #432 item ships. | [D5-B][rc3-d] | ADOPTED (release manager, 2026-10-22) |
| Phase C floor-host doctor invocation never reaches the tool | Reject an absent doctor result as sufficient proof; allow only a named replacement invocation on the same host, rc and closing store. | Name the missing invocation, complete replacement doctor result and kit #432 item; all doctor checks accounted for, no failed check; each warning needs its own matched previous-GA classification. Without that replacement the raw row stays INCONCLUSIVE and this kit-gap disposition is unavailable. Disclose in notes. Available until the named kit #432 item ships. | [raw C][rc3-c], [ruling][rc3-c-rm] | ADOPTED (release manager, 2026-10-22) |
| Phase C restart re-paste displaces a canary | Reject denominator reduction as proof of full probe coverage; require named replacement planting/read evidence. | Pin the same host, rc, material and restart/re-paste path; restore the full declared probe set by a replacement planting and read of each displaced probe in that cell after the restart/re-paste (or a rerun of the cell), with no LOSS. Name the missing probe, replacement evidence and kit #432 item and disclose in notes. Unsent probes never count as passed. Available until the named kit #432 item ships. | [raw C][rc3-c], [ruling][rc3-c-rm] | ADOPTED (release manager, 2026-10-22) |
| Phase C config-fixture version warning | Adopt only a named fixture-stamp kit-gap disposition. | Name the missing stamp, replacement effective-config evidence and kit #432 item; prove unchanged effective settings and no affected turns. An unclassified warning or execution exception is not covered. Disclose in notes. Available until the named kit #432 item ships. | [ruling][rc3-c-rm] | ADOPTED (release manager, 2026-10-22) |
| Phase C / D5-B lifecycle warnings and caught host detach/import traces | Adopt only a specific matched previous-GA, no-new-regression disposition, separate from the raw doctor/error verdict. | Name each warning/trace shape, host, config and both build SHAs; prove each shape has the same signature on the previous GA under the same host and config, its count per triggering host event (fork/detach, optional import) is no worse, and any affected stored row meets the Phase B identity/order rule. No loss, wrong-session row, reorder or unhandled error may be waived. A host without a matched previous-GA run cannot establish this disposition for that host; disclose the remaining gap. | [ruling][rc3-c-rm], [D5-B][rc3-d] | ADOPTED (release manager, 2026-10-22) |
| User-stopped organic turn pending host catch-up (#1623) | Adopt only for an explicit host stop interrupt followed by interrupted-by-user termination. | Exclude it from completed-organic credit. If a later completed turn exists, require every stopped-turn row in host order with role/content hash and complete tool identities; missing rows fail. Otherwise list pending rows at every read and disclose final pending catch-up. Duplicates stay zero; other incomplete-turn causes fail. Available until #1623's fix ships. | [predeclared ruling](https://github.com/electricsheephq/lcm-x/issues/1554#issuecomment-6267609596) | ADOPTED (release manager, 2026-10-22) |
| Local #1501 strict acceptance FAIL treated as base-equivalent | Reject equivalence as strict acceptance. The existing Phase B differential rule may decide respin only on its own evidence. | Keep strict raw FAIL and unsupported upgrade coverage visible; no green acceptance from equal projections, historical fixtures or an unperformed second-crash probe. Any disposition cites the exact differential cells and existing rule. | [local acceptance][rc3-d], [hands-on][rc3-b] | ADOPTED (release manager, 2026-10-22) |
| Raw host-suite failures / unsupported remote roundtrip treated as overall clearance | Reject workflow success or a test-defect label as suite or roundtrip acceptance. | Retain each failing or unsupported result; name matched baseline failure evidence (the same failing test ids on the previous GA's run of the same workflow and host) and tracked defect for any no-new-regression disposition. Missing roundtrip evidence remains a coverage gap, never a completed roundtrip. | [hands-on][rc3-b], [notes][rc3-notes] | ADOPTED (release manager, 2026-10-22) |
| Latency overlap accepted as conservative with an isolated rerun only on FAIL | Adopt narrowly: a timing PASS under disclosed overlap counts as conservative; an overlapping timing FAIL or INCONCLUSIVE gets one isolated rerun that replaces it. | Retain the load windows and name the overlapping work, which shares no store or process with the measured cell; the measured cell has 0 summary-call failures, timeouts and rate-limit markers; thresholds unchanged; disclose the overlap in notes. Otherwise the exclusive latency-envelope rule applies. | [overlap ruling][rc3-d] | ADOPTED (release manager, 2026-10-22) |
| v0.29.0 only: D1 LOCAL at the #1186 default-flip release, not run | Ship the GA with D1 LOCAL UNRUN, which is not green. As written, the arm embeds each question's full LongMemEval-M history with the harness cache off: about 7 minutes per question, and more than 100 hours with both roles and A/A′. | v0.29.0 only. D1 OFF passes, with 0 recall flips and 0 full-text flips. Local recall evidence is D4 LOCAL (paired, placebo accepted, PASS), the #1586 same-store row (no regression) and the Phase C LOCAL cells. The notes say D1 LOCAL is not green. Making D1 LOCAL fit a release window is #1971. v0.29.1 runs it or carries a new disclosed decision. No other row may cite this decision. | [D1][v029-d], [notes][v029-notes] | ADOPTED (release manager, 2026-10-28) |
| v0.29.0 only: D2/D3 with a failed prospective A/A | Ship the GA with D2/D3 not green. Raw verdict: UNRUN, because the prospective A/A failed. The base-against-base A/A failed its own bar after two provider-latency events: in each, a summary call ran close to the 120 s timeout (one timed out), and the survival fit then dropped live rows (#2004). | v0.29.0 only. Both arms ran the same build, so the failure measures the instrument and the service, not the release. Facts kept rests on D7 (production-mode Track S, all six bars PASS). A v0.29.1 rule that a replay with any summariser timeout or failure is service-invalid and re-run once is added to the D2 validity rule above before that run; this row does not create it. D2 stays not green until an A/A passes. No other row may cite this decision. | [D2][v029-d], [notes][v029-notes] | ADOPTED (release manager, 2026-10-28) |
| v0.29.0 only: D8 at the #1186 default-flip release, with the A/A′ placebo invalid on all four rows | Ship the GA with D8 not green. Raw verdict: UNRUN. The first run is invalid because its placebo failed on all four rows, base against base included where it was read, and the one re-run that the rule requires did not run. The LongMemEval-S OFF record is also INCOMPLETE, because one base answer failed. The disposition is a disclosed release-manager decision, not a pass. | v0.29.0 only. LOCAL non-inferiority passes on both datasets; OFF does not, and is recorded. The notes say D8 is not green and claim no D8 pass. The fleet's upgrade target is v0.29.1, not v0.29.0. v0.29.1 runs D8 as written. No other row may cite this decision. | [D8][v029-d], [notes][v029-notes] | ADOPTED (release manager, 2026-10-28) |
| v0.29.0 only: D5 window of a minor release that fails the strict integrity read on pre-existing classes and has no published organic compaction or synthetic top-up | Accept the window for this minor release by a disclosed release-manager decision. The raw verdict stays FAIL, and the compaction-activity bar is recorded as NOT MET. Compaction evidence is the existing controlled surfaces: Phase C, D7 and D5-B. | v0.29.0 only. The raw reader classes are named in the receipt and each matches a class seen on v0.28.x: delegated sub-agent sessions not stored (#1931); replay re-stores (#1674, bursts the same size as before T0); host-replaced image payloads that LCM-X kept in full; one cron row changed by the configured sensitive-pattern filter; the host-caught fail-closed detach refusal (8 traceback events, against 6 in the v0.28.1 window; the count per triggering event was not measured); three doctor warnings that were already present at T0; and 41 unresolved externalized payload references, the same historical references as both v0.28.1 reads (every reference in the window resolves). There are 0 publication-invariant lines, 0 engine-not-found lines and 0 failed doctor checks, and every organic turn completed. The notes state the raw FAIL and the unmet activity bar. The fleet's upgrade target is v0.29.1. This is not the patch-only alternative above, which grants no waiver for existing duplicate or traceback shapes; major and minor activity bars, and the no-waiver rule for duplicate and traceback shapes, are unchanged for later releases. No other row may cite this decision. | [window][v029-window], [notes][v029-notes] | ADOPTED (release manager, 2026-10-28) |
| v0.29.0 only: Phase C doctor bar with warnings on every host and no available previous-GA disposition | Ship the GA with the Phase C doctor bar not green. Raw verdict: FAIL on the six hosts where the doctor ran, because each reports warnings; INCONCLUSIVE on the Hermes 0.21.3 floor host, where the model never calls `lcm_doctor`. A replacement doctor on the floor host, rc and closing store reports 20 checks: 18 pass, 2 warn, 0 fail. Every other Phase C bar passes on all 7 hosts. | v0.29.0 only. No host reports a failed doctor check. The warnings are lifecycle_fragmentation on all 7 hosts, context_pressure on two, and embedding_provider_health ("FastEmbed is not installed") on the three hosts that run labelled degraded full text. The lifecycle disposition above is unavailable on every host: four hosts ran a different config from v0.28.1, because this release enables the embeddings companion; on six hosts the saved v0.28.1 doctor evidence keeps only numeric details, so the signature cannot be matched; and on all seven the count per triggering host event and the Phase B identity check of the affected rows were not measured. No adopted row covers the other two warnings; on v0.28.1 the embeddings check passed, with embeddings off by default. The notes say the doctor bar is not green. Recording the full doctor result is #432; recording the count per triggering host event is a v0.29.1 kit item. The fleet's upgrade target is v0.29.1. No other row may cite this decision. | [C][v029-c], [notes][v029-notes] | ADOPTED (release manager, 2026-10-28) |
| v0.29.1 only: Phase C doctor and caught-host-trace bars with warnings on every host and no matched previous-GA evidence | Ship the GA with the Phase C doctor bar and the caught-trace bar not green. Raw verdicts: doctor FAIL on the six hosts where the doctor ran, because each reports warnings; doctor INCONCLUSIVE on the Hermes 0.21.3 floor host, where the model never calls `lcm_doctor`, and a replacement doctor on the same host, rc and closing store reports 20 checks: 18 pass, 2 warn, 0 fail; caught-trace bar FAIL, because host-caught traces appear on all 7 hosts. Every other Phase C bar passes on all 7 hosts. | v0.29.1 only. No host reports a failed doctor check. The warnings are lifecycle_fragmentation on all 7 hosts, context_pressure on two, and embedding_provider_health ("FastEmbed is not installed") on the three hosts that run labelled degraded full text: the same classes and host groups as v0.29.0, except that one context_pressure warning moved between hosts. The caught host traces are detach refusals (6-9 per cell) and optional-import failures (4-5 on the three hosts without the companion), in the raw host log only. The lifecycle disposition above is unavailable. The v0.29.0 Phase C cells saved the full doctor result (six cells plus the floor replacement), but no host-event counts, and no Phase B identity check of the affected rows was recorded. Re-scoring the saved v0.29.0 logs offline would give per-file exception counts, but still neither counts per triggering host event nor that identity check. The v0.29.1 cells add per-file exception counts only, so a later release also needs counts per triggering host event and the identity check before the lifecycle row's disposition applies. The notes say the doctor and caught-trace bars are not green. No other row may cite this decision. | [C][v0291-c], [notes][v0291-notes] | ADOPTED (release manager, 2026-10-28) |
| v0.29.0 only: D5-B doctor and LCM-X-frame traceback bars without an available previous-GA disposition | Ship the GA with these two D5-B bars not green. Raw verdicts: doctor INCONCLUSIVE, because the close-turn doctor result was saved by reference and the reader missed it; tracebacks with an LCM-X frame FAIL, with 7 per cell against a bar of 0. Every other D5-B bar passes in both cells. | v0.29.0 only. The saved doctor results read back with 20 checks: 17 pass, 3 warn (config_validation, embedding_provider_health, lifecycle_fragmentation), 0 fail. Every LCM-X-frame traceback is the host-caught, fail-closed refusal to detach a background-review fork, one per detach attempt, as on v0.28.1 (6 and 8 there). The dispositions above are unavailable: the result card names no #432 item; two of the three warning messages differ in text from v0.28.1's; the detach counts come from log lines, not a recorded field; and the affected lifecycle rows were not checked against the Phase B identity rule. The notes say both bars are not green. The fleet's upgrade target is v0.29.1. No other row may cite this decision. | [D5-B][v029-d], [notes][v029-notes] | ADOPTED (release manager, 2026-10-28) |
| v0.29.1 only: D5-B doctor and LCM-X-frame traceback bars with no matched previous-GA evidence | Ship the GA with these two D5-B bars not green. Raw verdicts: doctor FAIL, because both cells report warnings; tracebacks with an LCM-X frame FAIL, with 6 and 8 per cell against a bar of 0. Every other D5-B bar passes in both cells. | v0.29.1 only. Both cells report 20 doctor checks: 17 pass, 3 warn (config_validation, embedding_provider_health, lifecycle_fragmentation), 0 fail. These are the same three warning names as v0.29.0's read-back and v0.28.1. Every LCM-X-frame traceback is the host-caught, fail-closed refusal to detach a background-review fork, one per detach attempt (7 and 7 on v0.29.0; 6 and 8 on v0.28.1). Each cell also has 2 host-caught optional-import tracebacks with no LCM-X frame. The matched disposition above is unavailable: the traceback counts come from scanning each cell's log files, not from a per-event record, and the affected lifecycle rows were not checked against the Phase B identity rule. The notes say both bars are not green. No other row may cite this decision. | [D5-B][v0291-d], [notes][v0291-notes] | ADOPTED (release manager, 2026-10-29) |
| v0.29.0 only: Phase B full host suites and reliability matrix with raw failures and no complete no-new-regression disposition | Ship the GA with both rows not green. Raw verdicts: FAIL on all 7 hosts in the full host suites, and FAIL in the remote reliability matrix. Phase B's CLEAR rests on the review sweep (0 confirmed P0/P1) and the native matrix (0 regressions), not on these rows. | v0.29.0 only. Host suites: 0 PASS→FAIL flips against v0.28.1's run of the same workflow on the same hosts, and the 29–30 pre-existing failures per host match v0.28.1 by test id, but no tracked issue is named for each. The failures new in this release are 12 per host on four hosts from one test fixture (#1968, a test defect, which the rule above does not accept as a suite pass) and the catalog-scanner test on the floor host (#1975). Reliability matrix: 65 of 66 failures and both inconclusive cells match v0.28.1, and the other failure is the known intermittent #861; tracked issues for the 65 are not named. The notes say both rows are not green. No other row may cite this decision. | [B][v029-b], [notes][v029-notes] | ADOPTED (release manager, 2026-10-28) |

[rc3-notes]: https://github.com/electricsheephq/lcm-x/blob/77b2b4f795fb6a4f84c55abf6d0441da4469971e/.github/release-notes/v0.28.1.md
[rc3-d]: https://github.com/electricsheephq/lcm-x/releases/download/v0.28.1/SCORECARD-PHASE-D.md
[rc3-window]: https://github.com/electricsheephq/lcm-x/releases/download/v0.28.1/SOAK-WINDOW-RECEIPT.md
[rc3-c]: https://github.com/electricsheephq/lcm-x/releases/download/v0.28.1/PHASE-C-RECEIPT.md
[rc3-c-rm]: https://github.com/electricsheephq/lcm-x/releases/download/v0.28.1/PHASE-C-RM-RULING.md
[rc3-b]: https://github.com/electricsheephq/lcm-x/releases/download/v0.28.1/PHASE-B-HANDSON-RECEIPT.md
[v029-notes]: https://github.com/electricsheephq/lcm-x/releases/tag/v0.29.0
[v029-d]: https://github.com/electricsheephq/lcm-x/releases/download/v0.29.0/SCORECARD-PHASE-D.md
[v029-window]: https://github.com/electricsheephq/lcm-x/releases/download/v0.29.0/SOAK-WINDOW-RECEIPT.md
[v029-b]: https://github.com/electricsheephq/lcm-x/releases/download/v0.29.0/PHASE-B-RECEIPT.md
[v029-c]: https://github.com/electricsheephq/lcm-x/releases/download/v0.29.0/PHASE-C-RECEIPT.md
[v0291-notes]: https://github.com/electricsheephq/lcm-x/releases/tag/v0.29.1
[v0291-c]: https://github.com/electricsheephq/lcm-x/releases/download/v0.29.1/PHASE-C-RECEIPT.md
[v0291-d]: https://github.com/electricsheephq/lcm-x/releases/download/v0.29.1/SCORECARD-PHASE-D.md

## Receipts

Each phase writes `PHASE-{A,B,C,D}-RECEIPT.md`: rc tag + tree sha, exact commands, matrix results
(per-row pass/fail), findings + dispositions, and the claim class per the gate-closeout
discipline (a phase receipt claims what it measured, never "customer ready"). The receipts are
published as assets of the GA release, redacted (no local paths, host names or people), with a
`SHA256SUMS` file, and the GA release notes link each asset.
