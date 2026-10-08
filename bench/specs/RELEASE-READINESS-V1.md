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
3. **Run the gauntlet** (phases A-D below) against the rc tag. Every phase produces a receipt
   file under the session-notes artifacts dir; the GA cut references all of them. Phase D's full
   tier runs at the rc1 of every release, major, minor or patch (later rcs: its carry-forward rule).
4. **Fix-and-respin**: any P0/P1 finding → fix PR (through the gate) → `-rc2` → re-run.
   Carry-forward rule (every receipt binds an exact tree, so carrying needs proof): **Phase B
   re-runs on every respin** (it is diff-scoped by definition). Phases A and C may carry a prior
   rc's receipt ONLY when the rcN→rcN+1 diff touches nothing outside bench/, docs/, tests/, and
   .github/release-notes/ — AND nothing under bench/instruments/release_gauntlet/ (a changed
   gauntlet invalidates its own receipts) — the carried receipt is referenced WITH the diff-scope proof
   (`git diff rcN..rcN+1 --name-only`) recorded beside it. Any product-code delta re-runs all
   affected phases (ingest/recall/privacy deltas re-run everything).
5. **GA tag `vX.Y.Z`** only when A+B+C+D receipts are green for the passing rc tree. Because
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
particular the candidate respins when, against the base, it adds loss, duplicate rows, a wedge,
a session reset, or a request over the model window. The probe, its rule and both counts go
into the Phase B receipt.

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
`UNAVAILABLE`, `INCOMPLETE` or `INCONCLUSIVE` is not green.

| gate | workload | producing unit | denominator | pass rule |
|---|---|---|---|---|
| D1 recall flips (fast tier, every rc) | LongMemEval-M retrieval, `OFF` | the recall runner + the per-question identity projection (`bench/tools/v1m-rebank/`) | 500 questions, per-question `lcm_recall`; every question stays in the denominator | Two conditions, both required. (1) Flips against the previous GA, re-scored under the current instrument: the row passes with 0 flips; if there are flips, it passes only when every flip is attributed to a named commit and classified intended, net R@10 ≥ previous GA − 0.005, and no category is down more than 0.02. A defect-class loss blocks either way. (2) Every executed recall's health matches its configuration's healthy status (abstention rows run no recall, carry no health and are exempt): no timeout, and the full-text arm's coverage is `ok`. In `OFF`, `degraded: true` from the absent semantic arm with `coverage.fts: ok` is the expected healthy status (RUN-SHEET-ROW1-FULLTEXT); any other degraded reason is unhealthy. Health is checked on its own because the identity projection carries only category, abstention and metrics. An unhealthy question is re-run once; still unhealthy, it counts as a loss. |
| D2 facts kept (full tier) | Track S v3 material, 3 seeds × 2 runs, checkpoints 176 and 304 | `bench/instruments/compaction_probe/track_s/` (`s2/run_s_lcmx.py` → `scorer/score_s.py` → `decision/render_decision.py`) | the run, stratified by seed, per checkpoint: the facts of a run share its compactions, so they are correlated and are not the unit (the run-level variance was 3–25× the independent-fact variance at cp-304 on real runs). A run's kept share is kept / scored facts on that run. Exact stratified run-level permutation test: the statistic is the mean over seeds of the candidate's mean run share minus the previous tree's, and the null is every within-seed split of the seed's 4 runs into 2 + 2 (216 for 3 seeds, so the minimum reachable p is 2/216 ≈ 0.0093), as `decision/analyze_paired.py` computes it (`d2.cp-<n>`). The per-fact sign-flip and sign tests and the per-occurrence McNemar are reported as diagnostics only and do not gate | Blocks when the run-level permutation test gives p < 0.05 (the unrounded value; receipts may display it rounded) AND the net loss is ≥ 5 points of the mean run kept share, at either checkpoint. Every run must be complete, and every seed needs both runs on both sides. `analyze_paired.py` marks a run with any reader-truncated probe `INCOMPLETE`, so a truncated run is re-run (the S1/S2 one-retry rule makes this rare) rather than having its facts excluded. When run1 and run2 of a seed differ by more than 0.10 at any gated checkpoint, both runs of that seed are repeated once and the repeat replaces them; if the spread is still above 0.10, D2 is `INCONCLUSIVE`. |
| D3 Track S absolutes | the D2 runs | the same scorer | per run | 0 admission-proven losses (a planted fact missing from the store at a checkpoint). Level-3 nodes follow Phase C's rule. |
| D4 QA rows (full tier) | LongMemEval-S (500) and LoCoMo (1,986), `OFF`, through the pinned benchmark bridge | the QA runner + its fail-closed scorer | per-question pairs, candidate vs previous GA | Blocks when the paired exact McNemar gives p < 0.05 with a net loss. The A/A′ placebo is the candidate's rerun: the 100-question subset for LongMemEval-S, the full set for LoCoMo. The placebo is accepted only when at most 5% of A/A′ pairs are discordant AND the A/A′ pair does not itself meet the blocking predicate (exact McNemar p < 0.05 with a net change in either direction). Otherwise the row re-runs once (A and A′, fresh stores) and the re-run replaces it; a second rejected placebo makes the row `INCONCLUSIVE`. Failed, missing and judge-fallback questions count against the candidate. |
| D5 latency | the gauntlet cells; the ≥ 24 h reference-agent acceptance window on the exact rc tree (not Phase C's scripted battery) | the field-window counter | per compaction event; the soak needs at least 10 events | Gauntlet cells: max ≤ 145 s, with p50/p90 recorded. Soak: p90 ≤ 60 s and max ≤ 120 s over at least 10 compaction events. Percentiles are nearest-rank: p90 is the ⌈0.9·n⌉-th smallest event time (the rule the Track S scorer uses), and the receipt names the method. If the organic window has fewer, a disclosed synthetic session on the reference agent supplies the rest (the v0.26.0 pattern) and the receipt says how many events each source gave; with fewer than 10 in all, D5 is `INCONCLUSIVE`. Track S event times are recorded but not judged: they are bound by the summariser lane's latency, so Track S runs off the provider's peak hours. |
| D6 recorded, not gating | `LOCAL` and `VOYAGE-TRACKER` recall and QA rows; the continuity suite (#659); the external arms (Codex native, lossless-claw, Hermes built-in, Claude Code) | as each row's kit | per row | Recorded in the receipt. A row becomes a gate only through a change to this spec that names its pass rule (D1's for a recall row, D4's for a QA row) and the release it starts at, and only after it has its own A/A′ with 0 discordant rows in the same session. The continuity suite's per-class bars are declared the same way, from its v0.27.0 recorded run, before v0.28.0 rc1 (#659). External arms run on demand as H1 evidence. |

**Comparability.**
- A candidate row is judged only against a previous-GA row with the same:
  - material or dataset sha, seeds, and fact or question ids;
  - reader and judge (model and effort);
  - summariser, for D2 and D3 (provider, served model, effort and route);
  - scoring path.
- A scorer change re-scores the stored runs of BOTH trees under the current scorer before the comparison.
- A harness change keeps the previous row comparable only when the receipt shows that it cannot change any scored row or denominator of the previous run. Examples: it only adds diagnostic text, or it only changes a failure path the previous run never took (failed, missing and fallback counts all 0). Any other harness change re-runs the previous GA row under the changed harness. That includes a retry that fills answers and a changed exclusion.

**Carry-forward.** Phase D evidence binds the exact candidate tree and the exact Phase D kit: scorer, material or dataset, reader, judge, summariser, benchmark bridge and harness.
- D1 runs on every rc, patch rcs included.
- D2-D4 run at the rc1 of every release: major, minor or patch.
- A later rc of the same release may carry D2-D4 forward only when both of these hold:
  - its diff from the measured rc touches nothing outside `docs/`, `tests/`, `.github/release-notes/`, `CHANGELOG.md` and the parts of `bench/` that are not in the Phase D kit;
  - the Phase D kit is unchanged.

  The receipt records that diff (`git diff --name-only`). Any other change re-runs D2-D4 on the new rc.
- D5's soak is the ≥ 24 h reference-agent acceptance window on the rc tree, not Phase C's scripted battery. A later rc of the same release may carry it forward under the same diff-scope rule as D2-D4 (no product change and an unchanged field counter), with the diff in the receipt; otherwise a new window runs on the new rc.

**Wall clock.** The target is ≤ 60 min per rc for D1. D2 and D4 are hours-long and run in parallel with Phases A-C:
- D2: about 2 h off-peak at two runs in parallel;
- D4: about 3-5 h per configuration, one reader lane per machine.

## Receipts

Each phase writes `PHASE-{A,B,C,D}-RECEIPT.md`: rc tag + tree sha, exact commands, matrix results
(per-row pass/fail), findings + dispositions, and the claim class per the gate-closeout
discipline (a phase receipt claims what it measured, never "customer ready"). The receipts are
published as assets of the GA release, redacted (no local paths, host names or people), with a
`SHA256SUMS` file, and the GA release notes link each asset.
