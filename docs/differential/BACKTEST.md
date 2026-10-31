# Previous-GA differential backtest

Measured 2026-10-15 against GA `6800cf45ad05280e6cf6fa9e1ad96c62b7132646`.
Earlier product candidate: rc7 `432bc4567c7b6201f83421bb9e5402eee7980302`; current product candidate: rc8 `4e603ed65f137a909606afccfc735c691e363d25`.
The corrected public comparison starts at harness head `d0a27b259cd5d1d04b974aeb7dee70ceadb810e0`, whose production files match rc7.
The original historical receipts used checkout `3c9b34bc1165905869369c7d93f914aec95fdf5c`.
The corrected public comparison fixes R2's stale objective probe, compares managed GA defaults by default, and adds C1 summary coverage.
The 32K #1247 shape has 734 uncovered rows on both managed GA and rc7; the bare-GA comparison exposes the gap.
The absolute zero-gap expectation is retained as a dedicated strict xfail. Two exact #1123 R3 cells are accepted
separately: 40K compress runs and 64K provider calls. No production code changed.
This is advisory synthetic evidence. It is not a release gate or proof of runtime readiness.

## Repinned to v0.28.0 (2026-10-19)

The default is `DEFAULT_BASELINE = "v0.28.0"`, pinned to
`709099b3a397a7893a7abae8802ebe7bf605f79e`, with `bare` GA-native defaults.
v0.27.0 remains selectable with `bare`; v0.26.2 remains selectable with `managed`.
The workflow fetches all three pinned commits. Explicit baseline/config overrides retain
precedence. All four accepted deltas remain scoped only to v0.26.2; the v0.26.2-only
oracle and inherited-mark branches are unchanged. v0.28.0 follows the same branches as v0.27.0.
Code receipt: [`a08759acfafc564bd85c19e1faa864e1cc4b9019`](https://github.com/electricsheephq/lcm-x/commit/a08759acfafc564bd85c19e1faa864e1cc4b9019),
on product/harness base `39757f7e9fc5595287b84075cb3cdc6e3baa1d21`.
No product code, input cases, oracle thresholds or delta predicates changed in this repin.

Counts below are raw / unabsorbed oracle events at that code receipt, seed 1227.
Each row includes 75 public cases (15 exact plus 60 generated), 1,000 projection cases,
and the separate #1281 pin on v0.26.2. The public count includes the strict absolute
objective-overflow shape, making the counting boundary explicit beside the earlier 74-case table.

| Selected baseline | Public | Projection | #1281 pin (v0.26.2) | Pytest | Exit | Wall s |
|---|---:|---:|---:|---|---:|---:|
| v0.28.0 default | 0 / 0 | 0 / 0 | 0 / 0 | 97 passed, 2 xfailed | 0 | 187.295 |
| v0.27.0 | 0 / 0 | 0 / 0 | 0 / 0 | 97 passed, 2 xfailed | 0 | 136.043 |
| v0.26.2 | 2 / 0 | 238 / 0 | 0 / 0 | 97 passed, 2 xfailed | 0 | 120.326 |

The v0.28.0 and v0.27.0 projection runs each record 238 inherited cases / 282 inherited
marks; neither has accepted deltas. A measured one-case `projection_cases(1, 1227)` run
on v0.28.0 records `baseline_config="bare"` and `inherited_user_marks={"cases": 1, "marks": 1}`;
the new parameter row pins exactly that result. Reverting only `DEFAULT_BASELINE` to
v0.27.0 makes the new default/SHA/config regression fail; restoring it passes.

The older v0.26.2 public/projection event counts are unchanged. The #1281 pin changes
from the historical 2 / 2 to 0 / 0 because the candidate now includes the
[#1281 fix](https://github.com/electricsheephq/lcm-x/commit/3f59baa7984ad54d9a60fe15c0a15add14110d36) that keeps the newest
request whole when marking cannot reach the budget and the list already fits. Its former
xfail now passes; the remaining xfails are #1225 absolute overflow and #1247 coverage.
Subsequent oracle, identity and restoration regressions also increase the pytest total
relative to the prior repin. These are current-harness results, not a rerun of the old harness.

The measured pytest command is below; the older-baseline rows additionally set
`LCM_DIFFERENTIAL_BASELINE` to the selected version. Wall seconds measure the whole process.
All runs use Python 3.14, the CI host stub, isolated temporary/cache directories,
`HERMES_DISABLE_LAZY_INSTALLS=1`, and `PYTHONDONTWRITEBYTECODE=1`, without network or model calls.

```bash
LCM_DIFFERENTIAL=1 nice -n 19 python -m pytest tests/differential -q -rx -o addopts= -p no:cacheprovider
```

The CLI's default sample is 75 public plus 60 projection cases (135 total), so its
projection event count is intentionally separate from the 1,000-case pytest sample.
All three CLI runs identify the clean code receipt above.

| CLI baseline | Public raw / unabsorbed | Projection raw / unabsorbed | Accepted events | Exit | Wall s |
|---|---:|---:|---:|---:|---:|
| v0.28.0 default | 0 / 0 | 0 / 0 | 0 | 0 | 88.577 |
| v0.27.0 | 0 / 0 | 0 / 0 | 0 | 0 | 90.016 |
| v0.26.2 | 2 / 0 | 14 / 0 | 16 | 0 | 82.015 |

The legacy CLI's 14 projection P_USER events are absorbed by D1233; its two public
R3 events are absorbed by the unchanged 40K and 64K D1123 cells. D916 absorbs none.
The two newer baselines have zero accepted events and each records 14 inherited cases /
15 marks in this smaller projection sample. CLI commands (baseline omitted for the default):

```bash
LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --out differential-results/v028
LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --out differential-results/v027 --baseline v0.27.0
LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --out differential-results/v026 --baseline v0.26.2
```

Compatibility-suite `PREVIOUS_GA` and rollback-reader pins remain a separate repin.
This is local synthetic evidence only; it does not establish merge, release or runtime readiness.

## Repinned to v0.27.0 (2026-10-16)

The default is `DEFAULT_BASELINE = "v0.27.0"`, pinned to
`c67d7633d963a2ffd24cf90995189ee58e2f88c7`. The v0.26.2 pin remains selectable.
`BASELINE_CONFIG` selects `managed` for v0.26.2 (the historic 8K-leaf overlay) and `bare`
for v0.27.0 (GA-native defaults, matching the candidate). Explicit `--baseline-config` and
`LCM_DIFFERENTIAL_BASELINE_CONFIG` overrides still take precedence. Unlisted refs use `bare`.
All four existing accepted deltas apply only to v0.26.2; their predicates are unchanged.
Code receipt: `82e41636ca926ce57be9fa3fc7c467b4d0ea52f7`; prior harness: `a65e9596`.
The candidate's product files match v0.27.0. No cases, other oracles, thresholds or deltas changed.

For v0.27.0 and other non-v0.26.2 baselines, P_USER reports only candidate marks absent from GA
(`set(candidate) - set(GA)`). Shared marks remain visible as `inherited_user_marks` cases/marks
in both summary formats, including shared marks in a case that also has a new mark.
v0.26.2 retains its absolute P_USER check and unchanged D1233/D916 absorption.

Counts below are raw / unabsorbed oracle events, with 74 public cases, 1,000 projection cases
and one separate historical #1281 pin. The default's 1,074 selected-baseline cases have **zero raw reds**.

| Harness / selected baseline | Public | Projection | #1281 pin (v0.26.2) | Pytest | Exit | Wall s |
|---|---:|---:|---:|---|---:|---:|
| Prior / v0.26.2 | 2 / 0 | 238 / 0 | 2 / 2 | 27 passed, 3 xfailed | 0 | 116.276 |
| Revised / v0.26.2 | 2 / 0 | 238 / 0 | 2 / 2 | 40 passed, 3 xfailed | 0 | 123.421 |
| Revised / v0.27.0 default | 0 / 0 | 0 / 0 | 2 / 2 | 40 passed, 3 xfailed | 0 | 113.343 |

The v0.27.0 projection summary records **238 inherited cases / 282 inherited marks**. These are
inherited #1233 last-resort marks, not accepted deltas or differential regressions. Eleven new tests
cover identical, additional and subset marks under legacy/newer baselines and summary accounting;
the two earlier repin tests still cover baseline-scoped delta behavior.
The v0.26.2 per-case tables, replay inputs, driver results (excluding elapsed seconds), oracle events
and coverage are identical to `a65e9596`, including the exact historical #1281 outcome.

Explicit `--baseline v0.27.0 --baseline-config managed` still yields exactly two unabsorbed reds
(exit 1, 92.030 seconds, 135 CLI cases with its default 60-case samples):

- `i1139_small_window_40k`: C1, uncovered rows GA 0 / candidate 29.
- `i1139_small_window_64k`: R3, summarizer calls GA 46 / candidate 65.

Both clear under native config parity. The earlier repin's 238 absolute P_USER reds are now
reported as inherited marks. No new accepted delta hides either configuration difference.
A scratch-only candidate-driver mutation adds mark 999 to projection p0; the default baseline/config
then fails with unabsorbed `P_USER: [999]` (exit 1, 0.869 seconds), while retaining its inherited mark count.

The #1281 pin remains explicitly on v0.26.2: v0.27.0 and the candidate both send 112,193 tokens,
mark `[0]`, and lose objective/newest, yielding zero R2 events. v0.26.2 sends 112,689 tokens and
retains both sentinels; its exact two-R2 expectation is unchanged.
Opt-out collection remains the identical ordered 7,712-test set. Serial nice-19 commands, per-case
tables, mutation patch and full-suite parity results are retained in the revised lane evidence report.
This is local synthetic harness evidence, not merge, release or runtime proof.

## Historical inputs and rc8 (2026-10-16)

The branch merged rc8 `4e603ed65f137a909606afccfc735c691e363d25` with merge commit
`cd2a17f0c7dcaa5a3df5cd564023b5149aec6703`. All new authored changes are tests/docs only.
Every oracle, threshold and pre-existing delta is unchanged. The sole added delta is the explicitly
requested `D1233_LAST_RESORT_USER_MARK`: projection / P_USER only, candidate marks a nonempty subset of GA marks, and
`not (ga["out_tokens"] <= case["limit"] < cand["out_tokens"])`, linked to [#1233](https://github.com/electricsheephq/lcm-x/issues/1233).
This does not accept R2, user marks absent from GA, public-layer marks, or candidate overflow where GA fit.

### Inputs and seams

- `i1221_historical_assembly_32k` ports `objective_scenario` from
  `tests/test_issue_1221_objective_default_bound.py`: four history rows including the 12K objective,
  one 2K leaf covering those rows, 20 tool-only tail rows, and a 32,768 window. Row sentinels replace
  text prefixes and timestamps retain ordering. The normal host loop chooses a 24-row fresh tail and
  keeps the objective; it cannot select this input. A case-scoped `assembly_tail` seam instead calls
  the regression's `_assemble_context(system, tail, include_lcm_note=False)` then `_survival_fit`.
  It records the same sent-view schema and runs unchanged R1/R2/R3/C1/A1/A2 oracles. No extra host
  retry or summarizer call is inserted on this assembly path.
- `i1139_historical_sweep_32k` starts with #1140's doubled 48-row backlog and 24-row tail, at 32,000,
  and seeds its four depth-zero nodes (2K token metadata each). It continues for 360 logical calls,
  15 seconds apart, with one assistant/tool pair (400/1,600 characters) per call and a user every ninth
  call. The issue supplies a cadence rather than the original transcript. The existing 8K tool-result
  workload filled the fresh tail and entered survival holding before prefix pressure was observable.
  This reconstruction keeps the tail smaller, so the summary frontier can drive the threshold.
  The fixture's `_summarize_leaf_chunk_with_rescue` and module-bound `summarize_with_escalation`
  seams return healthy 2K summaries, counted as provider calls. They bypass provider admission/rescue,
  as the regression fixture does; real ingest, threshold sweep, condensation and host gating still run.
  Module replacements are restored after each case, including the extracted compaction module.
- `i1225_historical_assembly_16k` uses the issue's 12K-objective / 16K-window boundary and declares
  a 4K system prefix plus an 8K tool result, with a seeded history leaf and two-row assembly tail.
  [#1225](https://github.com/electricsheephq/lcm-x/issues/1225) gives size ranges, not a serialized
  host input. The prefix makes room genuinely insufficient: the matrix records GA at 18,135 tokens and rc8 at 16,122
  into a 16,000 window. This is parity in overflow, not an R1 regression. The dedicated strict xfail
  raises only `ObjectiveOverflow`, after successful differential evaluation and a GA overflow check;
  driver errors and unrelated failures cannot satisfy it. The old 22K/32K input still has its fit check.

### Re-measured historical matrix

All rows compare managed GA `6800cf45ad05280e6cf6fa9e1ad96c62b7132646`, seed 1227,
public result schema (including the documented assembly seam). Counts are raw oracle events;
all omitted oracles are zero. `917-*` uses the parallel #1220 input, `917-seq-*` the three #1227 inputs,
`936-*` the new #1221 input, and `1129-*` / `1140-*` the new #1139 input. Each seven-case row uses all
three new historical inputs, parallel #1220, and sequential #1227 n90/n113/n128.

| Probe / receipt | Candidate SHA | Cases | Raw oracle reds | Unabsorbed | Wall s |
|---|---|---:|---|---:|---:|
| 917-parent | `bbcbf2d706706ddbb0f7a6aba63fdcebaf81447b` | 1 | 0 | 0 | 3.14 |
| 917-merge | `286e731800b95eb9bbf11d240c4ca6e2d7f3e8d4` | 1 | i1220:R1=1 | 1 | 2.11 |
| 917-seq-parent | `bbcbf2d706706ddbb0f7a6aba63fdcebaf81447b` | 3 | 0 | 0 | 2.16 |
| 917-seq-merge | `286e731800b95eb9bbf11d240c4ca6e2d7f3e8d4` | 3 | i1227:R1=2 | 2 | 2.13 |
| 936-parent | `7defa0b0fac8a4906d5e054302f6ef6d630544f4` | 1 | 0 | 0 | 2.25 |
| 936-merge | `112388f7f07665ec21a74a7b114c8ed2d07d6577` | 1 | i1221:R2=2 | 2 | 2.98 |
| 1129-parent | `86fc834ef0e995bc8e187b002db43268d033346a` | 1 | 0 | 0 | 10.78 |
| 1129-merge | `2c6dfef9c6425aca765113e769c0df7dd389d35e` | 1 | i1139:R3=2 | 2 | 18.98 |
| 1140-merge | `09e32580fdf95885474dbeb5ec4794e30136bb6a` | 1 | 0 | 0 | 13.98 |
| 1226-parent | `7676991390911e83ab725be313c8db2fec4cbaa0` | 7 | i1220:R1=1, i1221:R2=2, i1225:R2=2, i1227:R1=2 | 7 | 6.58 |
| rc6 | `bd46fd9fed63bc95f81c464065400deff143f667` | 7 | i1227:R1=2 | 2 | 23.19 |
| rc7 | `432bc4567c7b6201f83421bb9e5402eee7980302` | 7 | 0 | 0 | 7.24 |
| rc8 | `4e603ed65f137a909606afccfc735c691e363d25` | 7 | 0 | 0 | 9.71 |

Full SHAs, exact commands, case IDs, wall times, missing seams, checkout HEADs, exits, replay inputs,
both tree results/logs and per-tree means are recorded in the PR. Expected culprit reds return CLI
exit 1; they are successful negative probes, not execution errors. No required seams were missing
on any historical row.

- #1221: GA and #936's parent keep both sentinels; the culprit loses objective and newest (R2=2),
  then rc6/rc7/rc8 keep them. Parent / culprit / rc8 send 27,553 / 21,039 / 26,927 tokens.
- #1139: GA and #1129's parent use 16 compress runs / 54 provider calls; culprit 249 / 331
  (both R3 metrics red); #1140 uses 15 / 28 (green). This is a causal backtest, not a claim to match
  the original simulator's absolute counts.
- #1226 parent now has seven events in four cases; rc6 has two events in two cases. #1220's
  fit and #1221's retention clear; #1227 n113/n128 remain red until rc7. #1225 retention clears but
  the absolute overflow remains, so this is not a claim that #1226 fixes #1225.
- #1227's actual introduction is also measured: n113/n128 are red at #917's merge and green at
  its parent. It was already present before #1226; rc6 versus rc7 alone would not prove culprit/parent.

**Historical acceptance condition (a): 4 of 4 seed defects are red at their culprit and green at their parent**:
#1220 and #1227 at #917, #1221 at #936, and #1139 at #1129. Fix clearance is measured at rc6,
rc7 and #1140 respectively. The open #1225 absolute limitation is tracked separately, outside the
four historical regression seeds. Promotion remains report-only while open product findings remain.

### rc8 findings and original acceptance

**rc8 g0: documented #1281 case (P3, v0.27.1, not GA-blocking).** The exact generated case (seed 1227, 128,000 window,
112,407 input tokens) still fails R2 twice: GA sends 112,689 and retains objective/newest;
rc8 sends 112,193 with user mark `[0]`, losing both sentinels. The survival budget is 108,800,
so marking the newest user did not reach it. This regime marks the newest user even though the
marks cannot reach the survival budget and the unmarked list already fits the window.
The tool rows carry `name` without `tool_name` (#1232), which leaves them unmarked; host-form rows
need about 2,240+ distinct parallel calls in one assistant row at 128K (580+ at 32K).
The last-resort user pass is `survival_fit.py:427` at rc8. The reproduction is recorded in the PR.
Only g0 at seed 1227 is excluded from the general generated sample; a dedicated strict xfail pins
exactly the two R2 events at call 0 for objective/newest. It flips to a failing strict XPASS when the
#1281 fix lands. Other unexplained events fail normally. No delta, oracle change or production repair
was added for this finding in the backtest.

The original 4,000-case rc8 projection triage has P_USER=951, all accepted before D1233 was tightened,
with R1=P_INPUT=0 and zero unexplained events. D916 absorbs zero; it stays as a historical guard. Both #1123 deltas are also zero on this projection-only run. GA/candidate
wall times are 149.25/222.29 seconds, means 0.03692/0.05502 seconds per case; process wall
374.91 seconds. No seams are missing. The supplied recheck's 238 P_USER cases used 1,000
inputs, not 4,000 (confirmed from its original summary); these are different sample sizes.
Detailed projection receipts are recorded in the PR.

Default collection before/after is 7,712 tests (rc8 adds 14 versus the corrected rc7 count).
Opt-out: one module skip, expected exit 5, 0.09 seconds pytest / 1.90 seconds process wall.
Collection after: 14.02 seconds pytest / 15.21 seconds process wall. Ruff and whitespace pass.
The original final opt-in run has **27 passed, 2 strict xfailed (#1225 and #1247), 1 failed** in
434.16 seconds pytest / 434.94 seconds process wall. The sole failed test is the generated sample,
containing only g0's two R2 events. All 14 normal exact cases pass, including both new regression
shapes. Unique public inputs: 15 exact plus 60 generated; raw R2=2, R3=2 (the existing exact #1123
cells), all other differential oracles zero. The 1,000 projection cases have P_USER=238, all D1233,
with zero unexplained. D916 absorbs zero on the complete current run and remains a historical guard.
Detailed opt-in receipts are recorded in the PR.
That original run preserved g0 as a failure; the documented #1281 case now has the dedicated strict
xfail described above. Production repair is outside this tests-only change.

All commands used the supplied interpreter and host stub, serial execution at nice 19, no xdist,
no model/network calls, and no pytest cache. The explicitly required local offline matrix/affected
suite used a 15-minute per-run budget because remote execution would require forbidden push/CI
writes; the longest focused suite took 7m15s. Original failed input attempts are
recorded in the PR; the final matrix supersedes them. Ruff/whitespace are clean. No full
repository tests, remote CI, push, PR write, release or runtime action was performed.

### Final last-resort acceptance (2026-10-16)

D1233 requires a nonempty subset of GA user marks, and no candidate overflow where GA fit:
`bool(cand["user_marks"]) and set(cand["user_marks"]) <= set(ga["user_marks"]) and not (ga["out_tokens"] <= case["limit"] < cand["out_tokens"])`.
D916, all other deltas, oracles, thresholds and product files remain unchanged. The expanded
predicate regression fails with the preceding equality rule and passes with this subset rule.

| Projection probe, seed 1227 | Cases | Raw R1 | Raw P_USER | D916 | D1233 | Unexplained | Exit | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unchanged rc8 | 1,000 | 0 | 238 | 0 | 238 | 0 | 0 | 39.665 |
| Unchanged rc8 | 4,000 | 0 | 951 | 0 | 951 | 0 | 0 | 154.774 |
| User pass stops after first mark | 1,000 | 6 | 238 | 6 | 232 | 6 | 1 | 40.433 |
| User-pass limit guards removed | 400 | 0 | 106 | 0 | 97 | 9 | 1 | 7.092 |

The early-exit mutant retains exactly 6 unexplained P_USER events: p66, p295, p347, p543, p551
and p922; each candidate overflows where GA fit, while D916 still absorbs its R1 event.
The guard-removal mutant remains red with 9 unexplained events. Both unchanged rc8 samples
have zero unexplained events, every P_USER event is accepted by D1233, and no seams are missing.

Subset is needed because unchanged rc8 drops a GA mark while both outputs fit in p543, p1491,
p3277, p3341 and p3629; equality would reject all 5. In p543, GA marks `[2,3,5]` produce 495 tokens
versus candidate marks `[2,3]` at 537 tokens, under limit 679.

With the #1281 pin, the opt-in suite reports **27 passed, 3 strict xfailed (#1225, #1247 and #1281),
0 failed** in 117.80 seconds pytest / 118.505 seconds process wall. The generated sample passes
with only seed-1227 g0 excluded; its dedicated pin retains the two unexplained R2 events.
The F3 probe flips only the #1281 pin to XPASS(strict): **27 passed, 2 strict xfailed, 1 failed**,
119.71 seconds pytest / 120.401 seconds process wall. Replaying all 60 generated cases, including
g0, through the same evaluator on F3 gives zero unexplained events (34.221 seconds process wall).
Before the pin, the deduplicated suite reported **26 passed, 2 strict xfailed, 1 failed** in 115.77
seconds pytest / 116.644 seconds process wall. Its sole failure was g0; the projection sample was
green. The old 32K fit input remains excluded by ID
from `test_exact_shape`; its dedicated checks and the separate historical 16K strict xfail remain.
Default collection stays at 7,712 tests (7.959 seconds process wall); opt-out reports one skipped
module, expected exit 5 (0.786 seconds). Detailed commands, replay inputs and results are recorded
in the PR. This is advisory synthetic evidence, not readiness proof.
After the #1281 pin, enabled collection is 30 tests (previously 29), while disabled collection
remains 7,712 (7.211 seconds process wall). Opt-out still skips one module, expected exit 5
(0.656 seconds). A fresh unchanged-rc8 1,000-case projection run at seed 1227 has zero unexplained
events, all 238 P_USER events D1233, in 16.107 seconds process wall.

## Corrected public comparison

The newest-user objective rule replaces the static first-turn R2 probe. GA now defaults to the managed
configuration; select `--baseline-config bare|managed` or `LCM_DIFFERENTIAL_BASELINE_CONFIG` (default `managed`).
Only GA receives this selection; candidate defaults and explicit per-case overrides remain in force.
Managed values are `context_threshold=0.75`, `fresh_tail_count=24`, `fresh_tail_max_tokens=24000`,
`leaf_chunk_tokens=8000`, `threshold_full_sweep_enabled=True`, `large_output_externalization_enabled=True`,
and `large_output_active_replay_stubbing_enabled=True`. Explicit case overrides take precedence.

C1 counts non-system stored rows absent from the final live view and from all DAG message-source coverage.
Unique synthetic row sentinels identify live rows; summary-source IDs are not message IDs.
C1 is red only when candidate exceeds GA by more than two rows. This measures summary coverage, not store loss.
The absolute `i1247_*` strict xfail requires zero uncovered rows on the 32K i1139 shape and catches only
`SummaryCoverageGap`; a driver error or unrelated assertion cannot satisfy it.

Corrected receipts and the table below supersede the original public triage and acceptance results retained
under the historical sections. Historical rows used the bare GA defaults and the old static objective probe.

### rc7 public-layer results after correction

All 72 public cases (12 exact plus 60 generated), seed 1227. Counts are raw oracle events.
Managed receipts come from the final enabled pytest run; bare GA replays the identical saved inputs and
reuses those validated candidate records. No candidate result is regenerated or edited for the bare arm.

| GA config | Cases | R1 | R2 | R3 | C1 | A1 | A2 | Accepted | Unabsorbed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| managed | 72 | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| bare | 72 | 0 | 0 | 3 | 15 | 0 | 0 | 0 | 18 |

The managed gate has zero unexplained reds. Both R3 events have separate exact-cell deltas:
`D1123_64K_LEAF_TRADE` names #1123 / PR #1129 and accepts only provider calls 65 versus 46 at 64K;
`D1123_40K_COMPRESS_TRADE` accepts only compress runs 113 versus 93 at 40K (provider calls fall 166 to 165).
The latter is an interpretation correction: the supplied triage numbers already exceed R3's 10% + 2 run
threshold despite its prose saying 40K passes. Neither exception absorbs another metric, case, or count pair.

| i1139 window | Bare GA / rc7 uncovered | Managed GA / rc7 uncovered | Managed GA / rc7 provider calls |
|---|---:|---:|---:|
| 32K | 0 / 734 | 734 / 734 | 3 / 3 |
| 40K | 8 / 29 | 206 / 29 | 166 / 165 |
| 48K | 0 / 0 | 0 / 0 | 246 / 198 |
| 64K | 0 / 0 | 26 / 0 | 46 / 65 |
| 128K | 130 / 25 | 125 / 25 | 12 / 7 |

Bare remains a diagnostic arm: its three R3 reds are provider calls at 40K (165/53), 48K (198/85),
and g51 (4/1), candidate/GA. C1 is red at 32K, 40K, and g9, g17, g23, g27, g28, g29, g30, g37,
g38, g54, g55, g57, g58. These events stay visible and unabsorbed; no broad default-change delta hides them.
They are comparison signals against bare settings, not evidence of regression against managed GA.
At 32K, C1 reproduces #1247 exactly: 734 versus bare GA's 0, and 734 versus managed GA's 734.
A1 has no missing produced sentinels; C1 does not claim stored context was deleted.

Final enabled acceptance: 24 passed, 1 strict xfailed (#1247), 630.78 seconds pytest / 632.21 seconds process wall.
The 1,000-case managed projection sample has 54 R1 reds, all accepted by D916; zero unexplained.
Opt-out: one module skip (expected exit 5), 1.85 seconds wall. Full collection: 7,698 tests, unchanged,
19.25 seconds pytest / 21.19 seconds wall. Enabled collection grows from 18 to 25 tests.
Bare GA public replay: 184.85 seconds process wall; candidate records reused. No wall-time assertion decides an oracle.
The initial development run (362.18 seconds) found the 40K run-count discrepancy and a missing-ID unit-fixture
error in a new delta predicate; both are corrected in the final run. Original receipts below remain historical.

## Reproduce

With an interpreter providing pytest and the CI `agent.context_engine` host stub on `PYTHONPATH`:

```sh
HERMES_DISABLE_LAZY_INSTALLS=1 LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --baseline v0.26.2 --candidate 432bc4567c7b6201f83421bb9e5402eee7980302 --layers projection --cases 4000 --seed 1227 --out differential-results/projection
HERMES_DISABLE_LAZY_INSTALLS=1 LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --baseline v0.26.2 --candidate 432bc4567c7b6201f83421bb9e5402eee7980302 --baseline-config managed --layers public --cases 60 --seed 1227 --out differential-results/public-managed
HERMES_DISABLE_LAZY_INSTALLS=1 LCM_DIFFERENTIAL=1 nice -n 19 python -m tests.differential.run --baseline v0.26.2 --candidate 432bc4567c7b6201f83421bb9e5402eee7980302 --baseline-config bare --layers public --cases 60 --seed 1227 --out differential-results/public-bare
LCM_DIFFERENTIAL=1 nice -n 19 python -m pytest tests/differential -q -p no:randomly -p no:cacheprovider
```

For a historical row, substitute its full SHA and use `--layers public --cases 0 --only CASE_ID` (comma-separated IDs are accepted). Resolve merge parents with `git rev-parse SHA^1`. All required objects were already present; no fetch was needed.
`LCM_DIFFERENTIAL_BASELINE_CONFIG`, `LCM_DIFFERENTIAL_BASELINE`, `LCM_DIFFERENTIAL_CANDIDATE`, `LCM_DIFFERENTIAL_CASES`, `LCM_DIFFERENTIAL_SEED`, `LCM_DIFFERENTIAL_PROJECTION_CASES`, and `LCM_DIFFERENTIAL_OUT` control pytest. Defaults are managed, GA, worktree, 60, 1227, and 1000. The last variable selects the evidence directory.
Each run retains replayable `*-cases.jsonl`, per-tree `*.jsonl` and logs, `summary.json`, and `summary.md`. Replay a case file with `nice -n 19 python tests/differential/driver.py --tree TREE --layer public --cases CASES --out RESULTS` under the same opt-in environment.

## Historical matrix

All rows use the public layer, seed 1227, and only the named exact shapes. Counts are failing oracle events; omitted oracles have zero failures. Unexplained counts are events, not distinct cases. Wall time is GA plus candidate; means are GA/candidate seconds per case.

| Probe | Candidate SHA | Cases / exact shape | Oracle reds | Unexplained | Wall s | Mean s/case |
|---|---|---|---|---:|---:|---:|
| 917-parent | `bbcbf2d706706ddbb0f7a6aba63fdcebaf81447b` | 1: 1220 | 0 | 0 | 0.99 | 0.097/0.101 |
| 917-merge | `286e731800b95eb9bbf11d240c4ca6e2d7f3e8d4` | 1: 1220 | R1=1 | 1 | 1.06 | 0.109/0.169 |
| 936-parent | `7defa0b0fac8a4906d5e054302f6ef6d630544f4` | 1: 1221 | 0 | 0 | 1.02 | 0.071/0.055 |
| 936-merge | `112388f7f07665ec21a74a7b114c8ed2d07d6577` | 1: 1221 | 0 | 0 | 0.77 | 0.050/0.071 |
| 1129-parent | `86fc834ef0e995bc8e187b002db43268d033346a` | 1: 1139:32k | R2=2 | 2 | 16.41 | 11.988/3.457 |
| 1129-merge | `2c6dfef9c6425aca765113e769c0df7dd389d35e` | 1: 1139:32k | R2=2 | 2 | 15.43 | 2.955/11.676 |
| 1140-merge | `09e32580fdf95885474dbeb5ec4794e30136bb6a` | 1: 1139:32k | R2=2 | 2 | 6.33 | 2.322/2.914 |
| rc6 | `bd46fd9fed63bc95f81c464065400deff143f667` | 5: 1220, 1221, 1227:n90, 1227:n113, 1227:n128 | R1=2 | 2 | 2.13 | 0.130/0.161 |
| rc7 | `432bc4567c7b6201f83421bb9e5402eee7980302` | 5: 1220, 1221, 1227:n90, 1227:n113, 1227:n128 | 0 | 0 | 4.32 | 0.205/0.530 |

- #917: the parallel shape changes from green at the parent to R1 at the merge: GA sends 27,719 tokens, candidate 39,771, window 32,768. The short-text preservation branch is `survival_fit.py:349` at the merge.
- rc6: sequential outage n113 sends 33,641 tokens versus GA 27,716; n128 sends 38,111 versus GA 27,683. The tool-only second pass is `survival_fit.py:432`. Both shapes are green on rc7. n90, #1220, and #1221 are green on both rc6 and rc7.
- #936: both parent and merge are green. This public driver did not reproduce the assembly-only #1221 failure: its 24-row fresh tail still includes the objective, whereas the historical helper directly assembled the 20-row tool tail. This is a driver coverage gap, not evidence that #1221 was invalid.
- #1129 and #1140: the 32k shape has two R2 events at each parent/merge/fix; R3 stays green. GA has 119 compress runs / 52 provider calls; each compared tree has 49 / 3. This synthetic provider/cadence does not reproduce the historical progress regression, so it cannot establish that #1140 clears it. No oracle was tuned.

## Projection triage

4,000 instrumented p34 cases: 250 R1 events, all 250 absorbed by the sole `D916_GA_MARKED_USER` delta; zero unexplained, zero candidate user marks, zero input mutations. GA/candidate wall times were 51.52/74.08 seconds; mean case times 0.01085/0.01821 seconds. No missing seams.
The original uninstrumented p34 output artifacts independently contain 253 fit failures, all with GA mark-only user rows. Their SHA-256 receipts and all case IDs for those 253 failures are recorded in the PR. The new corpus replaces text prefixes with row sentinels, grows very short rows, and uses a fresh store per case (changing store-id widths). It is intentionally reported as 250, not presented as a byte-identical 253-case reproduction.

## Historical public triage and unexplained results

The rc7 run contains 12 exact shapes plus 60 generated cases, seed 1227. It reports 34 unexplained events in five cases: R2=31 and R3=3; R1=A1=A2=0. No delta absorbed a public case; D916 is zero-absorption on this layer but active in the full projection run.
- `i1139_small_window_32k`: objective missing versus GA at calls 10–11 (zero based).
- `i1139_small_window_40k`: objective missing at calls 13–36; provider calls 165 versus GA 53 (R3).
- `i1139_small_window_48k`: objective missing at calls 16–17; provider calls 198 versus GA 85 (R3).
- `i1139_small_window_64k`: objective missing at calls 21–23.
- `g51` (`mixed/1004/user`): provider calls 4 versus GA 1 (R3).
These are observed synthetic comparison failures, not newly confirmed product defects. In particular, R2 tracks the original tagged objective even after later scripted user turns. Every failure retains its full case JSON; no new delta suppresses it.
Absolute over-window outputs also occurred in g4, g10, g13, g15, g41, g42, and g44; GA did not fit either, so these are not R1 regressions. #1225 is separate: this 22k-objective/8k-tool-tail input is projected to 22,579 tokens and fits. Its strict absolute xfail XPASSes; the supplied shape does not reproduce the issue's absolute failure.

The rc6 enabled suite also reports 34 unexplained projection R1 cases among 1,000 samples (98 raw fit failures minus 64 D916 cases). These stay red; the rc7 projection sample clears them. The exact replay records are recorded in the PR.

| rc6 projection case | GA tokens | Candidate tokens | Limit |
|---|---:|---:|---:|
| p10 | 1414 | 1977 | 1474 |
| p92 | 643 | 944 | 835 |
| p95 | 43 | 805 | 754 |
| p111 | 1488 | 1575 | 1523 |
| p221 | 2203 | 2785 | 2374 |
| p234 | 2160 | 2422 | 2198 |
| p241 | 1160 | 1461 | 1368 |
| p250 | 3555 | 4093 | 3775 |
| p252 | 3479 | 4648 | 3744 |
| p291 | 3582 | 3862 | 3618 |
| p300 | 1167 | 1468 | 1388 |
| p340 | 85 | 736 | 643 |
| p367 | 43 | 280 | 271 |
| p400 | 1362 | 1795 | 1587 |
| p433 | 2659 | 3149 | 3139 |
| p452 | 1003 | 1319 | 1185 |
| p492 | 1292 | 2160 | 2005 |
| p503 | 655 | 1515 | 711 |
| p574 | 2230 | 2359 | 2233 |
| p601 | 330 | 588 | 372 |
| p617 | 369 | 670 | 464 |
| p644 | 1173 | 2865 | 2677 |
| p649 | 915 | 1217 | 933 |
| p670 | 3645 | 4099 | 3877 |
| p691 | 406 | 1101 | 910 |
| p704 | 1507 | 2547 | 2386 |
| p715 | 3085 | 3902 | 3464 |
| p759 | 1510 | 2272 | 1888 |
| p813 | 1132 | 1316 | 1312 |
| p874 | 1471 | 2276 | 1615 |
| p894 | 1065 | 1305 | 1239 |
| p927 | 57 | 303 | 212 |
| p964 | 544 | 806 | 786 |
| p983 | 1257 | 2427 | 1311 |

## Method and interpretations

- Default config construction disables only temporal rollups; fleet overrides are the seven settings in the prior public driver. #1107 declares a 4,000-token first-sight threshold because its stated 5,000-token result is below the product default 10,000 threshold; host overhead is 1,000 tokens.
- Both trees use the deterministic character counter, no plugin registration, an offline provider, and a fresh engine/store per case. TCP connections are refused. Healthy replies are 80 tokens, or approximately 2,000 for #1139. The logical host advances 15 seconds per model call; a provider attempt advances one second. Product cooldown, circuit, and spend-guard logic still run.
- All loaded tree modules with a time-module binding share the scripted clock. An early matrix patched only engine/escalation and was invalid after compaction moved to its own module; the corrected run recorded in the PR supersedes it. No timing assertion decides product correctness.
- Objective cases seed a real DAG leaf through the guarded seams used by the existing regression test. They then use public ingest/should_compress/compress/retry/update calls; the driver does not force private assembly results into the sent view.
- A1 reads every stored message, follows externalized references found in stored content with that tree's `extract_externalized_refs` and `load_externalized_payload`, and includes retrieved payload content in the sentinel census. Unretrievable references are recorded. Sentinel coverage is not a byte-for-byte durability proof.
- Every public call records token/row counts, compression runs, provider calls, overflow status, sentinel census, and tool-pair errors. The newest alias is derived from the latest produced user row's unique sentinel. Projection fixtures retain p34's synthetic externalized-placeholder and unstored-row cases.
- Generated samples weight windows at or below 128k at 70% exactly for a 60-case run. Larger windows use larger rows to bound local row count; the sampler is not an exhaustive Cartesian matrix. A stopped initial 1M-window/8,311-row pilot is retained as incomplete evidence. The final generator pads any density-floor shortfall deterministically.
- Private seam absence is recorded and unsupported essential paths become DRIVER failures. No tested tree lacked a required seam. Missing commits/host stubs skip locally and fail under CI.
- The provided interpreter has no xdist: runs are serial under nice 19, cacheprovider disabled. The backtest explicitly requires the local offline historical matrix; it does not run the full repository suite. A brief overlap during the pilot was suspended, then resumed serially.

## Historical acceptance evidence

Detailed command receipts and logs are recorded in the PR. The default collection baseline is 7,698 tests. The disabled differential module reports one skip in 0.04 seconds (0.99 seconds process wall time); pytest exits 5 because the opt-out module collects no runnable tests.
Full collection after the additions is still 7,698 tests (8.94 seconds), with the new module skipped. `ruff check tests/differential` passes.

Enabled worktree pytest: 12 passed, 6 failed in 146.22 seconds. Enabled rc6 pytest: 9 passed, 9 failed in 164.54 seconds. Four small-window tests and the generated sample fail on the documented public reds; #1225 fails as strict XPASS. Rc6 additionally fails the two #1227 exact shapes and its projection test. Three cheap oracle checks pass. These are report-only failures; the opt-out suite remains unchanged.

After the density-floor correction, the focused 60-case generated test still has only g51's R3 failure; all inputs measure 0.80022–1.46751 times their window. Its final GA/candidate wall times are 73.30/37.77 seconds and mean case times 1.208/0.488 seconds. The exact-shape receipts are unchanged; the final result recorded in the PR supersedes only the generated-sample receipt.

No production source, existing test, workflow, or root configuration changed. There was no push, PR, merge, release, or runtime action.

## Historical remaining evidence gate (superseded by the corrected public comparison)

The orchestrator should retain these unexplained cases and choose a bounded follow-up using the original historical host inputs for #1221, #1139, and #1225. The acceptance condition is the original failure on its culprit SHA and clearance on the fix while keeping the oracles unchanged. Treating a green blind spot or strict XPASS as proof of a fix would be misleading.
