# PR #1826 — Fix round 2

Claim class: advisory — local fixture proof for review; no answer-quality claim.
Candidate: [907043cb7ba9dbe5c5158b801a61acca0f9898c4](https://github.com/electricsheephq/lcm-x/commit/907043cb7ba9dbe5c5158b801a61acca0f9898c4). Base: `9eacedb9a667e517e40b9ed5c1725aa4f44df1cf`. Default-mode predecessor: `e06cc78879cc345c153a9e6afdc5eb3cde638c2e`. Tests first: `166de31f4e0e3957468213cf689a7beeb4745b0d`; test-only fixture correction: `f73f032f238872034908fcdcee3e792baec1aacd`.

## Ruling B

Non-date contracts use base occurrence semantics throughout normalization, carried-date helpers, availability/latest readers and grounding. Date dependence remains exactly `temporal_window is not None or operation in {date_filter, order, date_interval}`. This deliberately preserves base borrowed-day behavior for non-date latest/as-of operations; its follow-up belongs to the release manager.

One coherent order-label optimization stores short stripped labels in the existing bounded quote parse. Weekday patterns are precompiled and the full date-lookahead strip runs only after a cheap weekday precheck. Large-row cache values never retain label/source text. The large-map race returns the parsed local value after insertion. Grammar, relative resolution, policy version, recall source, ingest and stored rows are unchanged.

## Caller / mode table

| Consumer | Mode and boundary |
| --- | --- |
| `requirements_compiler._normalize_ref` (337) | `_date_dependent(contract)`; every baseline, adjacent-role, targeted-retrieval and finite-scan call supplies the contract. Direct helper calls without a contract retain default True. |
| `_date_candidates` (673) | `_date_dependent(contract)` on each exact date match. |
| `_carried_mixed_dates` (388) | Non-date carried rows use the base resolver with False in the original carried predicate; the hand-built ISO replica is removed. Date-dependent rows keep Amendment A. Ordinary non-date rows never enter this filter. |
| `_carried_absolute_day` / `_carried_known_day` | Receive the same date-dependent flag; use the shared named-date grammar and the selected carried predicate. |
| `_candidate_event_day` / `_available_as_of` | Pass `_date_dependent(contract)` to carried helpers and consume occurrence metadata already normalized in that mode. |
| `_candidate_time` / `_select_state` | Consume the normalized occurrence object without re-resolving a narrowed candidate; this preserves base latest/as-of behavior, including borrowed days under Ruling B. |
| Source pre-loop / finite scan | Source pre-loop is gated by `_date_dependent`; finite scan requires a temporal window. Their True defaults are therefore date-dependent only. |
| Compiler `_compute` / `_finite_enumeration` grounding | Pass `_date_dependent(contract)` into `ground_evidence`. |
| `reasoning.resolve_occurrence_time` (88) | Lazy wrapper forwards `withhold_mixed`, default True. |
| `reasoning._ground_one` (871) | Re-resolves the exact quote with the mode received through `ground_evidence`; compiler normalization and grounding agree. |
| `evidence_pack._resolve_candidate` (381) | Default True. It feeds the separate pack / legacy proposal path and its default-mode grounding, not the active AnswerContract compiler. `evidence_compiler.py:1307-1316` explicitly delegates the active path directly to requirements_compiler and preserves the legacy proposal compiler. No evidence_pack edit. |
| `recall_delivery` (248) | Default True, unchanged, per accepted Amendment 1. |
| `recall_session` (206) | Default True, unchanged, per accepted Amendment 1. |

Production inventory: exhaustive tracked Python grep for `resolve_occurrence_time`; no other production call sites. Test callers in `test_occurrence_time.py` and `test_issue_1811_occurrence_time_batch.py` retain True. The new mode regression explicitly exercises False and True. The standalone proof checks False with a withholding-reason helper that raises if called; zero calls occurred.

## RED / GREEN and identity

- Tests-first RED at the predecessor: `619 failed, 891 passed in 21.36s`. This includes all 609 reviewer-differing cells, five named repros, four mode tests and the cache-clear race.
- The committed guard contains 1,500 deterministic stratified cells, including every one of the 540 + 69 differences by generator index. It covers whole cited quotes, question dates None / 2023-03-22, two-candidate latest selection and carried mixed-separator ISO dates. The full 7,656-cell base capture took 172.07 s, so the allowed subset is committed; its four-worker runs meet the under-60-second target.
- Full non-date artifact: **7,656 / 7,656 byte-identical to base**, zero differences. Five named repro tests assert base operation/state/reason/value/computation outputs.
- Both resolver modes: **292,632 comparisons each, zero byte differences**: 40,000 generated comparisons (10,000 quotes × four session-date values), 246,750 LongMemEval-S occurrences and 5,882 LoCoMo occurrences. False equals base, default True equals the predecessor. Counts/digests only; no corpus text or identifiers saved or published.
- Date-dependent matrix: **14,552 / 14,552 byte-identical to the predecessor**; zero new answers and zero day changes versus base. Changed-cell list: empty.
- Focused GREEN: `2298 passed, 9 warnings in 31.81s`. Existing 12 pinned hashes, label/twin/cache controls and prior base hashes pass unchanged. Three recall golden files remain byte-identical to both base and predecessor; no pinned or golden edits.
- One hand-written test originally expected the predecessor to withhold a mixed-separator quote containing only one named day. The saved predecessor also returned the ISO day. A test-only correction uses the reviewer's conflicting-two-named-day shape; the production candidate did not change.

## Cost

| Operation | Input refs | Warm base ms | Warm head ms | Delta ms | Bar ms | Cold base ms | Cold head ms | Warm gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| date_filter | 2 | 0.104125 | 0.106688 | +0.002563 | 0.050000 | 0.104979 | 0.148770 | PASS |
| date_filter | 6 | 0.173104 | 0.179812 | +0.006708 | 0.050000 | 0.184271 | 0.292729 | PASS |
| date_filter | 10 | 0.229500 | 0.237958 | +0.008458 | 0.050000 | 0.273271 | 0.469396 | PASS |
| date_filter | 12 | 0.325375 | 0.341104 | +0.015729 | 0.050000 | 0.259250 | 0.449438 | PASS |
| date_filter | 24 | 0.316667 | 0.323958 | +0.007292 | 0.050000 | 0.275250 | 0.467062 | PASS |
| order | 2 | 0.175063 | 0.177500 | +0.002437 | 0.050000 | 0.188666 | 0.234375 | PASS |
| order | 6 | 0.281438 | 0.271562 | -0.009875 | 0.050000 | 0.278375 | 0.371395 | PASS |
| order | 10 | 0.421791 | 0.409562 | -0.012229 | 0.050000 | 0.407187 | 0.551666 | PASS |
| order | 12 | 0.509083 | 0.493563 | -0.015520 | 0.050908 | 0.422104 | 0.614854 | PASS |
| order | 24 | 0.437001 | 0.422104 | -0.014897 | 0.050000 | 0.483292 | 0.667729 | PASS |
| date_interval | 2 | 0.184500 | 0.189688 | +0.005188 | 0.050000 | 0.182334 | 0.236084 | PASS |
| date_interval | 6 | 0.499104 | 0.520416 | +0.021312 | 0.050000 | 0.545166 | 0.698042 | PASS |
| date_interval | 10 | 0.745812 | 0.780312 | +0.034500 | 0.074581 | 0.750750 | 1.014250 | PASS |
| date_interval | 12 | 0.521271 | 0.533937 | +0.012666 | 0.052127 | 0.518208 | 0.793646 | PASS |
| date_interval | 24 | 0.578896 | 0.606834 | +0.027938 | 0.057890 | 0.524042 | 0.796542 | PASS |
| finite count + window | 2 | 0.474187 | 0.483791 | +0.009604 | 0.050000 | 0.420417 | 0.479813 | PASS |
| finite count + window | 6 | 0.976042 | 0.964000 | -0.012042 | 0.097604 | 1.039083 | 1.199375 | PASS |
| finite count + window | 10 | 1.360979 | 1.366271 | +0.005292 | 0.136098 | 1.373355 | 1.560583 | PASS |
| finite count + window | 12 | 1.695375 | 1.618354 | -0.077021 | 0.169538 | 1.538417 | 1.813438 | PASS |
| finite count + window | 24 | 2.156687 | 2.301833 | +0.145146 | 0.215669 | 2.195104 | 2.672562 | PASS |
| non-date | 2 | 0.209146 | 0.210687 | +0.001542 | 0.050000 | 0.175583 | 0.192166 | PASS |
| non-date | 6 | 0.355750 | 0.362854 | +0.007104 | 0.050000 | 0.417291 | 0.423833 | PASS |
| non-date | 10 | 0.581458 | 0.585542 | +0.004083 | 0.058146 | 0.576145 | 0.564063 | PASS |
| non-date | 12 | 0.323271 | 0.327750 | +0.004479 | 0.050000 | 0.302667 | 0.310708 | PASS |
| non-date | 24 | 0.342625 | 0.350875 | +0.008250 | 0.050000 | 0.374000 | 0.382792 | PASS |

Every warm cell passes `head p50 - base p50 <= max(10% of base p50, 0.05 ms)`. Each row has 200 samples per arm, interleaved AB/BA in twenty ten-call blocks after thirty warmups. Cold: 200 samples per arm, report only; the head bounded quote caches are cleared per call. Process startup and setup are excluded. This is one paired run after one optimization; no timing rerun. The 1801 fixture shape, inputs, question dates and clock match between arms. No model calls or measured store writes. Point estimates; confidence intervals N/A. The ordinary hydration cap remains 12, including the 24-input-ref arm; finite scan still sees its store's rows.

## Suites and checks

Focused command:

```sh
python -m pytest tests/test_issue_1811_contract_mode.py tests/test_issue_1811_occurrence_time_batch.py tests/test_issue_1801_twin_parity.py tests/test_issue_1801_carried_followups.py tests/test_issue_1779_carried_time.py tests/test_occurrence_time.py -q -n 4 --tb=short -p no:cacheprovider --basetemp="$LCMX_SCRATCH/pt-focused" -o tmp_path_retention_policy=failed
```

```text
2298 passed, 9 warnings in 31.81s
```

The 32 matched files, in the exact order recorded in `matched-files.json`, use `python -m pytest <those files> -q -p no:cacheprovider --basetemp="$LCMX_SCRATCH/pt-matched" -o tmp_path_retention_policy=failed --tb=short`:

```text
head: 8 failed, 3179 passed, 3 xfailed, 21 warnings in 110.05s (0:01:50)
base: 419 failed, 2767 passed, 4 xfailed, 21 warnings in 138.39s (0:02:18)
```

The matched suite is **not green**. All eight head failures reproduce at base in the identical file order: four provider-alias coverage cases, the preflight-expiry coverage case (#1829), `test_bounded_read_stop[clock]`, and both `test_fix1_deadline_seeds` cases. Base-only failures are 411 expected failures in the overlaid new regression tests. Six new test/fixture files were added untracked to this lane's detached base solely to collect the identical command; no existing base source or test was overwritten. No recall/timing remediation or tracker write in this lane.

The requested full suite ran once, at `907043cb7ba9dbe5c5158b801a61acca0f9898c4`, one worker with a 2,400-second ceiling, the original host shim and isolated temporary root:

```sh
ulimit -n 1024
python -m pytest tests/ -q -p no:cacheprovider --basetemp="$LCMX_SCRATCH/pt-full" -o tmp_path_retention_policy=failed
```

```text
19402 passed, 141 skipped, 63 xfailed, 29 warnings in 1460.34s (0:24:20)
```

Full-suite disposition: Full suite GREEN at the final commit: 19,402 passed, 141 skipped, 63 xfailed; no full-suite failure. The separate matched-order failures remain disclosed with their base reproduction.

```sh
uv tool run --from ruff==0.15.13 ruff check . --no-cache
git diff --check
git diff 9eacedb9a667e517e40b9ed5c1725aa4f44df1cf HEAD --check
```

```text
All checks passed!
git diff --check: exit 0, no output
base-to-head diff --check: exit 0, no output
```

Production delta this round: **98 changed lines (+63/-35)** across the three allowed source files; no second production attempt after validation. Source digests match the fixture, matrix, resolver and cost candidate exactly. Final source commit only records those frozen bytes. No merge or rebase of main.

Testing improved: the non-date base-identity guard covers cited-quote noise, question dates, latest selection and carried separators, and the cost proof covers every date-dependent op.

Not proven: answer-quality improvement, production workload latency, independent semantic review, hosted exact-head CI, merge, release, installed runtime or customer behavior. Raw outputs remain preserved. T1/T2 registry/archive publication is outside this source lane; public-safe proof is linked separately. Release manager owns independent review and split/park decisions; this lane makes no review ping, reply, resolution, enqueue or merge.
