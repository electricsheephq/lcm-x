# PR #1860 fix round 1 acceptance

Claim class: advisory. Measured production source: `566c498c0ed8f256946e9aed2352a12ba571580b`.

## Fix round 1

This section supersedes the earlier final-head acceptance limits. The revision guard delays metadata-only extension until revision certification; cloud validation supplies coverage to its ranges and walk state; a low walk without a captured parent returns unproven; validation-hook exceptions are debug-logged so summary maintenance continues.

The RED-first regressions cover v0.28.1 mid-repair stores, a simulated revision bump, current-revision compatibility, the cloud legacy wider-scan shape, real manual provider-skip/privacy-withheld paths, NULL-parent capture and summary-hook isolation. The focused regression file passed all ten cells. No production bytes changed during this final validation round.

### Base / previous head / fixed probes

| Probe | Base 91a03989 | Previous head 04dc9385 | Fixed 566c498c |
|---|---|---|---|
| d1b (both arrival shapes) | repaired; no effective false certificate | row 2 falsely certified | repaired; no effective false certificate |
| d2 (both revision-bump shapes) | repaired | row 2 falsely certified | repaired |
| d3 current-revision control | sound | sound | sound |
| g legacy wider scan | row 3 visible then embedded | row 3 hidden by false certificate | row 3 visible then embedded |
| B1 manual skip/withheld | 2 (gap-fill); row 4 visible | 1; row 4 incorrectly covered | 2 (gap-fill); row 4 visible |
| b_parent_absent | N/A: no independent low-band key | row 4 falsely certified | safe early return before yield (EXPECTED_NEGATIVE) |

The NULL-parent review probe assumes a first yielded row. Its original `next(low)` now raises `StopIteration` at the prescribed safe early return, before the probe deletes low row 4. This is an expected negative of that old setup, preserved as such; it is not a passing pytest result. The committed regression checks an empty result, `unproven`, and no coverage marker directly.

### Oracle ruling and golden

The amended bar requires zero new undercount, false-coverage or false-certificate events; false certificates remain confined to the inherited legacy seeds 150, 212, 422 and 491. New overcounts are allowed only when final remaining/oracle work state after convergence matches base on every seed.

The preserved 600-seed runs in each arm plus the release-manager convergence probe satisfy that bar. All four probe controls reproduce the preserved events for all 600 seeds each. There are no new false-zero states or stuck overcounts after convergence, and final work counts match base on all 1,200 seed-runs. The three temporary legacy overcounts (207:7, 263:8 and 421:12) contain no missing rows and end at remaining 0 = oracle 0. Marker serialization differs on some seeds; work-count equality is not a claim of identical metadata. The oracle was not rerun because production hashes are unchanged. Inherited baseline limitations remain.

Only `chunks/incremental-revision-gap-checkpoint` changes in this fix round; 79 other canonical traces are identical. The revision guard replaces premature metadata-extension SQL with a revision read and preserves pending work until certification. Reports, rendered output and provider calls are unchanged; differences are confined to SQL ordering and saved metadata. Golden commit `566c498c` is retained. The PR carries no committed r2 oracle probe, so adding its after-convergence end-state record remains a follow-up rather than a code change here.

Testing improved: the revision-bump and v0.28.1 mid-repair false-certificate cells, the cloud legacy wider-scan cell, the NULL-parent low CAS cell and the hook-isolation cell. The external convergence probe adds a final pending-count liveness record.


## Remaining validation gates

| Gate | Observed result | Receipt |
|---|---|---|
| Bounds r2 | 6 passed; four long histories, at most 3 scans/call, no width violations or missing rows; 0 certified-steady scans | SUMMARY.json |
| Read-only status | 3 steady scenarios, no authorizer writes or data changes | SUMMARY.json |
| Full CI-style suite | 19511 passed, 141 skipped, 63 xfailed, 29 warnings in 1116.52s (0:18:36) | SUMMARY.json |
| Pinned ruff | All checks passed! | Commands below |
| Whitespace | exit 0, no output | Commands below |

## Steady cost delta

Paired/interleaved 200 samples per arm; threshold max(base × 1.10, base + 0.05 ms). Exact samples are retained in SUMMARY.json.

| Cell | Base p50 ms | Fixed p50 ms | Bound ms | Result |
|---|---:|---:|---:|---|
| pass | 2.2181 | 2.2142 | 2.4399 | PASS |
| status | 0.2709 | 0.2622 | 0.3209 | PASS |
| fresh-pass | 2.1891 | 2.2126 | 2.4080 | PASS |
| fresh-status | 0.2320 | 0.2331 | 0.2820 | PASS |

Arrival cost: sequential 30 samples per tree; threshold max(base × 1.5, base + 1 ms).

| Arm | Pass p50 ms | Status p50 ms |
|---|---:|---:|
| base | 14.314542 | 11.455063 |
| fixed | 4.229563 | 0.383479 |

All arrival-cost bars pass. Natural low-band cases fill holes on call 1 and certify on call 3; the 600-row case certifies on call 20. No missing rows, width/read-only violations or hidden-hole false-zero reports.

## Evidence

Machine-specific interpreter and scratch roots are normalized below; expanded argv, source hashes, raw bounds/liveness outputs and all inherited oracle inputs are retained under T2 tag `lcm-x/lcmx-1860-fix-r1c-20261023`.

```bash
bash -lc 'ulimit -n 1024 && python -m pytest tests/ -q -p no:cacheprovider --basetemp=$SCRATCH/pt-full -o tmp_path_retention_policy=failed'
19511 passed, 141 skipped, 63 xfailed, 29 warnings in 1116.52s (0:18:36)
uv tool run --from ruff==0.15.13 ruff check bench/fmp/delivery.py bench/fmp/judges.py bench/fmp/models.py bench/fmp/run.py bench/fmp/score.py bench/fmp/tests/test_dev.py bench/fmp/tests/test_fmp.py bench/fmp/tests/test_mcp_readiness.py bench/fmp/tests/test_r3.py bench/fmp/tests/test_reader_retry.py bench/fmp/tests/test_reader_retry_r1.py bench/fmp/tests/test_reader_retry_r2.py bench/fmp/tests/test_reader_retry_r3.py bench/fmp/tests/test_session_delivery.py bench/instruments/local_catchup/catchup.py bench/instruments/local_catchup/measurement.py bench/instruments/local_catchup/recall.py bench/instruments/local_catchup/rescore.py bench/instruments/local_catchup/rig.py bench/instruments/local_catchup/tests/conftest.py bench/instruments/local_catchup/tests/test_1586_fix1.py bench/instruments/local_catchup/tests/test_1586_fix2.py bench/instruments/local_catchup/tests/test_1586_v2.py bench/instruments/local_catchup/tests/test_1586_v23.py bench/instruments/local_catchup/tests/test_1586_v23_r1.py bench/instruments/local_catchup/tests/test_1586_v23_r2.py bench/instruments/local_catchup/tests/test_1586_v23_r3.py bench/instruments/local_catchup/tests/test_1653_round2.py bench/instruments/local_catchup/tests/test_1653_round3.py bench/instruments/local_catchup/tests/test_1760.py bench/instruments/local_catchup/tests/test_1767.py bench/instruments/local_catchup/tests/test_fix_round2.py bench/instruments/local_catchup/tests/test_fix_round3.py bench/instruments/local_catchup/tests/test_fix_round4.py bench/instruments/local_catchup/tests/test_pilot_handles.py benchmarks/results/render_results.py command.py embedding_maintenance.py evidence_tools.py occurrence_time.py reasoning.py requirements_compiler.py tests/issue_1811_r2_cells.py tests/test_issue_1586_gap_discovery.py tests/test_issue_1586_local_precursor.py tests/test_issue_1624_cloud_stream_liveness.py tests/test_issue_1801_carried_followups.py tests/test_issue_1801_twin_parity.py tests/test_issue_1811_contract_mode.py tests/test_issue_1811_occurrence_time_batch.py tests/test_issue_1811_round3.py tests/test_issue_1860_certificate_guards.py tests/test_longmemeval_medium_r2_core.py tests/test_public_results_page.py
All checks passed!
git diff --check
# exit 0 (no output)
git diff 91a039891afd969f9569897115990fe0c419e034 HEAD --check
# exit 0 (no output)
python rescore.py
# PASS: preserved timing samples, liveness and read-only summaries
```

Not proven: exact-head remote CI at report publication, independent final-head review, merge, release, installed runtime or customer behavior. The inherited oracle limitations remain. T1 registry is on this PR; raw T2 uses the tag above.

Full-suite process exit: 0. The preserved raw log also contains three post-summary shutdown import diagnostics for the synthetic `issue1603_profile_0` fixture module; no pytest node failed.
