# PR #1860 r2: initial read-only low-band backlog visibility

Claim class: merge-candidate fix with comparative probes and oracle evidence (advisory; no merge-readiness claim).
Measured source: `59d6118ee2e211a8364ff8e06ee75f29c4d396af`.

The d3 initial state has eligible unembedded row 2. The admitted source reported `0`, while base reported `1 (gap-fill)`. The missing-parent early return applied to a read-only walk, yielding no rows; the status estimate did not retain its unproven state. The fix requires the parent fence only when `record_progress=True`. Writable low walks retain the same missing-parent refusal.

The new initial-status regression failed at admitted `4c541493`, was committed RED as `b3df61c6`, and passes after the fix. It also checks that status changes no database contents or total_changes. All ten existing certificate-guard cells are unchanged and pass. Production delta: two additions and two deletions.

## Mechanical probe comparison

Every emitted step compares status, embedded rows, false rows and effective_false. This is 38 states and 152 field comparisons per comparator. The fixed negative self-check identifies precisely the d3 initial harmful difference (0 versus 1 while row 2 is missing). The candidate matches all required base fields and has only the benign d3 increase against fixed.

| Probe | Fields vs base | Harmful vs base | Benign vs base | Harmful vs fixed | Benign vs fixed |
|---|---:|---:|---:|---:|---:|
| d1b_v0281_midrepair_arrival | 24 | 0 | 0 | 0 | 0 |
| d1b_v0281_midrepair_no_arrival | 24 | 0 | 0 | 0 | 0 |
| d2_revision_bump_arrival | 20 | 0 | 0 | 0 | 0 |
| d2_revision_bump_selection_found_gap | 20 | 0 | 0 | 0 | 0 |
| d3_current_control | 20 | 0 | 0 | 0 | 1 |
| g_cloud_legacy_wider_scan | 20 | 0 | 0 | 0 | 0 |
| k-real-manual-fail | 8 | 0 | 0 | 0 | 0 |
| k-real-manual-skip | 8 | 0 | 0 | 0 | 0 |
| k-real-manual-withheld | 8 | 0 | 0 | 0 | 0 |
| b_parent_absent_capture | N/A: no truth emitted | 0 | expected negative | 0 | same safe early return |

The original parent probe calls next(low) before it can emit truth. Base skips because it has no independent low key. Fixed and candidate raise StopIteration at the same safe early return. This is an expected negative, not a passing pytest result or an invented truth snapshot. The unchanged committed F3 regression directly proves empty rows, unproven state and no coverage marker.

Re-score the public probe inputs:

```bash
python3 probe_diff.py.txt --candidate fixed --output negative-selfcheck.json
# expected exit 1: d3 initial harmful difference
python3 probe_diff.py.txt --candidate head --output recomputed-probe-diff.json
# exit 0: PASS, zero harmful differences
```

## Oracles

| Arm | Seeds | New undercount | New false coverage | New false certificate | Different final work state |
|---|---:|---:|---:|---:|---:|
| legacy | 600 | 0 | 0 | 0 | 0 |
| nolegacy | 600 | 0 | 0 | 0 | 0 |

Original r1b instruments ran unchanged in each arm, followed by the existing RM convergence observation. Its non-convergence events equal the original instrument for every seed. Final comparison is policy, remaining and oracle work count, per the preserved ruling; marker serialization is not asserted equal. Inherited baseline limitations remain. Legacy integer normalization includes already-existing certificates at seeds 107 and 450, as well as the previously reported JSON certificate seeds 150, 212, 422 and 491. There are zero new certificates against base.

Preliminary scorer diagnostics are preserved in the private working receipts: different missing-row tuples with smaller/equal inherited deficits and integer/JSON marker serialization initially produced false classifications. Scoring was corrected against saved records and the r1b step/kind contract; no oracle/test assertion, threshold or production source changed and no oracle was rerun. Positive/negative checks distinguish a newly covered missing row and an increased deficit. Raw oracle and convergence receipts are retained by the lane; this public snapshot contains their aggregate comparison, not their full re-score inputs.

## Evidence

Environment roots are normalized as in earlier evidence. The interpreter is the prescribed venv, with the prescribed host PYTHONPATH and bytecode disabled. Exact expanded argv are retained by the lane.

```bash
python -m pytest tests/test_issue_1860_initial_visibility.py -q -p no:cacheprovider --basetemp="$SCRATCH/pt-red" -o tmp_path_retention_policy=failed
1 failed in 0.84s
python -m pytest tests/test_issue_1860_initial_visibility.py tests/test_issue_1860_certificate_guards.py -q -p no:cacheprovider --basetemp="$SCRATCH/pt-focused" -o tmp_path_retention_policy=failed
11 passed in 1.21s
python -m pytest tests/ -q --basetemp="$SCRATCH/pt-full" -o tmp_path_retention_policy=failed -p no:cacheprovider
19512 passed, 141 skipped, 63 xfailed, 29 warnings in 1400.36s (0:23:20)
uv tool run --from ruff==0.15.13 ruff check .
All checks passed!
git diff --check
# exit 0 (no output)
```

The full-suite process exited 0. Its raw log preserves three post-summary synthetic-fixture shutdown import diagnostics, also observed in r1c; no pytest node failed. There was one full suite, no second full run.

Testing improved: a reusable mechanical base/head probe differ that classifies harmful and benign status differences per step, replacing prose summaries. It also adds a pinned d3 initial-visibility test.

T1: append-only result row on this PR. T2: not published; raw outputs are retained and no results are packed, moved or deleted.

Not proven: production backlog accuracy beyond these probes, independent final-head review, exact-head remote CI at this snapshot publication, merge, release, installed runtime or customer behavior.
