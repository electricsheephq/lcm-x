# #1451 PR-A r1 synthetic acceptance

Claim class: advisory. Product measured at f1df74ca5a051e33fca221fee06a5ba9c12b4f25, compared with supplied BASE matrix and the preserved r0 candidate. Full remote CI is recorded in the PR separately.

| Check | Result | Receipt |
|---|---|---|
| All 1,296 matrix cells, every checkpoint | 0 new/increased loss; errors 172 -> 0; 176 changed cells against supplied BASE | [summary](SUMMARY.json), [every BASE change](CHANGED-CELLS-base.md) |
| r1 against r0 matrix | 0 changed cells; 0 duplicate or restart-growth increases | [r0 changes](CHANGED-CELLS-r0.md) |
| Fault plus host merge, permanent regressions | RED 8/26 at bf57e64d; GREEN 26/26 | tests/test_issue_1451_r1_fault_merge.py |
| Old-writer upgrade, permanent regressions | actual v0.28.1 store; 335 -> 340/351 exactly equals BASE after last/all restamp | tests/compat/test_occurrence_snapshot_upgrade.py |
| Atomic fault grid | 132 runs, 96 fired; unchanged keys, exactly one WARNING after rollback, durable reader unavailable for all 96; no escaped exception | [fault summary](fault-summary.json) |
| Focused old-shape census | 606 valid publications, 0 rejected; all six original counterexamples included; 289 tests pass, 1 expected xfail | [census](census-summary.json) |
| Rollback reader | v0.28.1 opens candidate store and ingests; 1 passed | tests/compat/test_occurrence_snapshot_rollback.py |
| Restored assertions | rows+1; child rows==2; original carried_ids <= covered_ids in both arms; all pass | tests/test_compression_boundary.py; tests/test_probe_prrt_lu3v9.py |
| Commit path, 200 paired AB/BA samples | PASS; paired median delta -0.372313 ms, allowed +0.2273625 ms | [cost](cost.json), [samples](cost-samples.json) |
| Ruff 0.15.13 / actionlint / diff check | All checks passed! / exit 0 / exit 0 | exact commands below |

The matrix contains no publication-fault or old-writer upgrade axis; the permanent regressions and separate fault grid cover those paths. Inherited 594 loss cells remain. The supplied BASE matrix was generated at 5df5a3c, not implementation BASE10407e12; this run preserves that comparison boundary. There are zero increases in loss-row multisets, duplicates or positive restart growth at any checkpoint.

The legacy rule is derived from old writer cardinality and identity projection, not a scaffold-position allowlist. See [consumer table and writer citations](CONSUMERS.md).

Correction: the r0 receipt's "Test adaptations retain assertions" claim was inaccurate. r0 replaced exact count and carry-ID assertions with weaker bounds/content checks. r1 restores the original count and provenance assertions in the failed-write arm.

Exact checks: `uvx --from ruff==0.15.13 ruff check .` -> `All checks passed!`; `actionlint .github/workflows/ci.yml` -> exit 0; `git diff --check` -> exit 0. Test invocation details and full logs stay in the complete run packet.

Re-score: `python stream_compare.py inputs/red-base.jsonl candidate-r1.jsonl comparison-base.json` and `python stream_compare.py inputs/r0-candidate.jsonl candidate-r1.jsonl comparison-r0.json`. Raw inputs and runner are retained in the complete run packet; this committed directory is a public summary.

T1: results registry entry added with this summary. T2: unpublished; complete raw and re-score inputs preserved; no result disposal.
Not proven: merge readiness, release, live-host incidence, fleet/customer impact, PR-B carry rebind or the absence of inherited occurrence defects.
