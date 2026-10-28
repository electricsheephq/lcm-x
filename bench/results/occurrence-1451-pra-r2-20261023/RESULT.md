# occurrence-1451-pra-r2-20261023

Status: local proofs COMPLETE; final push and exact-head remote CI pending. Claim class: advisory. Measured source a2f5158972d9cd736a77c03fd23bf22fa399e683.

| Proof | Result | Receipt |
|---|---|---|
| BASE publication-fault control | 16->17 rows; new UID1; GREEN | base.log |
| r1 identical probe | 16->16; new UID0; expected RED | head.log |
| Fix identical probe | 16->17; new UID1; GREEN | green.log |
| Atomic/durability/retry-null/invalid shape | 35 passed in 3.05s | atomic.log |
| r1 fault+merge | 26 passed in 3.96s | r1-merge.log |
| Restored count/carry | 3 passed in 1.07s | r1-assertions.log |
| Actual v0.28.1 upgrade/rollback | 3 passed in 12.67s | compat.log |
| Matrix1296 vs supplied BASE | no new/increased loss; errors172->0;176 changed; no duplicate/growth rise | comparison-base.json, CHANGED-CELLS-base.md |
| Matrix vs r1 | 0 changed cells | comparison-r1.json, CHANGED-CELLS-r1.md |
| Expanded fault grid | 264 runs;192 fired,96 in timestamp/UID replacement arm; all keys unchanged/one postrollback warning/unavailable durable reader; no escaped exceptions | fault-summary.json, faults.json |
| Focused publication census | 372 valid,0 rejected;12 intentional invalid;266 passed,1 xfailed | census-summary.json, census.log |
| Retry-null fresh engine | exact JSON equality to BASE missing-snapshot control;16->17 | retry-cold-base.json, retry-cold-candidate.json |
| Commit path200 paired AB/BA | BASE1.5781035ms;fix1.3960005ms;paired delta-0.1714375ms;allowed+0.15781035ms;PASS | cost.json, cost-samples.json |
| Ruff0.15.13 / diff check | All checks passed! / exit0 | ruff-all.log |

## Re-score command

`python stream_compare.py supplied-BASE candidate-r2 comparison-base` and `python stream_compare.py preserved-r1 candidate-r2 comparison-r1`; raw inputs are retained with the complete local result card.

Raw outputs were preserved before scoring. Inputs are snapshots of the supplied immutable BASE/r1 records; this public card preserves only public-safe summaries.

Not proven: final pushed-head CI until read back; merge/release/runtime/fleet/customer readiness; live-host incidence or PR-B correctness. Inherited594 loss cells remain. The matrix has no publication-fault axis; separate permanent tests and fault grid cover that path.

T1: this public summary; registry row added before the same single push. T2: unpublished; full raw and re-score inputs retained. No result disposal.
