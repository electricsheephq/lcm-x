# B4 drained-coverage acceptance

Claim class: advisory. Instrument `6efbee1427db764ad2a967704f4040294d46f6f0`; pinned ref-0.21.5 host `f97608f178d1ffeca59860195ab7da295f7c8e5f`. In-process driver and scripted provider, isolated homes, unchanged pinned products.

| Cell receipt | Product | Baseline B4 | Candidate B4 | Final engine status | Coverage |
| --- | --- | --- | --- | --- | --- |
| [long-80/in-place](ga-L-inplace-result.json) | `77b2b4f795fb` | PASS | PASS | compacted | existing publication success |
| [long-80/rotation](ga-L-rotation-result.json) | `77b2b4f795fb` | PASS | PASS | compacted | existing publication success |
| [long-80/in-place](rc1-L-inplace-result.json) | `df9e8b490747` | INCONCLUSIVE | PASS | noop | 143 covered + 23 protected; 0 uncovered |
| [long-80/rotation](rc1-L-rotation-result.json) | `df9e8b490747` | INCONCLUSIVE | PASS | noop | 143 covered + 23 protected; 0 uncovered |
| [publication-failure/rotation-child-persistent](rc1-P-persistent-result.json) | `df9e8b490747` | FAIL | N/A | error | expected-negative diagnostic retained |
| [publication-failure/rotation-child](rc1-P-sibling-result.json) | `df9e8b490747` | PASS | PASS | compacted | existing publication success |

All six SPEC acceptance cells match their expected result. The persistent cell retains raw B1/B2 FAIL diagnostics: B1 mismatches 15; B2 deficits 0 and surplus 23, exactly matching the RCA baseline. Its B4 error and 2 conflict-log mentions remain recorded, without making B4 applicable. This is not a blanket PASS of the persistent cell.

## Run command

Copy of the RCA execute_cell.py launch/isolation/retention approach, using candidate instruments and new detached product worktrees. The full local packet retains exact invocations and isolation roots.

```sh
python execute_cell.py rc1-L-inplace <df9e8b490747-product-tree> long-80/in-place
python execute_cell.py rc1-L-rotation <df9e8b490747-product-tree> long-80/rotation
python execute_cell.py ga-L-inplace <77b2b4f795fb-product-tree> long-80/in-place
python execute_cell.py ga-L-rotation <77b2b4f795fb-product-tree> long-80/rotation
python execute_cell.py rc1-P-sibling <df9e8b490747-product-tree> publication-failure/rotation-child
python execute_cell.py rc1-P-persistent <df9e8b490747-product-tree> publication-failure/rotation-child-persistent
```

## Re-score command

```sh
PYTHONDONTWRITEBYTECODE=1 python rescore.py <checkout-at-6efbee1427db764ad2a967704f4040294d46f6f0>
```

Full-packet re-score reads retained standalone DB and payload snapshots; this public view contains payload-free acceptance receipts, not the DBs. Every verdict, applicable/failed/inconclusive bar and B4 diagnostic matches the original receipt.

Not proven: model quality, latency, unexecuted hosts, release, deployed runtime or customer readiness. Hosted CI and review have separate delivery receipts.

## Links

PR: https://github.com/electricsheephq/lcm-x/pull/1952

Baseline card: lcmx-v0290-b4-rca-20261024. T1/T2 not published; results and re-score inputs remain preserved.
