# #1880 part A final paired preflight cost

Claim class: advisory. Synthetic CI host stub; full preflight. Baseline and candidate swap both changed engine methods and both changed compaction methods. No provider calls.

Product: `8d71947d164346609fc0b59782ee01d0532e2646`; baseline `10407e12e33c1d7b09fe0e05141153555aec5d62`.

| Session rows | n pairs | Base median ms | Candidate median ms | Paired median delta ms | Allowed increase ms | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 10000 | 200 | 0.1789795 | 0.1924790 | 0.0114590 | 0.05 | PASS |
| 100000 | 200 | 0.1626040 | 0.1726250 | 0.0103755 | 0.05 | PASS |

Receipt per row: `summary.json`, derived from raw `pairs.jsonl`. 200 alternating-order measured pairs after 10 warmup pairs per session. Both marginal and paired median increase must be <= max(base*10%, 0.05ms). P10/P90 are retained.

Re-score: `python score.py`.

Not proven: live reference-agent host-boundary savings, live turn latency, part B hold, merge, release, runtime or customer behavior.

T1: final issue-1880-preflight-cost rows in `bench/scoreboard/results.jsonl`. T2 tag: `lcm-x/1880-preflight-cost-r2-final-20261023`.
