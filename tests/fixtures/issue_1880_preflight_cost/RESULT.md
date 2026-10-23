# #1880 part A Amendment 1 paired preflight cost

Claim class: advisory. Synthetic CI host stub, full preflight; base and candidate swap both changed engine methods and both changed compaction methods. No provider calls.

Code: `2cbb01e37e461ab439620514ecb05f92020620ae`; baseline `10407e12e33c1d7b09fe0e05141153555aec5d62`.

| Session rows | n paired | Base median ms | Candidate median ms | Paired median delta ms | Allowed increase ms | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 10000 | 200 | 0.1621250 | 0.1664170 | 0.0031040 | 0.05 | PASS |
| 100000 | 200 | 0.1662505 | 0.1691875 | 0.0031880 | 0.05 | PASS |

Receipt per row: `summary.json`, derived from 200 alternating-order pairs after 10 warmup pairs per session in `pairs.jsonl`. Both marginal and paired median increase must be <= max(base * 10%, 0.05ms). P10/P90 retained in summary.

Re-score: `python score.py` (reads preserved `pairs.jsonl`).

Not proven: live host boundary or reference-agent latency, release, runtime, customer behavior, or part B no-progress hold.

T1: the two issue-1880-preflight-cost-r2 rows in `bench/scoreboard/results.jsonl`. T2 tag: `lcm-x/1880-preflight-cost-r2-20261023`.

Public-safe copy: `measurement-source.txt` preserves the measurement source as data; the original executable source and all raw inputs also remain with the T2 receipt.
