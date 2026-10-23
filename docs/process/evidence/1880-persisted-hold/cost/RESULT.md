# #1880 B synthetic admission cost

Source candidate: `cd68a230e5d78cf7a8ad372e338cfd89acb528c6`; baseline: `10407e12e33c1d7b09fe0e05141153555aec5d62`.

| Rows | Gate | Base p50 ms | Candidate p50 ms | Delta ms | Candidate p95 ms | n | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 10000 | absent | 0.0036455000000000003 | 0.009208 | 0.005562999999999999 | 0.010000 | 200 | PASS |
| 10000 | mismatch | 0.003917 | 0.029166 | 0.025249499999999998 | 0.034792 | 200 | PASS |
| 10000 | held | N/A | 0.031896 | N/A | 0.037917 | 200 | PASS |
| 10000 | rebuilt-held | N/A | 0.087083 | N/A | 0.110292 | 20 | PASS |
| 100000 | absent | 0.003792 | 0.009083 | 0.0052910000000000006 | 0.009542 | 200 | PASS |
| 100000 | mismatch | 0.0044375000000000005 | 0.030791 | 0.0263535 | 0.033500 | 200 | PASS |
| 100000 | held | N/A | 0.032313 | N/A | 0.034334 | 200 | PASS |
| 100000 | rebuilt-held | N/A | 0.084917 | N/A | 0.103500 | 20 | PASS |

Paired, interleaved steady-state admissions, 20 warmups and 200 samples per arm. The absent and nonmatching metadata records remain unchanged throughout each steady cell; no per-sample writer churn. Held reconstruction measures admission after the current replay proof is established; engine construction and ingest are excluded. Raw samples are retained; one local run, no cross-run statistical claim. No model judge.

Unheld median limit: `max(base * 1.10, base + 0.05 ms)`. Held p95 limits: warm 50 ms, rebuilt 100 ms. Held admissions make zero model calls, full-history scans or metadata writes; bounded indexed query plans are recorded in `summary.json`.

Earlier instruments: the first used inconsistent timestamp reads and is excluded; two preserved writer-churn trials repeatedly rewrote the record outside the gate and are separate from the specified steady workload. The first valid trial prompted caching resolved store identity. The corrected steady workload and that optimization are disclosed rather than combining their timings.

## Re-score

```sh
python score_cost.py .
```

## Reproduce

```sh
python -B cost_probe.py --base BASE --candidate CANDIDATE --scratch SCRATCH --out RUN
```

BASE is the root Python modules from the pinned baseline. CANDIDATE is the pinned checkout. SCRATCH is an empty regenerable directory; RUN is a new evidence directory. Use a supported Python with the repository test dependencies.

## Not proven

Not proven: engine-construction or ingest latency, host prompt/tool refresh, live latency, the retained-occurrence repair (#1162), merge, release, runtime or customer readiness.

## Publication

T1: the registry row added with this PR. T2: not published; no result data was disposed. Public-safe raw timings, query plans, instrument and scorer accompany this card.
