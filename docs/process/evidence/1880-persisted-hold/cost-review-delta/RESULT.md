# #1880 B synthetic admission cost

Source candidate: `970e47f9b5ee33cd6cca2c71a27d777683eb412e`; baseline: `10407e12e33c1d7b09fe0e05141153555aec5d62`.

| Rows | Gate | Base p50 ms | Candidate p50 ms | Delta ms | Candidate p95 ms | n | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 10000 | absent | 0.003416 | 0.008458 | 0.005042 | 0.008958 | 200 | PASS |
| 10000 | mismatch | 0.0040205 | 0.030667 | 0.0266465 | 0.032500 | 200 | PASS |
| 10000 | held | N/A | 0.032333 | N/A | 0.034084 | 200 | PASS |
| 10000 | rebuilt-held | N/A | 0.083750 | N/A | 0.111333 | 20 | PASS |
| 100000 | absent | 0.0035 | 0.008625 | 0.005125000000000001 | 0.009375 | 200 | PASS |
| 100000 | mismatch | 0.0039165 | 0.029916 | 0.025999500000000002 | 0.031250 | 200 | PASS |
| 100000 | held | N/A | 0.032230 | N/A | 0.036291 | 200 | PASS |
| 100000 | rebuilt-held | N/A | 0.089292 | N/A | 0.115458 | 20 | PASS |

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

This run rechecks the consolidated review correction: completed-proof classification, L2 budget/prompt-version settings and non-secret current route identity. The previous steady run remains a separate historical result.
