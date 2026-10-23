# #1880 persisted-hold integration cost recheck

Product: `8fc7b6a5eb7fc2e9415e07636c6266b50f0758af`; BASE: `10407e12e33c1d7b09fe0e05141153555aec5d62`.

| Rows / mode | BASE p50 ms | Candidate p50 ms | Candidate p95 ms | n | Receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| 10000 / absent | 0.003375 | 0.00925 | 0.011459 | 200 | PASS; summary.json, samples.jsonl |
| 10000 / mismatch | 0.0036875 | 0.0307505 | 0.036708 | 200 | PASS; summary.json, samples.jsonl |
| 10000 / held | N/A | 0.032625 | 0.034542 | 200 | PASS; summary.json, samples.jsonl |
| 10000 / rebuilt-held | N/A | 0.0816665 | 0.099542 | 20 | PASS; summary.json, samples.jsonl |
| 100000 / absent | 0.003625 | 0.009771 | 0.010375 | 200 | PASS; summary.json, samples.jsonl |
| 100000 / mismatch | 0.0040625 | 0.032292 | 0.033916 | 200 | PASS; summary.json, samples.jsonl |
| 100000 / held | N/A | 0.03425 | 0.039083 | 200 | PASS; summary.json, samples.jsonl |
| 100000 / rebuilt-held | N/A | 0.101688 | 0.151708 | 20 | PASS; summary.json, samples.jsonl |

All four UNHELD cells satisfy max(BASE*1.10, BASE+0.05ms). HELD makes zero model calls, full-history scans and metadata writes; SQL plans are indexed searches.

Protocol: same paired/interleaved steady workload,20 warmups then200 admissions per arm at10k/100k synthetic rows; held200 and reconstructed-held20. Preparation, seed, ingest and engine construction are excluded. Completed durable proof setup now explicitly uses `persist=True`, outside timing, matching the refined local-versus-durable API. No other instrument change from the prior steady run. One run, raw timings retained; no cross-run significance or model judge.

Run: `python -B cost_probe.py --base BASE --candidate CANDIDATE --scratch SCRATCH --out RUN`. BASE/CANDIDATE are exact source identities above; scratch contains regenerable stores only.

Re-score: `python score_cost.py .`; reads only the preserved samples.jsonl and summary.json. All eight cells PASS.

Not proven: reference-session latency, host prompt/tool refresh, ingest/reconstruction cost, #1162 repair, merge, release, runtime or customer readiness. Synthetic only.

T1: pending pinned registry row after this evidence commit. T2: not published; no results disposed.
