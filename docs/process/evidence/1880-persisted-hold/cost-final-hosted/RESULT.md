# #1880 final hosted admission cost: PASS

Measured source:441c338ff49d3d1bd64a864a9191ce2723c509bb; product source778609b68bc39ef4f216728ccb2952d36b53a8b1. BASE10407e12e33c1d7b09fe0e05141153555aec5d62.
Runner: GitHub-hosted Ubuntu, Python3.13; run39912662905/job120278995028.

| Rows / mode | BASE p50 ms | Candidate p50 ms | Candidate p95 ms | n | Receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| 10000 / absent | 0.005328 | 0.008312 | 0.008613 | 200 | PASS; summary.json and samples.jsonl |
| 10000 / mismatch | 0.005593 | 0.0351775 | 0.038007 | 200 | PASS; summary.json and samples.jsonl |
| 10000 / held | N/A | 0.037461 | 0.044006 | 200 | PASS; summary.json and samples.jsonl |
| 10000 / rebuilt-held | N/A | 0.14063599999999998 | 0.167221 | 20 | PASS; summary.json and samples.jsonl |
| 100000 / absent | 0.005548 | 0.008603 | 0.008983 | 200 | PASS; summary.json and samples.jsonl |
| 100000 / mismatch | 0.0057135 | 0.0354435 | 0.039509 | 200 | PASS; summary.json and samples.jsonl |
| 100000 / held | N/A | 0.037737 | 0.043646 | 200 | PASS; summary.json and samples.jsonl |
| 100000 / rebuilt-held | N/A | 0.14606449999999999 | 0.15936 | 20 | PASS; summary.json and samples.jsonl |

All four UNHELD cells satisfy max(BASE*1.10,BASE+0.05ms). HELD makes zero model calls, full-history scans and metadata writes; indexed query plans are preserved.

Protocol: paired/interleaved200 admissions per arm after20warmups at10k/100k rows, static absent/nonmatching record; held200 and rebuilt-held20. Genuine clean ingest is outside timing. Seed count-1 plus its one unique replay append yields exact final rowcounts; cold preparation asserts idempotent rowcount. No failure generation is assigned manually. Seed/ingest/construction excluded. One run, no model judge or cross-run/cross-machine speedup claim. Earlier failed hosted run is preserved separately, not overwritten.

Re-score: python score_cost.py .; all eight cells PASS against snapshotted raw samples.

Not proven: construction/ingest cost, reference-session latency, host tool/prompt refresh, #1162 repair, merge/release/runtime/customer readiness.

T1: https://github.com/electricsheephq/lcm-x/blob/89d908dd34623ea2b67f79ffeb0c638a6230bc11/bench/scoreboard/results.jsonl. T2: not published; no results disposed.
