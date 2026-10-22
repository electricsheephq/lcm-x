# #1822 paired cost — COMPLETE

Run: `1822-r2-15819107-cost`. Candidate `158191079b4ec61e676f9d03ea3f34069543e8fb`.

| Rows | Path | Pattern | Base ms | Candidate ms | Pre-1628 ms | Candidate IQR ms | Limit ms | Gate |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 200 | rewind | True | 11.698021 | 7.564104 | 7.546416 | 0.134250 | 8.301058 | PASS |
| 200 | rewind | False | 4.674876 | 4.604708 | 4.607855 | 0.163906 | 4.674876 | PASS |
| 200 | steady | True | 4.219437 | 4.209146 | 4.193145 | 0.168219 | 4.641381 | PASS |
| 200 | steady | False | 1.119417 | 1.126626 | 1.107416 | 0.027365 | 1.231358 | PASS |
| 2000 | rewind | True | 109.691688 | 68.730020 | 68.776646 | 2.075531 | 75.654311 | PASS |
| 2000 | rewind | False | 41.247521 | 40.610042 | 40.920146 | 0.906593 | 41.247521 | PASS |
| 2000 | steady | True | 39.441437 | 39.348500 | 39.463959 | 0.922583 | 43.385581 | PASS |
| 2000 | steady | False | 9.668417 | 9.655771 | 9.676271 | 0.163031 | 10.635259 | PASS |

Each row receipt: cost-summary.json, re-scored from cost-200.jsonl. Median200 per arm/cell; 4800 total measured samples. Five excluded warmups; rotating/reversed interleaved arm order; same Mac/Python/host stub. Setup outside timer, in-memory SQLite; wall-clock ingest only; GC disabled only within timer. Embeddings, rollups, empty lifecycle GC and large externalization off. Exactly 200/2000 incoming rows, changed first row plus two fresh tail rows for rewind, no change for steady. Pattern uses existing literal _ContainsPattern matcher through ordinary configured pipeline, no benchmark row matches it. Full IQRs for every arm in summary.

Base: 238d1a9e876dea55b2e2f7a68110afd21c4e779e. Pre-1628: 8f121c723764be1ba49b182acd716642d94a4ecf.

Re-score: `python score.py`. Expected exit 0. Raw SHA256: 4f380e88cbd1263862d14a9e7ad4d8bfd9a6af7916e68f6f7d00db6dd953461f.

Not proven: production latency, final-head full-suite execution, merge/release/runtime/customer readiness.

T1: omitted because explicit SPEC limits owned source files. T2 tag: lcm-x/1822-r2-15819107-cost (publication verification recorded separately).
