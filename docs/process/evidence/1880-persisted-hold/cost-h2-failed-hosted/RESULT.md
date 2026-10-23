# #1880 H2 hosted-Linux cost result: FAIL

Measured source:849697b9e5eb24b921ea35799f94da55af3f2dcc; product unchanged from81229ac490067e2514a29601c0a09f771ae42de0. BASE10407e12e33c1d7b09fe0e05141153555aec5d62.
Runner: GitHub-hosted Ubuntu, Python3.13; run39910510425/job120272169764.

| Rows / mode | BASE p50 ms | Candidate p50 ms | Candidate p95 ms | n | Receipt |
| --- | ---: | ---: | ---: | ---: | --- |
| 10000 / absent | 0.0098585 | 0.024071 | 0.025161 | 200 | PASS; summary.json and samples.jsonl |
| 10000 / mismatch | 0.013053 | 0.09568399999999999 | 0.1042 | 200 | FAIL cost bar; summary.json and samples.jsonl |
| 10000 / held | N/A | 0.1028105 | 0.115645 | 200 | PASS; summary.json and samples.jsonl |
| 10000 / rebuilt-held | N/A | 0.2312265 | 0.256388 | 20 | PASS; summary.json and samples.jsonl |
| 100000 / absent | 0.0100245 | 0.024383500000000002 | 0.025585 | 200 | PASS; summary.json and samples.jsonl |
| 100000 / mismatch | 0.013382 | 0.0964425 | 0.107666 | 200 | FAIL cost bar; summary.json and samples.jsonl |
| 100000 / held | N/A | 0.1026995 | 0.114147 | 200 | PASS; summary.json and samples.jsonl |
| 100000 / rebuilt-held | N/A | 0.25764600000000004 | 0.294232 | 20 | PASS; summary.json and samples.jsonl |

Two of four UNHELD cells pass; both nonmatching-record cells exceed max(BASE*1.10,BASE+0.05ms). This is a failed acceptance measurement, not H2 cost acceptance. Held paths make zero model calls/full-history scans/metadata writes; indexed query plans are preserved.

Protocol: paired/interleaved200 admissions per arm after20warmups at10k/100k rows, same static absent/nonmatching record; held200 and rebuilt-held20. Actual clean ingest is outside timing. Seed count-1 plus its one unique replay append yields exact final rowcounts; cold preparation asserts idempotent rowcount. No failure generation is assigned manually. Seed/ingest/construction excluded. One run, no model judge or cross-machine equivalence claim.

Re-score: python score_cost.py .; EXPECTED_NEGATIVE at the10k mismatch cost-bar assertion. Raw samples and all computed cell summaries remain preserved.

Not proven: passing H2 cost bar, original-Mac H2 cost, live/reference latency, retained-occurrence repair, merge/release/runtime/customer readiness.

T1: https://github.com/electricsheephq/lcm-x/blob/89d908dd34623ea2b67f79ffeb0c638a6230bc11/bench/scoreboard/results.jsonl. T2: not published; no results disposed.
