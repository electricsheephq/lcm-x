# 20261022-occurrence-time-fix-r1b-public

Claim class: advisory. Source e06cc78879cc345c153a9e6afdc5eb3cde638c2e; baseline 9eacedb9a667e517e40b9ed5c1725aa4f44df1cf.

| Benchmark | Metric | Value | n | Receipt |
| --- | --- | ---: | ---: | --- |
| compiler-2-refs | warm_delta_ms | 0.0018330000000000013 | 200 | cost-summary.json |
| compiler-6-refs | warm_delta_ms | 0.005584000000000006 | 200 | cost-summary.json |
| compiler-10-refs | warm_delta_ms | 0.007417000000000007 | 200 | cost-summary.json |
| compiler-12-refs | warm_delta_ms | 0.011229000000000045 | 200 | cost-summary.json |
| compiler-24-refs | warm_delta_ms | 0.013228500000000032 | 200 | cost-summary.json |
| non-date-identity | matching_cells | 60.0 | 60 | matrix-summary.json |

Warm gates pass at every requested input count. Cold-cache latency is report-only; normal hydration cap is 12.

Resolver mismatches: zero across 292,632 comparisons. Large-input retention: 12,598 bytes ordinary and 14,071 bytes carried. Cache equality: 3,000 quotes.

Full suite: 17,777 passed, 141 skipped, 63 xfailed. Focused: 788 passed. Privacy: 77 passed. Matched: five accepted pre-existing #1829 failures, 1,310 passed.

Exact re-score command: `python3 rescore.txt`.

Not proven: board answer quality, production latency, independent review, hosted CI, merge, release, runtime or customer behavior.

T1/T2: not published within this source-fix lane; no cleanup of results authorized. Public proof: proof-receipt.md (public card) or RECEIPT.md (full local receipt).
