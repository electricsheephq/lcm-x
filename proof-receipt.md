# Fix round 1 local proof

Claim class: advisory. Candidate [e06cc78879cc345c153a9e6afdc5eb3cde638c2e](https://github.com/electricsheephq/lcm-x/commit/e06cc78879cc345c153a9e6afdc5eb3cde638c2e); base 9eacedb9a667e517e40b9ed5c1725aa4f44df1cf; reviewed predecessor c993816bec8e78c0c46c0b1a5b67f03afd00893e.

## Correctness

- Non-date output identity: all 60 cells match base serialization, including all carried partial refs. Expected hashes unchanged. Both Bali controls pass.
- Date-dependent full-row matrix: 1,700 cells. All 850 carried cells match predecessor as JSON objects; carried exclusion key order is restored to base for non-date byte identity. Date-dependent byte identity is not claimed. Of 850 ordinary cells, 680 match; 170 finite-count cells remain unknown with the same reason and answers while exclusion metadata changes. Item 1(a) makes extracted partial candidate refs quote-scoped; full-source pre-loop still withholds mixed rows. Coverage-certificate digests reflect the changed exclusion metadata. Every changed cell is listed in `date-matrix.json`.
- Ordinary Bali partial refs: all five date/path variants change as authorized; carried variants match predecessor. Date filter becomes answer sufficient; finite count becomes computation sufficient; three order variants retain unknown with `missing_operand_2` instead of an unrelated-row-time reason.
- Abbreviated weekdays: 80 carried/ordinary/date/no-date cases match ISO labels and dates.
- Resolver byte identity against predecessor: 40,000 generated comparisons (10,000 quotes and four session-date inputs), 246,750 LongMemEval-S occurrences and 5,882 LoCoMo occurrences; zero mismatches. Corpus text and identifiers are excluded.
- Cached/uncached quote parsing: 3,000 equal. Retention after 64 distinct 256 KiB strings: ordinary 12598 bytes; carried 14071 bytes, each below 2 MiB. Inputs up to 1,200 characters use a 4,096-entry LRU; larger inputs retain only SHA-256 digest keys and results, at most 4,096 entries.
- Recall goldens and pinned-output fixtures unchanged. No resolver grammar/decision, recall, ingest, engine or store write change.

## Scaled compiler cost

Warm interleaved AB/BA, 200 samples per arm per count after 20 warmups. Same fixture shape and question in both worktrees; timing excludes setup and RPC. Inputs are dated baseline refs; normal compiler hydration cap remains 12, so the 24-input arm hydrates its supported maximum of 12. No cap override or provider call. Cold-cache rows clear the candidate's quote caches before each compile and are report-only. One shared-cache design, one measured run, no second optimization. CI/variance: N/A; local point estimates.

| Input refs | Warm base ms | Warm head ms | Delta ms | Cold base ms | Cold head ms | Warm gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0.100104 | 0.101938 | +0.001833 | 0.103166 | 0.137292 | PASS |
| 6 | 0.180958 | 0.186542 | +0.005584 | 0.169167 | 0.248271 | PASS |
| 10 | 0.223125 | 0.230542 | +0.007417 | 0.227187 | 0.350438 | PASS |
| 12 | 0.268062 | 0.279292 | +0.011229 | 0.249062 | 0.386292 | PASS |
| 24 | 0.306437 | 0.319666 | +0.013229 | 0.327125 | 0.497750 | PASS |

Bar at each count: max(10% of base, 0.05 ms). Cold latency is not claimed cost-neutral.

Raw samples: `cost-raw.json`; rescore: `python3 rescore.txt`.

## Tests

New file: `515 passed in 9.30s`. Focused set: `788 passed, 9 warnings in 14.85s`.
Matched set: `5 failed, 1310 passed, 3 warnings in 28.37s`; exactly the five pre-existing order-dependent coverage failures accepted under #1829.
Full CI-style result: `17777 passed, 141 skipped, 63 xfailed, 32 warnings in 307.67s (0:05:07)`. Separate embedding privacy: `77 passed in 2.30s`. Pinned Ruff 0.15.13: `All checks passed!`; both diff checks exit 0 with no output. Both public recall paths match predecessor; pack/ground smoke is grounded with zero store writes.

Production fix-round delta: 73 changed lines (+48/-25), below 120. Tests were committed first in 8d6c8c6e7180cc5c641615836043d7fe9cf5686e.

Not proven: answer-quality improvements on boards, production workload latency, independent semantic review, hosted exact-head CI, merge, release, installed runtime, or customer behavior. Full run outputs stay preserved; T1/T2 publication remains outside the source-fix scope.

Resolver-only original paired run: base p50 0.002791 ms, head 0.002750 ms; delta -0.000041 ms; 5,000 calls per arm, PASS.
