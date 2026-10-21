# lcmx-1765-whitespace-20261022

Claim class: advisory local product catch-up proof. Refs #1765.

Code: [candidate 61148a0209136f65e135508950e0d9e2e851332c](https://github.com/electricsheephq/lcm-x/tree/61148a0209136f65e135508950e0d9e2e851332c). Baseline: main `5dd67c0dab62f409beec5f86951181f3d2affb1c`.

| Proof | Main | Fix | Receipt |
| --- | --- | --- | --- |
| 20k rows; newline runs 1999 / 2000 / 2001; published arrival every pass | 1999 drains in 11; 2000 and 2001 still have 12999 unfilled after 12 passes | all drain in 11, bound 12; exact eligible coverage and no duplicate publication | validation.json |
| Arrival interval of 2001 newline rows; SQL pages 32 and 2000 | 4001 fetched rows per pass | exactly 2000; empty walk saved at 4002; next arrival resumes to old history | validation.json |
| Rollback readers: pre-walk, v0.28.1, v0.28.0 | all 6 broken-trigger probes miss the body | all 6 detect abort/no-op bodies; normal updates succeed and clear both destination proofs | validation.json |
| Frozen base revision and eligibility drift | both drift probes fail to fail closed | both detect drift; fixed corpus covers all 3 policies | validation.json |
| Ordinary output identity and non-increased VM work | reference | 24/24 pass | ordinary-final.json |
| Focused regression + compatibility suite | 12 expected RED, 5 pass | 19 pass | validation.txt |
| Full CI-style main / isolated privacy | N/A | 16579 / 77 pass; 141 skipped, 64 xfailed in main | validation.txt |

## Ordinary-store identity and SQLite VM work

Generated #1758 shapes: 250000 user rows with 0/50000/240000 published; 100000 rows with 60000 published, steady probe then 20 published arrivals, publishing each arrival pass. Limit 2000; SQL page 32. Both arms receive the same rows/state via savepoint rollback, then actual candidate discovery/publication advances the live arm. Every chunk id and UTF-8 byte sequence is compared directly; per-cell digests are in ordinary-final.json. SQLite progress handler counts every VM step, without SQL predicate instrumentation. One deterministic observation per arm/cell; variance, confidence interval and model judge N/A. Counts are not a latency or throughput claim.

| Shape / pass | Filled before measurement | Main VM steps | Fix VM steps | Chunk count | IDs and UTF-8 bytes |
| --- | ---: | ---: | ---: | ---: | --- |
| prefix250k / 0 | 0 | 59056 | 59024 | 2000 | identical |
| prefix250k / 1 | 50000 | 101793 | 101793 | 2000 | identical |
| prefix250k / 2 | 240000 | 101793 | 101793 | 2000 | identical |
| live100k / 0 | 60000 | 101793 | 101793 | 2000 | identical |
| live100k / 1 | 60000 | 101873 | 101854 | 2000 | identical |
| live100k / 2 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 3 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 4 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 5 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 6 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 7 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 8 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 9 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 10 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 11 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 12 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 13 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 14 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 15 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 16 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 17 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 18 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 19 | 60000 | 101854 | 101854 | 2000 | identical |
| live100k / 20 | 60000 | 101854 | 101854 | 2000 | identical |

## Re-score

```sh
python3 rescore-final.txt
```

Expected: `24/24 ordinary cells preserve chunk ids/bytes and do not increase VM steps; all whitespace cases drain within 12 passes.`

`ordinary.json` and `rescore.txt` preserve the earlier endpoint-seeded measurement. They are excluded from the final headline; `ordinary-final.json` is authoritative for the ordinary-store table.

## Validation boundary

The full suite covered final production bytes before the final measurement-fixture-only correction. The two corrected ordinary-shape cases passed afterward. Remote CI and independent review remain separate gates. The PR is to remain open and unmerged; items 3-5 stay on #1765.

Not proven: #1586 floor performance, provider throughput, remote CI, merge/release readiness, runtime or customer behavior.

T1: no registry mutation authorized by this command.py/tests-only spec. T2: not published; local raw results retained. Public-safe proof is attached to the PR through an immutable evidence branch.
