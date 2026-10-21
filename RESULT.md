# #1749 row-handoff-v1 — advisory measurements on this Mac

Baseline: e0db1b846fd813a9d280b34229b799bdf6ce46e0 (contains #1750).
Candidate product: c9c8869314e8209b712fc8b2576267e4362c402e; exact product hashes in product-file-hashes.json.
Measurements/full run preceded commit; product files are byte-identical to this commit.
Python 3.14.6; SQLite 3.53.1; inherited nice 19; switch interval 0.005 s.
Generated #1586 short corpus: 250,000 messages, 500 sessions, 250,000 chunk binary rows,
500 summary binary rows; deterministic 384-dimensional int8 layouts. No provider/model calls.
Five interleaved idle/busy pairs per site per phase, one busy in-process Python thread.
Eight-second effective lcm_recall budget supplied to each path; no cache used for vector loads.
Row-factory counters delegate to the product factory and count interrupted rows too.

| Site | Baseline idle median [min,max], s | Baseline busy median [min,max], s | Ratio | Decision |
|---|---:|---:|---:|---|
| sessions | .002132 [.001767,.003999] | .099589 [.030788,6.942297] | 46.71 | fix |
| vector.chunk | .314021 [.286967,.353658] | 8.162329 [8.092741,8.441186] | 25.99 (deadline censored) | fix |
| vector.summary | .001467 [.001309,.007885] | 3.009937 [.001374,3.549656] | 2052.23 | fix |

| Site | Candidate idle median [min,max], s | Candidate busy median [min,max], s | Ratio | Python rows baseline → candidate |
|---|---:|---:|---:|---:|
| sessions | .001844 [.001684,.002777] | .182218 [.150958,.193630] | 98.80 | 1000 → 2 |
| vector.chunk | .404621 [.376926,.412938] | .573191 [.541204,.594674] | 1.42 | 250000 → 1 |
| vector.summary | .001967 [.001658,.006443] | .162150 [.145501,.177707] | 82.42 | 500 → 1 |

Baseline chunk timed out in all five busy runs (1174–3998 rows already handed to Python),
and exceeded its 8 s call-path budget in every trial. Sessions and summary did not exceed
8 s in these trials. Candidate: zero deadline overruns/timeouts at every site.
Session busy median REGRESSES from .099589 to .182218 s; its measured maximum drops
from 6.942297 to .193630 s. No session median-speedup or all-sites-small-factor claim.
Small workloads retain connection/watchdog/GIL overhead; vectors trade some idle cost
and aggregate memory for reduced contention. These are advisory, noisy five-pair samples.

| Proof | Receipt |
|---|---|
| Phase 1 decisions, before/after timings/counters | SUMMARY.json; SAMPLES.json; base/product uptime snapshots |
| Session set and complete ordered matrix-byte identity on generated store | rescore.py digest equality across every successful sample |
| Base-red counters | CHECKS.json: 3 failed, 15 deselected in 1.01s |
| Focused recall/vector and #1750 regressions | CHECKS.json: 387 passed, 1 deselected in 12.79s |
| Single CI-equivalent full run | CHECKS.json: 1 failed, 15951 passed, 137 skipped, 64 xfailed, 23 warnings in 253.67s (0:04:13) |
| Corrected 30-query identity test | CHECKS.json: 1 passed in 4.14s |
| CI's isolated privacy file | CHECKS.json: 77 passed in 2.35s |
| Ruff 0.15.13 | CHECKS.json: All checks passed! |

Exact re-score command: `python rescore.py` (reads immutable SAMPLES.json only).
T1: pending publication of the bounded public receipt and registry row.
T2: incomplete; no bench-evidence release was authorized or published.
Not proven: floor, smallest host, release, independent review, exact-head remote CI,
merge readiness, runtime safety or customer readiness; no general session median speedup.

The full run's sole failure was the newly tightened identity fixture's empty filtered result,
not a product/base divergence. Message source is an append keyword, supplied timestamp is
observed_at (messages.timestamp is ingestion time), and some fixed-session filter intersections
are correctly empty. Corrected the fixture and expectation; the final targeted identity test
passes 120 recall pairs and 120 filtered semantic-grep pairs over the same 30 queries.
Recall assertions require semantic full_approx coverage and compare hit order, full hit payload
hashes and provenance. Grep compares exact result payloads, including valid empty intersections.
The earlier 17 remaining new tests passed; the original focused suite passed 387 tests.
Full suite run exactly once as instructed; final test-only delta was not full-suite rerun.
No known local-only product failures occurred or were chased. Product delta: 128 changed lines.
