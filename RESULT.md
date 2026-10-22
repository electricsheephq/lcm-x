# Lifecycle CAS #730 acceptance

Main: `de14a7f6ec16b671daafc43049171af741f23479`; candidate: `a75ef0d90b49354f88472f7734827e03f7753e9a`.

Paired measurements: 20 warmups and 200 measured pairs for each shape; alternate arm order; local SQLite WAL databases. Main and candidate use identical dependencies and filesystem. Threshold: paired median delta <= max(10% of main median, 0.05 ms).

| Shape | Main median ms | Candidate median ms | Paired median delta ms | MAD ms | Gate |
|---|---:|---:|---:|---:|---|
| new_bind | 0.101666 | 0.097354 | -0.001938 | 0.012542 | PASS |
| bound_noop | 0.005375 | 0.005042 | -0.000291 | 0.000209 | PASS |
| rollover_bind | 0.080771 | 0.073667 | -0.006271 | 0.011000 | PASS |
| resume_bind | 0.100250 | 0.090875 | -0.006687 | 0.011666 | PASS |
| finalize | 0.112750 | 0.105813 | -0.004125 | 0.011959 | PASS |

Reproduce: save `lifecycle_cost.py.txt` as `lifecycle_cost.py`, then run `python lifecycle_cost.py --main "$MAIN" --candidate "$CANDIDATE" --scratch "$SCRATCH/cost-dbs" --output cost.json` with fresh scratch and checkouts at the commits above.

Raw per-pair timings and source-file hashes: `cost.json`.

Not proven: cross-process races, real Hermes host, fleet behavior, release readiness, or performance on other machines.

Publication: SHA-pinned public evidence branch; private archive publication is not claimed.
