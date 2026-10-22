# #1659 catch-up r2 acceptance

Claim class: advisory; synthetic implementation proof.

Candidate: `6dd578221b0522b0e9665fa1a39e1cb187bdcf0d`; BASE: `238d1a9e876dea55b2e2f7a68110afd21c4e779e`.

| Gate | Result | Receipt |
| --- | --- | --- |
| Four defects RED at BASE | 4 intended failures | [log](red-base-final.log) |
| Final regressions, small differential, named and affected suites | 438 passed in 38.84s | [log](final-focused.log) |
| Final full differential | 600 passed in 40.52s; 0 false coverage | [log](final-differential.log) |
| Full local suite, once on original candidate | 62 failed, 19896 passed, 141 skipped, 63 xfailed | [log](full-suite.log) |
| Final cost gate | 1 passed in 38.18s | [log](final-costs.log) |
| Rollup latency flake | 5/5 candidate and 5/5 BASE passed | [checks](latency-checks.log) |
| Ruff | All checks passed! | [log](ruff.log) |
| diff-check | exit 0, no output | [log](diff-check.log) |

Full-run disposition: 50 frozen traces, 8 summary writer cases, 1 override scan-cost case, 2 slow-slice drain cases. One consolidated correction and the final 438-test pass resolve these. The remaining rollup latency failure qualifies under the SPEC flake rule after five passes on each arm. No second full local run and no full-green claim. Final-head remote CI and independent review remain separate.

The repo harness regenerated all 80 trace scenarios, changing 27 chunk traces; every summary trace is byte-identical to BASE. Assertions and named suites are unchanged.

| Paired/interleaved cost | BASE median ms | Candidate median ms | n per arm | Candidate limit ms |
| --- | ---: | ---: | ---: | ---: |
| status | 50.230355 | 50.513771 | 200 | 55.253389950000006 |
| recall | 3.615229 | 3.577354 | 200 | 3.9767519000000005 |
| inflight | 224.345396 | 26.494083 | 20 | N/A |

Raw samples: [costs.json](costs.json); [harness](test_costs.py). Five warmups per arm, order alternates each pair, identical synthetic non-loopback Ollama fixture. In-flight probe: 3,060 uncertain rows from 60 large sources; discovery held empty identically in both arms to isolate counting/publication tail. No confidence interval claimed; samples retained.

## Re-score

Run `python -c 'import json,statistics; c=json.load(open("costs.json")); print({k:{a:statistics.median(v) for a,v in c[k]["samples_ms"].items()} for k in ("status","recall","inflight")})'`. Seed outcomes are the assertion results in final-differential.log; the seeded generator is test_full_differential.py. To rerun timing, obtain command-base.py from the BASE commit and run test_costs.py with the candidate root and supported host stub on PYTHONPATH, loading tests.conftest.

Not proven: live provider, release, deployed runtime, fleet, customer sessions or merge readiness.

T1: not published; registry edits are outside this SPEC. T2: not published; no results disposed. Public receipts are a SHA-pinned evidence branch, not T1/T2 completion.
