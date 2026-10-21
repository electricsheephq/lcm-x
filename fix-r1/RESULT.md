# Recall row handoff: fix round 1

Claim class: advisory product/performance evidence on one Mac. **PARTIAL: the RSS target fails.**

Base: `e0db1b846fd813a9d280b34229b799bdf6ce46e0`. Candidate: starting `8d18e7943152d132f0af292911c88098b66685aa` plus `fix.patch`; exact file hashes and dirty-tree binding are in `CODE.json`.

## Pass table

| Gate | Result | Receipt |
| --- | --- | --- |
| No temp B-tree; row identity; resource fallback | PASS | `EQP.json`, `focused.log` |
| Idle speed ≤1.1× base | PASS | `LIMITS.json` |
| Full matrix at 1.6 s, 4/4 | PASS, both base and candidate | `LIMITS.json` |
| Peak RSS delta ≤1.25× base | FAIL at both widths | `LIMITS.json`, `memory-stages.log` |

## Idle speed

| Shape | Base median s | Candidate median s | Ratio |
| --- | ---: | ---: | ---: |
| 250k-48 | 0.235500 | 0.212000 | 0.900× |
| 1m-128 | 1.082500 | 0.968500 | 0.895× |

Four interleaved base/candidate pairs per shape, fresh subprocesses, private connections. One first base 1M sample took 8.908 s; it remains in the samples and median. Successful ordered-ID and matrix-byte hashes match.

## Memory

| Shape | Base peak delta MB | Candidate peak delta MB | Ratio |
| --- | ---: | ---: | ---: |
| 1m-128 | 583 | 1127 | 1.933× |
| 1m-384 | 1107 | 2947 | 2.662× |

Fresh subprocess per load, private connection, macOS `ru_maxrss` delta in decimal MB. At 1M × 128 B the direct aggregate reaches 858 MB before decoding; lean decoding peaks at 1156 MB in the standalone stage probe. Releasing hex and avoiding per-row copies does not meet the required bound. The prescribed whole-stream hex aggregate remains the limiting mechanism. No alternate transport was introduced.

## Remeasured original sites

| Site | Base idle median [min,max] s | Candidate idle median [min,max] s | Base busy median [min,max] s | Candidate busy median [min,max] s |
| --- | ---: | ---: | ---: | ---: |
| sessions | 0.002511 [0.002404,0.002633] | 0.002110 [0.001992,0.002255] | 3.297513 [3.170188,4.272009] | 0.112816 [0.106986,0.142380] |
| vector.chunk | 0.259293 [0.257244,0.290627] | 0.198665 [0.197173,0.222263] | 8.160382 [8.026514,8.281800] | 0.335681 [0.316753,0.342580] |
| vector.summary | 0.001621 [0.001590,0.003829] | 0.001793 [0.001762,0.001881] | 1.656694 [1.390754,1.773543] | 0.117894 [0.088226,0.142537] |

Session busy median improves in this run; the prior run's regression is not reproduced. Timing noise remains large; these samples do not establish a general median guarantee.

Generated 250k-message store, 500 sessions, 250k chunk rows and 500 summary rows, 384-dimensional int8. Five idle/busy pairs per site, interleaved base/candidate, exactly one busy Python thread, 8 s call budget. Inherited nice 20, Python 3.14.6, SQLite 3.53.1, 5 ms switch interval. All successful session and ordered-vector digests match. Base busy chunk times out 5/5; candidate 0/5 at every site.

## Validation

`CHECKS.json` records exact commands/results. Full suite: `15963 passed, 140 skipped, 64 xfailed, 23 warnings in 280.18s (0:04:40)`. Privacy: `77 passed in 2.24s`. Pinned Ruff 0.15.13: `All checks passed!`. `git diff --check`: exit 0. No local-only failures occurred or were chased.

## Commands and rescore

```sh
python run_limits.py
python run_measure.py
python rescore.py
```

The execution scripts use lane-local inputs; the portable `rescore.py` reads only `SAMPLES.json` and `LIMITS.json`. `SUMMARY.json` is its exact output.

Not proven: the required RSS bound, an end-to-end OOM bound, general timing guarantees, independent review, exact-head remote CI, merge, release, runtime or customer readiness.

T1: the three updated row-handoff entries in `bench/scoreboard/results.jsonl` in this PR.
T2: not published; explicitly outside this round.

Recommended next steps: the release manager should revise the transport constraint to permit a bounded-memory representation or bounded aggregates, then re-run only the RSS and affected identity/speed gates. Do not describe the memory requirement as passed.
