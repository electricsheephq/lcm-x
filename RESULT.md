# Coverage marker round 3 acceptance

Refs #1624 #1853

Cloud maintenance validates bounded, contiguous chunk metadata coverage without chunking or provider dispatch. Discovery checks cancellation and budget before chunk validation. Independent parent-fenced low-band progress alternates after an actual upper-band deferral; a bounded postpublication metadata check lets coverage catch new arrivals.

Specified base: [91a03989](https://github.com/electricsheephq/lcm-x/commit/91a039891afd969f9569897115990fe0c419e034). Final production candidate: [c0b1f17e](https://github.com/electricsheephq/lcm-x/commit/c0b1f17e86868c5312e9752ac1d0594412856997). Previous GA: [709099b3](https://github.com/electricsheephq/lcm-x/commit/709099b3a397a7893a7abae8802ebe7bf605f79e) (`v0.28.0`). Pre-marker cost comparator: [20891a98](https://github.com/electricsheephq/lcm-x/commit/20891a980cae075cad177da7c19c549bac05e63c). Streaming and certified-steady samples were measured at [cef344e5](https://github.com/electricsheephq/lcm-x/commit/cef344e54167848829c22752c6a431191a763985); the later guard affects metadata-only revision promotion, a branch those workloads do not execute. Oracle/bounds and affected metadata-validation measurements name their final source in the identity map.

## RED / GREEN

RED regressions were committed before the fix; the exact-base run reported `4 failed in 0.88s`.

| Cell | RED | GREEN |
| --- | --- | --- |
| Cloud coverage and planted metadata gap | Two failures | Both pass; no chunking/provider resolution |
| Streaming first yield / cancellation | 69 chunk calls exceeds bound 8 | Cancellation uses 5 calls; same 32 selected |
| Arrival pair before every retry=False call | Low marker still 6 after 12 calls | Holes filled call 1; marker converged call 3 |
| Invalidated initial low CAS | Stale-parent retry could loop | One rejected attempt, conservative return |
| First disjoint-band priority | Initial candidate chose low too early | Actual-deferral flag preserves base selection |

Earlier focused/preservation checks reported `7 passed in 1.22s` and `254 passed in 41.03s`. Final affected checks report `94 passed in 8.33s`; final golden checks report `90 passed in 4.56s`. These final checks exercised the bytes committed as c0b1f17e while invocation HEAD was 7852ad7d, before the source and golden were committed together.

Testing improved: the cloud-profile coverage cell, the streaming first-yield/cancel timing cell, and the arrival-every-call liveness cell join the coverage-marker suite.

## Oracle, bounds and status

| Gate | Result |
| --- | --- | --- |
| 600 seeds without legacy step | 0 new step mismatches; 0 false certificates |
| 600 seeds with legacy step | 0 new step mismatches; false certificates confined to accepted seeds 150, 212, 422, 491 |
| Four long histories | At most 3 gap scans/call; 0 width violations; 0 missing rows |
| Certified steady state | 60 pass/status pairs per history; 0 scans |
| Read-only status authorizer | 0 attempted writes or data changes |

The exact r2 bounds instrument waits for full-band certification before measuring steady state. Earlier reviewer output includes initial proof scans before certification. Differential oracle parity preserves inherited baseline limitations; it does not claim those limitations were repaired.

## Cloud status

20k rows, revision 2 with no coverage keys, one maintenance pass; paired interleaved samples, n=60 per arm. All timings are milliseconds.

| Arm | Before p50 | After p50 | After IQR |
| --- | ---: | ---: | ---: |
| ga | 0.558146 | 0.505084 | 0.107719 |
| base | 12.705625 | 12.660979 | 1.408771 |
| head | 12.578187 | 0.585375 | 0.087833 |

PASS: head after one pass is within 2× measured GA; marker covers 1..20000. A planted gap remains outside coverage and visible in backlog.

## Streaming

2,000 enclosed 4,096-byte gap rows, batch limit 32; paired interleaved n=40 per arm. Values are p50 / IQR in milliseconds.

| Metric | GA | Specified base | Candidate |
| --- | ---: | ---: | ---: |
| first | 2.243313 / 0.097772 | 2.294625 / 0.151303 | 2.252730 / 0.122687 |
| cancel | 14.461042 / 0.482396 | 9.129084 / 0.325501 | 2.911208 / 0.226864 |

PASS: both candidate p50 values are within 2× GA. All arms select the identical 32 ids, `2001:0` through `1970:0`. Cancellation dispatches nothing.

## Steady pass and status costs

20k rows, paired interleaved n=200 per arm; threshold max(base × 1.10, base + 0.05 ms). Fresh arms replay a legacy integer marker and are distinct from the cloud no-key shape.

| Comparator / cell | Base p50 ms | Candidate p50 ms | Bound ms | Result |
| --- | ---: | ---: | ---: | --- |
| specified-base / pass | 2.5475 | 2.5433 | 2.8023 | PASS |
| specified-base / status | 0.2980 | 0.2949 | 0.3480 | PASS |
| specified-base / fresh-pass | 2.6436 | 2.6324 | 2.9080 | PASS |
| specified-base / fresh-status | 0.2583 | 0.2659 | 0.3083 | PASS |
| pre-marker / pass | 2.6372 | 2.7535 | 2.9010 | PASS |
| pre-marker / status | 0.2062 | 0.2128 | 0.2562 | PASS |
| pre-marker / fresh-pass | 2.0463 | 2.1089 | 2.2510 | PASS |
| pre-marker / fresh-status | 0.2329 | 0.2423 | 0.2829 | PASS |

## Arrival-every-call liveness

The natural and upper-pending probes fill holes on call 1 and certify on call 3; the 600-row saved-walk shape certifies on call 20. Each receives a new arrival pair before every call. Zero missing rows, width violations, read-only violations or false-zero backlog reports.

20k one-arrival-pair-per-pass cost probe, n=30 pairs per tree; arms run sequentially. Threshold max(pre-marker × 1.5, pre-marker + 1 ms).

| Arm | Pass p50 ms | Status p50 ms |
| --- | ---: | ---: |
| prebase | 4.819750 | 0.461729 |
| base | 11.969146 | 9.980438 |
| head | 3.114937 | 0.295896 |

PASS against the pre-marker comparator for both costs. The defective specified base is shown separately.

## Explained golden changes

77 of 80 traces are unchanged. Messages, chunk metadata, vectors, provider inputs and dispatch sequences are preserved in all three changed traces:

- `chunks/stop-before-dispatch`: cancellation occurs before selection, so selected/pending/estimates become zero. Dispatch remains zero and remaining backlog remains 3. The existing classifier changes partial to complete; this direct earlier-cancellation tradeoff is explicitly accepted. The unused empty partial-message record disappears.
- `chunks/incremental-boundary-resume`: approved epoch/turn and parent-fence bookkeeping; reports and dispatch unchanged.
- `chunks/incremental-revision-gap-checkpoint`: metadata coverage/progress is recorded independently from repair revision. The metadata-only walk leaves revision 1 intact; the subsequent normal repair walk promotes revision 2. Reports and dispatch are unchanged.

One independent initial semantic review and three bounded targeted delta passes covered the scheduling/CAS correction, golden-only update, and final revision guard. Both verified blockers are fixed; no unresolved finding remains in the inspected scope. No general review was restarted.

## Evidence

Claim class: advisory; exact-source local implementation and synthetic acceptance evidence.

The prescribed virtual environment and Hermes host fixture were used. Machine-specific environment roots are omitted; `$SCRATCH` denotes the isolated temporary-data directory. Exact expanded commands and complete raw oracle inputs are retained under evidence tag `lcm-x/lcmx-1624-r3-20261022`.

```bash
python -m pytest tests/test_issue_1624_cloud_stream_liveness.py -q -p no:cacheprovider --basetemp="$SCRATCH/pt-delta1" -o tmp_path_retention_policy=failed
# 7 passed in 1.22s
python -m pytest tests/behaviour/test_backfill_trace.py -q -p no:cacheprovider --basetemp="$SCRATCH/pt-golden" -o tmp_path_retention_policy=failed
# 90 passed in 2.25s
python rescore.py
# PASS: cloud, stream, same 32 selected chunks, and all steady-cost timing cells
```

Full-suite receipt (run once at 7852ad7d):

```bash
bash -lc 'ulimit -n 1024 && python -m pytest tests/ -q -p no:cacheprovider --basetemp="$SCRATCH/pt-full" -o tmp_path_retention_policy=failed'
1 failed, 17460 passed, 141 skipped, 64 xfailed, 29 warnings in 1319.85s (0:21:59)
```

The sole failure was `tests/test_issue_1197_short_turn_chunks.py::test_revision_gapfill_and_partial_boundaries_both_finish`; that node passes at the exact specified base. The introduced premature revision promotion was fixed in c0b1f17e, with the original assertion preserved and the affected 94 tests plus 90 goldens passing on its committed content. No second full-suite run was performed; a green full suite at the final head is not claimed.

```bash
UV_OFFLINE=1 uv tool run --from ruff==0.15.13 ruff check command.py embedding_maintenance.py tests/test_issue_1586_gap_discovery.py tests/test_issue_1624_cloud_stream_liveness.py
All checks passed!
git diff --check
# exit 0 (no output)
git diff 91a039891afd969f9569897115990fe0c419e034 HEAD --check
# exit 0 (no output)
```

Public timing re-score: `python rescore.py`. The complete oracle/liveness re-score uses the raw bundle's `worker/repro/delta2/rescore.py worker`.

Not proven: a green full suite at the final head, remote supported-Python CI matrix, merge, release, installed runtime or customer behavior. The four accepted legacy false-certificate seeds remain inherited limitations.
