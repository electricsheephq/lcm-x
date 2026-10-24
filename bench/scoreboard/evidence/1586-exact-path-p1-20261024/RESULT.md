# Exact-path P1: mandatory exactness stop

Claim class: **advisory**. Draft remains **not ready**.

Base `849da257b52fc9833955928816dd1585184988b1`; candidate
`76abda9fcce73706c2656a038098532be4a8e0e7`. Product code was frozen throughout.
The permanent instrument at that candidate uses the same synthetic store/query
classes/seed as r1, nice 19, BLAS/VECLIB/OMP 1 and product ORT 2.
No competing Python thread was added during this preflight.

| Gate | Observed result | Receipt |
|---|---|---|
| Exactness preflight | **STOP**: 4 top-k differences in 36 observed pairs; 32 pairs match within 4 float32 ULP | [SUMMARY.json](SUMMARY.json), [per-query projection](partial-recalls.jsonl), [first difference](exactness-stop.json) |
| First difference | index 3/class a: base full chunks in 1.584662 s; candidate timeout/chunk none in 8.003818 s | [first difference](exactness-stop.json) |
| RSS | Base first block peak 880984064 bytes; candidate interrupted before peak summary; **incomplete** | Original E-A1 summary retained with full local run |
| C1 / C2 / C3 | **Not run** after exactness hard stop | [SUMMARY.json](SUMMARY.json) |
| Store pre/post audit | Complete 250000 chunks/500 leaves; integrity OK; raw rows, metadata and vectors unchanged | [pre](pre-audit-p1.json), [post](post-audit-p1-stop.json) |

All 200 base calls and all 36 completed candidate calls are retained, with no
filtering. The first top-k difference was detected from the live completed prefix;
the lane then interrupted only its own measured child. Additional calls completed
before the interrupt drained, leaving the 36-pair prefix. Candidate peak RSS was
not emitted, so no memory PASS is claimed. The four mismatches each coincide with
a candidate timeout and absent chunk coverage; this does not establish a ranking
error between two completed full scans.

| Block | Load1 before | Load1 after | Receipt |
|---|---:|---:|---|
| E-A1 base | 15.3501 | 19.2124 | [load.json](load.json) |
| E-B1 candidate, interrupted | 19.2124 | 22.4761 | [load.json](load.json) |

P1 permits current-load measurement and treats **C2** failure at load1 >=4 as
PENDING-QUIET. It retains the independent rule that **any exactness difference
stops the lane**. This is EXACTNESS_STOP, not PENDING-QUIET. No product correction,
extra latency cell, or acceptance waiver followed the stop. No C1/C2 improvement,
complete ABAB comparison, failure rate, or confidence interval is inferred from
this partial preflight.

## Re-score

From this receipt directory:

```sh
python3 -B rescore-partial.py
```

Observed output includes `"status": "EXACTNESS_STOP"`, `"paired_queries": 36`,
and four difference entries. The projection retains every completed call's query,
ID/score, timing, timeout and coverage. Full payloads, effective configs, argv,
stdout/stderr, interrupt trace, runner and audit records remain in the original
run, protected by its result card. The public card covers this reproducible
projection; it is not a T2 archival receipt.

Not proven: completed exactness, memory bar, C1/C2/C3 acceptance, 2-vCPU floor,
saturation compaction, fresh GA readiness, merge, release, runtime or customer behavior.

T1: a diagnostic results.d row is added in the same PR, pinned to this evidence.
T2: incomplete; publication requires the release manager because this lane's SPEC
forbids tags and releases. Intended tag: `lcm-x/20261024-1586-exact-path-p1`.
All original results and re-score inputs remain preserved. No cleanup occurred.

## Recommended next steps

The release manager should authorize a quiet-window exactness/RSS rerun on the
unchanged product, then C1/C2/C3 if preflight passes. This avoids changing exactness
or deadline semantics to accommodate censored calls. Resume only after that explicit
stop disposition, exact source/store revalidation, a fresh output path, load1 <4
for the recommended preflight, and disk >=40 GiB. A C2-only quiet rerun cannot
complete this gate because C1/exactness/memory have not passed.
