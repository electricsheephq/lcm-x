# #1586 selected-ID exact chunk loads

Measured product: `59b2289558521289a71a0c959ce7a7b9854d0639`; final source: `cdd4c2629ebca159de3f3093227206935d4ee06b` (only removes one SQL whitespace character to preserve the existing golden trace). BASE: `2ce2de7ff93a6d2d8b60321033fdce5c117f59c4`.

Claim: local synthetic source/performance proof. No schema, index, cache, service, default or history change. Two candidate-first CROSS JOIN queries use existing unique `(chunk_id, identity_hash)` indexes. Existing 500-ID staging chunks use one parameterized multi-row INSERT each; the temp-table schema and request isolation stay unchanged.

## RED / GREEN

| Check | BASE | Final candidate |
|---|---|---|
| Same 16 IDs; identity 64 → 8,192 rows, NumPy | 1,475 → 123,393 VM steps (RED) | 819 → 817 (GREEN) |
| Same check, Python decoder | 1,475 → 123,393 (RED) | 819 → 817 (GREEN) |
| Exact float32 multi-batch oracle | PASS before product edit | PASS; unchanged IDs, scores, ties and full/bounded coverage fields |
| Affected summary/chunk/int8 tests | baseline oracle calibrated before product edit | `102 passed in 25.62s` |

The oracle was calibrated to BASE’s newer-row-first tie key and existing bounded-total denominator before implementation. Initial oracle expectation failures are retained in red.txt, separately from the valid VM-step RED.

## Public recall scaling

RCA-derived real `lcm_recall` driver on the same synthetic covered store clone, float32/384, local BGE, 25k batch, no binary mirror, unchanged 8s deadline. Two synthetic queries (canary + broad NL), interleaved arms, four declared warmups and ten measured calls per arm/size. No measured samples omitted.

| Chunks | BASE p50 / p90 (s) | Candidate p50 / p90 (s) | Candidate full coverage |
|---:|---:|---:|---:|
| 62,500 | 2.646720 / 3.035249 | 0.338742 / 0.398635 | 10/10 |
| 125,000 | 8.002861 / 8.004191 (deadline-censored) | 1.155392 / 1.642400 | 10/10 |
| 250,000 | 8.001972 / 8.004638 (deadline-censored) | 1.534938 / 1.914184 | 10/10 |

BASE half/full rows are observed request latency at the unchanged deadline, not successful exact-recall latency. Local host contention occurred during this lane; preliminary and final raw samples are retained, and absolute timings are not pooled across windows.

## Fixed selected batch

Exactly the same 25,000 source IDs at every size; twenty paired timing samples per arm, plus separate progress-handler counting. Timing runs exclude the progress handler. Matrix bytes, rowids, IDs and kinds agree exactly between loaders. Indexed work stays constant; wall-clock/cache noise is shown rather than asserted away.

| Identity chunks | BASE load p50 / p90 (ms) | Candidate p50 / p90 (ms) | BASE VM steps | Candidate VM steps |
|---:|---:|---:|---:|---:|
| 62,500 | 132.231 / 137.390 | 59.897 / 62.662 | 1,512,645 | 1,025,602 |
| 125,000 | 178.340 / 185.302 | 69.667 / 77.365 | 2,450,145 | 1,025,602 |
| 250,000 | 244.869 / 251.292 | 78.515 / 82.412 | 4,325,145 | 1,025,602 |

## Small-store recall costs

Public `knn_chunks`, paired/interleaved median of 200 per arm; two warmup pairs. Cold clears existing caches outside timing; warm preserves the normal single-batch cache policy. Every returned ID/score/coverage field agrees exactly. Limit: candidate ≤ max(BASE × 1.10, BASE + 0.05 ms).

| Chunks | Cache | BASE median (ms) | Candidate median (ms) | Limit (ms) | Result |
|---:|---|---:|---:|---:|---|
| 1,000 | cold | 2.805063 | 2.314646 | 3.085569 | PASS |
| 1,000 | warm | 0.770042 | 0.771375 | 0.847046 | PASS |
| 10,000 | cold | 418.317354 | 32.853020 | 460.149089 | PASS |
| 10,000 | warm | 10.455312 | 10.543396 | 11.500844 | PASS |

## Exactness and checks

- BASE and candidate pass the independent multi-batch oracle with two identities, source/session/time exclusions, archived and malformed rows, an old best match and tied scores. Full and bounded coverage fields are exact.
- Both 250k query result sets (50 raw KNN rows each) match completed pristine-BASE captures exactly, including full coverage/scanned/total. Current BASE deadline misses are not substituted for complete reference results.
- Quarter-size BASE/candidate full scans and all fixed-batch byte comparisons agree. Half-size broad-NL BASE never completed in the interleaved cell; no direct complete-result comparison is claimed for that cell.
- `ruff check . --no-cache` with ruff 0.15.13: `All checks passed!`.
- `git diff --check`: exit 0, no output.
- Independent acceptance and distinct isolation/fallback reviews PASS at initial source and targeted staging delta; see REVIEWS.md. This is semantic coverage only.
- CI at measured59b22895 reported five golden trace failures, all the same extra space between VALUES and its placeholder;19887other tests passed in Python3.12. Finalcdd4c262 removes only that space; `95 passed in 7.66s` in affected assembly/phase0 checks. No golden was regenerated. Final-source full CI is a separate PR gate, reported in the PR Evidence block.

## Re-score and retention

Run `python3 analyze.py` against the retained raw JSON, then `python3 write_result.py`. Re-score reads no scratch DBs, live home or model cache. The unchanged synthetic input recipe comes from covered-store run 39768769622. BASE method snapshot is byte-verified against the pinned git source; CODE.json binds measured code and environment.

T1: the fixing PR appends run `lcmx-1586-chunk-load-20261023` to `bench/scoreboard/results.jsonl`. T2 tag: `lcm-x/lcmx-1586-chunk-load-20261023` (readback recorded in the local result manifest).

Not proven: #1586 floor bars on the 2-CPU runner, concurrent publication/compaction, release, deployment, runtime or customer readiness. This PR does not close #1586.
