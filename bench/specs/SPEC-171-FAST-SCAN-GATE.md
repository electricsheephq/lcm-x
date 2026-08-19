# SPEC — #171 fast-scan: design + parity gate **v2** (re-registered after F39 KILL; v1 preserved in git history). FROZEN on commit.

## v2 changes (F39 adjudication — registered BEFORE the re-run)
- **R2′ replaces R2:** residency is dtype-AGNOSTIC — the resident int8 matrix is quantized FROM float32
  vectors at build time (same RAM budget). Scoring on resident paths therefore differs from exact
  float32 — recall parity is exactly what this gate tests, not an assumption.
- **Size-aware path selection:** below a measured N threshold the original simple loader serves (kills
  the small-rung regression by construction). The threshold is chosen by measurement, stated in the PR.
- **PASS-5 gains failure semantics (v1 authoring gap):** small-rung p50 ≤1.25× F34 = PASS; (1.25,1.6]× =
  GRAY; >1.6× = KILL.
- All other clauses unchanged from v1.

## Design decision (architect, 2026-07-29; full design legwork archived at
## session-notes/2026-07-29/hermes-r3-1/artifacts/DESIGN-171-FAST-SCAN.md)

**#171 was misdiagnosed as an ANN problem.** Decomposition (reproduced 6.63s on same-shape 185k×384
synthetic): per-row `struct.unpack` decode in `_load_vectors_for_ids` = 2.58s (duplicated for chunks);
93 temp-table batch JOINs in `_scan_ranked` = 5.56s vs 3.2s streaming. Adopted:
- **R1 (unconditional):** vectorized loader — `numpy.frombuffer` + one streaming cursor, no temp table —
  on BOTH `knn` and `knn_chunks` via a shared helper (the two loaders are copy-paste twins). Measured
  ~400ms warm, bit-identical results, `coverage='full'` unchanged, no migration, no deps.
- **R2 (primary for #171):** persistent int8 matrix residency behind a RAM budget
  (`LCM_KNN_RESIDENT_MAX_MB`, default 128 ≈ 330k vectors; over-budget → fall back to R1). Measured 39ms
  warm / 624ms one-time load. `coverage='full'` stays truthful (every live vector scored). Keyed like
  `_binary_matrix_cache` (data_version). **Accepted cost (decided):** invalidation on write = full
  reload; ingest bursts drop residency and fall back to R1 (~400ms) — append-delta is a follow-up, not
  this train.
- **Rejected:** IVF (breaks `full_approx` semantics as written; unpredictable recall), HNSW (build
  infeasible pure-python; wheel unpackageable — install.sh is symlink-only by design), sqlite-vec
  (HARD BLOCKER: system python 3.9.6 has enable_load_extension compiled out).
- **Deferred to the >1M-vector tier:** sign-bit prescreen — and its M must be a FRACTION of N (≥5%),
  never `mult*k` (see the landmine issue filed from this analysis).
- Amendment-7 coupling DISCHARGED with reasoning: the reframed #171 touches message/chunk vectors; #179
  touches trajectory-state embeddings — disjoint tables, and the F31/F34 instrument measures the former
  only. #171's gate run no longer waits on #179.

## Implementation traps (bind the implementing lane)
1. `_release_matrix_caches()` clears every cache at the start of any multi-batch scan — it will destroy
   residency on exactly the path residency serves. Rework it; do not leave it.
2. Residency load must stay interruptible through the existing deadline seams
   (`_prescreen_deadline_expired`, `_monotonic`) — the hardened budget contract must not regress.
3. No `identity_hash` change of any kind (frozen stores must stay comparable).

## Pre-registered parity gate (adopted from the design legwork verbatim)
Instrument: F31/F34 389× family, all four rungs, frozen probe set, content-based joins, bench/tools
(storefreeze + pinverify + failclose), fastembed stamp, SAME store copies.
Recall: paired per-question all-gold delivery, arm A3 new-build vs F34-build, **u-arm to u-arm**
(deadline disabled; F34's archived 0.233 is deadline-censored ≈5.3% at top rung — not a deterministic
reference).
- **PASS:** paired net (b−c) ≥ −3 every rung; A3 all-gold ≥ arm B every rung; p50 @19,829 ≤ **350ms**;
  0% queries hit the 8s deadline at any rung; p50 at 500/2k ≤ 1.25× F34's 18–45ms; returned coverage
  string exactly matches what was scored.
- **GRAY (owner + power memo):** net in [−8,−4] any rung, or p50 @19,829 in (350,388]ms.
- **KILL:** net ≤ −9 any rung; all-gold < arm B any rung; p50 @19,829 > 388ms; any coverage overstate.
- Latency vs a SAME-SESSION re-measured arm B; warm p50 AND first-query cold cost both reported.
- Power memo first: discordant pool per rung from F34's per-question output; < 20 at the top two rungs →
  widen the probe set, do not run (the F38 lesson).
