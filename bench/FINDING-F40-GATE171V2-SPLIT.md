# F40 — #171 gate v2: KILL on PASS-5 only; the scale clause passes with margin (241ms @19,829, 26× vs F34). Root cause of the small-rung residual: residency dies with per-query instance lifetime.

**Date:** 2026-07-29 · **Run:** gate171-v2 on 590c1f4 (R2′), instrument @29fcdd2, 20 min, zero LLM,
pins PASS pre+post, stores sha-matched to F39's pinned values, 0 fail-closes. Artifacts:
session-notes/2026-07-29/hermes-r3-1/artifacts/gate171-v2-run/.

## Clause reading (v2 spec, unchanged)
MET: PASS-1 recall (0/0/0/0 vs F34 archive 20-pair; 0/+1/0/0 vs the v1-build n=50 — quantize-on-load
changed ONE delivery in 200 paired questions, favorably) · PASS-2 (A3≥B every rung) · **PASS-3 p50
@19,829 = 240.96ms ≤ 350** (F34: 6,311; v1 build: 2,847) · PASS-4 (0 deadline hits) · PASS-6 (coverage
truthful). **KILL: PASS-5** — 500/2k at 2.91×/3.73× vs F34 (1.97×/2.74× machine-normalized; >1.6× on
every basis). Same-session B control ran 1.03–1.48× vs F34 (ambient predates comparison; normalized
numbers stated).

## Root cause (direct evidence, EVICTION-EVIDENCE.json — not the trap the spec anticipated)
A VectorStore is constructed per query at every rung, and at 500/2k the residency-holding instance is
FINALIZED between queries (`_read_with_deadline:2535 → __del__:3627 → close:3622`; clear_nonempty 9/10
queries vs 1/10 at 8k/19829) — the resident matrix rebuilds on 100/100 queries (~42ms @500, ~126ms @2k
per rebuild = the entire residual). `_release_matrix_caches()`: 0 calls — the reworked trap held. The
2,500 crossover behaved as coded: chunk corpora at 500/2k (5,261/19,322 rows) exceed it and went
resident-but-evicted; summary routed to the old loader.

## Adjudication
- KILL stands (PASS-5, v2's own band). No claim ships.
- **R2″ (revision, no spec change needed — the v2 gate is unchanged and simply re-runs):** residency
  keyed to POOLED lifetime, not instance lifetime — module/pool-level cache keyed
  (db_path, data_version, identity, budget), shared by per-query instances, invalidation semantics
  unchanged. Expected effect: small rungs build once (~42/126ms) then serve warm — faster than F34, not
  2.9–3.7× slower.
- Banked pending a full pass (no claims): the scale-clause margin, the 200-question parity table, and
  the instrument's third stability reproduction. The run agent also self-corrected a CPython id-reuse
  artifact in its own build counter before reporting — instrument discipline noted.
