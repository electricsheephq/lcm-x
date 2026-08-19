# RUN SHEET — LongMemEval-V1 MEDIUM, RERANK-ON variant (lever L-R1a, from F55)

Status: REGISTERED (this document is the registration; lands on main before any run).
Basis: FINDING-F54 (structural decomposition) + FINDING-F55 (rank-1 forensics: 51/97
delivered misses have gold at rank 2; 82/97 within ranks 2–5).

## 1. The variant (config-only — no code change)
Identical to the banked F53 configuration (RUN-SHEET-V1M-REGISTERED + amendments; lcm-x
main instrument, Voyage voyage-context-3, prepared-m corpus manifest 300cf936…, embed cache,
6-shard layout) with exactly ONE env delta:
- `LCM_RERANK_ENABLED=1` — enables `_lcm_recall_rerank` (tools.py): a fail-open, single
  voyage-rerank API call that reorders the fused top-window INSIDE the production
  `lcm_recall` path. All other arms untouched by design.
The full `_EnvFieldSpec` inventory is captured per shard as always; the diff vs the F53
run-env captures must show ONLY this key (verification step V1).

## 2. Spend (ladder rung: measured-basis estimate, then cap)
Embeddings: cache-served (F55's re-run was zero-spend; identical corpus/queries).
NEW spend: one voyage rerank call per scored question per store — ≤500 calls/arm on
short summary snippets (window ≈ top fused candidates). Voyage rerank pricing puts the
full run + A/A′ in the cents-to-low-dollars range on the existing Voyage account. Hard cap:
if the A/A′ subset (100q) shows > $2 total rerank billing, STOP and re-estimate before the
full 500 (bill read from the Voyage dashboard, not assumed).

## 3. Order of operations
1. A/A′ FIRST (100q fixed-seed subset, seed 20260802, same as F53's): the reranker is a
   provider call, so the 0.00pt determinism of F53 does NOT carry over — the variant's own
   noise floor must be measured before the full run is interpreted.
2. Full 500 (6 shards, same layout/stagger), `--dump-candidates` ON (forensics sidecar is
   now standard; zero marginal cost).
3. Paired per-question comparison vs the F53 checkpoints (the deterministic baseline makes
   every delta attributable to the rerank stage alone).

## 4. Pre-declared bands (aggregate lcm_recall r@1; baseline 0.4999, structural ceiling 0.6552)
- **ADOPT**: r@1 ≥ 0.530 (+3pt) AND demotions ≤ ¼ of promotions (per-question: a demotion =
  gold held rank 1 in F53 and lost it here; a promotion = the reverse) AND recall@10 + ndcg@10
  not degraded by more than the variant's own measured A/A′ spread AND no category's r@1
  drops by more than 2× that spread.
- **GRAY**: +1pt to +3pt, or bands mixed → publish at full resolution; adopt only if the
  demotion/promotion profile and category pattern match the F55 prediction (gains concentrated
  in rank-2/3 misses; preference + temporal categories move most).
- **FAIL**: < +1pt, or demotion-heavy → publish at full resolution; lever L-R1a rejected;
  the fusion tie-break lever (the 31 lcm_recall-specific artifacts, F55 §2) becomes primary.
- Invariant check (not a band): where the rerank window ≤ 10, recall@10 is unchanged BY
  CONSTRUCTION (pure intra-window reorder) — any r@10 delta beyond the window boundary is an
  instrument bug, stop and investigate before reading results.

## 5. Verification (per program discipline)
V1: run-env diff vs F53 captures shows only LCM_RERANK_ENABLED. V2: aggregates recomputed
from per-question checkpoints, never read from the run's own summary alone. V3: rerank-mode
telemetry confirms the REAL reranker ran (no silent placeholder fallback — the F49/F53
mixed-mode class; the per-question `rerank_mode`-style status must be voyage for scored
rows, and any fallback count is disclosed). V4: seven-point disclosure on the row;
RUN-LOG entry; scoreboard row only on ADOPT (GRAY/FAIL publish as findings, not rows).

## 6. Ownership
Program session (release-manager lane) owns launch + verdict. Finding number on completion:
next free (F56 expected). C1's LoCoMo verdict (N1) is independent and takes precedence on
wake.
