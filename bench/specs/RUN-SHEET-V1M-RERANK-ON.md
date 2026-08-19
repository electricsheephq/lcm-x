# RUN SHEET — LongMemEval-V1 MEDIUM, RERANK-ON variant (lever L-R1a, from F55)

Status: REGISTERED (this document is the registration; lands on main before any run).
Basis: FINDING-F54 (structural decomposition) + FINDING-F55 (rank-1 forensics: 51/97
delivered misses have gold at rank 2; 82/97 within ranks 2–5).

## 1. The variant (one explicit instrument flag; NOT env-only — verified)
Identical to the banked F53 configuration (RUN-SHEET-V1M-REGISTERED + amendments; lcm-x
main instrument, Voyage voyage-context-3, prepared-m corpus manifest 300cf936…, embed cache,
6-shard layout) with exactly ONE binding delta: the `lcm_recall` arm runs with the product's
`_lcm_recall_rerank` stage enabled (tools.py — fail-open, single voyage-rerank call,
intra-window reorder). All other arms untouched by design.
⚠ VERIFIED 2026-08-20 before registration: `LCM_RERANK_ENABLED=1` alone does NOT reach the
instrument — `evaluate_question` constructs `LCMConfig(...)` explicitly, which does not
apply env specs (only `LCMConfig.from_env()` does). An env-only "variant" would silently
run rerank-OFF while the env inventory said ON — the F49 silent-mixed-mode class. The
variant therefore requires a small instrument change, landed BEFORE launch:
- runner flag `--recall-rerank` threading `rerank_enabled=True` into the per-question
  `LCMConfig`;
- the binding recorded in the checkpoint header AND the dump-candidates header (config-
  binding fields, so resume/append validation fails closed across mismatched variants);
- per-question `lcm_recall` rerank status telemetry (applied / skipped:<reason>), aggregated
  like the hybrid arm's `rerank_mode`, so a silent fallback is visible and countable (V3).

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
V1: checkpoint + dump headers record the recall-rerank binding; run-env diff vs F53
captures shows no other delta. V2: aggregates recomputed
from per-question checkpoints, never read from the run's own summary alone. V3: rerank-mode
telemetry confirms the REAL reranker ran (no silent placeholder fallback — the F49/F53
mixed-mode class; the per-question `rerank_mode`-style status must be voyage for scored
rows, and any fallback count is disclosed). V4: seven-point disclosure on the row;
RUN-LOG entry; scoreboard row only on ADOPT (GRAY/FAIL publish as findings, not rows).

## 6. Ownership
Program session (release-manager lane) owns launch + verdict. Finding number on completion:
next free (F56 expected). C1's LoCoMo verdict (N1) is independent and takes precedence on
wake.

## Amendment 1 — 2026-08-20: window-10 sub-variant (registered before its run)
**Window-50 A/A′ outcome (the §3 step-1 subset, 95 scored):** V3 clean (95/95 `applied`,
both arms), **A/A′ spread 0.00pt (0/95 discordant — the voyage reranker is deterministic on
identical inputs)**, r@1 +5.93pt (0.4639→0.5232) — but recall@10 **−1.84pt** and r@5 −1.00pt:
the effective window is `min(50, limit×4)` = **50** in the instrument's call path
(tools.py rerank_window; the instrument fetches 50), so the reorder crosses the top-10
boundary and trades delivery for precision. With a 0.00pt noise floor, any delivery loss
violates the §4 ADOPT guard → **the full 500 was NOT run on window-50** (band discipline).
Demotion gate at n=95: promotions 11, demotions 4 (needs ≤2.75) — unresolved at subset n,
binds at the full run.

**Zero-spend synthetic window-10** (exact for a pointwise reranker: relative order of any
candidate subset is window-invariant; computed from the rerank-run dumps + the baseline
dumps, artifacts/synthetic-window10.txt): r@1 **+5.93pt with r@10 +0.00pt** — the entire
gain lives inside the top-10; the 11-50 range contributes only damage.

**Registered sub-variant:** identical to §1 plus a bounded rerank window:
- product change (landed before the run): configurable `rerank_window_limit`
  (LCMConfig field + `LCM_RERANK_WINDOW_LIMIT` env spec; 0 = existing `min(50, limit×4)`
  behavior, byte-identical default) clamping the window in the lcm_recall rerank stage;
- instrument threading: `--recall-rerank-window N` recorded in both config-binding headers;
- this run uses window **10**, making the §4 invariant (r@10 unchanged BY CONSTRUCTION)
  active — any r@10 delta is an instrument/product bug, stop and investigate.
**Order:** A/A′ first (re-measure determinism under the bounded window; expected 0.00pt by
the pointwise argument, but measured, not assumed) → full 500 → paired verdict per §4
(unchanged bands; the demotion gate resolves at n=500) → F56.
