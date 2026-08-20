# FINDING F56 — RERANK-ON window-10 full-500: +4.24pt r@1, and the bands said no

Date: 2026-08-20. Registration: RUN-SHEET-V1M-RERANK-ON + Amendment 1 (this run followed the
registered order exactly: window-50 A/A′ → synthetic window-10 → product/instrument change
(#243) → window-10 A/A′ → full 500). Instrument main @ e4e34493. Artifacts:
session-notes/2026-08-20/v1m-rerank-{aa,full}/.

## 1. The paired full-500 result (470 scored, vs the F53 deterministic baseline)
| metric | F53 base | rerank w10 | delta |
|---|---|---|---|
| recall@1 | 0.4999 | **0.5423** | **+4.24pt** |
| ndcg@10 | 0.8610 | 0.8835 | +2.25pt |
| recall@5 | 0.8977 | 0.8900 | −0.77pt |
| recall@10 | 0.9559 | 0.9495 | −0.64pt |

Determinism carried: BOTH A/A′ pairs (window-50 and window-10) were 0/95 discordant, 0.00pt —
the voyage reranker is deterministic on identical inputs, so every delta above is attributable.
V3 telemetry: 467/470 `applied`, 3 `skipped: no candidates to rerank` (fail-open, disclosed).

## 2. Verdict per the pre-declared §4 bands: **GRAY → NOT ADOPTED**
- r@1 ≥ +3pt: PASS (+4.24).
- delivery guard (r@10/ndcg not degraded beyond the 0.00pt spread): **FAIL** (r@10 −0.64pt).
- demotion gate (demotions ≤ promotions/4): **FAIL** — 46 promotions, 16 demotions (2.9:1 vs
  the required 4:1). Demotions by category: temporal 7, multi-session 4, preference 2, user 2,
  assistant 1.
- GRAY adopt condition (profile matches the F55 prediction): PARTIAL — preference moved as
  predicted (+6.67pt) but temporal barely (+1.78pt), and the largest mover was unpredicted
  single-session-user (+14.06pt). Per-category deltas: user +14.06, preference +6.67,
  knowledge-update +3.47, multi-session +2.62, assistant +1.79, temporal +1.78.
The declared V1-M config of record REMAINS the F53 configuration (rerank off).

## 3. Correction (append-only): the Amendment 1 invariant was mis-derived
Amendment 1 claimed r@10 is unchanged BY CONSTRUCTION at window ≤ 10. Wrong: the reorder acts
on the top-10 ITEMS, but the scorer consumes the DEDUPED SESSION list — when the item head
contains duplicate sessions, reordering changes which sessions enter the deduped top-10 from
below the boundary. Investigated per the sheet's stop-and-investigate clause: exactly 3
questions changed r@10 (all down), and all 3 mechanically confirm this dedup-boundary
mechanism (session set changed + duplicate sessions present in the item head). Not an
instrument or product bug; the invariant clause is corrected by this finding (Amendment 2
marks it in the sheet). The window-10 A/A′ subset showed exact invariance only because its
95 questions happened to contain no duplicate-session heads among the reordered items.

## 4. What the result teaches (next lever, declared)
The reranker finds real top-slot signal (+4.24pt, ndcg up) but overrides 16 previously-correct
rank-1s — an unconditional reorder is too blunt. Next registered iteration:
**margin-gated override** — keep the incumbent rank-1 unless the reranker's top score beats
the incumbent's by a pre-set margin (directly attacks the demotion count while keeping most
promotions), optionally combined with the zero-provider fusion tie-break for the 31
lcm_recall-specific artifacts (F55 §2). Each iteration costs ~30 min + cents at full-500
scale thanks to the deterministic instrument; the bands stay as registered.

## 5. Disclosures
(1) Spend: ~1,140 voyage rerank calls total across all runs (~cents; hard-cap margin >100×,
dashboard not separately read). (2) The 3 rerank-skipped questions are scored normally (the
stage fail-opens to the incoming order). (3) GRAY/FAIL rows do not enter the scoreboard per
the run sheet; this finding is the full-resolution publication. (4) All aggregates recomputed
from per-question checkpoints (V2); headers bind recall_rerank=true, window=10 on all shards.
