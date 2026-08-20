# FINDING F57 — margin-gated rerank override: rejected by its own pre-registered rule, no full run spent

Date: 2026-08-20. Registration: RUN-SHEET-V1M-RERANK-ON Amendment 3 (#248; product/instrument
support #249). Artifacts: session-notes/2026-08-20/v1m-rerank-aa/artifacts/
(margin-selection.txt, w10s run outputs).

## 1. What ran
The 100q subset once more on the post-#249 tree with rerank-score telemetry on (margin 0).
Cross-tree determinism held again: 95/95 questions metric-identical to the pre-#249
window-10 run — third consecutive 0-discordance measurement on this lever.

## 2. The pre-registered selection rule fired REJECT
Gap = challenger relevance − incumbent relevance, on the 13 override questions
(10 promotions, 3 demotions vs the F53 baseline):
- demotion gaps: 0.0117, 0.0195, **0.1855**
- promotion gaps: 0.0254, 0.0273, 0.0273, 0.0313, 0.0430, 0.0703, 0.0781, 0.0996, 0.1484, 0.3770
The rule (M = smallest promotion gap strictly greater than every demotion gap; valid only if
≥80% of promotions retained at gap ≥ M) yields M = 0.3770 retaining **1/10 = 10%** →
**REJECTED without a full run**, exactly per the registered no-valid-M/retention clause.

## 3. What it means
The reranker's confidence gap does NOT separate its correct overrides from its wrong ones
here: one demotion (gpt4_d6585ce9) carries a 0.186 gap — the reranker is confidently wrong —
while most correct promotions sit below 0.10. Margin thresholding on rerank scores is not a
viable demotion guard for this corpus/model pair. The rerank lever family closes with:
unconditional window-10 = GRAY not adopted (F56); margin-gated = rejected pre-run (this).
Declared V1-M config REMAINS F53 (rerank off).

## 4. Next lever (from F55 §2, unchanged)
The 31 lcm_recall-specific rank-1 artifacts (no other arm makes the same mistake) — a
zero-provider fusion tie-break. First step is a zero-spend decomposition of those 31 from
the existing dumps (what does fusion rank above gold that every individual arm ranks
below?). Registered before any product change, as usual.

## 5. Disclosures
(1) Incumbent relevance was matched by session id; on questions whose window contains
duplicate entries of the incumbent session (most of the 13), the MAX relevance among them
was used — this biases gaps conservatively and applies to promotions and demotions alike;
the near-total distribution overlap is robust to it. (2) Spend: 95 rerank calls (~cents).
(3) All numbers recomputed from checkpoints/dumps; scripts in the artifacts dir.
