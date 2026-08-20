# FINDING F57 — margin-gated rerank override: rejected by its own pre-registered rule, no full run spent

Date: 2026-08-20. Registration: RUN-SHEET-V1M-RERANK-ON Amendment 3 (#248; product/instrument
support #249). Artifacts: session-notes/2026-08-20/v1m-rerank-aa/artifacts/
(margin-selection.txt, w10s run outputs).

## 1. What ran
The 100q subset once more on the post-#249 tree with rerank-score telemetry on (margin 0).
Cross-tree determinism held on every pass: 95/95 metric-identical across the pre-#249,
post-#249, and post-#251 trees — four consecutive 0-discordance measurements on this lever.

## 2. The pre-registered selection rule fired REJECT (exact gaps)
A review round caught that session-matched incumbent identification only lower-bounds the
gaps on duplicate-session windows; the telemetry was extended to carry the provider-order
INPUT index (#251) and the deterministic subset re-run (metrics identical — the FOURTH
consecutive 0-discordance check on this lever). EXACT gaps, challenger − exact incumbent
(input index 0), on the 13 override questions (10 promotions, 3 demotions vs F53):
- demotion gaps: 0.0195, 0.1855, **0.1953**
- promotion gaps: 0.0273, 0.0273, 0.0313, 0.0430, 0.0449, 0.0703, 0.0996, 0.1484, 0.1992, 0.3770
The rule (M = smallest promotion gap strictly greater than every demotion gap; valid only if
≥80% of promotions retained at gap ≥ M) yields M = 0.1992 retaining **2/10 = 20%** →
**REJECTED without a full run**, exactly per the registered no-valid-M/retention clause.
(The earlier session-matched pass reached the same verdict with M = 0.377 retaining 1/10;
exactness made the picture sharper — the top demotion gap had been UNDERESTIMATED at 0.186.)

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
(1) The first analysis pass matched incumbents by session id (gaps = lower bounds on
duplicate-session windows); a reviewer correctly flagged that the ordering was not provably
exact, so the rule was recomputed from exact input-indexed telemetry (#251) — same verdict,
sharper numbers (§2). (2) Spend: 190 rerank calls across both telemetry passes (~cents).
(3) All numbers recomputed from checkpoints/dumps; scripts + both passes' outputs in the
artifacts dir (margin-selection.txt, margin-selection-exact.txt).
