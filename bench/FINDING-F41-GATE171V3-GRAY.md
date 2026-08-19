# F41 — #171 gate, third execution: GRAY on PASS-5 alone (owner decision per the registered spec); every other clause passes with margin. The scale result is banked-quality: 222.7ms @19,829 (28× vs F34), recall parity 280/280.

**Date:** 2026-07-29 · **Run:** gate171-v3 on 027c738 (R2″ pooled residency), spec v2 UNCHANGED,
pins/stores/probe-lists sha-identical to the prior executions, 0 fail-closes, zero LLM. Artifacts:
session-notes/2026-07-29/hermes-r3-1/artifacts/gate171-v3-run/.

## Mechanism verified fixed
Residency builds **1/100 queries at every rung** (F40: 100/100 at 500/2k); registry survives close()
(structural probe: entry counts identical across close at every rung); 0 LRU evictions (121.5MB under
the 128MB budget); `_release_matrix_caches` 0 calls. Per-query store churn itself is UNCHANGED (the
finalization site still fires 9/10 at small rungs) — the pooled registry simply makes it harmless to
residency. That churn remains a real product observation → follow-up issue.

## Clause reading
MET with margin: PASS-1 (net 0 across all 280 paired questions, both baselines — the pooled change
produced ZERO delivery changes), PASS-2, **PASS-3 (222.73ms ≤ 350; F34 6,311 → 28.3×; 8k: 98.9ms →
18.1×)**, PASS-4 (0/400 ≥8s), PASS-6 (coverage truthful, 100% of rows).
**GRAY: PASS-5.** 500: 1.26–1.42× raw, **0.98–1.22× machine-normalized (meets the 1.25 PASS line
normalized)**; 2000: 1.48–1.57× raw, 1.32–1.40× normalized. No basis touches the 1.6× KILL line.
Same-session B control ran 1.02–1.17× vs F34 (ambient normalization stated per basis).
Power note: the F34-archive pairing keeps its inherent 20-pair ceiling (<20 discordants at top rungs);
the v2-baseline n=50 pairing carries pools of 33/36 there — the parity conclusion rests on both.

## Absolute framing for the GRAY decision (the numbers behind the ratios)
Small-store penalty in absolute terms: 500-session stores 20.8→28.9ms (+8.1ms); 2k stores 45.5→69.9ms
(+24.4ms) — imperceptible in product terms — purchased for 18–28× at 8k–19.8k where the product
previously took 1.8–6.3 seconds. Cold first-touch published alongside (176–626ms small rungs).

## Architect recommendation (decision is the owner's per the registered GRAY clause)
**ACCEPT the GRAY and ship the mechanism**, with the small-rung cost published in the notes (the R2
candour pattern: both sides of the curve, always). The residual 2k overhead traces to per-query store
churn — a follow-up product issue, not another in-train revision (diminishing returns; two revisions
already landed cleanly). Alternative if declined: revision targeting store-instance pooling (larger
product change), gate runs a fourth time.
