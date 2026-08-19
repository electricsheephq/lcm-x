# F36 — Gate 4 GREEN: fail-close class eliminated (16→0), and the R2-train V1 gain is REAL — confirmed stable across the citability rebuild.

**Date:** 2026-07-29 · **Run:** `sanity100-r2citable-…021239Z`, product `2edb8fc` (PR #174 merged after the
8-round review cycle), F32 pins verified pre/post byte-identical, stores 200/200 unchanged, 351k harness-unit
tokens. **Adjudicated against the F35 §4 pre-declared gate.**

## 1. The gate

| clause | bar | observed | verdict |
|---|---|---|---|
| fail-closes | ≤ 2 (F32 baseline) | **0** (16 in the RED run) | **PASS** |
| paired flips | within ±6 same-code band OR stable measured gain | +18 vs banked (p=4.0e-05) — the GAIN branch | **PASS, see §2** |

All 16 previously-destroyed rows scored (10 right / 6 wrong); zero failures in any phase. The uncitable-delivery
class is eliminated at the harness seam, not just in unit tests.

## 2. The +17 question is answered: the gain is real, and the citable fix did not manufacture it

The load-bearing comparison: **vs the previous consolidated arm (543e9ea) on the 84 common scored rows: b=2,
c=3, net −1, p=1.0.** Eight review rounds and a delivery-engine rebuild moved the previously-scored population
by *nothing* — the gate-population +0/−0 invariant held in the live harness. Therefore:

62/100 = 52 (the R2-train effect on scored rows — now measured TWICE independently: 53/84 pre-fix, 52/84
post-fix) + 10 (recovered fail-closed rows). Against baselines: +18 vs banked A=44 (p=4.0e-05), +12 vs the
same-code repeat A′=50 (p=0.017), both far outside the A/A′ noise reference (+6, p=0.109). Per-category:
broad-based (multi-session +6, assistant +5, temporal +3 vs A) — not a single-category artifact.

**Mechanism attribution (from F35's chain):** the #168 in-product sanitization made the provider's internal
FTS/summary-arm queries work on V1 for the first time; the improvement predates the citable fix and survives it
unchanged. This is the first measured V1 accuracy gain attributable to our own code in the program's history —
on a failure-enriched slice.

## 3. What this licenses and what it does not (§6e.8 discipline)

The slice is failure-enriched: **62/100 is not a category rate and licenses NO /500 claim.** Per the
consolidation checklist's own trigger ("full re-run ONLY if the slice moves beyond the band" — it did), the
**full-500 confirm run on 2edb8fc** is now dispatched under the F32 pins; its paired result vs the banked 444
becomes the release's V1 number. Expectation-setting from the slice: the banked run's 56 failures are heavily
sampled here; the full-500 gain will be SMALLER than +18 (most of the 500 are passes with nowhere to go up and
some room to break). No number is promised.

## 4. The review cycle, banked as method (summary; ledger in routing-ledger.jsonl)

Eight rounds · fifteen P2s · three structural rewrites (entry-lifecycle ledger → positional cursor → derived
resume), ending at a provable termination condition: **no walk state exists outside the ledger or its
derivations.** The test suite out-ran the max-effort reviewer in round 7 (the extended property generator found
a defect review had passed over twice) and the histories upgrade caught an unreachable-transition gap on its
first run. Gate population +0/−0 across every round. Standing lesson reinforced from four separate incidents
this cycle: specifications (mine included) are hypotheses — the author corrected my prescribed mechanism once,
my repro shape once, and my finding text twice. The pipeline's redundancy is the product.
