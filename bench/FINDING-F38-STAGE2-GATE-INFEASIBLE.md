# F38 — The Stage-2 V1-small gate is INFEASIBLE AS REGISTERED (zero-spend adjudication); session expansion re-aims at the scale regime. Plus a material caveat on F33's mechanism attribution.

**Date:** 2026-07-29 · **Basis:** zero-spend recounts on existing artifacts (no new runs). Method validated
by reproducing F29's numbers on the banked 444 run to the digit before touching the new baseline. Full
tables/scripts: `session-notes/2026-07-29/hermes-r3-0/artifacts/RECOUNTS-A-B-POWER.md` (+ JSON).

## 1. The current baseline's completeness headroom (Recount A — replaces "oracle ceiling ≈459")

On the release run (455/500): all-answer-turn recall **86.4%** (was 85.6), strata 414/46/19, accuracy
gradient **94.2 / 76.1 / 63.2** (still monotone). The 45 failures: 24 complete-evidence (reader-bound),
11 partial, 7 none, 3 abstention variants. Addressable: counterfactual **2.85 pts**, hard cap **3.6 pts**,
oracle-calibrated realistic **≈1.6 pts** (≈0.6 pt at 40% mechanism efficiency). R2's +11 did not spend the
completeness headroom (per-hit content unchanged; the gain was citability) — the headroom was simply never
as large as the cross-baseline arithmetic implied. §6e.8/§6e.11 disciplines applied throughout.

## 2. The frozen subset is dead as a gate vehicle (Recount B + power memo)

Of the frozen 80 (selected on 444-era wrongness): **18 still wrong** under the current baseline; 18 flipped
right since freezing (11 of them already had complete evidence at freeze — the flips were not
retrieval-driven); same-base placebo b=3/c=1. Binding constraint: **15 of the oracle's 23 flip-qids are
already right** — delivery changes banked two-thirds of the subset effect — and **10 of the 18 still-wrong
rows are oracle-refractory** (failed WITH byte-exact gold injected). Empirical b-ceiling ≈ 8 = bare pass
only at c=0, with c=1 already the measured placebo mode. Pass probability ≈2.7% generous, ≈0% under the
oracle-record bound. **Running the paired experiment as registered would burn 300–600k harness units on a
gate that cannot pass for reasons unrelated to the mechanism.**

## 3. Adjudication (pre-registration discipline, not gate relaxation)

- The gate is **not relaxed and not failed** — its PRECONDITION (subset validity) failed on zero-spend
  evidence before spend. Recorded as infeasible-as-registered. The mechanism is NOT killed by this.
- **RE-SELECT on V1-small is declined** (my call as architect): re-freezing on the 45 current failures
  needs a fresh flag-OFF control arm (selecting on E-wrongness and controlling with E manufactures ≈+4 by
  construction — the memo's mandatory condition), a second oracle run for any efficiency comparator, and a
  second pre-registration — ≈800k units aimed at a lane whose realistic total yield is §1's ≈1.6 pts and
  whose mechanism premise now carries §4's confound. That spend chases a battlefield the program has four
  findings saying is the wrong one (F27→F28→F29→§1).
- **Session expansion re-aims at the scale regime**, where sessions are genuinely scattered and the R3
  thesis lives ("win at scale, provably"): its next gate is retrieval-completeness on the 389× instrument
  (F31/F34 family — zero-to-cheap LLM spend), pre-registered before any run. V1-small keeps only the
  metric-standard story: 86.4% answer-turn completeness with a monotone gradient IS the story; the last
  ~1.6 pts are not.
- **PR #173 merges after its bot-pass fixes** as dormant capability: flag-OFF byte-identical (proven
  twice), review-hardened, and required for scale-regime experiments without branch drift.
- #172/#177 remain product-correctness workstreams with paired non-inferiority gates (unchanged); their
  V1-accuracy upside expectation is written down to §1's numbers.

## 4. Candour caveat on F33's mechanism attribution (publishes forward; R2's shipped text stays as-measured)

The oracle flipped complete-evidence wrong rows at **69.2%** vs incomplete at **63.6%** — the intervention
effect is nearly independent of whether evidence was missing. F33's interventional result (23/35 flips,
p=1.9e-05) stands as measured; its ATTRIBUTION narrows: injection largely worked by **salience/emphasis**,
not completeness per se, so b=23 was a contaminated upper bound for any completeness mechanism. R3
messaging states the metric-standard claim on the observational gradient + the market argument, and
qualifies "causally validated" to "interventionally supported, salience confound documented." Residual
open question (would need an oracle re-run on the current baseline): whether the 10 refractory rows resist
evidence per se or the card FORMAT.

## 5. Owner visibility

This closes an owner-approved lane's V1-small chapter on evidence (#150/#25 → the lane continues at scale).
Reversible: the frozen-80 artifacts, the re-selection procedure, and the estimation-run option (a spec
change, owner's call) are all documented in the memo if the owner prefers a different disposition.
