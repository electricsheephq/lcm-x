# SPEC M7b — conditional absence disclosure, on an enriched slice, dual-consumer

**Issue:** #157 · **Author:** orchestrator · **Date:** 2026-07-25 · **Status:** FROZEN before run
**Depends on:** M15 (M7 NO-GO diagnosis) · M16 (enriched slice + dual-consumer architecture)

## 1. Why this variant exists
M7 met its primary (abstention 29.4%→52.9%, +4q) and breached its answerable floor (−7.0 pts
artifact-adjusted). Diagnosis (M15): **all 7 answerable losses carried `directly_supported`** and the reader
went UNKNOWN on **0** of them — so the harm is pack-quality collateral on questions the mechanism never
targeted, NOT over-abstention. Absence statuses score **8/10** where they fire. Two independent measures show
`directly_supported` packs were perturbed anyway: spans +0.19 / ctx +779 tok / 11-of-37 differing, and
**answerable searches +12% (5.81→6.49)**.

## 2. The two changes (both required — rendering alone is insufficient)
**(a) Conditional RENDER.** Emit `## Evidence Assessment` only when `evidence_status` ∈
{`near_match_only`, `contradicts_premise`, `insufficient`}. Never on `directly_supported`.
**(b) Conditional SEARCH.** The M7 contract demanded a targeted absence search *unconditionally, before any
status was known* — the direct cause of the +12% answerable tax. New contract wording: *do the normal
search first; only if it fails to surface the entity/field/relation the question presumes, spend ONE targeted
confirmation search and record the queries and hit counts.*
Everything else identical to M7: gate enabled, declarative-only rendering (no `answer_policy` verbatim, no
reader directives — that constraint held 60/60 and remains binding), low effort, decoding pinned 0.6/0.95/20.

**Also test render POSITION.** M7 prepended the section FIRST, taking the slot a weak 9B reader weights most
(M1/M4) — a plausible contributor to the collateral. Place it AFTER Support Analysis. If that inverts a
result, it is a finding about the reader, not a bug.

## 3. Slice — ENRICHED (M16), not random
**All 128 abstention questions + 64 randomly sampled answerable = 192q**, frozen manifest with sha256.
Rationale: a random 60q slice holds only ~17 abstention questions and cannot see a real effect (p=0.22 for
the +23.5pt lift we actually measured). At n=128 the same effect reads p<0.0001 and even a modest +8pt
effect reads p=0.011.
**A matched CONTROL arm (gate off, same manifest, same config) is part of the run, not optional** — L3 does
not cover this slice, and per M12 comparisons must be paired on identical questions.

**★ PREMISE CORRECTION (see FINDING-M15 §8):** an earlier version of this spec justified the control ruling by
claiming the control's `INSTRUCTION.md` carries "M7-era instruction improvements". **It does not** — the
control's `INSTRUCTION.md` is **byte-identical to L3's** (`450668d0…`) and the bullet cited as evidence is
verbatim in d8f1d90. The ruling's outcome is unchanged (hold everything constant except the gate + conditional
logic; carry no L3 numbers across), but **non-comparability with L3 is purely SLICE-BASED**, and any inference
of the form "instruction changes barely move X" drawn from control-vs-L3 is void.

## 4. Dual-consumer measurement (M16 §2b) — the product objective is not the leaderboard objective
- **PRIMARY (leaderboard):** official fixed Qwen3.5-9B reader.
- **PRODUCT CHECK (Sol):** re-read the **same stored `memory_context` packs** with a frontier reader.
  Curation is the expensive shared stage, so this is a reader-only second pass (~1.2x, not 2x). Do NOT
  re-run the agent.
Report both. **Sol is the product the owner and customers actually run**; a mechanism that helps the weak
reader and harms the frontier one is a product regression and must not ship default-on.

## 5. PREDECLARED GATE (frozen — not revisable at one-short)
| axis | measure | bar |
|---|---|---|
| **PRIMARY** | abstention-subset accuracy, n=128, paired | **up AND McNemar p<0.05** |
| **FLOOR (hard)** | answerable-subset accuracy, n=64, paired | **not down >2.0 pts** vs THIS run's CONTROL (artifact-adjusted **symmetrically** — see M15 §7; never against M7's superseded −7.0) |
| **PRODUCT** | frontier-reader accuracy on same packs | **not down** vs control |
| SECONDARY | overall accuracy, latency, `searches_per_question` by class | reported |
| INSTRUMENT | provider-error rows, both arms, both readers | counted BEFORE any comparison |

Primary now requires **significance**, not just direction — the enriched slice makes that affordable and
M11–M15 showed direction alone is noise. One-primary law holds. 192q is powered on the primary but remains a
**screen for banking**: promotion to a published/submitted number still requires the full 451 (M16 Tier 2).

**Outcomes.** Both pass → full-451 confirmation. Primary passes, product harmed → **config-gate it off by
default, labelled benchmark-only** (M16 ship rule). Floor breached again → the mechanism is not separable
from its collateral; **stop the M7 family and say so** rather than producing M7c.

## 5b. ★ GATE CALIBRATION NOTE — recorded BEFORE the M7b arm produced any numbers

Computed from the completed control's enterprise half, while the web half was still scoring. **The answerable
FLOOR is miscalibrated relative to the noise it must survive**, and this is stated in writing now so it cannot
later be rationalised away (cf. `feedback_gate_proxy_calibration`: miscalibration must be documented in
writing, not discovered at one-short).

- answerable subset n=64 → **one question = 1.56 points**; the 2.0-point bar therefore permits losing **exactly
  one** question. Two questions = 3.12 points = breach.
- 95% CI half-width on a proportion at n=64 is **±11.7 points**. **The bar sits far inside the noise band.**
- That variance is not hypothetical: M11 §8 measured identical config, identical 6 questions, different run →
  3/6 vs 5/6.

**The PRIMARY is fine** — abstention n=128 gives p≈0.00001 for an M7-sized lift. The enriched slice fixed the
axis it was designed to fix. It did not fix the floor, because the floor rides on the 64 answerable questions.

**The bar is NOT relaxed.** A breach is still a breach and still yields NO-GO. What is predeclared here is what
a NO-GO *procedurally means*, so the verdict cannot be over- or under-read:

| result | verdict | correct next action |
|---|---|---|
| primary passes (p<0.05) **and** floor holds | **PASS** | full-451 confirmation |
| primary passes **and** floor breaches by **≤2 questions** | **NO-GO, floor INCONCLUSIVE** | confirmation run at larger n on the answerable side — **not** ship, and **not** abandon the mechanism |
| primary passes **and** floor breaches by **≥3 questions** | **NO-GO, floor HARMFUL** | the mechanism is not separable from its collateral; stop the M7 family |
| primary fails | **NO-GO** | floor is moot; the channel is not the constraint |

**Design lesson for the Phase-2 eval loop:** an auto-powered slice generator must power **every gated axis**,
not just the primary. Powering the primary and leaving the floor at coin-flip resolution is a half-fix, and I
shipped it into this spec.

## 5c. ★ M7 PREMISE RE-VALIDATED at n=128 — and it yields a sharp adjudication criterion

Measured on the completed control (128 abstention questions, vs the original finding's smaller samples):

| | control (n=128) | original M7 finding |
|---|---|---|
| reader **asserted** on abstention questions | **115/128 = 89.8%** | 122/128 = 95% |
| of the 89 FAILED abstention questions, assertions | **76/89 = 85.4%** | 73 wrong hallucinations |
| reader said UNKNOWN | 13/128 = 10.2% | 6/128 |
| **score when the reader DID say UNKNOWN** | **0 / 13 = 0%** | **0 / 6 = 0%** |

**The premise holds.** The dominant abstention failure is the reader confidently asserting against a false
premise, not abstaining.

**★ ADJUDICATION CRITERION (predeclared): a bare "UNKNOWN" is worth ZERO.** Now 0/19 across both samples. The
official checker requires asserting the **specific negative conclusion** ("there is no second field"), not a
bare refusal. Therefore, when reading M7b:

- if the abstention gains come with **increased UNKNOWN responses** → the mechanism is teaching the reader to
  refuse, which scores nothing. Gains would have to come from elsewhere, and the mechanism is mis-shaped.
- if the gains come with **specific negative assertions** → the mechanism is doing exactly what is required.
- **Report the UNKNOWN rate on the abstention subset for both arms.** A rise in UNKNOWN alongside a rise in
  accuracy is a coincidence to be explained, not a success to be banked.

Encouraging prior signal: in the M7 pilot, all 5 abstention gains were **non-UNKNOWN** — specific negative
assertions. That is the behaviour we want, and it is what the declarative-evidence design (§2) was chosen to
produce rather than an imperative "say you don't know."

## 6. Ship rule reminder
helps both → default-on · leaderboard-only/product-harmful → config-gated off · product-only → ship anyway.
