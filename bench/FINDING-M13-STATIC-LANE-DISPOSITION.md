# M13 — [PARTIALLY VOIDED — see §0] Static lane disposition

## 0. ★★ VOIDED ANALYSIS — my §1 numbers were produced by BLENDING ARMS (owner challenge, 07-26)

**The §1 table is wrong and its method was broken.** The owner asked why we would retire working code; checking
the evidence showed my analysis, not the code, was the problem.

Each `W3B-DEV*-RAW` directory contains **multiple arms**, not one: DEV1 = armA + armB **+ a directory explicitly
named `web-run-INVALID-wrong-branch`**; DEV3 = armE + armF; DEV4 = armG + armH. My loader globbed
`**/per_question.jsonl` into a dict keyed by `question_id`, so **later files silently overwrote earlier ones** —
each "DEV" figure was an arbitrary blend of two arms, and DEV1 ingested a run I had myself labelled INVALID.

**Correct per-arm scores (arms separated, INVALID excluded):**

| arm | M13 §1 claimed | ACTUAL | decoding |
|---|---|---|---|
| armA | 20/60 | **19/60 = 31.7%** | pinned — valid |
| armB | (blended away) | **8/60 = 13.3%** | pinned — valid |
| armC | 20/60 | **20/60 = 33.3%** | pinned — valid |
| armE | 21/60 | **22/60 = 36.7%** | **UNPINNED — void (M9)** |
| armF | (blended away) | 21/60 = 35.0% | **UNPINNED — void (M9)** |
| armG | 22/60 | 18/60 = 30.0% | **UNPINNED — void (M9)** |
| armH | (blended away) | **15/32 = 46.9%** (web only) | **UNPINNED — void (M9)** |

**Paired McNemar, pinned arms only:** armA→armB **−11q, p=0.027 (a SIGNIFICANT REGRESSION I missed entirely)**;
armB→armC **+12q, p=0.017 (a SIGNIFICANT RECOVERY I missed)**; armA→armC +1q, p=1.000.

**WHAT SURVIVES:** across the *pinned* arms the net is flat (armA→armC = +1 question). That single conclusion
holds.

**WHAT IS RETRACTED:** the claim "every step p=1.000, whole loop +2q p=0.791" — false, built on blended arms.
And critically, **the claim that the W3B mechanisms are flat.** Arms **E/F/G/H are the anti-boilerplate MMR,
title-boost and compact-delivery mechanisms on `bench/w3b-compact-delivery`, and all four ran UNPINNED**
(M9) — they are **CONFOUNDED, NOT NEGATIVE.** They also carry the two highest observations in the whole static
lane (armE 36.7%, armH 46.9% web-only). **"They don't work" was never established; "we never measured them
properly" is the truth.**

Related: the H3.1 composition A+D policy is recorded in its own quarantine commit as **"2nd powered fail +7 vs
≥+8"** — a one-question miss on a powered test, not a demonstration of no effect.

**ROADMAP CORRECTION.** §4's "defund the mechanism SEARCH" is **withdrawn as stated**. The correct disposition:
- **Do NOT retire the `bench/w3b-compact-delivery` code.** Rebase it onto `upstream-wave-1` so one branch is
  genuinely current. It is default-off and costs nothing until enabled.
- **The static mechanisms need ONE properly-powered, pinned, paired measurement** — an enriched slice per M16,
  pinned decoding per M9, arms separated. Until then their status is **UNKNOWN**, not flat.
- What remains true from §2/§3: the value structure is a cliff (below 51.0% static scores exactly 0.0000) and
  the arithmetic to reach it is daunting. That argues for *sequencing* static behind the scale work — **not for
  discarding the code.**

**Process lesson (§6e-class):** I globbed a directory and assumed it was one arm. Same failure family as M17/M18
(asserting a property of the data instead of enumerating it) — and this one killed a lane's funding for a day.
**RULE: before scoring a results directory, list its subdirectories and assert one arm per score.**

---

# M13 (original text follows — §1 numbers VOID per §0)


**Date:** 2026-07-25 · **Status:** decision record (architect) · **Supersedes:** PLAN v6 §3 P3 "re-aim or defund"
**Method:** M11 §8 / M12 §4 rules applied to the static lane's own record.

---

## 1. The static dev loop produced nothing statistically detectable

Paired McNemar across the W3b dev arms (same 60q manifest throughout):

| step | accuracy | delta | McNemar p |
|---|---|---|---|
| DEV1 | 20/60 = 33.3% | — | — |
| DEV1 → DEV2 | 20/60 = 33.3% | +0 q | **1.000** |
| DEV2 → DEV3 | 21/60 = 35.0% | +1 q | **1.000** |
| DEV3 → DEV4 | 22/60 = 36.7% | +1 q | **1.000** |
| **DEV1 → DEV4 (whole loop)** | | **+2 q (+3.3 pts)** | **0.791** |

**Four iterations, no detectable improvement.** DEV3/DEV4 are additionally the unpinned-decoding arms
voided by M9 — so the two "best" arms are both confounded *and* non-significant.

## 2. The value structure is a cliff, and it is unforgiving

Static latency is 0.109s — roughly 250x inside window B's 26.9s ceiling. So static is a **pure accuracy
problem**, and its value function is a step:

| static accuracy | LAFS |
|---|---|
| 27.7% (banked) · 35% · 42.8% · 51.0% | **0.0000** |
| 51.5% | 0.311 |
| 55.0% | **2.485** |
| 58.6% | **4.722** |
| 62.0% | **7.729** |

Above the cliff static is **the steepest thing in the program** — 4.72 at 58.6% versus agentic's 1.89 at
72.1%. And the lanes **add** (a submission is a curve, not a point): static 55% + agentic 66.1%@51.6s =
**3.535**.

So static is not low-value. It is **high-value and currently unreachable**: it needs **+23.3 points**
(125→230 of 451) before it contributes a single thousandth.

## 3. The arithmetic on the two known static loss masses

- **M7 abstention class:** static abstention accuracy is 12.5% (16/128). Lifting it to 50% = **+48 q**
  → 173/451 = 38.4%. *Still below the cliff.*
- **M1 spurious-unknowns:** the reader answers "unknown" on 95/323 answerable questions while holding
  full untruncated ~22k contexts. Recovering half = **+47 q** → ~220/451 = 48.8%. *Still below the cliff.*

**Both of the largest known masses, both fixed at optimistic rates, still land short of 51%.** That is
the honest case, and it is why a broad mechanism search here is a bad bet.

## 4. DECISION

**(a) Defund the static mechanism SEARCH.** No further exploratory dev-loop iterations, no arm-E/F
re-measurement for its own sake. Four iterations of that loop bought nothing measurable; a fifth has no
better prior. The code already built (diversity-cap, adaptive-excerpt, state-semantic-quota,
antiboilerplate, title-boost) is retained as valid, merge-worthy code and may ship upstream on its
engineering merits — it is simply not an accuracy strategy.

**(b) Port M7 to static once it validates on agentic.** This is the one mechanism with a real mass
behind it (+48 q projected, the largest single static class), it is content-only and cheap to port
(the static compilation stage emits a deterministic absence line when a question-derived key term has
zero pool hits), and M7's own text always specified both lanes. **Conditional on the agentic pilot
passing its gate** — we do not port an unvalidated mechanism.

**(c) Re-evaluate static only after M7-static lands.** If M7-static delivers near its +48 projection
(→ ~38%), the cliff is 13 points away and a *targeted* spurious-unknown attack becomes a rational
follow-on. If it underdelivers, static is done as an accuracy lane and we say so publicly.

**(d) Meanwhile the agentic lane carries the program.** It is 2 points from its window (not 23), it has
a mechanism in flight, and per M12 it has the program's only statistically established capability
result (−22% latency, p<0.0001).

## 5. Why this is not the same as "defund"
Static remains the biggest prize on the board (7.73 at 62%). We are not abandoning it — we are refusing
to keep paying for a search that has produced four consecutive null results, while keeping the one
port that has evidence behind it. **Fund mechanisms with measured mass; stop funding exploration that
four iterations have shown to be flat.**

## 6. Instrument note that makes all of the above possible
None of §1 was visible from headline percentages — 33.3 → 36.7 reads like progress. It is only visible
paired. Per M12 §4, **every future dev-loop step ships its McNemar and discordant counts, or it is not a
result.**
