# F24 — V1 run-to-run noise floor is ±6 questions per 100, measured with IDENTICAL code

**Date:** 2026-07-26 · **Credit:** the executing cross-test agent, which added a repeat-baseline arm I did not
specify in the dispatch. Without it the cross-test would have been unreadable.

---

## 1. The measurement

Same #423 code, same 100-question slice, same answerer/judge (`gpt-5.6-sol`, effort medium):

| arm | correct |
|---|---|
| banked — verdicts from the original 444/500 run | **44/100** |
| **repeat — fresh run, identical code** | **50/100** |
| **difference** | **+6 questions, zero mechanism change** |

## 2. What it changes

**(a) The bar for the cross-test moved.** wave-1 must beat **50**, not 44, and by more than the ±6 band, before
"wave-1 is better on V1" means anything. A wave-1 result of 45–49 would have *looked* like a win against the
banked 44 while actually sitting **below** the same-code control.

**(b) The banked 444/500 is not a point, it is a sample.** A ±6 swing on 100 questions implies the 500-question
figure carries real variance too. The slice here is enriched with the 56 hardest questions (those #423 got
wrong), and boundary questions flip most easily, so ±6 likely *overstates* variance on a random slice — but the
banked 444 should be treated as **444 ± several**, not as an exact integer.

**(c) It tempers F23's per-category decomposition.** F23 attributed the 22-question gap to OMEGA as
single-session −7 / preference −5 / multi-session −4 / knowledge-update −4 / temporal −2. With ±6 noise per 100,
the smaller per-category figures are **not individually reliable**. What survives:
- the **total** 22-question gap is well outside noise;
- **preference application (25/30 vs 30/30)** remains the sharpest signal because OMEGA is at a **ceiling** —
  30/30 cannot be noise upward, so the deficit is real even if its exact size is ±2;
- the −2 temporal and −4 figures should be treated as directional only, not as work items sized to the question.

**(d) It retroactively strengthens the M11/M16 power argument.** We now have a second independent measurement of
run-to-run variance on a different benchmark and harness: V2 gave 3/6 vs 5/6 on identical config (M11 §8), V1
gives 44 vs 50 per 100. **Stochastic decoding means every arm we run carries a several-question band, and any
mechanism claim smaller than that band is unmeasurable by a single run.**

## 3. Standing rule (new)
**Every A/B on V1 or V2 must include a repeat of the CONTROL arm**, not just control-vs-variant. Without a
same-code repeat there is no way to distinguish a real delta from resampling, and the temptation is always to
read the difference as signal. Budget the repeat as part of the experiment, not as an optional extra.

This is the same class as the M16 enriched-slice fix (power the instrument before trusting the reading) and it
is cheap: one extra arm converts an uninterpretable number into an interpretable one.

## 4. Process note
I specified a paired comparison against a single banked baseline. The executing agent added the repeat arm
unprompted. **That is the third time in two days that a subagent's addition or objection corrected the
orchestrator's design** (M15 §7/§8 asymmetric adjustment and false premise; now this). Dispatch prompts should
keep explicitly inviting exactly this.
