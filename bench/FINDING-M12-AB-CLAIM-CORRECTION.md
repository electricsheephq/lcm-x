# M12 — The headline A/B claim is wrong on accuracy and right on latency

**Date:** 2026-07-25 · **Status:** correction, decision-bearing · **Trigger:** M11 §8 (paired comparison
beats headline percentages) applied to our own capability claim before anyone else applied it for us.

---

## 1. The claim as it stood

`H6-P4-FULL-REPORT.md` and PLAN v5 both carried:

> **vs OUR vanilla-Codex baseline (the controlled A/B): WIN on both LAFS axes** — hermes 66.1% @ 197s vs
> vanilla 63.3% @ 256s... **First evidence the memory measurably helps an agent.**

Two defects.

**(a) The two numbers are from different question sets.** hermes 66.1% is the **full 451**; vanilla 63.3%
is the **60q slice** (`H6-P1`). A vanilla@451 run does not exist — the report itself says so. Comparing
them as a controlled A/B is invalid.

**(b) The accuracy effect is not significant on the paired data.** Three hermes arms exist against the
same vanilla control on the same 60 qids, and McNemar on all three is null:

| hermes arm | hermes | vanilla | delta | discordant (b/c) | McNemar p |
|---|---|---|---|---|---|
| P3 (first) | 40/60 | 38/60 | +2 q (+3.3 pts) | 5 / 3 | **0.727** |
| P3R (post-fix) | 38/60 | 38/60 | **0 q (0.0 pts)** | 5 / 5 | **1.000** |
| P4-valslice (the banked config) | 40/60 | 38/60 | +2 q (+3.3 pts) | 7 / 5 | **0.774** |

The "+3.4 pts" in the report is the P3 / P4-valslice point estimate. **The spread across our own three
hermes arms (40, 38, 40 — range 2 questions) is exactly the size of the claimed effect.** Per M11 §8 that
is the expected magnitude of run-to-run noise at temperature 0.6. Note also that P3R — the *post-fix*
revision run, which was supposed to be better — scored *lower* than P3.

**Accuracy verdict: our memory has NOT been shown to improve agent accuracy.** The honest statement is
that the controlled A/B is null-to-underpowered on accuracy.

## 2. What the same data DOES establish — and it is the more valuable half

Paired latency, same 60 questions, same agent/model/effort, P4-valslice vs P1-vanilla:

| | mean | median |
|---|---|---|
| vanilla | 256.4s | 226.8s |
| hermes | **200.1s** | **168.9s** |

- hermes faster on **48/60 questions (80%)**
- mean paired difference **−56.3s per question** (−22%)
- **95% CI −32.4s to −80.1s** (excludes zero) · t = 4.72 · **sign test p < 0.0001**

This is a large, strongly significant, paired effect measured with the only variable being our memory.

**Under LAFS this is the better half of the claim, not a consolation.** M8 established that latency is a
first-class axis and a multiplier on accuracy; M11 established that our accuracy is flat across
configurations while latency is what moves. A mechanism that buys ~22% latency at equal accuracy moves
us *left* along exactly the axis the metric rewards.

## 3. The corrected claim (use this wording; retire the old one)

> Against a vanilla-Codex control on the same 60 questions with the same agent, model, and reader, our
> memory makes the agent **~22% faster (−56.3s/question, 95% CI −32 to −80s, p<0.0001, faster on 80% of
> questions)** at **statistically indistinguishable accuracy** (McNemar p=0.77; three arms spanning
> −0 to +2 questions). The accuracy difference is within run-to-run noise and is **not** claimed.

## 4. Why this was caught, and the rule that generalises

It was caught by applying M11 §8 — *paired same-question comparison, not headline percentages* — to our
own headline. Two independent errors survived until then because a percentage difference was read as a
result: a cross-question-set comparison, and a noise-sized effect reported as a finding.

**Binding rules (add to standing discipline):**
1. **Never compare arms measured on different question sets.** State n and the manifest for both sides.
2. **Any claimed accuracy delta ships with its paired test** (McNemar for binary scoring) and its
   discordant counts. A delta smaller than the spread across your own repeated arms is not a finding.
3. **Report run-to-run spread wherever repeated arms exist** — we had three and never compared them.
4. **Audit your own headline before publishing it.** This claim was one review comment away from being
   destroyed in public, and it was ours to catch.

## 5. Consequences
- `H6-P4-FULL-REPORT.md` and PLAN "headline capability result" wording must be replaced with §3.
- The **R2 fork release** (owner-agreed, sequenced after P1) must carry the corrected claim. Releasing
  the old wording would publish an unsupported result.
- The banked numbers themselves are untouched: static 125/451, agentic 298/451, V1 444/500 all stand.
  **Only the A/B interpretation changes.**
- A vanilla@451 run (~$2–5) would make the full-scale A/B rigorous on both axes. Owner-gated; now better
  motivated, since the latency claim is the one worth hardening.
