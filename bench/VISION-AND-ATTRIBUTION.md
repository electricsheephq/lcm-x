# VISION & ATTRIBUTION — what the frontier model does, what WE do, and where our value is measurable

**Date:** 2026-07-25 · **Status:** program vision (architect, under owner directive)
**Owner directive:** *"Our users care about the frontier... understand what percentage of performance comes
from the frontier model itself and what you're actually improving. That gives you a baseline to build on."*

---

## 1. The attribution, measured and honest

| capability | who actually delivers it | our marginal contribution |
|---|---|---|
| corpus → pack compression (~7.6M tok → 17k, ~444x) | **the coding agent, either one** | **none** — vanilla Codex is handed a `trajectories/` dir and greps it to the same ~20k pack |
| accuracy | agent + fixed reader | **none measurable** — McNemar null across 3 paired arms (p=0.727/1.000/0.774) |
| context tokens delivered | either | **none meaningful** — +3.2%, 95% CI spans zero (sign test p=0.027 on direction only) |
| **retrieval latency** | **our indexed store vs its file scan** | **−56.3s/question (−22%), 48/60 questions, p<0.0001** |

**One real, measured advantage: speed.** Everything else attributed to us so far was either the agent's work
or noise. This is the baseline the owner asked for, and it is deliberately unflattering.

### 1b. ★ The V1 attribution is now DECOMPOSED, not just unflattering (F27 + F28, 2026-07-25)

The owner's question was *"what percentage of performance comes from the frontier model itself and what are you
actually improving."* For LongMemEval-V1 there is now a measured answer, from the banked 444/500's own artifact at
zero spend.

**Our layer's job is to put the right evidence in front of the reader. Measured, it does that:**

| stage we own | measured | verdict |
|---|---|---|
| at least one gold session retrieved | **500/500 = 100.0%** | saturated |
| every gold session retrieved | **486/500 = 97.2%** | saturated |
| gold evidence *represented* in the 25-hit budget | failures get **more** gold hits than passes (11.5 vs 11.2) | adequate — no crowding-out |
| evidence *structured* for the reader | renderer already groups by session, dates every item | adequate — nothing to add |

**And the failures are not ours:** 52 of the 56 failures had the **complete** gold evidence in the prompt. On
multi-session — our worst category and the largest single loss — **25 of its 26 failures had every gold session
retrieved.**

So the 11.2-point V1 gap decomposes as:

- **≤4 questions (0.8 pts): retrieval.** Ours. Partial gold coverage.
- **~14 questions (2.8 pts): abstention calibration.** Contested — the reader abstains *correctly* on 20
  genuinely-unanswerable questions and incorrectly on 14 where evidence was present. A memory-layer *signal*
  might move this; the M7/M7b lane tried and got a NO-GO.
- **~38 questions (7.6 pts): the reader's synthesis.** **Not ours.** Complete, well-represented, grouped, dated
  evidence in; wrong answer out.

> **⛔ WITHDRAWN 2026-07-29 (F29):** the "~90% belongs to the model" attribution and the bound below rested on
> session-level recall read as evidence completeness. Answer-turn measurement: **~30% of the gap (≈3.4 pts) is
> retrieval-addressable**; and F22's leaderboard (OMEGA 466 / Mastra ~474 on GPT-4.1) directly contradicts the
> "no memory system can win those questions" claim. The corrected attribution: our layer is saturated at
> session granularity, ~86% at evidence granularity, and the difference is ours to win.

**Why this is the right answer to give, not a disappointing one.** It converts "we can't beat 444" from a failure
into a *bound*: no memory system can win those 38 questions by retrieving or presenting better, so any competitor
claiming a large V1 margin over us is either using a stronger reader, running extra reader passes, or decomposing
questions in the agent — all of which are agent-layer work, not memory-layer work. That is a claim we can state
publicly and defend with the decomposition above, and it reframes the competitive question from "who retrieves
better" (settled, both saturated) to "who spends more reader compute" (an honest axis, and one where LAFS already
prices the trade-off).

**What it does NOT license.** It says nothing about V2's trajectory subsystem, and nothing about the scaling
regime — where the corpus stops being grep-exhaustible and retrieval becomes load-bearing again. Both remain open,
and Phase 1A (#159) is the test. **V1-small is where our ceiling is measured; it is not where our thesis lives.**

### 1c. ★ R2 REVISED THE ATTRIBUTION — two rows changed sign (2026-07-30, post-R2 addendum)

The 07-25 table above was deliberately unflattering and is now measurably out of date in two places:

| capability | 07-25 verdict | R2-era measurement | what changed |
|---|---|---|---|
| accuracy | "none measurable" (McNemar null ×3) | **V1 455/500, +11 vs banked 444 (b20/c9, p=0.061, reported as measured; direction consistent across all three baselines, placebo flat)** | The completeness lane worked: waking the provider's internal retrieval (#168) + citable delivery (gate-4 rebuild) produced genuine flips. F33 then showed the mechanism is interventionally supported — injecting missing gold answer turns flips 23/35 (p=1.9e-5; salience confound documented, so public claims use the observational gradient. |
| retrieval latency | −56.3s/q at small tier | that claim stands, **plus F44 at 389×: 263.6ms @ 19,829 sessions = 24× vs the pre-fast-scan baseline, recall parity net 0** | The latency advantage now has a published scale curve behind it, not just the small-tier A/B. |

Unchanged and still honest: compression is the agent's, not ours; token delivery is noise; V1-small is
where our ceiling is measured, not where the thesis lives (F38: realistic remaining V1-small headroom
≈1.6 pts). The thesis now runs through the scale regime (the 389× instrument family), the metric-standard
play (answer-turn evidence-completeness), and — per the owner's 07-30 decision — the LoCoMo sidecar and
the V1-medium tier as the official surfaces. The measurement discipline that produced §1's unflattering
table is the same one that later moved two of its rows: that is the point of the discipline.

## 2. Why the advantage is so thin HERE — and it is structural, not a failure

**The benchmark corpus is 1,870 trajectories total, and each question is handed 100 pre-selected candidates.**

That is the whole explanation. At 100 candidate files, brute-force exploration is *competitive* — `grep` over
100 files is fast, so an index has almost nothing to beat. The benchmark **pre-solves the candidate-selection
problem**, which is precisely the problem a memory system exists to solve, and then measures only what
remains: reading a small pre-filtered set.

**So LME-V2 structurally cannot showcase our architecture.** Not because the benchmark is bad at what it
measures, but because what it measures is downstream of our value. Vanilla Codex scoring 63.3% without us is
not a threat to the product — **it is evidence that this task does not need a memory system**, and we should
say so plainly rather than pretend a 2-point wobble is a moat.

## 3. What that implies — the scaling hypothesis (our actual thesis, now testable)

File scanning is **O(n)** in corpus size. Indexed retrieval is sub-linear. So:

> **The latency advantage we measure should GROW with corpus size.** 22% at 100 candidates is the floor, not
> the ceiling — it is our advantage leaking through at a scale engineered to suppress it.

That is a falsifiable, product-relevant claim, and it is the owner's lossless-raw + read-time-intelligence
thesis stated as an experiment: *keep everything, and still answer fast.* The value proposition is not
"+2 accuracy points"; it is **"your memory can grow without your queries getting slower."**

**This is what we should be building and publishing.** It also explains why the corpus-coverage ceiling (M5:
1/451 gold absent) was reassuring but not decisive — storage completeness is necessary and we have it; the
differentiator is retrieval cost at scale.

## 4. The dual-objective consequence (with M16 §2b)

| objective | consumer | can it show our value? | action |
|---|---|---|---|
| LME-V2 leaderboard | fixed Qwen3.5-9B, both tiers (`EXPECTED_READER_MODEL_SUBSTRING = "qwen3.5-9b"`) — **no frontier track exists** | Weakly — pre-filtered small corpus | compete for the LAFS latency axis; do not expect a moat |
| **Product / Sol (frontier)** | what customers actually run | **Yes — but only if the corpus is realistic** | **our own benchmark, scaled corpus, published on our repo** |
| LME-V1 | more permissive | Yes (444/500 banked) | publishable now |

Frontier results cannot be submitted to LME-V2 (weak-reader-only by construction) but **can** be published on
our own repo — which is what users read anyway.

## 5. What to build next (priority, revised by this analysis)

1. **SCALING BENCHMARK (new, top priority).** Same questions, but sweep the candidate corpus: 100 → 1,000 →
   10,000+ trajectories per query, no oracle pre-filter. Measure accuracy, latency, and tokens for
   (a) our indexed store, (b) vanilla agent + file scan, (c) naive RAG. **The hypothesis is that (b)
   degrades linearly and we do not.** This is the experiment that either proves the product or kills the
   thesis — and it is the only one that can, because LME-V2 cannot.
2. **Frontier-consumer measurement** (H6-P5, #156) — read the same packs with a frontier reader. Establishes
   the product-truth number and how much of the score is the model vs us.
3. **M7b** (running) — the leaderboard path. Keep it, but it is no longer the strategic centre.
4. **Full-scale confirmation of the latency claim** (vanilla@451) — hardens our one real result.

## 6. The discipline this vision inherits
Today's audit cycle retired four claims we believed (M12–M15) and its root cause was one underdesigned
instrument (M16). The compression number in §1 was *one step from becoming a fifth* — it was the owner who
asked "didn't vanilla score well?", which is exactly the question that caught it. **Attribution claims get
the same paired, adversarial treatment as accuracy claims: name who does the work, and measure the
counterfactual where we are absent.**
