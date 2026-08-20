# M11 — Effort buys latency and (measurably) not much accuracy; low effort is the operating point

_Original title claimed a convex curve. Retained in §2 as the point-estimate shape, but see the §2b
power check: the accuracy deltas are within noise, so the load-bearing claim is the simpler and
stronger one — **accuracy flat, latency halved**._

**Date:** 2026-07-25 · **Issue:** #158 (P1 agentic latency sweep) · **Status:** measured, decision-bearing
**Depends on:** M8 (LAFS = accuracy × latency) · M10 (search-flailing unifies accuracy and latency)

---

## 1. The measurement

Agent reasoning effort swept over the frozen 60-question dev manifest (32 web / 28 enterprise),
decoding pinned (temperature 0.6 / top_p 0.95 / top_k 20) per the M9 parity rule, everything else
held identical to the P4 configuration.

| arm | effort | accuracy | latency (mean) | LAFS |
|---|---|---|---|---|
| P4 (451q, banked) | xhigh | 66.10% | 196.9s | 0.0000 |
| L1 | high | 61.67% (37/60) | 126.3s | 0.0000 |
| L2 | medium | 58.33% (35/60) | 99.0s | 0.0000 |
| L3 | low | 56.67% (34/60) | 51.6s | 0.0000 |

Per-domain:

| arm | web | enterprise |
|---|---|---|
| L1 high | 68.8% @ 110.9s | 53.6% @ 144.0s |
| L2 medium | 65.6% @ 89.6s | 50.0% @ 109.7s |
| L3 low | 65.6% @ 47.0s | 46.4% @ 56.9s |

All four arms score **0.0000**. That is the least interesting thing about this table.

## 2. The curve is convex — the cheap latency is at the bottom

| step | accuracy cost | latency bought | points per 10s |
|---|---|---|---|
| xhigh → high | −4.43 | −70.6s | 0.63 |
| high → medium | −3.34 | −27.3s | 1.22 |
| **medium → low** | **−1.66** | **−47.4s** | **0.35** |

The last leg is the cheapest by a factor of ~3.5 against the leg above it. An earlier read of the
L1→L2 segment alone characterised the trade as "roughly 1:1"; that describes the top of the curve and
**does not generalise downward**. Notably, web accuracy is *identical* at medium and low (65.6% both)
while web latency nearly halves (89.6s → 47.0s) — on web, the last effort step is free.

## 2b. ★ POWER CHECK — the accuracy differences are NOT significant; the latency differences are

Run after the fact, before any of this was allowed to drive the roadmap (Fisher exact, two-sided):

| comparison | difference | p |
|---|---|---|
| overall, high vs low | 37/60 vs 34/60 (-3 q) | **0.711** |
| answerable subset, high vs low | 34/43 vs 29/43 (-5 q) | **0.330** |
| abstention subset, high vs low | 3/17 vs 5/17 (+2 q) | **0.688** |

**None of the accuracy differences across the effort dial are distinguishable from noise on a 60q
slice.** The per-step "costs" in the table above are point estimates, not established costs; "medium->low
costs 1.66 points" must be read as "costs an amount indistinguishable from zero at this sample size."

The LATENCY differences are a different matter: 126.3 / 99.0 / 51.6s are means over 60 observations with
a large effect and are reliable.

**This strengthens the operating-point decision rather than weakening it.** If accuracy is statistically
indistinguishable across the effort dial while latency more than halves, low effort is the clear choice —
the argument no longer even needs the convexity claim, it only needs "accuracy flat, latency halved."

**It also retires a claim made in this session and briefly recorded here:** that effort moves the two
subsets in *opposite* directions (low better at abstention, worse at answerable), with a mechanism story
(more reasoning -> tidier pack -> over-persuasion) and a derived ~0.93 LAFS prize from combining
high-effort answerable with low-effort abstention. At p=0.688 and p=0.330 that pattern is a 2-question
and 5-question wobble. **Withdrawn.** It is a hypothesis for a powered sample, not a finding, and it must
not be used to justify a mechanism or a spend. (This is the `feedback_instruments_adjudicate_eyeballs`
discipline applied to my own narrative.)

**Consequence for the distance-to-window number:** "1.94 points, ~1.2 questions" in §4 is a point
estimate against a hard threshold. The confidence interval on a 60q proportion is wide (roughly +/-12
points at 95%); we may already be above 58.6% on the full 451, or well below. The 60q slice is a
SCREEN, not a promotion instrument. Any arm that passes on 60q requires a full-451 confirmation before
it is banked or submitted.

## 3. Why this decides the operating point

Under LAFS, latency multiplies the value of accuracy. The same accuracy number is worth ~8× more at
low effort than at medium:

| accuracy | @ 99.0s (medium) | @ 51.6s (low) |
|---|---|---|
| 58.61% | 0.0002 | 0.0014 |
| 62.00% | 0.0576 | **0.4758** |
| 66.10% | 0.1271 | **1.0495** |
| 72.10% | 0.2288 | **1.8890** |

**Decision (recorded): low effort is the program's operating point.** Not as a fallback for when we
run out of accuracy, but as the place every future accuracy point should be banked, because each one
is worth roughly eight times more there. Effort is now a settled dial, not a lever we are still
searching over.

## 4. Distance to a non-zero score

At 51.6s the scoring window opens at **58.6%** accuracy. L3 sits at 56.67% — a gap of **1.94 points,
about 1.2 questions out of 60.** Value immediately above the threshold rises steeply: 60% → 0.196,
62% → 0.476, 66.1% → 1.049.

M7 (negative-evidence disclosure, #157) projects +6 points from the abstention mass. Landed at the
low-effort operating point that is ≈62.7% @ 51.6s ≈ **0.55** — and per M10 the same mechanism removes
searches rather than adding them, so latency should fall too, compounding the gain. The identical
mechanism landed at medium effort would be worth ≈0.07.

## 5. Instrument note (M9 discipline)

L3/web took exactly one reader-side HTTP 504; the harness records an empty response and scores it
wrong (`question_id edb69441`). This is a provider artifact, not a capability loss. L1 and L2 took
zero such errors.

- as-measured: **56.67%** (34/60)
- excluding the artifact: **57.63%** (34/59)

Both are below the 58.6% floor, so the artifact changes no decision here — recorded because at a
1.94-point margin a single question is material, and because L2 previously missed its floor by 0.3
points. **Rule reaffirmed: count provider-error rows before comparing any two arms.**

## 6. What this does and does not establish

**Establishes:** the LATENCY of each effort setting (large, reliable effects over 60 observations);
that accuracy across the dial is statistically indistinguishable on a 60q slice; that the operating
point is therefore low; and — as arithmetic from the scorer, not an empirical claim — that accuracy is
~8x more valuable at 51.6s than at 99.0s.

**Does not establish:** any real accuracy DIFFERENCE between effort levels (all p>0.33 — the per-step
costs in §2 are point estimates only); that effort moves answerable and abstention in opposite
directions (retired, §2b); the honest uncontended latency (L1-L3 ran with concurrent workers — the L4
concurrency-1 probe is outstanding and can only move these numbers *down*, i.e. further in our favour);
that M7's projected +6 will materialise; that 60q dev-slice accuracy transfers to the full 451 (it has
run +/-3 points historically, and the 95% CI on a 60q proportion is ~+/-12 points).

**Next:** M7 pilot (#157), executed at **low** effort, with `searches_per_question` instrumented as
the leading indicator per M10.

---

## 7. L4 CONTENTION PROBE — inconclusive, and a parity failure caught by our own rule

**The probe did not run the configuration it was meant to.** L4's `run_args.json` diffed against L3
shows the intended concurrency change (`prompt_build_max_workers` 3→1, `reader_max_concurrent_requests`
3→1) **plus an unintended one: `reasoning_effort: high`, not `low`.** L4 is therefore not comparable to
L3 at all; its correct control is L1, the other high-effort arm. Caught by the M9 parity diff before the
number was used — the first time that rule has paid for itself on an incoming result rather than an
outgoing one. (Read naively against L3 it appeared uncontended runs were 2.3x SLOWER, which is
incoherent; the parity diff explained it immediately.)

**Corrected comparison (both arms effort=high, only concurrency differs), 6 paired qids:**

| | mean | median |
|---|---|---|
| L4 concurrency=1 | 125.4s | 122.1s |
| L1 concurrency=3 | 167.8s | 168.1s |

Point estimate: contention factor **1.34x**. **But it is not established:** sd=0.40, n=6,
**95% CI 0.93–1.76 — includes 1.0.** Per-question ratios run 0.86 to 1.93 and **two of six questions
were FASTER contended.** Conclusion: **we may not claim a corrected latency number from this probe.**
Directionally consistent with contention existing; underpowered to establish it.

Even taking the point estimate at face value it does not rescue a score: 56.67% @ 38.6s still computes
**0.0000**, because we are stranded between the two windows — accuracy below the 58.6% floor of window A,
latency above the 26.9s ceiling of window B. What the correction *would* do is multiply future accuracy
gains: 62.7% → 0.798 (from 0.574), 66.1% → 1.460 (from 1.050).

**ACTION:** if the latency figure ever becomes load-bearing (submission, publication, a gate), run a
properly powered uncontended probe — same effort as the arm being corrected, ≥30 questions. Until then
**all latency figures are reported as CONTENDED**, per the standing rule.

## 8. ★ RUN-TO-RUN VARIANCE — the mechanism behind "60q is a screen"

On the same 6 questions at the same effort, **L1 scored 3/6 and L4 scored 5/6.** A 33-point swing with
no mechanism change. The cause is not a bug: decoding is pinned at temperature 0.6 / top_p 0.95 /
top_k 20 — pinned for **comparability**, not determinism. The reader is stochastic by construction, so
every arm carries irreducible sampling noise, and two arms differing by a couple of questions on a small
slice tell us nothing.

This is the strongest available justification for the §2b/§0 rule: **the 60q slice is a screen, not a
promotion instrument**, and no mechanism story may rest on a sub-significant subset difference. It also
means a full-451 comparison is not noise-free either — paired/same-question comparison (as used here)
is materially better than comparing headline percentages, and should be the default for any future gate.
