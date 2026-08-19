# M17 — The MEDIUM track exists, is 5x scale, has LOWER bars, and is where our thesis is officially testable

**Date:** 2026-07-25 · **Status:** major strategic finding · **Credit:** owner question ("there are larger
longmemeval data sets... we used small I think?") — I had asserted the corpus was uniformly small.

---

## 1. What I got wrong

I described the LME-V2 corpus as small (1,870 trajectories, 100 candidates per question) and concluded the
benchmark structurally cannot exercise our architecture. **The 100-candidate figure is the `small` tier only.**
The dataset ships two haystacks:

| tier | file | candidates per question | questions |
|---|---|---|---|
| small | `haystacks/lme_v2_small.json` (803K) | **100** (min/median/max all 100) | 451 |
| **medium** | `haystacks/lme_v2_medium.json` (3.9M) | **387 / 500 / 500** (min/median/max) | 451 |

**We have only ever run `small`.** A real, published, 5x scale point was available the whole time.

**Second correction, same class:** I implied the LME-V2 *dataset* is not useful for Sol/frontier work. Wrong.
The questions, gold answers, and haystacks are reader-agnostic and perfectly usable with any consumer. Only
the **leaderboard submission** is locked to `qwen3.5-9b`. "Cannot submit frontier numbers" ≠ "data is not
useful for frontier evaluation."

## 2. Everyone degrades at 5x — and the retrieval-heavy systems degrade most

| system | small | medium | Δ accuracy |
|---|---|---|---|
| RAG: query → slice + notes | 51.0 @ 0.2s | 45.9 @ 0.3s | **−5.1** |
| AgentRunbook-C | 74.9 @ 108.3s | 70.1 @ 139.9s | **−4.8** |
| AgentRunbook-R | 58.6 @ 26.9s | 57.0 @ 25.8s | −1.6 |
| Codex | 69.9 @ 177.2s | 68.7 @ 185.8s | −1.2 |

**This is the scaling hypothesis visible in the benchmark's own published numbers.** The systems that lean on
retrieval quality (RAG, AgentRunbook-C) lose the most when the candidate pool grows 5x; AgentRunbook-C also
gets *slower* (108.3s → 139.9s). That is precisely the degradation pattern our indexed architecture is
supposed to resist.

## 3. Consequence: the bars are LOWER, so results that score 0 on small score on medium

Windows — small: beat 51.0 @<0.2s · 58.6 @<26.9s · 74.9 @<108.3s · 69.9 @<177.2s.
Medium: beat **45.9** @<0.3s · **57.0** @<25.8s · **70.1** @<139.9s · **68.7** @<185.8s.

Computed with the repo scorer, using accuracies **as measured on small** (hypothetical — see §4):

| our arm | small LAFS | **medium LAFS** |
|---|---|---|
| low effort 56.67% @ 51.6s | 0.0000 | 0.0000 |
| **medium effort 58.33% @ 99.0s** | 0.0000 | **0.0868** |
| **high effort 61.67% @ 126.3s** | 0.0000 | **0.0901** |
| xhigh 66.10% @ 196.9s | 0.0000 | 0.0000 |

Two arms we already own would score **non-zero** on the medium track. Note the operating point flips: at
medium the winning region sits between 25.8s and 139.9s, which favours the **medium/high** effort arms, not
the low-effort one M11 selected for the small track. **M11's "low effort is the operating point" is a
SMALL-TRACK conclusion and does not transfer.**

## 4. The honest caveat — this is a hypothesis, not a result

Those accuracies were measured against **100 candidates**. Submitting them as medium-track numbers would be
invalid, and our accuracy will very likely drop at 500 candidates too — that is the whole question. What §3
establishes is that **the medium bar is low enough that a modest degradation still scores**, which makes the
track worth measuring rather than assumed.

## 5. Revised plan

1. **Run the medium tier** (`--tier medium`, 500 candidates) at medium and high effort. This simultaneously
   (a) tests the scaling thesis on real published data, (b) targets an official track with lower bars, and
   (c) gives a genuine 5x scale point without synthesis. **Highest-value run available.**
2. **The scaling benchmark keeps its scope-widening ladder** for scales beyond 500 (S3=1,870), but S1/S2 are
   now *replaced by the real medium tier* wherever possible — official data beats synthesized scope.
3. **Re-derive the operating point per tier.** Effort is a per-track decision, not a program constant.
4. **Check whether LME-V1 also has larger tiers** (owner believes so; we ran small there too at 444/500).

## 5b. CONFIRMED — LME-V1 has the same story, and a BIGGER jump

Checked at the owner's prompting. Locally we hold only `longmemeval-data/longmemeval_s` (265MB) — **the small
variant**. The canonical LongMemEval-V1 release also ships **`longmemeval_m`**, roughly **10-13x larger per
question** (~1.5M tokens of haystack vs ~115k), plus `longmemeval_oracle` (evidence-only).

**Our banked 444/500 on V1 is the SMALL variant.** `longmemeval_m` is the variant where retrieval actually
matters — a larger scale jump than V2's 5x, and squarely our thesis territory. Acquiring it is a download
(HuggingFace), not a spend; LEXAR has room.

**Combined ladder now available on REAL published data, no synthesis at all:**

| dataset | variant we ran | larger variant available | scale factor |
|---|---|---|---|
| LME-V2 | small (100 cand) | **medium (500 cand)** | ~5x |
| LME-V1 | small (~115k tok) | **`longmemeval_m` (~1.5M tok)** | **~13x** |

That is a two-benchmark, two-scale grid on official data — strictly better evidence than the scope-widening
ladder I specced, and it carries published competitor numbers to compare against.

## 6. Process lesson (third of its kind today)
M12 (cross-question-set comparison), M15 §4c (wrong repo in a dispatch packet), and now M17 (an entire larger
tier unexamined) share one root: **asserting a property of the data instead of enumerating it.** The fix is
mechanical — before any strategic claim about a dataset, list its files and print its shape. All three were
caught by someone else asking a plain question.
