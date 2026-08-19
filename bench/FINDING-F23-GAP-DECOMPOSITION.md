# F23 — The 22-question gap to OMEGA decomposes exactly, and it is concentrated

**Date:** 2026-07-26 · **Basis:** per-question verdicts from the banked 444/500 run
(`v1l1-full500-primary-2026-07-23T18-26-13-780Z`, `phases.evaluate.score`) vs OMEGA's published category
breakdown on omegamax.co/benchmarks. Both on LongMemEval-V1 `longmemeval_s`, both with frontier readers
(ours `gpt-5.6-sol`, theirs GPT-4.1).

---

## 1. The decomposition — it sums exactly to the gap

| category | us (#423 code) | OMEGA | gap |
|---|---|---|---|
| single-session recall (user + assistant) | 118/126 = **93.7%** | 125/126 = 99% | **−7** |
| **preference application** | 25/30 = **83.3%** | 30/30 = **100%** | **−5** |
| multi-session reasoning | 107/133 = **80.5%** | 111/133 = 83% | **−4** |
| knowledge updates | 71/78 = **91.0%** | 75/78 = 96% | **−4** |
| temporal reasoning | 123/133 = **92.5%** | 125/133 = 94% | **−2** |
| **total** | **444/500 = 88.8%** | **466/500 = 95.4%** | **−22** |

**7 + 5 + 4 + 4 + 2 = 22 = 466 − 444.** The gap is fully accounted for with no residual.

## 2. Our own loss profile (where to look first)

| question type | correct | wrong | loss rate |
|---|---|---|---|
| multi-session | 107 | **26** | 20% |
| temporal-reasoning | 123 | 10 | 8% |
| knowledge-update | 71 | 7 | 9% |
| single-session-assistant | 51 | 5 | 9% |
| **single-session-preference** | 25 | **5** | **17%** |
| single-session-user | 67 | 3 | 4% |

## 3. What the shape says

**(a) Preference application is the sharpest signal.** OMEGA scores **30/30**; we score 25/30 — a 16.7-point
deficit on the smallest category in the benchmark. A perfect score on 30 questions is not luck: it implies a
*specific mechanism* for persisting and applying stated user preferences. We lose 1 in 6. **Small category,
large rate gap, well-defined semantics — this is the most likely single identifiable missing feature in the
product.**

**(b) Single-session recall is the largest absolute gap (−7)** despite a low 4–9% loss rate, simply because the
category is large (126). OMEGA at 125/126 is effectively solved. Ours is 118/126. These should be the *easiest*
questions — the evidence is in one session — so 8 losses suggests a delivery or answer-layer issue rather than
retrieval.

**(c) Multi-session is where we lose most in absolute count (26 wrong) but the gap to OMEGA is only −4** —
because it is *their* weakest category too (83%). This is the benchmark's genuinely hard class. Closing it is
research; closing (a) and (b) is likely engineering.

**Priority implied: preference application (−5) and single-session recall (−7) = 12 of the 22 questions, in the
two categories where the competitor is at or near perfect.** That is over half the gap in the two places where
"they solved something we did not" is most likely to be true and findable.

## 4. Why this is credible as a comparison
Both sides ran the same benchmark, same `longmemeval_s` variant ("40 clean sessions"), and both used a
**frontier reader** — so this is not a weak-reader artifact. OMEGA's stack is modest
(`bge-small-en-v1.5` ONNX, GPT-4.1, a 16GB M1 laptop, 12 tools), so the gap is method, not compute.

## 5. Immediate use
Feeds **#154** (competitor-technique survey) with a precise target instead of a general question: *what does
OMEGA do for preference application and single-session recall that we do not?* Also gives the V1 lane (#150,
re-aimed at ≥466) a concrete work-breakdown rather than a bare number.
