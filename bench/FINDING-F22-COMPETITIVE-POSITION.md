# F22 — We are 3rd on LongMemEval-V1 and have been optimising the wrong benchmark

**Date:** 2026-07-26 · **Status:** major strategic correction · **Trigger:** owner asked me to review
omegamax.co/benchmarks and the LME-V2 leaderboard, at 65% confidence that we had NOT "eclipsed" the benchmark.
**Their 65% was better calibrated than my framing.**

---

## 1. The competitive picture I did not have

**LongMemEval-V1 (Wang et al., ICLR 2025) has a live, populated leaderboard with real separation:**

| system | score | reader / judge |
|---|---|---|
| OMEGA (OmegaMax / Sosa Research) | **95.4% = 466/500** | GPT-4.1 (both) |
| Mastra | 94.87% | — |
| **hermes-lcm (us)** | **88.8% = 444/500** | Codex |
| Emergence AI | 86% | — |
| Zep / Graphiti | 71.2% | — |

OMEGA's stated setup: `OMEGA v1.0.0, GPT-4.1 as generation and grading LLM, bge-small-en-v1.5 ONNX embeddings,
M1 MacBook Pro 16GB`, on "40 clean sessions" — i.e. the same `longmemeval_s` variant we ran, with a **frontier
reader**. Their own note that the corpus "structurally requires memory" is consistent with a 24-point spread
between the best and worst listed systems.

**A 24-point spread across five systems means V1 is a WORKING instrument that differentiates memory systems.**

## 2. Where my reasoning was self-serving

I established, correctly and with measurement, that **V2-small cannot exercise retrieval**: 2 distinct candidate
sets serve all 451 questions, a 200-trajectory union, exhaustible by grep (M18). That finding stands.

**I then let it generalise into "the benchmark can't show our value", which the V1 evidence contradicts.**
V1 differentiates systems clearly, permits frontier readers, and ranks us **third with a 22-question gap to
first**. "The benchmark cannot measure us" is what a losing team says; the accurate statement is narrower:
*V2-small* cannot measure retrieval. V1 measures memory systems fine, and we are behind on it.

## 3. Consequence: V1 is the competitive target, not V2

| | **LME-V1** | LME-V2 |
|---|---|---|
| leaderboard | **live, 5 systems, 71.2–95.4% spread** | **EMPTY, both tiers** |
| reader | **frontier permitted (OMEGA uses GPT-4.1)** | fixed weak Qwen3.5-9B, no frontier track |
| corpus | 1M+ tokens; "structurally requires memory" | small tier: no retrieval problem (M18) |
| our standing | **88.8%, 3rd, gap to #1 = 22 questions** | 0.0000, unranked |
| matches product reality (Sol/frontier) | **yes** | no |

**The entire program has been aimed at the benchmark with the empty leaderboard, the deprecated reader, and the
corpus that does not need memory.** V1's 22-question gap is tractable, visible, and measured on the consumer we
actually ship. That is where effort belongs.

This also re-values work already done: the V1 lane (W2A-C2, #150, "444 → ≥450") was **never** a sideline — it is
the competitive lane, and its target should be re-aimed at **≥466 to take first place**, not ≥450.

## 4. On building our own benchmark — validated, but sequence it correctly

OmegaMax already proves the move works and is honest: they built **MemoryStress** (1,000 sessions, 625 facts,
10 simulated months) and **publish their own 38.3% (115/300) on it.** A harder self-authored benchmark that
exposes what the incumbent hides, with the author's own weak score published, is credible rather than
self-serving.

**But sequencing is the whole risk.** Publishing our own benchmark while ranked third on the existing one reads
as avoidance, however good the science. Order:
1. **close the V1 gap** (444 → 466+), on a frontier reader, competing where the field is watching;
2. **then publish the scaling benchmark** from credibility — the longitudinal axis nobody has published well.

Phase 1A (#159) already builds toward (2): a 389x corpus-scaling probe on V1 data is structurally what
MemoryStress does, and it is the axis with no strong published result.

## 5. Actions
- **Re-aim the V1 lane at ≥466** (first place), not ≥450. Update #150.
- **Keep Phase 1A running** — it is the seed of our own benchmark and costs nothing.
- **Demote V2-medium (#161)** below V1 work: an empty leaderboard with a weak fixed reader is worth less than a
  visible third place we can convert to first.
- **Audit the gap**: what do OMEGA/Mastra do that we do not? Their disclosed stack is unremarkable
  (bge-small-en-v1.5, GPT-4.1, a laptop) — so a 22-question gap is unlikely to be infrastructure. #154's
  competitor-technique survey is now high-priority rather than a curiosity.

## 6. Process note
This correction came from the owner asking me to *read the competitive landscape* rather than reason about it.
I had a benchmark-portfolio decision record (§6a) that never enumerated where competitors actually stood on V1.
**Same failure family as M17/M18/M13: I asserted a property of the landscape instead of enumerating it.** The
fix is identical — go and look.
