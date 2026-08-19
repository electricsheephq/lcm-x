# SPEC — THE SCALING BENCHMARK (our own; the experiment LME-V2 cannot run)

**Date:** 2026-07-25 · **Status:** DRAFT for immediate build · **Priority:** top (supersedes M7b as strategic centre)
**Basis:** `bench/VISION-AND-ATTRIBUTION.md` · **Owner directive:** frontier-first; latency / quality / tokens

---

## 1. The claim under test

> **File scanning is O(n) in corpus size; indexed retrieval is sub-linear. Therefore our latency advantage
> GROWS with corpus size.** The measured −22% at 100 candidates is a floor produced at a scale engineered
> to suppress it.

Falsifiable and symmetric: **if our latency degrades at the same rate as file-scan, the thesis is wrong and
we publish that.** That is the point of running it.

**Why LME-V2 cannot answer this:** its corpus is 1,870 trajectories and it *hands each question 100
pre-selected candidates*. It pre-solves candidate selection — the problem a memory system exists for — so an
index has almost nothing to beat, which is exactly why vanilla Codex scores 63.3% without us.

## 2. Design — the key move is to REMOVE the oracle pre-filter

For each question, candidate set = **its gold trajectories + N distractors**, N swept upward. Gold must
remain present at every scale, or accuracy collapses for reasons unrelated to retrieval and the experiment
measures nothing.

**Scale ladder (fully honest, zero synthesis first):**

| point | candidates/query | source |
|---|---|---|
| S0 | 100 | the benchmark's own haystack (reproduces today's numbers — sanity anchor) |
| S1 | 500 | real trajectories sampled from the 1,870 corpus |
| S2 | 1,000 | same |
| S3 | **1,870 (whole corpus)** | same — **18.7x sweep with NO synthetic material** |
| S4 | 5,000+ | *only if needed*: real text from the LME-V1 corpus as additional distractors — **labelled as cross-corpus**, never fabricated |

S0→S3 alone is an 18.7x sweep on entirely real material. If the hypothesis holds, divergence should already
be visible there; S4 exists to extend the curve, not to rescue it.

**Arms (three, per scale point):**
- **A — hermes indexed store** (our product)
- **B — vanilla agent + file scan** (the *actual* competitor identified by the attribution analysis)
- **C — naive RAG**, embedding top-k (commodity baseline)

## 3. Two phases, so the expensive part runs only where it is needed

**Phase 1 — RETRIEVAL-ONLY latency/token sweep (cheap: NO reader, NO judge).**
The hypothesis is about *retrieval*, so measure the memory stage alone. No reader calls, no judge calls —
this removes almost all cost and isolates exactly the variable under test. Full ladder × 3 arms.
Metrics per (arm, scale): retrieval latency (mean/p50/p95), tokens delivered, and **recall of gold
trajectories in the returned set** (the quality proxy that needs no LLM).

**PRIMARY OUTPUT: the SLOPE of latency vs corpus size per arm.** Report the fitted exponent — is arm B
linear (~1.0) and arm A sub-linear (<0.5)? That single number is the thesis.

**Phase 2 — accuracy confirmation at the ENDPOINTS ONLY (S0 and the largest point).**
Confirms retrieval wins convert to answer quality and that nothing degraded. Run with **both consumers**
per M16 §2b: the official fixed 9B reader *and* a frontier reader (product truth). Endpoints only, because
the middle of the ladder adds cost without changing the conclusion.

## 4. Predeclared expectations (write these down BEFORE running — M11–M15 discipline)
- Arm B latency scales ~**linearly** with N (grep reads every file).
- Arm A latency scales ~**flat-to-log** (indexed lookup).
- Arm C latency scales ~linearly in embedding cost but sub-linearly in query.
- **Gold recall:** arm B ~1.0 by construction at any N (it can see everything, given time); arms A and C
  must be *measured* — **if our recall drops as N grows, that is a real weakness and must be reported as
  prominently as any latency win.** A fast retriever that misses the needle is worse than a slow one.
- Accuracy at S0 must reproduce today's numbers within run-to-run noise, else the harness is broken.

## 5. What each outcome means
| result | conclusion |
|---|---|
| A flat, B linear, A recall holds | **thesis proven** — publish; this is the product |
| A flat, B linear, **A recall degrades** | we bought speed with misses — fix recall before any claim |
| A and B both linear | **thesis wrong**; our store is not architecturally better. Publish that and re-plan |
| A wins but only past S4-synthetic | weak/inconclusive — say so, do not lean on synthetic scale |

## 6. Publication
Frontier results cannot go to the LME-V2 leaderboard (no frontier track: `qwen3.5-9b` required on both
tiers). This publishes on **our own repo** as our own benchmark, with the harness, manifests, and raw
outputs — the reproducibility standard we already applied to R1. Include the negative attribution honestly:
the agent does the compression, and on a pre-filtered 100-candidate corpus vanilla is our equal.

## 7. Discipline inherited
Predeclared bars · paired comparison per scale point · provider-error counts · run-to-run spread ·
slice composition reported · no claim without its counterfactual (VISION §6).
