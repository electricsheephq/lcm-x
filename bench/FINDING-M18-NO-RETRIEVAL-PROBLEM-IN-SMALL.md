# M18 — The `small` tier's candidate pool is SHARED and TINY, so exhaustive scan is cheap

_Original title claimed "NO retrieval problem". **That was an overstatement — see §0.** The measurement
(2 distinct candidate sets for 451 questions) is correct; the interpretation was too strong._

## 0. ★ SELF-CORRECTION (same day, before this drove a spend)

I wrote that the small tier "contains no retrieval task — not a small one, none." **Too strong.** Each
individual question still requires locating 1–2 gold trajectories among 100 candidates — a needle density of
~0.01–0.02, which is a real per-question retrieval task.

What is actually true, and it is still decisive:

| | V2 small | V2 medium (enterprise) | **V1 `longmemeval_s`** |
|---|---|---|---|
| candidate pool per question | 100 | 500 | ~50 (39–66) |
| **pool SHARED across questions?** | **yes — 2 sets for 451 q** | no — 211 unique of 211 | **no — 500 unique of 500** |
| union / total corpus | **200** | 874 | **19,829** |
| needle density | ~0.01–0.02 | ~0.002–0.004 | 2/50 = **0.04** |
| mean overlap between questions | **1.00** | 0.48 | **0.001** |

**The precise defect in V2 small is not the absence of a retrieval decision — it is that the pool is SHARED
and TINY (200 trajectories total).** A 100-file exhaustive scan is cheap, so an index has nothing to beat.
That is exactly what I said in VISION §2 originally; M18's stronger framing was a regression, not an
improvement, and I am reverting to the accurate version.

**The attribution conclusion is unchanged:** vanilla Codex greps 100 fixed files and that is competitive
because the pool is small, not because retrieval is absent. Our indexed architecture is still untested at
scale — for a reason I can state correctly.

## 1. The measurement (unchanged, correct)

**Date:** 2026-07-25 · **Status:** decisive — explains the entire attribution result
**Supersedes the explanation in:** VISION-AND-ATTRIBUTION §2 (right conclusion, understated cause)

---

## 1. The measurement

| tier | questions | union of candidates | candidates/q | **DISTINCT candidate sets** | selectivity |
|---|---|---|---|---|---|
| **small** | 451 | **200** | 100 | **2** | 100/100 per domain = **1.00** |
| **medium** | 451 | **1,473** | 387–500 | **433** | 500/1473 = **0.34** |

**In the `small` tier every web question receives the identical 100 trajectories, and every enterprise
question receives the identical 100.** Two candidate sets serve all 451 questions. Mean pairwise overlap in
medium is 0.48 — genuinely different pools per question.

## 2. What this settles

**There is no retrieval task in the `small` tier.** Not a small one — none. The candidate set is fixed,
identical across questions, and equals the entire domain pool. Nothing has to be *found*.

This is the complete explanation for the attribution result (vanilla Codex 63.3% vs ours, McNemar null):
**vanilla Codex is handed a folder of 100 fixed files and greps it, and on this tier that IS the whole task.**
Earlier I framed this as "the corpus is small so grep is competitive." Too weak. The correct statement is
that the tier we have spent the entire program on **does not exercise retrieval at all**, so a memory
system's core function is unmeasured by construction.

It also retires any residual worry that our architecture underperformed: **it was never tested at a scale
where an index matters.** (Not "never tested at all" — the per-question needle-finding did happen.)

## 3. `medium` is different in KIND, not degree
433 distinct candidate sets, 34% selectivity, 0.48 mean overlap. **Medium is the first tier where a
per-question retrieval decision exists.** That is why the retrieval-heavy published systems degrade there
(M17: RAG −5.1, AgentRunbook-C −4.8 and 31s slower) while the agentic ones barely move — the ones that
depend on retrieval quality start actually needing it.

Combined with M17, the strategic picture is settled: **every number this program has banked comes from a tier
with no retrieval problem.** 125/451, 298/451, and the V1 444/500 are all small-variant results.

## 3a. CORRECTION to my own selectivity figure — compute it PER DOMAIN

I first reported medium selectivity as 0.34. That divided by the **combined cross-domain union (1,473)**, but
retrieval happens **within** a domain, so the figure was wrong. Corrected, per domain:

| domain | small union | medium union | ingest factor | medium candidates/q | **distinct candidate sets** | selectivity |
|---|---|---|---|---|---|---|
| enterprise | 100 | **874** | **8.7x** | 500 | **211 of 211 questions — every one UNIQUE** | **0.572** |
| web | 100 | **599** | 6.0x | 387–500 | **222 of 240** | **0.829** |
| total | 200 | 1,473 | 7.4x | | | |

**What survives, strengthened:** the qualitative shift is the real story. Small gives **one shared candidate
set per domain** for every question; medium gives a **near-unique set per question** (211/211 in enterprise).
That is the difference between "no retrieval decision exists" and "a retrieval decision exists for every
question", and it is not a matter of degree.

**What must be tempered:** the *pool-narrowing* in medium is modest — a question sees 57% of its domain pool in
enterprise and **83% in web**. So medium is not a hard retrieval problem either; it is a real but gentle one.
**Do not oversell medium as the scaling test.** Enterprise is the more discriminating domain (0.572, 8.7x
ingest) and should be weighted accordingly when reading results. For genuinely sharp selectivity we still need
the larger variants (V1 `longmemeval_m`, ~13x) or the scope-widening ladder.

## 3b. IMPORTANT SCOPE LIMIT — what M18 does and does NOT invalidate

This finding is easy to over-apply. Be precise:

**M18 invalidates RETRIEVAL claims made from the small tier.** Any statement of the form "our store retrieves
better / our memory finds the right evidence" cannot be supported by small-tier results, in either direction.
That includes the negative: vanilla Codex's parity with us says nothing about retrieval quality.

**M18 does NOT invalidate READING claims.** The small tier is a perfectly valid test of *delivery into a
consumer* — which is what M1/M3 identified as the static→agentic gap, and what the whole M7 family targets.
Negative-evidence disclosure is a mechanism about how the reader handles absence in a curated pack; it does
not depend on retrieval selectivity. **So M7/M7b remain valid experiments on the small tier and must not be
discarded on the strength of M18.**

Likewise still valid from small-tier data: the M12 latency result (a wall-clock measurement of the query path,
not a retrieval-quality claim), M11's effort/latency curve (for the small track), M14's latency-distribution
finding, and M5's corpus-coverage audit.

**Rule of thumb: small tier = reading instrument. Medium tier = reading AND retrieval instrument.** Label every
future claim with which instrument supports it.

## 4. Practical blocker for the medium run (must be planned, not assumed)
The current store holds **exactly 100 trajectory sources per domain** (`lcm_trajectory_sources`: web 100,
enterprise 100 = 200 total) — precisely the small tier's union. **Medium needs 1,473 distinct trajectories,
a 7.4x ingest**, roughly 900MB of source text (corpus mean ~640KB/trajectory).

So `--tier medium` is **not a config flip**; it requires a real ingest job first. Plan it explicitly:
build a medium store, verify `lcm_trajectory_sources` reaches the required union per domain, and re-run the
corpus-coverage audit (M5) against the medium haystack before trusting any accuracy number from it.

## 5. Why this was missed for the whole program
The store had exactly the trajectories the small tier needed, every run succeeded, and every number looked
plausible. Nothing failed. **A benchmark that quietly omits the hard half of the task produces perfectly
clean, perfectly misleading results** — and the tell was available from one query
(`count DISTINCT candidate sets`) that nobody ran until the owner asked whether larger variants existed.

**Standing check, added to discipline:** before treating any benchmark as a measure of retrieval, compute the
number of distinct candidate sets and the per-question selectivity. If selectivity is ~1.0, the benchmark
measures reading, not retrieval, and must not be used to justify a retrieval claim in either direction.
