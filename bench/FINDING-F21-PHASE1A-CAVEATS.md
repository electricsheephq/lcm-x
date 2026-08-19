# F21 — Phase 1A interpretive caveats, recorded BEFORE the result lands

**Date:** 2026-07-26 · **Issue:** #159 · **Status:** pre-registered caveats + first write-cost measurement
Recorded while the probe was still ingesting, so they cannot be selected after seeing the curve.

---

## 1. ★ The V1 path is a SEMANTIC+LEXICAL hybrid, not the pure-lexical index the V2 runs used

The ingest log shows `provider=fastembed model=BAAI/bge-small-en-v1.5 dim=384`, and each store carries
`summary_vectors` and `chunk_vectors` (500 / 5,261 at the N=500 rung). **The V1 memorybench path embeds.**

Every V2 agentic run this program has done used `semantic_enabled: false` — pure FTS/BM25. So:

- **The Phase 1A scaling curve measures a hybrid vector+lexical retriever**, not the lexical index behind
  our banked V2 numbers. The two are different systems with different scaling behaviour (ANN/flat vector
  search scales differently from an inverted index).
- **Do not generalise the Phase 1A slope to the V2 agentic lane**, and vice versa. If the hybrid scales well,
  that is a claim about the V1 path; the V2 lexical path needs its own measurement.
- This is not a flaw in the probe — V1's banked 444/500 *was* produced by this hybrid path, so measuring it is
  the correct before/after. It is a limit on what the number licenses.

**Consequence for the file-scan arm:** grep is purely lexical. So arm B is not a like-for-like comparison of
*retrieval strategy* — it is a comparison of *"what a coding agent handed a folder can do"* versus *"what our
system does"*, which is the product question and the right one. But it should not be read as
"vector beats lexical".

## 2. First measurement of our WRITE cost — and it is not free

| metric | value |
|---|---|
| ingest rate | **~2.2 sessions/second** (stable: 2.3 / 2.2 / 2.2 across checkpoints) |
| store size | **~52KB per session** (500 sessions → 27.6MB; 2,000 → ~105MB) |
| N=500 rung | 261.5s for 500 sessions / 5,190 messages |
| projected 19,829-session rung | **~2.5 hours, ~1GB** |
| full ladder (30,329 sessions) | **~3.8 hours of pure ingest** |

**Why this matters strategically.** The owner's product thesis is *lossless raw storage + read-time
intelligence* — deliberately trading write cost for read benefit. **This is the first time we have measured
the write side.** A user whose memory grows by 20,000 sessions pays ~2.5 hours of ingest and ~1GB. That is a
real product characteristic, not setup overhead to be discounted, and it belongs in any published scaling
claim alongside the query-latency win.

It also means the *honest* framing of a good Phase 1A result is **"flat read cost, linear write cost"** — not
"scales for free".

## 2b. ★ PROVIDER MISMATCH — Phase 1A is NOT on the production embedding stack

Confirmed by the owner: **Voyage is the production embedding provider.** The banked 444's stores carry
`lcm_embedding_profile` = `voyage / voyage-context-3 / 1024-dim / float32`. **Phase 1A is running
`fastembed / BAAI/bge-small-en-v1.5 / 384-dim`.**

| | provider | model | dims |
|---|---|---|---|
| production / banked 444 | **voyage** | `voyage-context-3` | **1024** |
| Phase 1A (this probe) | fastembed | `bge-small-en-v1.5` | 384 |

**Consequence for the scaling claim:** vector-search cost and recall both scale with dimensionality, and a
1024-dim index is ~2.7x the vector volume per session. So a flat latency curve on 384-dim fastembed **does not
transfer directly to production**. The *direction* (index vs linear scan) is a property of index structure and
should hold; the *magnitude* will not. **Any published scaling number must state the provider, and a
Voyage-provider run is a required follow-up before the claim is made about the product.**

**Not restarting the probe over this.** The question it answers — does indexed retrieval degrade like O(n)
file-scan — is about index structure, and the probe is 1.5h into its final ingest. Restarting would cost more
than the caveat.

**Accidental upside:** `fastembed` + `bge-small-en-v1.5` is **exactly OMEGA's disclosed stack**
(omegamax.co/benchmarks). So Phase 1A measures scaling on the same embedding configuration as our closest
competitor — a fair comparison against them, even though it is not our production config. Worth keeping for
that purpose regardless of the Voyage rerun.

**Also note the write-cost angle:** F21 §2 recorded ~2.2 sessions/second on fastembed/384-dim. Voyage at
1024-dim is an API call per batch rather than local compute, so production write cost has a completely
different shape (network-bound, rate-limited, paid) and the 2.2 sess/s figure must NOT be quoted as the
production ingest rate.

## 3. Pre-registered: what would make me distrust the curve
- **file-scan arm too fast at large N** — if arm B's latency does not grow roughly linearly, it is probably
  short-circuiting (early exit on first match, or not scanning the full corpus). A suspiciously flat arm B
  invalidates the comparison in our favour and must be investigated before the slope is cited.
- **our arm too fast at large N** — check gold recall first. Flat latency with falling recall is a fast
  retriever that misses, which is worse than slow.
- **S0 anchor mismatch** — the probe evaluates 100 of 500 questions, so S0 reproduces the banked *condition*
  (~51 sessions in scope), **not** the banked *score* of 444/500. Any comparison to 444/500 is invalid.
- **per-scale stores** (the agent built one store per rung rather than one store with narrowed scope) is the
  MORE rigorous choice: each rung's IDF statistics match the corpus actually searched. A single store searched
  at narrowed scope would have leaked full-corpus IDF into the small-N rungs, flattering our curve.
