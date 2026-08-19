# F31 — Phase 1A verdict: the scaling thesis is neither confirmed nor refuted — it is BLOCKED by two named, fixable product ceilings. The latency half wins outright.

**Date:** 2026-07-29 · **Instrument:** Phase 1A (#159), single store scaled 389× (51 → 19,829 sessions;
199,641 messages), retrieval-only, zero LLM spend, 50-question fixed set × 3 reps, LIMIT=25.
**Provider stamp (every recall figure):** fastembed / bge-small-en-v1.5 / 384-dim — NOT production Voyage.
**Concurrency:** B and A3 (the load-bearing latency comparison) ran on an idle machine before any other work —
**UNCONTENDED**. A2/A1's top rungs saw light intermittent contention (doc edits, audit agents ≤seconds); immaterial
because their verdicts are deadline-dominated, not latency-precision claims. A2u pending at authoring; attach on
completion. **B×S0 excluded** (non-functional, protocol §3g). Adjudication by pre-registered protocol
(`PHASE1A-READING-PROTOCOL.md`), script + full JSON in
`session-notes/2026-07-28/hermes-phase1a-adjudication/artifacts/`.

---

## 1. The censored curves (p50 ms / all-gold recall@25)

| rung | B file-scan | A3 semantic+safe-query | A2 semantic+raw-query | A1 lexical+raw |
|---|---|---|---|---|
| S0 (~51) | *(excluded)* | 22 / 0.98 | 105 / 0.94 | 92 / 0.02 (deg) |
| 500 | 18 / 0.60 | 23 / 0.82 | 2,737 / 0.86 | 2,828 / 0.02 (deg) |
| 2,000 | 48 / 0.36 | 54 / 0.60 | 8,010 / 0.10 (66% EMPTY) | 8,011 / 0.00 |
| 8,000 | 179 / 0.26 | 115 / 0.08 (deg) | 8,044 / 0.00 (100% EMPTY) | 8,041 / 0.00 |
| 19,829 | 411 / 0.20 | 267 / 0.00 (deg) | 8,130 / 0.00 (100% EMPTY) | 8,126 / 0.00 |

Distrust checks: §3a arm-B short-circuit does NOT fire (23× latency growth over the ladder's 40× span — an
honest scan). A3u (no deadline) reproduces A3's recall crash exactly → **the crash is not a timeout artifact.**

## 2. ★ Product ceiling #1 — `recall_scan_rows = 25_000` silently blinds semantic search at scale

The instrument names its own cause: every A3 query at 8,000+ is flagged degraded with
*"chunk arm coverage bounded: scored the 25,000 most-recent of 185,175 vectors (older vectors excluded)."*
**Verified in product source:** `config.py:638` (`recall_scan_rows: int = 25_000`), engaged at
`tools.py:3725/3781`. The brute-force vector scan caps at the 25k most-recent vectors; at 389× scale, **86% of
memory is invisible to semantic retrieval**. The recall crash from 0.82→0.00 is this cap, not embedding quality —
recall goes to literal zero exactly when the gold sessions age out of the scan window (the eval questions target
the original 51-session store, the oldest content in the merged corpus — worst case by construction, and the
honest one: real memories age).

It is a *configurable default*, but raising it alone trades the crash for latency (the default exists because
the scan is O(rows)); the real fix is an ANN index (or sharded scan) behind the same API. **Filed as a product
issue.** Until it lands, "Hermes-LCM at >25k vectors" is structurally a recency-window memory.

## 3. ★ Product ceiling #2 — the default query form returns NOTHING at scale

A2/A1 send questions as-is; FTS5 rejects `?`/apostrophes/commas; hermes falls back to a LIKE full-scan; at
2,000+ sessions that blows `recall_query_timeout_s = 8.0` (`config.py:691`) and returns **empty — 100% of
queries at 8,000+.** A naive integrator gets zero memories, silently, at exactly the scale where memory matters.
A3 proves the fix is trivial and already understood (§3e pre-registered): sanitize to the FTS5-safe term form
**inside the product**, not in benchmark harnesses. **Filed as a product issue.**

## 4. The half that WON: latency at scale

A3: 23ms → 267ms p50 over 40× corpus growth (11.5×, sub-linear) — **faster than file-scan at every rung**
(B: 18 → 411ms, 23×, linear-ish) and 30× faster than the banked contended V1 figure. p95 stays sane except the
A3-19829 tail (1.24s — the scan cap's cost showing). UNCONTENDED, fastembed stack. This is the "fast memory
access at scale" half of the thesis, and it held — *given* the safe query form (ceiling #2 fixed) and *within*
the scan window (ceiling #1's shadow).

## 5. What the recall LEVELS do and do not mean

All recall levels carry the §3i persona-collapse confound (500 personas in one store; gold-only scoring), and
the §3h cap-fairness bias (favouring B). Note B *also* collapses — 0.60→0.20 — so file-scan is no refuge at
scale either; its recall halves while its latency quadruples, and it keeps growing linearly. **No recall level
in this experiment transfers to production (fastembed ≠ Voyage; personas ≠ real users). The causal findings
transfer: the scan cap and the query-form death are code, not corpus.**

## 6. Verdict against the pre-registered readings (§2)

- Row 2 fired — recall degrades with scale — and per protocol it is the most valuable outcome. But the
  degradation has **named mechanisms in our own code**, so the honest statement is: **the current
  implementation has a scaling ceiling at 25k vectors and a query-form defect; the architecture's scaling
  behaviour beyond those ceilings is still unmeasured.**
- Row 4 does NOT fire as written: file-scan does not beat A3 on both axes — A3 wins latency at every rung, and
  wins recall below the cap (0.82 vs 0.60 @500; 0.60 vs 0.36 @2,000). Above the cap the comparison is
  cap-vs-crowding, not index-vs-scan.
- **Phase 1A did exactly its job: it found the two ceilings before a customer did, for zero LLM spend.**

## 7. What follows (Phase 1B, spec-first)

1. Fix ceiling #2 (query sanitization in-product) — small, unambiguous, benchmark-independent.
2. Fix ceiling #1 (ANN or sharded scan behind the same API; `degraded` signal already exists and must stay).
3. **Phase 1B = re-run this exact instrument** (same corpus, same 4 rungs, same arms) with both fixes — that is
   the run that actually tests the scaling thesis. Same zero-LLM cost. Pre-register: if recall STILL collapses
   with full coverage and safe queries, the thesis takes the hit with no excuses left.
4. R2 language: the §2 "no scaling language before the curve" rule stands — the curve now says *"fast at scale;
   recall ceiling found and being fixed"*. The candour framing (§3b of the R2 draft) gets stronger, not weaker:
   we measured our own ceiling and published it.

---

## 8. A2u attachment (chain completed 02:51 — 28/28 runs; recall-only per the §6 early-release condition)

The no-deadline raw-query arm, **CONTENDED** (the release run and other work shared the machine) and therefore
**recall figures only** — u-arm latency is void as pre-registered:

| rung | A2u all-gold | empty |
|---|---|---|
| 500 | 0.900 | 0.00 |
| 2,000 | 0.550 | 0.00 |
| 8,000 | 0.200 (deg: scan cap) | 0.00 |
| 19,829 | **0.000** (deg: scan cap) | **0.00** |

**This cleanly decomposes the two ceilings.** With the timeout removed, the raw-query path returns results at
every rung (empty = 0 everywhere — ceiling #2 is purely the 8s budget) and reaches 0.90 all-gold at the small
rung — but still hits **exactly 0.000 at 389×**, with the degraded reason naming the 25k scan cap. Ceiling #1
(#167) is therefore independent of ceiling #2 (#168): unlimited patience does not recover recall, because the
gold vectors are outside the scanned window. Each defect is confirmed by an arm that isolates it. Phase 1A is
complete: 28/28 runs, both ceilings measured, the fix acceptance criteria in #167/#168 unchanged.
