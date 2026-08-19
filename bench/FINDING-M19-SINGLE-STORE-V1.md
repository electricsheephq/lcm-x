# M19 — The 389x scaling test is already on disk: re-ingest V1 as ONE store

**Date:** 2026-07-25 · **Status:** decisive roadmap change · **Cost:** zero new data, no download
**Follows:** M17 (larger variants exist) · M18 + §0 self-correction (small pools, cheap exhaustive scan)

---

## 1. What our banked results actually measured

| result | retrieval corpus per query | shared across questions? |
|---|---|---|
| V2 small — 125/451 static, 298/451 agentic | **100 trajectories** (200 total union) | **yes — 2 sets for 451 questions** |
| **V1 — 444/500** | **51 sessions / 514 messages** (one isolated .db per question) | no, but each store is tiny |

Verified directly: `mb-workdir-500q` holds **500 separate `.db` files**, one per question, 2.7MB each
(`messages: 514`, `summary_nodes: 51`), 1.3GB total.

**Neither banked headline result tested retrieval at meaningful scale.** V2 small is a shared 100-file pool;
V1 is 514 messages in an isolated store. Both are trivially exhaustible, so an index has nothing to beat in
either. This completes the attribution story: our architecture has never been measured where it should win.

## 2. The experiment is already on disk

Ingest V1 **once**, as a single store over the union of every question's haystack, and scope per question:

| | as we ran it | **single store** | factor |
|---|---|---|---|
| sessions | 51 | **19,829** | **389x** |
| messages | 514 | **199,641** | **388x** |
| new data required | — | **none** (265MB already local) | — |

**389x scale, zero download, and the same 500 questions with the same gold answers** — so it compares
directly against the banked 444/500. That is a far larger and cheaper scaling test than the V2 medium run I
was about to commission (enterprise union 874, only 8.7x).

## 3. Why this is the right experiment
- **Direct before/after on identical questions.** Any accuracy change is attributable to corpus scale alone,
  not to a different question set — the error that invalidated our A/B claim (M12).
- **It tests the actual product claim.** "Your memory can grow without your queries getting slower" is exactly
  51 → 19,829 sessions.
- **It is the honest version of the vanilla comparison.** A coding agent grepping 514 messages is trivial; a
  coding agent grepping 199,641 messages is not. **This is where file-scan should break and an index should
  not.** V2 small could never show that.
- **Both failure modes are informative.** If accuracy holds and latency stays flat → thesis proven on real
  data. If accuracy DROPS at 389x → our retrieval does not scale, which is a critical product finding we
  currently have no way of knowing.

## 4. Predeclared expectations (before running)
- **Latency:** near-flat for our indexed store (FTS/BM25 lookup); ~linear for a file-scan arm.
- **Accuracy:** the risk case. At 389x more distractors, precision may fall — some questions that resolved
  from 51 sessions may now surface wrong evidence. **A drop here is a genuine negative about our retrieval and
  must be reported as prominently as any latency win.**
- **Gold recall** is the diagnostic that separates the two: if recall holds and accuracy falls, the reader is
  being confused by distractors (a delivery problem). If recall falls, retrieval itself does not scale.
- **IDF caution (M18 lesson):** a 389x larger corpus changes corpus-level IDF and therefore lexical ranking.
  That is not a confound to remove — it IS the phenomenon under test — but it must be stated, and the
  single-store build must NOT overwrite the per-question stores that back the banked 444/500.

## 5. Roadmap change
1. **Single-store V1 (this finding) — now the top scaling experiment.** No download, 389x, direct
   before/after against a banked number.
2. **V2 medium** — demoted below it. Still worth running (official track, lower bars, submittable), but as a
   *leaderboard* play rather than the scaling proof. Its 8.7x is dwarfed by 389x.
3. **`longmemeval_m` download** — deferred. Only needed if single-store V1 proves inconclusive.
4. **Custom scope-widening harness** — keep only the LLM-free retrieval slope measurement.

## 6. Process note
This was found by asking what our own store actually contained, after the owner asked whether larger dataset
variants existed. Three findings in a row (M17, M18, M19) came from enumerating artifacts we already had
rather than acquiring anything new. **The cheapest experiment available was sitting in a working directory the
whole time.**
