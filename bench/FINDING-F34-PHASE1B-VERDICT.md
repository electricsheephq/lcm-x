# F34 — Phase 1B verdict: the recall cliff is GONE and the index now out-recalls file-scan at every rung; the price is full-scan latency at big rungs — ANN is mandated for the next train.

**Date:** 2026-07-29 · **Instrument:** the F31 chain re-run byte-identical (paths-only diff, verified), same
corpus/stores/questions/arms/rungs/reps/LIMIT. Chain: 52 min vs F31's 3h26m — the fixes made the instrument
itself 4× faster. **Provenance (the load-bearing part):** F31 measured `e463cd7` (reflog-proven stationary
through the F31 window; probed: lacks all three fixes). 1B measured `543e9ea` (the merged R2 train; 8/8 probes
pass; fixes on by default). The venv resolves `hermes_lcm` to the checkout working tree via `sys.path`
insertion — recorded as the instrument's build-selection mechanism. **Stability check:** arm B (untouched code
path) reproduced F31 within 5.5% at every rung with identical all-gold — cross-run comparability holds.
**Concurrency:** ambient desktop load both runs (recorded pre/post); no competing benchmark processes.
**Artifacts:** `session-notes/2026-07-29/hermes-phase1b-rerun/artifacts/` (adjudication JSON, probes, machine
state, 56 result files).

---

## 1. #168 (raw queries return nothing at scale): **FIXED — VERIFIED. Issue closes.**

| A2 (raw questions) | F31 | 1B |
|---|---|---|
| empty @19,829 | **100%** | **0%** |
| p50 @2,000 | 8,010 ms (deadline) | **44.7 ms** |
| all-gold @19,829 | 0.000 | **0.247** |
| p50 ratio A2/A3 @19,829 | — | **1.05** (criterion ≤2.0) |

A2 and A3 converged to near-identity at every rung: sanitization now lives in the product and the harness-side
trick is redundant. Every acceptance clause met.

## 2. #167 (25k recency window): **COVERAGE FIXED — VERIFIED. Latency clause FAILS as pre-registered → ANN successor filed.**

**Recall (the acceptance's stated criterion — shape, no collapse):**

| all-gold | 500 | 2,000 | 8,000 | 19,829 |
|---|---|---|---|---|
| A3 (F31) | 0.82 | 0.60 | 0.08⚠cap | **0.000**⚠cap |
| **A3 (1B)** | 0.82 | 0.60 | **0.34** | **0.233** |
| B file-scan (crowding control) | 0.60 | 0.36 | 0.26 | 0.20 |

The cliff is eliminated; the index **out-recalls file-scan at every rung** (and §3h's cap bias favours B, so
the true margin is larger). The residual decline (3.5× over the ladder) parallels B's pure-crowding decline
(3.0×) — the corpus-design confound (§3i, unquantified, both runs) explains the shape; the coverage defect no
longer does. u-arms confirm: no-deadline recall identical — nothing is time-starved.

**Latency (the second clause):** A3 p50 @19,829 = **5,567 ms** (A2 5,842) vs B's 388 ms → **FAILS** "below
file-scan", exactly as pre-registered as a possible outcome. The full scan re-reads the corpus per query
(cold-by-design above one batch, per the PR #169 cache semantics), and 8/150 top-rung queries (5.3%) still hit
the 8 s deadline — so the true uncensored figure is slightly worse. **Consequence, mandated by data, not
softened: an ANN index (or persistent matrix residency) is the next train's work.** Below ~2,000 sessions the
product is 18–45 ms — the fast regime is real and its boundary is now published.

## 3. Anomaly worth its own issue: the lexical arm is now cheaply dead instead of expensively dead

A1 (semantic OFF) returned useless LIKE-fallback rows in F31 (gold 0.02, 8 s burns); in 1B it returns **nothing,
fast** (empty 94–100%, 2–66 ms): the sanitized ~10-term question becomes an implicit-AND FTS query nothing
satisfies — the conjunctivity consequence flagged in the PR #169 comment. Product impact is bounded (the vector
arms carry recall; A2≈A3 end-to-end), but the FTS arm contributes ~nothing to the 3-arm fusion for NL queries.
Filed as a product question (bounded OR-join / IDF-top-terms rescue), next-train candidate alongside ANN.

## 4. What this closes and what it opens

- **#168 closed verified. #167 closes as coverage-verified with the ANN successor filed.** The R2 release-notes
  scaling section is corrected (the flattering 267 ms figure described the broken build and died before
  shipping; the two-sided curve ships instead).
- Release gates: gate 3 (Phase 1B) is **green** — both fixes verified against their pre-registered criteria,
  with the failed latency clause converted into the published boundary + successor work, which is what the
  candour framing is for.
- Open: §3i persona-collision quantification (would sharpen the recall-decline attribution; not blocking),
  the ANN train, the FTS-arm rescue decision.
