# FINDING F55 — V1-M rank-1 forensics: a near-miss profile, and the lever is already in the tree

Date: 2026-08-20. Zero-spend continuation of F54, using the `--dump-candidates` sidecar the
F54 PR added (#233). Source: a full re-run of the F53 config with dumps enabled — verified
**bitwise-identical to F53 on every metric field for all 500 questions** before any analysis
(the fully deterministic instrument doing its job; wall-clock fields differ, nothing else).
Method + artifacts: `session-notes/2026-08-20/v1m-dump-forensics/` (classify_rank1.py,
rank1-classification-full.json, hit-side-control.txt, determinism-check.txt).

## 1. The miss pool is a near-miss pool
97 delivered-but-not-first questions (cross-validates F54's independently derived count).
First-gold rank: **51 at rank 2, 82 within ranks 2–5**, 8 at ranks 6–9.
| first gold rank | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| questions | 51 | 21 | 10 | 7 | 2 | 4 | 1 | 1 |
Median rank by category: preference 4, temporal 3, user 3, others 2.

## 2. Whose fault is the wrong rank-1?
For each miss, does the wrong top session also top the OTHER arms?
- **systemic (≥2 other arms agree): 43** — the corpus genuinely confuses these;
- shared (exactly 1 other arm): 23;
- **lcm_recall-specific: 31** — pure fusion/ordering artifacts; no other arm makes this
  mistake. Directly addressable without new signals.
Other arms already rank gold first on many misses (hybrid_rerank 25, chunk_vectors 24,
summary_vectors 24 of the 97).

## 3. A signal that did NOT survive its control (kept for the record)
On all 97 misses the rank-1 turn item is a precise turn from the wrong session — which looked
diagnostic until the hit-side control: on all 364 hits the rank-1 turn item is ALSO a precise
turn (of the gold session). The turn projection structurally leads with precise items;
item-type says nothing about misses. (Controls before conclusions.)

## 4. The lever selection
The near-miss profile (§1) is precisely the shape a slate reranker fixes, and the product
already ships one, off by default: `_lcm_recall_rerank` (tools.py) — an opt-in
(`LCM_RERANK_ENABLED`), fail-open, single-API-call, intra-window reorder of the fused
top-window inside the `lcm_recall` path, deliberately never spliced onto the RRF score scale.
The instrument's `lcm_recall` arm runs the production path, so the variant is **config-only**.
Registered as `RUN-SHEET-V1M-RERANK-ON.md` (this PR). Known risk, pre-declared as a gate:
the reranker can DEMOTE currently-correct rank-1s (364 questions at stake) — the paired
per-question promotion/demotion count is part of the acceptance bar, not an afterthought.
The 31 lcm_recall-specific artifacts (§2) remain a separate, zero-provider lever (fusion
tie-break) if the reranker underdelivers.

## 5. Disclosures
(1) Zero new spend; dump re-run was embed-cache-served (one shard needed a resume after a
transient Voyage network timeout — same incident class as the C1 stall, lcm-x#235; the
instrument's checkpoint+sidecar resume machinery handled it, and the resumed shard is part
of the 500/500 bitwise verification). (2) Abstention questions (30) excluded by design.
(3) Rank positions read from the dump's deduped session ranking = the scorer's exact input
(equivalence tested in #233).
