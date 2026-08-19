# FINDING F54 — V1-M r@1 miss decomposition: the 50% headline is three different problems

Date: 2026-08-19. Zero-spend analysis of the banked F53 run (no new measurement; source =
the six `m-full2-shard-*` per-question checkpoints, 470 scored questions). Method + outputs:
`session-notes/2026-08-19/v1m-r1-decomposition/` (decompose_r1_v2.py, decomposition-v2.json).
Claim-boundary check: recomputed aggregates match F53 exactly (r@1 0.4999 / r@10 0.9559 /
ndcg 0.861).

## 1. The aggregate r@1 metric is structurally capped at 0.655, not 1.0
`recall@1 = |gold in top-1| / |gold|`, and 300/470 questions are multi-gold (n_gold inferred
from metric denominators: 170×1, 224×2, 39×3, 17×4, 11×5, 3×6, 6 unknown). One slot can hold
one item, so the per-question max is 1/n_gold → aggregate ceiling **0.6552**. Actual 0.4999 →
**true headroom = 15.5pt**, not 50pt.

## 2. Top-slot reality: 77.4% of questions already put A gold at rank 1
364/470 rank a gold item first. The real miss pool is **106 questions (22.6%)**: 97 with gold
delivered in top-10 but not at rank 1, and 9 hard failures (nothing gold delivered; 6 of the 9
are temporal).

## 3. Where the misses live (full 106-question pool = delivered-not-first + hard failures)
| category | miss pool/scored | delivered + hard | signal |
|---|---|---|---|
| single-session-preference | **18/30 (60%)** | 17 + 1 | worst by far — preference statements not surfacing first |
| temporal | **35/127 (28%)** | 29 + 6 | biggest absolute pool; owns 6 of the 9 hard failures |
| multi-session | 26/121 (21%) | 25 + 1 | |
| single-session-user | 17/64 (27%) | 16 + 1 | |
| knowledge-update | 7/72 (10%) | 7 + 0 | |
| single-session-assistant | 3/56 (5%) | 3 + 0 | essentially solved (94.6% r@1 rate) |

Numerators sum to the full 106-question pool of §2; the split shows delivered-but-not-first
(rescuable by ranking) vs hard failures (nothing gold delivered — a retrieval gap, not a
ranking one).

## 4. Cross-arm rescue: half the pool is already ranked correctly by SOME arm
On 52/97 delivered misses another arm puts gold at #1 (hybrid_rerank 25, chunk_vectors 24,
summary_vectors 24, hybrid_rrf3 21, hybrid_rrf 18, fts 2). Oracle per-question best-arm
aggregate = **0.5661** → perfect arm arbitration alone is worth **+6.6pt**; the remaining
~9pt of headroom needs ranking no current arm achieves. Note: the F53 run's `hybrid_rerank`
ran in `placeholder-cosine` mode — the instrument already supports the real Voyage reranker
(`rerank_sessions_voyage`), untested on this corpus at full scale.

## 5. Levers this ranks (for the next registered variant)
1. **L-R1a — real reranker arm** (config, not code): re-run with rerank enabled; the
   deterministic instrument makes this a paired ~43-min run. Caveat: the reranker is an
   LLM-adjacent provider call — A/A′ needed again (determinism no longer guaranteed on that arm).
2. **L-R1b — product-path arbitration**: lcm_recall already delivers (95.6% r@10); teach its
   final ordering the signals the rescuing arms use (summary-vs-chunk arbitration). Needs the
   candidates dump (below) to see what outranks gold before designing anything.
3. **L-R1c — category levers**: preference-statement handling (30q category, 57% pool rate) and
   temporal anchoring (biggest absolute pool + the hard failures).

## 6. Instrument gap + fix
Checkpoints persist metrics only; ranked lists are discarded → "what outranked gold" is not
answerable offline. Fix in this PR: opt-in `--dump-candidates` JSONL (per-arm ranked ids +
gold sets, byte-identical outputs otherwise). Next step: free re-run with dumps, then classify
the 97 delivered misses by what sits at rank 1.

## 7. Disclosures
(1) n_gold is INFERRED from metric denominators (Fraction.limit_denominator(64) + LCM across
arms/metrics), not read from labels — 6 questions unknowable (no arm retrieved anything);
ceiling treats them optimistically as n_gold=1. (2) Analysis is checkpoint-only; no new
retrieval executed. (3) Multi-gold capping means published r@1 for ANY config on this corpus
should be read against the 0.655 ceiling; scoreboard row unchanged (metric definition is
standard), but future V1-M findings should quote gold@1-rate alongside. (4) The 30
abstention-excluded questions are outside this analysis, per the instrument's design.
