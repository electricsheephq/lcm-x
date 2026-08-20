# [wave 1 · R2] Trajectory/experience-memory subsystem + scaling fixes + instrument integrity — DRAFT PR body for #436 (update at consolidation)

> ⚠ DRAFT. Update ⟨TBD⟩ from F34 + sanity slice, then replace the #436 description with this at upstream push.

## What this PR is
The consolidated wave-1 release: the V2 trajectory/experience-memory subsystem, the two scaling fixes found by
our 389× single-store probe, the query-path spend-guard configurability (supersedes #434), the summary-hit
`store_id` fix, the reference-strict citable-delivery engine, and the benchmark evidence trail
(`bench/`, F20–F37).

## Measured results (all paired, pinned, reproducible — provenance blocks in bench/)
- LongMemEval-V1: **455/500 (91.0%)** on the consolidated base — paired vs the banked 444/500: net +11 on 29
  discordant rows, p=0.061 (reported as measured, not claimed as significant); instrument fail-closes 0 vs
  the prior run's 8 (F36, F37).
- LongMemEval-V2 agentic: 298/451 (66.1%), tag bench-H6-P4-298.
- Latency: −56.3 s/question end-to-end vs file-scan exploration (p<0.0001, 48/60 faster).
- 389× scaling probe (F31 → F34): recall cliff eliminated (0.000 → 0.233 at ~200k messages, out-recalling
  file-scan at every rung); raw-query empty rate 100% → 0%. Full-coverage scan cost published: 20–45 ms at
  ≤2k sessions, 5.6 s at 20k — the ANN successor is #171. (An earlier "267 ms at 200k" figure measured the
  broken build's partial scan; it does not ship.)
- Evidence-delivery causality: injecting missing gold evidence flips 23/35 wrong answers (p=1.9e-05).

## What we found wrong and fixed (the honest part)
- 25k-vector recency window blinded semantic recall at scale (recall → 0.000 at 389×) → full batched scan.
- Raw NL queries → FTS5 rejection → LIKE scan → timeout → empty results at scale → in-product sanitization
  (tokenizer-parity verified incl. NFC/combining marks, operator neutralization, compound terms).
- Summary hits missing `store_id` cost whole answers under strict evidence validation (1.6% measured) → fixed.
- Spend-guard starved query-path retrieval after 60 calls/min → configurable with generous defaults (#434).
- The woken summary arm delivered hits strict validation could not cite (16% fail-close on an enriched slice)
  → reference-strict citable delivery with backfill (fork PR #174); fail-closes 0/500 at full scale.

## Review guide (for maintainers)
Highest-leverage files: `search_query.py` (sanitizer — six adversarial findings already fixed in review rounds,
regression-tested), `vector_store.py` (batched scan + cache semantics), `tools.py` (recall arms), the
`lcm_trajectory_*` subsystem (disjoint from V1's message store). Fork-side review: ⟨N⟩ rounds
(Codex sol·max, CodeRabbit, evaOS bot) — logs linked in bench/release-kit/. Known open harness-side issues
(NOT this PR): our reports upstream at LongMemEval-V2 #6/#7 and fork #165.

## Relationship to other PRs
Supersedes #423 (closed; its answer-layer commits measured at no V1 delta — F32) and #434 (fix carried here
verbatim; closing with pointer). #436's earlier test-infra fix (the 3.13 encoder-thread leak) is already in.
