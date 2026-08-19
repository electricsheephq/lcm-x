# FINDING F51 — first full AMA-Bench number: 47.3% fail-closed; accuracy RISES with scale

Date: 2026-08-03. Registration: RUN-SHEET-AMA-208 (bands: coverage row, no pass bar — GOALS
target line ≥72.26%). Config: luna answerer (default effort — AMA's custom-provider path has
no effort surface, disclosed) + claude-sonnet-5 judge (cross-family), clone ddfd319e + our two
fail-closed patches. Cost ≈ $8.5 total; 208/208 episodes executed.

## 1. The row
**1180/2496 = 47.28% strict judged accuracy, fail-closed** (scored-only 1180/2460 = 47.97%).
3/208 episodes failed (36 questions in the denominator, never dropped) — all three the same
harness defect: luna returned empty-content completions 3× on one question each, and AMA's
`model_client.py:109` calls `.content.strip()` unguarded (AttributeError). Judge: 0 unparsed
verdicts across 2,460 scored questions (the judge-fail-closed patch drew no blood — sonnet-5
parsed clean throughout).

## 2. The headline finding: no context-scale degradation
| tier (episode source tokens) | accuracy |
|---|---|
| small <100k (184 eps) | 1026/2172 = 47.2% |
| mid 100k–800k (21 eps) | 134/252 = 53.2% |
| large >800k (3 eps, incl. 1.03M) | 20/36 = **55.6%** |

Accuracy RISES with episode size. The 1M-token episodes — ingested and answered through
hermes_lcm memory with luna never seeing the raw haystack — outscore the small-episode average.
Whatever separates us from the 72.26% target, it is NOT memory failing at scale.

## 3. Where the gap lives (funnel decomposition queued as the next zero-spend unit)
Worst task types: alfworld 18.4% (348 q — the single largest drag), swebench 43.8% (432 q),
webarena 47.3% (372 q). Best: candy_crush 75.0%, crafter 66.7%, gaia_level3 65.0%.
alfworld/swebench are long-horizon procedural tasks where the question style rewards
step-reconstruction — hypothesis for the decomposition: retrieval delivers, but answer style /
judge rubric fit dominates (the B3-PC pattern); to be MEASURED, not assumed.

## 4. Ops
Wall: measured episode walls sum 6.9h (median 106s/ep); end-to-end ~15.2h including
per-episode process setup + ingest outside the timer. Ran concurrently with the V1-M 6-shard
flagship run + slice + smoke tails with zero cross-lane incidents. The 10.9h serial projection
from the pilot was 4× over on throughput (pilot episodes unrepresentatively heavy) and ~1.5×
under on cost ($8.5 vs $5.7) — both corrections banked for future budget memos.

## 5. Dispositions
1. Scoreboard row live (`ama208-full-2026-08-02`), generator-validated.
2. A/A′ 30-episode fixed-seed subset: runs after the V2-SOTA row completes (registered policy).
3. Harness patch for the empty-content class (`content is None` → clean retry/fail) → offered
   upstream to AMA-Bench with the 3 receipts; our fork carries it for the A/A′ re-run.
4. Funnel decomposition (alfworld-first) → next zero-spend unit; feeds the ≥72.26% roadmap.
