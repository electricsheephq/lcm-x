# BASELINE LEDGER — merged changes that move eval baselines

Rule (architect, 2026-08-20, #252 verdict batch): a scoreboard row may be compared against a
row from the other side of a listed boundary ONLY after the noted re-baseline has run. Rows
from pinned non-main trees (e.g. the LoCoMo product worktree) are unaffected by main-side
boundaries. Boundaries are appended, never edited.

| boundary (merge) | date | surface | required re-baseline | status |
|---|---|---|---|---|
| #192 assembly adjacent-assistant row-merge | 2026-08-19 | answer-quality rows (assembled context) | any future main-tree LoCoMo/AMA answer row records "post-#192" | OPEN — no main-tree answer row exists yet |
| #199 preflight maintenance deferral | 2026-08-19 | compaction timing on default path | next full-500 V1-M run doubles as re-baseline (deterministic); LoCoMo on next main-tree config. Containment note: both post-#199 benchmark trees reproduced the pre-#199 95q subset BITWISE → empirically inert on the V1-M declared config | PARTIALLY DISCHARGED (subset evidence); full-500 pending |
| #245 summary-contract acceptance widened | 2026-08-19 | summary ingest (fail-open widening) | ledger entry (this row) + zero-moved-rows confirmation on next full run | OPEN — check on next full run |
| #180 query arg coercion | 2026-08-19 | AMA/agentic arms only | scoped re-baseline on next AMA row. Retrieval rows PROVABLY inert: only retro PR inside the F56 comparison trees; F56 non-override rows bitwise-identical | OPEN (AMA); DISCHARGED (retrieval) |

Standing caveats adopted from the #252 retro pack (recorded, no re-baseline owed):
- #158 / #170: if any bench profile ever routes through `LCMConfig.from_env()`, per-model
  threshold overrides and `LCM_ABSOLUTE_THRESHOLD_TOKENS` silently outrank the ratio-based
  setpoint and would not appear in the run's own config record. Same trap class as
  RUN-SHEET-V1M-RERANK-ON §1 (explicit-ctor vs from_env).
- #212: `LCM_DISABLED_TOOLS` on a customer box is a memory-coverage change (disabled-tool
  turns stop entering the corpus), not just a prompt-size change.
- #211: a SiliconFlow embedding arm, if ever wanted, enters the scoreboard as a NEW row,
  never as a delta against F53.
