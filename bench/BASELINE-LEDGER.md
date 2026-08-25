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
| #263-fix (gpt-5.6 OAuth cap 372K→272K) | 2026-08-21 | compaction cadence on gpt-5.6 codex-OAuth routes where the host advertises >272K (threshold now computed on the real window; compaction earlier, overflow risk closed) | banked rows predate and pinned their effective windows (F53 tree-pinned; F59 arms recorded windows per-arm); next main-tree row on a 5.6 OAuth route records post-#263-fix | OPEN |
| #332/#333/#338 cloud-embedding privacy transform (privacy trio, v0.23.1) | 2026-08-23 | **production recall-path content shaping (proven boundary, audited 2026-08-26)**: `protect_embedding_text` shapes what the PRODUCTION recall path embeds when sensitive patterns are enabled; patterns default OFF (config.py:701) and a cloud-provider config then raises — pre-#370 `lcm_recall` could swallow that raise into a silent FTS degrade (#367). Durable sensitive redaction PRE-DATES the trio (6376272c, ancestor of v0.22.0) and the trio's transform does not mutate durable source. ⚠ HARNESS GAP: `benchmarking/longmemeval.py` applies NO privacy transform (constructs LCMConfig without enabling patterns; embeds directly) — the benchmark and production embedding paths DIVERGE | **cross-boundary comparisons held pending verification**: F53 reproducibility on current main is UNVERIFIED (source-level analysis only, #367 — no completed rerun). The re-bank run must (a) declare whether the harness enables the privacy transform to match production or measures the raw path, and (b) settle reproducibility empirically. Ledger row was MISSING at merge (reconciled 2026-08-26; boundary corrected same day per the independent audit) | OPEN |
| #352 V1-M provider identity + accounting | 2026-08-24 | benchmarking/longmemeval.py instrument (provider identity/resume/accounting) | LAND-AFTER-REBASELINE (verdict of record: #252 2026-08-24 11:09; ledger entry was comment-only until this row — reconciled 2026-08-26) | OPEN |

Standing caveats adopted from the #252 retro pack (recorded, no re-baseline owed):
- #158 / #170: if any bench profile ever routes through `LCMConfig.from_env()`, per-model
  threshold overrides and `LCM_ABSOLUTE_THRESHOLD_TOKENS` silently outrank the ratio-based
  setpoint and would not appear in the run's own config record. Same trap class as
  RUN-SHEET-V1M-RERANK-ON §1 (explicit-ctor vs from_env).
- #212: `LCM_DISABLED_TOOLS` on a customer box is a memory-coverage change (disabled-tool
  turns stop entering the corpus), not just a prompt-size change.
- #211: a SiliconFlow embedding arm, if ever wanted, enters the scoreboard as a NEW row,
  never as a delta against F53.

Rows appended by the v0.23.0 maintainer train (authorized by the architect ruling, #252
comment 5356005454; classes per its derivation rule; ARCHITECT-TBD cells to be filled by the
architect):

| boundary (merge) | date | surface | required re-baseline | status |
|---|---|---|---|---|
| #168 durable compaction totals in status | 2026-08-20 | status/observability output only | ledger entry only, no re-baseline owed | RECORDED |
| #196 expand-query evidence provenance | 2026-08-20 | answer/assembly (expand-query provenance object) | next main-tree answer row records "post-#196" | OPEN — no main-tree answer row exists yet |
| #173 recall FTS sub-budget + hardened corpus preflight | 2026-08-20 | retrieval path (FTS/vector budget split; scan-budget, session-scope, expiry-evidence preflight; timeout-semantics fix ed586a27) | next full-500 V1-M run + next main-tree LoCoMo row | OPEN |
| #203 replay scaffolding normalization + OOB unique-identity binding | 2026-08-20 | ingest/reconcile (replay proofs; ID-less rows fail closed) | corpus-shaping: next full-500 V1-M run re-baselines (deterministic) WITH an ingest corpus-count parity check vs F53 (fail-closed ID-less drops must show up as count deltas, never silently); next main-tree LoCoMo row records post-#203 | OPEN |
| #273 crashed-search disclosure (coverage 'none', never 'ok') | 2026-08-20 | disclosure-only, crash paths | ledger entry only, no re-baseline owed (verdict 5355967878) | RECORDED |
| #177 partial tool tail replay: occurrence-bounded proofs + tool-name identity | 2026-08-20 | ingest/reconcile (replay identity 5-tuple; replay-snapshot digests change once post-upgrade — persists instead of matching) | corpus-shaping, replay-heavy corpora only: bench corpora ingest fresh (no replay tails) so declared configs are expected-inert — verified by the same V1-M corpus-count parity check as #203; any future bench that REPLAYS real sessions (e.g. the compaction program P1) must re-baseline against post-#177 digests explicitly | OPEN |
| #261 startup FTS bootstrap race + integrity-state publish contract | 2026-08-20 | startup bootstrap (verified-pass clearing; CAS-fenced scan publish) | FTS-availability at process start: next main-tree LoCoMo row (FTS-sensitive; C1-lineage runs start fresh processes, so a fixed bootstrap race can only move FTS-arm coverage upward — treat as part of the joint FTS program decision). V1-M declared config unaffected while its reference fts arm stays dark (F49 class) | OPEN |
| #183 cross-session summary DAG expansion | 2026-08-21 | retrieval path (expand/describe/expand_query cross-session; provenance session attribution) | next full-500 V1-M run + next main-tree LoCoMo row | OPEN |
| #286 teams-slice-1 AccessContextV1 contract | 2026-08-21 | additive, eval-unreachable (all-new paths; verdict 5356722388) | no re-baseline owed | RECORDED |
| #297 auto-focus keeps newest request (summary-steering) | 2026-08-21 | compaction/assembly focus string for over-cap user turns (ingest-side summarization included) | next full-500 V1-M run re-baselines (deterministic); next main-tree LoCoMo row records post-#297 | OPEN |
| #300 teams-slice-2 policy/catalog/scope-storage | 2026-08-21 | additive, eval-unreachable (all-new paths, zero call sites; verdict 5358203472) | no re-baseline owed | RECORDED |
