# FINDING F53 — V1-M flagship: first full-500 retrieval row (43 minutes, not 45 hours)

Date: 2026-08-19. Registration: `bench/specs/RUN-SHEET-V1M-REGISTERED.md` (+ Amendments 1–2).
Instrument: **lcm-x v0.22.0** (92fd69b7 — carries the ported #198 instrument + #199
checkpoint/embed-cache). Corpus: prepared-m, manifest sha 300cf936… (recovered from the LEXAR
drive and re-verified). Config identical to the smoke basis (Voyage voyage-context-3, FTS
prose flag unset — disclosed, F49-class).

> ⚠ **REPRODUCIBILITY NOTE (appended 2026-08-26).** This row was measured on lcm-x v0.22.0
> (92fd69b7), BEFORE the v0.23.1 privacy trio (#332/#333/#338) changed embedding-input content
> and BEFORE the #365 fix bumped the privacy transform revision. It **no longer reproduces
> bitwise on current main** (see #367): the redaction shapes the embedded corpus, and a
> disabled-policy raise can silently degrade `lcm_recall` to FTS-only. F53 stands as the
> **pre-privacy-trio flagship of record**; a re-banked successor row (its own registration +
> A/A′, with the BASELINE-LEDGER privacy-trio boundary) is the reproducible replacement, to be
> run when it serves a product decision (not speculative spend). Do NOT compare a post-v0.23.1
> row against F53 without that re-bank.


## 1. The row (500 questions, 6 shards, fail-closed accounting)
470 scored + 30 abstention-excluded by the instrument's design; 0 instrument failures.
| arm | recall@1 | recall@10 | ndcg@10 |
|---|---|---|---|
| **lcm_recall (primary)** | **0.4999** | **0.9559** | **0.8610** |
| hybrid_rerank | 0.3652 | 0.7790 | 0.6681 |
| summary_vectors | 0.3631 | 0.7790 | 0.6674 |
| chunk_vectors | 0.3009 | 0.4238 | 0.4273 |
| hybrid_rrf | 0.1429 | 0.6968 | 0.4559 |
| fts | 0.0071 | 0.0455 | 0.0255 |

Delivery headline: **95.6% recall@10** on the primary path — the gold evidence reaches the
delivered set almost always. Ranking headline: **50.0% recall@1** — top-slot precision is the
real product frontier for the medium tier, and the honest lever target.

## 2. Findings inside the row
1. **The smoke was unrepresentative** (disclosure): the 25-question smoke (first-N by manifest
   order) showed r@1 0.96 vs the full run's 0.50 — first-N sampling selected easy questions.
   Standing rule reinforced: seeded-random smokes only; first-N smokes are retired.
2. **The embedding cache collapsed the run 60×**: 43 minutes wall for 500 questions vs the
   ~45h original projection (and a prior 17h run killed at ~40%). Prewarm (505,695 cached
   embeddings, determinism probe 20/20 bitwise = measurement-neutral) made ingest ≈ free;
   per-shard wall ≈ 13–16 min. The A/A′ subset now costs minutes, so noise floors per config
   become routine rather than budgeted.
3. **FTS is dark on V1-M too** (0.7% r@1) — same F49 class (prose flag unset in the declared
   env, disclosed in run-env captures). A V1-M FTS-ON variant follows the C1 pattern if the
   LoCoMo C1 result justifies it.
4. Ops: one launch failure caught by the fail-closed preflight (recovered shard symlinks still
   pointed at the dead LEXAR absolute paths; all 500 rewritten relative; relaunch clean).

## 3. A/A′ noise floor (100-question fixed-seed subset, seed 20260802)
A′ EXECUTED same-day (fresh stores, warm shared cache): **95/95 scored questions
per-question IDENTICAL on all three metrics; aggregate spread 0.00pt** (r@1 0.4639 = 0.4639;
r@10 0.9623 = 0.9623; ndcg 0.8355 = 0.8355). With bitwise-deterministic embeddings (the 20/20
probe) and no LLM in the retrieval loop, this instrument is fully reproducible — the first
zero-discordance config in the program (every LLM-reader config carries 3–11% question churn).

## 4. Dispositions
- Scoreboard row + RUN-LOG entries in this PR.
- GOALS context: the ≥90% flagship target line was set for the QA (reader) row; this retrieval
  row's delivery metric (95.6% r@10) clears that bar, its ranking metric (50% r@1) defines the
  next lever. The reader row is a separate registration (OpenRouter lane, funded).
- Next levers: top-slot ranking (r@1) on medium-tier; V1-M FTS-ON variant decision post-C1.
