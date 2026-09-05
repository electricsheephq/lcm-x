# FINDING F53 — V1-M flagship: first full-500 retrieval row (43 minutes, not 45 hours)

Date: 2026-08-19. Registration: `bench/specs/RUN-SHEET-V1M-REGISTERED.md` (+ Amendments 1–2).
Instrument: **lcm-x v0.22.0** (92fd69b7 — carries the ported #198 instrument + #199
checkpoint/embed-cache). Corpus: prepared-m, manifest sha 300cf936… (recovered from the LEXAR
drive and re-verified). Config identical to the smoke basis (Voyage voyage-context-3, FTS
prose flag unset — disclosed, F49-class).

> ⚠ **REPRODUCIBILITY NOTE (appended 2026-08-26; boundary corrected same day per the
> independent audit).** This row was measured on lcm-x v0.22.0 (92fd69b7), before the v0.23.1
> privacy trio (#332/#333/#338). Reproducibility on current main is **UNVERIFIED — not proven
> broken**: the analysis behind #367 is source-level, not a completed rerun. What IS proven:
> the PRODUCTION recall path is incompatible with the F53 config as-declared (sensitive
> patterns default off ⇒ the cloud-embedding gate raises; pre-#370 `lcm_recall` could swallow
> that into a silent FTS-only degrade), while `benchmarking/longmemeval.py` itself applies no
> privacy transform (it embeds directly) — the benchmark and production embedding paths
> diverge. F53 stands as the **pre-privacy-trio flagship of record**; the re-banked successor
> (own registration + A/A′, with the BASELINE-LEDGER privacy-trio boundary, declaring whether
> the harness matches the production transform) settles reproducibility empirically.
> **Scope of the freeze (arm-precise):** F53's flagship numbers ARE the production `lcm_recall`
> arm, and that arm is frozen — do NOT compare any later production-path (`lcm_recall`) row
> against it until the re-bank. Harness-raw arms (summary/chunk/hybrid, which never apply the
> privacy transform) did not cross the privacy boundary and remain comparable within the raw
> path, subject separately to the #352 instrument boundary in BASELINE-LEDGER.md.
> **Boundary naming (exact):** the frozen boundary is the embedding-transform content change —
> #332/#333/#338 (shipped in v0.23.1) plus any subsequent embedding-transform revision change
> (e.g. the pending #365/#366 privacy-revision bump, which adds its own ledger row at merge) —
> not the version label alone.
> **Update 2026-09-05 (FINDING-F62):** the registered re-bank (RUN-SHEET-V1M-REBANK, merged 22c12b21) executed on the
> shipped posture and **PARKED at the pre-spend gate**: the production provider-copy transform re-shapes 599 unit
> occurrences (258 of 500 questions) of this corpus and the residual validator REFUSES 3 distinct units (18 questions),
> so the production `lcm_recall` arm cannot be run over LongMemEval-M at 22c12b21 without a posture or product change.
> This settles the note above one step short of a number: F53 is **not executable under the shipped posture as of
> 22c12b21**; the raw path F53 was measured on has not been re-run; F53 remains the row of record and the freeze stands.


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
