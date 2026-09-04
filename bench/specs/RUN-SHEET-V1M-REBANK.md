# RUN SHEET — LongMemEval-V1 MEDIUM RE-BANK (F53 successor row on the shipped v0.23.2 posture)

Status: REGISTERED before spend — this sheet is the registration (#380; closes the remaining scope of
#367). Verdict channel: #252. Architect: registration written under the owner's delegation rule
(2026-09-05): re-running a registered design on the shipped tree cannot move a banked row (append-only
successor row), so the ≥95% bar is met for benchmark integrity; the sequencing choice (re-bank before B3-A
productization) is a product-priority call the owner may override.

## 0. Why this run
F53 (r@1 0.4999 / r@10 0.9559 / ndcg 0.8610; A/A′ 0.00pt) banked on the v0.22.0 tree with a harness that
embedded RAW text. Production now shapes every provider-bound copy: the v0.23.1 privacy trio
(#332/#333/#338), the #365 fix (#366, privacy:v1→v3), loud policy failures (#370), lossless-by-default with
provider-copy protection and rerank protection (#374), and the truncated-PEM scanner (#384/#391). PR #370 and
#374 aligned `benchmarking/longmemeval.py` to that posture. The banked row is therefore UNVERIFIED on current
main — not proven broken (#380). This run measures the production path once, settles reproducibility
empirically, and is the single discharge event named by the OPEN rows in `bench/BASELINE-LEDGER.md` (§6).

## 1. Declared transform posture (MANDATORY, #380)
- **(a) Production path.** Harness constructs `LCMConfig(sensitive_patterns_enabled=False,
  embedding_privacy_enabled=None)` — durable store lossless, provider copies protected because the provider
  is cloud (`voyage`). This is the only posture the harness supports (there is no raw switch); it is also the
  posture production ships. Row label: **post-#352 instrument · production privacy posture · privacy:v3**.
- The privacy revision (`privacy:v3:<sha256 of the sorted active pattern names>`) is written into the
  checkpoint header by the harness and validated on resume; the run's pins record the exact string.
- **Rerank-protection boundary (#371/#374): NOT exercised.** The primary arm runs without `--recall-rerank`,
  exactly as F53's primary arm did, so the row stays comparable to F53. A rerank-protected row is a separate
  registration (RUN-SHEET-V1M-RERANK-ON lineage).

## 2. Design (= F53's declared config; nothing else moves)
- Dataset `longmemeval_m` (sha fb5413e3…, HF revision 2ec2a557…, prepared-m manifest 300cf936… — pinned in
  PREP-V1-MEDIUM-DATASET.md; the prepared-dir manifest verifies per-question shas fail-closed). 500 questions.
- 6 shards, fixed interleave qid[i::6] (`prepared-m-shards/shard-K`), 5-minute launch stagger, own
  `HERMES_HOME` / `TMPDIR` / output dir per shard (the F53 `run_shard.sh` topology, re-parametrised for the
  new worktree — shard topology is operational, not measured surface).
- Provider voyage / voyage-context-3; batched embedding path; **no reader, no judge** (OpenRouter dependency
  NONE — retrieval metrics are deterministic). Primary arm `lcm_recall`; FTS arm stays dark (F49 class,
  disclosed as in F53).
- Instrument: the post-#352 `benchmarking/longmemeval.py` + `scripts/lcm_longmemeval.py run --resume`
  (per-question `per_question_checkpoint.jsonl`, fail-closed on a foreign checkpoint — the AMENDMENT-2
  lost-progress class is closed).
- Product tree: `origin/main` at launch, which must contain this sheet's merge. Sha from `git rev-parse`.
- A/A′: the F53 fixed 100-question subset (`prepared-m-aprime100`, seed 20260802); discordance and spread
  reported like F53.

## 3. Pins (finalized at launch — every value from a command, none hand-typed)
Repo sha; `benchmarking/longmemeval.py`, `scripts/lcm_longmemeval.py`, `ingest_protection.py`, `config.py`
blob shas; dataset + prepared-dir + A/A′-subset manifest shas; provider identity (model, `privacy:v3:…`
revision string, active pattern set); FULL `_EnvFieldSpec` inventory diff (F49 §5 rule); Voyage key from
Keychain at runtime, never in configs or logs; embed-cache path + size + mtime before the run; per-shard
`run-env-captured.txt`.

## 4. Pre-declared reporting bars
1. Headline metrics (r@1 / r@10 / ndcg over all scored questions) with fail-closed accounting:
   0-delivered rows and instrument failures disclosed per question, never dropped; abstention-excluded count
   reported as in F53 (470 scored + 30 excluded).
2. **Transform-change count** = the number of prepared-corpus documents for which
   `protect_embedding_text` returned `changed=True` or raised, measured by a full prewarm pass over
   `prepared-m` under the declared posture (input-hash parity with the F53-era cache). This is the ledger
   discharge measurement for the #366 / #374 / #384-#391 rows: **0 ⇒ inert for this corpus**; >0 ⇒ the
   boundary is live and the metric delta is attributed to it.
3. Reproducibility verdict vs the banked F53 row, three pre-declared outcomes:
   - **REPRODUCED** — transform-change count 0 AND per-question results identical to F53's on-disk outputs
     (`lme-runs/m-full2-shard-*`) — then the F53 row is confirmed on the current instrument and the successor
     row records "post-#352 · reproduced".
   - **MOVED-EXPLAINED** — transform-change count >0 and the per-question deltas are confined to questions
     whose corpus documents changed — successor row banked; F53 annotated "pre-#332, superseded" (#380).
   - **MOVED-UNEXPLAINED** — deltas on questions with no changed documents — instrument drift: STOP, do not
     bank, root-cause first (compare candidates via `--dump-candidates`).
4. Corpus-count parity vs F53 (#203/#177 rows): per-question ingested message/node/chunk counts must match
   F53's per-question outputs exactly; any delta is disclosed and blocks the #203/#177 discharge.
5. A/A′: discordance count + aggregate spread; given a deterministic retrieval metric any discordance is a
   finding, not noise.
6. Negative results ship at the same resolution as positive.

## 5. Cost and time (F53 measured basis)
F53's full 500-question run took **43 minutes** wall across 6 shards on the warm content-hash cache
(505,695 entries; 4.1 GB, present on disk) versus a ~45 h cold projection. The cache key is the sha256 of the
post-transform text, so cost is driven entirely by the transform-change count: 0 ⇒ query embeddings only
(single-digit dollars); a full re-embed worst case is estimated $15–40 (scaling AMENDMENT-2's $5–15 per ~190
questions). **Cost cap: $40 Voyage.** Order of operations enforces it: the 20-sample determinism probe and
then the full prewarm pass run FIRST and report the transform-change count and cache-hit rate; if the
projected spend exceeds the cap the run is PARKED and reported before any shard launches. Public V1-M data
only. OpenRouter: not used.

## 6. Ledger discharges (appended at banking, never edited)
- `#352` → RECORDED (finding records "post-#352 instrument").
- `#332/#333/#338` trio row → posture recorded (durable lossless + provider-copy transform on) and F53
  reproducibility settled per §4.3.
- `#199` → full-500 confirmation (subset already bitwise).  `#245` → zero-moved-rows confirmation.
- `#203` / `#177` → corpus-count parity check result (§4.4).  `#173` / `#183` / `#297` → next full-500 row.
- `#366`, `#374`, `#384/#391` (v0.23.2 train rows) → transform-change count (§4.2).
- `#364` → expected-inert: compaction counts unchanged vs F53 (single-writer harness).

## 7. Abort / park criteria
- >2 shards dead of the same cause → park, root-cause first (no blind restarts).
- Voyage 429 sustained >30 min → halve the shard count, document, continue (operational).
- Any `EmbeddingPrivacyPolicyError` instrument failure → disclosed row-level; >1% of questions → park.
- Projected spend > $40 → park before launch (§5).
- Host reboot / kill → resume with `--resume` per shard (checkpoint verified); no partial results carry
  over across a re-registration.
- Known blockers named by #380, both verified DOES-NOT-APPLY to this run (2026-09-05, read-only trace):
  #235's network-dependent warmup lives in `FastembedProvider.warmup()` and is reached only for the
  `fastembed` provider — this run pins `--provider voyage` / `LCM_EMBEDDING_PROVIDER=voyage` (F53 `run_shard.sh`
  lines 16 and 25) and the voyage branch performs at most one `embed_query("warmup")` HTTPS call, no HF download;
  #236 is the bun orchestrator of `memorybench-benchmark-tool` (leaked bridge children) — this run invokes
  `scripts/lcm_longmemeval.py run` in Python directly (`run_shard.sh` lines 22-27), no bun, no bridge.
- Fresh output root per shard is REQUIRED: the F53 output dirs cannot be resumed on current main — the
  checkpoint-header validator diffs the union of header keys and main's header carries five keys F53's lacks
  (`chunk_provider`, `chunk_model`, `recall_rerank`, `recall_rerank_window`, `embedding_privacy_revision`).
  The shared embed cache (`embed-cache.sqlite`) IS reused — that is the whole cost model.

## 8. Procedure
1. Merge this sheet (gate: acceptance + independent factual audit). 2. New worktree at `origin/main`;
record pins (§3). 3. `scripts/lcm_longmemeval.py probe --sample-size 20` → transform-change count + cache
hit rate on the sample. 4. Full prewarm pass over `prepared-m` → corpus transform-change count, reported by the instrument-only
`privacy` counters this registration PR adds to the harness (today every `protect_embedding_text` call site
discards `changed`, and `EmbeddingPrivacyPolicyError` is never referenced in the harness or driver — a block would
abort a shard uncounted): `changed` / `blocked` counts in the prewarm and determinism reports and in
`ingest_report["privacy"]` (NOT the checkpoint header — a new header key breaks resume), plus per-question
`corpus_counts` (messages / summary nodes / chunks ingested) in each per-question record so future rows have
a per-question parity field. **Cache-parity proxy for this row:** because the cache key is the sha256 of the
post-transform text and the F53 cache is fully warm (F53 `ingest.embed_cache = {hits: 398139, misses: 0}`),
an identical hits/misses pair on this run proves corpus identity AND a no-op transform in one number; any miss
is a changed or new document and must reconcile with the transform-change count. 5. Six shards via
the re-parametrised `run_shard.sh`, 5-minute stagger. 6. A/A′ on `prepared-m-aprime100`. 7. **Snapshot raw
outputs to the session-notes artifacts dir before anything else runs.** 8. Recompute every metric from
`per_question_checkpoint.jsonl` — never the run's own aggregate. 9. Corpus-count parity vs
`lme-runs/m-full2-shard-*`. 10. FINDING-F62 + scoreboard row + §6 ledger rows + F53 annotation, one PR
through the gate with an independent audit; #380/#367 close on merge; #252 verdict record.
