# RUN SHEET — LongMemEval-V1 MEDIUM RE-BANK (F53 successor row on the shipped v0.23.2 posture)

Status: REGISTERED before spend — this sheet is the registration (#380; closes the remaining scope of
#367). Verdict channel: #252. Architect: registration written under the owner's delegation rule
(2026-09-05, Asia/Bangkok calendar — UTC+7; commit timestamps carry the UTC instant): re-running a registered design on the shipped tree cannot move a banked row (append-only
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
  posture production ships. Row label: **post-#352 tree · F53 chunk semantics (flat) · production privacy posture · privacy:v3**.
- The privacy revision (`privacy:v3:<sha256 of the sorted active pattern names>`) is written into the
  checkpoint header by the harness and validated on resume; the run's pins record the exact string.
- **Rerank-protection boundary (#371/#374): NOT exercised.** The primary arm runs without `--recall-rerank`,
  exactly as F53's primary arm did, so the row stays comparable to F53. A rerank-protected row is a separate
  registration (RUN-SHEET-V1M-RERANK-ON lineage).
- **Chunk-embedding mode (F53 semantics): `LCM_LONGMEMEVAL_CHUNK_EMBEDDING_MODE=flat`, recorded by the harness as
  `ingest.chunk_embedding_mode`.** F53 (2026-08-19) embedded every chunk independently through the content-hash cache.
  The #352 instrument change (e38ac74b, 2026-08-24) added contextualized chunk grouping (`embed_chunk_group_batches`)
  for Voyage context models — `voyage-context-3` advertises it — which the cache wrapper cannot serve: an unpinned
  cached run would have taken the grouped path on the raw provider (paid, invisible to the cache pair) and embedded
  contextual vectors F53 never had, breaking §4.4 and §5 by construction. This row therefore holds chunk semantics
  constant and varies ONE thing — the privacy posture; the harness refuses (fails loud) any cached run that would need
  the grouped path, in every mode. Contextual grouping is a separate registration (it needs cache support for grouped
  requests and its own cost basis).

## 2. Design (= F53's declared config; nothing else moves)
- Dataset `longmemeval_m` (sha fb5413e3… and HF revision 2ec2a557… pinned in PREP-V1-MEDIUM-DATASET.md; prepared-m
  manifest 300cf936… pinned in FINDING-F53's pins and the scoreboard row; the prepared-dir manifest verifies
  per-question shas fail-closed). 500 questions.
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
   `protect_embedding_text` returned `changed=True` or raised, measured by the dry-run full pass (`prewarm-cache --dry-run`: every unit validated, nothing embedded; the real prewarm
   repeats the count) over
   `prepared-m` under the declared posture (input-hash parity with the F53-era cache). This is the ledger
   discharge measurement for the #366 / #374 / #384-#391 rows: **0 ⇒ inert for this corpus**; >0 ⇒ the
   boundary is live and any metric delta is attributed to it. Only prepared-corpus DOCUMENTS count: query-text
   transforms are tracked in separate `queries*` counters, reported alongside but never part of this bar.
3. Reproducibility verdict vs the banked F53 row, four pre-declared outcomes:
   - **REPRODUCED** — transform-change count 0 AND per-question results identical to F53's on-disk outputs
     (`lme-runs/m-full2-shard-*`) — then the F53 row is confirmed on the current instrument and the successor
     row records "post-#352 · reproduced".
   - **REPRODUCED-TRANSFORM-INERT** — transform-change count >0 AND per-question results identical to F53's
     on-disk outputs — the F53 row is confirmed on the current instrument and the boundary is recorded as
     live-but-inert for retrieval on this corpus; the successor row records "post-#352 · reproduced; transform
     live, retrieval-inert" and the finding publishes the changed-document manifest (question id + raw/protected unit
     digests per changed unit, written by `prewarm-cache --changed-manifest` during the dry run) so the inertness is checkable.
   - **MOVED-EXPLAINED** — at least one per-question delta, and every delta row passes BOTH tests: (i) its own checkpoint
     row shows a live transform (`privacy.changed > 0` — a corpus document changed — or `privacy.queries_changed > 0`
     — its provider-bound query text changed; query transforms never touch the embed cache pair but can move
     retrieval, so they are an explanation, not drift), AND (ii) the transform is LINKED to the moved retrieval at the
     delta level — question-scope correlation is not causation (a transformed-but-irrelevant unit can coexist with
     unrelated drift on the same row). The link holds when the row's query text changed (`queries_changed > 0`) AND every
     arm that moved on that row is embedding-driven — any arm but `fts`: the lexical arm searches the lossless durable
     store with the RAW question (`fts_hits(store, question.question, …)`; only the summary/chunk/rerank queries are
     protected), so a query transform cannot reach it and an `fts`-only move is never explained by a changed query (the
     hybrid arms fuse a vector component and can be); or when at least one changed unit of that question (a
     `--changed-manifest` record: `question_id` + `unit_index` in the harness's deterministic unit order, mapped back to
     its session by replaying the same walk over the prepared question's `haystack_sessions` that
     `iter_ingest_embedding_request_units` performs, counting units per session — no embedding call)
     belongs to a session or turn that appears in the re-bank run's `--dump-candidates` row for that question, either in
     the top-10 of an arm whose metric moved or among the row's gold sessions/turns (a transformed gold unit that
     dropped out of the top-10 explains a loss the same way). The document transform-change count (bar 2) may be 0
     here: a query-only transform that moves retrieval is MOVED-EXPLAINED, not undefined. Disclosed limitation: F53
     dumped no candidates (its rows carry per-arm metrics only), so the link is established on the re-bank side and
     the finding says so. Outcome: successor row banked; F53 annotated "pre-#332, superseded" (#380).
   - **MOVED-UNEXPLAINED** — any delta on a question whose own row shows NO live transform of either kind
     (`privacy.changed == 0` AND `privacy.queries_changed == 0`), OR whose live transform has no delta-level link
     under test (ii) — instrument drift, or a transform whose effect on this row is unproven: STOP, do not bank,
     root-cause first (root-cause procedure: re-run the delta questions once with the transform disabled into a scratch
     root and diff the two `--dump-candidates` rows — a diagnostic, never banked). The two MOVED outcomes partition the
     delta set by that per-row test (live transform AND delta-level link); a run with rows of both kinds is
     MOVED-UNEXPLAINED. Together the four outcomes cover every run
     and no run matches two: results identical to F53 → REPRODUCED or REPRODUCED-TRANSFORM-INERT (split by the
     document transform-change count); results differ → MOVED-EXPLAINED or MOVED-UNEXPLAINED (split by the per-row
     test alone — the document count does not enter). `--dump-candidates` is therefore ON for every shard and for A/A′
     (§8 steps 5–6); a shard without its candidate dump, or whose dump is missing rows (a `--resume` after a run that lacked the flag does
     not re-dump completed questions), cannot be classified and is re-run from a fresh output root.
4. Corpus identity vs F53 (#203/#177 rows). F53 recorded NO per-question message/node/chunk counts (its checkpoint
   records carry only abstention/arms/category/ingest_ms/question_id/rerank_mode, and each question database was
   deleted after scoring), so per-question count parity CANNOT be executed against F53. The executable identity
   check is the embed-cache pair, at two levels: (a) prewarm — every unique post-transform request unit is already
   in the F53 cache (`prewarm-cache` report: `populated == 0`, `already_cached == unique_request_units`; the F53 cache
   holds 505,695 entries), which proves corpus identity AND a no-op transform in one number; (b) run — each shard's embed-cache pair, RECOMPUTED as the sum of the per-question `embed_cache` {hits, misses}
   rows in `per_question_checkpoint.jsonl` (the run aggregate `ingest.embed_cache` is process-local and would
   cover only the final process of a `--resume`d shard; this PR records the per-question deltas so the sum
   survives an interruption, and restores the aggregate from them on resume), equals F53's per-shard pair (shards 0–5:
   398,139 / 397,857 / 390,572 / 389,980 / 394,388 / 390,617 hits, 2,361,553 total, 0 misses everywhere). Any miss
   is a changed or new document and must reconcile with §4.2. This run's per-question `corpus_counts` are recorded
   as the FORWARD baseline for future rows; the #203/#177 discharge = the cache-pair result + that baseline.
5. A/A′: discordance count + aggregate spread; given a deterministic retrieval metric any discordance is a
   finding, not noise.
6. Negative results ship at the same resolution as positive.

## 5. Cost and time (F53 measured basis)
F53's full 500-question run took **43 minutes** wall across 6 shards on the warm content-hash cache
(505,695 entries; 4.1 GB, present on disk) versus a ~45 h cold projection. The cache key is the sha256 of the
post-transform text, so cost is driven entirely by the transform-change count: 0 ⇒ query embeddings only
(single-digit dollars); a full re-embed worst case is estimated $15–40 (scaling AMENDMENT-2's $5–15 per ~190
questions). **Cost cap: $40 Voyage, scoped to the prewarm (re-embed) spend — the one step whose volume is unknown before the
dry run.** Order of operations enforces it: the 20-sample determinism probe (sample-scoped
transform count; runs without the cache and reports no hit rate) and then `prewarm-cache --dry-run` (cache lookups + privacy validation only — no embedding call, no spend; for cloud
providers the CLI also skips provider warmup, so the dry run makes no provider call of any kind) run
FIRST and report the transform-change count and `would_populate`, the exact number of request units the real
prewarm would embed; projected spend = `would_populate` × **$40 / 505,695 ≈ $0.0000791 per request unit** (the cap divided
by the full-corpus worst case — conservative: it prices a full re-embed at exactly the cap). If the projection reaches the
cap (≥ $40) the run is PARKED and reported before the real prewarm or any shard launches — the real `prewarm-cache` embeds
every miss as it goes, so it is never the first spend-bearing step. Only after the dry-run clears the cap does
the real prewarm run (expected `populated == 0`). Non-prewarm Voyage calls are bounded by construction and are disclosed
rather than capped: the determinism probe embeds 2 × 20 documents (no warmup, no query); each shard process issues one
provider warmup (one more per `--resume` restart); and the run embeds each question text twice — once by the harness and once
by the production `lcm_recall` arm, never cache-served — ≈ 1,000 query embeddings for the 500 questions (A′: ≈ 200). The
harness does not record these dispatches (provider accounting is transient and unwired in the CLI); the bound is the count
itself: ≈ 1,050 request units ≈ $0.08 at the cap-basis price above. Public V1-M data only. OpenRouter: not used.

## 6. Ledger discharges (appended at banking, never edited)
- `#352` → RECORDED (finding records "post-#352 instrument").
- `#332/#333/#338` trio row → posture recorded (durable lossless + provider-copy transform on) and F53
  reproducibility settled per §4.3.
- `#199` → full-500 confirmation (subset already bitwise).  `#245` → zero-moved-rows confirmation.
- `#203` / `#177` → cache-pair identity result + forward `corpus_counts` baseline (§4.4).  `#173` / `#183` / `#297` →
  next full-500 row.
- `#366`, `#374`, `#384/#391` (v0.23.2 train rows) → transform-change count (§4.2).
- `#364` → expected-inert: compaction counts unchanged vs F53 (single-writer harness).

## 7. Abort / park criteria
- >2 shards dead of the same cause → park, root-cause first (no blind restarts).
- Voyage 429 sustained >30 min → halve the shard count, document, continue (operational).
- Any `EmbeddingPrivacyPolicyError` is fail-loud by design: the pre-launch `prewarm-cache` / `determinism-probe`
  commands print a `{"status": "blocked", "privacy": …}` report and exit non-zero (park before launch, root-cause);
  a block inside a shard aborts that shard uncaught, and a `--resume` re-hits the same question → park, root-cause,
  disclose row-level. There is no percentage threshold: one block is a park.
- Projected prewarm spend ≥ $40 → park before launch (§5; the same threshold as §5 — one observable outcome).
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
record pins (§3). 3. `scripts/lcm_longmemeval.py determinism-probe --sample-size 20` → determinism verdict + SAMPLE-scoped privacy
counts (`privacy_scope: sample`; the probe runs with the embed cache disabled and reports no cache statistics).
4. `prewarm-cache --dry-run --changed-manifest <evidence path>` over `prepared-m` FIRST (cache lookups + privacy validation, no
embedding call, no spend — §5 ordering) → `would_populate`, projected prewarm spend vs the cap (≥ $40 → PARK, §5/§7), the
changed-document manifest, and the CORPUS transform-change count (`privacy_scope: corpus`); only after that gate clears, the real
`prewarm-cache` over the same corpus (expected `populated == 0`). Both are reported by the instrument-only
`privacy` counters this registration PR adds to the harness (baseline at the merge-base 0301405b, BEFORE this PR: every
`protect_embedding_text` call site discarded `changed`, and `EmbeddingPrivacyPolicyError` was never referenced in the
harness or driver — a block would have aborted a shard uncounted; this PR replaces that state): `changed` / `blocked` counts in the prewarm and determinism reports and in
`ingest_report["privacy"]` (NOT the checkpoint header — a new header key breaks resume), plus per-question
`corpus_counts` (messages / summary nodes / chunks ingested) and per-question `embed_cache` {hits, misses}
deltas in each per-question record so future rows have per-question parity fields (the forward baseline —
F53 has none, §4.4) and the cache pair can be recomputed from rows. A privacy block is counted wherever it
occurs, including the determinism probe's uncounted de-duplication scan (`count=False` suppresses only the
documents/changed totals, never `blocked`), and the CLI's blocked report falls back to the live module
counters when the raising validator carried none — a blocked report is never all-zero. `prewarm-cache
--dry-run` performs the cache lookups and privacy validation without embedding and reports `would_populate`. **Cache-parity proxy for this row:** the
cache key is the sha256 of the post-transform text and the F53 cache (505,695 entries) is fully warm, so the
`prewarm-cache --dry-run` report must show `would_populate == 0` and `already_cached == unique_request_units`
(then the real prewarm `populated == 0`), and at run time each shard's cache pair — the sum of its per-question
`embed_cache` rows — must equal F53's per-shard pair (§4.4 — the 398,139 figure is shard 0's, not the run's); any miss is a
changed or new document and must reconcile with the transform-change count. 5. Six shards via
the re-parametrised `run_shard.sh` (`--dump-candidates <output>/candidates.jsonl` ON — the per-question top-10 candidate
sidecar §4.3 needs; the CLI refuses a dump path that aliases the checkpoint, the metrics files or the dataset), 5-minute
stagger. 6. A/A′ on `prepared-m-aprime100`, same flags including `--dump-candidates`. 7. **Snapshot raw
outputs to the session-notes artifacts dir before anything else runs.** 8. Recompute every metric from
`per_question_checkpoint.jsonl` — never the run's own aggregate (the aggregates `ingest.privacy` and
`ingest.embed_cache` are restored across `--resume` from the rows, but the checkpoint rows are the record).
9. Per-shard cache-pair parity = the sum of per-question `embed_cache` rows vs `lme-runs/m-full2-shard-*/`
`longmemeval_metrics.json` `ingest.embed_cache`; record `corpus_counts` as the forward baseline. 10. FINDING-F62 + scoreboard row + §6 ledger rows + F53 annotation, one PR
through the gate with an independent audit; #380/#367 close on merge; #252 verdict record.
