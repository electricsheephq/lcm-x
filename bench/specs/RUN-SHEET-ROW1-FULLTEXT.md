# RUN SHEET — Row 1: recall on the shipped default (full-text recall, embeddings off)

Status: REGISTERED (2026-10-02), after the v0.24.9 GA cut. The kit is merged first: lcm-x #811 adds the LongMemEval
`--embeddings` arm, and memorybench PR #7 adds the full-text arm to `feat/locomo-hermes-prep` at `b47f92f7`.
Nothing in this sheet spends money. Roadmap reference: H2 in `ROADMAP.md` (recall re-baseline).
Public-copy rule: the merged copy carries no customer, box or person names, no internal aliases, no local paths.

## 0. Why this row
- The product default is `embeddings_enabled=False`: recall is full-text unless a deployment turns embeddings on.
- The headline recall rows (V1-M r@10 95.6, V1-S QA 91.0, LoCoMo 54.6 / 67.4) were all measured with embeddings ON.
- So no registered number exists for the default configuration. This row is a NEW baseline. It is never scored against the
  embeddings-on history; the history is shown beside it, labelled with its own configuration.

## 1. Sub-rows (each is its own registered row; same product sha)
| Sub-row | Instrument | Data | Configuration | Scored output |
|---|---|---|---|---|
| R1-M retrieval | `scripts/lcm_longmemeval.py run --embeddings off --provider stub --dataset-label m` (shorthand), once per shard with that shard's own prepared directory and output directory | LongMemEval-M, 500 q (the F53 `prepared-m` manifest), run as the 6 F53 shard directories `prepared-m-shards/shard-K` (fixed interleave `qid[i::6]`; each run scores only its shard); A/A′ on `prepared-m-aprime100` | arms `fts` + `lcm_recall` only; vector arms report `run: false` | r@1 / r@5 / r@10 / ndcg@10, session and turn level |
| R1-S QA | memorybench LongMemEval-S, hermes-lcm provider, `HERMES_MB_EMBEDDINGS=off`, fusion unset | LongMemEval-S cleaned, 500 q (dataset sha pinned at prep) | stores rebuilt from scratch with no embedder | QA accuracy (judge verdicts), per-category |
| R1-L QA | memorybench LoCoMo, hermes-lcm provider, `HERMES_MB_EMBEDDINGS=off`, fusion unset | LoCoMo-10, 1,986 q (adversarial gold per the harness pin; the 99 corrupted-gold rows of the F46 lineage stay in and are scored as-is) | stores rebuilt from scratch | QA accuracy, strict judge rubric (the F46/F61 lineage) |

## 2. Pins (every value from a command at launch, none typed)
- Product: the tree each sub-row actually imports, by commit sha; plugin version line; `config.py` blob sha.
  - `git rev-parse v0.24.9^{commit}` must equal the canonical GA commit `e36ae9c866757d531292db2bf64f5f0b59710bc8`
    (the tag is mutable); any mismatch stops the row.
  - R1-M: `scripts/lcm_longmemeval.py` imports the product from its own checkout, so R1-M measures the instrument
    commit's product tree. R1-S / R1-L: record the tree the bridge loads.
  - Recall-path identity: `git diff --stat e36ae9c8 <measured sha>` over the product files is recorded, and the recall
    path (`tools.py`, `retrieval_core.py`, `search_query.py`, `adaptive_retrieval.py`, `store.py`, `vector_store.py`)
    must be byte-identical to v0.24.9. If it is, the row is
    labelled "v0.24.9 recall path at <sha>"; if not, it is labelled with the measured sha only and the diff is listed.
- Instruments: the lcm-x commit holding the `--embeddings` arm (#811's merge or later); the memorybench commit holding
  `HERMES_MB_EMBEDDINGS` and `scripts/run-with-watchdog.sh` (`b47f92f7` or later on `feat/locomo-hermes-prep`); blob
  shas of the harness files.
- Data: dataset file sha256 and prepared-dir manifest sha for each sub-row; question-id list sha.
- Reader and judge (R1-S, R1-L): model id, reasoning effort, codex CLI version + binary sha256. Reader = the current
  Sol generation at medium; judge = Sol at low with the strict rubric. The reader differs from the 07-29 row
  (gpt-5.6-sol), which is one more reason R1-S is a new baseline.
- Served model: every reader answer and judge verdict, watchdog-resumed work included, carries the model that
  actually served it (from the transport log), and it must equal the pinned id (the F59 lesson: a requested model
  was silently served by another). If the transport does not expose the served model, that sub-row is blocked
  until it does.
- Recorded configuration: `retrieval_config` from every R1-M report; `embeddings_enabled` from every bridge
  `initialize` reply; an allowlisted inventory of the non-secret `LCM_*` / `HERMES_MB_*` configuration values.
  Credential variables (for example `LCM_EMBEDDING_API_KEY`) are recorded as present or absent only, never by value;
  in this row they are expected absent. Never capture an unrestricted environment dump.

## 3. Proof before any scored run (positive controls)
1. Kit unit tests green at the pinned instrument commits (off path, on path unchanged, refusal cases).
2. R1-M control: `--limit 10` off vs on (stub provider) on `prepared-s`. Off: `embeddings_enabled false`, vector arms
   `run: false` with null metrics, `fts` + `lcm_recall` with n = 10 minus abstentions. On: vector arms `run: true`.
3. R1-S / R1-L control: one conversation, three searches, off vs on. Off: 0 embed calls and a full-text answer. On: > 0.
4. A control that does not separate its two arms stops the row (no scored run on an unproven arm).

## 4. Pre-declared reporting bars
1. Headline metric with fail-closed accounting: every question scored, failed, or excluded (abstentions) is counted
   and listed; a failed question is never dropped.
2. Noise floor before any claim. R1-M: deterministic, A/A′ on the F53 100-question subset must give 0 discordant
   rows (any discordance is a finding). R1-S: A′ on the same fixed 100-question subset. R1-L: A/A′ full run;
   the discordant count is the floor.
3. Cost and size per query, recorded or marked UNMEASURED with the reason: delivered context tokens per question,
   reader input and output tokens (from the transport log), wall time p50 / p90, metered dollars (expected $0:
   subscription lanes only, no embedding provider).
4. R1-L: the report states the 99 corrupted-gold rows and the known-corruption ceiling of 95.02% for this 1,986-row set (F46 §6) next to
   the headline, as the earlier LoCoMo rows did. The rows stay scored as-is; no corrected-gold rescoring.
5. Comparison rule: within this configuration only. The embeddings-on rows stay visible with their own pins; a
   difference between the two is reported as "configuration difference", never as a regression or a gain.
6. Negative results ship at the same resolution as positive ones.

## 5. Operations
- Snapshot outputs first: copy every run's output dir to the evidence folder before any analysis touches it.
- memorybench runs go through `scripts/run-with-watchdog.sh <run-id> -- <command>` (lcm-x #236: a stalled pool
  is stopped by process group and resumed with the same run id, at most 3 times; every action logged with UTC time).
- R1-M: own `HERMES_HOME` / `TMPDIR` / output dir per shard, fresh output roots (no resume across instruments).
- A healthy off-mode recall reports `degraded: true` (semantic retrieval disabled) with `provenance.coverage.fts: "ok"`:
  this is the product's label for recall without the semantic arm, the configuration under test, not a failure.
  Record it; never filter a hit on it. A full-text failure is different: `provenance.coverage.fts: "none"` is a
  failed search and is counted as one.
- A watchdog resume is safe to rerun: the bridge records each fully ingested session per container and refuses a
  session cut off mid-ingest (rebuild that container's store, then resume). Count any such refusal in the run log.
- Readers run one lane at a time per machine; no other heavy local run in parallel.

## 6. Abort / park
- Any arm reports the wrong `embeddings_enabled` or a non-null vector metric with embeddings off → stop, root-cause.
- Any embed call counted in an off run → stop.
- Watchdog used all resumes on one run → park that sub-row, report the stall with the log.
- More than 2 R1-M shards dead of one cause → park, root-cause first.
- The `v0.24.9` tag does not resolve to `e36ae9c8…`, or a served model differs from its pin → stop.

## 7. Procedure
1. Merge the kit PRs; merge this sheet (fact-check of every pin source).
2. Fresh worktrees at the pinned commits; record pins (§2).
3. Positive controls (§3) → evidence folder.
4. R1-M full 500 (6 shards) + A/A′ subset → snapshot → score.
5. R1-S 500 with A′ subset → snapshot → judge → score.
6. R1-L 1,986 A and A′ → snapshot → judge → score.
7. Finding per sub-row, scoreboard rows (new rows; nothing superseded), ledger lines, issue on the H2 milestone.

## 8. What this row does not prove
- Nothing about the embeddings-on configuration (row 2) or production privacy with embeddings (row 3, INCOMPLETE).
- Nothing about recall inside live sessions on customer profiles; it measures the plugin's recall path on public data.
