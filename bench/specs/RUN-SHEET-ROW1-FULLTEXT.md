# RUN SHEET — Row 1: recall on the shipped default (full-text recall, embeddings off)

Status: REGISTERED (2026-10-02), after the v0.24.9 GA cut. The kit is merged first: lcm-x #811 adds the LongMemEval
`--embeddings` arm, and memorybench PR #7 adds the full-text arm to `feat/locomo-hermes-prep` at `b47f92f7`.
Nothing in this sheet spends money. Roadmap reference: H2 in `ROADMAP.md` (recall re-baseline).
Public-copy rule: the merged copy carries no customer, box or person names, no internal aliases, no local paths.
Amended (#818, before any scored run): the product is the lcm-x GA current at launch, run from a detached GA worktree
(§2); one product sha across sub-rows; the R1-S dataset digest; phase-separated reader and judge launches; no reader
tool use. Amended again (#833): no judge tool use either; the R1-M overlay must equal the instrument commit's blobs;
the per-session summary-node maximum of every R1-S / R1-L store is recorded (§2, §6, §7).

## 0. Why this row
- The product default is `embeddings_enabled=False`: recall is full-text unless a deployment turns embeddings on.
- The headline recall rows (V1-M r@10 95.6, V1-S QA 91.0, LoCoMo 54.6 / 67.4) were all measured with embeddings ON.
- So no registered number exists for the default configuration. This row is a NEW baseline. It is never scored against the
  embeddings-on history; the history is shown beside it, labelled with its own configuration.

## 1. Sub-rows (each is its own registered row; one product sha, gated in §2)
| Sub-row | Instrument | Data | Configuration | Scored output |
|---|---|---|---|---|
| R1-M retrieval | `scripts/lcm_longmemeval.py run --embeddings off --provider stub --dataset-label m` (shorthand), once per shard with that shard's own prepared directory and output directory | LongMemEval-M, 500 q (the F53 `prepared-m` manifest), run as the 6 F53 shard directories `prepared-m-shards/shard-K` (fixed interleave `qid[i::6]`; each run scores only its shard); A/A′ on `prepared-m-aprime100` | arms `fts` + `lcm_recall` only; vector arms report `run: false` | r@1 / r@5 / r@10 / ndcg@10, session and turn level |
| R1-S QA | memorybench LongMemEval-S, hermes-lcm provider, `HERMES_MB_EMBEDDINGS=off`, fusion unset | LongMemEval-S cleaned, 500 q (dataset sha pinned at prep) | stores rebuilt from scratch with no embedder | QA accuracy (judge verdicts), per-category |
| R1-L QA | memorybench LoCoMo, hermes-lcm provider, `HERMES_MB_EMBEDDINGS=off`, fusion unset | LoCoMo-10, 1,986 q (adversarial gold per the harness pin; the 99 corrupted-gold rows of the F46 lineage stay in and are scored as-is) | stores rebuilt from scratch | QA accuracy, strict judge rubric (the F46/F61 lineage) |

## 2. Pins (every value from a command at launch, none typed)
- Product: the latest lcm-x GA at launch (`v0.25.0` when this amendment lands), by commit sha; plugin version line;
  `config.py` blob sha.
  - `git rev-parse <GA tag>^{commit}` must equal the sha in the "Latest stable" row of `docs/project-status.md` on
    `origin/main` at launch (`vX.Y.Z@<sha>`; written after the cut, outside the GA tree, because the tag is mutable and
    a GA tree cannot name its own commit). That row must name the GA under test; if it still names an older release,
    update the docs first and do not launch. Any mismatch stops the row.
  - R1-M runs in a detached worktree at the GA commit. Only the instrument files (`benchmarking/`,
    `scripts/lcm_longmemeval.py`) come from the instrument commit, and `git status --porcelain` lists only those
    paths. The overlay must equal the instrument commit's blobs:
    `git diff --quiet <instrument sha> -- benchmarking/ scripts/lcm_longmemeval.py` exits 0, and
    `git status --porcelain --untracked-files=all` shows no untracked file under those paths; record both outputs.
    The harness imports the product from its own checkout, so the measured product is the GA tree byte for byte,
    including modules the recall path imports indirectly (for example `store.py` applies
    `message_content.normalize_content_value` before full-text insertion, so a listed-files check alone could miss a
    corpus change).
  - R1-S / R1-L: the lcm-x checkout each bridge loads is a detached worktree at the same GA commit with
    `git status --porcelain` empty; record its commit sha and the empty status output. A bridge that loads any other
    checkout, or one with local changes, stops the row (an equal `HEAD` sha alone does not prove the loaded code).
  - One product sha: at launch the three sub-rows' product commit shas must be equal. If they cannot be, each sub-row
    is labelled with its own sha and the sub-rows are never reported as one row.
  - Recall path since `v0.24.9`: record `git diff --stat v0.24.9 <GA>` over the recall modules (`tools.py`,
    `retrieval_core.py`, `search_query.py`, `adaptive_retrieval.py`, `store.py`, `vector_store.py`, `dag.py`,
    `db_bootstrap.py`, `config.py`, `message_content.py`). From `v0.24.9` to `v0.25.0` it is a comment in `config.py`
    and a depth query in `dag.py` that condensation and context assembly use, outside recall retrieval (#750).
    For a session with fewer than 1,000 summary nodes the new query returns the same depth set as the previous capped
    path, so the scored context is unchanged there. Record the maximum per-session summary-node count of every R1-S /
    R1-L store once it is built, before scoring; if any session reaches 1,000, label the change as in the scored
    context path for that sub-row.
- Instruments: the lcm-x commit holding the `--embeddings` arm (#811's merge or later); the memorybench commit holding
  `HERMES_MB_EMBEDDINGS` and `scripts/run-with-watchdog.sh` (`b47f92f7` or later on `feat/locomo-hermes-prep`); blob
  shas of the harness files.
- Data: dataset file sha256 and prepared-dir manifest sha for each sub-row; question-id list sha.
  - R1-S: the LongMemEval-S cleaned file `longmemeval_s_cleaned.json`, sha256
    `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` (the banked V1-S / F37 lineage). R1-S runs
    exactly the banked F37 question list: 500 ids, one per line, each line ending in `\n`, list sha256
    `42903357eb3c866f0bba2331dccd8d321a6c7ab57099eb2979c172c1d4f2bc6f`. Prep checks that list's sha and that the
    pinned dataset holds exactly those 500 ids (set equality; the list order is not the file order). A file with
    another sha, a list with another sha, or any id mismatch stops the row. The original (uncleaned) LongMemEval-S
    release is a different file and is not this dataset.
- Reader and judge (R1-S, R1-L): model id, reasoning effort, codex CLI version + binary sha256. Reader = the current
  Sol generation at medium; judge = Sol at low with the strict rubric. The reader differs from the 07-29 row
  (gpt-5.6-sol), which is one more reason R1-S is a new baseline.
- Reader and judge launches: when the memorybench CLI transport reads one process-wide reasoning-effort setting, the
  answer phase and the judge phase are separate launches, each with its own pinned environment (reader: Sol at medium;
  judge: Sol at low) and its own receipt (model id, effort, CLI version, binary sha256, and the served model, or for
  the Codex CLI transport the configured model labelled as configured). A harness commit with separate reader and
  judge settings (`HERMES_MB_CODEX_MODEL` / `HERMES_MB_CODEX_JUDGE_MODEL` and their effort variables) may run both
  phases in one launch; the receipt then records both settings.
- Reader and judge tool use: the reader answers from the delivered recall context only, and the judge grades from
  the question, gold answer and reader answer only. Every reader call and every judge call keeps a durable per-call
  tool-event record, and any tool call by either (filesystem, shell, search, web) stops the sub-row: the F59
  reference arm was invalidated by a reader that searched files instead of using its context, and a judge that can
  look things up makes the strict-judge score unauditable. A transport that cannot call tools satisfies this by
  construction (record which, per phase). If neither is available for a phase, the sub-row is blocked.
- Served model: every reader answer and judge verdict, watchdog-resumed work included, carries the model that
  actually served it (from the transport log), and it must equal the pinned id (the F59 lesson: a requested model
  was silently served by another). If the transport does not expose the served model, that sub-row is blocked
  until it does, except for the Codex CLI transport, which is handled as follows.
  The Codex CLI transport (`codex exec`) does not report the model that served a call: its `--json` events carry no
  model field, and its session record holds only the configured model. It reports only a mismatch: when the server's
  response names a model other than the requested one, the CLI emits an `item.completed` event whose item has type
  `error` and a message starting `model rerouted:`, and the process still exits 0. For that transport the model
  evidence is the harness's per-call telemetry record (requested model and its pin source, provider, effort, CLI
  version, and every attempt's status), which must name the pinned id and the pinned provider. The calls use the
  pinned `-m` argument and a pinned `model_provider`, with the user configuration ignored and no endpoint variables in
  the child environment, so neither a user profile nor an endpoint variable can change the model or the backend. The
  receipt records that no system or managed Codex configuration is present on the run host. Any `--json` error event
  or error item, a `model rerouted` item included, fails that attempt closed: the attempt is recorded as failed,
  its answer is never scored, and it is counted in the run log. The Codex CLI sub-row is blocked until the harness
  commit under test enforces this. The run receipt records the CLI binary sha256, and every transport error fails
  closed (a failed call is never answered by another model). Calls stay `--ephemeral`: a session record would repeat
  only the configured model, and persisting it would store benchmark content in the operator's agent history. This is
  configured-model evidence, not served-model evidence: a Codex CLI sub-row never satisfies a served-model claim, and
  the row's `caveats` states that the model is the configured one as recorded by the client and that the server's
  model is checked only through the mismatch the CLI reports, never confirmed per call.
- Recorded configuration: `retrieval_config` from every R1-M report; `embeddings_enabled` from every bridge
  `initialize` reply; an allowlisted inventory of the non-secret `LCM_*` / `HERMES_MB_*` configuration values.
  Credential variables (for example `LCM_EMBEDDING_API_KEY`) are recorded as present or absent only, never by value;
  in this row they are expected absent. Never capture an unrestricted environment dump.
- Shipped defaults only: apart from the row's declared configuration (embeddings off) and the storage paths the
  harness sets for its own runs (listed), every `LCM_*` / `HERMES_MB_*` value that changes an `LCMConfig` field the
  recall path reads (for example `LCM_RECALL_QUERY_TIMEOUT_S`, `LCM_RECALL_REFERENCE_STRICT`) must be absent or equal
  to the product default. Otherwise the row stops (§6); it is not published as the shipped-default baseline.
- Per-question recall provenance: every sub-row records each question's `degraded` flag and
  `provenance.coverage` in a durable output. R1-M's harness keeps only the hits today, so R1-M is blocked until #817
  lands. A sub-row whose bridge does not expose provenance is blocked the same way.

## 3. Proof before any scored run (positive controls)
1. Kit unit tests green at the pinned instrument commits (off path, on path unchanged, refusal cases).
2. R1-M control: `--limit 10` off vs on (stub provider) on `prepared-s`. Off: `embeddings_enabled false`, vector arms
   `run: false` with null metrics, `fts` + `lcm_recall` with n = 10 minus abstentions. On: vector arms `run: true`.
3. R1-S / R1-L control: one conversation, three searches, off vs on. Off: 0 embed calls and a full-text answer. On: > 0.
4. A control that does not separate its two arms stops the row (no scored run on an unproven arm).

## 4. Pre-declared reporting bars
1. Headline metric with fail-closed accounting: every question scored, failed, or excluded is counted and listed; a
   failed question is never dropped. Exclusion applies to R1-M only: its abstention (`_abs`) questions have no
   evidence-session target. R1-S scores all 500 and R1-L all 1,986; a false abstention there is an error, as in the
   banked V1-S protocol (F37).
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
- A recall-affecting override that differs from the product default (§2) → stop, or relabel the row as a distinct
  configuration.
- Watchdog used all resumes on one run → park that sub-row, report the stall with the log.
- More than 2 R1-M shards dead of one cause → park, root-cause first.
- The GA tag does not resolve to the `docs/project-status.md` pin, the R1-M worktree shows a change outside the
  instrument files or an overlay file that differs from the instrument commit, a bridge checkout is not clean,
  sub-row product shas differ without separate labels, the R1-S dataset or question-list sha differs from its pin, a
  reader or judge tool call appears, a served model differs from its pin, or a Codex CLI call reports
  `model rerouted` → stop.

## 7. Procedure
1. Merge the kit PRs; merge this sheet (fact-check of every pin source). R1-M also waits for #817 (per-question
   provenance).
2. A detached worktree at the GA commit with the instrument files overlaid and checked against the instrument commit
   (R1-M), a clean detached worktree at the same commit for the bridges (R1-S / R1-L); record pins (§2), including
   the per-session summary-node maximum of every R1-S / R1-L store once it is built.
3. Positive controls (§3) → evidence folder.
4. R1-M full 500 (6 shards) + A/A′ subset → snapshot → score.
5. R1-S 500 with A′ subset → snapshot → judge → score.
6. R1-L 1,986 A and A′ → snapshot → judge → score.
7. Finding per sub-row, scoreboard rows (new rows; nothing superseded), ledger lines, issue on the H2 milestone.

## 8. What this row does not prove
- Nothing about the embeddings-on configuration (row 2) or production privacy with embeddings (row 3, INCOMPLETE).
- Nothing about recall inside live sessions on customer profiles; it measures the plugin's recall path on public data.
- For a Codex CLI sub-row, that the server's model equalled the pin on every call: the CLI reports the server's model
  only when it differs from the requested one, and only when the server names it.
