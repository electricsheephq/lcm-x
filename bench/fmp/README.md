# Frontier Measurement Protocol v1

Refs electricsheephq/lcm-x#1189. This harness produces local measurement
artifacts; the smoke (30 LME-S questions plus every LoCoMo conv-30 question outside
category 5) is not a published quality row. No host contract changes. LCM-X is imported from the explicitly selected checkout, never edited.

Run from the checkout with Python 3.11+ and `httpx`; reuse an existing LCM-X
environment. Local embeddings require exactly fastembed 0.8.0. No install is
performed by this harness. `run.sh` and `run_smoke.sh` re-exec under nice 19,
cap thread environment variables at four (production FastEmbed uses two),
disable dependency logs, and load keys from macOS Keychain into child env only.
Do not enable shell tracing or HTTP debug logging. The CLI suppresses incidental
stdout/stderr and emits only a sanitized error class on failure. Corpus,
questions, golds, hypotheses and tool JSON belong only to private run outputs.

```bash
export FMP_PYTHON=/path/to/venv/bin/python FMP_LME_DATA=/path/to/longmemeval_s_cleaned.json
export FMP_LOCOMO=/path/to/locomo10.json   # optional: FMP_CACHE_ROOT=/dir/with/earlier/embedding/caches
export FMP_RESULT_CARD_TOOL=/path/to/installed/bench-card.py
nice -n 19 bash bench/fmp/run_smoke.sh /absolute/external/OUTPUT_DIR
```

## v1.1: subscription-plan transport (no metered API)

Model specs: `codex:<model>@<effort>` runs `codex exec --json` on the ChatGPT plan login;
`claude:<model>` runs `claude -p` on the Claude plan; `openrouter:<id>` (or a bare legacy id) is the
explicit opt-in metered path and needs `--max-spend` and `--spend-ledger`. Defaults: reader
`codex:gpt-6.1-sol@high`, primary judge `codex:gpt-6.1-sol@medium` on every answer, agreement judge
`claude:claude-opus-5-5` on a seeded 10% sample of question ids (`--agreement-judge none` disables),
4 workers. `run.sh` loads the OpenRouter key only for an OpenRouter spec.

`glm:<model>` uses the z.ai Coding Plan subscription through the isolated Claude Code runner
at `api.z.ai`, with low thinking and no metered API charge. The operator exports `GLM_API_KEY`
from their own secret store; the harness reads only that environment variable and passes it only
to the child. For run 1, judge every answer with two families:
`--judge codex:gpt-6.1-sol@low glm:glm-5.3 --agreement-judge none`.
GLM readers support `single`, `full_context` (a conservative 128k admission cap), and the unchanged Claude MCP agentic route.

- Codex: isolated `CODEX_HOME` (auth is a symlink to the plan login, never copied) and `HOME`,
  `--ignore-user-config --ignore-rules -s read-only`, every built-in tool feature disabled, a fresh
  empty `--cd` per call under `<output>/harness/`, prompt from a file, stdin from that file. Code mode
  stays on: in codex-cli 0.162.1 it is the only route to MCP tools, and its JS runtime has no file
  system or imports. Any model call other than the code-mode wrapper, any executed item other than an
  lcmx MCP call (or an MCP resource listing), fails the question as `tool_violation`. Model, tokens and
  `rate_limits` come from the rollout; the weekly window at `--quota-stop` (95%) stops new questions.
- Claude: isolated `CLAUDE_CONFIG_DIR` with `{}` settings, an allowlisted environment (no host-session
  markers), `CLAUDE_CODE_OAUTH_TOKEN` read from the keychain per call into the child only,
  `--tools '' --strict-mcp-config --system-prompt <reader/judge system>`, stream-json transcript. Agentic
  adds `--mcp-config` (lcmx only), `--allowedTools mcp__lcmx` and a PreToolUse deny hook (`deny_hook.py`)
  for every other tool. Every reply must come from the requested model (a safeguard fallback to another
  model fails the call); Claude Code's own Haiku side calls are recorded as auxiliary. A usage-limit
  error stops further Claude calls (`not_run`).
- `mcp_reader.py`: stdio MCP server over one frozen store, opened with the same query-only delivery
  open; the agentic toolset unchanged; tools run at the question date; opaque session ids; at most 30
  counted calls; per-call log of tool, argument hash, latency and output size only.
- `canary`: for one harness, a unique reference code is written next to the run output and into the
  dataset directory, and the reader is asked for it, without and with MCP. It passes only if the code
  reaches no transcript, no non-MCP tool call appears, and the MCP run reached the lcmx tools.
- Stores ingest at each session's date through the production store clock (format 3), so recency and
  event times behave as in a live store. Paired tests drop a question with an error in either arm.

The subscription smoke selects 30 cleaned LME-S questions stratified over question
type plus abstention (seed 1189; ids in `lme-ids.json`/`lme-strata.json`) and one
LoCoMo conversation (conv-30, excluding category 5). Sol/high answers both sets
with paired `off` and `local` configurations under `single` and `agentic` deliveries;
`full_context` is available in the CLI but is not part of this smoke. A separate
Opus agentic `off` arm uses a seeded 10 of the 30 LME-S ids (`opus-ids.json`).
Both reader harnesses must pass their leakage canaries before the answer arms run.
Judging uses the CLI defaults: Sol/medium on every answer and Opus on a seeded
10% agreement sample, rather than two judges on every answer.
All commands use four workers and share the subscription quota ledger `quota.json`;
there is no spend ledger or dollar cap in this smoke. OpenRouter remains a separate
opt-in path. The output must be outside the checkout (for example, an absolute run
directory). The script skips an answer step only when both paired arms' `answers.jsonl`
files already exist, and it does not check that they are complete; individual `answer`
and `judge` commands refuse to overwrite existing files. After an interrupted answer step,
or for a new run, use a new output directory.

Smoke matrix and selection: [run_smoke.sh](run_smoke.sh). CLI defaults and options:
[run.py](run.py); model defaults: [models.py](models.py).

For individual commands, use `nice -n 19 bash bench/fmp/run.sh COMMAND` with:

- `--lcmx-repo CHECKOUT --dataset lme-s|lme-m|locomo --data FILE`
- `--question-ids FILE --split-label LABEL` (JSON string array or one id per line)
- `--config off|local|voyage[+rerank:MODEL] --summaries none|deterministic`
- `--store-root DIR --output DIR --quota-ledger FILE --quota-stop PERCENT`
  (quota ledger defaults to `<output>/quota.json`; Codex weekly stop defaults to 95%)
- `--max-spend USD --spend-ledger FILE` (both required only for the metered OpenRouter path)
- `--reader MODEL --delivery agentic|single|full_context --workers N` (default 4)
- `--allow-dirty` (permit an LCM-X tree with uncommitted changes; recorded as
  `<sha>-<content-sha256>-dirty`, hashing the tracked diff plus untracked names/content;
  see [common.py](common.py))
- `--allow-partial` (score incomplete runs or arms with unequal executed question sets
  after a budget stop; declared question ids must still match)
- `--pair-across-commits` (opt in to scoring different LCM-X commit identities;
  default scoring requires matching commits and toolsets)

Use `build-stores`, then `answer`, then `judge` with identical dataset, selection,
split and output arguments. `answer` verifies/builds missing stores before API
calls. For `score`, additionally supply `--runs RUN_DIR ...`; shared CLI options
are still required. LoCoMo uses `sample_id:zero_based_qa_index` question ids and
one store per conversation. LME uses one store per question. Dates are UTC.
LoCoMo speaker identity is retained in a speaker sidecar alongside the production roles.
`--store-text speaker-prefix|plain|name-date` controls ingestion (default `speaker-prefix`,
byte-identical store data): `Name: text`, turn text only, or `Name (YYYY-MM-DD): text`
using the session UTC date. It feeds search, embeddings and deterministic summaries;
LME store data stays identical across variants. Store manifests bind this option.
`--hit-label speaker|role` controls single-delivery labels (default `speaker`); role
uses the stored role and ignores the sidecar. Stored prefixes are stripped once.
Both options are recorded in run manifests and checked for judging; the label is
`n/a` for agentic and full-context delivery. Score keys distinguish these arm axes.
For [#995 step 1](https://github.com/electricsheephq/lcm-x/issues/995):
A0 = `--store-text plain --hit-label role`; A1 = `--store-text plain --hit-label speaker`;
A2 = `--store-text name-date --hit-label speaker`. The agentic check compares
`plain` with `name-date` under `--delivery agentic` (hit label recorded as `n/a`).
Image captions, when supplied by the dataset, remain in the evidence.

For the #995 A0/A1/A2 measurement, include category 5 explicitly and check retrieval
and rendered labels before judging. Defaults are **agentic delivery** and a **Claude
agreement judge**; override both. With `DATA`, `IDS`, `STORES`, and `OUTPUT` set to
the local dataset, frozen all-category IDs file, store root, and a new output root:

```bash
common=(--dataset locomo --data "$DATA" --question-ids "$IDS" --split-label 995
        --include-adversarial --delivery single --summaries none
        --judge codex:gpt-6.1-sol@low glm:glm-5.3 --agreement-judge none
        --store-root "$STORES")
for step in build-stores answer; do
  for arm in A0 A1 A2; do
    text=plain; label=speaker
    if [[ "$arm" == A0 ]]; then label=role; fi
    if [[ "$arm" == A2 ]]; then text=name-date; fi
    bash bench/fmp/run.sh "$step" "${common[@]}" --store-text "$text" \
      --hit-label "$label" --output "$OUTPUT/$arm" || exit 1
  done
done
bash bench/fmp/run.sh check-arms --runs "$OUTPUT/A0" "$OUTPUT/A1" || exit 1
for arm in A0 A1 A2; do
  text=plain; label=speaker
  if [[ "$arm" == A0 ]]; then label=role; fi
  if [[ "$arm" == A2 ]]; then text=name-date; fi
  bash bench/fmp/run.sh judge "${common[@]}" --store-text "$text" \
    --hit-label "$label" --output "$OUTPUT/$arm" || exit 1
done
bash bench/fmp/run.sh score "${common[@]}" --runs "$OUTPUT/A0" "$OUTPUT/A1" \
  "$OUTPUT/A2" --output "$OUTPUT/score"
```

`check-arms` makes no model calls and leaves run files untouched. It compares ordered
recall-hit store IDs, question-ID sets and IDs-file hashes, counts role/speaker labels
in saved reader prompts, and reports median/p90 estimated prompt tokens. Speaker
arms fail on any role-labelled line; missing prompts/recall traces also fail. Keep
`harness/<call_dir>/prompt.txt` alongside `answers.jsonl`. It prints counts/hashes
only and creates `check-arms-<unique-id>.json` beside the runs; any mismatch exits
nonzero. A2 changes ingestion, so A0/A1 are the retrieval-identity control.
Score reports all-category and cats-1–4 totals separately; #995 headlines use cats 1–4.

Every session id (LME and LoCoMo) is replaced at normalisation by an opaque
`s0001`, `s0002`, … in session-date order; readers never see dataset ids (LME
evidence ids start `answer_`). The mapping lives only in each store's
`build-manifest.json`, for scoring. LoCoMo category 5 (adversarial) is excluded by default,
as Mem0 and OmniMemEval do; the run manifest records the count under `excluded`.
`--include-adversarial` loads it with the shared dev/run abstention rubric, records
`include_adversarial: true` and `excluded: {}`, and binds scoring/judging identity.
The decoy `adversarial_answer` is never gold. Without the flag, the manifest field
stays absent and the default question selection and exclusion receipt stay unchanged.

Ingestion manifests bind commit, dataset bytes, id, session map, model, summary
mode and explicit raw-public-benchmark embedding posture. The store directory is
`<config>-<summaries>-<commit12>-<identity hash>/<id hash>`, and every engine
open re-checks the manifest. Rerank variants reuse ingestion;
LCM-X currently skips reranking unless the embedding provider is Voyage. Inspect
verbatim trace results rather than treating a requested variant as proof it ran.
Existing caches are queried read-only by exact provider/model/text digest;
misses go into a run-owned production cache. Rebuilds preserve old generations.
Production SQL connections are query-only during delivery. The agentic toolset is
the engine's production registration (`get_tool_schemas` with its config-dormant
and `LCM_DISABLED_TOOLS` omissions) under the store's effective config, minus
`lcm_expand_query` (calls another model) and the operator tools; the manifest
records it. Malformed arguments and tool exceptions return `{"error": ...}` as the
tool result, as the Hermes host does. Tools run with "now" = the question date:
the harness swaps product modules' `time`/`threading` names for proxies (no product
code change) that fall back to the real clock on threads without a question.

The agentic system prompt states the history is not in the prompt and must be
retrieved with the tools; single and full-context carry the evidence block.
For the metered OpenRouter loop, the budget is 12 tool calls and 200k cumulative
reported `prompt_tokens` ([delivery.py](delivery.py)); subscription agentic readers
use the MCP harness's 30-call cap described above. In the metered loop, the next
call's size is the previous call's reported `prompt_tokens` plus ceil(chars/4) of
the appended messages; a tool round is offered only while a forced final after it
still fits, reserving the largest tool result seen (at least 8k tokens) plus 4k for
the tool-call turn. A tool result that would break the budget is replaced by an
error. The forced final keeps `tools` with `tool_choice: "none"`; rows record
`forced_final` and every call's `finish_reason` (reader `max_tokens` 32k).

For the metered OpenRouter path, the spend ledger reserves conservative maximum
request cost (including model pricing overrides) before dispatch. A timeout or missing accounting consumes its
reservation as estimated cost; actual and estimated charges are distinguished
per call. In-flight reservations prevent dispatch beyond the cap. 429/5xx and
transport timeouts, and HTTP-200 error payloads, get at most three retries. Provider fallback is disabled;
returned model/provider identities and all attempts are recorded. High reasoning
is requested for every pinned model, including Anthropic extended thinking.

Full-context fit uses UTF-8 byte upper bounds with framing. This can
conservatively skip contexts that a model tokenizer would fit; nothing is silently
truncated. Reasoning tokens are a subset of output tokens, not an additional
billable count. Reader errors (never judged) and judge errors count incorrect;
full-context skips have explicit counts and denominators. A spend-guard stop is a
run state (as is HTTP 402, an exhausted OpenRouter balance): those rows are `not_run`, excluded from scoring, the manifest says
`incomplete`, and `score` refuses unequal run sets unless `--allow-partial`.
Every `--judge` model is required: a stop from any required judge excludes the row.
Only `--agreement-judge` stops and failures leave required verdicts intact and drop the sample from agreement.
By default, `score` requires equal LCM-X commits, offered toolsets, split label,
dataset and id-list hashes. Explicit `--pair-across-commits` requires at least two
distinct LCM-X commit identities, permits toolset changes and records each arm's
commit and toolset. Harness identity and the other pairing checks still apply
([score.py](score.py)). Agreement uses only
valid paired verdicts and reports its N. Paired comparisons exclude a question
if either arm skipped it. Bootstrap uses 10,000 resamples with seed 1189.

`report.md` contains tables only; `report.json` retains statistics, per-type
accuracies and smoke-scaled estimates. `adjudicate.jsonl` has a seeded random
sample of at most 50 blinded disagreements without arm labels. Token averages include all attempted
rows and their calls, excluding `skipped` and `not_run` rows, including failures.
Full-run estimates scale only supplied arms to 500 questions with the same token counts and latency;
local build time, retrieval quality changes, unmeasured readers and worker scaling are excluded.
Paired arms record their own active work time, excluding time spent in the other arm.
Reader specs without a known context length are rejected at CLI parse time.

Judges: LME parses a leading yes/no word (any case, trailing punctuation). LoCoMo
sends OmniMemEval's system prompt and temperature 0 where the pinned endpoint lists
`temperature` (the pinned reasoning models do not, so it is omitted and recorded
per call), and parses the JSON `label` field first (fenced or not), else one
CORRECT/WRONG standing alone at the end of the last line with no other capital label
word; anything ambiguous is retried once, then `judge_error`. Every judgment keeps its raw judge text. `judge`
re-hashes the prompt files and refuses a mismatch with the answer manifest.

Judge sources are copied verbatim into prompts.py with attribution and bundled
MIT/Apache-2.0 license texts in licenses/. No upstream runtime imports are used.

Scoring requires `FMP_RESULT_CARD_TOOL` pointing to the installed `bench-card.py`. It snapshots
judged rows and manifests into `score-inputs/`, then writes and checks `RESULT-MANIFEST.json`
and `RESULT.md`; an unavailable writer fails scoring. Use a new score output directory. The card
records the exact re-score command and inputs. T1/T2 publication remains a separate owner action.
For Voyage MCP retrieval, the trusted reader loads the existing `VOYAGE_API_KEY` Keychain service
directly; the model CLI environment, arguments, prompts and MCP configuration never carry its value.

## Cached development loop

`python -m bench.fmp.dev` compares one baseline checkout with a candidate checkout
or JSON configuration arm. This is an outcome-blind frozen development screen:
**kill/futility screen, not a promotion instrument**. It does not replace confirmation, board
measurements, or release gates. The frozen panel contains 25 IDs in each of four
strata; a separate guard retains all development LoCoMo category 5 and LME
abstention items. Main/guard overlap is intentional and evaluated once per arm.
LoCoMo guards use FMP's abstention rubric, without the adversarial decoy as gold.

Set `FMP_LME_DATA`, `FMP_LOCOMO`, and `FMP_RESULT_CARD_TOOL` to the dataset files
and installed card writer. Use the existing FMP environment and local embedding
cache, offline mode, an isolated `TMPDIR`, and a new output directory per command.
The baseline and candidate must provide the supported FMP/product import contract.

```sh
# Cold ingestion is separate from the timed comparison; no model calls here.
nice -n 19 python -m bench.fmp.dev --baseline BASELINE --cache-root CACHE \
  --output PREPARE_OUTPUT --prepare --workers 2
# Cold A/A, then repeat with a new output for warm A/A.
nice -n 19 python -m bench.fmp.dev --baseline BASELINE --cache-root CACHE \
  --output COLD_AA_OUTPUT --workers 2 --deadline-min 30
nice -n 19 python -m bench.fmp.dev --baseline BASELINE --cache-root CACHE \
  --output WARM_AA_OUTPUT --workers 2 --deadline-min 30
nice -n 19 python -m bench.fmp.dev --baseline BASELINE --cache-root CACHE \
  --output D50_OUTPUT --candidate-config '{"recall_depth":50}' --workers 2
```

Default reader/judge are `codex:gpt-6.1-sol@medium` / `codex:gpt-6.1-sol@low`;
only Codex subscription transport is admitted. Default stores use `local` with
summaries disabled, speaker-prefixed text, and FMP's single-delivery renderer.
Only an explicit `recall_depth` override promotes the depth-screen
limit/pool/session/expansion caps and checks their effective readback. Otherwise
the renderer keeps its normal request (limit 25), without patching product caps;
observed product defaults, including a changed checkout's caps, are recorded and
strictly checked against response limits and expansion counts.

The recall-time configuration allowlist is derived from `tools.py`'s
`lcm_recall`, `_lcm_recall_scan_bounds`, summary/chunk scans and
`_lcm_recall_rerank` (the delivery module uses their passed values):
`recall_scan_max_rows`, `recall_scan_budget_s`, `recall_scan_rows`,
`recall_reference_strict`, `recall_query_timeout_s`, `recall_arm_weights`,
`recall_fts_anchor`, `rerank_window_limit`, `rerank_enabled`, `rerank_model`,
`rerank_margin`, `embeddings_enabled`. These and the harness-only `recall_depth`
are the only override keys. Ingestion-only `leaf_chunk_tokens`,
`extraction_enabled`, `assertion_extraction_enabled`, unknown keys, and
embedding identity/privacy settings are refused with a clear error: a frozen
store cannot measure an ingestion change. Rerank still needs a supported provider;
an allowed knob can be inert on a given store, so identical delivered evidence
SHA256s for every paired row yield `INCOMPLETE` with reason `no-op-arm`.
A different product source
requires preparation for that checkout (`--candidate CANDIDATE --prepare`).
Store generations conservatively hash product Python, schema/config/embedding
inputs and ingestion adapters. Retrieval keys also cover query/date, rendered
source identities and overrides. Response keys cover complete messages, endpoint
(including `codex --version`, read once per process),
model/effort, temperature, output limit, tools, rubric and renderer/source hashes.
Private caches contain response/evidence bytes; reports contain IDs/hashes and
numeric metadata. Failed/uncertain sends and unparseable judge replies are stored
as errors, never cache hits, without automatic replay.

Reports expose paired correct-count changes and stratified cluster bootstrap
ci95 intervals (10,000 draws, seed 1188): resample LoCoMo conversations and LME
clusters separately, preserving each stratum's cluster count per draw, then
combine question-weighted. The LoCoMo part rests on **2 conversations**, so its
interval is approximate. The combined guard and both guard breakdowns have
reported ci95 intervals; **LME abstention has only 5 guard items**.
Gains confined to just one LoCoMo conversation can still have a zero lower
bound: stratification does not guarantee a directional win for every +20 pattern.

Rules (guard risk takes precedence over a directional win):
- `DIRECTIONAL-WIN`: diff ≥ `--min-gain` (default 1 item) AND stratified
  ci95 lower bound > 0.
- `GUARD-RISK`: the combined guard panel's net paired delta is ≤
  −`--guard-margin` (integer items, default 2). Guard ci95 is reported only;
  it is not part of this rule.
- `INCOMPLETE`: missing/error rows, invalid judge or recall readback, deadline,
  cache/store/transport failure, or all delivered evidence hashes identical
  (`no-op-arm`). Other complete comparisons report `NO-SIGNAL`.

At n=100 the minimum detectable effect is about **7–11 items** for 7–15%
discordance. Each report computes and prints the approximation
`2.8·sqrt(discordant)/1` items from its measured paired discordance; this is
a screen sensitivity estimate, not independent confirmation.
Reports also retain per-ID caps/tokens, call-cache counts and timing; result
cards name the guard margin and measured MDE. A warm cached A/A is an identity
control (`INCOMPLETE/no-op-arm`), never an independent placebo sample. Result cards cover advisory timing;
publication and the full acceptance pilot remain separate owner actions.

### DM0 derived-memory instrument (#1191)

`python -m bench.fmp.derived [inventory|freeze|extract|answer|score] --product PRODUCT
--source INPUT.json --run RUN [--cache W1_CACHE]` defaults to read-only inventory.
It is bench-only and leaves product flags, V4 publication and existing FMP defaults alone.
`freeze` requires complete LOCAL/25 retrieval/store identities, rebuilds DM0-886 from
hash-pinned datasets, and snapshots a create-only manifest before any model call.
Incomplete LOCAL/25 receipts refuse admission; OFF/25 cannot substitute.
Freeze binds panel IDs and A0 bytes to verified retrievals; live `w1_cache` cannot be null.
Added questions need `added_receipt` in the same verified LOCAL/25 cache. Imported
controls must match their original response records and verified source identities.
Freeze requires 44 unique repeat IDs in the panel, each with its control record and
baseline/source/renderer/endpoint identity; missing repeat admission refuses before calls.
The manifest pins renderer/endpoints and byte digests of every `bench/fmp` Python module.
Resume refuses changed modules by name; a fresh run directory freezes the new identity.

INPUT is a private JSON export: schema `dm0-v1`, pinned `product` source hash and
`product_commit`, `source_files`/`dataset_sha256`, `normalizer_sha256`, `dev_ids`,
`w1_panel`, `w1_receipt`, `w1_cache`, exact `baselines` messages by canonical ID,
original `controls` response records, `w1_endpoint`/`w1_renderer`/`w1_sources`,
versioned Codex/GLM `endpoints`, `renderer`, frozen `repeat_ids` (44), and an absolute
Unix `deadline`. Missing added-question baselines require pinned-product preparation
before freeze; this command never builds W1 stores or substitutes retrievals.
Synthetic fixtures instead provide `questions` and `panel` and set `synthetic=true`.
Semantic-screen evidence remains a separate prerequisite (`semantic_valid`); fabricated
hard cases and the conservative `v4_row_candidate` diagnostic cannot certify X05 quality.

Only separately authorized live runs use `extract`/`answer --activate`: subscription
Sol/medium extraction/readers and Sol/low plus GLM 5.3 judgments, both on every answer.
The launcher supplies GLM credentials; the instrument never reads credential files.
Resume uses immutable inputs and successful calls; uncertain sends require disposition.
Completed turns remain successful despite non-fatal reconnects. Failed calls use only
the fatal outcome (last error if no turn failure) and stderr tail for classification.
Timeouts, HTTP 5xx/unmarked 429 outside backend usage-limit predicates, resets/refusals
and stream/protocol failures retry unchanged after 60, 180 and 600 seconds.
Every retry rechecks the frozen deadline after sleeping, before recording or sending.
Deadline expiry settles a partial PARK-INCOMPLETE with scoring snapshots; resume is a no-op.
Usage limits match each backend, including GLM error-result 429 and rate-limit markers.
Quota/usage stops return `PAUSED` without exclusions or a settled report. Rerun the
same directory after reset to retry the paused call and reuse saved calls; paused
wall time does not consume the frozen deadline.
Exhausted questions are excluded from both arms and judges; more than 3% is INCOMPLETE.
Each exclusion budget counts only IDs in its population, including targets-only interim.
The drift cap is 3% of the 44 repeat IDs; two lost repeats fail admission.
Extraction exclusions above 3% of panel IDs settle before any drift or main-panel call.
Valid extraction siblings survive rejection; more than 5% rejected items is INCOMPLETE.
Failed semantic admission (including missing approval or empty extraction) settles before
drift or main-panel sends. Invalid extraction JSON (including non-string content),
envelope identities or array types receive one repair attempt. All item fields must be
strings except nullable string dates and sources; malformed items get one repair while
valid siblings survive. A failed envelope/JSON repair is terminal:
the saved disposition settles PARK-INCOMPLETE; resume neither retries nor sends again.
Invalid judge replies in the main panel and drift retry once unchanged; persistent
invalids are reported per arm/judge. Main-panel invalids above 2% are INCOMPLETE;
drift allows `max(1, ceil(0.02 * n))` invalids per judge/arm (one at 44 repeats).
Invalid drift settles PARK-INCOMPLETE before any main-panel send; drift quota pauses.
Drift exhaustion excludes the question from all main-panel arms and judges.
Drift snapshots omit volatile cache-hit labels so an interrupted checkpoint replays identically.
Unsettled calls remain resumable.
When memory is empty, controls reuse the A0 answer. Cards include rejection counts/rates.
Each judge's headline reports its scored overall n; panel and evaluated-prefix lengths appear separately.
Source-only 16k-token windows plus at most 2k context are batched three per call.
Chunk identity `overlap-v2` overlaps by the admissible quote limit (2,400 characters,
or half the chunk limit for smaller windows); exact source offsets deduplicate overlap.
Whole source-linked groups, including quotes/references/header, fit an 8,192-token cap.
A0 hit/question bytes stay fixed; only the declared memory insertion changes M8192.

The fixed target runs first; conservative paired and clustered bounds may stop futility.
GO requires both judges' +5/+2-point, Holm, positive cluster-lower, category and strict
combined-guard gates. Every original frozen category needs evaluable items after exclusions;
an empty category is INCOMPLETE. PARK variants distinguish futility, uncertainty, no-go
and incomplete coverage/transport/identity. GO funds Stage 1 only. `score` replays frozen scoring inputs
and writes a result card; T1/T2 publication, live quality and product benefit remain unproven.
