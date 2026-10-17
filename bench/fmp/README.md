# Frontier Measurement Protocol v1

Refs electricsheephq/lcm-x#1189. This harness produces local measurement
artifacts; the 20-question smoke is not a published quality row. No host contract
changes. LCM-X is imported from the explicitly selected checkout, never edited.

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
nice -n 19 bash bench/fmp/run_smoke.sh OUTPUT_DIR
```

## v1.1: subscription-plan transport (no metered API)

Model specs: `codex:<model>@<effort>` runs `codex exec --json` on the ChatGPT plan login;
`claude:<model>` runs `claude -p` on the Claude plan; `openrouter:<id>` (or a bare legacy id) is the
explicit opt-in metered path and needs `--max-spend` and `--spend-ledger`. Defaults: reader
`codex:gpt-6.1-sol@high`, primary judge `codex:gpt-6.1-sol@medium` on every answer, agreement judge
`claude:claude-opus-5-5` on a seeded 10% sample of question ids (`--agreement-judge none` disables),
4 workers. `run.sh` loads the OpenRouter key only for an OpenRouter spec.

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

The smoke runs 30 cleaned LME-S questions stratified over question type plus
abstention (seed 1189; ids in `lme-ids.json`/`lme-strata.json`) with `off`,
Sol/high, three deliveries and two judges, and one LoCoMo conversation (conv-30)
agentic and single. Its workers=3 and persistent shared $9.50 ledger cover all
deliveries and judgments. Reruns must retain that ledger and cap. The default
output is artifacts/1189/r1/smoke. Existing run files are never overwritten by
answer/judge; use a new output directory.

For individual commands, use `nice -n 19 bash bench/fmp/run.sh COMMAND` with:

- `--lcmx-repo CHECKOUT --dataset lme-s|lme-m|locomo --data FILE`
- `--question-ids FILE --split-label LABEL` (JSON string array or one id per line)
- `--config off|local|voyage[+rerank:MODEL] --summaries none|deterministic`
- `--store-root DIR --output DIR --max-spend USD --spend-ledger FILE`
- `--reader MODEL --delivery agentic|single|full_context --workers N` (default 12)
- `--allow-dirty` (answer on an LCM-X tree with uncommitted changes; recorded as `<sha>-dirty`)
- `--allow-partial` (score arms whose run question sets differ after a spend stop)

Use `build-stores`, then `answer`, then `judge` with identical dataset, selection,
split and output arguments. `answer` verifies/builds missing stores before API
calls. For `score`, additionally supply `--runs RUN_DIR ...`; shared CLI options
are still required. LoCoMo uses `sample_id:zero_based_qa_index` question ids and
one store per conversation. LME uses one store per question. Dates are UTC.
LoCoMo speaker identity is retained in stored content and a speaker sidecar alongside the production roles.
Image captions, when supplied by the dataset, remain in the evidence.

Every session id (LME and LoCoMo) is replaced at normalisation by an opaque
`s0001`, `s0002`, … in session-date order; readers never see dataset ids (LME
evidence ids start `answer_`). The mapping lives only in each store's
`build-manifest.json`, for scoring. LoCoMo category 5 (adversarial) is excluded,
as Mem0 and OmniMemEval do; the run manifest records the count under `excluded`.

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
The budget is 12 tool calls and 200k cumulative reported `prompt_tokens`. The next
call's size is the previous call's reported `prompt_tokens` plus ceil(chars/4) of
the appended messages; a tool round is offered only while a forced final after it
still fits, reserving the largest tool result seen (at least 8k tokens) plus 4k for
the tool-call turn. A tool result that would break the budget is replaced by an
error. The forced final keeps `tools` with `tool_choice: "none"`; rows record
`forced_final` and every call's `finish_reason` (reader `max_tokens` 32k).

The spend ledger reserves conservative maximum request cost (including model
pricing overrides) before dispatch. A timeout or missing accounting consumes its
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
`score` also requires equal commits, split label, dataset and id-list hashes. Agreement uses only
valid paired verdicts and reports its N. Paired comparisons exclude a question
if either arm skipped it. Bootstrap uses 10,000 resamples with seed 1189.

`report.md` contains tables only; `report.json` retains statistics, per-type
accuracies and smoke-scaled estimates. `adjudicate.jsonl` has a seeded random
sample of at most 50 blinded disagreements without arm labels. Full-run estimates assume the same token counts,
Opus reader pricing twice Sol and equal latency; local build time, retrieval
quality changes and actual Opus reader latency are unmeasured.

Judges: LME parses a leading yes/no word (any case, trailing punctuation). LoCoMo
sends OmniMemEval's system prompt and temperature 0 where the pinned endpoint lists
`temperature` (the pinned reasoning models do not, so it is omitted and recorded
per call), and parses the JSON `label` field first (fenced or not), else one
CORRECT/WRONG standing alone at the end of the last line with no other capital label
word; anything ambiguous is retried once, then `judge_error`. Every judgment keeps its raw judge text. `judge`
re-hashes the prompt files and refuses a mismatch with the answer manifest.

Judge sources are copied verbatim into prompts.py with attribution and bundled
MIT/Apache-2.0 license texts in licenses/. No upstream runtime imports are used.
