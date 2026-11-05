# R1 frontier shim

Run from the LCM-X root with its Python environment (httpx is already an FMP
dependency). No added dependency or product change. Only the pinned `sol-low`
name is served: `codex:gpt-6.1-sol@low`. GLM-5.3 judging goes directly through
the harness to `https://api.z.ai/api/coding/paas/v4`; FMP's `glm:` transport is
not used. Add a thinking-disabled GLM route only if R1-0 demonstrates that the
direct endpoint thinks by default. No Claude instrument is used.

The operator supplies `LCMX_SHIM_CODEX_HOME`, an existing absolute directory
leased through Switchboard. The shim passes that path to FMP without inspecting
credentials or creating auth links. FMP's existing default home and timeouts
are unchanged for other callers; total-token pass-through is opt-in too.
Launch in an otherwise idle model-stage window:

```sh
python -m bench.frontier_shim.server --root "$R1_SHIM_STATE" \
  --log "$R1_CALL_LOG" --concurrency 4
```

The bind address is always **127.0.0.1**, default port 8765 (`--port` changes only
the port). `POST /v1/chat/completions` accepts any bearer key, including a dummy
key. Messages must have system/user roles and string content; streaming and
additional generation controls are rejected. Responses include assistant text,
the model FMP reads back, and only reported token fields. Missing usage fields
remain absent, including total tokens if FMP reports no total; no zero or total
is invented. A missing rollout returns 503; receipts retain FMP's identity source
and weekly usage readback. Model mismatch, tool violation or invalid answer shape
returns 422 so the harness does not retry a deterministic failure.

Codex **flattens system and user messages** in order with blank lines. Requested
temperature is ignored and `temperature_applied: false` is recorded per call.
The Codex agent wrapper adds about 20k tokens: returned prompt usage is wrapper
inclusive, not answer-context usage. Measure answer-context tokens separately
from the stage inputs; never label the returned usage as context-only.
R1 token figures come from the shim call log, not the harness token report: that
report fabricates total_tokens=0 when absent and includes Codex's wrapper in
answer prompt tokens (about 9.3k observed; overhead varies).

Codex concurrency is 1–4. Set **`--concurrency 2`** (or lower) when a Codex-route
row runs, accounting for that row's own parallelism; set the harness's answer
and Sol re-judge `--llm-workers` to the same size. Slot acquisition times out at
30 seconds with 503. Each text call has FMP's **540-second subprocess timeout**
(agentic timeout stays 1800); `subprocess.run` kills and reaps on timeout. A slot
is released only when the backend returns, even if the HTTP client disconnects.
Keep the harness client timeout at 600 seconds. Transport failures return 503.
BEAM eval must satisfy `llm_workers * num_runs <= shim concurrency` because each
worker gathers `num_runs` calls. Set `ANSWER_MAX_RETRIES=1` for every shim answer
stage and `EVAL_MAX_RETRIES=1` for every Sol re-judge stage in the harness env.
Put these in the env file: the harness overrides exported settings when loading it.
Pass `--skip-failed-answer 0` and `--skip-failed-judge 0` on all applicable shim
runner stages (PersonaMem's runner has only the answer flag). Skipped failures
are saved permanently and resume does not ask them again.

A reported usage limit or FMP quota stop returns **429, Retry-After: 60** and
latches the shim stopped until restart. No reset wait and no internal retries.
On the first `codex_stopped_restart_required`, kill the stage immediately; its
per-item results files allow safe resume. Stop the shim with **SIGTERM**
(`kill -TERM <shim-pid>`), wait for it to exit, use `codex-account-swap` to swap
the lease, restart with the leased home, and resume the stage. SIGINT also drains
active calls; stalled client sockets time out after 60 seconds. Each restart uses a fresh
FMP quota ledger; prior ledgers and call receipts remain. Calls already in flight
may finish after a stop. Client retries can duplicate calls; repeated request
hashes expose them. The log's retries count is shim retries (always zero), not
the harness retry ordinal or hidden CLI retries.

JSONL receipts contain UTC time, pinned name/spec, served model, latency, reported
usage, HTTP status, retries, temperature disclosure and SHA-256 of the exact
request/response body bytes. Headers, environment values, prompts, answers and
exception text are never included. FMP traces and leased-home rollouts are
separate, sensitive raw artifacts; rollouts stay enabled for identity and quota
readback. Record leased-home disk growth during smoke and pilot; prune nothing.

## Release-manager-only five-question real-lanes smoke (not run by this lane)

Prerequisites: Parts A2/B have supplied a **scratch harness copy**, the five
original-id questions and their offline ingest/search output, complete embedding
catch-up receipts (`remaining = 0` for both corpora), zero degraded payloads,
and copied tiktoken/fastembed caches. Use a fresh version `r1-smoke-5`, with only
search results and status files, no prior responses/judgments. The scratch copy
must contain only those five dataset rows. `R1_SMOKE_HARNESS` names that copy;
activate its Python environment first. `R1_MODEL_ENV` is the operator-managed
harness env file: `ANSWER_MODEL=sol-low`, dummy `ANSWER_API_KEY`,
`ANSWER_BASE_URL=http://127.0.0.1:8765/v1`, `EVAL_MODEL=glm-5.3`,
`EVAL_BASE_URL=https://api.z.ai/api/coding/paas/v4`, and `EVAL_API_KEY` supplied
through the approved local mechanism. Keep both client timeouts at 600 seconds.
No keys or auth files belong in the shim tree or its receipts.

With the shim running at concurrency 4, the exact **model-stage** smoke command
is below. It does not ingest or search on this Mac; those stages must already
have passed offline. Run outside latency rows and the reserved UTC test window.

```sh
set -eu
cd "$R1_SMOKE_HARNESS"
python - <<'PY'
import json
from pathlib import Path
p = Path("results/lme/lcmx-r1-smoke-5")
rows = json.loads((p / "lcmx_lme_search_results.json").read_text())
assert len(rows) == 5 and all(len(v) == 1 for v in rows.values())
assert not (p / "lcmx_lme_responses.json").exists()
assert not (p / "lcmx_lme_judged.json").exists()
PY
nice -n 19 bash scripts/run_lme_eval.sh --lib lcmx --env "$R1_MODEL_ENV" \
  --version r1-smoke-5 --from-step 3 --to-step 4 --llm-workers 4 --num-runs 1 \
  --skip-failed-answer 0 --skip-failed-judge 0
```

The release manager checks five completed answers and judgments, captures the
GLM endpoint's default thinking behavior (reasoning metadata may require a direct
response receipt), and records leased-home growth. If thinking is enabled, stop
before the pilot: add/review the specified thinking-off route. On quota failure,
recover and resume the same stage without the fresh-output assertions.

## Proof boundary

Offline suites mock backends and subprocesses: request/response, missing usage,
400/422/429/503, stop latch, 2/4-slot caps, 30-second wait and 540-second timeout
forwarding, safe log hashes, home pass-through, and untouched FMP defaults.
They do not prove real-lane identity, kill latency, quota behavior or throughput.
Part A remains pending real smoke and independent review. A passing five-question
smoke proves integration only, **no benchmark number**. Stop if defaults must
change, auth handling is needed, or this lane exceeds 400 non-test lines.
