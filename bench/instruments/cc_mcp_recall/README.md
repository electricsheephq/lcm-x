# Recall over MCP: the #2225 gate kit

The bars live in **BARS-2225-mcp-recall-v1.md**, approved before any run
and held by the orchestrator. That approved protocol is authoritative; this kit
does not change its bars. Use the GLM gate launcher, never Anthropic billing.
Only invented fixture facts and synthetic answers are written by this kit.

The eight session labels include a resumed session and a fork. D checks all
forty facts and both tokens of each superseded pair: forty-eight tokens,
aggregated into forty-four positive/latest question rows. The empty T control
queries forty facts and the four current values, exactly forty-four calls.

* Capture must audit ok and verified after each session, or the run stops.
* D reads the expected session and allowed roles directly with read-only SQL.
  A subagent fact can belong to the parent's recorded subagent children.
* T keeps one MCP stdio connection, searches tokens, and expands paged content
  and externalized payloads. Its content hash must match D's full content hash.
* A starts a fresh reader session and working directory per question, with
  hooks disabled, built-in tools off, and only the connected lcm-x MCP server.
  Connection failures remain explicit errors, never ordinary recall misses.

Run from the repository root. Set KIT to this folder, CMD to the orchestrator's
GLM launcher, PLUGIN to the unpacked plugin, and CAPTURE, EMPTY, READER and OUT
to fresh absolute scratch/evidence directories. The launcher must isolate HOME
and CLAUDE_CONFIG_DIR using KIT_HOME; it must pass KIT_LCMX_CC_HOME to its reader.
The orchestrator checks the owner's settings hash before and after batches;
the kit never reads that file or any authentication file.

```bash
python "$KIT/kit.py" --help
# Empty-store controls FIRST; no corpus build or measured arm yet.
python "$KIT/kit.py" make-empty-store --home "$EMPTY/.lcm-x/claude-code"
python "$KIT/kit.py" arm-t --plugin "$PLUGIN" --home "$EMPTY" \
  --d "$KIT/corpus.json" --empty --out "$OUT/t-empty"
python "$KIT/kit.py" arm-a --claude "$CMD" --plugin "$PLUGIN" \
  --store-home "$EMPTY/.lcm-x/claude-code" --reader-home "$READER/control" \
  --only positive --out "$OUT/a-empty"
# Proceed only if all empty T checks pass and empty A scores no positives.
python "$KIT/kit.py" build-corpus --claude "$CMD" --plugin "$PLUGIN" \
  --home "$CAPTURE" --out "$OUT/corpus"
python "$KIT/kit.py" arm-d --home "$CAPTURE" \
  --manifest "$OUT/corpus/corpus-manifest.json" --out "$OUT/d"
python "$KIT/kit.py" arm-t --plugin "$PLUGIN" --home "$CAPTURE" \
  --d "$OUT/d/arm-d.json" --out "$OUT/t"
python "$KIT/kit.py" arm-a --claude "$CMD" --plugin "$PLUGIN" \
  --store-home "$CAPTURE/.lcm-x/claude-code" --reader-home "$READER/main" \
  --parallel 3 --out "$OUT/a"
python "$KIT/kit.py" score --corpus-manifest "$OUT/corpus/corpus-manifest.json" \
  --d "$OUT/d/arm-d.json" --t "$OUT/t/arm-t.json" \
  --t-empty "$OUT/t-empty/arm-t.json" --a "$OUT/a/arm-a.json" \
  --a-empty "$OUT/a-empty/arm-a.json" --out "$OUT/score"
```

D misses are capture findings: stop and report. T misses block release.
A needs at least thirty-six positives and at most one fabricated negative.
When the first A run scores thirty-four through thirty-eight positives, run A
again on the same store using a fresh reader home and supply `--a2`; the lower
positive score and worse false-recall count govern. Latest values are classified
current/stale/both/miss, and are not gated. Fill each empty `classification`
field for T-pass/A-fail rows. Latency is ungated, but p90 above one second or
any T call above five seconds requires a written classification before GA.

The scorer snapshots its JSON inputs beside RESULT.md and RESULT-MANIFEST.json.
Preserve them for re-scoring; publication and GA decisions belong to the
orchestrator. Missing controls void the result. Limit errors stop new calls;
already-running parallel calls may finish. No real sessions are run by tests.

## What it does not prove

Synthetic facts in a fixture project, not the owner's real history; one model
(the GLM plan) as the reader; one store of 8 sessions, so nothing about recall
at a large store size; no Grok Build sessions (that is #2226); latency on a
shared Mac. Not a "customer ready" claim.
