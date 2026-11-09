# Recall over MCP: the #2225 gate kit

Measures whether a Claude Code agent finds what was said in earlier Claude Code sessions through the
read-only `lcm-x` MCP reader, and whether anything the capture store holds goes missing on the way
through the endpoint. Gate item for Claude Code GA (#2225).

## Protocol

- **Corpus:** 8 real Claude Code sessions (S1-S8, including a resumed session, a fork, a compacted
  session and a session with subagents) in a fixture project, captured by the plugin into one store.
  40 invented facts, each a unique value, planted across eight classes: user text 8, assistant-stated
  derived values 6, tool inputs read back 6, inline tool results 8, large tool outputs that Claude Code
  saves to `tool-results/` 4, before a compaction 4, inside a subagent 2, in a resumed or forked
  session 2. Plus 4 superseded values (stated, then changed in a later session). All facts are
  invented, so the per-question outputs can be published. `corpus.json` holds the sessions and facts.
- **Questions** (`questions.json`, fixed before any run): 40 positives that name the topic but never
  the value, 4 latest-value questions, 6 negatives about topics never discussed.
- **Arm D, direct:** read-only SQL over the store. Is each value in a row of the expected session and
  role (inline or in its externalized payload)? It also checks that each value reached the session's
  own Claude Code files (`plant_failed`), and records per session the compactions, subagent
  transcripts and saved outputs that really happened. A pre-compaction value counts as covered only
  when a compaction boundary follows the line that planted it.
- **Arm T, endpoint:** the reader started as the plugin's `.mcp.json` starts it. `lcm_grep`
  (full text) must find each value in the expected session, and paged `lcm_expand` must return content
  byte-equal (sha256) to the stored row or payload.
- **Arm A, agent:** headless Claude Code with hooks off, built-in tools off and only the `lcm-x`
  server, one fresh session per question. The reader prompt is fixed in `kit.py` (`PROMPT`).
  Deterministic judge (`judge`): a positive is correct only if the answer holds the gold value and no
  other value-shaped string; a negative only if the answer is exactly `NOT FOUND`, and any other
non-empty answer to a negative counts as a false recall. Arm T also counts a
  value found in a saved tool output (a payload hit), expanded by its ref and byte-equal to the payload.

## Bars (set before any data)

| Bar | Pass |
|---|---|
| Empty-store control (first) | T returns nothing for all 44 values, and A scores 0/40 positives, on an empty store. Otherwise the run is void. |
| Capture | The capture audit is ok and verified on all 8 sessions. |
| D | 44/44 (40 facts + 4 current values). A miss is a capture finding: stop. |
| T | 44/44, byte-equal expand. A miss is endpoint loss and blocks release. |
| A | At least 36/40 positives. |
| False recall | At most 1/6 negatives name a value or invent one. |
| T-pass / A-fail | Each such question is classified in the receipt. |
| Latest values | Recorded and classified (current/stale/both/miss), not gated. |
| Latency | Not gated. p50/p90/max per tool. A T p90 above 1 s, or any T call above 5 s, needs a written classification before GA. |

If the first A run scores 34-38 positives, a second A run on the same store decides, and the lower
score governs. Old superseded values are checked by D and T but reported apart, not gated. An
infrastructure error in A (timeout, error exit, empty answer, a limit) gives `INCOMPLETE`, not a
result. Verdict precedence: VOID, STOP, FAIL (T), INCOMPLETE, FAIL (other bars), SECOND RUN
REQUIRED, FAIL (A), CLASSIFICATION REQUIRED, PASS.

## Running it

Run from the repository root. `KIT` is this folder (`bench/instruments/cc_mcp_recall`). `CMD` is a
launcher for the Claude Code CLI that runs it with `HOME="$KIT_HOME"` and
`CLAUDE_CONFIG_DIR="$KIT_HOME/.claude"` (the kit reads transcripts from `$KIT_HOME/.claude/projects`),
and passes `KIT_LCMX_CC_HOME` through as the reader's `LCMX_CC_HOME`. `PLUGIN` is the unpacked Claude
Code unit asset. `EMPTY`, `READER`, `CAPTURE` and `OUT` are fresh directories.

Stop after the control if `t-empty` finds anything or `a-empty` names a value (the run is void), and
after arm D if any value is missing (a capture finding); the scorer reports both, but the later arms
would only spend model calls.

```bash
python "$KIT/kit.py" make-empty-store --home "$EMPTY/.lcm-x/claude-code"
python "$KIT/kit.py" arm-t --empty --plugin "$PLUGIN" --home "$EMPTY" --d "$KIT/corpus.json" --out "$OUT/t-empty"
python "$KIT/kit.py" arm-a --claude "$CMD" --plugin "$PLUGIN" --store-home "$EMPTY/.lcm-x/claude-code" \
  --reader-home "$READER/control" --only positive --parallel 3 --out "$OUT/a-empty"
python "$KIT/kit.py" build-corpus --claude "$CMD" --plugin "$PLUGIN" --home "$CAPTURE" --out "$OUT/corpus"
python "$KIT/kit.py" arm-d --home "$CAPTURE" --manifest "$OUT/corpus/corpus-manifest.json" --out "$OUT/d"
python "$KIT/kit.py" arm-t --plugin "$PLUGIN" --home "$CAPTURE" --d "$OUT/d/arm-d.json" --out "$OUT/t"
python "$KIT/kit.py" arm-a --claude "$CMD" --plugin "$PLUGIN" --store-home "$CAPTURE/.lcm-x/claude-code" \
  --reader-home "$READER/main" --parallel 3 --out "$OUT/a"
python "$KIT/kit.py" score --corpus-manifest "$OUT/corpus/corpus-manifest.json" --d "$OUT/d/arm-d.json" \
  --t "$OUT/t/arm-t.json" --t-empty "$OUT/t-empty/arm-t.json" --a "$OUT/a/arm-a.json" \
  --a-empty "$OUT/a-empty/arm-a.json" --out "$OUT/score"
```

When the first A run scores 34-38, run arm A again on the same store and pass it as `--a2`. A
T-pass/A-fail row, or a latency tripwire, needs a written classification: pass `--classifications`
with a JSON object of question id to text (and `"latency"` to text). The scorer copies its inputs
beside `RESULT.md` and `RESULT-MANIFEST.json` so a result can be re-scored. The tests (`tests/test_cc_mcp_recall_kit.py`) run no model and no Claude Code.

## What it does not prove

Invented facts in a fixture project, not anyone's real history. One model as the agent. One store of
8 sessions, so nothing about recall at a large store size. No Grok Build sessions (#2226). Latency on
a shared machine. It is not a customer-ready claim. A value that appears only in a tool call's input
(not echoed in any result) is not covered: the tool-input facts here are read back by the agent.
