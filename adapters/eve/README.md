# LCM-X for Eve (`lcmx-eve`)

LCM-X for Eve agents, 1:1 with LCM-X under Hermes where Eve allows it: the real LCM-X engine (upstream,
unchanged, vendored at a pinned commit) runs beside each Eve agent and is driven the way the Hermes plugin
drives it. Built for `eve@1.0.0-rc.6` exactly. Install: [INSTALL.md](INSTALL.md).

## Parity with Hermes

| LCM-X under Hermes | Eve agent with this package |
|---|---|
| Lossless store of every message (exactly-once ingest) | Same engine ingest. Plus a raw ledger of every message Eve showed |
| Summary DAG (leaf summaries, condensation, escalation) | Same, on the engine's own view of each session |
| Auxiliary LLM for summaries and `expand_query` (`call_llm`, default: the agent's own model) | Same: `call_llm` goes over stdio to the Eve process and runs on the agent's model (`model`, or `summaryModel`). Separate background calls, never inside a turn |
| Tools: 13 by default, 3 behind flags | Same 13 as `lcmx__<name>`; the flags work the same (`LCM_*` env) |
| Vector recall (FastEmbed local, or an API provider) | OpenAI embeddings by default (key for embeddings only), caught up between turns; FastEmbed optional |
| LCM-X builds the model's context window (compaction, assembly) | **Not possible**: Eve rc.6 has no hook. Eve compacts its own window; LCM-X compacts its own view |
| Proactive recall at `pre_llm_call` | `recall["turn.started"]`: Eve calls it before every turn (also scheduled runs). The best hits for the turn's user text (at most 5, 2 per session; also earlier, compacted parts of the current session; text the model still sees is left out, the same text only once), one message replaced each turn |

## How it works

```
Eve agent process                              <root>/scopes/<scope id>/
 agent/memory/lcmx.ts (defineLcmxMemory)
  capture turn.completed / compaction.requested ──> spool/*.json (durable before the handler returns)
  tools(): lcmx__recall, __grep, __expand, ...  ──┐
                                                  ▼
 child: host/lcmx_eve_host.py serve (Python, one per store, holds host.lock)
   plan_record(): align the record with what it saw (generation, position, Eve's compaction tail)
   shadow += new messages → engine.ingest(shadow); note_turn_complete()   (Hermes post_llm_call)
   engine.should_compress → shadow = engine.compress(shadow)              (Hermes before a model call)
   tools → engine.handle_tool_call()                                      ──> lcm.db (only the engine writes)
   every message Eve showed, raw; views; shadows                          ──> eve-ledger.db
```

- **Capture never fails Eve.** Both capture handlers write a spool record (temporary file, fsync, rename), then
  nudge the host. They never throw and return within 15 s even when the disk stalls.
- **Identity across Eve's compaction.** Eve hands over the full history with no message ids. A message is
  (session, generation, position); a history rewrite starts a new generation, and the tail Eve kept (also capped
  tool results, text-only assistant copies, the replayed user message) is matched and not fed again. Eve's own
  marker and summary stay raw in the ledger; the engine makes its own summaries, as under Hermes.
- **One writer per store.** The host holds `host.lock` while it runs. A second Eve process spools only; the
  holder drains its records (it scans the spool every 2 s). `lcmx-eve drain` (for a timer) exits when the lock
  is held.
- **Fixed namespace.** `defineLcmxMemory({ agent })` sets the namespace `lcmx:<agent>`. Eve's default local
  namespace hashes the build folder, so a rebuild would give a new, empty store.

## Develop

```bash
cd adapters/eve && npm ci && node scripts/vendor-core.mjs
LCMX_PYTHON=$(uv python find 3.12.16) npm test
uv run --no-project --python 3.12.16 --with pytest==9.1.1 pytest -c adapters/eve/writer/pytest.ini adapters/eve/writer adapters/eve/host
npm pack            # prepack builds dist/ and vendors core/
```

- `e2e/`: Eve rc.6 test agent with `mockModel` (compaction, tool call, recall through `lcmx__grep`).
- `e2e-real/`: the same with a real OpenAI model (`OPENAI_API_KEY`), two sessions in two server processes.
- After a rebase on upstream: `node scripts/vendor-core.mjs <commit>` and `python -I scripts/gen_engine_tools.py`.

## Limits

- LCM-X does not build Eve's context window (no Eve hook). At turn start it adds one recall message (Eve's memory-slot
  recall; bounded to 8 s, never throws; option `recall: false` turns it off).
- The bridge needs the Eve process: a timer drain stores messages but writes no summaries.
- Not yet tested with a real `chatgpt()` subscription model (only with `mockModel` and the OpenAI API).
- Subagent sessions are stored as their own sessions, without a parent link.
- Sessions from before the install: `lcmx-eve import` (INSTALL) copies only their text (no tool calls, reasoning
  or attachments). A message with no matching Eve event takes the time of the message before it.
- After `clear()`, an exact repeat of the last exchange that is not the new turn's input may be taken as carried.
- `writer/lcmx_eve_writer.py` still has its own store mode (r1, Claude Code unit pattern); the package uses only
  its planning functions.
