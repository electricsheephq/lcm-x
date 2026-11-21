# Install LCM-X in an Eve agent

This guide installs `lcmx-eve` into one Eve agent. The agent then has LCM-X as Hermes has it:
a lossless store of every session, the summary DAG (written by the agent's own model), vector recall and the LCM-X tools
(`lcmx__recall`, `lcmx__grep`, `lcmx__expand`, `lcmx__expand_query`, `lcmx__describe`, `lcmx__load_session`,
`lcmx__recent`, `lcmx__compute`, `lcmx__compile_evidence`, `lcmx__evidence_pack`, `lcmx__status`,
`lcmx__inspect`, `lcmx__doctor`). Eve keeps its own context window and compaction (Eve has no hook to replace
them).

## Requirements

- Node.js 24 or newer, and `eve@1.0.0-rc.6` in the agent.
- `uv` on the PATH. Setup uses it to install CPython 3.12.16. Without `uv`, give a Python 3.11+ whose SQLite is
  3.51.3 or newer (`--python`).
- The agent's model, for example `chatgpt("gpt-6.1-terra")` (ChatGPT subscription). LCM-X's summaries and
  `lcmx__expand_query` run on it, in separate background calls, as Hermes runs them on its auxiliary model.
- An OpenAI platform key for embeddings only (`OPENAI_API_KEY`). Use one key for each machine. LCM-X never
  uses this key for chat.

## Steps

1. Install the package in the agent app:

   ```bash
   npm install --save-exact /path/to/lcmx-eve-<version>.tgz
   ```

2. Set up the runtime. Use a folder outside the app and outside git, one folder for each agent:

   ```bash
   npx lcmx-eve setup --root /srv/agents/scout/.lcm-x
   ```

   The command writes `<root>/runtime.json`. The output must show `"ok": true` and `"tools": 13`. Add
   `--fastembed` only if you use local FastEmbed vectors instead of OpenAI embeddings.

3. Put the agent's model in one module, and use it in `agent.ts` and in the memory slot:

   ```ts
   // agent/lib/model.ts
   import { chatgpt } from "eve/models/openai";
   export const model = chatgpt("gpt-6.1-terra");
   ```

   ```ts
   // agent/agent.ts: model, reasoning and the rest as before
   import { defineAgent } from "eve";
   import { model } from "./lib/model";
   export default defineAgent({ model, reasoning: "high" });
   ```

   ```ts
   // agent/memory/lcmx.ts
   import { defineLcmxMemory } from "lcmx-eve";
   import { model } from "../lib/model";

   export default defineLcmxMemory({ agent: "scout", root: "/srv/agents/scout/.lcm-x", model });
   ```

   - `agent` sets the fixed memory namespace `lcmx:scout`. Do not leave it out: Eve's default namespace changes
     when the app is built again, and the agent then gets a new, empty store.
   - `model`: LCM-X's own calls (summaries, `expand_query`) run on it with reasoning effort `low`
     (`summaryReasoningEffort`). Each call is separate: LCM-X's prompt only, no agent history, no tools, nothing in
     the agent's session, one at a time. Summaries run only between turns, never inside one.
   - `summaryModel`: optional, another model for LCM-X's calls (Hermes: the auxiliary model).
   - `embeddings`: default `{ provider: "openai" }` (`text-embedding-3-small`, 512 dims int8, key from
     `OPENAI_API_KEY`). Other values: `"fastembed"`, `"off"`.

4. Set the environment of the Eve process:

   | Variable | Value |
   | --- | --- |
   | `OPENAI_API_KEY` | the machine's key, for embeddings only (another name: `embeddings.apiKeyEnv`) |
   | `LCMX_CONTEXT_LENGTH` | the window LCM-X compacts its own view for, default `200000` (or `contextLength`) |

   For `chatgpt()` on Linux, Eve needs the ChatGPT login: Codex on the `PATH`, or `secret-tool`, a session D-Bus and
   an unlocked keyring (Eve docs, "ChatGPT subscription models").

5. Add a timer that drains the spool when the agent is stopped. Example for systemd (user units):

   ```ini
   # ~/.config/systemd/user/lcmx-scout-drain.service
   [Service]
   Type=oneshot
   ExecStart=/usr/bin/env npx --prefix /srv/agents/scout lcmx-eve drain --root /srv/agents/scout/.lcm-x

   # ~/.config/systemd/user/lcmx-scout-drain.timer
   [Timer]
   OnUnitActiveSec=10min
   OnBootSec=2min
   [Install]
   WantedBy=timers.target
   ```

   While Eve runs, its LCM-X host holds the store and drains at once. The timer then reports `"lock": "held"`
   and does nothing. A timer drain only stores messages; it writes no summary (no model without Eve). The next
   live turn writes the summaries.

6. Check the agent:

   ```bash
   npx lcmx-eve check --root /srv/agents/scout/.lcm-x
   npx lcmx-eve audit --root /srv/agents/scout/.lcm-x
   ```

   `audit` must show `"ok": true` for each store: no missing message, the same number of rows fed and stored,
   an empty spool.

## What is in the root folder

| Path | What |
|---|---|
| `runtime.json` | Python, core and host paths (from `setup`) |
| `venv/` | Python with FastEmbed (from `setup --fastembed`); do not back it up |
| `scopes/<id>/lcm.db` | the LCM-X store (only the engine writes it) |
| `scopes/<id>/eve-ledger.db` | raw copy of every message Eve showed, and the views and shadows of the sessions |
| `scopes/<id>/spool/` | capture records not yet drained |
| `scopes/<id>/logs/` | `host.log`, `writer.jsonl`; `<root>/logs/provider.jsonl` |

Back up `scopes/` (both `.db` files with an SQLite online backup, or with the agent stopped).

## Upgrade

Install the new package version, then run `setup` again. The store stays. The package pins its LCM-X core
commit in `core/BUNDLE.json`.

## Import sessions from before the install

An agent that ran on Eve before LCM-X has sessions that the store does not have. Copy their conversations in once,
as the agent user, while the agent's Eve server runs:

```bash
npx lcmx-eve import --root /srv/agents/scout/.lcm-x --eve-url http://127.0.0.1:<eve port> \
  --bearer-file <file with the Eve HTTP bearer> --namespace lcmx:scout \
  --workflow-data <agent app>/.eve/.workflow-data --before <install time, ISO>
```

Each Eve session becomes its own LCM-X session `import-<eve session id>`: user messages and completed replies (no
tool calls, reasoning or attachments). The running Eve's engine host stores them within seconds; when Eve is
stopped, run `lcmx-eve drain`. A second run stores nothing new for a session that did not grow. After the spool is
drained, the import sets each imported message's original time (Eve's event time of the delivery or the reply;
`retime` in the last output line). The engine itself stores the ingest time. `retime` changes only the
`timestamp` column of the rows of `import-*` sessions, and only when a session's row count matches its transcript.
