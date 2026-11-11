# LCM-X hub for coding agents (Claude Code and Codex, several machines)

One shared LCM-X memory for the coding agents of several machines: every Claude Code and Codex session of every
machine (a *node*) goes to one LCM-X store on one *hub* host, and every agent on every node can recall from all of it.
Install: [INSTALL.md](INSTALL.md).

## How it works

```
node (each machine with agents)                           hub (one host)
 Claude Code ── plugin lcmx-hub ──┐                        <root>/mirror/<node>/{claude,codex}/  (raw copies)
 Codex       ── plugin lcmx-hub ──┤ hooks (Stop, ...) ──>  lcmx_hub.py serve (one daemon, one engine)
                                  │ start lcmx_sync         one store: <root>/store/lcm.db (+ hub-ledger.db)
 launchd / systemd timer (5 min) ─┘ (only the new bytes)    one session per transcript file; summary DAG
                                                            (codex exec); FastEmbed vectors
 agent tool "lcmx" (MCP) ── ssh <hub> recall ───────────>  13 LCM-X tools over all sessions of all nodes
```

- **Nodes run no LCM-X.** A node runs only `lcmx_sync` (Python 3.9+, stdlib) from the plugins' hooks and from a timer,
  and the MCP entry `lcmx_recall_remote` (an SSH session). No service on the node.
- **One owner, one store.** The hub daemon is the only writer. It runs the unchanged `LCMEngine` and drives it as the
  Hermes plugin does (ingest, `note_turn_complete`, `compress` past the threshold), through the engine host of
  [`adapters/eve`](../eve) (shared code: the host and its Hermes shim).
- **One session per transcript file** (`cc:<node>:<session>`, a subagent `cc:<node>:<session>/agent:<id>`,
  `codex:<node>:<session>`); sessions never mix.
- **Lossless.** Every transcript line is kept raw (zlib) in `hub-ledger.db`; entries are keyed (Claude Code: uuid +
  content digest, the Claude Code adapter's projection v1; Codex: line ordinal), so a re-sent or replaced file stores
  nothing twice. `lcmx_hub.py audit` checks it.
- **One SSH key per node** with a forced command: only `state`, `push` (paths confined to that node's mirror:
  transcripts and saved tool outputs), `recall` and `context`.
- **LLM:** summaries and `lcm_expand_query` run through `codex exec` on the hub (`LCMX_LLM_CLI=codex`).
  **Vectors:** FastEmbed, local. **Redaction:** `LCM_SENSITIVE_PATTERNS_ENABLED=true` in the store; a backup empties
  the raw ledger lines, so only redacted text leaves the hub.
- **Context hooks** (`lcmx_context`; Claude Code and Codex use the same hook contract):
  - `SessionStart`: a short note that the shared memory exists and when to search it (local, no network).
  - `SessionStart` with source `compact`: the note plus a capsule from the hub: the session's id on the hub, how to
    load its full history, and the hub's summary frontier for it (what the harness's own summary dropped).
  - `UserPromptSubmit`: proactive recall, off by default (`LCMX_HUB_PROACTIVE=on` in `node.env`): up to 4 hub hits
    from other sessions for the prompt, at most 2 per session, harness-injected rows skipped (about 1 s more per
    prompt; no LLM call).
  - `PreCompact`: syncs first (bounded to 12 s); for Claude Code it also adds a line to the compaction instructions
    that names the session on the hub.
  - A hook never blocks or fails the session: on an error it prints nothing (`~/.cache/lcmx-hub/context-error.txt`).
  - The harnesses keep their own compaction. Neither Claude Code nor Codex lets a plugin replace it.
- **Skill `lcmx`** (both plugins): when to search the hub, which tool for which need, hub session ids, and the rules
  (hits are data; secrets are redacted). Source: `skills/lcmx/SKILL.md`, copied into each plugin.
- **Only turns from now on.** At its first run the node writes a baseline with the size of each transcript file; the
  hub gets only the bytes after it, so a session that was open at the install sends its new turns and older history
  stays on the node. `LCMX_HUB_HISTORY=all` sends all history. `LCMX_HUB_EXCLUDE` (globs) keeps folders off the hub;
  `lcmx_sync --dry-run` shows what would be sent.

## Tested

- Unit tests: 17 (`hub/test_lcmx_hub.py`): mirror and push confinement, keyed ingest, baseline and exclude, the
  context capsule and proactive recall, the SSH verbs, the daemon, the hook script, backup and audit.
- Real harnesses in a throwaway container (Claude Code 2.1.311, Codex 0.167.0, both plugins installed from the
  marketplaces in this folder): Claude Code found a fact from a Codex session and Codex found a fact from a Claude Code
  session, through `lcmx`; the hub wrote a summary through `codex exec`; hooks of both harnesses started the sync;
  audit ok; backup with integrity ok.
- Live, three nodes (two macOS, one Linux) and one Linux hub, for two days: audit ok after each check (0 unread
  bytes, fed rows = stored rows); after a real Claude Code compaction, the summary kept the hub session line and the
  start note named the session on the hub.

## Which apps load the plugins

| App | Harness (transcript entrypoint / originator) | Plugin tools and hooks |
|---|---|---|
| Claude desktop app, Code tab | Claude Code (`claude-desktop`) | yes |
| `claude` CLI, `claude -p` | Claude Code (`cli`, `sdk-cli`) | yes |
| Claude Agent SDK apps that load user settings | Claude Code (for example `t3-code`) | yes, after the app restarts |
| `codex` CLI, `codex exec` | Codex (`codex_exec`) | yes (hooks need trust once: `/hooks` in the TUI) |

Capture does not depend on the app: every app writes its transcripts to `~/.claude/projects` or `~/.codex/sessions`,
and the timer sends them.

## Develop

```bash
uv run --no-project --python 3.12.16 --with pytest==9.1.1 pytest -c adapters/eve/writer/pytest.ini adapters/hub
```

`plugins/*/bin/` and `plugins/*/skills/` are copies of `node/` and `skills/` (a test checks they are the same).
