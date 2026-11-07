# LCM-X for Claude Code 0.1.0 (preview)

LCM-X for Claude Code keeps a lossless, local record of your Claude Code sessions and lets the agent search it. This is a preview: you opt in per launch, and it changes nothing until you do.

## What it does

- **Captures every session.** Each message, tool call and tool result of your Claude Code sessions is written to a local LCM-X store, including the large tool outputs Claude Code saves to files, subagent sessions, resumed sessions and forks. Nothing is summarized away or dropped: the store keeps every transcript line it reads, and an audit command checks each captured line against its transcript file (see Known issues for the two cases it cannot see).
- **Lets the agent recall it.** The plugin adds five read-only tools: `lcm_grep` (exact and full-text search), `lcm_recall`, `lcm_expand` (returns a stored message or tool output in full), `lcm_load_session` and `lcm_describe`. They work across sessions, so the agent can find what an earlier session said or ran.
- **Leaves Claude Code alone.** Claude Code's own compaction is unchanged. The plugin never rewrites the conversation and adds nothing to the context by itself; the agent reads memory only when it calls a tool. The hooks print nothing: most run in a few milliseconds and hand the work to a background worker. Before Claude Code compacts, the `PreCompact` hook drains the session into the store first, which can delay compaction by up to 10 seconds (about 0.1 to 1.1 seconds in our runs).

## How to opt in

You need Claude Code 2.1.139 or later, `python3` 3.9 or later on `PATH` (the macOS system Python is fine), and Python 3.11 or later somewhere on `PATH` for capture and recall (for example from Homebrew or python.org). macOS and Linux are supported.

1. Download `lcm-x-claude-code-0.1.0.tar.gz` and its `.sha256` from this release, then check and unpack it:
   ```bash
   shasum -a 256 -c lcm-x-claude-code-0.1.0.tar.gz.sha256 && tar -xzf lcm-x-claude-code-0.1.0.tar.gz
   # Linux: sha256sum -c lcm-x-claude-code-0.1.0.tar.gz.sha256 && tar -xzf lcm-x-claude-code-0.1.0.tar.gz
   ```
2. Start Claude Code with the plugin:
   ```bash
   claude --plugin-dir "$PWD/lcm-x-claude-code-0.1.0"
   ```
   Capture starts with the session. The store lives in `$HOME/.lcm-x/claude-code` (set `LCMX_CC_HOME` to move it), outside the plugin folder, so removing or updating the plugin never removes your memory.
3. Without a Python 3.11 or later, capture waits and records `runtime_missing`. Nothing is lost while it waits as long as Claude Code keeps its transcripts (its default; see the local transcript GC under Known issues), because they stay the source and capture reads them once a runtime is there. One step fixes it: install Python 3.11 or later, or run `uv venv --python 3.11 $HOME/.lcm-x/claude-code/venv`.

To check what was captured, run the audit with any Python 3.11 or later (here `python3.12`): `python3.12 lcm-x-claude-code-0.1.0/lcm-x/scripts/lcm_x_capture.py audit --home "${LCMX_CC_HOME:-$HOME/.lcm-x/claude-code}"`. It prints counts and ids only, and says whether every captured line is accounted for (`ok`) and nothing is pending (`verified`).

To turn capture off for a launch, set `LCMX_CAPTURE=off`, or start Claude Code without `--plugin-dir`.

## What it does not do yet

- **No compaction or context injection.** Recall is on demand, through the tools; memory is not added to the context automatically.
- **Not searchable yet:** thinking blocks, attachments (`@`-file contents, plans, hook-injected context) and bookkeeping entries are kept in the store's ledger but are not indexed for search.
- **Vector recall needs a setup step:** FastEmbed installed in the venv above, with its model already downloaded. Without it, recall uses full-text search. The plugin never downloads a model.
- **No forget command** for Claude Code sessions yet.
- **One store per harness.** It does not share a store with the Hermes plugin; Hermes and Claude Code memories are separate in this release.
- **Grok Build** sessions are not captured.
- **No plugin marketplace entry** yet: you pass `--plugin-dir` on each launch.

## Known issues

- **Claude Code's local transcript GC.** Lossless capture needs Claude Code's opt-in local transcript GC off (`CLAUDE_CODE_TRANSCRIPT_LOCAL_GC` unset, the default). With it on, capture is best effort and the audit lists those sessions without a lossless claim.
- **A subagent transcript removed before the worker first reads it** cannot be captured, and the audit cannot see that it existed.
- **Audit memory.** The audit needs memory roughly in proportion to the largest session in the store: about 1.2 GB for a 625,000-line session. Capture itself stays under about 400 MB.
- **Hook timing under heavy load.** Hooks usually hand off in milliseconds. Under heavy disk and memory pressure from other work, the hook's write of its small pointer file can stall:
  - In one full run of the release gate, 8 of 670 hook runs took over 2 s. The slowest took 8,341 ms, at a 1-minute load of about 20 (the runs with stalls started at loads of 20 to 37), and each delay was in that write.
  - In a rerun with no other gate job on the machine, the slowest of 54 hooks took 526 ms.

  `PostToolBatch` runs in the background. The other hooks run before Claude Code goes on, with a 5-second limit (15 seconds for `PreCompact`), so a stall there adds to the turn.

  No audit lost anything. If a hook hits its time limit, that turn is captured by the next hook or the next time Claude Code starts. A kill at Claude Code's own hook timeout was not observed directly in the gate. Tracked in #2350.
- **Leftover temporary files.** A hook that is killed mid-write can leave an empty `.tmp` file in the store's `spool` folder. It holds nothing and is safe to delete; capture is unaffected.

## What it bundles

Unit version 0.1.0, tag `claude-code-v0.1.0`, with LCM-X core vendored from the tagged commit. The asset's `bundle.json` records the full core commit and core version, and the release workflow adds them as the last line of these notes.

## Release evidence

The unit gate ran on the tagged commit: ten real Claude Code sessions (multi-turn tool use, resume, forks, manual and automatic compaction, parallel subagents, four parallel sessions, killed workers and sessions, reruns, cross-session recall, and a driven 32-turn session with idle gaps and compactions). Each was audited with 0 lost and 0 duplicated lines. The gate also ran a growth check on one real transcript of about 690,000 lines, the CLI smoke in both runtime modes, and install records on macOS and Linux. The result card is attached to this release.
