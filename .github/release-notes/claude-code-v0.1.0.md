# LCM-X for Claude Code 0.1.0 (preview)

LCM-X for Claude Code keeps a lossless, local record of your Claude Code sessions and lets the agent search it. This is a preview: you opt in per launch, and it changes nothing until you do.

## What it does

- **Captures every session.** Each message, tool call and tool result of your Claude Code sessions is written to a local LCM-X store, including the large tool outputs Claude Code saves to files, subagent sessions, resumed sessions and forks. Nothing is summarized away or dropped: the store keeps every transcript line, and an audit command proves it.
- **Lets the agent recall it.** The plugin adds five read-only tools: `lcm_grep` (exact and full-text search), `lcm_recall`, `lcm_expand` (returns a stored message or tool output in full), `lcm_load_session` and `lcm_describe`. They work across sessions, so the agent can find what an earlier session said or ran.
- **Leaves Claude Code alone.** Claude Code's own compaction is unchanged. The plugin never blocks a turn, never rewrites the conversation and adds nothing to the context by itself; the agent reads memory only when it calls a tool. The hooks print nothing and hand the work to a background worker.

## How to opt in

You need Claude Code 2.1.139 or later, `python3` 3.9 or later on `PATH` (the macOS system Python is fine), and Python 3.11 or later somewhere on `PATH` for capture and recall (for example from Homebrew or python.org). macOS and Linux are supported.

1. Download `lcm-x-claude-code-0.1.0.tar.gz` and its `.sha256` from this release, then check and unpack it:
   ```bash
   shasum -a 256 -c lcm-x-claude-code-0.1.0.tar.gz.sha256   # Linux: sha256sum -c
   tar -xzf lcm-x-claude-code-0.1.0.tar.gz
   ```
2. Start Claude Code with the plugin:
   ```bash
   claude --plugin-dir "$PWD/lcm-x-claude-code-0.1.0"
   ```
   Capture starts with the session. The store lives in `$HOME/.lcm-x/claude-code` (set `LCMX_CC_HOME` to move it), outside the plugin folder, so removing or updating the plugin never removes your memory.
3. Without a Python 3.11 or later, capture waits and records `runtime_missing`; nothing is lost while it waits, because Claude Code's own transcripts stay the source. One step fixes it: install Python 3.11 or later, or run `uv venv --python 3.11 $HOME/.lcm-x/claude-code/venv`.

To check what was captured: `python3.11 lcm-x-claude-code-0.1.0/lcm-x/scripts/lcm_x_capture.py audit --home $HOME/.lcm-x/claude-code` (any Python 3.11 or later) prints counts and ids only, and says whether every line is accounted for.

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
- **A subagent transcript removed before the worker first reads it** cannot be captured.
- **Hook timing under heavy load.** Hooks hand off in milliseconds on an idle machine, but on a heavily loaded machine an asynchronous hook took up to about 0.4 s in our runs. It never blocks a turn. A quiet-host measurement is tracked in #2141.
- **Leftover temporary files.** A hook that is killed mid-write can leave an empty `.tmp` file in the store's `spool` folder. It holds nothing and is safe to delete; capture is unaffected.

## What it bundles

Unit version 0.1.0, tag `claude-code-v0.1.0`, with LCM-X core vendored from the tagged commit. The asset's `bundle.json` records the full core commit and core version, and the release workflow adds them as the last line of these notes.

## Release evidence

The unit gate ran on the tagged commit: ten real Claude Code sessions (multi-turn tool use, resume, forks, manual and automatic compaction, parallel subagents, four parallel sessions, killed workers and sessions, reruns, cross-session recall and a 24-hour lane), each audited with 0 lost and 0 duplicated lines; a growth check on a 670,000-line transcript; and the CLI smoke in both runtime modes. The result card is attached to this release.
