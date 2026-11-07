# LCM-X for Claude Code 0.1.0 (preview)

LCM-X for Claude Code keeps a lossless, local record of your Claude Code sessions and lets the agent search it. This is a preview: you opt in per launch, and it changes nothing until you do.

## What it does

- **Captures every session.** Each message, tool call and tool result of your Claude Code sessions is written to a local LCM-X store, including the large tool outputs Claude Code saves to files, subagent sessions, resumed sessions and forks. Nothing is summarized away or dropped: the store keeps every transcript line it reads, and an audit command checks each captured line against its transcript file (see Known issues for the two cases it cannot see).
- **Lets the agent recall it.** The plugin adds five read-only tools: `lcm_grep` (full-text search, including exact phrases in quotes), `lcm_recall`, `lcm_expand` (returns a stored message or tool output in full, paging through long ones), `lcm_load_session` and `lcm_describe`. They work across sessions, so the agent can find what an earlier session said or ran.
- **Leaves Claude Code alone.** Claude Code's own compaction is unchanged. The plugin never rewrites the conversation and adds nothing to the context by itself; the agent reads memory only when it calls a tool. The hooks print nothing: most run in a few milliseconds and hand the work to a background worker. Before Claude Code compacts, the `PreCompact` hook drains the session into the store first. That can delay compaction: the drain stops after 10 seconds, and Claude Code stops the hook at its 15-second limit. In the gate runs on earlier 0.1.0 candidates the drain took up to about 1.1 seconds, and the whole hook up to about 2 seconds when its pointer write stalled (see Hook timing). The result card attached to this release lists this release's own run.

## How to opt in

You need Claude Code 2.1.139 or later (the release that added the hook `args` this plugin uses; this release was tested on 2.1.319), `python3` 3.9 or later on `PATH` (the macOS system Python is fine), and Python 3.11 or later for capture and recall, on `PATH` or in the venv below (for example from Homebrew or python.org). macOS and Linux are supported.

1. Download `lcm-x-claude-code-0.1.0.tar.gz` and its `.sha256` from this release, then check and unpack it:
   ```bash
   shasum -a 256 -c lcm-x-claude-code-0.1.0.tar.gz.sha256 && tar -xzf lcm-x-claude-code-0.1.0.tar.gz
   # Linux: sha256sum -c lcm-x-claude-code-0.1.0.tar.gz.sha256 && tar -xzf lcm-x-claude-code-0.1.0.tar.gz
   ```
2. Start Claude Code with the plugin:
   ```bash
   claude --plugin-dir "$PWD/lcm-x-claude-code-0.1.0"
   ```
   Capture starts with the session. The store lives in `$HOME/.lcm-x/claude-code`, outside the plugin folder, so removing or updating the plugin never removes your memory (set `LCMX_CC_HOME` to move it; keep it outside the plugin folder).
3. Without a Python 3.11 or later, capture waits and records `runtime_missing`. Nothing is lost while it waits as long as Claude Code keeps its transcripts (its default; see the local transcript GC under Known issues), because they stay the source and capture reads them once a runtime is there. One step fixes it: install Python 3.11 or later, or run `uv venv --python 3.11 "${LCMX_CC_HOME:-$HOME/.lcm-x/claude-code}/venv"`.

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

- **Claude Code's local transcript GC.** Lossless capture needs Claude Code's opt-in local transcript GC off (`CLAUDE_CODE_TRANSCRIPT_LOCAL_GC` unset, the default). With it on, capture is best effort, and the audit lists those sessions under `local_gc`; its `ok` and `verified` cannot cover lines the GC removed before capture.
- **A subagent transcript removed before the worker first reads it** cannot be captured, and the audit cannot see that it existed.
- **Audit memory.** The audit needs memory roughly in proportion to the largest session in the store: about 1.3 GB for a 690,000-line session in our gate run. Capture itself peaked at about 0.4 GB in that run.
- **Hook timing.** Hooks usually hand off in milliseconds, but the hook's write of its small pointer file sometimes stalls for a few seconds, at moderate as well as heavy machine load.
  - **The bound.** `PostToolBatch` runs in the background. The other hooks run before Claude Code goes on, with a 5-second limit (15 seconds for `PreCompact`), so a stall holds up the turn until the hook finishes or Claude Code stops it. In our runs Claude Code recorded the hooks it stopped after 5.5 to 11.5 seconds.
  - **What we measured.** These figures are from three full release-gate runs on 0.1.0 candidates with the same hook code:
    - Of the hook runs that finished, 8 of 670, 2 of 679 and 5 of 682 took over 2 s, every one of them in that write.
    - The slowest blocking hook that finished in an ordinary session took 3.8 s, 4.1 s and 4.7 s.
    - The background hook and the stress tests' scripted hook bursts reached up to 8.3 s.
    - Claude Code's transcripts record 3, 0 and 3 hooks stopped at their time limit, all `Stop` or `SubagentStop` hooks in ordinary sessions. The hook log has no line for them, because it records a hook only when it finishes.
  - **This release's own run.** The result card attached to this release lists the hook timings measured on this release's own gate run.

  In all three runs no audit lost anything. A hook cut off at its time limit is designed to be caught up by the session's next hook. If an earlier hook had already recorded the session, a later worker run, even one started from another session, also catches it up. If every hook of a new session were cut off before any of them recorded it, that session would not be captured.

  For each of the six hooks Claude Code stopped, the session's next hook ran within half a second and started the worker, and that session's audit was verified with 0 lost and 0 duplicated lines. Tracked in #2350.
- **Leftover temporary files.** A hook that is killed mid-write can leave a hidden temporary file (`.<id>.<pid>.tmp`, empty or partly written) in the store's `spool` folder. Capture ignores it, and it is safe to delete.

## What it bundles

Unit version 0.1.0, tag `claude-code-v0.1.0`, with LCM-X core vendored from the tagged commit. The asset's `bundle.json` records the full core commit and core version, and the release workflow adds them as the last line of these notes.

## Release evidence

The unit gate made ten runs on the tagged commit: nine real Claude Code sessions (multi-turn tool use, resume, forks, manual and automatic compaction, parallel subagents, four parallel sessions, killed workers and sessions, cross-session recall, and a driven 32-turn session with idle gaps and compactions) and a capture rerun over every transcript. Each was audited with 0 lost and 0 duplicated lines. The gate also ran the CLI smoke in both runtime modes and install records on macOS and Linux. Its growth check, on one real transcript of about 690,000 lines, ran on the tagged commit, or on an earlier candidate when nothing that shapes the stored bytes changed since; the result card says which. The result card is attached to this release.
