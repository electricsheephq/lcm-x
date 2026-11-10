# LCM-X for Claude Code 1.0.0

LCM-X for Claude Code keeps a lossless, local record of your Claude Code sessions and lets the agent search it. This is the first full release, and we recommend it for every Claude Code user on macOS or Linux. It is still opt-in: it changes nothing until you start Claude Code with it.

## What it does

- **Captures every session.** Each message, tool call and tool result of your Claude Code sessions is written to a local LCM-X store, including the large tool outputs Claude Code saves to files, subagent sessions (workflow agents at any depth too), resumed sessions and forks. Nothing is summarized away or dropped: the store keeps every transcript line it reads, and an audit command checks each captured line against its transcript file, and every transcript in each session folder against the store (see Known issues for what it cannot see).
- **Lets the agent recall it.** The plugin adds five read-only tools: `lcm_grep` (full-text search, including exact phrases in quotes), `lcm_recall`, `lcm_expand` (returns a stored message or tool output in full, paging through long ones), `lcm_load_session` and `lcm_describe`. They work across sessions, so the agent can find what an earlier session said or ran. In `full_text` mode, the default, `lcm_grep` also searches the large tool outputs Claude Code saved to files, by the same term rule as messages: every word of the query, in any order, with quoted phrases kept together.
- **Leaves Claude Code alone.** Claude Code's own compaction is unchanged. The plugin never rewrites the conversation and adds nothing to the context by itself; the agent reads memory only when it calls a tool. The hooks print nothing and hand the work to a background worker; most hooks now answer Claude Code before they write anything to disk. Before Claude Code compacts, the `PreCompact` hook drains the session into the store first. That can delay compaction: the drain stops after 10 seconds, and Claude Code stops the hook at its 15-second limit. The result card attached to this release lists this release's own hook timings.
- **Recall from Codex too.** The same store can be read from Codex through the plugin's read-only reader; see `docs/codex-reader.md` in the repository.

## How to opt in

You need Claude Code 2.1.139 or later (the release that added the hook `args` this plugin uses), `python3` 3.9 or later on `PATH` (the macOS system Python is fine), and Python 3.11 or later for capture and recall, on `PATH` or in the venv below (for example from Homebrew or python.org). macOS and Linux are supported.

1. Download `lcm-x-claude-code-1.0.0.tar.gz` and its `.sha256` from this release, then check and unpack it:
   ```bash
   shasum -a 256 -c lcm-x-claude-code-1.0.0.tar.gz.sha256 && tar -xzf lcm-x-claude-code-1.0.0.tar.gz
   # Linux: sha256sum -c lcm-x-claude-code-1.0.0.tar.gz.sha256 && tar -xzf lcm-x-claude-code-1.0.0.tar.gz
   ```
2. Start Claude Code with the plugin:
   ```bash
   claude --plugin-dir "$PWD/lcm-x-claude-code-1.0.0"
   ```
   Capture starts with the session. The store lives in `$HOME/.lcm-x/claude-code`, outside the plugin folder, so removing or updating the plugin never removes your memory (set `LCMX_CC_HOME` to move it; keep it outside the plugin folder). Upgrading from 0.1.0: unpack 1.0.0 and point `--plugin-dir` at it; the store is kept and read as it is.
3. Without a Python 3.11 or later, capture waits and records `runtime_missing`. Nothing is lost while it waits as long as Claude Code keeps its transcripts (its default; see the local transcript GC under Known issues), because they stay the source and capture reads them once a runtime is there. One step fixes it: install Python 3.11 or later, or run `uv venv --python 3.11 "${LCMX_CC_HOME:-$HOME/.lcm-x/claude-code}/venv"`.

To check what was captured, run the audit with any Python 3.11 or later (here `python3.12`): `python3.12 lcm-x-claude-code-1.0.0/lcm-x/scripts/lcm_x_capture.py audit --home "${LCMX_CC_HOME:-$HOME/.lcm-x/claude-code}"`. It prints counts and ids only, and says whether every captured line is accounted for (`ok`) and nothing is pending or unexplained (`verified`).

To turn capture off for a launch, set `LCMX_CAPTURE=off`, or start Claude Code without `--plugin-dir`.

## What it does not do yet

- **No compaction or context injection.** Recall is on demand, through the tools; memory is not added to the context automatically.
- **Not searchable yet:** thinking blocks, attachments (`@`-file contents, plans, hook-injected context) and bookkeeping entries are kept in the store's ledger but are not indexed for search.
- **Saved tool outputs in `hybrid` and `semantic` search.** In those modes `lcm_grep` searches messages only; use `mode='full_text'` (the default) to include saved tool outputs (#2592).
- **Vector recall needs a setup step:** FastEmbed installed in the venv above, with its model already downloaded. Without it, recall uses full-text search. The plugin never downloads a model.
- **No forget command** for Claude Code sessions yet.
- **One store per harness.** It does not share a store with the Hermes plugin; Hermes and Claude Code memories are separate in this release.
- **Grok Build is not supported yet.** Grok Build sessions are not captured: the hooks log each Grok event as `grok_capture_off` and spool nothing.
- **No plugin marketplace entry** yet: you pass `--plugin-dir` on each launch.

## Known issues

- **Claude Code's local transcript GC.** Lossless capture needs Claude Code's opt-in local transcript GC off (`CLAUDE_CODE_TRANSCRIPT_LOCAL_GC` unset, the default). With it on, capture is best effort, and the audit lists those sessions under `local_gc`; its `ok` and `verified` cannot cover lines the GC removed before capture.
- **A subagent transcript removed before the worker first reads it** cannot be captured, and the audit cannot see that it existed.
- **Files in subfolders of a session's tool-results folder** (for example page images rendered from a PDF) are counted by the audit but not stored. A session that has such files does not read as verified (#2554).
- **Many very large saved outputs in one batch** can stall capture of that session: the worker can run out of memory on the batch, roll it back and retry it. Nothing is lost, and the audit reads the session as unverified until it is captured (#2549).
- **A second transcript path for one session or subagent.** If Claude Code ever writes one lineage's lines to a second file, the audit reads it as unverified rather than capturing it (#2551). We have not seen this in real sessions.
- **A Claude Code process killed before any end hook** (no `SessionEnd` or `SubagentStop`) can leave a partly written last line. The audit counts it in its verdict, but the session can still read as verified. A subagent cut off by its session's end reads as unverified. After a resume, a cut-off helper agent's partial last line can leave the session reading as verified (#2566).
- **A subagent whose tool call and its result fall on both sides of a `/clear`** leaves the new session's audit not `ok`, with one orphan line. Nothing is lost.
- **Stores upgraded from 0.1.0 where Claude Code reused one agent id at two paths in one session.** Capture keeps both transcripts' lines (#2576), but on such a store the audit can report the second path's lines as lost while nothing is lost, and a tool call and its result can sit in two sessions. Deleting the first subagent session leaves copies of the shared lines in the ledger of the second path's session until that session is deleted too; recall never reads them (#2566).
- **Audit memory.** The audit needs memory roughly in proportion to the largest session in the store: about 1.3 GB for a 690,000-line session in the 0.1.0 gate run. The result card attached to this release lists this release's own run.
- **Hook timing.** Most hooks now return to Claude Code before they write anything (#2531), so a slow disk no longer holds up the turn for them. `SessionStart` and `PreCompact` still do their work before Claude Code goes on, with a 5-second limit (15 seconds for `PreCompact`). A hook that Claude Code stops at its limit is caught up by the session's next hook. The result card attached to this release lists the hook timings measured on this release's own gate run.
- **Leftover temporary files.** A hook that is killed mid-write can leave a hidden temporary file (`.<id>.<pid>.tmp`, empty or partly written) in the store's `spool` folder. Capture ignores it, and it is safe to delete.

## What it bundles

Unit version 1.0.0, tag `claude-code-v1.0.0`, with LCM-X core vendored from the tagged commit. The asset's `bundle.json` records the full core commit and core version, and the release workflow adds them as the last line of these notes.

## Release evidence

The unit gate ran on the tagged commit: nine real Claude Code sessions (multi-turn tool use, resume, forks, manual and automatic compaction, parallel subagents, four parallel sessions, killed workers and sessions, cross-session recall, and a driven 32-turn session with idle gaps and compactions) and a capture rerun over every transcript, each audited. It also ran the CLI smoke in both runtime modes on Python 3.11 and 3.14, and install records on macOS and Linux. Its growth check ran on one real transcript of about 690,000 lines, on the tagged commit or on an earlier commit when nothing that shapes the stored bytes changed since; the result card says which. Recall through the plugin's tools was measured on a 40-question set before this release (#2225). The result card is attached to this release.
