- New: `adapters/hub`, one shared LCM-X memory for the Claude Code and Codex sessions of several machines. Each node
  sends only new transcript bytes over a forced-command SSH key; one hub daemon runs the unchanged engine over one
  store (one session per transcript file, raw lines kept, keyed so nothing is stored twice) and serves the 13
  `lcm_*` tools to every agent through the `lcmx-hub` plugins for Claude Code and Codex. The plugins add a start note,
  a capsule after compaction, a sync before compaction, opt-in proactive recall per prompt, and an `lcmx` skill.
