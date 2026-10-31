# LCM-X for Claude Code

LCM-X memory for Claude Code. The plugin captures every message and tool result of your
Claude Code sessions into a local, lossless LCM-X store, and gives the agent read-only tools to
search and read it (`lcm_recall`, `lcm_grep`, `lcm_expand`, `lcm_load_session`, `lcm_describe`).
Claude Code's own compaction is unchanged, and the plugin adds nothing to the context by itself.

**Status: draft.** The hooks record which transcript changed and start a capture worker; the
worker and the store writer are not in this build yet, so nothing is captured. Design:
[docs/claude-code-adapter.md](../../docs/claude-code-adapter.md).

## Try it from a checkout

```bash
claude --plugin-dir adapters/claude-code
```

Hooks need the system `python3` (3.9 or newer). Capture and recall need Python 3.11 or newer with
LCM-X's dependencies, at `<store home>/venv` or named by `LCMX_PYTHON`.

## Settings (environment)

| Variable | Default | Meaning |
|---|---|---|
| `LCMX_CC_HOME` | `~/.lcm-x/claude-code` | Store home: `lcm.db`, payloads, spool, logs |
| `LCMX_CAPTURE` | on | `off` disables capture |
| `LCMX_PYTHON` | `<store home>/venv/bin/python` | Runtime interpreter |
| `LCMX_PACKAGE_ROOT` | the plugin's vendored copy, else this checkout | LCM-X package |
| `LCMX_GROK_CAPTURE` | off | `1` enables the experimental Grok Build reader |

The store lives outside the plugin's folders, so removing or updating the plugin never removes
your memory. Capture runs on macOS and Linux.
