# LCM-X for Claude Code

LCM-X memory for Claude Code. The plugin captures every message and tool result of your
Claude Code sessions into a local, lossless LCM-X store, and gives the agent read-only tools to
search and read it (`lcm_recall`, `lcm_grep`, `lcm_expand`, `lcm_load_session`, `lcm_describe`).
Claude Code's own compaction is unchanged, and the plugin adds nothing to the context by itself.

**Status: preview in development (0.1.0-dev).** The hooks record which transcript changed and
start the capture worker, which writes every message and tool result to the store. The worker
and recall need the read-only MCP reader (#1216), which lands first. Design:
[docs/claude-code-adapter.md](../../docs/claude-code-adapter.md).

## Try it from a checkout

```bash
claude --plugin-dir adapters/claude-code
```

Hooks need the system `python3` (3.9 or newer). Capture and recall need Python 3.11 or newer. The
plugin looks for one in this order:
- `LCMX_PYTHON`;
- a venv in the plugin's data folder, or at `<store home>/venv`;
- any Python 3.11+ on `PATH` (`python3.11` to `python3.14`, or a `python3` that is 3.11 or newer),
  run with the standard library only.

Without one, capture records `runtime_missing`. One step sets it up: install Python 3.11 or newer,
or run `uv venv --python 3.11 ~/.lcm-x/claude-code/venv`. Vector recall also needs FastEmbed
installed in that venv, with its model already downloaded; without it, recall uses full-text
search.

## Settings (environment)

| Variable | Default | Meaning |
|---|---|---|
| `LCMX_CC_HOME` | `~/.lcm-x/claude-code` | Store home: `lcm.db`, payloads, spool, logs |
| `LCMX_CAPTURE` | on | `off` disables capture |
| `LCMX_PYTHON` | `<store home>/venv/bin/python` | Runtime interpreter |
| `LCMX_PACKAGE_ROOT` | the plugin's vendored copy, else this checkout | LCM-X package |
| `LCMX_GROK_CAPTURE` | off | `1` enables the experimental Grok Build reader |
| `LCMX_GROK_HOME` | `~/.lcm-x/grok-build` | Store home for Grok Build sessions (one owning harness per store) |

Home settings must be absolute paths; a relative one disables capture and the server says so.

A `reader-override.json` in the store home makes the server start another reader in place of the
plain one, and hands it the plain reader's command. Anything wrong with the file starts the plain
reader, and so does a file or command that another user could have written. The plugin never writes
that file; its format and fallbacks are in the design doc's Recall surface section.

The command's first item must be an absolute path; there is no `PATH` search. Every folder on the
way to it, through any symlinks, must be writable only by you or root, or be a shared sticky folder
such as `/tmp` whose entry is yours. A folder writable by its group passes only when that group's
members, listed or by primary group, are you and root, and only where the user database is provably
local: on Linux, glibc with `nsswitch.conf` taking users and groups only from `files`, `systemd` or
`altfiles`; on macOS, a directory search path of local nodes only. On a machine bound to a directory
service (SSSD, LDAP, Active Directory), which may not list its users, a command under a folder
writable by its group starts the plain reader, and the line on stderr says "user database not
provably local": keep the command under folders only you can write. So does a command under
Homebrew's folders on a Mac with several admins; groups nested in another group are not expanded.
Script arguments, such as the script in
`["/usr/bin/python3", "/path/to/override.py"]`, are not checked: keep them in a folder only you can
write. Nor are a script's `#!` interpreter, the `PATH` search of `#!/usr/bin/env`, or folder ACLs.

The store lives outside the plugin's folders, so removing or updating the plugin never removes
your memory. Capture runs on macOS and Linux.
