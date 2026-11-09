# Recall LCM-X memory from Codex (read-only)

Codex can search and read an LCM-X store through LCM-X's read-only MCP reader, the same one the
Claude Code plugin uses. This is step 1 of the Codex harness (#2227): **install and recall only**.
Codex sessions are not captured into any store yet; capture needs its own design.

What you get in Codex: five read-only tools, `lcm_recall`, `lcm_grep`, `lcm_expand`,
`lcm_load_session` and `lcm_describe`, over one store. The reader opens the store read-only and
never writes to it.

## Which store

The reader serves one store home. The usual choice is the store that LCM-X for Claude Code fills
(`$HOME/.lcm-x/claude-code` unless you set `LCMX_CC_HOME`), so Codex can recall what you did in Claude
Code. Any LCM-X capture store home works the same way. Do not point it at a Hermes profile folder.

## Install

1. **Get the reader.** Download the Claude Code unit asset from the latest `claude-code-v*` release
   (`lcm-x-claude-code-<version>.tar.gz` and its `.sha256`), check it and unpack it somewhere that
   will stay put:

   ```bash
   shasum -a 256 -c lcm-x-claude-code-<version>.tar.gz.sha256
   mkdir -p "$HOME/.lcm-x/units" && tar -xzf lcm-x-claude-code-<version>.tar.gz -C "$HOME/.lcm-x/units"
   ```

   The reader needs Python 3.11 or newer. It uses `LCMX_PYTHON` if set, else
   `<store home>/venv/bin/python`, else a Python 3.11+ found on `PATH`.

2. **Add it to Codex.** Use absolute paths. `$HOME` is expanded by your shell here, not by Codex:

   ```bash
   codex mcp add lcm-x --env LCMX_CC_HOME="$HOME/.lcm-x/claude-code" -- \
     python3 "$HOME/.lcm-x/units/lcm-x-claude-code-<version>/bin/lcmx_mcp.py"
   codex mcp list
   ```

   That writes this to `$HOME/.codex/config.toml` (or `$CODEX_HOME/config.toml`), which you can also
   write by hand, with `<home>` replaced by your home folder's absolute path:

   ```toml
   [mcp_servers.lcm-x]
   command = "python3"
   args = ["<home>/.lcm-x/units/lcm-x-claude-code-<version>/bin/lcmx_mcp.py"]

   [mcp_servers.lcm-x.env]
   LCMX_CC_HOME = "<home>/.lcm-x/claude-code"
   ```

3. **Try it.** In a new Codex session, ask about something from an earlier Claude Code session,
   for example "Use the lcm-x tools: what did we decide about the release freeze last week?".

To remove it: `codex mcp remove lcm-x`. That leaves the store and the unpacked unit untouched.

## When it does not work

- `codex mcp list` shows the server but Codex has no `lcm_*` tools: the reader printed why on
  stderr. Run the command from step 2 by hand to see it. The usual causes are no Python 3.11+
  (install one, or run `uv venv --python 3.11 <store home>/venv`), a relative `LCMX_CC_HOME`, and
  a store home with no `lcm.db` yet.
- Recall finds nothing: check that the store home is the one your Claude Code capture writes
  (`lcm_describe` shows the store's counts).

## What this does not do yet

- No Codex capture: Codex's own sessions do not go into the store.
- No shared store: each harness has its own store home, and this reader reads one of them.
- The proof for this step is one install on a fresh Codex setup and one end-to-end recall from
  Codex. It is not a recall measurement; the Claude Code reader's recall is measured under #2225.
