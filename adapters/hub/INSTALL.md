# Install the LCM-X hub and the lcmx-hub plugins

Part 1 is the hub (once, on one Linux host). Part 2 is each node (each machine where Claude Code or Codex runs).
The examples use the hub host name `hub`, the hub user `lcmx`, the checkout `/opt/lcm-x` and the data root
`/srv/lcmx-hub`; change them as you need.

## Part 1: the hub

1. Make a user for the hub, with no password and no sudo: `lcmx`. Give it `~/.hushlogin` (a login banner would break
   the SSH protocol lines).
2. Check out this repository at a pinned commit, for example `/opt/lcm-x`. If root owns the checkout, run git there
   as root (`sudo git -C /opt/lcm-x ...`); as another user, git stops with "dubious ownership".
3. As `lcmx`: install CPython 3.12 and a venv with FastEmbed:

   ```bash
   uv python install 3.12.16
   uv venv --python 3.12.16 ~/lcmx-venv
   uv pip install --python ~/lcmx-venv/bin/python -r /opt/lcm-x/adapters/eve/host/requirements-fastembed.txt
   ```

4. As `lcmx`: install Codex (`@openai/codex@0.167.0` was tested) and sign in once with `codex login --device-auth`.
   The hub uses it for summaries and `lcm_expand_query`.
5. Start the daemon as a systemd user service of `lcmx` (linger on):

   ```ini
   [Service]
   Environment=LCMX_LLM_CLI=codex LCM_EMBEDDINGS_ENABLED=true LCM_EMBEDDING_PROVIDER=fastembed LCM_SENSITIVE_PATTERNS_ENABLED=true
   ExecStart=%h/lcmx-venv/bin/python -I /opt/lcm-x/adapters/hub/hub/lcmx_hub.py serve --root /srv/lcmx-hub
   Restart=on-failure
   ```

   `LCM_SENSITIVE_PATTERNS_ENABLED=true` turns on LCM-X's durable redaction (API keys, bearer tokens, passwords,
   private keys) in the store. The raw lines in `hub-ledger.db` and the mirror stay unredacted on the hub only.
6. Check: `~/lcmx-venv/bin/python -I /opt/lcm-x/adapters/hub/hub/lcmx_hub.py check --root /srv/lcmx-hub`.
7. Backup: run `lcmx_hub.py backup --root /srv/lcmx-hub` as `lcmx` (a tar on stdout) and encrypt it. The raw mirror is
   not in the backup, and the backup empties the raw lines in its copy of `hub-ledger.db`: only redacted text leaves
   the hub. Restore test: `PRAGMA integrity_check` plus `lcmx_hub.py audit`.

## Part 2: each node

1. Make the node's own SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/lcmx_node -C lcmx-node-<node>`.
2. On the hub, add one line to `~lcmx/.ssh/authorized_keys` for this key. `<node>` is the node's name on the hub (a
   short lower-case word):

   ```
   command="/home/lcmx/lcmx-venv/bin/python -I /opt/lcm-x/adapters/hub/hub/lcmx_hub.py ssh --root /srv/lcmx-hub <node>",restrict ssh-ed25519 AAAA... lcmx-node-<node>
   ```

3. Settings on the node, `~/.config/lcmx-hub/node.env`:

   ```
   LCMX_HUB_HOST=hub         # SSH host name of the hub
   LCMX_HUB_USER=lcmx
   LCMX_HUB_KEY=~/.ssh/lcmx_node
   ```

4. Plugins, from a checkout of this repository (`adapters/hub` is a marketplace for both CLIs):

   ```bash
   claude plugin marketplace add <checkout>/adapters/hub
   claude plugin install lcmx-hub@lcm-x-hub
   codex plugin marketplace add <checkout>/adapters/hub
   codex plugin add lcmx-hub@lcm-x-hub
   ```

   - Claude Code: allow the tools once, `mcp__plugin_lcmx-hub_lcmx__*` (settings, `permissions.allow`).
   - Codex: trust the plugin's hooks once (`/hooks` in the TUI). Codex skips untrusted hooks; the timer still syncs.
5. Timer every 5 minutes: launchd on macOS, a systemd user timer on Linux, running
   `python3 -I -S <checkout>/adapters/hub/node/lcmx_sync --quiet`.
6. Keep `CLAUDE_CODE_TRANSCRIPT_LOCAL_GC` unset (lossless capture needs the transcripts).
7. Optional: `LCMX_HUB_EXCLUDE` in `node.env`: comma-separated globs on the relative path that are never sent, for
   folders that are not coding sessions (for example `claude/projects/-Users-me-agents-*`). Check with
   `lcmx_sync --dry-run` before the first sync.
8. Optional: proactive recall for each prompt: add `LCMX_HUB_PROACTIVE=on` to `node.env`. Each prompt then waits
   about 1 s for the hub. Off by default.
9. Apps built on the Claude Agent SDK that load user settings get the plugin after they restart.
10. Check: `python3 -I -S <checkout>/adapters/hub/node/lcmx_sync` prints the files and bytes it sent; in a session, ask
    the agent to search the shared memory with `lcmx`.

Only turns from now on: the first run of `lcmx_sync` writes `~/.cache/lcmx-hub/baseline-v2.json` with the size of
every file at that moment. The hub gets only the bytes after it: a session that was open at the install sends its new
turns, and a new file goes whole. Do not delete the baseline files (that would send the old history).
`LCMX_HUB_HISTORY=all` sends all history.

## Update a node

Check out the new commit, update both plugins from it (Claude Code: `claude plugin marketplace update lcm-x-hub` and
`claude plugin update lcmx-hub@lcm-x-hub`; Codex: add the plugin again), trust changed Codex hooks once (`/hooks`), and
point the timer at the new `lcmx_sync`. Update the hub first when a new node verb is added.
