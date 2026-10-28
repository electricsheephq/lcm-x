#!/usr/bin/env bash
# Prepare a Linux cloud sandbox (Codex cloud tasks, Claude Code on the web, a bench VM) to run
# the test suite the way CI's `test` job does: CI's own test-tool install, the SQLite release the
# supported Hermes runtimes bundle, the hermes-agent ContextEngine stub and the rollback-reader
# commits. Each of those is CI's own step script, read from .github/ at run time, so this
# script follows CI when CI changes. A green run here is advisory; GitHub CI gates.
# The script owns the checkout's .venv: it creates or recreates it and removes the YAML
# libraries from it, so run it in a sandbox checkout, not one whose .venv you manage.
#
# Usage, from anywhere in the checkout (needs network, python3 >= 3.11 with venv, curl,
# unzip, gcc, sha256sum):
#   bash scripts/dev/cloud_setup.sh
# Then run the printed commands; they start by sourcing the environment file, which
# non-interactive shells (agents, ssh) do not read from ~/.bashrc.
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
cd "$repo"
cache="${XDG_CACHE_HOME:-$HOME/.cache}/lcmx-ci"
mkdir -p "$cache"

"${PYTHON:-python3}" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || { echo "cloud_setup: Python >= 3.11 is required" >&2; exit 1; }
# Reuse .venv only when it was made here, on Python >= 3.11, without the system site packages
# (a system PyYAML would stay importable). Otherwise recreate it; a venv copied or moved from
# another path keeps shebangs that point there.
reuse=0
case "$(head -1 .venv/bin/pip 2>/dev/null)" in
  "#!$repo/.venv/bin/python"*) reuse=1 ;;
esac
if [ "$reuse" = 1 ]; then
  .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null || reuse=0
  if grep -qsiE '^include-system-site-packages *= *true' .venv/pyvenv.cfg; then reuse=0; fi
fi
[ "$reuse" = 1 ] || "${PYTHON:-python3}" -m venv --clear .venv
export PATH="$repo/.venv/bin:$PATH"
# PyYAML only to read the workflow files; it is removed again below, because CI's main
# pytest run has no PyYAML (config.py falls back to its own parser without it).
python -m pip install -q PyYAML==6.0.3

# Print the `run:` script of the first step whose name starts with $2, from the composite
# action or the workflow's `test` job in file $1. A missing step is an error.
step_script() {
  python - "$1" "$2" <<'PY'
import sys
import yaml

path, prefix = sys.argv[1], sys.argv[2]
doc = yaml.safe_load(open(path))
steps = doc["runs"]["steps"] if "runs" in doc else doc["jobs"]["test"]["steps"]
runs = [step["run"] for step in steps if step.get("name", "").startswith(prefix)]
if not runs:
    sys.exit(f"cloud_setup: no step named {prefix!r} in {path}")
print(runs[0])
PY
}

# Assign each script first: under `set -e` a failed command substitution inside an argument
# would not stop the script, and an empty script would "succeed".
install="$(step_script .github/workflows/ci.yml 'Install test dependencies')"
sqlite="$(step_script .github/actions/sqlite-runtime/action.yml 'Build and load SQLite')"
stub="$(step_script .github/workflows/ci.yml 'Stub hermes-agent ABC')"
rollback="$(step_script .github/workflows/ci.yml 'Fetch the rollback-reader commits')"
ruamel="$(step_script .github/workflows/ci.yml 'Test the upstream ruamel loader')"
eval "$(python - <<'PY'
import shlex
import yaml

inputs = yaml.safe_load(open(".github/actions/sqlite-runtime/action.yml"))["inputs"]
for key in ("version", "url", "sha256"):
    print(f"export SQLITE_{key.upper()}={shlex.quote(inputs[key]['default'])}")
PY
)"

# 1. CI's test tools. 2. SQLite: CI's build-and-verify step with the action's defaults.
bash -c "$install"
: > "$cache/github-env"
RUNNER_TEMP="$cache" GITHUB_ENV="$cache/github-env" bash -c "$sqlite"
# 3. The hermes-agent ContextEngine stub (agent/ is gitignored).
bash -c "$stub"
# 4. The rollback-reader commits. Without them the old-reader tests fail here (#841).
if ! bash -c "$rollback"; then
  echo "cloud_setup: warning: could not fetch the rollback-reader commits; the old-reader tests will fail" >&2
fi
# CI's main pytest run has neither YAML library (config.py then uses its own parser); CI's ruamel
# step below installs both, so a re-run removes both again.
python -m pip uninstall -q -y PyYAML ruamel.yaml ruamel.yaml.clib 2>/dev/null || true

# Keep the venv and the SQLite library for every later shell. Paths are shell-quoted, so a path
# with spaces or `$` in it is read as written.
printf 'export PATH=%q:"$PATH"\nexport LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' \
  "$repo/.venv/bin" "$cache/sqlite-runtime" > "$cache/env.sh"
source_line=". $(printf '%q' "$cache/env.sh")"
grep -qsF "$cache/env.sh" "$HOME/.bashrc" || grep -qsxF "$source_line" "$HOME/.bashrc" \
  || echo "$source_line" >> "$HOME/.bashrc"
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  cat "$cache/env.sh" >> "$CLAUDE_ENV_FILE"
fi
. "$cache/env.sh"
python - <<'PY'
import os
import sqlite3
import sys

want = os.environ["SQLITE_VERSION"]
if sqlite3.sqlite_version != want:
    sys.exit(f"cloud_setup: loaded SQLite {sqlite3.sqlite_version}, expected {want}")
print("ready: python", sys.version.split()[0], "sqlite", sqlite3.sqlite_version)
PY
if [ "$(id -u)" = 0 ]; then
  # 14 tests in test_issue_1186_dependency_companion, test_stemmed_recall_index and
  # test_track_s_integrity fail as root and pass as any other user (PR #2038).
  echo "cloud_setup: warning: running as root. Some tests fail as root, and another user may not be able" \
    "to read this checkout's .venv or $cache. Run this setup and pytest as the same non-root user." >&2
fi
cat <<EOF
Run the suite as CI does (each shell first loads the environment):
  $source_line
  ulimit -n 1024
  python -m pytest tests/ -q -m "not latency" -n 4 --ignore=tests/test_embedding_privacy.py
  python -m pytest tests/test_embedding_privacy.py -q -m "not latency"
Then CI's ruamel loader step (it installs ruamel.yaml and PyYAML), and remove them again so the
main suite keeps CI's no-YAML environment:
$ruamel
python -m pip uninstall -q -y PyYAML ruamel.yaml ruamel.yaml.clib
EOF
