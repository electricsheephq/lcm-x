#!/usr/bin/env bash
# Prepare a Linux cloud sandbox (Codex cloud tasks, Claude Code on the web, a bench VM) to run
# the test suite the way CI's `test` job does: the pinned test tools, the SQLite release the
# supported Hermes runtimes bundle, the hermes-agent ContextEngine stub and the rollback-reader
# commits. The SQLite build, the stub and the commit list are read from .github/ at run time,
# so this script follows CI when CI changes. A green run here is advisory; GitHub CI gates.
#
# Usage, from anywhere in the checkout (needs network, python3 >= 3.11 with venv, curl,
# unzip, gcc, sha256sum):
#   bash scripts/dev/cloud_setup.sh
# Later shells pick up the environment from ~/.bashrc (and $CLAUDE_ENV_FILE when set).
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
cd "$repo"
cache="${XDG_CACHE_HOME:-$HOME/.cache}/lcmx-ci"
mkdir -p "$cache"

"${PYTHON:-python3}" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || { echo "cloud_setup: Python >= 3.11 is required" >&2; exit 1; }
[ -x .venv/bin/python ] || "${PYTHON:-python3}" -m venv .venv
export PATH="$repo/.venv/bin:$PATH"
# The versions CI's test job installs, plus PyYAML to read the workflow files.
python -m pip install -q pytest==9.1.1 pytest-xdist==3.8.0 numpy ruff==0.15.13 hypothesis==6.141.1 PyYAML==6.0.3

# Print the `run:` script of the first step whose name starts with $2, from the composite
# action or the workflow's `test` job in file $1.
step_script() {
  python - "$1" "$2" <<'PY'
import sys
import yaml

path, prefix = sys.argv[1], sys.argv[2]
doc = yaml.safe_load(open(path))
steps = doc["runs"]["steps"] if "runs" in doc else doc["jobs"]["test"]["steps"]
print(next(step["run"] for step in steps if step.get("name", "").startswith(prefix)))
PY
}

# 1. SQLite: CI's own build-and-verify step, with the action's default version, URL and hash.
eval "$(python - <<'PY'
import shlex
import yaml

inputs = yaml.safe_load(open(".github/actions/sqlite-runtime/action.yml"))["inputs"]
for key in ("version", "url", "sha256"):
    print(f"export SQLITE_{key.upper()}={shlex.quote(inputs[key]['default'])}")
PY
)"
: > "$cache/github-env"
RUNNER_TEMP="$cache" GITHUB_ENV="$cache/github-env" \
  bash -c "$(step_script .github/actions/sqlite-runtime/action.yml 'Build and load SQLite')"

# 2. The hermes-agent ContextEngine stub (agent/ is gitignored).
bash -c "$(step_script .github/workflows/ci.yml 'Stub hermes-agent ABC')"

# 3. The rollback-reader commits. Without them the old-reader tests fail here (#841).
if ! bash -c "$(step_script .github/workflows/ci.yml 'Fetch the rollback-reader commits')"; then
  echo "cloud_setup: warning: could not fetch the rollback-reader commits; the old-reader tests will fail" >&2
fi

# Keep the venv and the SQLite library for every later shell.
cat > "$cache/env.sh" <<EOF
export PATH="$repo/.venv/bin:\$PATH"
export LD_LIBRARY_PATH="$cache/sqlite-runtime\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
grep -qsF "$cache/env.sh" "$HOME/.bashrc" || echo ". \"$cache/env.sh\"" >> "$HOME/.bashrc"
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  cat "$cache/env.sh" >> "$CLAUDE_ENV_FILE"
fi
. "$cache/env.sh"
python -c "import sqlite3, sys; print('ready: python', sys.version.split()[0], 'sqlite', sqlite3.sqlite_version)"
if [ "$(id -u)" = 0 ]; then
  # Root ignores file permissions, so tests that expect a read-only file or directory to refuse a
  # write fail (14 at 0eb6e873: test_issue_1186_dependency_companion, test_stemmed_recall_index,
  # test_track_s_integrity); they pass as any other user.
  echo "cloud_setup: warning: running as root; permission-based tests fail as root. Run pytest as a non-root user." >&2
fi
cat <<'EOF'
Run the suite as CI does:
  ulimit -n 1024
  python -m pytest tests/ -q -m "not latency" -n 4 --ignore=tests/test_embedding_privacy.py
  python -m pytest tests/test_embedding_privacy.py -q -m "not latency"
EOF
