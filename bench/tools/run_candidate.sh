#!/usr/bin/env bash
set -u
fail() { echo "$1" >&2; exit 2; }
repo=$(git rev-parse --show-toplevel) || fail NO_REPO
helper_dir=$(cd "$(dirname "$0")/../../scripts/dev" && pwd) || fail SETUP_FAILED
helper="$helper_dir/ci_step.sh"
base='' patch='' focused='' red='' label=candidate full=0 fetch=1 stub=1 wf=4 wa=6 priority=10
root="$repo/candidate-runs" env_file="$repo/.venv/lcmx-ci/env.sh"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --full) full=1; shift; continue;;
    --no-fetch) fetch=0; shift; continue;;
    --no-stub) stub=0; shift; continue;;
  esac
  [ "$#" -ge 2 ] || fail BAD_ARGUMENT
  case "$1" in
    --base) base=$2;; --patch) patch=$2;; --label) label=$2;;
    --focused) focused=$2;; --red) red=$2;; --root) root=$2;;
    --workers-focused) wf=$2;; --workers-full) wa=$2;; --nice) priority=$2;;
    --env-file) env_file=$2;; *) fail BAD_ARGUMENT;;
  esac
  shift 2
done
[[ "$base" =~ ^[a-fA-F0-9]{40}$ && "$label" =~ ^[a-zA-Z0-9_-]+$ ]] || fail BAD_ARGUMENT
[[ "$wf" =~ ^[1-9][0-9]*$ && "$wa" =~ ^[1-9][0-9]*$ && "$priority" =~ ^[0-9]+$ ]] || fail BAD_ARGUMENT
[ -f "$env_file" ] || fail 'NO_ENV: run bash scripts/dev/cloud_setup.sh'
# A custom env file may expand unset variables, so source it with set -u off.
set +u
# shellcheck disable=SC1090
. "$env_file"
env_status=$?
set -u
[ "$env_status" = 0 ] || fail NO_ENV
# cloud_setup.sh installs both through ci.yml's "Install test dependencies" step.
command -v python > /dev/null 2>&1 || fail 'NO_PYTHON: run bash scripts/dev/cloud_setup.sh'
command -v ruff > /dev/null 2>&1 || fail 'NO_RUFF: run bash scripts/dev/cloud_setup.sh'
[ -f "$patch" ] || fail NO_PATCH
patch_dir=$(cd "$(dirname "$patch")" && pwd) || fail NO_PATCH
patch="$patch_dir/$(basename "$patch")"
[ "$fetch" = 0 ] || git fetch -q origin || fail FETCH_FAILED
git cat-file -e "$base^{commit}" || fail FETCH_FAILED
mkdir -p "$root" || fail SETUP_FAILED
root=$(cd "$root" && pwd) || fail SETUP_FAILED
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run="$root/run-$label-$stamp" green="$root/wt-$label-green-$stamp" redwt="$root/wt-$label-red-$stamp"
mkdir "$run" || fail SETUP_FAILED
[ ! -e "$green" ] && [ ! -e "$redwt" ] || fail SETUP_FAILED
if [ "$stub" = 1 ]; then
  # shellcheck disable=SC1090
  . "$helper" || fail STUB_FAILED
fi
# Print the prune command for each worktree this run made, after a failure as well.
prune=''
trap '[ -z "$prune" ] || printf "%s" "$prune"' EXIT
prepare() {
  local wt=$1 mode=$2 path
  git worktree add --detach "$wt" "$base" > "$run/$mode-setup.log" 2>&1 || fail WORKTREE_FAILED
  printf -v path 'git worktree remove %q\n' "$wt"
  prune="$prune$path"
  if [ "$mode" = red ]; then
    git -C "$wt" apply --index --include='tests/*' "$patch" || fail APPLY_FAILED
  else
    git -C "$wt" apply --index "$patch" || fail APPLY_FAILED
  fi
  if git -C "$wt" diff --cached --quiet; then
    [ "$mode" != red ] || fail NO_TEST_HUNKS
    fail EMPTY_PATCH
  fi
  # Commit the whole index: in this fresh worktree it holds only what the patch staged, and a
  # path list would leave a rename's deletion uncommitted.
  git -C "$wt" -c user.name=Candidate -c user.email=candidate@example.invalid -c commit.gpgsign=false \
    commit --no-verify -qm candidate || fail COMMIT_FAILED
  if [ "$stub" = 1 ]; then
    local script
    script=$(step_script "$wt/.github/workflows/ci.yml" 'Stub hermes-agent ABC') || fail STUB_FAILED
    # GitHub runs a step's script with `bash -e`.
    (cd "$wt" && bash -e -c "$script") > "$run/$mode-stub.log" 2>&1 || fail STUB_FAILED
  fi
}
record() {
  python - "$run" "$base" "$patch" "$label" "$stamp" "$@" <<'PY'
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

run, base, patch, label, stamp, phase, command, code = sys.argv[1:]
out = Path(run) / "summary.json"
data = json.loads(out.read_text()) if out.exists() else dict(
    base=base, patch_sha256=hashlib.sha256(Path(patch).read_bytes()).hexdigest(),
    label=label, stamp=stamp, python=sys.version.split()[0], sqlite=sqlite3.sqlite_version, phases={})
lines = (Path(run) / f"{phase}.log").read_text(errors="replace").splitlines()
final = re.sub(r"(?<!\w)/\S+", "<path>", lines[-1] if lines else "")
data["phases"][phase] = dict(command=command, exit_code=int(code), final_line=final)
out.write_text(json.dumps(data, indent=2) + "\n")
PY
}
# pytest_phase PHASE WORKTREE TARGETS WORKERS [EXTRA_PYTEST_ARG...]; WORKERS 0 runs without xdist.
pytest_phase() {
  local phase=$1 wt=$2 targets=$3 workers=$4 code cmd rel home target args=()
  shift 4
  # TESTS is a whitespace-separated target list, never shell code.
  read -r -a args <<< "$targets"
  [ "${#args[@]}" -gt 0 ] || fail BAD_ARGUMENT
  for target in "${args[@]}"; do [[ "$target" != /* ]] || fail BAD_ARGUMENT; done
  # This runner's spec, not CI: the red receipt lists every test outcome.
  [ "$phase" != red-on-base ] || args+=(-rA)
  [ "$#" -eq 0 ] || args+=("$@")
  # ci.yml: Run pytest shard under low file descriptor limit (ulimit, -q, -m, -n).
  # ci.yml: Test the upstream ruamel loader (-p no:cacheprovider).
  # pytest.ini: tmp_path_retention_policy; basetemp isolates each phase's output.
  rel="../$(basename "$run")"
  args+=(-q -p no:cacheprovider -m 'not latency' -o tmp_path_retention_policy=failed "--basetemp=$rel/bt-$phase")
  # ci.yml: Run embedding privacy tests alone (no -n).
  [ "$workers" = 0 ] || args+=(-n "$workers")
  # CI supplies a disposable HOME. It sits in the run directory, outside the worktree, so the
  # tree stays clean; nice is the host resource policy from this runner's spec.
  home="$run/home-$phase"
  mkdir "$home" || fail SETUP_FAILED
  printf -v cmd '%q ' nice -n "$priority" python -m pytest "${args[@]}"
  cmd="cd ../$(basename "$wt") && ulimit -n 1024 && export HOME=\"\$PWD/$rel/home-$phase\" && $cmd"
  (
    cd "$wt" || exit 126
    ulimit -n 1024 || { echo 'ulimit -n 1024 failed'; exit 126; }
    export HOME="$home"
    nice -n "$priority" python -m pytest "${args[@]}"
  ) > "$run/$phase.log" 2>&1
  code=$?
  echo "$phase: pytest_exit=$code"
  tail -n 2 "$run/$phase.log"
  if [ "$phase" = red-on-base ]; then grep -E '^(FAILED|ERROR|PASSED|XFAIL|XPASS)' "$run/$phase.log" || :
  else grep -E '^(FAILED|ERROR)' "$run/$phase.log" || :; fi
  record "$phase" "$cmd" "$code" || fail SUMMARY_FAILED
  case "$code" in 4|126|127) fail PYTEST_SETUP_FAILED;; esac
}
prepare "$green" green
changed=()
while IFS= read -r -d '' path; do changed+=("$path"); done < <(git -C "$green" diff "$base" HEAD --name-only --diff-filter=ACMR -z -- '*.py')
# Bash before 4.4 treats "${changed[@]}" of an empty array as unset under set -u.
printf 'changed_py:'; [ "${#changed[@]}" -eq 0 ] || printf ' %s' "${changed[@]}"; echo
cmd=:
if [ "${#changed[@]}" -gt 0 ]; then printf -v cmd '%q ' ruff check -- "${changed[@]}"; fi
(cd "$green" && bash -c "$cmd") > "$run/ruff.log" 2>&1
code=$?; echo "ruff_exit=$code"; head -n 80 "$run/ruff.log"
record ruff "cd ../$(basename "$green") && $cmd" "$code" || fail SUMMARY_FAILED
[ -z "$focused" ] || pytest_phase green-focused "$green" "$focused" "$wf"
if [ -n "$red" ]; then prepare "$redwt" red; pytest_phase red-on-base "$redwt" "$red" "$wf"; fi
if [ "$full" = 1 ]; then
  # ci.yml: Collect deterministic test shards (--ignore) and Run embedding privacy tests alone.
  pytest_phase green-full "$green" tests/ "$wa" --ignore=tests/test_embedding_privacy.py
  # CI runs this file even if a candidate removes it; a missing file fails as PYTEST_SETUP_FAILED.
  pytest_phase green-privacy "$green" tests/test_embedding_privacy.py 0
fi
echo "dirs: $run $green"
exit 0
