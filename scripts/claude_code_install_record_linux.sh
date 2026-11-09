#!/bin/bash
# Native Linux install record for LCM-X for Claude Code: no containers. Run as an unprivileged user
# on an Ubuntu host (written for 24.04, python3 3.12). Every case gets a fresh scratch HOME and
# CLAUDE_CONFIG_DIR under WORKDIR and runs under `env -i`, so the user's own setup is never read or
# written. Output: counts, ids and timings only. The model key is read from KEY_FILE into the model
# calls' environment and is never printed. BASE_URL, when set, points Claude Code at an
# Anthropic-compatible endpoint (left unset, Claude Code uses its default).
# The unit gate's Linux install record (bench/specs/RELEASE-READINESS-V1.md, "Claude Code unit
# releases") when no fresh container is available (#2348).
#
#   CANDIDATE=<full sha on main> KEY_FILE=<mode-600 key file> WORKDIR=<new or empty dir> [BASE_URL=<url>] \
#     bash scripts/claude_code_install_record_linux.sh 2>&1 | tee linux-install-record.out
#
# Case A: the host's python3 (3.11+) on PATH; expect zero manual steps.
# Case B: PATH whose only Python is a uv-managed 3.10 (the host's python3* names are left out of a
#   linked copy of the system bin dirs); expect runtime_missing, then the documented one step
#   `uv venv --python 3.11 ~/.lcm-x/claude-code/venv`, a resume, and a verified audit.
# Proxy variables (http_proxy, https_proxy, no_proxy, any case) pass through when set.
set -euo pipefail
: "${CANDIDATE:?set CANDIDATE}"; : "${KEY_FILE:?file holding the model key, mode 600}"; : "${WORKDIR:?new or empty dir}"
[ "$(stat -c %a "$KEY_FILE")" = 600 ] || { echo "KEY_FILE must be mode 600"; exit 2; }
if [ -e "$WORKDIR" ] && [ -n "$(ls -A "$WORKDIR")" ]; then echo "WORKDIR must be new or empty"; exit 2; fi
mkdir -p "$WORKDIR"; WORKDIR=$(cd "$WORKDIR" && pwd); KEY_FILE=$(cd "$(dirname "$KEY_FILE")" && pwd)/$(basename "$KEY_FILE")
SYSPATH=/usr/local/bin:/usr/bin:/bin
PASS=(LANG=C.UTF-8 TERM=dumb CANDIDATE="$CANDIDATE" KEY_FILE="$KEY_FILE" BASE_URL="${BASE_URL:-}"
      CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1)
for v in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do [ -n "${!v:-}" ] && PASS+=("$v=${!v}"); done
echo "== host: $(. /etc/os-release && echo "$PRETTY_NAME"), $(/usr/bin/python3 -V), native (clean environment, not a fresh container), endpoint $(if [ -n "${BASE_URL:-}" ]; then echo custom; else echo default; fi), candidate $CANDIDATE, start $(date -u +%FT%TZ)"

# The release asset stand-in: built once from the candidate with the host's python3 (the tag
# workflow builds the same bytes); each case then installs it from its own Downloads folder.
env -i "${PASS[@]}" HOME="$WORKDIR/build-home" PATH="$SYSPATH" bash -s <<'BUILD'
set -euo pipefail
mkdir -p "$HOME" && cd "$HOME/.."
git clone -q https://github.com/electricsheephq/lcm-x src && git -C src checkout -q "$CANDIDATE"
python3 src/scripts/build_claude_code_unit.py "$CANDIDATE" --out asset >/dev/null
echo "asset: $(cd asset && ls ./*.tar.gz | xargs -n1 basename) $(cut -c1-16 asset/*.sha256)"
BUILD

# uv for case B (it supplies the 3.10 and, later, the documented 3.11 venv); kept outside the case HOMEs.
env -i "${PASS[@]}" HOME="$WORKDIR/tools-home" PATH="$SYSPATH" UV_INSTALL_DIR="$WORKDIR/tools/bin" UV_NO_MODIFY_PATH=1 \
  bash -c 'mkdir -p "$HOME" && curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null'
UVENV=(UV_PYTHON_INSTALL_DIR="$WORKDIR/tools/python" UV_CACHE_DIR="$WORKDIR/tools/cache" UV_PYTHON_PREFERENCE=only-managed)
env -i "${PASS[@]}" "${UVENV[@]}" HOME="$WORKDIR/tools-home" PATH="$SYSPATH" "$WORKDIR/tools/bin/uv" python install -q 3.10
PY310=$(env -i "${UVENV[@]}" HOME="$WORKDIR/tools-home" PATH="$SYSPATH" "$WORKDIR/tools/bin/uv" python find 3.10)
mkdir -p "$WORKDIR/caseB/py310" "$WORKDIR/caseB/sysbin"
ln -s "$PY310" "$WORKDIR/caseB/py310/python3"; ln -s "$WORKDIR/tools/bin/uv" "$WORKDIR/caseB/py310/uv"
for d in /usr/local/bin /usr/bin /bin; do for f in "$d"/*; do b=$(basename "$f")
  case "$b" in python*|pydoc*|idle*|2to3*) continue;; esac
  [ -e "$WORKDIR/caseB/sysbin/$b" ] || ln -s "$f" "$WORKDIR/caseB/sysbin/$b"; done; done

one_case() {  # $1 case name, $2 PATH after the case HOME's ~/.local/bin
  local home="$WORKDIR/case$1/home"; mkdir -p "$home"
  env -i "${PASS[@]}" "${UVENV[@]}" CASE="$1" ASSET="$WORKDIR/asset" HOME="$home" CLAUDE_CONFIG_DIR="$home/.claude" \
    PATH="$home/.local/bin:$2" bash -s <<'CASE'
set -euo pipefail
cd "$HOME"; T0=$(date +%s)
curl -fsSL https://claude.ai/install.sh | bash >/dev/null 2>&1      # the documented Claude Code install
mkdir -p Downloads && cp "$ASSET"/*.tar.gz "$ASSET"/*.sha256 Downloads/
( cd Downloads && sha256sum -c ./*.sha256 >/dev/null && tar -xzf ./*.tar.gz )
PLUGIN=$(ls -d "$HOME"/Downloads/lcm-x-claude-code-*/)
mkdir -p proj && cd proj && printf 'alpha\nbeta\n' > a.txt
if [ -n "$BASE_URL" ]; then export ANTHROPIC_BASE_URL="$BASE_URL"; fi
# An assignment prefix, not `env`: the key never appears in any process's argument list.
run() { ANTHROPIC_AUTH_TOKEN="$(cat "$KEY_FILE")" \
        claude -p "$1" --plugin-dir "$PLUGIN" --dangerously-skip-permissions --output-format json "${@:2}" < /dev/null \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print("turn", {k: d.get(k) for k in ("session_id","is_error","num_turns")})'; }
outcomes() { python3 -c "import json,collections; print(dict(collections.Counter(json.loads(l)['event']+':'+str(json.loads(l)['outcome']) for l in open('$HOME/.lcm-x/claude-code/logs/hook.log'))))" 2>/dev/null || echo "{}"; }
echo "== case $CASE: claude $(claude --version 2>/dev/null | head -1), python3 on PATH $(python3 -V 2>&1)"
SID=$(python3 -c 'import uuid; print(uuid.uuid4())')
run "Use Bash to run 'wc -l a.txt' and Read a.txt, then report both." --session-id "$SID"
run "Use Bash to run 'seq 1 50000' exactly with no pipe and report the line count." --resume "$SID"
sleep 5; STEPS=0
PY=$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3.11 || true)
if [ -z "$PY" ] && python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY=$(command -v python3); fi
if [ -z "$PY" ]; then
  echo "before step: hook outcomes $(outcomes)"
  echo "one step: uv venv --python 3.11 ~/.lcm-x/claude-code/venv (uv was provided on PATH, not installed by the case)"
  uv venv -q --python 3.11 ~/.lcm-x/claude-code/venv; STEPS=1
  run "Use Bash to run 'echo after-setup' and report it." --resume "$SID"; sleep 5
  PY=~/.lcm-x/claude-code/venv/bin/python
fi
"$PY" -I "$PLUGIN/lcm-x/scripts/lcm_x_capture.py" audit --home ~/.lcm-x/claude-code > audit.json || true
"$PY" -I - "$CASE" "$STEPS" "$T0" "$HOME/.lcm-x/claude-code/lcm.db" <<'PYEND'
import json, sqlite3, sys, time
case, steps, t0, db = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
a = json.load(open("audit.json"))
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
rows = con.execute("select count(*) from messages").fetchone()[0]
print("RECORD", json.dumps({"case": case, "manual_steps": steps, "rows": rows, "audit_ok": a.get("ok"),
      "verified": a.get("verified"), "lineages": len(a.get("lineages") or []),
      "failures": {k: v for k, v in (a.get("failures") or {}).items() if v}, "spool": a.get("spool"),
      "wall_s": int(time.time()) - t0}, sort_keys=True))
PYEND
echo "after: hook outcomes $(outcomes)"
CASE
}
one_case A "$SYSPATH"
one_case B "$WORKDIR/caseB/py310:$WORKDIR/caseB/sysbin"
echo "== done $(date -u +%FT%TZ)"
