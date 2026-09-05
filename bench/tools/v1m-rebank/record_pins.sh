#!/bin/zsh
# Record the RUN-SHEET-V1M-REBANK §3 pins — every value from a command, none hand-typed.
# Usage: REBANK_REPO=<worktree at merged main> ./record_pins.sh [phase]   (phase = launch | postrun; default launch)
set -uo pipefail
: "${REBANK_REPO:?set REBANK_REPO to the worktree at the merged registration sha}"
PHASE="${1:-launch}"
D=/Users/m1/hermes-work/longmemeval-data
OUTDIR=/Users/m1/Codex/session-notes/2026-09-05/v1m-rebank/artifacts; mkdir -p "$OUTDIR"
OUT="$OUTDIR/pins-rebank-$PHASE-$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "# V1-M re-bank pins ($PHASE) — recorded $(date -u +%FT%TZ) by record_pins.sh"
  echo "## repo"
  echo "path=$REBANK_REPO"
  echo "head=$(git -C "$REBANK_REPO" rev-parse HEAD)"
  echo "branch=$(git -C "$REBANK_REPO" rev-parse --abbrev-ref HEAD)"
  echo "origin_main=$(git -C "$REBANK_REPO" rev-parse origin/main 2>/dev/null)"
  echo "dirty_files=$(git -C "$REBANK_REPO" status --porcelain | wc -l | tr -d ' ')"
  echo "## blob shas"
  for f in benchmarking/longmemeval.py scripts/lcm_longmemeval.py ingest_protection.py config.py bench/specs/RUN-SHEET-V1M-REBANK.md; do
    echo "$f=$(git -C "$REBANK_REPO" rev-parse "HEAD:$f" 2>/dev/null || echo MISSING)"
  done
  echo "## dataset + prepared manifests (sha256)"
  for f in "$D/longmemeval_m" "$D/prepared-m/manifest.json" "$D/prepared-m-aprime100/manifest.json" $D/prepared-m-shards/shard-{0,1,2,3,4,5}/manifest.json; do
    if [ -f "$f" ]; then echo "$(shasum -a 256 "$f" | awk '{print $1}')  $f"; else echo "MISSING  $f"; fi
  done
  echo "## embed cache"
  ls -la "$D/embed-cache.sqlite" | awk '{print "size="$5" mtime="$6" "$7" "$8}'
  for t in $(sqlite3 "$D/embed-cache.sqlite" ".tables" 2>/dev/null); do echo "rows[$t]=$(sqlite3 "$D/embed-cache.sqlite" "select count(*) from $t" 2>/dev/null)"; done
  echo "## provider"
  echo "voyage_key_in_keychain=$(security find-generic-password -s VOYAGE_API_KEY -w >/dev/null 2>&1 && echo yes || echo NO)"
  echo "python=$(${REBANK_PY:-python3} --version 2>&1)  (${REBANK_PY:-python3})"
  echo "## _EnvFieldSpec inventory (config.py) — name=env_var=value-in-this-shell ((unset) = product default applies)"
  # Structural inventory (PR #416 review): import ENV_FIELD_SPECS instead of regexing config.py (the single-line regex dropped
  # multi-line declarations), assert emitted == declared, and REDACT the value of every credential-shaped variable
  # (KEY|TOKEN|SECRET|PASS|COOKIE|AUTH|CREDENTIAL) — this file is retained and shared as evidence.
  "${REBANK_PY:-python3}" - "$REBANK_REPO" <<'PYEOF'
import os, re, sys
repo = sys.argv[1]
sys.path.insert(0, repo)
import benchmarking.longmemeval as lme  # noqa: E402
lme._ensure_hermes_lcm_package()
from hermes_lcm import config as cfg  # noqa: E402
SECRET = re.compile(r"KEY|TOKEN|SECRET|PASS|COOKIE|AUTH|CREDENTIAL", re.I)
def shown(var):
    if var not in os.environ:
        return "(unset)"
    return "(set, redacted)" if SECRET.search(var) else os.environ[var]
emitted = 0
for spec in cfg.ENV_FIELD_SPECS:
    print(f"{spec.name}={spec.env_key}={shown(spec.env_key)}")
    emitted += 1
declared = len(cfg.ENV_FIELD_SPECS)
print(f"inventory_count={declared} emitted={emitted}")
if emitted != declared:
    sys.exit(f"env inventory incomplete: emitted {emitted} of {declared}")
print("## LCM_/HERMES_/REBANK_ env in this shell (credential-shaped values redacted)")
for var in sorted(os.environ):
    if var.startswith(("LCM_", "HERMES_", "REBANK_")):
        print(f"{var}={shown(var)}")
PYEOF
  if [ "$PHASE" = "postrun" ]; then
    echo "## checkpoint headers (provider identity / privacy revision) per shard"
    for k in 0 1 2 3 4 5; do f=/Users/m1/hermes-work/lme-runs/m-rebank-shard-$k/per_question_checkpoint.jsonl; [ -f "$f" ] && echo "shard-$k $(head -1 "$f" | cut -c1-600)" || echo "shard-$k MISSING"; done
  fi
} | tee "$OUT"
echo "pins written: $OUT"
