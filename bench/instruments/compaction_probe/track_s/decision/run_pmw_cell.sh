#!/bin/bash
# Track S per-message-wait cell: one repeat per seed, GLM reader, Codex summariser.
# Only an observed 429 / usage-limit marker brakes the cell (exit 3).
# FACTS_FLOOR is the cp-304 floor the release declares (RELEASE-READINESS-V1 D7: the previous GA's D7 mean - 0.10).
set -eu
if [ "${S2_FAULT_SLEEP_S+x}" = x ]; then
  echo "STOP: S2_FAULT_SLEEP_S must be unset for a cell" >&2; exit 2
fi
FAULT=${FAULT:-}
case "$FAULT" in ''|T|P) ;; *) echo "STOP: FAULT must be empty, T or P" >&2; exit 2;; esac
for var in PRODUCT_WT PRODUCT_SHA KIT_DIR MATERIAL OUT PY GLM_API_KEY FACTS_FLOOR; do
  if [ -z "${!var:-}" ]; then echo "STOP: required environment variable $var is unset" >&2; exit 2; fi
done
export GLM_API_KEY
ARM=${ARM:-LCMX-fleet-rc3}
SEEDS=${SEEDS:-1 2 3}
PARALLEL=${PARALLEL:-2}
case "$PARALLEL" in ''|*[!0-9]*|0) echo "STOP: PARALLEL must be positive" >&2; exit 2;; esac
read -r -a seeds <<< "$SEEDS"
for seed in "${seeds[@]}"; do
  case "$seed" in ''|*[!0-9]*) echo "STOP: SEEDS must contain integers" >&2; exit 2;; esac
done
[ "$(git -C "$PRODUCT_WT" rev-parse HEAD)" = "$PRODUCT_SHA" ] &&
  [ -z "$(git -C "$PRODUCT_WT" status --porcelain)" ] || {
    echo "STOP: product worktree must be clean at PRODUCT_SHA" >&2; exit 2;
  }
if ! kit_sha=$(git -C "$KIT_DIR" rev-parse HEAD) ||
   ! kit_status=$(git -C "$KIT_DIR" status --porcelain --untracked-files=all -- .) ||
   [ -n "$kit_status" ]; then
  echo "STOP: KIT_DIR must be inside a clean git checkout: $KIT_DIR" >&2; exit 2
fi
H=$OUT/decision
mkdir -p "$H/logs" "$OUT/tmp"
[ ! -f "$OUT/STOP" ] || { echo "STOP: existing limit marker" >&2; exit 3; }
"$PY" -B - "$OUT/cell.json" "$PRODUCT_SHA" "$kit_sha" "$ARM" "$SEEDS" "$FACTS_FLOOR" "${FAULT:-}" <<'PY'
import json
import sys
from pathlib import Path

path, product_sha, kit_sha, arm, seeds, floor, fault = sys.argv[1:]
reasons = []
if arm != "LCMX-fleet-rc3":
    reasons.append(f"ARM={arm} is not LCMX-fleet-rc3")
if seeds != "1 2 3":
    reasons.append(f"SEEDS={seeds!r} is not exactly '1 2 3'")
if fault:
    reasons.append(f"FAULT={fault} injects summariser hangs (cell 1096-H)")
if reasons:
    print("NON-GATING: " + "; ".join(reasons), file=sys.stderr)
Path(path).write_text(json.dumps(dict(product_sha=product_sha, kit_sha=kit_sha, arm=arm,
                                     seeds=[int(n) for n in seeds.split()], facts_floor=float(floor), fault=fault or None,
                                     gating=not reasons), indent=1) + "\n")
PY

limited() {
  local seed=$1 file
  for file in "$H/logs/s2-$ARM-d$seed-r1.log" "$OUT/lcmx-runs/$ARM/seed-$seed/d$seed-r1/engine.log"; do
    if [ -f "$file" ] && grep -q -i -E 'HTTP 429|HTTP Error 429|usage[- ]limit|too many requests' "$file"; then
      echo "LIMIT marker in $file" >&2
      touch "$OUT/STOP"
      return 0
    fi
  done
  return 1
}

one() {
  local seed=$1 name pid rc=0 wait_rc
  set --
  [ -z "$FAULT" ] || set -- --fault-hang "$FAULT"
  name=s2-$ARM-d$seed-r1
  echo "start $(date +%s) $(date -u +%FT%TZ)" > "$H/logs/$name.log.wall"
  TRACK_S_OUT=$OUT TMPDIR=$OUT/tmp TRACK_S_MATERIAL=$MATERIAL S2_PRODUCT_WORKTREE=$PRODUCT_WT S2_PRODUCT_SHA=$PRODUCT_SHA \
    perl -e 'setpgrp or die; exec @ARGV or die' nice -n 19 "$PY" -B "$KIT_DIR/s2/run_s_lcmx.py" \
    --arm "$ARM" --seed "$seed" --run "d$seed-r1" --lane codex-sol --reader glm \
    --turn-end-notify --checkpoints 176,304 "$@" > "$H/logs/$name.log" 2>&1 < /dev/null & pid=$!
  trap 'kill -TERM -- "-$pid" 2>/dev/null || true' EXIT
  while kill -0 "$pid" 2>/dev/null; do
    limited "$seed" || true
    if [ -f "$OUT/STOP" ]; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      sleep 1
      kill -KILL -- "-$pid" 2>/dev/null || true
      rc=3
      break
    fi
    sleep 1
  done
  wait "$pid" || { wait_rc=$?; [ "$rc" -ne 0 ] || rc=$wait_rc; }
  limited "$seed" && rc=3
  echo "exit $rc end $(date +%s) $(date -u +%FT%TZ)" >> "$H/logs/$name.log.wall"
  trap - EXIT
  return "$rc"
}

pids=()
trap 'for pid in "${pids[@]:-}"; do [ -z "$pid" ] || kill -TERM "$pid" 2>/dev/null || true; done' EXIT
i=0
while [ "$i" -lt "${#seeds[@]}" ]; do
  pids=()
  for ((slot=0; slot<PARALLEL && i<${#seeds[@]}; slot++, i++)); do
    one "${seeds[$i]}" & pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  pids=()
  [ ! -f "$OUT/STOP" ] || exit 3
  [ "$status" -eq 0 ] || exit "$status"
done
"$PY" -B "$KIT_DIR/decision/score_decision.py" --run-root "$OUT/lcmx-runs" --logs "$H/logs" \
  --material "$MATERIAL" --out "$H" --arms "$ARM" --seeds "${seeds[@]}" --checkpoints 176 304 > "$OUT/score.log" 2>&1
"$PY" -B "$KIT_DIR/decision/pmw_read.py" --run-root "$OUT" --material "$MATERIAL" \
  --arm "$ARM" --seeds "${seeds[@]}" --facts-floor "$FACTS_FLOOR" --json "$OUT/pmw-read.json"
