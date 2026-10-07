#!/usr/bin/env bash
# rc3 aged-tier gate; serial GLM summariser/reader. r1 (default), r2, or one smoke pair.
set -eu
T=$(cd "$(dirname "$0")/.." && pwd)
H=${TRACK_S_OUT:?set TRACK_S_OUT}/stub2k-gate
mkdir -p "$H/logs"
: "${S2_PRODUCT_WORKTREE:?set S2_PRODUCT_WORKTREE}" "${TRACK_S_MATERIAL:?set TRACK_S_MATERIAL}"
export S2_PRODUCT_SHA=${S2_PRODUCT_SHA:-0b26ef7d6ce63ad430f4e676464eb4eaae428c92}
[ "$S2_PRODUCT_SHA" = 0b26ef7d6ce63ad430f4e676464eb4eaae428c92 ] || { echo 'gate requires rc3 SHA' >&2; exit 2; }
MODE=${1:-r1}
case "$MODE" in r1|r2|smoke) ;; *) echo 'usage: run_stub2k_gate.sh [r1|r2|smoke]' >&2; exit 2;; esac
# A second invocation must not share the single GLM lane. A crash leaves a visible lock for owner inspection.
mkdir "$H/.glm-lane-lock" || { echo 'GLM gate lane already locked' >&2; exit 2; }
pid=
cleanup() { if [ -n "$pid" ]; then kill -TERM "$pid" 2>/dev/null || :; fi; rmdir "$H/.glm-lane-lock"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run() { name=$1; est=$2; shift 2; L=$H/logs/$name.log
  echo "start $(date +%s) $(date) est ${est} min" > "$L.wall"
  "$@" > "$L" 2>&1 & pid=$!; echo "pid $pid" >> "$L.wall"; lim=$((est * 120)); t=0; stopped=0
  while ps -p "$pid" >/dev/null 2>&1; do sleep 10; t=$((t + 10))
    if [ "$t" -ge "$lim" ]; then
      kill -TERM "$pid" 2>/dev/null || :
      echo "STOPPED at 2x estimate (pid $pid)" >> "$L.wall"; stopped=1; break
    fi
  done
  if [ "$stopped" -eq 1 ]; then sleep 2; kill -KILL "$pid" 2>/dev/null || :; fi
  rc=0; wait "$pid" || rc=$?; pid=
  echo "exit $rc end $(date +%s) $(date)" >> "$L.wall"
  return "$rc"
}
PY=${PYTHON:-python3}
# Adapt only output location and metadata, keeping run_s_lcmx.py and scoring/material unchanged.
ADAPTER=$(cat <<'PY'
import importlib.util, os, sys
from pathlib import Path
t = Path(sys.argv.pop(1))
spec = importlib.util.spec_from_file_location("gate_s2_runner", t / "s2/run_s_lcmx.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.RUNS = Path(os.environ["TRACK_S_OUT"]).resolve() / "stub2k-gate/runs"
original = runner.config_dict
def config_receipt(cfg):
    out = original(cfg)
    engine = sys.modules["hermes_lcm.engine"].LCMEngine.__new__(sys.modules["hermes_lcm.engine"].LCMEngine)
    engine._config = cfg
    out.update(product_sha=runner.seam.PINNED,
               aged_tier_resolved_tokens=engine._active_replay_stub_aged_threshold_tokens())
    return out
runner.config_dict = config_receipt
runner.main()
PY
)
if [ "$MODE" = smoke ]; then
  for arm in LCMX-fleet-agedoff LCMX-fleet; do
    id=g9-$arm-smoke-1-r1
    run "$id" 5 "$PY" -B -c "$ADAPTER" "$T" --arm "$arm" --seed smoke-1 --run "$id" --lane glm --reader glm
  done
else
  for N in 1 2 3; do
    for arm in LCMX-fleet LCMX-fleet-agedoff; do
      id=g9-$arm-d$N-$MODE
      run "$id" "${G9_EST_MIN:-30}" "$PY" -B -c "$ADAPTER" "$T" --arm "$arm" --seed "$N" --run "$id" \
        --lane glm --reader glm --checkpoints 176,304
    done
  done
fi
