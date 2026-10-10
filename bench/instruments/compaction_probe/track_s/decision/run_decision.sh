#!/usr/bin/env bash
# S8 decision run: run_decision.sh glm <N> | s4 <N>   (seed N = 1..3). Every run goes to the end of the material (row_index
# 304) with probes at rows 176 and 304. glm-lane runs (S1 + S2, summariser + reader glm-5.3, one coding-plan key) run one after
# the other; the S4 writer (codex, no glm call) runs beside them, its two runs of a seed side by side (token-expiry window).
# Each run is stopped (process group) at >=3x its estimate (minutes). Estimates: S7 walls (lossless-claw 35.6, tuned 38.9); LCMX-fleet
# 30 (S7 12.9 min ran with the summariser blacked out by the spend guard from row ~240; the guard is off here); -open = plain +
# tool rounds; S4 = s4/dry-run seed-1 dictation-file plan (87 turns, smoke fit) + probes.
# A failed or stopped run does not stop the lane: the other arms still run, and the script exits nonzero at the end.
T=$(cd "$(dirname "$0")/.." && pwd)
: "${TRACK_S_OUT:?set TRACK_S_OUT}" "${TRACK_S_MATERIAL:?set TRACK_S_MATERIAL}"
H=$TRACK_S_OUT/decision; mkdir -p "$H/logs"
run() { local name=$1 est=$2 L pid lim t rc timed_out=0; shift 2; L=$H/logs/$name.log
  echo "start $(date +%s) $(date) est ${est} min" > "$L.wall"
  perl -e 'setpgrp or die "setpgrp: $!"; exec @ARGV or die "exec: $!"' "$@" > "$L" 2>&1 & pid=$!; echo "pid $pid" >> "$L.wall"; lim=$((est * ${S2_TIMEOUT_MULTIPLIER:-3} * 60)); [ "${S2_TIMEOUT_MULTIPLIER:-3}" -ge 3 ] || return 2; t=0
  while ps -p $pid > /dev/null 2>&1; do sleep 10; t=$((t + 10))
    if [ $t -ge $lim ]; then
      timed_out=1; kill -TERM -- "-$pid" 2>/dev/null || true
      echo "STOPPED at configured cap (>=3x estimate) (process group $pid)" >> "$L.wall"
      sleep 10; kill -KILL -- "-$pid" 2>/dev/null || true; break
    fi; done
  rc=0; wait "$pid" || rc=$?; [ "$timed_out" = 0 ] || rc=124
  echo "exit $rc end $(date +%s) $(date)" >> "$L.wall"; return "$rc"; }
PY=${PYTHON:-python3}; N=${2:-}; LC_WORKTREE=${3:-}; CODEX_AUTH=${4:-}; CP="--checkpoints 176,304"
if [[ "$1" = eval2* ]]; then
  status=0
  for N in ${2:-1 2 3 4 5 6 7 8}; do
    est=$("$PY" -c 'import json,math,os,sys; m=json.load(open(sys.argv[1])); print(math.ceil(max(60,m["decision_checkpoint"]["tokens"]/float(os.environ.get("S2_REPLAY_TOKENS_PER_SECOND","333")))/60))' "$TRACK_S_MATERIAL/seed-$N/material.manifest.json")
    if [ "$1" = eval2-c ]; then
      run "s2-L1-d$N-astra" "$est" "$PY" -B "$T/s2/run_s_lcmx.py" --arm L1 --reader astra-low --seed "$N" --run "d$N-astra" --checkpoints lifecycle || status=1
      run "s4-codex-native-d$N-astra" 160 "$PY" -B "$T/s4/run_s_codex.py" --seed "$N" --run "d$N-astra" --auth-file "${CODEX_AUTH:?}" --checkpoints lifecycle || status=1
      continue
    fi
    for arm in L0 L1 L1-H L1-noptr H; do
      driver=("$PY" -B "$T/s2/run_s_lcmx.py" --arm "$arm")
      [ "$arm" != H ] || driver=(bash "$T/s2/launch_h.sh")
      run "s2-$arm-d$N-r1" "$est" "${driver[@]}" --seed "$N" --run "d$N-r1" --checkpoints lifecycle || status=1
    done
  done
  exit "$status"
fi
case "$1" in
glm)
status=0
run s2-LCMX-fleet-d$N-r1 30 "$PY" -B "$T/s2/run_s_lcmx.py" --arm LCMX-fleet --seed $N --run d$N-r1 --lane glm --reader glm $CP || status=1
run s2-LCMX-fleet-d$N-r2 30 "$PY" -B "$T/s2/run_s_lcmx.py" --arm LCMX-fleet --seed $N --run d$N-r2 --lane glm --reader glm $CP || status=1
run s1-lossless-claw-d$N-r1 36 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-r1-default --arm lossless-claw $CP || status=1
run s1-lossless-claw-d$N-r2 36 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-r2-default --arm lossless-claw $CP || status=1
run s1-lossless-claw-tuned-d$N-r1 39 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-r1-tuned --arm lossless-claw-tuned $CP || status=1
run s1-lossless-claw-tuned-d$N-r2 39 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-r2-tuned --arm lossless-claw-tuned $CP || status=1
run s2-LCMX-fleet-open-d$N-open 40 "$PY" -B "$T/s2/run_s_lcmx.py" --arm LCMX-fleet-open --seed $N --run d$N-open --lane glm --reader glm $CP || status=1
run s1-lossless-claw-open-d$N-open 50 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-open-default --arm lossless-claw --open $CP || status=1
run s1-lossless-claw-tuned-open-d$N-open 50 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-open-tuned --arm lossless-claw-tuned --open $CP || status=1
exit "$status"
;;
s4)
S4=$T/s4/run_s_codex.py
run s4-codex-native-d$N-r1 160 "$PY" -B "$S4" --seed "$N" --dictation file --stop-row 304 --auth-file "${CODEX_AUTH:?pass CLI auth file as argument 4}" --run d$N-r1 &
p1=$!
sleep 30; run s4-codex-native-d$N-r2 160 "$PY" -B "$S4" --seed "$N" --dictation file --stop-row 304 --auth-file "${CODEX_AUTH:?pass CLI auth file as argument 4}" --run d$N-r2 &
p2=$!; status=0
wait "$p1" || status=$?; wait "$p2" || status=$?
exit "$status"
;;
*) echo "usage: run_decision.sh glm|s4 <seed>" >&2; exit 2;;
esac
