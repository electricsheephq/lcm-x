#!/usr/bin/env bash
# S8 decision run: run_decision.sh glm <N> | s4 <N>   (seed N = 1..3). Every run goes to the end of the material (row_index
# 304) with probes at rows 176 and 304. glm-lane runs (S1 + S2, summariser + reader glm-5.3, one coding-plan key) run one after
# the other; the S4 writer (codex, no glm call) runs beside them, its two runs of a seed side by side (token-expiry window).
# Each run is stopped (exact PID) at 2x its estimate (minutes). Estimates: S7 walls (lossless-claw 35.6, tuned 38.9); LCMX-fleet
# 30 (S7 12.9 min ran with the summariser blacked out by the spend guard from row ~240; the guard is off here); -open = plain +
# tool rounds; S4 = s4/dry-run seed-1 dictation-file plan (87 turns, smoke fit) + probes.
T=$(cd "$(dirname "$0")/.." && pwd)
: "${TRACK_S_OUT:?set TRACK_S_OUT}" "${TRACK_S_MATERIAL:?set TRACK_S_MATERIAL}" "${S2_PRODUCT_WORKTREE:?set S2_PRODUCT_WORKTREE}"
H=$TRACK_S_OUT/decision; mkdir -p "$H/logs"
run() { name=$1; est=$2; shift 2; L=$H/logs/$name.log
  echo "start $(date +%s) $(date) est ${est} min" > "$L.wall"
  "$@" > "$L" 2>&1 & pid=$!; echo "pid $pid" >> "$L.wall"; lim=$((est * 120)); t=0
  while ps -p $pid > /dev/null 2>&1; do sleep 10; t=$((t + 10))
    if [ $t -ge $lim ]; then kill $pid; echo "STOPPED at 2x estimate (pid $pid)" >> "$L.wall"; fi; done
  wait $pid; echo "exit $? end $(date +%s) $(date)" >> "$L.wall"; }
PY=${PYTHON:-python3}; N=$2; LC_WORKTREE=${3:-}; CODEX_AUTH=${4:-}; CP="--checkpoints 176,304"
case "$1" in
glm)
run s2-LCMX-fleet-d$N-r1 30 "$PY" -B "$T/s2/run_s_lcmx.py" --arm LCMX-fleet --seed $N --run d$N-r1 --lane glm --reader glm $CP
run s2-LCMX-fleet-d$N-r2 30 "$PY" -B "$T/s2/run_s_lcmx.py" --arm LCMX-fleet --seed $N --run d$N-r2 --lane glm --reader glm $CP
run s1-lossless-claw-d$N-r1 36 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-r1-default --arm lossless-claw $CP
run s1-lossless-claw-d$N-r2 36 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-r2-default --arm lossless-claw $CP
run s1-lossless-claw-tuned-d$N-r1 39 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-r1-tuned --arm lossless-claw-tuned $CP
run s1-lossless-claw-tuned-d$N-r2 39 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-r2-tuned --arm lossless-claw-tuned $CP
run s2-LCMX-fleet-open-d$N-open 40 "$PY" -B "$T/s2/run_s_lcmx.py" --arm LCMX-fleet-open --seed $N --run d$N-open --lane glm --reader glm $CP
run s1-lossless-claw-open-d$N-open 50 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-open-default --arm lossless-claw --open $CP
run s1-lossless-claw-tuned-open-d$N-open 50 "$T/s1/run.sh" "${LC_WORKTREE:?pass lossless checkout as argument 3}" seed-$N d$N-open-tuned --arm lossless-claw-tuned --open $CP
;;
s4)
S4=$T/s4/run_s_codex.py
run s4-codex-native-d$N-r1 160 "$PY" -B "$S4" --seed "$N" --dictation file --stop-row 304 --auth-file "${CODEX_AUTH:?pass CLI auth file as argument 4}" --run d$N-r1 &
sleep 30; run s4-codex-native-d$N-r2 160 "$PY" -B "$S4" --seed "$N" --dictation file --stop-row 304 --auth-file "${CODEX_AUTH:?pass CLI auth file as argument 4}" --run d$N-r2 &
wait
;;
*) echo "usage: run_decision.sh glm|s4 <seed>" >&2; exit 2;;
esac
