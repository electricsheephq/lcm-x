#!/usr/bin/env bash
# S8 continuation (release manager, after the coder lane died on a quota error): seeds 2 and 3 with the coder's own
# run_decision.sh. glm runs one after the other (one coding-plan key); the S4 writer runs beside them, as in seed 1.
S=$(cd "$(dirname "$0")" && pwd)
: "${TRACK_S_OUT:?set TRACK_S_OUT}"
H=$TRACK_S_OUT/decision; mkdir -p "$H"
echo "seeds-2-3 start $(date -u +%FT%TZ)" >> "$H/RUNLOG.txt"
( bash "$S/run_decision.sh" glm 2 "${1:?lossless checkout}"; bash "$S/run_decision.sh" glm 3 "${1:?lossless checkout}" ) &
( bash "$S/run_decision.sh" s4 2 "${1:?lossless checkout}" "${2:?CLI auth file}"; bash "$S/run_decision.sh" s4 3 "${1:?lossless checkout}" "${2:?CLI auth file}" ) &
wait
echo "seeds-2-3 done $(date -u +%FT%TZ)" >> "$H/RUNLOG.txt"
echo SEEDS_2_3_DONE
