#!/usr/bin/env bash
# Score every existing Track S smoke run (S1 lc-runs, S2 lcmx-runs, S4 codex-runs) and render SMOKE-SCORES.md.
set -eu
S=$(cd "$(dirname "$0")" && pwd)
T=${TRACK_S_OUT:?set TRACK_S_OUT}; O=$T/scorer/smoke-scores; M=${TRACK_S_MATERIAL:?set TRACK_S_MATERIAL}/smoke-seed-1
mkdir -p "$O"
sc() { python3 "$S/score_s.py" --material "$M" --run "$1" --arm "$2" --out "$O/$2.smoke-seed-1.r$3.json"; }
sc "$T/lc-runs/smoke-seed-1/1" lossless-claw 1
sc "$T/lc-runs/smoke-seed-1/1" lossless-claw-open 1
for r in 1 2; do sc "$T/lcmx-runs/LCMX-fleet/smoke-1/$r" LCMX-fleet "$r"; done
sc "$T/lcmx-runs/LCMX-fleet-open/smoke-1/1" LCMX-fleet-open 1
sc "$T/lcmx-runs/LCMX-fleet-stub10k/smoke-1/1" LCMX-fleet-stub10k 1
sc "$T/lcmx-runs/C0/smoke-1/1" C0 1
for r in 1 2; do sc "$T/codex-runs/smoke-seed-1/$r" codex-native "$r"; done
python3 "$S/report_s.py" --scores "$O" --out "$T/scorer/SMOKE-SCORES.md" --json "$O/report.json" \
  --title "Track S SMOKE scores (smoke-seed-1, WIRING-ONLY timing; not a gate result)"
