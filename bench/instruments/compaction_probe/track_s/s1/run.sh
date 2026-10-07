#!/usr/bin/env bash
# S1 runner: run.sh <lossless-checkout> <seed> <run> [--slice N] [--dry-run] [--batches N] [--open-probes id,id] [--cadence-s 20]
# Bundles replay.ts against the pinned worktree (lc-wt @ 988dee8) with esbuild, then runs it under node with
# HOME, LCM_LOG_FILE and the database inside lc-runs/<seed>/. Only LCM_LOG_FILE is set: the plugin config is DEFAULT.
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: run.sh <lossless-checkout> <seed> <run> [--slice N] [--dry-run] [...]" >&2; exit 2; }
WT=$1 SEED=$2 RUN=$3; shift 3
S1=$(cd "$(dirname "$0")" && pwd)
TS=${TRACK_S_OUT:?set TRACK_S_OUT}; MATERIAL=${TRACK_S_MATERIAL:?set TRACK_S_MATERIAL}
[ "$(git -C "$WT" rev-parse HEAD)" = 988dee85592b9066ffe1c859542e8e18c23f2345 ] || { echo "lc-wt is not at 988dee8" >&2; exit 3; }
git -C "$WT" diff --quiet HEAD || { echo "lc-wt has tracked modifications" >&2; exit 3; }
[ -d "$MATERIAL/$SEED" ] || { echo "no material: $MATERIAL/$SEED" >&2; exit 2; }
mkdir -p "$TS/s1/build" "$TS/lc-runs/$SEED/home" "$TS/lc-runs/$SEED/$RUN"
"$WT/node_modules/.bin/esbuild" "$S1/replay.ts" --bundle --platform=node --target=node22 --format=esm \
  --alias:track-s-lossless="$WT" --outfile="$TS/s1/build/replay.mjs" --external:openclaw --external:'@earendil-works/*' --log-level=warning
for v in $(env | grep -o '^LCM_[A-Z_]*' || true); do unset "$v"; done
export HOME="$TS/lc-runs/$SEED/home" OPENCLAW_STATE_DIR="$TS/lc-runs/$SEED/home/.openclaw"
export LCM_LOG_FILE="$TS/lc-runs/$SEED/$RUN/lcm.log" TMPDIR="$TS/lc-runs/$SEED/home/tmp"
mkdir -p "$TMPDIR"
exec node --no-warnings "$TS/s1/build/replay.mjs" --worktree "$WT" --seed "$SEED" --run "$RUN" "$@"
