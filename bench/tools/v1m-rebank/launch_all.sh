#!/usr/bin/env bash
# Launch the 6 re-bank shards with the F53 5-minute stagger. Requires REBANK_REPO + REBANK_PY exported.
# PRECONDITIONS (sheet §5/§8, r3): registration merged; `prewarm_gate.sh` exited 0 (probe → prewarm-cache --dry-run →
# projected spend under the $40 cap → real prewarm; evidence in artifacts/prewarm-gate/); `record_pins.sh launch` done.
# Do NOT run before those. After the run: `cache_pair_check.py` (per-shard pair = sum of per-question embed_cache rows vs F53).
set -euo pipefail
: "${REBANK_REPO:?}"; : "${REBANK_PY:?}"
R=/Users/m1/Codex/session-notes/2026-09-05/v1m-rebank
for K in 0 1 2 3 4 5; do
  nohup "$R/run_shard.sh" "$K" > "$R/artifacts/shard-$K.log" 2>&1 &
  echo "shard-$K pid $!"
  [ "$K" -lt 5 ] && sleep 300
done
