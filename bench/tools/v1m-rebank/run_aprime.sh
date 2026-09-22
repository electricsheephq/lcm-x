#!/usr/bin/env bash
# V1-M RE-BANK (RUN-SHEET-V1M-REBANK, PR #413) — the A′ arm: prepared-m-aprime100 (100 questions, seed 20260802), same
# env/provider/cache/chunk mode as run_shard.sh; fresh output root m-rebank-aprime. Run AFTER the six shards complete
# (sheet §8 step 6: same flags incl. --dump-candidates). Then identity_all.sh computes F53-A′ vs re-bank-A′ and the A/A′ discordance.
set -euo pipefail
: "${REBANK_REPO:?set REBANK_REPO to the worktree at the merged registration sha}"
: "${REBANK_PY:?set REBANK_PY to the python that has the harness deps (F53 used wt-v1m-run2/.venv/bin/python)}"
VOYAGE_API_KEY="$(security find-generic-password -s VOYAGE_API_KEY -w 2>/dev/null)"
[ -n "$VOYAGE_API_KEY" ] || { echo "no voyage key"; exit 78; }
export VOYAGE_API_KEY
OUT=/Users/m1/hermes-work/lme-runs/m-rebank-aprime
export HERMES_HOME=/Users/m1/hermes-work/lme-m-run/rebank-aprime-home
export TMPDIR=/Users/m1/hermes-work/lme-m-run/rebank-aprime-tmp
export LCM_LONGMEMEVAL_FASTEMBED_CACHE=/Users/m1/hermes-work/fastembed-cache
export LCM_LONGMEMEVAL_EMBED_CACHE=/Users/m1/hermes-work/longmemeval-data/embed-cache.sqlite
export LCM_LONGMEMEVAL_CHUNK_EMBEDDING_MODE=flat
export LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS=-1
export LCM_EMBEDDINGS_ENABLED=1
export LCM_EMBEDDING_PROVIDER=voyage
export LCM_EMBEDDING_MODEL=voyage-context-3
mkdir -p "$HERMES_HOME" "$TMPDIR" "$OUT"
env | grep -E '^(LCM_|HERMES_|REBANK_)' | sort > "$OUT/run-env-captured.txt"
git -C "$REBANK_REPO" rev-parse HEAD > "$OUT/product-sha.txt"
cd "$REBANK_REPO"
# --resume only when a checkpoint already exists (see run_shard.sh).
RESUME=(); [ -s "$OUT/per_question_checkpoint.jsonl" ] && RESUME=(--resume)
echo "[rebank-aprime] start $(date -u +%FT%TZ) sha=$(cat "$OUT/product-sha.txt") resume=${#RESUME[@]}"
"$REBANK_PY" scripts/lcm_longmemeval.py run \
  --prepared-dir /Users/m1/hermes-work/longmemeval-data/prepared-m-aprime100 \
  --dataset-label m \
  --provider voyage --model voyage-context-3 \
  --output "$OUT" \
  --allow-external-output \
  --dump-candidates "$OUT/candidates.jsonl" \
  "${RESUME[@]}"
echo "[rebank-aprime] end $(date -u +%FT%TZ) exit=$?"
