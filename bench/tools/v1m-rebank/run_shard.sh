#!/usr/bin/env bash
# V1-M RE-BANK (RUN-SHEET-V1M-REBANK, PR #413) — one shard. Usage: run_shard.sh <k>
# Mirrors the F53 kit (session-notes/2026-08-19/v1m-launch2/run_shard.sh): same prepared shards,
# same shared content-hash embed cache (THE cost model), same provider/model/env; only the product
# tree (REBANK_REPO at the merged registration sha), the venv, and the output/home roots change.
# Shard topology is operational, not measured surface. Fresh output roots are REQUIRED (sheet §7).
set -euo pipefail
K="$1"
: "${REBANK_REPO:?set REBANK_REPO to the worktree at the merged registration sha}"
: "${REBANK_PY:?set REBANK_PY to the python that has the harness deps (F53 used wt-v1m-run2/.venv/bin/python)}"
VOYAGE_API_KEY="$(security find-generic-password -s VOYAGE_API_KEY -w 2>/dev/null)"
[ -n "$VOYAGE_API_KEY" ] || { echo "no voyage key"; exit 78; }
export VOYAGE_API_KEY
OUT=/Users/m1/hermes-work/lme-runs/m-rebank-shard-$K
export HERMES_HOME=/Users/m1/hermes-work/lme-m-run/rebank-$K-home
export TMPDIR=/Users/m1/hermes-work/lme-m-run/rebank-$K-tmp
export LCM_LONGMEMEVAL_FASTEMBED_CACHE=/Users/m1/hermes-work/fastembed-cache
export LCM_LONGMEMEVAL_EMBED_CACHE=/Users/m1/hermes-work/longmemeval-data/embed-cache.sqlite
export LCM_LONGMEMEVAL_CHUNK_EMBEDDING_MODE=flat   # F53 semantics: independent chunk embeddings through the cache (sheet §1; #352 contextual grouping is NOT this row)
export LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS=-1
export LCM_EMBEDDINGS_ENABLED=1
export LCM_EMBEDDING_PROVIDER=voyage
export LCM_EMBEDDING_MODEL=voyage-context-3
mkdir -p "$HERMES_HOME" "$TMPDIR" "$OUT"
env | grep -E '^(LCM_|HERMES_|REBANK_)' | sort > "$OUT/run-env-captured.txt"
git -C "$REBANK_REPO" rev-parse HEAD > "$OUT/product-sha.txt"
cd "$REBANK_REPO"
# --resume only when a checkpoint already exists: the harness refuses --resume on a fresh root ("cannot resume
# without a non-empty checkpoint") and refuses a bare run over an existing one — so the flag is derived from disk.
RESUME=(); [ -s "$OUT/per_question_checkpoint.jsonl" ] && RESUME=(--resume)
echo "[rebank-$K] start $(date -u +%FT%TZ) sha=$(cat "$OUT/product-sha.txt") resume=${#RESUME[@]}"
"$REBANK_PY" scripts/lcm_longmemeval.py run \
  --prepared-dir /Users/m1/hermes-work/longmemeval-data/prepared-m-shards/shard-$K \
  --dataset-label m \
  --provider voyage --model voyage-context-3 \
  --output "$OUT" \
  --allow-external-output \
  --dump-candidates "$OUT/candidates.jsonl" \
  "${RESUME[@]}"
echo "[rebank-$K] end $(date -u +%FT%TZ) exit=$?"
