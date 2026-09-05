#!/usr/bin/env bash
# RUN-SHEET-V1M-REBANK §4.3 bar 3 — run the identity projection over every F53 ↔ re-bank pair BEFORE naming an outcome,
# plus the §4 bar 5 A/A′ discordance (re-bank A′ vs the same 100 ids in the re-bank shards).
#   pairs 1–6 : F53 shard K (lme-runs/m-full2-shard-K)  vs  re-bank shard K (REBANK_SHARD_PREFIX + K)
#   pair  7   : F53 A′ (lme-runs/m-full2-aprime, 100 q)  vs  re-bank A′ (REBANK_APRIME_DIR)
#   pair  8   : re-bank A′  vs  re-bank shards restricted to A′'s ids (the A/A′ discordance; deterministic metric → expect 0)
# Exit 0 = every pair identical; exit 1 = deltas somewhere (then the §4.3 MOVED-* diagnosis applies row by row); exit 2 = usage/missing input.
# Self-test against F53 itself: REBANK_SHARD_PREFIX=/Users/m1/hermes-work/lme-runs/m-full2-shard- REBANK_APRIME_DIR=/Users/m1/hermes-work/lme-runs/m-full2-aprime
set -euo pipefail
R=/Users/m1/Codex/session-notes/2026-09-05/v1m-rebank
PY=${REBANK_PY:?export REBANK_PY (the F53 venv python)}
F53=/Users/m1/hermes-work/lme-runs
REBANK_SHARD_PREFIX=${REBANK_SHARD_PREFIX:-/Users/m1/hermes-work/lme-runs/m-rebank-shard-}
REBANK_APRIME_DIR=${REBANK_APRIME_DIR:-/Users/m1/hermes-work/lme-runs/m-rebank-aprime}
OUT=${1:-$R/artifacts/identity}
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARGS=()
for K in 0 1 2 3 4 5; do
  A="$F53/m-full2-shard-$K/per_question_checkpoint.jsonl"; B="$REBANK_SHARD_PREFIX$K/per_question_checkpoint.jsonl"
  [ -s "$A" ] || { echo "missing F53 shard $K: $A" >&2; exit 2; }
  [ -s "$B" ] || { echo "missing re-bank shard $K: $B" >&2; exit 2; }
  ARGS+=("$A" "$B")
done
echo "# identity_all $STAMP — six shard pairs (F53 vs re-bank)" | tee "$OUT/identity-shards-$STAMP.txt"
set +e
"$PY" "$R/result_identity.py" "${ARGS[@]}" | tee -a "$OUT/identity-shards-$STAMP.txt"
RC_SHARDS=${PIPESTATUS[0]}
set -e
RC_APRIME=0; RC_AA=0
APRIME="$REBANK_APRIME_DIR/per_question_checkpoint.jsonl"
if [ -s "$APRIME" ]; then
  echo "# identity_all $STAMP — F53 A′ vs re-bank A′" | tee "$OUT/identity-aprime-$STAMP.txt"
  set +e
  "$PY" "$R/result_identity.py" "$F53/m-full2-aprime/per_question_checkpoint.jsonl" "$APRIME" | tee -a "$OUT/identity-aprime-$STAMP.txt"
  RC_APRIME=${PIPESTATUS[0]}
  set -e
  # A/A′ discordance: the re-bank shards' rows for exactly A′'s question ids, as one file (header copied from shard 0).
  RESTRICTED="$OUT/rebank-shards-restricted-to-aprime-ids-$STAMP.jsonl"
  "$PY" - "$APRIME" "$RESTRICTED" "${REBANK_SHARD_PREFIX}"{0,1,2,3,4,5}/per_question_checkpoint.jsonl <<'EOF'
import json, sys
aprime, out, *shards = sys.argv[1:]
ids = {json.loads(l)["question_id"] for l in open(aprime) if l.strip() and "question_id" in json.loads(l)}
header = None; rows = []
for path in shards:
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        if "question_id" not in rec:
            header = header or line
            continue
        if rec["question_id"] in ids:
            rows.append(line)
with open(out, "w") as fh:
    fh.write(header if header else "")
    fh.writelines(rows)
print(f"restricted file: {len(rows)} rows for {len(ids)} A′ ids")
EOF
  echo "# identity_all $STAMP — re-bank A′ vs re-bank shards restricted to A′ ids (A/A′ discordance)" | tee "$OUT/identity-aa-$STAMP.txt"
  set +e
  "$PY" "$R/result_identity.py" "$APRIME" "$RESTRICTED" | tee -a "$OUT/identity-aa-$STAMP.txt"
  RC_AA=${PIPESTATUS[0]}
  set -e
else
  # A′ is a mandatory input for naming any outcome (sheet §4.3/§8 step 6): its absence is a missing-input status, never a pass
  # (PR #416 review) — RC 2 so the final test below exits 2, not 0.
  echo "re-bank A′ dir not present ($REBANK_APRIME_DIR) — MISSING MANDATORY INPUT; shard pairs computed, no outcome may be named" | tee "$OUT/identity-aprime-$STAMP.txt"
  RC_APRIME=2; RC_AA=2
fi
echo "RC shards=$RC_SHARDS aprime=$RC_APRIME aa=$RC_AA" | tee "$OUT/identity-rc-$STAMP.txt"
# exit 2 = invalid input (result_identity.py's own code) must not collapse into 1 (= a confirmed identity delta) — review of PR #416
if [ "$RC_SHARDS" = 2 ] || [ "$RC_APRIME" = 2 ] || [ "$RC_AA" = 2 ]; then echo "INVALID INPUT in at least one pair — no delta verdict"; exit 2; fi
[ "$RC_SHARDS" = 0 ] && [ "$RC_APRIME" = 0 ] && [ "$RC_AA" = 0 ]
