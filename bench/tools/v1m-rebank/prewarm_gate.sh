#!/bin/zsh
# Re-bank pre-spend gate (RUN-SHEET-V1M-REBANK §5/§8, r3 semantics):
#   determinism-probe (sample-scoped, no cache) → prewarm-cache --dry-run (lookups + privacy validation, no spend)
#   → projected spend vs the $40 cap → real prewarm-cache → consistency checks.
# Requires: REBANK_REPO (worktree at the merged registration sha), REBANK_PY (venv python), VOYAGE_API_KEY exported
# by the caller from Keychain (never printed here). Evidence → $EVID (default: this dir's artifacts/prewarm-gate/).
# Exit 0 = proceed to launch_all.sh. Exit 3 = PARK (blocked / over cap / would_populate>0 without REBANK_ACCEPT_SPEND=1).
set -uo pipefail
: "${REBANK_REPO:?export REBANK_REPO=<worktree at merged main>}"; : "${REBANK_PY:?export REBANK_PY=<venv python>}"
: "${VOYAGE_API_KEY:?export VOYAGE_API_KEY from Keychain first (security find-generic-password -s VOYAGE_API_KEY -w)}"
export LCM_LONGMEMEVAL_EMBED_CACHE=${LCM_LONGMEMEVAL_EMBED_CACHE:-/Users/m1/hermes-work/longmemeval-data/embed-cache.sqlite}
export LCM_LONGMEMEVAL_CHUNK_EMBEDDING_MODE=flat   # prewarm populates flat units; the run must embed the same units (sheet §1)
PREPARED_DIR=${PREPARED_DIR:-/Users/m1/hermes-work/longmemeval-data/prepared-m}
SHARDS_MANIFEST=${SHARDS_MANIFEST:-/Users/m1/hermes-work/longmemeval-data/prepared-m-shards}
MODEL=voyage-context-3
CAP_USD=40
CORPUS_UNITS=505695          # F53 cache entries — the full-re-embed worst case the $15–40 basis was scaled from
EVID=${EVID:-/Users/m1/Codex/session-notes/2026-09-05/v1m-rebank/artifacts/prewarm-gate}
mkdir -p "$EVID"
for p in "$PREPARED_DIR" "$SHARDS_MANIFEST" "$LCM_LONGMEMEVAL_EMBED_CACHE"; do [ -e "$p" ] || { echo "missing: $p"; exit 2; }; done
CLI="$REBANK_REPO/scripts/lcm_longmemeval.py"
echo "[gate] repo=$(git -C "$REBANK_REPO" rev-parse --short HEAD) cache=$LCM_LONGMEMEVAL_EMBED_CACHE evid=$EVID"

# The CLI interleaves progress lines with its JSON report on stdout; keep the last top-level JSON object.
lastjson() { python3 -c '
import sys, json
lines = sys.stdin.read().splitlines()
starts = [i for i, l in enumerate(lines) if l.startswith("{")]
if not starts: sys.exit("no JSON object in output")
print(json.dumps(json.loads("\n".join(lines[starts[-1]:]))))'; }
field() { python3 -c 'import sys, json; d = json.load(open(sys.argv[1])); v = d
for k in sys.argv[2].split("."): v = v[k]
print(v)' "$1" "$2"; }

# 1. determinism probe — sample-scoped transform count; uses no cache and reports no hit rate.
"$REBANK_PY" "$CLI" determinism-probe --prepared-dir "$PREPARED_DIR" --shards-manifest "$SHARDS_MANIFEST" \
  --dataset-label m --model "$MODEL" --sample-size 20 --seed 0 > "$EVID/probe.stdout" 2> "$EVID/probe.stderr"
RC=$?; lastjson < "$EVID/probe.stdout" > "$EVID/probe.json" || { echo "[gate] probe produced no JSON (rc=$RC)"; exit 3; }
if [ $RC -ne 0 ]; then echo "[gate] PARK — probe rc=$RC status=$(field "$EVID/probe.json" status 2>/dev/null)"; exit 3; fi
echo "[gate] probe ok: privacy=$(field "$EVID/probe.json" privacy) scope=$(field "$EVID/probe.json" privacy_scope)"

# 2. dry run — cache lookups + privacy validation, no embedding call, no spend.
rm -f "$EVID/changed-manifest.jsonl"   # since r5 --changed-manifest truncates on open (and r7 refuses a path that aliases the cache); the rm keeps each gate run auditable from an empty file
"$REBANK_PY" "$CLI" prewarm-cache --prepared-dir "$PREPARED_DIR" --shards-manifest "$SHARDS_MANIFEST" \
  --dataset-label m --provider voyage --model "$MODEL" --dry-run --changed-manifest "$EVID/changed-manifest.jsonl" > "$EVID/dry-run.stdout" 2> "$EVID/dry-run.stderr"
RC=$?; lastjson < "$EVID/dry-run.stdout" > "$EVID/dry-run.json" || { echo "[gate] dry-run produced no JSON (rc=$RC)"; exit 3; }
if [ $RC -ne 0 ]; then echo "[gate] PARK — dry-run rc=$RC status=$(field "$EVID/dry-run.json" status 2>/dev/null) privacy=$(field "$EVID/dry-run.json" privacy 2>/dev/null)"; exit 3; fi
WOULD=$(field "$EVID/dry-run.json" would_populate); CACHED=$(field "$EVID/dry-run.json" already_cached); UNITS=$(field "$EVID/dry-run.json" unique_request_units)
CHANGED=$(field "$EVID/dry-run.json" privacy.changed); BLOCKED=$(field "$EVID/dry-run.json" privacy.blocked)
# r23: the scan is corpus-wide only if the shard union equals the prepared manifest's question set — the report says so itself.
SCOPE=$(field "$EVID/dry-run.json" privacy_scope); SEL=$(field "$EVID/dry-run.json" question_coverage.selected); PREP=$(field "$EVID/dry-run.json" question_coverage.prepared)
if [ "$SCOPE" != "corpus" ] || [ -z "$SEL" ] || [ -z "$PREP" ] || [ "$SEL" != "$PREP" ]; then echo "[gate] PARK — dry run covered '$SEL' of '$PREP' prepared questions (privacy_scope=$SCOPE): the shard union is not the corpus (empty counts = missing report key, fail closed)"; exit 3; fi
echo "[gate] coverage: $SEL/$PREP prepared questions, privacy_scope=$SCOPE"
PROJ=$(python3 -c "print(round($WOULD * $CAP_USD / $CORPUS_UNITS, 2))")
echo "[gate] dry-run: unique=$UNITS already_cached=$CACHED would_populate=$WOULD transform_changed=$CHANGED blocked=$BLOCKED projected_usd=$PROJ (cap $CAP_USD)"
if [ "$BLOCKED" != "0" ]; then echo "[gate] PARK — privacy block in the corpus scan (§7: one block is a park)"; exit 3; fi
if python3 -c "import sys; sys.exit(0 if $PROJ >= $CAP_USD else 1)"; then echo "[gate] PARK — projected prewarm spend reaches the cap (>= $CAP_USD; sheet §5 r8)"; exit 3; fi
if [ "$WOULD" != "0" ] && [ "${REBANK_ACCEPT_SPEND:-0}" != "1" ]; then
  echo "[gate] PARK — would_populate=$WOULD (>0: the transform is live on this corpus; expected 0). Inspect $EVID/dry-run.json,"
  echo "        reconcile with §4.2/§4.3 (MOVED-* or REPRODUCED-TRANSFORM-INERT territory), then rerun with REBANK_ACCEPT_SPEND=1 to embed."
  exit 3
fi

# 3. real prewarm — the only spend-bearing step, taken after the cap check.
"$REBANK_PY" "$CLI" prewarm-cache --prepared-dir "$PREPARED_DIR" --shards-manifest "$SHARDS_MANIFEST" \
  --dataset-label m --provider voyage --model "$MODEL" > "$EVID/prewarm.stdout" 2> "$EVID/prewarm.stderr"
RC=$?; lastjson < "$EVID/prewarm.stdout" > "$EVID/prewarm.json" || { echo "[gate] prewarm produced no JSON (rc=$RC)"; exit 3; }
if [ $RC -ne 0 ]; then echo "[gate] PARK — prewarm rc=$RC status=$(field "$EVID/prewarm.json" status 2>/dev/null)"; exit 3; fi
POP=$(field "$EVID/prewarm.json" populated); CACHED2=$(field "$EVID/prewarm.json" already_cached); UNITS2=$(field "$EVID/prewarm.json" unique_request_units)
echo "[gate] prewarm: unique=$UNITS2 already_cached=$CACHED2 populated=$POP privacy=$(field "$EVID/prewarm.json" privacy)"
[ "$(field "$EVID/prewarm.json" privacy_scope)" = "corpus" ] || { echo "[gate] PARK — real prewarm was not corpus-scoped ($(field "$EVID/prewarm.json" question_coverage.selected)/$(field "$EVID/prewarm.json" question_coverage.prepared))"; exit 3; }
[ "$POP" = "$WOULD" ] || { echo "[gate] PARK — populated ($POP) != dry-run would_populate ($WOULD): the dry run did not predict the spend"; exit 3; }
[ "$UNITS2" = "$UNITS" ] || { echo "[gate] PARK — unique_request_units differ between passes ($UNITS vs $UNITS2)"; exit 3; }
if [ "$WOULD" = "0" ]; then
  [ "$CACHED2" = "$UNITS2" ] || { echo "[gate] PARK — already_cached != unique_request_units on a zero-populate pass"; exit 3; }
  echo "[gate] §4.4(a) IDENTITY HOLDS: every unique post-transform unit was already in the F53 cache; transform-change count=$CHANGED"
fi
sqlite3 "$LCM_LONGMEMEVAL_EMBED_CACHE" "select count(*) from sqlite_master where type='table';" > /dev/null 2>&1 && \
  sqlite3 "$LCM_LONGMEMEVAL_EMBED_CACHE" ".tables" > "$EVID/cache-tables.txt" 2>/dev/null
date -u +%FT%TZ > "$EVID/gate-passed-at.txt"
echo "[gate] PASS — proceed: launch_all.sh"
