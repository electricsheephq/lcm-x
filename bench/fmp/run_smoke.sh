#!/bin/bash
# FMP v1.1 smoke on the subscription plans: no metered API calls. Codex runs on the
# ChatGPT plan login; claude -p derives its OAuth token per call into the child env only.
set +x
set -euo pipefail
if [[ "${FMP_NICED:-}" != 1 ]]; then
  export FMP_NICED=1
  exec nice -n 19 bash "$0" "$@"
fi
umask 077
export PYTHONDONTWRITEBYTECODE=1 HERMES_DISABLE_LAZY_INSTALLS=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 ORT_NUM_THREADS=4
unset OPENROUTER_API_KEY
PYTHON="${FMP_PYTHON:?set FMP_PYTHON to the Python that has LCM-X dependencies}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?usage: run_smoke.sh OUTPUT_DIR}"
LME_DATA="${FMP_LME_DATA:?set FMP_LME_DATA to longmemeval_s_cleaned.json}"
LOCOMO_DATA="${FMP_LOCOMO:?set FMP_LOCOMO to locomo10.json}"
SOL=codex:gpt-6.1-sol@high
OPUS=claude:claude-opus-5-5
cd "$REPO"
mkdir -p "$OUT"
# 30 LME-S ids stratified over question type plus abstention (seed 1189); LoCoMo conv-30 without
# category 5; a seeded 10 of the 30 for the Opus agentic arm. Ids only, no corpus text.
"$PYTHON" - "$REPO" "$OUT" "$LME_DATA" "$LOCOMO_DATA" <<'INNER'
import json, random, sys
from pathlib import Path
repo, out, lme_data, locomo_data = sys.argv[1:]
sys.path.insert(0, repo)
from bench.fmp import datasets
from bench.fmp.common import load_lcm, quiet
quiet()
lme = load_lcm(repo)
strata = {}
for q in datasets.load(lme_data, "lme-s", lme)[0]:
    strata.setdefault("abstention" if q.is_abstention else q.qtype, []).append(q.qid)
rng, keys, chosen = random.Random(1189), sorted(strata), {}
for i, key in enumerate(keys):
    for qid in rng.sample(sorted(strata[key]), 30 // len(keys) + (i < 30 % len(keys))):
        chosen[qid] = key
Path(out, "lme-ids.json").write_text(json.dumps(sorted(chosen)) + "\n")
Path(out, "lme-strata.json").write_text(json.dumps(dict(sorted(chosen.items())), indent=1) + "\n")
Path(out, "opus-ids.json").write_text(json.dumps(sorted(random.Random(1189).sample(sorted(chosen), 10))) + "\n")
ids = [q.qid for q in datasets.load(locomo_data, "locomo", lme)[0] if q.store_id == "conv-30"]
Path(out, "locomo-ids.json").write_text(json.dumps(ids) + "\n")
INNER
COMMON=(--lcmx-repo "$REPO" --split-label smoke --store-root "$OUT/stores" --quota-ledger "$OUT/quota.json" --workers 4)
LME=("${COMMON[@]}" --dataset lme-s --data "$LME_DATA" --question-ids "$OUT/lme-ids.json")
LOCOMO=("${COMMON[@]}" --dataset locomo --data "$LOCOMO_DATA" --question-ids "$OUT/locomo-ids.json")
OPUS_LME=("${COMMON[@]}" --dataset lme-s --data "$LME_DATA" --question-ids "$OUT/opus-ids.json")
fmp() { "$PYTHON" -m bench.fmp "$@"; }
for CONFIG in off local; do
  fmp build-stores "${LME[@]}" --config "$CONFIG" --output "$OUT/build/lme-$CONFIG"
  fmp build-stores "${LOCOMO[@]}" --config "$CONFIG" --output "$OUT/build/locomo-$CONFIG"
done
# Leakage canary for each harness before any real question (spec item 3); a failure stops here.
fmp canary "${LME[@]}" --config off --reader "$SOL" --output "$OUT/canary/codex"
fmp canary "${LME[@]}" --config off --reader "$OPUS" --output "$OUT/canary/claude"
for MODE in single agentic; do
  for CONFIG in off; do
    for SET in lme locomo; do
      if [[ "$SET" == lme ]]; then ARGS=("${LME[@]}"); else ARGS=("${LOCOMO[@]}"); fi
      [[ -f "$OUT/$SET/off-$MODE/judged.jsonl" && -f "$OUT/$SET/local-$MODE/judged.jsonl" ]] && continue
      [[ -f "$OUT/$SET/off-$MODE/answers.jsonl" && -f "$OUT/$SET/local-$MODE/answers.jsonl" ]] || \
        fmp answer "${ARGS[@]}" --config off --paired-config local --delivery "$MODE" --reader "$SOL" --output "$OUT/$SET"
      for CONFIG in off local; do
        RUN="$OUT/$SET/$CONFIG-$MODE"
        [[ -f "$RUN/judged.jsonl" ]] || fmp judge "${ARGS[@]}" --config "$CONFIG" --delivery "$MODE" --reader "$SOL" --output "$RUN"
      done
    done
  done
done
RUN="$OUT/opus/off-agentic"
if [[ ! -f "$RUN/judged.jsonl" ]]; then
  [[ -f "$RUN/answers.jsonl" ]] || fmp answer "${OPUS_LME[@]}" --config off --delivery agentic --reader "$OPUS" --output "$RUN"
  fmp judge "${OPUS_LME[@]}" --config off --delivery agentic --reader "$OPUS" --output "$RUN"
fi
fmp score "${LME[@]}" --output "$OUT/lme" --runs "$OUT"/lme/{off,local}-{single,agentic}
fmp score "${LOCOMO[@]}" --output "$OUT/locomo" --runs "$OUT"/locomo/{off,local}-{single,agentic}
fmp score "${OPUS_LME[@]}" --output "$OUT/opus" --runs "$OUT/opus/off-agentic"
