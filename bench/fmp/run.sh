#!/bin/bash
set +x
set -euo pipefail
if [[ "${FMP_NICED:-}" != 1 ]]; then
  export FMP_NICED=1
  exec nice -n 19 bash "$0" "$@"
fi
umask 077
PYTHON="${FMP_PYTHON:?set FMP_PYTHON to the Python that has LCM-X dependencies}"
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  echo "FMP_PYTHON requires Python >= 3.11" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 ORT_NUM_THREADS=4
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export HERMES_DISABLE_LAZY_INSTALLS=1
unset OPENROUTER_API_KEY
for ARG in "$@"; do
  # OpenRouter is an explicit opt-in backend: its key is loaded only for an openrouter spec.
  if [[ "$ARG" == openrouter:* || "$ARG" == openai/* || "$ARG" == anthropic/* ]]; then
    OPENROUTER_API_KEY="$(nice -n 19 security find-generic-password -s OPENROUTER_API_KEY -w 2>/dev/null)"
    export OPENROUTER_API_KEY
  fi
  if [[ "$ARG" == voyage* || "$ARG" == --config=voyage* ]]; then
    VOYAGE_API_KEY="$(nice -n 19 security find-generic-password -s VOYAGE_API_KEY -w 2>/dev/null)"
    export VOYAGE_API_KEY
  fi
done
trap 'unset OPENROUTER_API_KEY VOYAGE_API_KEY' EXIT
"$PYTHON" -m bench.fmp "$@"
