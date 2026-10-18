#!/bin/bash
set -euo pipefail
# Never enable xtrace: the token must stay in process memory.
set +x
umask 077
SERVER_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
export LCMX_REPO="${1:?usage: run_aml_server.sh /absolute/path/to/lcm-x-tree}"
export AML_DB_ROOT="${AML_DB_ROOT:?AML_DB_ROOT is required}"
export AML_OMNIMEMEVAL_REPO="${AML_OMNIMEMEVAL_REPO:?AML_OMNIMEMEVAL_REPO is required}"
[[ -f "$AML_OMNIMEMEVAL_REPO/scripts/client_factory/lcmx_client.py" ]] || {
  echo "AML_OMNIMEMEVAL_REPO has no scripts/client_factory/lcmx_client.py" >&2; exit 1; }
: "${AML_ADDSEARCH_TOKEN:?AML_ADDSEARCH_TOKEN is required}"
mkdir -p -- "$AML_DB_ROOT"
chmod 0700 -- "$AML_DB_ROOT"
export AML_PORT=8791
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export AML_ADDSEARCH_TOKEN
exec "${AML_PYTHON:-python3}" -B "$SERVER_DIR/aml_server.py"
