#!/usr/bin/env bash
set -euo pipefail
: "${TMPDIR:?}" "${TRACK_S_OUT:?}"
: "${HERMES_SRC:?set HERMES_SRC to the pinned Hermes source checkout (its parent holds venv/)}"
export HERMES_SRC HERMES_HOME
HERMES_HOME=$(mktemp -d "$TMPDIR/eval2-h.XXXXXX")
export HERMES_DISABLE_LAZY_INSTALLS=1 S2_H_SANDBOX=1
mkdir -p "$TRACK_S_OUT"
profile=$HERMES_HOME/isolation.sb
python3 - "$profile" "$TMPDIR" "$TRACK_S_OUT" <<'PY'
import json, os, pathlib, sys
profile, scratch, output = map(pathlib.Path, sys.argv[1:])
profile.write_text('(version 1)\n(allow default)\n' + ('(deny network*)\n' if os.environ.get('S2_OFFLINE') == '1' else '') +
                   '(deny file-write*)\n(allow file-write* ' +
                   ' '.join('(subpath ' + json.dumps(str(p.resolve())) + ')' for p in (scratch, output)) + ')\n')
PY
cp "$profile" "$TRACK_S_OUT/h-isolation.sb"
exec sandbox-exec -f "$profile" "$(dirname "$HERMES_SRC")/venv/bin/python" -B \
  "$(dirname "$0")/run_s_hermes_builtin.py" "$@"
