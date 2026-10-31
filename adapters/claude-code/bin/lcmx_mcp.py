"""Start LCM-X's read-only MCP server (scripts/lcm_x_mcp.py, #1216) over the Claude Code store.

Stdlib only and Python 3.9 compatible: Claude Code starts it with the system ``python3``,
and it replaces itself with the LCM-X runtime interpreter.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcmx_runtime  # noqa: E402


def main() -> int:
    environ = os.environ
    host = "grok-build" if environ.get("GROK_SESSION_ID") else "claude-code"
    home = lcmx_runtime.store_home(environ, host)
    if home is None:
        print("lcm-x: %s must be an absolute path" % lcmx_runtime.HOMES[host][0], file=sys.stderr)
        return 1
    python = lcmx_runtime.runtime_python(environ, home)
    package = lcmx_runtime.package_root(environ)
    reader = package / "scripts" / "lcm_x_mcp.py" if package else None
    if python is None:
        problem = "no LCM-X runtime: set LCMX_PYTHON or create <store home>/venv"
    elif reader is None or not reader.is_file():
        problem = "the LCM-X package with scripts/lcm_x_mcp.py was not found"
    else:
        problem = ""
    if problem:
        print("lcm-x: " + problem, file=sys.stderr)
        return 1
    logs = home / "logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.execv(str(python), [str(python), str(reader), "--store", str(home / "lcm.db"),
                           "--hermes-home", str(home), "--log-dir", str(logs)])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
