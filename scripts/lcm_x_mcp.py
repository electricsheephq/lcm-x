#!/usr/bin/env python3
"""lcm-x mcp: a read-only stdio MCP server over one Hermes LCM-X store.

Register it with Claude Code (run with the Hermes venv python):

    claude mcp add lcm-x -- <hermes-venv-python> <plugin_dir>/scripts/lcm_x_mcp.py \\
        --store <path/to/lcm.db> --hermes-home <HERMES_HOME>

It never writes the store. See docs/operator-guide.md.
"""

import sys

if sys.version_info < (3, 10):
    sys.stderr.write("lcm-x mcp needs Python 3.10 or newer; run it with the Hermes venv python\n")
    raise SystemExit(2)

import types  # noqa: E402
from pathlib import Path  # noqa: E402

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "hermes_lcm"


def _ensure_local_package_importable():
    """Register a bare package for the plugin modules; never runs __init__.py or engine.py."""
    if PACKAGE_NAME in sys.modules:
        return
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(PLUGIN_DIR)]
    pkg.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = pkg


_ensure_local_package_importable()

from hermes_lcm.mcp_reader import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
