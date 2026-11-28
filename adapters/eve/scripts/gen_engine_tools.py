#!/usr/bin/env python3
"""Write src/engine-tools.json: the LCM-X engine's advertised tools (name, description, parameters) at the vendored
core, with default settings, so the Eve provider offers them without starting the engine.

    node scripts/vendor-core.mjs && uv run --no-project --python 3.12.16 python -I scripts/gen_engine_tools.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "host"))
os.environ["LCM_EMBEDDINGS_ENABLED"] = "false"
import lcmx_eve_host  # noqa: E402

core = PKG / "core"
lcmx_eve_host.planner.bootstrap(str(core))
commit = json.loads((core / "BUNDLE.json").read_text())["commit"]
with tempfile.TemporaryDirectory() as home:
    host = lcmx_eve_host.Host(Path(home), context_length=200_000, model="gen")
    try:
        tools = host.tool_schemas()
    finally:
        host.close()
(PKG / "src" / "engine-tools.json").write_text(json.dumps({"core": commit, "tools": tools}, indent=2, sort_keys=True) + "\n")
print(f"{len(tools)} engine tools from {commit[:12]}")
