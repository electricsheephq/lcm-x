"""Packaged CLI smoke (docs/claude-code-adapter.md, "Release mechanics")."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("runtime", ("venv", "found"))
def test_claude_code_unit_smoke_at_head(tmp_path, runtime):
    reader = subprocess.run(["git", "cat-file", "-e", "HEAD:scripts/lcm_x_mcp.py"], cwd=ROOT,
                            capture_output=True, timeout=5)
    if reader.returncode:
        pytest.skip("HEAD has no scripts/lcm_x_mcp.py; CLI smoke needs the #1216 reader")
    # The CLI constructs a separate allowlisted environment for all of its children.
    done = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/claude_code_unit_smoke.py"),
                           "--ref", "HEAD", "--runtime", runtime], cwd=ROOT,
                          capture_output=True, text=True, timeout=65,
                          env={"PATH": os.defpath, "TMPDIR": str(tmp_path)})
    summary = json.loads(done.stdout)
    assert done.returncode == 0, summary
    assert summary["pass"] is True and summary["ref"] == "HEAD"
    assert summary["runtime"] == runtime
    assert summary["core_commit"] and summary["unit"]
    assert all(step["outcome"] == "pass" for step in summary["steps"])
    assert {"hook worker capture", "tools/list", "lcm_grep", "lcm_expand", "MCP exit"} <= {
        step["step"] for step in summary["steps"]}
    for field in ("hook_wall_ms", "hook_in_process_ms"):
        assert set(summary[field]) == {"SessionStart", "UserPromptSubmit", "PostToolBatch", "Stop"}
        for event, stats in summary[field].items():
            assert stats["n"] == (20 if event == "PostToolBatch" else 1)
            assert 0 <= stats["p50"] <= stats["p95"] <= stats["max"]
    assert summary["hook_wall_ms"]["PostToolBatch"]["n"] == 20
