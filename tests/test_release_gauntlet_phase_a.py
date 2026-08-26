from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_local_posture_covers_every_runtime_registered_tool(tmp_path):
    worktree = Path(__file__).resolve().parents[1]
    runner = worktree / "bench/instruments/release_gauntlet/phase_a_tool_matrix.py"
    env = os.environ.copy()
    env.pop("VOYAGE_API_KEY", None)
    env.pop("LCM_EMBEDDING_API_KEY", None)
    env.pop("SILICONFLOW_API_KEY", None)
    completed = subprocess.run(
        [sys.executable, str(runner), "--worktree", str(worktree), "--out", str(tmp_path)],
        cwd=worktree, env=env, text=True, capture_output=True, timeout=60,
    )
    receipt = (tmp_path / "PHASE-A-RECEIPT.md").read_text(encoding="utf-8")
    assert completed.returncode == 0, completed.stdout + completed.stderr + receipt
    assert "Registry coverage: `COMPLETE`" in receipt
    assert "Missing matrix rows: `none`" in receipt
    assert receipt.count("| local | `lcm_") == 15
    assert "| cloud | `lcm_recall` | SKIP | SKIP: VOYAGE_API_KEY is absent |" in receipt
    assert "Exit verdict: `PASS`" in receipt
