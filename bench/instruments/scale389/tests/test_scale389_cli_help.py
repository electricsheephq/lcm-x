from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "script",
    ["archive_regression.py", "build_corpus.py", "phase1a.py", "run_scale.py"],
)
def test_script_has_help(script):
    completed = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.startswith("usage:")
