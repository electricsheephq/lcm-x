from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "failclose.py",
        "pinverify.py",
        "storefreeze.py",
        "pairedgate.py",
        "deliveryprofile.py",
    ],
)
def test_tool_has_help(name):
    completed = subprocess.run(
        [sys.executable, str(TOOLS / name), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.startswith("usage:")
