import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/dev/cloud_setup.sh"


def test_missing_tools_stop_before_venv(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Only command presence is checked: none of these tools should be invoked.
    for tool in ("git", "curl", "gcc", "sha256sum"):
        (bin_dir / tool).symlink_to(shutil.which(tool) or sys.executable)
    result = subprocess.run(
        [shutil.which("bash"), str(SCRIPT)], cwd=tmp_path,
        env=dict(os.environ, PATH=str(bin_dir), PYTHON=sys.executable),
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "cloud_setup: missing prerequisite: unzip",
        "cloud_setup: missing prerequisite: node",
    ]
    assert not (tmp_path / ".venv").exists()


def test_missing_venv_support_stops_before_work(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("git", "curl", "unzip", "gcc", "sha256sum", "node"):
        (bin_dir / tool).symlink_to(sys.executable)
    python = bin_dir / "python"
    python.write_text('#!/bin/sh\nexit 1\n')
    python.chmod(0o755)
    result = subprocess.run(
        [shutil.which("bash"), str(SCRIPT)], cwd=tmp_path,
        env=dict(os.environ, PATH=str(bin_dir), PYTHON=str(python)),
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 1
    assert result.stderr.splitlines() == [f"cloud_setup: missing prerequisite: {python} venv/ensurepip"]
    assert not (tmp_path / ".venv").exists()
