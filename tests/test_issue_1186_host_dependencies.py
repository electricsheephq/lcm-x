"""Permanent #1186 exact-host cells; opt-in integration runs on hosted CI only."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bench.instruments.reliability.dependency_qualification import (
    Qualification, allocated_bytes, answer_prompt,
)

REPO = Path(__file__).resolve().parents[1]


def test_test_user_declines_only_the_actual_host_dependency_question():
    assert answer_prompt("Enable 'companion' now? [y/N]: ", decline=True) == "y"
    assert answer_prompt("Prepare these with Hermes through PM now? [y/N]: ", decline=True) == "n"
    assert answer_prompt("Prepare these with Hermes through PM now? [y/N]: ", decline=False) == "y"


def test_allocated_measurement_counts_hardlinks_once(tmp_path):
    first = tmp_path / "one"
    first.write_bytes(b"x" * 8192)
    (tmp_path / "two").hardlink_to(first)
    assert allocated_bytes(tmp_path) == first.stat().st_blocks * 512


def test_workflow_keeps_exact_host_arms_and_failure_artifacts():
    workflow = (REPO / ".github/workflows/real-host.yml").read_text()
    assert "dependency-qualification:" in workflow
    qualification = workflow.split("  dependency-qualification:", 1)[1]
    assert "host: [ref-0.21.5, upstream-main]" in qualification
    assert "route: [native, installer]" in qualification
    assert "bench.instruments.reliability.ci prep" in qualification
    assert "tests/test_issue_1186_host_dependencies.py" in qualification
    assert "if: always()" in qualification
    assert "1186-qualification/" in qualification


@pytest.mark.skipif(os.environ.get("LCM_1186_QUALIFY") != "1", reason="hosted exact-host qualification only")
def test_exact_host_dependency_cells():
    host = os.environ["LCM_1186_HOST"]
    src, python = Path(os.environ["LCM_REAL_HERMES_SRC"]), Path(os.environ["LCM_REAL_HERMES_PYTHON"])
    out = Path(os.environ["LCM_1186_ARTIFACTS"])
    scratch = Path(os.environ["LCM_1186_SCRATCH"])
    qualification = Qualification(host, python, src, scratch, out,
                                  route=os.environ["LCM_1186_ROUTE"],
                                  allow_locked_tools=os.environ.get("LCM_1186_ALLOW_LOCKED_TOOLS") == "1")
    try:
        qualification.execute()
    finally:
        qualification.save()
    cells = json.loads((out / "cells.json").read_text())["cells"]
    assert any(cell["cell"] == "a" and cell["status"] == "COMPLETE" for cell in cells)
