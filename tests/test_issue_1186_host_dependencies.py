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


@pytest.mark.parametrize("hosted,synthetic", [(False, True), (True, False), (True, True)])
def test_core_scanner_exception_requires_hosted_synthetic_home(tmp_path, monkeypatch, hosted, synthetic):
    qualification = Qualification("ref-0.21.5", Path("python"), REPO,
                                  tmp_path / "scratch", tmp_path / "out")
    home = qualification.scratch / "native" if synthetic else tmp_path / "outside"
    monkeypatch.setenv("GITHUB_ACTIONS", "true" if hosted else "false")
    calls = []

    def stop_before_host(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("host call reached")

    monkeypatch.setattr(qualification, "cli", stop_before_host)
    monkeypatch.setattr(qualification, "run", lambda *a, **k: "")
    if hosted and synthetic:
        with pytest.raises(RuntimeError, match="host call reached"):
            qualification.install(home, "native")
        assert calls
    else:
        with pytest.raises(AssertionError, match="hosted CI only|synthetic home only"):
            qualification.install(home, "native")
        assert not calls
        assert not (home / "config.yaml").exists()


@pytest.mark.parametrize("route", ["native", "installer"])
@pytest.mark.parametrize("host_hint", [False, True, "wrapped"])
def test_decline_cell_requires_host_hint_independent_of_installer_echo(tmp_path, monkeypatch, route, host_hint):
    qualification = Qualification("upstream-main", Path("python"), REPO,
                                  tmp_path / "scratch", tmp_path / "out", route=route)
    text = "If the companion is disabled, run: hermes -p default plugins enable hermes-lcm-x-local-embeddings"
    if host_hint:
        text += "\nPlugin installed but not enabled. Run `hermes plugins enable hermes-lcm-x-local-embeddings` to activate."
    if host_hint == "wrapped":
        text = text.replace("Run `hermes plugins enable ", "Run `hermes plugins enable \r\n")
    monkeypatch.setattr(qualification, "install", lambda *a, **k: text)
    monkeypatch.setattr(qualification, "probe", lambda home, mode: {"companion_enabled": mode == "import"})
    monkeypatch.setattr(qualification, "cli", lambda *a: "")
    if host_hint:
        qualification.decline_routes()
        assert any(cell["cell"] == "d" for cell in qualification.cells)
    else:
        # Deleting only the host's hint must fail even if install.sh echoes enable.
        with pytest.raises(AssertionError, match="host.*hint"):
            qualification.decline_routes()


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
