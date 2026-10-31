"""#1820: module reloads must not unfreeze the real compress engine's clock."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

from tests.behaviour.compress_trace_harness import Clock, install_clock
from tests.harness.engine import LCMEngine


def test_stress_polluter_then_compress_victims(tmp_path):
    """Collect together, then run the real polluter before the real victims."""
    command = [
        sys.executable, "-m", "pytest", "tests/test_stress_release_check.py",
        "tests/behaviour/test_compress_auxiliary_stack_budget.py::"
        "test_sweep_phase_host_callbacks_preserve_auxiliary_at_host_depth",
        "tests/behaviour/test_compress_trace.py::test_compress_golden[leaf-phase-rescue-loses]",
        "tests/behaviour/test_compress_trace.py::test_compress_golden[boundary-cleanup-only]",
        "-q", "-n", "0", "-p", "no:randomly", "-p", "no:cacheprovider",
        "-o", "addopts=", "-o", "tmp_path_retention_policy=failed",
        f"--basetemp={tmp_path / 'pair'}",
    ]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                            env={**os.environ, "PYTEST_ADDOPTS": ""},
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]


def test_clock_patches_retained_engine_and_reimported_modules(monkeypatch):
    """Exercise hardening even when a future test evicts the entire package."""
    from benchmarking.stress import _clear_hermes_lcm_submodules

    package = sys.modules["hermes_lcm"]
    saved = {name: module for name, module in sys.modules.items()
             if name.startswith("hermes_lcm.")}
    attributes = vars(package).copy()
    original_globals = LCMEngine.__init__.__globals__
    store_globals = original_globals["MessageStore"].get.__globals__
    try:
        _clear_hermes_lcm_submodules()
        fresh = importlib.import_module("hermes_lcm.engine")
        assert fresh.LCMEngine is not LCMEngine
        install_clock(monkeypatch, Clock())
        assert original_globals["time"] is fresh.time
        assert LCMEngine._compress_impl.__globals__["time"] is fresh.time
        assert store_globals["time"] is fresh.time
    finally:
        _clear_hermes_lcm_submodules()
        sys.modules.update(saved)
        vars(package).clear()
        vars(package).update(attributes)
