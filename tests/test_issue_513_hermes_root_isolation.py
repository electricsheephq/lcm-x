"""#513: the LCM_TEST_HERMES_AGENT_ROOT opt-in must not leak the real host package into this process.

With the root named, tests/test_issue_488_emission_proof.py used to put the checkout on sys.path and
import agent.agent_runtime_helpers at module import; later modules that stub the host (for example
tests/test_native_publication_fallback.py) then saw the real package and failed.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_ROOT_ENV = os.environ.get("LCM_TEST_HERMES_AGENT_ROOT")
ROOT = Path(_ROOT_ENV) if _ROOT_ENV else None
OPT_IN_MODULE = "tests.test_issue_488_emission_proof"
# Its hash check is about which helper bytes the root carries, not isolation; it runs in the suite itself.
PIN_TEST = "tests/test_issue_488_emission_proof.py::test_real_hermes_merge_fixture_is_pinned"

pytestmark = pytest.mark.skipif(
    ROOT is None or not (ROOT / "agent" / "agent_runtime_helpers.py").is_file(),
    reason="LCM_TEST_HERMES_AGENT_ROOT unset or not a hermes-agent checkout (the real host is opt-in)",
)


def _host_modules():
    return {name for name in sys.modules if name == "agent" or name.startswith("agent.")}


def test_opt_in_module_then_host_stub_module_run_green_in_one_process():
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if importlib.util.find_spec("pytest_randomly") is not None:
        cmd += ["-p", "no:randomly"]
    cmd += [
        "tests/test_issue_488_emission_proof.py",
        "tests/test_native_publication_fallback.py",
        "--deselect",
        PIN_TEST,
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env={**os.environ, "LCM_TEST_HERMES_AGENT_ROOT": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=900,
    )
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert re.search(r"\b\d+ passed\b", summary), summary
    assert not re.search(r"\b\d+ (failed|errors?)\b", summary), summary


def test_importing_the_opt_in_module_adds_no_host_package(monkeypatch):
    monkeypatch.setenv("LCM_TEST_HERMES_AGENT_ROOT", str(ROOT))
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(ROOT)])
    # Force a real import of the helper if the module still reaches for it (order-independent).
    monkeypatch.delitem(sys.modules, "agent.agent_runtime_helpers", raising=False)
    before = _host_modules()
    try:
        importlib.reload(importlib.import_module(OPT_IN_MODULE))
        assert _host_modules() - before == set()
        assert str(ROOT) not in sys.path
    finally:
        for name in _host_modules() - before:
            sys.modules.pop(name, None)
