"""Repeat the unchanged synthetic regression in both host-facing arms."""

import importlib.util
import os
from pathlib import Path

import pytest


_path = Path(__file__).parents[1] / "test_issue_1306_cli_new.py"
_spec = importlib.util.spec_from_file_location("diag_1753_original", _path)
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
engine = _original.engine


@pytest.mark.parametrize("iteration", range(int(os.environ.get("DIAG_1753_ITERATIONS", "300"))))
@pytest.mark.parametrize("gateway", [False, True], ids=["cli", "gateway"])
def test_new_stores_all_turns_under_new_session(engine, gateway, caplog, iteration):
    _original.test_new_stores_all_turns_under_new_session(engine, gateway, caplog)
