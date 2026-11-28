"""Rewrite recall_snippets_envelope_main_windows_9d9f0e8c.json from main's own recall_delivery.py.

Run from the repo root (not collected by default):
    python -m pytest -p no:cacheprovider tests/fixtures/capture_recall_snippets_envelope_main_windows.py
"""
import json
from pathlib import Path
import subprocess
from types import ModuleType

from hermes_lcm import recall_delivery
from tests import test_recall_snippets_envelope as envelope


def test_capture_main_windows(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(["git", "show", f"{envelope.MAIN_BASE}:recall_delivery.py"], cwd=root, text=True)
    main = ModuleType("hermes_lcm._snippets_main_delivery")
    main.__package__ = "hermes_lcm"
    exec(compile(source, "main:recall_delivery.py", "exec"), main.__dict__)
    monkeypatch.setattr(recall_delivery, "deliver_recall", main.deliver_recall)
    cases = envelope.windows_cases(lambda d, **kwargs: d.invoke(**kwargs))
    payload = {"base": envelope.MAIN_BASE, "cases": {key: envelope.pinned(raw) for key, raw in sorted(cases.items())}}
    envelope.MAIN_WINDOWS.write_text(json.dumps(payload, indent=1) + "\n")
