"""Exercise the real target's report/artifact boundary without exposing payloads."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["pass", "fail", "empty", "missing"])
def test_failure_capture(tmp_path, mode):
    sentinel = "SYNTHETIC_1753_CONTENT_SENTINEL"
    probe = tmp_path / "probe_1753.py"
    probe.write_text('''
import logging
import pytest

@pytest.fixture(autouse=True)
def inject(request, monkeypatch):
    from hermes_lcm.store import MessageStore
    module = request.module
    turn = module.turn
    def planted(history, text):
        turn(history, text + "SYNTHETIC_1753_CONTENT_SENTINEL" if text == "before-one" else text)
    monkeypatch.setattr(module, "turn", planted)
    original = MessageStore._append_protected_batch
    def append(self, *args, **kwargs):
        messages = args[1] if len(args) > 1 else kwargs["messages"]
        # The target's last ingest (gamma) has no later ingest that could store it again. A reclassify
        # or busy retry re-enters here with the same rows (#2756), so match the content, not a call count.
        last = [m.get("content") for m in messages] == ["gamma", "answer gamma"]
        if last and MODE == "empty":
            return []
        ids = original(self, *args, **kwargs)
        if last and MODE == "missing":
            with self.connection:
                self.connection.executemany("DELETE FROM messages WHERE store_id = ?", [(i,) for i in ids])
        return ids
    monkeypatch.setattr(MessageStore, "_append_protected_batch", append)
    yield
    if MODE == "fail":
        logging.getLogger("hermes_lcm").error("SYNTHETIC_1753_CONTENT_SENTINEL")
        print("SYNTHETIC_1753_CONTENT_SENTINEL")
        raise AssertionError("SYNTHETIC_1753_CONTENT_SENTINEL")
'''.replace('MODE', repr(mode)))
    output, xml = tmp_path / "capture", tmp_path / "results.xml"
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               LCM_DIAG_1753_DIR=str(output))
    run = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "probe_1753",
                          "-q", "tests/test_issue_1306_cli_new.py::test_new_stores_all_turns_under_new_session[cli]",
                          f"--basetemp={tmp_path / 'pt'}", f"--junitxml={xml}"],
                         cwd=Path(__file__).parents[1], env=env, capture_output=True, timeout=60)
    assert run.returncode == (0 if mode == "pass" else 1), "unexpected target outcome"
    files = list(output.glob("*.json"))
    assert len(files) == (0 if mode == "pass" else 1)
    assert sentinel.encode() not in run.stdout + run.stderr + xml.read_bytes()
    if files:
        raw = files[0].read_text()
        assert sentinel not in raw
        data = json.loads(raw)
        assert set(data) == {"schema", "pins", "events"}
        events = data["events"]
        assert len(events) <= 512
        allowed = {"kind", "time", "thread", "owner", "conn", "in_transaction", "cursor", "proof_present",
                   "end_consumed", "frontier", "pending_tails", "input_count", "protected_count",
                   "estimate_count", "returned_ids", "statement", "rows", "phase", "outcome"}
        assert all(set(event) <= allowed for event in events)
        assert not any(key in raw for key in ('"content"', '"sql"', '"parameters"', '"exception"'))
        if mode in {"empty", "missing"}:
            final = [e for e in events if e["kind"] == "append_return"][-1]
            # A retry after the probe's own commit can return later ids than 9 and 10 (#2756).
            # The gamma batch stores exactly two rows, on every attempt.
            assert (final["returned_ids"] == []) if mode == "empty" else len(final["returned_ids"]) == 2
            for kind in ("same_reader", "fresh_reader"):
                rows = next(e["rows"] for e in events if e["kind"] == kind)
                assert [r[0] for r in rows] == list(range(1, 9))
                assert not set(final["returned_ids"]) & {r[0] for r in rows}


def test_ring_is_bounded():
    from tests.diag_1753_capture import Capture
    capture = Capture()
    for _ in range(600):
        capture.record("synthetic")
    assert len(capture.events) == 512
