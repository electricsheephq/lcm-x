"""Fixture-only Linux counters plus the platform fallback and shell interface."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bench.tools import hoststat


def snap(monkeypatch, text):
    monkeypatch.setattr(hoststat.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", lambda self: text)
    return hoststat.snapshot()


@pytest.mark.parametrize("steal", [0, 10])
def test_known_jiffies(monkeypatch, steal):
    before = snap(monkeypatch, "cpu 10 0 5 80 5 0 0 0 200 100\ncpu0 0 0 0\n")
    after = snap(monkeypatch, f"cpu 30 0 15 {140 - steal} 15 0 0 {steal} 999 888\n")
    assert before["counters"] == dict(zip(hoststat.FIELDS, [10, 0, 5, 80, 5, 0, 0, 0]))
    before.update(monotonic=10, loadavg=[1, 2, 3])
    after.update(monotonic=12, loadavg=[4, 5, 6])
    assert hoststat.delta(before, after) == dict(elapsed_s=2, load1m_before=1, load1m_after=4,
        host_load_peak1m=4, steal_pct=steal, iowait_pct=10, busy_pct=30)
    assert hoststat.delta(before, before)["steal_pct"] is None
    assert hoststat.delta(after, before)["busy_pct"] is None


@pytest.mark.parametrize("text", ["", "cpu 1 2", "cpu 1 2 3 4 5 6 7 nope", "cpu0 1 2 3 4 5 6 7 8"])
def test_bad_counters(monkeypatch, text):
    found = snap(monkeypatch, text)
    assert found["counters"] is None and found["reason"]
    assert hoststat.delta(found, found)["steal_pct"] is None


def test_missing_file_and_platform(monkeypatch):
    monkeypatch.setattr(hoststat.sys, "platform", "linux")
    def missing(self):
        raise FileNotFoundError("fixture missing")
    monkeypatch.setattr(Path, "read_text", missing)
    assert "FileNotFoundError" in hoststat.snapshot()["reason"]
    monkeypatch.setattr(hoststat.sys, "platform", "darwin")
    assert "off Linux" in hoststat.snapshot()["reason"]
    monkeypatch.delattr(hoststat.os, "getloadavg")
    assert hoststat.snapshot()["loadavg"] is None


def test_cli_roundtrip(tmp_path):
    command = [sys.executable, "-B", hoststat.__file__]
    before = json.loads(subprocess.check_output([*command, "snapshot"], text=True))
    after = dict(before, monotonic=before["monotonic"] + 2)
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, value in zip(paths, (before, after)):
        path.write_text(json.dumps(value))
    found = json.loads(subprocess.check_output([*command, "delta", *map(str, paths)], text=True))
    assert found == hoststat.delta(before, after)
    assert found["elapsed_s"] == 2


def test_maxima_missing_and_mixed():
    assert hoststat.maxima([{}, {"host": {"steal_pct": None}}]) == dict(max_steal_pct=None, host_load_peak1m=None)
    assert hoststat.maxima([{"host": {"steal_pct": 3, "host_load_peak1m": 2}},
                            {"host": {"steal_pct": 1, "host_load_peak1m": 9}}]) == dict(max_steal_pct=3, host_load_peak1m=9)


@pytest.mark.parametrize("shape", ["zero", "regression", "missing"])
def test_null_percentages_have_reason(shape):
    before = dict(monotonic=1, loadavg=[1, 0, 0], counters=dict.fromkeys(hoststat.FIELDS, 5))
    after = dict(before, monotonic=2)
    if shape == "regression":
        after["counters"] = dict.fromkeys(hoststat.FIELDS, 4)
    elif shape == "missing":
        after["counters"] = None
    found = hoststat.delta(before, after)
    assert all(found[key] is None for key in ("steal_pct", "iowait_pct", "busy_pct"))
    assert found.get("reason")
