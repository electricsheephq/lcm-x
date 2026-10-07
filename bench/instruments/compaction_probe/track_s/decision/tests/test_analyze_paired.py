"""Offline regression for committed, failed, and held-pass latency accounting."""

import json
import subprocess
import sys
from pathlib import Path


def test_failed_sweeps_require_summariser_calls(tmp_path):
    runs, logs, material = (tmp_path / name for name in ("runs", "logs", "material"))
    logs.mkdir()
    (material / "seed-1").mkdir(parents=True)
    (material / "seed-1" / "facts.json").write_text("[]")
    events = [
        {"is_compaction": True, "compress_wall_s": 10, "summariser_calls": [{}]},
        {"is_compaction": False, "compress_wall_s": 5, "summariser_calls": []},
        {"is_compaction": False, "compress_wall_s": 40, "summariser_calls": [{}]},
        {"is_compaction": False, "compress_wall_s": 0, "summariser_calls": [{}]},
    ]
    for arm in ("first", "second"):
        run = runs / arm / "seed-1" / "d1-r1"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({"events": [{**e, "levels": []} for e in events]}))
        (logs / f"s2-{arm}-d1-r1.log.wall").write_text("start 1\nexit 0 end 2\n")
    command = [
        sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "analyze_paired.py"),
        "--arms", "first", "second", "--seeds", "1", "--checkpoints", "304",
        "--run-root", str(runs), "--logs", str(logs), "--material", str(material),
        "--decision-root", str(tmp_path / "decision"),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    for stats in result["per_arm"].values():
        assert stats["compaction_wall_s"] == {"n": 1, "p50": 10, "p90": 10, "max": 10}
        assert stats["failed_sweep_count"] == 2
        assert stats["failed_sweep_wall_s"] == {"n": 2, "p50": 20, "p90": 40, "max": 40}
        assert stats["compaction_including_failed_sweeps_wall_s"] == {"n": 3, "p50": 10, "p90": 40, "max": 40}
