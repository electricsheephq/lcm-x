"""Eval-2 pinned arms run the fleet's stub threshold (10000); the v3 LCMX-fleet arm keeps 6000."""
import json
import sys
from pathlib import Path

TRACK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACK / "s2"))
from s2lib import arms  # noqa: E402

KEY = "LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS"


def test_eval2_arms_use_fleet_stub_threshold(monkeypatch):
    pins = {a: dict(worktree="/nonexistent", sha="0" * 40) for a in ("L0", "L1", "L1-H", "L1-noptr")}
    monkeypatch.setenv("S2_ARM_PINS", json.dumps(pins))
    for arm in pins:
        assert arms.resolve(arm)["env"][KEY] == "10000"
    assert arms.resolve("LCMX-fleet")["env"][KEY] == "6000"
