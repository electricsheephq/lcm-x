"""An actual v0.28.1 writer retains restamped occurrences across upgrade (I1/I5)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.compat.test_occurrence_snapshot_rollback import V0281

BASE = "10407e12e33c1d7b09fe0e05141153555aec5d62"


@pytest.mark.parametrize("restamp,expected", [("last", 340), ("all", 351)])
def test_v0281_snapshot_upgrade_matches_base(tmp_path, old_tree, host_root, restamp, expected):
    old = old_tree("v0.28.1", {"v0.28.1": V0281})
    base = old_tree("BASE", {"BASE": BASE})
    root = Path(__file__).resolve().parents[2]
    probe = tmp_path / "upgrade_probe.py"
    shutil.copy2(Path(__file__).with_name("occurrence_upgrade_probe.py"), probe)
    home, result = tmp_path / "writer", tmp_path / "result.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("LCM_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", P4F_HOME=str(home), P4F_OUT=str(result),
               P4F_HEAD="summary", P4F_FLOW="drive", P4F_RESTAMP=restamp)

    def phase(tree, action):
        env.update(PYTHONPATH=str(tree) + os.pathsep + str(host_root), P4F_PHASE=action)
        command = [sys.executable, "-B", "-c",
                   "import tests.conftest; import pytest, sys; sys.exit(pytest.main(sys.argv[1:]))",
                   str(probe), "-q", "-p", "no:cacheprovider", "--confcutdir=" + str(probe.parent),
                   "--basetemp=" + str(tmp_path / (action + tree.name))]
        run = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
        return json.loads(result.read_text()) if action == "read" else None

    phase(old, "write")
    saved = tmp_path / "old-writer.db"
    shutil.copy2(home / "lcm.db", saved)
    control = phase(base, "read")
    shutil.copy2(saved, home / "lcm.db")
    candidate = phase(root, "read")
    assert candidate == control
    assert candidate["attached"] and candidate["rows_before"] == 335
    assert candidate["rows_after"] == expected and candidate["ingest_failures"] == 0
