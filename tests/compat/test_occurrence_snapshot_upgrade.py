"""An actual v0.28.1 writer retains restamped occurrences across upgrade (I1/I5)."""
import json
import os
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest

from tests.compat.test_occurrence_snapshot_rollback import V0281

BASE = "0e9cef1e97b3248dd899903e587062cce7bc42be"


def _save_database(source, saved):
    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(saved)) as dst:
        src.backup(dst)


def _restore_database(saved, target):
    for suffix in ("-wal", "-shm"):
        Path(str(target) + suffix).unlink(missing_ok=True)
    shutil.copy2(saved, target)


def test_save_database_includes_uncheckpointed_wal(tmp_path):
    source, saved = tmp_path / "lcm.db", tmp_path / "saved.db"
    with closing(sqlite3.connect(source)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("CREATE TABLE sample (value TEXT)")
        writer.execute("INSERT INTO sample VALUES ('committed in WAL')")
        writer.commit()
        assert source.with_name("lcm.db-wal").stat().st_size > 32
        _save_database(source, saved)
        with closing(sqlite3.connect(saved)) as snapshot:
            assert snapshot.execute("SELECT value FROM sample").fetchall() == [("committed in WAL",)]


@pytest.mark.parametrize("restamp,expected", [("last", 340), ("all", 351)])
@pytest.mark.parametrize("stale_wal", [False, True])
def test_v0281_snapshot_upgrade_matches_base(tmp_path, old_tree, host_root, restamp, expected, stale_wal):
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
    _save_database(home / "lcm.db", saved)
    if stale_wal:
        # A second connection keeps the control phase's WAL alive after its exit.
        keeper = sqlite3.connect(home / "lcm.db")
        try:
            assert keeper.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            control = phase(base, "read")
            sidecars = {suffix: (home / ("lcm.db" + suffix)).read_bytes()
                        for suffix in ("-wal", "-shm")}
            assert len(sidecars["-wal"]) > 32
        finally:
            keeper.close()
        # Plant the captured control state after closing every connection.
        for suffix, data in sidecars.items():
            (home / ("lcm.db" + suffix)).write_bytes(data)
    else:
        control = phase(base, "read")
    _restore_database(saved, home / "lcm.db")
    candidate = phase(root, "read")
    assert candidate == control
    assert candidate["attached"] and candidate["rows_before"] == 335
    assert candidate["rows_after"] == expected and candidate["ingest_failures"] == 0
