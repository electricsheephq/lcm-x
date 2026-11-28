"""#2711: drain FTS workers before pytest deletes and reuses a test's tmp_path."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("loader", ["spillover", "matrix"])
def test_fixture_load_after_previous_test_scan(tmp_path, loader):
    root = Path(__file__).resolve().parents[1]
    probe = tmp_path / "test_reused_path.py"
    probe.write_text(textwrap.dedent('''
        import sqlite3
        import threading
        from contextlib import closing

        import pytest

        from hermes_lcm import db_bootstrap as db
        from tests.test_db_bootstrap_fts import _spec
        from tests.test_issue_2521_spillover_upgrade_replay import _v0292_store
        from tests.invariants.test_issue_2539_compare_matrix import _v0292_matrix_store

        entered, release = threading.Event(), threading.Event()
        original_scan = db._run_background_integrity_scan
        previous_path = worker = None

        def held_scan(*args):
            entered.set()
            assert release.wait(10), "scan was never released"
            original_scan(*args)

        @pytest.mark.parametrize("phase", ["start", "load"])
        def test_fixture_loader_reuses_same_directory(phase, tmp_path, monkeypatch):
            global previous_path, worker
            if phase == "start":
                previous_path = tmp_path
                _v0292_store(tmp_path)
                monkeypatch.setattr(db, "_run_background_integrity_scan", held_scan)
                with closing(sqlite3.connect(tmp_path / "hermes/lcm.db")) as conn:
                    assert db._dispatch_background_integrity_scan(conn, _spec())
                assert entered.wait(5)
                worker = next(iter(db._integrity_scan_threads.values()))
                original_join = worker.join

                def release_on_drain(timeout=None):
                    release.set()
                    original_join(timeout)

                # A harness drain releases the worker before tmp_path teardown;
                # without a drain it stays held until the next cell's load.
                monkeypatch.setattr(worker, "join", release_on_drain)
                return

            assert tmp_path == previous_path  # pytest truncates both names to 30 chars.
            home = tmp_path / "hermes"
            home.mkdir()
            assert not list(home.iterdir())  # No database, WAL or journal survives.
            release.set()
            worker.join(5)
            assert not worker.is_alive()
            # The real scan reopens its pinned path and creates metadata if it
            # escaped the prior test. Each real dump must still load cleanly.
            if LOADER == "spillover":
                _v0292_store(tmp_path)
            else:
                temp = tmp_path / "host-temp"
                temp.mkdir()
                _v0292_matrix_store(tmp_path, temp, "data-uri", "hermes-results")
    ''').replace("LOADER", repr(loader)))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "tests.conftest",
         "-p", "no:cacheprovider", "-p", "no:randomly", "-c", str(root / "pytest.ini"),
         "-o", "tmp_path_retention_policy=failed", "--basetemp", str(tmp_path / "inner-bt"), str(probe)],
        cwd=root, capture_output=True, text=True, timeout=60,
        # The child options are explicit above; an outer -k filter would deselect its probe.
        env={key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
