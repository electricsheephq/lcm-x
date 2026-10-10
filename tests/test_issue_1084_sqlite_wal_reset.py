"""#1084: report SQLite WAL-reset exposure without changing DB behavior."""

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command
import hermes_lcm.db_bootstrap as bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize("version, affected", [
    ((3, 45, 1), True), ((3, 51, 2), True), ((3, 51, 3), False),
    ((3, 50, 7), False), ((3, 50, 6), True), ((3, 44, 6), False),
    ((3, 44, 5), True), ((3, 6, 23), False), ((3, 54, 0), False),
    ((3, 7, 0), True), ((3, 51, 0), True), ((3, 45, 0), True),
])
def test_version_table(version, affected):
    assert bootstrap.sqlite_wal_reset_affected(version) is affected


def test_version_defaults_to_linked_library(monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 45, 1))
    assert bootstrap.sqlite_wal_reset_affected() is True
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 51, 3))
    assert bootstrap.sqlite_wal_reset_affected() is False


@pytest.mark.parametrize("affected", [True, False])
@pytest.mark.parametrize("concurrent", [False, True])
def test_warning_once_per_process(tmp_path, monkeypatch, caplog, affected, concurrent):
    monkeypatch.setattr(bootstrap, "sqlite_wal_reset_affected", lambda: affected)
    monkeypatch.setattr(bootstrap, "_wal_reset_warned", False)
    db_path = tmp_path / "warning.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sample (value INTEGER)")
    conn.close()

    def open_connection(_):
        conn = sqlite3.connect(db_path)
        try:
            bootstrap.configure_connection(conn)
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()

    with caplog.at_level(logging.WARNING, logger=bootstrap.__name__):
        if concurrent:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(open_connection, range(2)))
        else:
            for index in range(2):
                open_connection(index)
    warnings = [record for record in caplog.records if "walresetbug" in record.message]
    assert len(warnings) == int(affected)
    if affected:
        assert warnings[0].levelno == logging.WARNING
        assert warnings[0].message == (
            f"LCM-X: the linked SQLite {sqlite3.sqlite_version} has the WAL-reset bug "
            "(https://sqlite.org/wal.html#walresetbug), which can rarely corrupt "
            "a WAL database written by several connections; upgrade to a Python "
            "whose SQLite is 3.51.3 or newer (or 3.50.7+ on the 3.50 branch, 3.44.6+ on the 3.44 branch)."
        )
    else:
        assert caplog.records == []


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_doctor_advisory_preserves_verdict_and_other_lines(tmp_path, monkeypatch, journal_mode):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "doctor.db"), temporal_rollups_enabled=False,
    ), hermes_home=str(tmp_path))
    try:
        # Only substitute the journal-mode read; never convert the test database.
        real_conn = engine._store.connection

        class JournalReadProxy:
            def execute(self, sql, *args, **kwargs):
                if sql == "PRAGMA journal_mode":
                    return SimpleNamespace(fetchone=lambda: (journal_mode,))
                return real_conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(real_conn, name)

        monkeypatch.setattr(engine._store, "_conn", JournalReadProxy())
        monkeypatch.setattr(command, "sqlite_wal_reset_affected", lambda: False)
        baseline = command._doctor_text(engine).splitlines()
        assert f"sqlite_library: {sqlite3.sqlite_version} (WAL-reset bug: not affected)" in baseline
        assert not any(line.startswith("sqlite_wal_reset_advice:") for line in baseline)
        monkeypatch.setattr(command, "sqlite_wal_reset_affected", lambda: True)
        affected = command._doctor_text(engine).splitlines()
        assert f"sqlite_library: {sqlite3.sqlite_version} (WAL-reset bug: affected)" in affected
        advice = "sqlite_wal_reset_advice: upgrade Python/SQLite to 3.51.3+; see https://sqlite.org/wal.html#walresetbug"
        assert (advice in affected) is (journal_mode == "wal")
        # This compares status, issues, recommended_actions and every old line byte for byte.
        def old_lines(lines):
            return [line for line in lines if not line.startswith(("sqlite_library:", "sqlite_wal_reset_advice:"))]
        assert old_lines(affected) == old_lines(baseline)
    finally:
        engine.shutdown()
