"""#1084: report SQLite WAL-reset exposure without changing DB behavior."""

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command
import hermes_lcm.db_bootstrap as bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tools import lcm_doctor


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
    if concurrent and affected:
        # The flag is deliberately unsynchronized (see db_bootstrap), so racing callers may both warn.
        assert 1 <= len(warnings) <= 2
    else:
        assert len(warnings) == int(affected)
    if affected:
        assert warnings[0].levelno == logging.WARNING
        assert warnings[0].message == (
            f"LCM-X: Linked SQLite {sqlite3.sqlite_version} has the WAL-reset bug (#1085). "
            "In WAL mode with several connections, whole recently committed transactions can vanish. "
            "Fixed in SQLite 3.51.3. Upgrade to a Python whose SQLite is 3.51.3 or newer "
            "(or 3.50.7+ on the 3.50 branch, 3.44.6+ on the 3.44 branch). "
            "See https://sqlite.org/wal.html#walresetbug."
        )
    else:
        assert caplog.records == []


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
@pytest.mark.parametrize("version, affected", [
    ((3, 46, 1), True), ((3, 51, 3), False), ((3, 53, 1), False),
    ((3, 50, 7), False), ((3, 44, 6), False),
])
def test_linked_version_on_all_warning_surfaces(
    tmp_path, monkeypatch, caplog, journal_mode, version, affected,
):
    version_text = ".".join(map(str, version))
    monkeypatch.setattr(sqlite3, "sqlite_version_info", version)
    monkeypatch.setattr(sqlite3, "sqlite_version", version_text)
    monkeypatch.setattr(bootstrap, "_wal_reset_warned", False)
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "doctor.db"), temporal_rollups_enabled=False,
        embeddings_enabled=False,
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
        status = command._status_text(engine)
        doctor = command._doctor_text(engine).splitlines()
        tool = json.loads(lcm_doctor({}, engine=engine))
        startup = [record.message for record in caplog.records
                   if record.name == bootstrap.__name__ and "WAL-reset bug" in record.message]
        library_status = "affected" if affected else "not affected"
        assert f"sqlite_library: {version_text} (WAL-reset bug: {library_status})" in doctor
        assert ("sqlite_wal_reset: warn " in status) is affected
        assert any(line.startswith("sqlite_wal_reset: warn ") for line in doctor) is affected
        assert doctor[1] == ("status: action-recommended" if affected else "status: ok"), "\n".join(doctor)
        assert "issues: none" in doctor, "\n".join(doctor)
        assert tool["overall"] == ("warnings" if affected else "healthy"), tool["guidance"]
        assert [check["status"] for check in tool["checks"] if check["check"] == "sqlite_wal_reset"] == (
            ["warn"] if affected else []
        )
        assert bool(startup) is affected
        if affected:
            assert "whole recently committed transactions can vanish" in startup[0]
    finally:
        engine.shutdown()
