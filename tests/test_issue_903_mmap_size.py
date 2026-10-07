"""#903: configurable SQLite memory mapping, with unchanged default reads."""

import logging
import sqlite3
from contextlib import closing

import pytest

from hermes_lcm import config as config_mod
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.db_bootstrap import configure_connection
from hermes_lcm.engine import LCMEngine


DEFAULT_BYTES = 268_435_456


def _pre_change_value(path, requested=DEFAULT_BYTES):
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"PRAGMA mmap_size={requested}")
        return conn.execute("PRAGMA mmap_size").fetchone()[0]


def _configured_value(path):
    with closing(sqlite3.connect(path)) as conn:
        configure_connection(conn)
        return conn.execute("PRAGMA mmap_size").fetchone()[0]


def test_unset_keeps_pre_change_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LCM_SQLITE_MMAP_SIZE", raising=False)
    expected = _pre_change_value(tmp_path / "baseline.db")
    assert _configured_value(tmp_path / "default.db") == expected


def test_zero_disables_mmap(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_SQLITE_MMAP_SIZE", "0")
    assert _configured_value(tmp_path / "disabled.db") == 0


def test_positive_bytes_are_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_SQLITE_MMAP_SIZE", "1048576")
    expected = _pre_change_value(tmp_path / "baseline.db", 1_048_576)
    assert _configured_value(tmp_path / "configured.db") == expected


@pytest.mark.parametrize("raw", ["abc", "-5", ""])
def test_invalid_uses_default_and_warns_once(tmp_path, monkeypatch, caplog, raw):
    monkeypatch.setattr(config_mod, "_sqlite_mmap_size_warned", False, raising=False)
    monkeypatch.setenv("LCM_SQLITE_MMAP_SIZE", raw)
    expected = _pre_change_value(tmp_path / "baseline.db")
    with caplog.at_level(logging.WARNING, logger=config_mod.__name__):
        for index in range(3):
            assert _configured_value(tmp_path / f"invalid-{index}.db") == expected
        monkeypatch.setenv("LCM_SQLITE_MMAP_SIZE", "another-invalid-value")
        assert _configured_value(tmp_path / "other-invalid.db") == expected
    warnings = [r for r in caplog.records if "LCM_SQLITE_MMAP_SIZE" in r.message]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING


@pytest.mark.parametrize("raw", [None, "0", "1048576", "9223372036854775807"])
def test_doctor_reports_effective_connection_value(tmp_path, monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("LCM_SQLITE_MMAP_SIZE", raising=False)
    else:
        monkeypatch.setenv("LCM_SQLITE_MMAP_SIZE", raw)
    config = LCMConfig(database_path=str(tmp_path / "doctor.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        effective = engine._store.connection.execute("PRAGMA mmap_size").fetchone()[0]
        # Changing the env after opening must not misreport the live connection.
        monkeypatch.setenv("LCM_SQLITE_MMAP_SIZE", "2097152")
        assert f"sqlite_mmap_bytes={effective}" in handle_lcm_command("doctor", engine)
    finally:
        engine.shutdown()
