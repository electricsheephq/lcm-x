"""#1266: embedding failures and warning re-arming belong to each database."""

import logging

import pytest

from hermes_lcm import embedding_maintenance as maintenance


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    for name in ("_FAILURES", "_FAILURE_WARNED", "_LAST_PASS"):
        monkeypatch.setattr(maintenance, name, type(getattr(maintenance, name))())


def test_two_databases_have_independent_counts_and_warnings(tmp_path, caplog):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    with caplog.at_level(logging.DEBUG, logger=maintenance.__name__):
        maintenance._record(b, "failed", "B missing model")
        maintenance._record(b, "failed", "B still missing")
        maintenance._record(a, "failed", "A missing model")
    assert maintenance.incremental_embedding_state(a)[1] == 1
    assert maintenance.incremental_embedding_state(b)[1] == 2
    assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.DEBUG, logging.WARNING]


def test_success_resets_only_its_database_and_rearms_warning(tmp_path, caplog):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    maintenance._record(a, "failed", "A failed")
    maintenance._record(b, "failed", "B failed")
    maintenance._record(a, "complete", None)
    assert maintenance.incremental_embedding_state(a)[1] == 0
    assert maintenance.incremental_embedding_state(b)[1] == 1
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=maintenance.__name__):
        maintenance._record(a, "failed", "A failed again")
        maintenance._record(b, "failed", "B still failed")
    assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.DEBUG]


def test_single_database_repeats_and_success(tmp_path, caplog):
    db = tmp_path / "single.db"
    with caplog.at_level(logging.DEBUG, logger=maintenance.__name__):
        maintenance._record(db, "failed", "missing")
        maintenance._record(db, "failed", "missing")
    assert maintenance.incremental_embedding_state(db)[1] == 2
    assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.DEBUG]
    maintenance._record(db, "complete", None)
    assert maintenance.incremental_embedding_state(db)[0][1] == "complete"
    assert maintenance.incremental_embedding_state(db)[1] == 0
