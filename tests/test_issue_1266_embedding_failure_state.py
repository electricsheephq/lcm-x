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


@pytest.mark.parametrize("outcome", [
    "circuit_open", "model_preparing", "model_unavailable", "lease_held",
    "slice_budget", "stopped", "no_profile", "model_prepare_exhausted",
    "summaries=refused", "summaries=complete:0 chunks=lease_held",
    "chunks=partial:0 row_backoff", "chunks=partial:0 no_progress",
])
def test_deferral_preserves_failure_and_spent_warning(tmp_path, monkeypatch, caplog, outcome):
    db = tmp_path / "deferred.db"
    outcomes = iter([("error", "first"), (outcome, None), ("error", "second")])
    monkeypatch.setattr(maintenance, "_run_pass", lambda *_a: next(outcomes))
    with caplog.at_level(logging.DEBUG, logger=maintenance.__name__):
        for _ in range(3):
            maintenance.run_incremental_embedding_pass(db, object())
    assert maintenance.incremental_embedding_state(db)[1] == 2
    assert [r.levelno for r in caplog.records if "pass skipped" in r.message] == [
        logging.WARNING, logging.DEBUG,
    ]


@pytest.mark.parametrize("outcome", [
    "complete", "summaries=complete:0 chunks=complete:0 complete",
    "summaries=complete:3", "chunks=partial:2", "chunks=complete:0 complete",
])
def test_completed_pass_rearms_warning(tmp_path, monkeypatch, caplog, outcome):
    db = tmp_path / "completed.db"
    outcomes = iter([("error", "first"), (outcome, None), ("error", "second")])
    monkeypatch.setattr(maintenance, "_run_pass", lambda *_a: next(outcomes))
    maintenance.run_incremental_embedding_pass(db, object())
    maintenance.run_incremental_embedding_pass(db, object())
    assert maintenance.incremental_embedding_state(db)[1] == 0
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=maintenance.__name__):
        maintenance.run_incremental_embedding_pass(db, object())
    assert maintenance.incremental_embedding_state(db)[1] == 1
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_mixed_pass_with_embedded_rows_still_counts_failure(tmp_path, monkeypatch, caplog):
    db = tmp_path / "mixed.db"
    monkeypatch.setattr(maintenance, "_run_pass", lambda *_a: (
        "summaries=complete:3 chunks=failed", "chunks: provider failed"))
    with caplog.at_level(logging.DEBUG, logger=maintenance.__name__):
        for expected_count in (1, 2, 3):
            maintenance.run_incremental_embedding_pass(db, object())
            assert maintenance.incremental_embedding_state(db)[1] == expected_count
    assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
