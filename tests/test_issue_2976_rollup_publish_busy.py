"""A finished rollup waits for a competing WAL writer before publishing."""

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timezone
from unittest.mock import Mock

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.rollup_builder import build_day
from hermes_lcm.rollup_store import RollupStore

DAY = date(2026, 11, 1)
SCOPE = "synthetic-rollup-scope"


@pytest.fixture
def rollup_db(tmp_path):
    db_path = tmp_path / "rollups.db"
    timestamp = datetime(2026, 11, 1, 12, tzinfo=timezone.utc).timestamp()
    with closing(SummaryDAG(db_path)) as dag:
        source_ids = [
            dag.add_node(SummaryNode(
                session_id=SCOPE, summary=summary, created_at=timestamp,
                earliest_at=timestamp, latest_at=timestamp,
            ))
            for summary in ("first source", "second source")
        ]
        with (
            closing(RollupStore(db_path)) as store,
            closing(sqlite3.connect(db_path, check_same_thread=False)) as writer,
        ):
            assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            writer.execute("CREATE TABLE synthetic_ingest (id INTEGER)")
            yield store, dag, source_ids, writer


def _hold_write_lock(writer):
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO synthetic_ingest VALUES (1)")


def _release_during_publish(store, writer, pool):
    # Start the delay from publication's first SQL statement so a slow start
    # cannot let the competing transaction commit before mark_ready enters SQLite.
    publish_started = threading.Event()
    store.connection.set_trace_callback(lambda _sql: publish_started.set())

    def commit():
        assert publish_started.wait(30), "publication did not reach SQLite"
        time.sleep(1)
        with closing(sqlite3.connect(f"{store.db_path.as_uri()}?mode=ro", uri=True)) as reader:
            assert reader.execute(
                "SELECT status FROM lcm_rollups WHERE period_kind = 'day' "
                "AND period_start = ? AND scope = ?",
                (DAY.isoformat(), SCOPE),
            ).fetchone() == ("building",)
        writer.commit()

    return pool.submit(commit)


def test_mark_ready_waits_for_other_writer(rollup_db, monkeypatch):
    store, _dag, source_ids, writer = rollup_db
    token = store.upsert_building("day", DAY.isoformat(), SCOPE)
    mark_failed = Mock(wraps=store.mark_failed)
    monkeypatch.setattr(store, "mark_failed", mark_failed)

    _hold_write_lock(writer)
    with ThreadPoolExecutor(max_workers=1) as pool:
        release = _release_during_publish(store, writer, pool)
        assert store.mark_ready(token, "completed summary", 3, source_ids, "fp") is True
        release.result()

    row = store.get_rollup("day", DAY.isoformat(), SCOPE)
    assert row["status"] == "ready"
    assert row["source_node_ids"] == source_ids
    mark_failed.assert_not_called()


def test_mark_ready_respects_busy_timeout(rollup_db):
    store, _dag, source_ids, writer = rollup_db
    token = store.upsert_building("day", DAY.isoformat(), SCOPE)
    store.connection.execute("PRAGMA busy_timeout = 200")

    _hold_write_lock(writer)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        store.mark_ready(token, "completed summary", 3, source_ids, "fp")

    assert writer.in_transaction
    assert not store.connection.in_transaction
    row = store.get_rollup("day", DAY.isoformat(), SCOPE)
    assert row["status"] == "building"
    assert row["source_node_ids"] == []


def test_build_day_publishes_once_after_other_writer_commits(rollup_db, monkeypatch):
    store, dag, source_ids, writer = rollup_db
    config = LCMConfig(database_path=str(store.db_path))
    mark_failed = Mock(wraps=store.mark_failed)
    monkeypatch.setattr(store, "mark_failed", mark_failed)
    releases = []

    with ThreadPoolExecutor(max_workers=1) as pool:
        def summarize(_text, **_kwargs):
            _hold_write_lock(writer)
            releases.append(_release_during_publish(store, writer, pool))
            return "completed daily summary", 1

        summarizer = Mock(side_effect=summarize)
        result = build_day(store, dag, config, SCOPE, DAY, summarizer=summarizer)
        summarizer.assert_called_once()
        releases[0].result()

    assert result is not None
    assert result["status"] == "ready"
    assert result["summary"] == "completed daily summary"
    assert result["source_node_ids"] == source_ids
    mark_failed.assert_not_called()
