"""Recorded catch-up backlog is distinct from kNN coverage and ordinary ingest lag."""
import json
import sqlite3

import pytest

from hermes_lcm import command, embedding_maintenance, recall_delivery, tools
from hermes_lcm.dag import SummaryNode
from hermes_lcm.vector_store import VectorStore
from tests.test_issue_1014_incremental_embedding import FakeProvider, _make_engine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, tmp_path / "home")
    monkeypatch.setattr(command, "resolve_provider", lambda c, **kw: FakeProvider(c.embedding_provider, c.embedding_model))
    monkeypatch.setattr(tools, "_resolve_recall_provider", lambda *a, **kw: FakeProvider("ollama", "model-a"))
    monkeypatch.setattr(tools, "_resolve_recall_chunk_provider", lambda *a, **kw: FakeProvider("ollama", "model-a"))
    monkeypatch.setattr(embedding_maintenance, "probe_provider_availability", lambda c: {"available": True}, raising=False)
    from hermes_lcm import embedding_provider
    monkeypatch.setattr(embedding_provider, "probe_provider_availability", lambda c: {"available": True})
    for i in range(4):
        engine._store.append("seed", {"role": "user", "content": f"cobalt backlog {i} " + "detail " * 60})
        engine._dag.add_node(SummaryNode(session_id="seed", depth=0, summary=f"cobalt backlog {i}",
                                        created_at=float(i), latest_at=float(i)))
    yield engine
    engine.shutdown()


def backfill(engine, task, limit=1, incremental=False):
    kwargs = dict(apply=True, limit=limit, retry_uncertain=False)
    if task == "chunk":
        return command._chunk_backfill_run(engine._config, engine._store.db_path,
                                          policy="", incremental=incremental, **kwargs)
    return command._embedding_backfill_summary_run(engine._config, engine._store.db_path, **kwargs)


def record(engine, task):
    with sqlite3.connect(engine._store.db_path) as conn:
        identity = conn.execute("SELECT identity_hash FROM lcm_embedding_profile WHERE active=1 AND task=?", (task,)).fetchone()[0]
        row = conn.execute("SELECT value FROM metadata WHERE key=?", (f"embedding_catchup:{task}:{identity}",)).fetchone()
    assert row is not None, "backfill did not persist catch-up state"
    result = json.loads(row[0])
    assert result["updated_at"] > 0
    return result


def recall(engine, task="all"):
    include = {"summary": "summaries", "chunk": "verbatim"}.get(task, task)
    return json.loads(tools.lcm_recall({"query": "cobalt", "include": include}, engine=engine))


@pytest.mark.parametrize("task,incremental", [("summary", False), ("chunk", False), ("chunk", True)])
def test_bounded_pass_discloses_then_drain_and_fresh_ingest_stay_clear(engine, task, incremental):
    result = backfill(engine, task, incremental=incremental)
    assert result["embedded"] > 0 and result["remaining"] > 0
    payload = recall(engine, task)
    assert payload["provenance"]["coverage"][task] == "full"
    assert payload["degraded"] is True
    assert "semantic index catching up" in payload["degraded_reason"]
    assert record(engine, task)["remaining"] > 0
    backfill(engine, task, limit=100, incremental=incremental)
    assert record(engine, task)["remaining"] == 0
    assert recall(engine, task)["degraded"] is False
    engine._store.append("seed", {"role": "user", "content": "cobalt fresh ingest " + "detail " * 60})
    assert recall(engine, task)["degraded"] is False


def test_automatic_pass_and_new_identity_rebuild_record_both_corpora(engine):
    engine._config.embedding_max_batch_items = 1
    assert "error" not in embedding_maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config)
    for task in ("summary", "chunk"):
        assert record(engine, task)["remaining"] > 0
    for _ in range(5):
        embedding_maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config)
    for task in ("summary", "chunk"):
        assert record(engine, task)["remaining"] == 0
    engine._config.embedding_model = "model-b"
    store = VectorStore(engine._store.db_path, config=engine._config)
    for task in ("summary", "chunk"):
        store.register_profile("model-b", "ollama", 2, task=task)
    store.close()
    for task in ("summary", "chunk"):
        backfill(engine, task)
        assert record(engine, task)["remaining"] > 0
        backfill(engine, task, limit=100)
        assert record(engine, task)["remaining"] == 0


def test_no_record_reasons_are_byte_equal_to_base(engine, monkeypatch):
    # Frozen base-1d991241 lists for this same store, captured before implementation.
    observed = []
    deliver = recall_delivery.deliver_recall

    def capture(*args, **kwargs):
        observed.append(json.dumps(kwargs["degraded_reasons"]).encode())
        return deliver(*args, **kwargs)

    monkeypatch.setattr(recall_delivery, "deliver_recall", capture)
    recall(engine)
    assert observed.pop() == b'["summary vectors are unavailable", "chunk vectors are unavailable"]'
    for task in ("summary", "chunk"):
        backfill(engine, task, limit=100)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("DELETE FROM metadata WHERE key LIKE 'embedding_catchup:%'")
    assert recall(engine)["degraded"] is False
    assert observed.pop() == b'[]'


def test_recall_reads_one_key_per_enabled_arm_without_record_scan(engine, monkeypatch):
    for task in ("summary", "chunk"):
        backfill(engine, task)
    statements = []
    connect = sqlite3.connect

    def traced(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced)
    recall(engine)
    reads = [s for s in statements if "embedding_catchup:" in s and s.lstrip().upper().startswith("SELECT")]
    assert len(reads) == 2
    for statement in reads:
        assert "COUNT(" not in statement.upper()
        assert "WHERE KEY" in statement.upper()
        with connect(engine._store.db_path) as conn:
            plan = conn.execute("EXPLAIN QUERY PLAN " + statement).fetchall()
        assert any("SEARCH metadata" in row[3] for row in plan)
        assert not any("SCAN metadata" in row[3] for row in plan)
    statements.clear()
    recall(engine, "summary")
    assert len([s for s in statements if "embedding_catchup:" in s]) == 1


@pytest.mark.parametrize("task", ["summary", "chunk"])
def test_metadata_write_failure_does_not_fail_pass(engine, task):
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("CREATE TRIGGER refuse_catchup BEFORE INSERT ON metadata "
                     "WHEN NEW.key LIKE 'embedding_catchup:%' "
                     "BEGIN SELECT RAISE(ABORT, 'simulated metadata failure'); END")
    result = backfill(engine, task)
    assert result["error"] is None
    assert result["embedded"] > 0
