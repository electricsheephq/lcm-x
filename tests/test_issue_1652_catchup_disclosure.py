"""Recorded catch-up backlog is distinct from kNN coverage and ordinary ingest lag."""
from contextlib import nullcontext
import json
import sqlite3
import threading
import time

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


def backfill(engine, task, limit=1, incremental=False, retry=False):
    kwargs = dict(apply=True, limit=limit, retry_uncertain=retry)
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


def mark_uncertain(engine, task, embedded_id):
    with sqlite3.connect(engine._store.db_path) as conn:
        identity = conn.execute("SELECT identity_hash FROM lcm_embedding_profile WHERE active=1 AND task=?", (task,)).fetchone()[0]
        conn.execute(
            "INSERT INTO lcm_embedding_backfill_inflight "
            "(embedded_id, identity_hash, lease_id, generation, claimed_at, state, updated_at) "
            "VALUES (?, ?, 'dead-lease', 1, ?, 'uncertain', ?)",
            (str(embedded_id), identity, time.time() - 3600, time.time() - 3600),
        )


@pytest.mark.parametrize("source", ["deleted", "policy"])
def test_orphan_uncertain_chunk_does_not_hold_backlog(engine, source):
    backfill(engine, "chunk", limit=100)
    store_id = engine._store.append("doomed", {"role": "user", "content": "cobalt doomed " + "detail " * 60})
    mark_uncertain(engine, "chunk", f"{store_id}:{0 if source == 'deleted' else 999}")
    if source == "deleted":
        engine._store.delete_session_messages("doomed")
        with sqlite3.connect(engine._store.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_backfill_inflight").fetchone()[0] == 0
    backfill(engine, "chunk", limit=100)
    assert record(engine, "chunk")["remaining"] == 0
    assert recall(engine, "chunk")["degraded"] is False
    backfill(engine, "chunk", limit=100, retry=True)
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_backfill_inflight").fetchone()[0] == (source == "policy")


@pytest.mark.parametrize("task", ["summary", "chunk"])
def test_uncertain_rows_name_operator_retry(engine, task):
    backfill(engine, task, limit=100)
    if task == "summary":
        row = engine._dag.add_node(SummaryNode(session_id="seed", depth=0, summary="cobalt timeout row", created_at=9.0, latest_at=9.0))
        embedded_id = getattr(row, "node_id", row)
    else:
        row = engine._store.append("seed", {"role": "user", "content": "cobalt timeout " + "detail " * 60})
        embedded_id = f"{row}:0"
    mark_uncertain(engine, task, embedded_id)
    for _ in range(4):
        embedding_maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config)
    reason = recall(engine, task)["degraded_reason"]
    corpus = "chunks" if task == "chunk" else "summary"
    assert f"{task} semantic index has 1 rows awaiting /lcm embed backfill --apply --retry-uncertain --corpus {corpus}" in reason
    assert "catching up" not in reason
    backfill(engine, task, limit=100, retry=True)
    assert recall(engine, task)["degraded"] is False


@pytest.mark.parametrize("batch", [False, True])
def test_remote_chunk_ingest_marks_backlog_until_manual_backfill(engine, batch):
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    assert not embedding_maintenance._automatic_chunks_allowed(engine._config, command)
    backfill(engine, "chunk", limit=100)
    assert recall(engine, "chunk")["degraded"] is False
    message = {"role": "user", "content": "cobalt later " + "detail " * 60}
    if batch:
        engine._store.append_batch("seed", [message])
    else:
        engine._store.append("seed", message)
    for _ in range(3):
        embedding_maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config)
    assert record(engine, "chunk")["remaining"] > 0
    assert recall(engine, "chunk")["degraded"] is True
    backfill(engine, "chunk", limit=100)
    assert record(engine, "chunk")["remaining"] == 0
    assert recall(engine, "chunk")["degraded"] is False


def test_completing_record_write_waits_for_foreground_transaction(engine, monkeypatch):
    backfill(engine, "summary")
    assert record(engine, "summary")["remaining"] > 0
    write = command._record_embedding_catchup
    held = threading.Event()

    def hold_lock():
        with sqlite3.connect(engine._store.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            held.set()
            time.sleep(0.3)

    def contended(*args, **kwargs):
        thread = threading.Thread(target=hold_lock)
        thread.start()
        try:
            assert held.wait(5)
            write(*args, **kwargs)
        finally:
            thread.join(5)

    monkeypatch.setattr(command, "_record_embedding_catchup", contended)
    result = backfill(engine, "summary", limit=100)
    assert result["remaining"] == 0
    assert record(engine, "summary")["remaining"] == 0
    assert recall(engine, "summary")["degraded"] is False


def test_unrebuildable_summary_is_not_counted(engine):
    backfill(engine, "summary", limit=100)
    mark_uncertain(engine, "summary", 999999)
    backfill(engine, "summary", limit=100)
    assert record(engine, "summary")["remaining"] == 0
    assert record(engine, "summary")["operator_retry"] == 0
    assert recall(engine, "summary")["degraded"] is False


def test_remote_chunk_metadata_failure_does_not_fail_ingest(engine):
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    backfill(engine, "chunk", limit=100)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("CREATE TRIGGER refuse_catchup BEFORE UPDATE ON metadata "
                     "WHEN NEW.key LIKE 'embedding_catchup:%' "
                     "BEGIN SELECT RAISE(ABORT, 'simulated metadata failure'); END")
    message = {"role": "user", "content": "cobalt survives advisory failure"}
    single = engine._store.append("seed", message)
    batch = engine._store.append_batch("seed", [message])
    for store_id in [single, *batch]:
        assert engine._store.get(store_id)["content"] == message["content"]


@pytest.mark.parametrize("override", ["conversational", "heads"])
def test_retry_policy_override_preserves_configured_backlog(engine, override):
    # Supplied C-policy/verify-policy probes: tool chunks exist only under full.
    engine._config.embedding_content_policy = "full"
    backfill(engine, "chunk", limit=100)
    sid = engine._store.append("seed", {"role": "tool", "content": "cobalt output. " * 600})
    mark_uncertain(engine, "chunk", f"{sid}:1")
    backfill(engine, "chunk", limit=100)
    assert record(engine, "chunk")["operator_retry"] == 1
    command._embedding_backfill_text(
        ["--apply", "--retry-uncertain", "--corpus", "chunks", "--policy", override], engine)
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT state FROM lcm_embedding_backfill_inflight WHERE embedded_id=?",
                            (f"{sid}:1",)).fetchone() == ("uncertain",)
    assert record(engine, "chunk")["operator_retry"] == 1
    assert recall(engine, "chunk")["degraded"] is True
    for _ in range(3):
        embedding_maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config)
    assert recall(engine, "chunk")["degraded"] is True
    backfill(engine, "chunk", limit=100, retry=True)
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT 1 FROM lcm_chunk_meta WHERE chunk_id=?", (f"{sid}:1",)).fetchone()
    assert recall(engine, "chunk")["degraded"] is False


@pytest.mark.parametrize("content", [None, ""])
def test_retry_purges_only_source_gone_or_empty(engine, content):
    backfill(engine, "chunk", limit=100)
    sid = engine._store.append("seed", {"role": "user", "content": "cobalt source"})
    mark_uncertain(engine, "chunk", f"{sid}:0")
    with sqlite3.connect(engine._store.db_path) as conn:
        if content is None:
            conn.execute("DELETE FROM messages WHERE store_id=?", (sid,))
        else:
            conn.execute("UPDATE messages SET content='' WHERE store_id=?", (sid,))
    backfill(engine, "chunk", limit=100, retry=True)
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_backfill_inflight").fetchone()[0] == 0


@pytest.mark.parametrize("batch", [False, True])
def test_backlog_mark_rollback_cannot_rollback_ingest(engine, batch, caplog):
    # Supplied F1 probe: RAISE(ROLLBACK) models a transaction-aborting error.
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    backfill(engine, "chunk", limit=100)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("CREATE TRIGGER abort_txn BEFORE UPDATE ON metadata "
                     "WHEN NEW.key LIKE 'embedding_catchup:%' "
                     "BEGIN SELECT RAISE(ROLLBACK, 'simulated transaction-aborting error'); END")
    message = {"role": "user", "content": "cobalt committed"}
    with caplog.at_level("DEBUG", logger="hermes_lcm.store"):
        ids = (engine._store.append_batch("seed", [message] * 3) if batch
               else [engine._store.append("seed", message)])
    with sqlite3.connect(engine._store.db_path) as conn:
        for sid in ids:
            assert conn.execute("SELECT content FROM messages WHERE store_id=?", (sid,)).fetchone() == (message["content"],)
    assert "Could not mark manual chunk backlog" in caplog.text
    assert engine._store.append("seed", message) > max(ids)


@pytest.mark.parametrize("task,corpus", [("summary", "summary"), ("chunk", "chunks")])
def test_operator_retry_reason_is_executable_for_its_corpus(engine, monkeypatch, task, corpus):
    backfill(engine, task, limit=100)
    if task == "summary":
        row = engine._dag.add_node(SummaryNode(session_id="seed", depth=0, summary="cobalt retry", created_at=9., latest_at=9.))
        embedded_id = getattr(row, "node_id", row)
    else:
        embedded_id = f"{engine._store.append('seed', {'role': 'user', 'content': 'cobalt retry'})}:0"
    mark_uncertain(engine, task, embedded_id)
    backfill(engine, task, limit=100)
    reason = recall(engine, task)["degraded_reason"]
    tokens = reason.split("awaiting /lcm embed backfill ", 1)[1].split()
    options = command._embedding_backfill_options(tokens)
    assert not isinstance(options, str), options
    assert options[0] is True and options[2] is True and options[3] == corpus
    seen = []
    monkeypatch.setattr(command, "_chunk_backfill_text", lambda *a, **kw: seen.append("chunks") or "ok")
    monkeypatch.setattr(command, "_embedding_backfill_summary_run", lambda *a, **kw: seen.append("summary") or "ok")
    command._embedding_backfill_text(tokens, engine)
    assert seen == [corpus]


@pytest.mark.parametrize("initial", [0, False, 7])
def test_backlog_mark_makes_count_unknown_and_does_not_rewrite(engine, initial):
    # A known count goes stale once new rows arrive, so ingest turns it into the
    # unknown-backlog marker; once marked, later ingest does not rewrite the row.
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    backfill(engine, "chunk", limit=100)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE metadata SET value=json_set(value, '$.remaining', json(?)) "
                     "WHERE key LIKE 'embedding_catchup:chunk:%'", (json.dumps(initial),))
        conn.execute("CREATE TABLE mark_updates(n INTEGER)")
        conn.execute("INSERT INTO mark_updates VALUES(0)")
        conn.execute("CREATE TRIGGER count_marks AFTER UPDATE ON metadata "
                     "WHEN NEW.key LIKE 'embedding_catchup:chunk:%' "
                     "BEGIN UPDATE mark_updates SET n=n+1; END")
    for _ in range(3):
        engine._store.append("seed", {"role": "user", "content": "cobalt later"})
    assert record(engine, "chunk")["remaining"] is True
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT n FROM mark_updates").fetchone()[0] == 1


def test_import_marks_cloud_chunk_backlog_after_commit(engine, tmp_path, monkeypatch):
    from tests.test_import_lossless_claw import load_importer_module, write_jsonl_session, jsonl_header, jsonl_message
    from hermes_lcm.ingest_protection import embedding_privacy_revision
    engine._config.embedding_provider = "voyage"
    store = VectorStore(engine._store.db_path, config=engine._config)
    store.register_profile("model-a", "voyage", 2, task="chunk", revision=embedding_privacy_revision(engine._config))
    store.close()
    monkeypatch.setattr(tools, "_resolve_recall_chunk_provider", lambda *a, **kw: FakeProvider("voyage", "model-a"))
    def fill():
        result = command._chunk_backfill_run(engine._config, engine._store.db_path, apply=True, limit=100,
                                             retry_uncertain=False, policy="", confirm_raw_text=True)
        assert result["error"] is None, result
    fill()
    assert recall(engine, "chunk")["degraded"] is False
    for key, value in {"LCM_EMBEDDINGS_ENABLED": "true", "LCM_EMBEDDING_PROVIDER": "voyage",
                       "LCM_EMBEDDING_MODEL": "model-a"}.items():
        monkeypatch.setenv(key, value)
    source = tmp_path / "import" / "new.jsonl"
    write_jsonl_session(source, [jsonl_header("imported")] + [
        jsonl_message(f"m{i}", "user", "cobalt imported " + "detail " * 60) for i in range(3)])
    importer = load_importer_module()
    result = importer.import_jsonl_sessions(files=[source], target_db=engine._store.db_path,
                                           namespace="openclaw-jsonl", agent="fixture", import_id="fix2", apply=True)
    assert result.imported == 3
    assert record(engine, "chunk")["remaining"] is True
    assert recall(engine, "chunk")["degraded"] is True
    fill()
    assert recall(engine, "chunk")["degraded"] is False


def test_actionable_count_chunks_each_source_once(engine, monkeypatch):
    # Supplied D cost probe, bounded to deterministic rebuild-call counts.
    from hermes_lcm.chunking import chunk_message
    backfill(engine, "chunk", limit=100)
    sources = set()
    total = 0
    for _ in range(3):
        sid = engine._store.append("bulk", {"role": "assistant", "content": "cobalt sentence. " * 3750})
        sources.add(sid)
        stored = engine._store.get(sid)
        for chunk in chunk_message(sid, stored["role"], stored["content"], policy="conversational"):
            mark_uncertain(engine, "chunk", chunk.chunk_id)
            total += 1
    calls = []
    def counted(sid, *args, **kwargs):
        calls.append(sid)
        return chunk_message(sid, *args, **kwargs)
    monkeypatch.setattr(command, "chunk_message", counted)
    # Isolate actionable counting from the separately bounded discovery scan.
    monkeypatch.setattr(command, "_chunk_pending_rows", lambda *a, **kw: (0, [], {}))
    with sqlite3.connect(engine._store.db_path) as conn:
        identity = conn.execute("SELECT identity_hash FROM lcm_embedding_profile WHERE active=1 AND task='chunk'").fetchone()[0]
    assert command._chunk_backfill_remaining(engine._store.db_path, identity, "conversational", 0, 0, True) == (0, 0, total)
    assert len(calls) <= len(sources)


@pytest.mark.parametrize("abort", [False, True])
def test_atomic_ingest_defers_backlog_mark_until_outer_commit(engine, abort):
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    backfill(engine, "chunk", limit=100)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("CREATE TRIGGER abort_txn BEFORE UPDATE ON metadata "
                     "WHEN NEW.key LIKE 'embedding_catchup:%' "
                     "BEGIN SELECT RAISE(ROLLBACK, 'simulated transaction-aborting error'); END")
    ids = []
    try:
        with engine._store.atomic():
            with engine._store.atomic():
                ids.append(engine._store.append("seed", {"role": "user", "content": "cobalt atomic"}))
            with sqlite3.connect(engine._store.db_path) as conn:
                assert conn.execute("SELECT 1 FROM messages WHERE store_id=?", (ids[0],)).fetchone() is None
            if abort:
                raise RuntimeError("abort ingest")
    except RuntimeError:
        assert abort
    with sqlite3.connect(engine._store.db_path) as conn:
        assert bool(conn.execute("SELECT 1 FROM messages WHERE store_id=?", (ids[0],)).fetchone()) is not abort
    assert engine._store._manual_chunk_backlog_pending is False


@pytest.mark.parametrize("atomic", [True, False])
def test_backlog_policy_error_preserves_ingest(engine, monkeypatch, caplog, atomic):
    engine._config.embedding_provider = "voyage"
    assert not embedding_maintenance._automatic_chunks_allowed(engine._config, command)
    store = engine._store
    assert not store._conn.in_transaction
    message = {"role": "user", "content": "cobalt survives backlog policy error"}

    def fail_policy(*args, **kwargs):
        raise RuntimeError("simulated backlog policy error")

    with monkeypatch.context() as patch:
        patch.setattr(embedding_maintenance, "_automatic_chunks_allowed", fail_policy)
        with caplog.at_level("DEBUG", logger="hermes_lcm.store"):
            with store.atomic() if atomic else nullcontext():
                sid = store.append("seed", message)

    assert "Could not mark manual chunk backlog" in caplog.text
    assert "simulated backlog policy error" in caplog.text
    assert not store._conn.in_transaction
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT content FROM messages WHERE store_id=?", (sid,)).fetchone() == (message["content"],)
    later = store.append("seed", {"role": "user", "content": "cobalt store still usable"})
    assert later > sid
    assert store.get(later)["content"] == "cobalt store still usable"
