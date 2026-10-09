"""The eight acceptance groups from SPEC-1021 (synthetic content only)."""
import json
from pathlib import Path
import re
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from hermes_lcm import command, db_bootstrap as boot, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, build_nodes_fts_spec
from hermes_lcm.store import MessageStore, build_message_fts_spec
from hermes_lcm.trajectory_store import CorpusIdentity, TrajectoryStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "dispatch_background_stem_backfill", lambda *a: None)
    obj = MessageStore(tmp_path / "stem.db")
    yield obj
    boot.join_background_stem_backfills(10)
    obj.close()


def partial(store, count=3):
    from hermes_lcm.store import build_message_stem_fts_spec
    conn = store.connection
    boot._drop_fts_artifacts(conn, build_message_stem_fts_spec())
    conn.execute("DELETE FROM metadata WHERE key LIKE 'fts_stem_%'")
    conn.executemany(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES(?, 'tool', ?, 1)",
        [(f"session-{i}", "visit deployment organization generic") for i in range(count)],
    )
    conn.commit()
    boot.ensure_message_stem_fts(conn, build_message_stem_fts_spec())
    return conn


def finish(store, stamp="test"):
    from hermes_lcm.store import build_message_stem_fts_spec
    boot._run_stem_backfill(str(store.db_path), build_message_stem_fts_spec(), stamp)


def consistent(conn):
    assert conn.execute("SELECT COUNT(*) FROM messages_fts_stem_docsize").fetchone()[0] == conn.execute(
        "SELECT COUNT(*) FROM messages"
    ).fetchone()[0]
    conn.execute("INSERT INTO messages_fts_stem(messages_fts_stem, rank) VALUES('integrity-check', 1)")
    conn.commit()
    assert boot._stem_metadata(conn, "fts_stem_state") == "ready"


def test_1_guarded_triggers_and_bulk_writers(store):
    conn = partial(store, 5)
    conn.execute("INSERT INTO messages_fts_stem(rowid, content) SELECT store_id, content FROM messages WHERE store_id=1")
    conn.commit()
    # Both not-yet-indexed and already-indexed rows traverse guarded triggers.
    assert store.gc_externalized_tool_result(4, "visit rewritten")
    assert store.delete_session_messages("session-4") == 1
    conn.execute("UPDATE messages SET content='visit updated' WHERE store_id=1")
    conn.commit()
    dag = SummaryDAG(store.db_path)
    try:
        engine = SimpleNamespace(_store=store, _dag=dag, _session_id="active")
        assert command._delete_clean_candidates_atomically(engine, {"session-1"})["messages_deleted"] == 1
    finally:
        dag.close()
    finish(store)
    consistent(conn)


def test_2_backfill_low_disk_interrupted_resumed_concurrent(store, monkeypatch):
    conn = partial(store, 2501)
    real_disk = boot._check_disk_space
    monkeypatch.setattr(boot, "_check_disk_space", lambda _: False)
    finish(store)
    assert conn.execute("SELECT COUNT(*) FROM messages_fts_stem_docsize").fetchone()[0] == 0
    monkeypatch.setattr(boot, "_check_disk_space", real_disk)
    real_sleep = boot.time.sleep
    batches = []

    def interrupt(_):
        batches.append(1)
        if len(batches) == 2:
            raise RuntimeError("simulated interrupted backfill")

    monkeypatch.setattr(boot.time, "sleep", interrupt)
    finish(store)
    assert conn.execute("SELECT COUNT(*) FROM messages_fts_stem_docsize").fetchone()[0] == 2000
    assert boot._stem_metadata(conn, "fts_stem_state") == "backfilling"
    monkeypatch.setattr(boot.time, "sleep", real_sleep)
    workers = [threading.Thread(target=finish, args=(store, str(i))) for i in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert not worker.is_alive()
    consistent(conn)
    assert conn.execute("SELECT COUNT(DISTINCT id) FROM messages_fts_stem_docsize").fetchone()[0] == 2501


@pytest.mark.parametrize("state", ["absent", "backfilling", "error", "ready"])
def test_3_recall_degradation_and_stem_hit(store, monkeypatch, state):
    conn = partial(store)
    finish(store)
    dag = SummaryDAG(store.db_path)
    config = LCMConfig(database_path=str(store.db_path), embeddings_enabled=False)
    engine = SimpleNamespace(_store=store, _dag=dag, _config=config,
                             _hermes_home=str(store.db_path.parent), current_session_id="active")
    if state == "absent":
        conn.execute("DELETE FROM metadata WHERE key='fts_stem_state'")
    elif state == "backfilling":
        boot._stem_metadata(conn, "fts_stem_state", state)
    conn.commit()
    if state == "error":
        class ErrorConnection:
            def execute(self, sql, *args):
                if "messages_fts_stem MATCH" in sql:
                    raise sqlite3.OperationalError("injected stem failure")
                return conn.execute(sql, *args)

        # Deadline workers have their own connection, so inject on their read copy.
        real_copy = tools.copy.copy

        def read_copy(obj):
            result = real_copy(obj)
            if isinstance(obj, MessageStore):
                class ReadStore(type(obj)):
                    def search(self, *a, **kw):
                        self._conn = ErrorConnection()
                        return super().search(*a, **kw)
                result.__class__ = ReadStore
            return result
        monkeypatch.setattr(tools.copy, "copy", read_copy)
    try:
        for query in ("visit", "visited", "missingword"):
            original = tools._lcm_grep_full_text_with_deadline
            with monkeypatch.context() as plain_only:
                plain_only.setattr(tools, "_lcm_grep_full_text_with_deadline", lambda args, **kw:
                                   original({**args, "_stemmed": False}, **kw))
                golden = json.loads(tools.lcm_recall({"query": query}, engine=engine))
            result = json.loads(tools.lcm_recall({"query": query}, engine=engine))
            ids = [h["store_id"] for h in result["hits"]]
            if state == "ready" and query == "visited":
                assert ids
            elif state != "ready":
                assert result["hits"] == golden["hits"]
            assert result["provenance"]["fts_index"] == ("stem" if state == "ready" else "plain")
            assert result["provenance"]["coverage"]["fts"] == "ok"
    finally:
        dag.close()


def test_4_rollback_writes_and_missing_triggers_reset(store, monkeypatch):
    from hermes_lcm.store import build_message_stem_fts_spec
    conn = partial(store)
    finish(store)
    first_marker = conn.execute("SELECT completed_at FROM lcm_migration_state WHERE step_name='fts_stem_v1'").fetchone()[0]
    conn.execute("INSERT INTO messages(session_id, role, content, timestamp) VALUES('old', 'user', 'visit', 1)")
    conn.execute("UPDATE messages SET content='visited' WHERE session_id='old'")
    conn.execute("DELETE FROM messages WHERE session_id='old'")
    conn.commit()
    consistent(conn)
    boot._drop_fts_triggers(conn, build_message_stem_fts_spec().trigger_sqls)
    conn.execute("UPDATE messages SET content='new visit' WHERE store_id=1")
    conn.commit()
    old_count = conn.execute("SELECT COUNT(*) FROM messages_fts_stem_docsize").fetchone()[0]
    reopened = MessageStore(store.db_path)
    try:
        assert boot._stem_metadata(conn, "fts_stem_state") == "backfilling"
        assert boot._stem_metadata(conn, "fts_stem_reset") == "1"
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_stem_docsize").fetchone()[0] == old_count
        monkeypatch.setattr(boot, "_check_disk_space", lambda _: False)
        finish(store)
        assert boot._stem_metadata(conn, "fts_stem_reset") == "1"
        monkeypatch.undo()
        finish(store)
        consistent(conn)
        assert conn.execute("SELECT completed_at FROM lcm_migration_state WHERE step_name='fts_stem_v1'").fetchone()[0] == first_marker
    finally:
        reopened.close()


def test_5_grep_prefix_and_exact_semantics(store):
    partial(store)
    finish(store)
    dag = SummaryDAG(store.db_path)
    engine = SimpleNamespace(_store=store, _dag=dag, current_session_id="active",
                             _config=LCMConfig(database_path=str(store.db_path), embeddings_enabled=False),
                             _hermes_home=str(store.db_path.parent))
    try:
        for query, expected in [("deploy*", True), ("organiz*", True), ("general", False)]:
            result = json.loads(tools.lcm_grep(
                {"query": query, "session_scope": "all", "_stemmed": True}, engine=engine
            ))
            assert bool(result["results"]) == expected
    finally:
        dag.close()


def test_6_open_has_no_synchronous_backfill(store):
    conn = partial(store, 20000)
    reopened = MessageStore(store.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts_stem_docsize").fetchone()[0] == 0
        assert boot._stem_metadata(conn, "fts_stem_state") == "backfilling"
    finally:
        reopened.close()


@pytest.mark.parametrize("scenario", ["backfill", "missing-trigger", "doctor"])
def test_round2_no_background_ddl(tmp_path, monkeypatch, scenario):
    from hermes_lcm.store import build_message_stem_fts_spec
    db = tmp_path / "trace.db"
    with monkeypatch.context() as setup:
        setup.setattr(boot, "dispatch_background_stem_backfill", lambda *a: None)
        obj = MessageStore(db)
        partial(obj, 2501)
        if scenario != "backfill":
            finish(obj)
            if scenario == "missing-trigger":
                obj.connection.execute("DROP TRIGGER msg_fts_stem_update")
                obj.connection.execute("UPDATE messages SET content='visited changed' WHERE store_id=1")
            else:
                boot._drop_fts_artifacts(obj.connection, build_message_stem_fts_spec())
            obj.connection.commit()
        obj.close()
    boot.join_background_integrity_scans(10)
    statements = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(lambda sql: statements.append((threading.current_thread().name, sql)))
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    if scenario == "doctor":
        # Keep the defect until repair apply; opening a store would already restore it.
        conn = sqlite3.connect(str(db))
        obj = SimpleNamespace(connection=conn, db_path=db)
        dag = SummaryDAG(db)
        monkeypatch.setattr(command, "backup_database", lambda _: {
            "ok": True, "db_path": str(db), "backup_path": "synthetic-backup", "backup_size": 0,
        })
        real_dispatch = command.dispatch_background_stem_backfill

        def dispatch_after_foreground_create(conn, spec):
            assert boot.get_existing_table_names(conn, [spec.table_name])
            assert not boot._fts_missing_triggers(conn, spec)
            real_dispatch(conn, spec)

        monkeypatch.setattr(command, "dispatch_background_stem_backfill", dispatch_after_foreground_create)
        try:
            assert "status: ok" in command._doctor_repair_apply_text(SimpleNamespace(_store=obj, _dag=dag))
        finally:
            boot.join_background_stem_backfills(10)
            dag.close()
    else:
        obj = MessageStore(db)
        conn = obj.connection
    try:
        boot.join_background_stem_backfills(10)
        background = [(name, sql) for name, sql in statements if "_run_stem_backfill" in name]
        assert background, "The trace must exercise the real backfill thread"
        assert not [(name, sql) for name, sql in background if re.match(r"\s*(CREATE|DROP|ALTER)\b", sql, re.I)]
        consistent(conn)
        if scenario != "backfill":
            assert any("VALUES('delete-all')" in sql for _, sql in background)
    finally:
        conn.close()


@pytest.mark.parametrize("readonly", ["query-only", "uri", "filesystem"])
def test_round2_readonly_pre_stem_open_and_recall(tmp_path, monkeypatch, readonly):
    db = tmp_path / "pre-stem.db"
    # The pre-stem constructor differs only by omitting this new optional ensure.
    with monkeypatch.context() as setup:
        setup.setattr("hermes_lcm.store.ensure_message_stem_fts", lambda *a: None)
        original = MessageStore(db)
    conn = original.connection
    conn.executemany(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES('synthetic', 'user', ?, 1)",
        [("visit deployment",), ("visited organization",), ("unrelated",)],
    )
    conn.commit()
    dag = SummaryDAG(db)
    engine = SimpleNamespace(_store=original, _dag=dag, current_session_id="active",
                             _config=LCMConfig(database_path=str(db), embeddings_enabled=False),
                             _hermes_home=str(tmp_path))
    args = {"query": "visit", "session_scope": "all"}
    golden = json.loads(tools.lcm_recall(args, engine=engine))
    boot.join_background_integrity_scans(10)
    dag.close()
    original.close()
    statements = []
    real_connect = sqlite3.connect

    def readonly_connect(path, *a, **kw):
        if str(path) == str(db):
            if readonly == "uri":
                path, kw["uri"] = f"{db.as_uri()}?mode=ro", True
            conn = real_connect(path, *a, **kw)
            if readonly == "query-only":
                conn.execute("PRAGMA query_only=ON")
            conn.set_trace_callback(statements.append)
            return conn
        return real_connect(path, *a, **kw)

    if readonly == "filesystem":
        db.chmod(0o444)
    monkeypatch.setattr(sqlite3, "connect", readonly_connect)
    reopened = None
    try:
        reopened = MessageStore(db)
        assert not boot.get_existing_table_names(reopened.connection, ["messages_fts_stem"])
        assert not boot._stem_metadata(reopened.connection, "fts_stem_state")
        engine._store = reopened
        result = json.loads(tools.lcm_recall(args, engine=engine))
        assert golden["hits"]
        assert result["hits"] == golden["hits"]
        assert result["provenance"]["fts_index"] == "plain"
        assert not any("INSERT OR REPLACE INTO metadata" in sql and "fts_stem_" in sql for sql in statements)
    finally:
        if reopened is not None:
            reopened.close()
        dag.close()
        db.chmod(0o644)


def test_round2_missing_history_table_skips_marker(store, caplog):
    conn = partial(store)
    conn.execute("DROP TABLE lcm_migration_state")
    conn.commit()
    finish(store)
    consistent(conn)
    assert not boot.get_existing_table_names(conn, ["lcm_migration_state"])
    assert "history" in caplog.text.lower()


def test_round2_trigger_recreation_and_flags_are_atomic(store, monkeypatch):
    from hermes_lcm.store import build_message_stem_fts_spec
    conn = partial(store)
    finish(store)
    conn.execute("DROP TRIGGER msg_fts_stem_update")
    conn.commit()
    real_metadata = boot._stem_metadata

    def crash_before_flags(conn, key, value=None):
        if key == "fts_stem_reset" and value == "1":
            raise sqlite3.OperationalError("simulated crash before reset flag")
        return real_metadata(conn, key, value)

    monkeypatch.setattr(boot, "_stem_metadata", crash_before_flags)
    with pytest.raises(sqlite3.OperationalError, match="simulated crash"):
        boot.ensure_message_stem_fts(conn, build_message_stem_fts_spec())
    conn.rollback()
    with sqlite3.connect(str(store.db_path)) as observer:
        assert boot._fts_missing_triggers(observer, build_message_stem_fts_spec())
        assert real_metadata(observer, "fts_stem_state") == "ready"
        assert not real_metadata(observer, "fts_stem_reset")


def test_7_schema_and_interim_classifier(store):
    assert boot.SCHEMA_VERSION == 5
    conn = partial(store)
    dag = SummaryDAG(store.db_path)
    dag.close()
    assert boot.classify_version_mismatch(conn) == boot.VERSION_MISMATCH_INTERIM_STAMP
    assert not any("messages_fts_stem" in str(p) or "msg_fts_stem" in str(p) for p in boot._interim_family_drops(conn))
    assert "messages_fts_stem" not in boot.REQUIRED_CORE_TABLES
    finish(store)
    consistent(conn)


def test_8_plain_ddl_and_structural_checks_unchanged(store):
    expected = json.loads((Path(__file__).parent / "fixtures/fts_plain_7b5e4b21.json").read_text())
    dag = SummaryDAG(store.db_path)
    identity = CorpusIdentity(dataset_name="synthetic", dataset_revision="1", harness_commit="1",
                              tier="small", domain="test", ingest_config_digest="1")
    trajectory = TrajectoryStore(store.db_path, identity, asset_root=store.db_path.parent)
    try:
        for name, sql in expected.items():
            assert store.connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()[0] == sql
        for spec in (build_message_fts_spec(), build_nodes_fts_spec()):
            assert spec.tokenize == ""
            assert not boot._fts_needs_rebuild_structural(store.connection, spec)
        # A plain spec still accepts a non-default tokenizer, as on origin/main.
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE messages(store_id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='store_id', tokenize='porter unicode61')")
        assert not boot._fts_needs_rebuild_structural(conn, build_message_fts_spec())
        conn.close()
    finally:
        trajectory.close()
        dag.close()


def test_round3_backfilling_open_takes_no_write_lock(store, monkeypatch):
    """#622: an open of a store whose stem backfill is still in flight (or skipped for low disk)
    repairs nothing, so it takes no write lock and re-dispatches the backfill."""
    import hermes_lcm.store as store_mod

    partial(store)
    assert boot._stem_metadata(store.connection, "fts_stem_state") == "backfilling"
    dispatched = []
    monkeypatch.setattr(boot, "dispatch_background_stem_backfill", lambda *a: dispatched.append(1))
    original = store_mod.configure_connection
    monkeypatch.setattr(store_mod, "configure_connection",
                        lambda conn: (original(conn), conn.execute("PRAGMA busy_timeout=1")))
    holder = sqlite3.connect(str(store.db_path), isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        reopened = MessageStore(store.db_path)  # raised "database is locked" before the fix
        reopened.close()
    finally:
        holder.rollback()
        holder.close()
    assert dispatched == [1]


def test_round4_trigger_repair_during_backfill_resets(store):
    """A trigger repaired while the state is already ``backfilling`` must reset the index: an
    update made while the trigger was absent leaves a stale entry the resumed backfill skips."""
    from hermes_lcm.store import build_message_stem_fts_spec
    conn = partial(store)
    conn.execute("INSERT INTO messages_fts_stem(rowid, content) SELECT store_id, content FROM messages WHERE store_id=1")
    conn.commit()
    boot._drop_fts_triggers(conn, build_message_stem_fts_spec().trigger_sqls)
    conn.execute("UPDATE messages SET content='zebra crossing' WHERE store_id=1")
    conn.commit()
    reopened = MessageStore(store.db_path)
    try:
        assert boot._stem_metadata(conn, "fts_stem_reset") == "1"  # unset before the fix
        finish(store)
        consistent(conn)
        match = "SELECT rowid FROM messages_fts_stem WHERE messages_fts_stem MATCH ? ORDER BY rowid"
        assert conn.execute(match, ("zebra",)).fetchall() == [(1,)]
        assert conn.execute(match, ("deployment",)).fetchall() == [(2,), (3,)]
    finally:
        reopened.close()


def test_round4_readiness_flip_during_query_falls_back_to_plain(store):
    """A repair or reset that clears the stem index between the readiness read and the query must
    not let the partial index answer as ``stem``; the plain index answers instead."""
    conn = partial(store)
    finish(store)
    expected = [hit["store_id"] for hit in store.search("visit", stemmed=False)]
    assert expected

    class FlipOnStemQuery:
        def __init__(self, inner):
            self.inner = inner

        def execute(self, sql, *args):
            if "messages_fts_stem MATCH" in sql:
                other = sqlite3.connect(str(store.db_path))
                other.executemany("INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                                  [("fts_stem_state", "backfilling"), ("fts_stem_reset", "1")])
                other.execute("INSERT INTO messages_fts_stem(messages_fts_stem) VALUES('delete-all')")
                other.commit()
                other.close()
            return self.inner.execute(sql, *args)

    store._conn = FlipOnStemQuery(conn)
    try:
        hits = store.search("visit", stemmed=True)
    finally:
        store._conn = conn
    assert [hit["store_id"] for hit in hits] == expected  # [] answered as "stem" before the fix
    assert store._last_fts_index == "plain"


def test_round4_backfill_pages_by_key_not_id_span(store, monkeypatch):
    """Sparse store_ids cost no empty batches: one page covers three rows 50M ids apart."""
    conn = partial(store, 2)
    conn.execute("INSERT INTO messages(store_id, session_id, role, content, timestamp) "
                 "VALUES(50000000, 'sparse', 'tool', 'visit deployment', 1)")
    conn.commit()
    batches = []

    def count_batch(_):
        batches.append(1)
        if len(batches) > 10:
            raise RuntimeError("backfill walked the id span")

    monkeypatch.setattr(boot.time, "sleep", count_batch)
    finish(store)
    consistent(conn)
    assert len(batches) == 1
