"""S0 cannot change any delivered or indexed message byte."""
from copy import deepcopy
import json
from threading import Event, current_thread, main_thread
import time
from types import SimpleNamespace

import pytest

from hermes_lcm import command, tools
from hermes_lcm.chunking import chunk_message
from tests.compat.storage_fixtures import install_fakes
from tests.harness import engine_context
from tests.test_sender_capture import begin, expected, message, senders
from tests.test_sender_capture import clean_capture_registry  # noqa: F401


def delivered(engine, view, provider):
    conn = engine._store.connection
    shadows = sorted(r[1] for r in conn.execute("PRAGMA table_list")
                     if r[2] == "shadow" and r[1].startswith("messages_fts"))
    fts = {name: sorted(conn.execute(f'SELECT * FROM "{name}"').fetchall(), key=repr) for name in shadows}
    matches = list(conn.execute("SELECT rowid, bm25(messages_fts), snippet(messages_fts, 0, '[', ']', '...', 12) "
                                "FROM messages_fts WHERE messages_fts MATCH 'synthetic' ORDER BY rowid"))
    rows = engine._store.get_session_messages("S")
    chunks = [chunk for row in rows for chunk in chunk_message(row["store_id"], row["role"], row["content"])]
    # Finish bind/ingest passes before warmup registers profiles; otherwise
    # they can race our explicit backfill and consume its recorded documents.
    assert engine.drain_embedding_maintenance(timeout=10)
    assert "status: ready" in command.handle_lcm_command("embed warmup", engine)
    provider.documents.clear()
    result = command.handle_lcm_command("embed backfill --corpus chunks --apply", engine)
    assert "status: complete" in result
    docs = list(provider.documents)
    search = tools.lcm_grep({"query": "synthetic", "session_id": "S", "session_scope": "session"}, engine=engine)
    recall = tools.lcm_recall({"query": "synthetic", "exclude_current_session": False}, engine=engine)
    assert json.loads(search).get("results")
    assert json.loads(recall).get("hits")
    assembled = engine._assemble_context(None, deepcopy(view), include_lcm_note=False, persist=False)
    return fts, matches, chunks, docs, rows, search, recall, assembled


@pytest.mark.parametrize("anchor", ["0", "1"])
@pytest.mark.parametrize("capture", ["0", "1"])
def test_t9_three_store_delivered_identity(tmp_path, monkeypatch, anchor, capture):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", anchor)
    monkeypatch.setenv("LCM_SENDER_CAPTURE", capture)
    provider = install_fakes(monkeypatch)
    # Hold write timestamps identical; sender writes must not alter either clock.
    # Pin only store.py's wall clock: its other time functions stay real, and
    # other modules keep the real time.time (#2266).
    clock = {name: getattr(time, name) for name in dir(time) if not name.startswith("_")}
    clock["time"] = lambda: 1700000000.0
    monkeypatch.setattr("hermes_lcm.store.time", SimpleNamespace(**clock))
    # lcm_recall scores recency from tools.py's wall clock; pin it the same way.
    monkeypatch.setattr("hermes_lcm.tools.time", SimpleNamespace(**clock))
    outcomes = []
    for shape in ("stamped", "null", "pre-s0"):
        view = [message(), dict(role="assistant", content="synthetic answer", timestamp=101.0)]
        with engine_context(tmp_path / shape, window=128000, conversation_id="conv",
                            embeddings_enabled=True, embedding_provider="ollama",
                            embedding_model=provider.model_id) as e:
            assert {r[1] for r in e._store.connection.execute("PRAGMA table_info(messages)")} >= {"sender_id", "sender_source"}
            if shape == "stamped":
                begin(e, view[:1])
            elif shape == "pre-s0":
                # Synthetic DB only: represent the supported old physical schema.
                e._store.connection.execute("ALTER TABLE messages DROP COLUMN sender_source")
                e._store.connection.execute("ALTER TABLE messages DROP COLUMN sender_id")
                e._store.connection.commit()
            e.ingest(view)
            if shape == "stamped":
                wanted = [expected(), (None, None)] if capture == "1" else [(None, None)] * 2
                if senders(e) != wanted:
                    raise AssertionError("capture switch did not control stamping")
            outcomes.append(delivered(e, view, provider))
    assert outcomes[0] == outcomes[1] == outcomes[2]


def test_t9_waits_for_background_embedding_lease(tmp_path, monkeypatch):
    """Force #2266's worker-release ordering without relying on thread timing."""
    held, resume = Event(), Event()
    release = command._BackfillLease.release
    handle = command.handle_lcm_command
    from hermes_lcm.engine import LCMEngine

    drain = LCMEngine.drain_embedding_maintenance
    released = []

    def delayed_release(lease):
        if current_thread() is not main_thread() and not held.is_set():
            held.set()
            assert resume.wait(10), "foreground never reached drain or backfill"
            release(lease)
            assert lease.conn.execute(
                "SELECT value FROM metadata WHERE key=?",
                (command._EMBEDDING_BACKFILL_CLAIM_KEY,),
            ).fetchone() is None, "background worker leaked its lease"
            released.append(True)
        else:
            release(lease)

    def controlled_command(text, engine):
        try:
            return handle(text, engine)
        finally:
            if text.startswith("embed backfill"):
                # The red control must also let shutdown finish.
                resume.set()

    def controlled_drain(engine, timeout=None):
        resume.set()
        return drain(engine, timeout=timeout)

    provider = install_fakes(monkeypatch)
    with engine_context(tmp_path, window=128000, embeddings_enabled=True,
                        embedding_provider="ollama", embedding_model=provider.model_id) as engine:
        assert "status: ready" in handle("embed warmup", engine)
        assert engine.drain_embedding_maintenance(timeout=10)
        monkeypatch.setattr(command._BackfillLease, "release", delayed_release)
        monkeypatch.setattr(command, "handle_lcm_command", controlled_command)
        monkeypatch.setattr(LCMEngine, "drain_embedding_maintenance", controlled_drain)
        try:
            view = [message(), dict(role="assistant", content="synthetic answer", timestamp=101.0)]
            engine.ingest(view)
            # Re-enqueue the bind's pass after profiles exist, forcing the
            # ordering a slow worker can encounter in the original test.
            engine._schedule_embedding_maintenance()
            assert held.wait(10), "background worker did not reach lease release"
            delivered(engine, view, provider)
        finally:
            resume.set()
            assert drain(engine, timeout=10)
        assert released == [True]
