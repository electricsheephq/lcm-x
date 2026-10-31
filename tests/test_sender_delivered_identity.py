"""S0 cannot change any delivered or indexed message byte."""
from copy import deepcopy
import json

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
    monkeypatch.setattr("hermes_lcm.store.time.time", lambda: 1700000000.0)
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
