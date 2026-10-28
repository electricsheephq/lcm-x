"""#1197: short turns and resumable revision-1 gaps on synthetic stores."""

from types import SimpleNamespace

import pytest

import hermes_lcm.chunking as chunking
import hermes_lcm.command as command
from hermes_lcm.config import LCMConfig
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore


FACT = "I adopted a grey cat named Pixel"
EMPTY = "\t\n"  # SQLite trim leaves this eligible; the chunker normalises it to empty.


class Provider:
    provider_id = "ollama"
    model_id = "model-a"
    last_skipped_documents = []

    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "short-turns.db"
    config = LCMConfig(
        database_path=str(db), embeddings_enabled=True,
        embedding_provider="ollama", embedding_model="model-a",
    )
    vectors = VectorStore(db, config=config)
    vectors.ensure_chunk_schema()
    identity_hash = vectors.register_profile("model-a", "ollama", 2, task="chunk")
    identity = EmbeddingIdentity.canonical("ollama", "model-a", "", 2, "float32", "little", "chunk")
    assert identity.identity_hash == identity_hash
    vectors.connection.execute(
        "CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT, "
        "session_id TEXT NOT NULL DEFAULT 'synthetic', source TEXT DEFAULT '', "
        "timestamp REAL NOT NULL DEFAULT 0)"
    )
    provider = Provider()
    monkeypatch.setattr(command, "resolve_provider", lambda *_a, **_k: provider)
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=db))
    yield engine, vectors, identity, provider
    vectors.close()


def seed_old(store):
    _, vectors, identity, _ = store
    rows = [
        (1, "user", "history fact"),
        (2, "user", "long low boundary. " * 60),
        (3, "user", FACT),
        (4, "assistant", "Pixel is grey"),
        (5, "user", EMPTY),
        (6, "assistant", "I work Fridays"),
        (7, "tool", "tool stays unchanged"),
        (8, "assistant", "long high boundary. " * 60),
        (9, "user", "new fact"),
    ]
    vectors.connection.executemany("INSERT INTO messages(store_id, role, content) VALUES(?, ?, ?)", rows)
    # Simulate revision 1: only the two long turns have chunks; no revision stamp.
    for store_id in (2, 8):
        role, content = rows[store_id - 1][1:]
        for chunk in chunking.chunk_message(store_id, role, content, policy="full"):
            vectors.record_chunk_embedding(
                chunk.chunk_id, "model-a", [1.0, 0.0], store_id=store_id,
                chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                char_end=chunk.char_end, token_estimate=chunk.token_estimate,
                identity=identity,
            )


def revision(store):
    _, vectors, identity, _ = store
    row = vectors.connection.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (f"chunk_policy_revision:{identity.identity_hash}",),
    ).fetchone()
    return None if row is None else int(row[0])


def run(store, *, limit=100, incremental=True, policy="conversational", retry=False):
    engine, _, _, _ = store
    result = command._chunk_backfill_run(
        engine._config, engine._store.db_path, apply=True, limit=limit,
        retry_uncertain=retry, policy=policy, incremental=incremental,
    )
    assert isinstance(result, dict), result
    assert result["error"] is None, result
    return result


@pytest.mark.parametrize("policy", ["conversational", "heads"])
def test_t1_short_fact_has_exact_span_and_empty_and_tool_rules(policy):
    chunks = chunking.chunk_message(1, "user", FACT, policy=policy)
    assert len(chunks) == 1
    assert (chunks[0].text, chunks[0].char_start, chunks[0].char_end) == (FACT, 0, len(FACT))
    assert chunking.chunk_message(2, "user", EMPTY, policy=policy) == []
    assert chunking.chunk_message(3, "tool", FACT, policy="conversational") == []
    assert len(chunking.chunk_message(3, "tool", FACT, policy="heads")) == 1
    assert chunking.CHUNK_POLICY_REVISION == 2


@pytest.mark.parametrize("policy", ["conversational", "heads"])
def test_t2_bounded_gapfill_finishes_without_duplicates(store, monkeypatch, policy):
    seed_old(store)
    _, vectors, identity, provider = store
    # Another identity's vector must not hide a gap in this identity.
    vectors.connection.execute(
        "INSERT INTO lcm_chunk_meta(chunk_id, identity_hash, store_id) VALUES('3:0', 'other', 3)"
    )
    monkeypatch.setattr(command, "_CHUNK_HISTORY_SCAN_MESSAGES", 2)
    for _ in range(12):
        run(store, limit=1, policy=policy)
        if revision(store) == 2 and not command._chunk_pending_rows(
            vectors.connection, identity.identity_hash, policy, 1, incremental=True
        )[0]:
            break
    assert revision(store) == 2
    ids = [row[0] for row in vectors.connection.execute(
        "SELECT store_id FROM lcm_chunk_meta WHERE identity_hash = ? ORDER BY store_id",
        (identity.identity_hash,),
    )]
    assert ids == ([1, 2, 3, 4, 6, 8, 9] if policy == "conversational" else [1, 2, 3, 4, 6, 7, 8, 9])
    assert len(provider.texts) == len(set(provider.texts)) == len(ids) - 2
    # New rows, gaps newest-first, then history. The empty row cannot hide row 4.
    expected_first = ["new fact", "I work Fridays" if policy == "conversational" else "tool stays unchanged"]
    assert provider.texts[:2] == expected_first
    before = list(provider.texts)
    run(store, policy=policy)
    assert provider.texts == before
    rebuilt = command._rebuild_chunk_document(vectors.connection, "3:0", policy)
    assert rebuilt[0] == FACT and rebuilt[2:] == (0, len(FACT))
    assert command._rebuild_chunk_document(vectors.connection, "2:0", policy) is not None


def test_t3_deadline_after_one_batch_preserves_all_unprocessed_gaps(store, monkeypatch):
    seed_old(store)
    _, vectors, identity, provider = store
    # Stay within the old cursor range to isolate the fill's deadline path.
    vectors.connection.execute("DELETE FROM messages WHERE store_id IN (1, 9)")
    monkeypatch.setattr(command, "_embedding_backfill_batch_size", lambda *_: 1)
    monkeypatch.setattr(command, "_embedding_backfill_budget_s", lambda: 1)
    monotonic = command.time.monotonic
    monkeypatch.setattr(command.time, "monotonic", lambda: monotonic() + (10 if provider.texts else 0))
    first = run(store)
    assert first["embedded"] == 1 and first["stop_reason"] == "op_budget_exhausted"
    assert revision(store) != 2
    monkeypatch.setattr(command.time, "monotonic", monotonic)
    for _ in range(4):
        run(store)
    assert revision(store) == 2
    assert set(provider.texts) == {FACT, "Pixel is grey", "I work Fridays"}
    assert len(provider.texts) == 3
    assert command._chunk_pending_rows(
        vectors.connection, identity.identity_hash, "conversational", 1, incremental=True
    )[0] == 0


def test_t4_backlog_status_counts_gaps_without_writing(store):
    seed_old(store)
    engine, vectors, identity, _ = store
    vectors.connection.execute("DELETE FROM messages WHERE store_id IN (1, 9)")
    before = vectors.connection.execute("SELECT * FROM metadata ORDER BY key").fetchall()
    status = "\n".join(command._embedding_maintenance_status_lines(engine))
    assert "embedding_backlog_chunks: 3 (gap-fill)" in status
    assert before == vectors.connection.execute("SELECT * FROM metadata ORDER BY key").fetchall()
    for _ in range(4):
        run(store)
    assert command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, "conversational") == "0"
    assert "embedding_backlog_chunks: 0" in "\n".join(command._embedding_maintenance_status_lines(engine))
    assert revision(store) == 2


def test_t5_fresh_identity_records_current_revision_without_fill(store):
    _, vectors, identity, provider = store
    vectors.connection.execute("INSERT INTO messages(store_id, role, content) VALUES(1, 'user', ?)", (FACT,))
    assert command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, "conversational") == "1"
    assert run(store)["embedded"] == 1
    assert revision(store) == 2
    assert provider.texts == [FACT]
    assert not vectors.connection.execute(
        "SELECT 1 FROM metadata WHERE key LIKE 'chunk_gapfill_skip_below:%'"
    ).fetchall()


@pytest.mark.parametrize("policy", ["conversational", "full"])
def test_t2_manual_backfill_and_full_policy_stamp_revision(store, policy):
    seed_old(store)
    assert run(store, incremental=False, policy=policy)["embedded"] == (6 if policy == "full" else 5)
    assert revision(store) == 2


@pytest.mark.parametrize("policy", ["conversational", "full"])
def test_limited_manual_backfill_preserves_automatic_gapfill(store, policy):
    seed_old(store)
    _, vectors, identity, provider = store
    assert command._chunk_policy_revision(vectors.connection, identity.identity_hash) == 1
    first = run(store, incremental=False, policy=policy, limit=1)
    assert first["pending"] > first["embedded"] == 1
    assert first["remaining"] > 0
    assert revision(store) != chunking.CHUNK_POLICY_REVISION

    for _ in range(12):
        if run(store)["embedded"] == 0 and revision(store) == chunking.CHUNK_POLICY_REVISION:
            break
    else:
        pytest.fail("automatic conversational gap-fill did not finish")
    for store_id, text in ((3, FACT), (4, "Pixel is grey"), (6, "I work Fridays")):
        assert provider.texts.count(text) == 1
        assert vectors.connection.execute(
            "SELECT COUNT(*) FROM lcm_chunk_meta WHERE store_id = ? AND identity_hash = ?",
            (store_id, identity.identity_hash),
        ).fetchone()[0] == 1
    assert revision(store) == chunking.CHUNK_POLICY_REVISION
    before = list(provider.texts)
    assert run(store)["embedded"] == 0
    assert provider.texts == before


def test_t2_full_policy_never_fills_old_range(store):
    seed_old(store)
    _, vectors, identity, provider = store
    run(store, policy="full")
    assert revision(store) is None  # Full skips the old gaps, so it cannot certify their revision.
    assert provider.texts == ["new fact", "history fact"]
    assert command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, "full") == "0"
    assert command._chunk_backlog_estimate(
        vectors.connection, identity.identity_hash, "conversational"
    ) == "3 (gap-fill)"


def test_full_pass_preserves_later_conversational_gapfill(store):
    seed_old(store)
    _, vectors, identity, provider = store
    assert command._chunk_policy_revision(vectors.connection, identity.identity_hash) == 1
    run(store, policy="full")
    for _ in range(12):
        if run(store)["embedded"] == 0 and revision(store) == 2:
            break
    assert provider.texts.count(FACT) == 1
    assert vectors.connection.execute(
        "SELECT COUNT(*) FROM lcm_chunk_meta WHERE chunk_id = '3:0' AND identity_hash = ?",
        (identity.identity_hash,),
    ).fetchone()[0] == 1
    assert revision(store) == 2
    before = list(provider.texts)
    assert run(store)["embedded"] == 0
    assert provider.texts == before


def test_t3_uncertain_gap_is_not_automatically_replayed(store):
    seed_old(store)
    _, vectors, identity, provider = store
    command._ensure_inflight_table(vectors.connection)
    vectors.connection.execute(
        "INSERT INTO lcm_embedding_backfill_inflight(embedded_id, identity_hash, state) "
        "VALUES('6:0', ?, 'uncertain')", (identity.identity_hash,),
    )
    for _ in range(3):
        run(store)
    assert "I work Fridays" not in provider.texts
    assert revision(store) != 2
    assert run(store, retry=True)["embedded"] == 1
    for _ in range(2):
        run(store)  # first advances past the empty row; then the query is empty
    assert revision(store) == 2
    assert provider.texts.count("I work Fridays") == 1


def test_revision_gapfill_and_partial_boundaries_both_finish(store):
    _, vectors, identity, provider = store
    rows = [
        (1, "user", "Low boundary factual sentence. " * 400),
        (2, "user", FACT),
        (3, "assistant", "High boundary factual sentence. " * 400),
        (4, "user", "new fact"),
    ]
    vectors.connection.executemany("INSERT INTO messages(store_id, role, content) VALUES(?, ?, ?)", rows)
    expected = [chunk for row in rows for chunk in chunking.chunk_message(*row, policy="conversational")]
    # Revision 1 plus a provider-boundary stop: each frontier has only chunk 0.
    for row in (rows[0], rows[2]):
        chunks = chunking.chunk_message(*row, policy="conversational")
        assert len(chunks) > 1
        chunk = chunks[0]
        vectors.record_chunk_embedding(
            chunk.chunk_id, "model-a", [1.0, 0.0], store_id=chunk.store_id,
            chunk_index=chunk.chunk_index, char_start=chunk.char_start,
            char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=identity,
        )
    for _ in range(len(expected) + 2):
        run(store, limit=1)
        if revision(store) == 2:
            break
    assert revision(store) == 2  # Requires record_progress to reach the frontier.
    actual = {row[0] for row in vectors.connection.execute(
        "SELECT chunk_id FROM lcm_chunk_meta WHERE identity_hash = ?", (identity.identity_hash,),
    )}
    assert actual == {chunk.chunk_id for chunk in expected}  # Requires both boundary rows.
    assert len(provider.texts) == len(expected) - 2  # Never replay committed boundary chunks.


@pytest.mark.parametrize("boundary", ["high", "low"])
def test_revision_gapfill_with_inflight_only_boundary(store, boundary):
    _, vectors, identity, provider = store
    boundary_id = 3 if boundary == "high" else 1
    content = "".join(
        f"Boundary fact {index} records the grey cat beside the window. "
        for index in range(240)
    )
    rows = [
        (1, "user", content if boundary_id == 1 else "Old low boundary. " * 60),
        (2, "user", FACT),
        (3, "assistant", content if boundary_id == 3 else "Old high boundary. " * 60),
    ]
    vectors.connection.executemany("INSERT INTO messages(store_id, role, content) VALUES(?, ?, ?)", rows)
    chunks = [chunk for row in rows for chunk in chunking.chunk_message(*row, policy="conversational")]
    boundary_chunks = [chunk for chunk in chunks if chunk.store_id == boundary_id]
    assert len(boundary_chunks) >= 3
    for chunk in chunks:
        if chunk.store_id not in (2, boundary_id):
            vectors.record_chunk_embedding(
                chunk.chunk_id, "model-a", [1.0, 0.0], store_id=chunk.store_id,
                chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=identity,
            )
    uncertain_id = f"{boundary_id}:0"
    command._ensure_inflight_table(vectors.connection)
    vectors.connection.execute(
        "INSERT INTO lcm_embedding_backfill_inflight(embedded_id, identity_hash, state) "
        "VALUES(?, ?, 'uncertain')", (uncertain_id, identity.identity_hash),
    )
    assert revision(store) is None
    assert command._chunk_policy_revision(vectors.connection, identity.identity_hash) == 1
    _, documents, _ = command._chunk_pending_rows(
        vectors.connection, identity.identity_hash, "conversational", 100, incremental=True,
    )
    ids = [document[0] for document in documents]
    assert len(ids) == len(set(ids)), ids
    assert uncertain_id not in ids
    run(store)  # The real automatic pass must reserve its candidates without error.
    for _ in range(len(chunks) + 2):
        if revision(store) == chunking.CHUNK_POLICY_REVISION:
            break
        run(store)
    assert revision(store) == chunking.CHUNK_POLICY_REVISION
    actual = {row[0] for row in vectors.connection.execute(
        "SELECT chunk_id FROM lcm_chunk_meta WHERE identity_hash = ?", (identity.identity_hash,),
    )}
    assert actual == {chunk.chunk_id for chunk in chunks} - {uncertain_id}
    expected_texts = [
        chunk.text for chunk in chunks
        if chunk.store_id in (2, boundary_id) and chunk.chunk_id != uncertain_id
    ]
    assert sorted(provider.texts) == sorted(expected_texts)
    assert boundary_chunks[0].text not in provider.texts
    inflight = vectors.connection.execute(
        "SELECT embedded_id, state FROM lcm_embedding_backfill_inflight WHERE identity_hash = ?",
        (identity.identity_hash,),
    ).fetchall()
    assert [tuple(row) for row in inflight] == [(uncertain_id, "uncertain")]
