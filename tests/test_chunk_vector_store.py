from __future__ import annotations

import sqlite3

import pytest

import hermes_lcm.vector_store as vector_store_module
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

MODEL = "voyage-context-4"
PROVIDER = "voyage"
DIM = 4


def _seed_messages(db_path, rows):
    """Create the messages columns the chunk KNN filters need, then insert rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            store_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO messages(store_id, session_id, source, role, content, timestamp) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _chunk_identity():
    """Return the canonical float32 identity shared by the chunk fixtures."""
    return EmbeddingIdentity.canonical(
        PROVIDER, MODEL, "", DIM, "float32", "little", "chunk"
    )


@pytest.fixture
def store(tmp_path):
    """Yield an isolated vector store with synthetic messages and a chunk profile."""
    db_path = tmp_path / "lcm.db"
    _seed_messages(
        db_path,
        [
            (10, "sess-a", "history", "user", "first message", 100.0),
            (11, "sess-a", "history", "assistant", "second message", 200.0),
            (12, "sess-b", "other", "tool", "tool output", 300.0),
        ],
    )
    vs = VectorStore(db_path)
    vs.register_profile(MODEL, PROVIDER, DIM, task="chunk")
    yield vs
    vs.close()


def _write(store, chunk_id, store_id, chunk_index, vec):
    """Record a synthetic vector with consistent chunk provenance."""
    store.record_chunk_embedding(
        chunk_id,
        MODEL,
        vec,
        store_id=store_id,
        chunk_index=chunk_index,
        char_start=0,
        char_end=10,
        token_estimate=5,
        identity=_chunk_identity(),
    )


class TestChunkWriteAndKnn:
    @pytest.mark.parametrize("loader", ["python", "numpy"])
    @pytest.mark.parametrize(
        ("stored_value", "storage_class"),
        [
            (DIM * 4, "integer"),
            (float(DIM * 4) + 0.5, "real"),
            ("x" * (DIM * 4), "text"),
            (b"short", "blob"),
        ],
    )
    def test_chunk_loaders_skip_malformed_sqlite_values(
        self, store, loader, stored_value, storage_class
    ):
        """Non-BLOB storage must not be coerced into valid float32 vectors."""
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "11:0", 11, 0, [0.0, 1.0, 0.0, 0.0])
        identity = _chunk_identity().identity_hash
        store.connection.execute(
            "UPDATE lcm_chunk_vectors SET vec = ? WHERE chunk_id = ? AND identity_hash = ?",
            (stored_value, "11:0", identity),
        )
        store.connection.commit()
        actual_type = store.connection.execute(
            "SELECT typeof(vec) FROM lcm_chunk_vectors WHERE chunk_id = ?",
            ("11:0",),
        ).fetchone()[0]
        assert actual_type == storage_class

        if loader == "numpy":
            numpy = pytest.importorskip("numpy")
            rowids, ids, kinds, values = store._load_chunk_matrix(
                numpy, identity, DIM, ["10:0", "11:0"], "float32"
            )
            assert values.shape == (1, DIM)
            values = values.tolist()
        else:
            rowids, ids, kinds, values = store._load_chunk_vectors_for_ids(
                identity, DIM, ["10:0", "11:0"], "float32"
            )
        assert rowids == [1]
        assert ids == ["10:0"]
        assert kinds == ["chunk"]
        assert values == [[1.0, 0.0, 0.0, 0.0]]

    def test_numpy_loader_reads_valid_float32_blobs_without_python_decode(
        self, store, monkeypatch
    ):
        """The NumPy path keeps float32 BLOBs binary until matrix construction."""
        numpy = pytest.importorskip("numpy")
        _write(store, "10:0", 10, 0, [1.0, -2.0, 3.0, -4.0])
        _write(store, "11:0", 11, 0, [4.0, 3.0, 2.0, 1.0])
        _write(store, "12:0", 12, 0, [0.0, 1.0, 0.0, 0.0])
        identity = _chunk_identity().identity_hash

        # Match the legacy decoder's rejection of malformed storage and archive
        # filtering while proving that valid float32 vectors do not visit it.
        store.connection.execute(
            "UPDATE lcm_chunk_vectors SET vec = ? WHERE chunk_id = ? AND identity_hash = ?",
            (b"short", "11:0", identity),
        )
        store.connection.commit()
        assert store.archive_chunks_for_messages([12]) == 1

        def unexpected_decode(*_args, **_kwargs):
            """Fail when float32 loading regresses to Python object expansion."""
            pytest.fail("valid float32 BLOBs should not be decoded through Python floats")

        monkeypatch.setattr(store, "_load_chunk_vectors_for_ids", unexpected_decode)
        rowids, chunk_ids, kinds, matrix = store._load_chunk_matrix(
            numpy, identity, DIM, ["10:0", "11:0", "12:0"], "float32"
        )

        assert rowids == [1]
        assert chunk_ids == ["10:0"]
        assert kinds == ["chunk"]
        assert matrix.dtype == numpy.float32
        numpy.testing.assert_allclose(
            matrix,
            numpy.array(
                [[1.0, -2.0, 3.0, -4.0]], dtype=numpy.float32
            ) / numpy.sqrt(numpy.float32(30.0)),
            rtol=0.0,
            atol=1e-7,
        )

    def test_numpy_loader_keeps_int8_on_the_existing_decoder_path(self, store, monkeypatch):
        """Keep quantized vectors on the established decoder and preserve metadata."""
        numpy = pytest.importorskip("numpy")
        identity = EmbeddingIdentity.canonical(
            PROVIDER, MODEL, "", DIM, "int8", "little", "chunk"
        )
        store.register_profile(MODEL, PROVIDER, DIM, dtype="int8", task="chunk")
        store.record_chunk_embedding(
            "10:int8",
            MODEL,
            [0.5, -0.25, 0.125, -0.0625],
            store_id=10,
            chunk_index=1,
            char_start=0,
            char_end=10,
            token_estimate=5,
            identity=identity,
        )
        original = store._load_chunk_vectors_for_ids
        calls = []

        def observed_decode(*args, **kwargs):
            """Record the chosen dtype while executing the real legacy decoder."""
            calls.append(args[3])
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "_load_chunk_vectors_for_ids", observed_decode)
        rowids, chunk_ids, kinds, matrix = store._load_chunk_matrix(
            numpy, identity.identity_hash, DIM, ["10:int8"], "int8"
        )

        assert calls == ["int8"]
        assert rowids == [1]
        assert chunk_ids == ["10:int8"]
        assert kinds == ["chunk"]
        assert matrix.shape == (1, DIM)

    def test_write_and_retrieve(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "11:0", 11, 0, [0.0, 1.0, 0.0, 0.0])
        result = store.knn_chunks([1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER)
        assert result.coverage == "full"
        assert result[0][0] == "10:0"
        assert {row[0] for row in result} == {"10:0", "11:0"}
        assert all(row[2] == "chunk" for row in result)

    def test_unbackfilled_identity_returns_none(self, store):
        result = store.knn_chunks([1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER)
        assert result.coverage == "none"

    def test_session_filter(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "12:0", 12, 0, [1.0, 0.0, 0.0, 0.0])
        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER,
            conversation_ids=["sess-b"],
        )
        assert {row[0] for row in result} == {"12:0"}

    def test_source_filter(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "12:0", 12, 0, [1.0, 0.0, 0.0, 0.0])
        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER, source="other",
        )
        assert {row[0] for row in result} == {"12:0"}

    def test_recency_window(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "12:0", 12, 0, [1.0, 0.0, 0.0, 0.0])
        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER, since=250.0,
        )
        assert {row[0] for row in result} == {"12:0"}

    def test_bounded_coverage(self, tmp_path):
        db_path = tmp_path / "lcm.db"
        _seed_messages(
            db_path,
            [(i, "s", "history", "user", "m", float(i)) for i in range(5)],
        )
        vs = VectorStore(db_path, bounded_scan_rows=2)
        vs.register_profile(MODEL, PROVIDER, DIM, task="chunk")
        try:
            for i in range(5):
                _write(vs, f"{i}:0", i, 0, [1.0, 0.0, 0.0, 0.0])
            result = vs.knn_chunks([1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER)
            assert result.coverage == "bounded"
            assert len(result) <= 2
        finally:
            vs.close()

    def test_full_scan_reaches_best_chunk_outside_recency_bound(self, tmp_path):
        """Salvaged from PR #191 (upstream hermes-lcm#461, @stephenschoettler).

        The PR's ``scanned == total`` assertion is a design fork against main's
        KNNResult contract (both stay None outside bounded coverage) and is
        deliberately not asserted here.
        """
        db_path = tmp_path / "lcm.db"
        _seed_messages(
            db_path,
            [
                (1, "s", "history", "user", "old exact match", 1.0),
                (2, "s", "history", "user", "new distractor", 2.0),
            ],
        )
        vs = VectorStore(db_path, bounded_scan_rows=1)
        vs.register_profile(MODEL, PROVIDER, DIM, task="chunk")
        try:
            _write(vs, "1:0", 1, 0, [1.0, 0.0, 0.0, 0.0])
            _write(vs, "2:0", 2, 0, [0.0, 1.0, 0.0, 0.0])

            bounded = vs.knn_chunks(
                [1.0, 0.0, 0.0, 0.0], k=1, model=MODEL, provider=PROVIDER
            )
            exhaustive = vs.knn_chunks(
                [1.0, 0.0, 0.0, 0.0],
                k=1,
                model=MODEL,
                provider=PROVIDER,
                full_scan=True,
            )

            assert bounded.coverage == "bounded"
            assert [row[0] for row in bounded] == ["2:0"]
            assert exhaustive.coverage == "full"
            assert [row[0] for row in exhaustive] == ["1:0"]
        finally:
            vs.close()

    def test_numpy_absent_reports_full_when_scan_covers_corpus(
        self, store, monkeypatch
    ):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "11:0", 11, 0, [0.0, 1.0, 0.0, 0.0])

        def unavailable():
            raise ImportError("numpy not installed")

        monkeypatch.setattr(vector_store_module, "_load_numpy", unavailable)
        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER
        )

        assert result.coverage == "full"
        assert {row[0] for row in result} == {"10:0", "11:0"}


class TestArchiveOnPurge:
    def test_archive_drops_from_knn(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "11:0", 11, 0, [0.0, 1.0, 0.0, 0.0])
        archived = store.archive_chunks_for_messages([10])
        assert archived == 1
        result = store.knn_chunks([1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER)
        assert {row[0] for row in result} == {"11:0"}

    def test_archive_noop_without_schema(self, tmp_path):
        db_path = tmp_path / "lcm.db"
        _seed_messages(db_path, [(1, "s", "", "user", "m", 1.0)])
        vs = VectorStore(db_path)
        try:
            assert vs.archive_chunks_for_messages([1]) == 0
        finally:
            vs.close()

    def test_archive_batch_on_connection(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        conn = store.connection
        archived = VectorStore.archive_chunks_for_messages_on_connection(conn, [10])
        assert archived == 1


class TestCoexistence:
    def test_summary_and_chunk_profiles_coexist(self, store):
        # Register a summary profile alongside the existing chunk profile.
        store.register_profile("summary-model", PROVIDER, DIM, task="summary")
        chunk = store._current_chunk_profile()
        summary = store._current_profile()
        assert chunk is not None and summary is not None
        assert chunk["task"] == "chunk"
        assert summary["task"] == "summary"
        assert chunk["identity_hash"] != summary["identity_hash"]
        # Both remain active.
        assert int(chunk["active"]) == 1
        assert int(summary["active"]) == 1
