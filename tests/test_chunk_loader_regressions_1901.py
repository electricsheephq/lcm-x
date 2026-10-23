"""Selected-batch regressions for #1901; reference the pre-#1893 index order."""
from __future__ import annotations

import random
import sqlite3

import pytest

from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

MODEL = "synthetic-loader-regression"
PROVIDER = "synthetic"


def _seed(db_path, count):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, "
            "source TEXT, role TEXT, content TEXT, timestamp REAL)"
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, 'keep', 'history', 'user', 'synthetic', ?)",
            [(i, float(i)) for i in range(count)],
        )


def _reference(numpy, rows, batches, dim, query):
    """Score each selected batch in the old metadata index order, without loaders."""
    scored = []
    for batch in batches:
        selected = set(batch)
        ordered = [row for row in rows if row["chunk_id"] in selected]
        matrix = numpy.frombuffer(
            b"".join(row["vec"] for row in ordered), dtype="<f4"
        ).reshape(len(ordered), dim)
        scored.extend(
            (row["rowid"], row["chunk_id"], float(score))
            for row, score in zip(ordered, matrix @ query)
        )
    return scored


def _assert_scores(numpy, actual, reference, blobs, query):
    expected = {cid: score for _, cid, score in reference}
    # Dot-product rounding scales with vector norms, not a near-zero result.
    # Use four float32 ulp at unit score scale, scaled by both input norms.
    scale = 4 * float(numpy.spacing(numpy.float32(1.0))) * float(numpy.linalg.norm(query))
    for cid, score, kind in actual:
        assert kind == "chunk"
        base = numpy.float32(expected[cid])
        vector = numpy.frombuffer(blobs[cid], dtype="<f4")
        tolerance = scale * float(numpy.linalg.norm(vector))
        assert abs(float(score) - float(base)) <= tolerance, (
            cid, score, float(base), tolerance
        )


def _assert_ranked(numpy, actual, reference, blobs, query, k):
    ordered = sorted(reference, key=lambda row: (-row[2], -row[0], row[1]))
    expected_ids = [cid for _, cid, _ in ordered[:k]]
    actual_ids = [cid for cid, _, _ in actual]
    assert len(actual_ids) == len(set(actual_ids)) == len(expected_ids)
    # Only byte-identical stored vectors may exchange IDs at the top-k boundary.
    assert [blobs[cid] for cid in actual_ids] == [blobs[cid] for cid in expected_ids]
    _assert_scores(numpy, actual, reference, blobs, query)


@pytest.mark.parametrize("dim", [384, 1024])
@pytest.mark.parametrize("count", [*range(3, 16), 2011])
@pytest.mark.parametrize("binary", [False, True], ids=["exact", "binary-survivors"])
def test_float32_scores_match_base_order(tmp_path, dim, count, binary):
    """Default batches, both loaders, top-k ties and the real binary rescore."""
    numpy = pytest.importorskip("numpy")
    db_path = tmp_path / "scores.db"
    _seed(db_path, count)
    vs = VectorStore(db_path)
    try:
        identity = vs.register_profile(MODEL, PROVIDER, dim, task="chunk")
        vs.ensure_chunk_schema()
        rng = random.Random(9)
        vector = numpy.asarray([rng.gauss(0, 1) for _ in range(dim)], dtype=numpy.float32)
        vector /= numpy.linalg.norm(vector)
        query = [rng.gauss(0, 1) for _ in range(dim)]
        query_array = numpy.asarray(vs._normalized(query, expected_dim=dim), dtype=numpy.float32)
        duplicate = vector.astype("<f4").tobytes()
        # A distinct, well-separated best match pins real IDs as well as the
        # duplicate-content tie set inherited from the unmerged ORDER BY tests.
        blobs = {f"{i}:0": duplicate for i in range(count)}
        blobs["0:0"] = query_array.astype("<f4").tobytes()
        insertion = rng.sample(range(count), count)
        vs.connection.executemany(
            "INSERT INTO lcm_chunk_vectors(chunk_id, identity_hash, vec) VALUES(?, ?, ?)",
            [(f"{i}:0", identity, blobs[f"{i}:0"]) for i in insertion],
        )
        vs.connection.executemany(
            "INSERT INTO lcm_chunk_meta(chunk_id, identity_hash, store_id, chunk_index, "
            "char_start, char_end, token_estimate, embedded_at, archived) "
            "VALUES(?, ?, ?, 0, 0, 1, 1, ?, 0)",
            [(f"{i}:0", identity, i, float(i // 2)) for i in insertion],
        )
        if binary:
            vs.connection.executemany(
                "INSERT INTO lcm_chunk_binary(chunk_id, identity_hash, bits) VALUES(?, ?, ?)",
                [(cid, identity, numpy.packbits(numpy.frombuffer(blob, dtype="<f4") >= 0).tobytes())
                 for cid, blob in blobs.items()],
            )
        vs.connection.commit()
        rows = vs.connection.execute(
            "SELECT v.rowid, v.chunk_id, v.vec FROM lcm_chunk_meta m "
            "JOIN lcm_chunk_vectors v ON v.chunk_id=m.chunk_id "
            "AND v.identity_hash=m.identity_hash WHERE m.identity_hash=? AND m.archived=0 "
            "ORDER BY m.embedded_at DESC, m.rowid",
            (identity,),
        ).fetchall()
        selected = [f"{i}:0" for i in reversed(range(count))]
        batches = [selected[start:start + vs.bounded_scan_rows]
                   for start in range(0, count, vs.bounded_scan_rows)]
        assert len(batches[-1]) in range(3, 16)
        if count == 2011:
            assert [len(batch) for batch in batches] == [2000, 11]
        rowids = {row["chunk_id"]: row["rowid"] for row in rows}
        for batch in batches:
            reference = _reference(numpy, rows, [batch], dim, query_array)
            for loader in ("python", "numpy"):
                loaded = (vs._load_chunk_matrix(numpy, identity, dim, batch, "float32")
                          if loader == "numpy" else
                          vs._load_chunk_vectors_for_ids(identity, dim, batch, "float32"))
                ids, kinds = loaded[1:3]
                assert set(ids) == set(batch)
                assert loaded[0] == [rowids[cid] for cid in ids]
                matrix = numpy.asarray(loaded[3], dtype=numpy.float32)
                assert [row.tobytes() for row in matrix] == [blobs[cid] for cid in ids]
                _assert_scores(
                    numpy, list(zip(ids, matrix @ query_array, kinds)), reference, blobs, query_array
                )
        # The default prescreen multiplier retains every odd-sized survivor
        # pool, while k still splits the byte-identical duplicate tie set.
        k = max(2, (count + vs.knn_prescreen_multiplier - 1) // vs.knn_prescreen_multiplier)
        reference = _reference(numpy, rows, [selected] if binary else batches, dim, query_array)
        result = vs.knn_chunks(query, k=k, model=MODEL, provider=PROVIDER, full_scan=True)
        _assert_ranked(numpy, result, reference, blobs, query_array, k)
        assert (result.coverage, result.scanned, result.total) == (
            "full_approx" if binary else "full", None, None
        )
        if not binary:
            # Force a real scan cap so scanned/total are integers, not only None.
            cap = count - 1
            bounded_ids = selected[:cap]
            bounded_batches = [bounded_ids[start:start + vs.bounded_scan_rows]
                               for start in range(0, cap, vs.bounded_scan_rows)]
            bounded = vs.knn_chunks(
                query, k=k, model=MODEL, provider=PROVIDER, full_scan=True, scan_max_rows=cap
            )
            reference = _reference(numpy, rows, bounded_batches, dim, query_array)
            _assert_ranked(numpy, bounded, reference, blobs, query_array, min(k, cap))
            assert (bounded.coverage, bounded.scanned, bounded.total) == ("bounded", cap, count)
    finally:
        vs.close()


@pytest.mark.parametrize("loader", ["python", "numpy"])
def test_direct_chunk_loaders_exclude_archived_messages(tmp_path, loader):
    """Explicit archived IDs must be rejected even when their BLOBs remain valid."""
    numpy = pytest.importorskip("numpy")
    db_path = tmp_path / "archived.db"
    _seed(db_path, 3)
    vs = VectorStore(db_path)
    try:
        identity = vs.register_profile(MODEL, PROVIDER, 384, task="chunk")
        embedding_identity = EmbeddingIdentity.canonical(
            PROVIDER, MODEL, "", 384, "float32", "little", "chunk"
        )
        for i in range(3):
            vs.record_chunk_embedding(
                f"{i}:0", MODEL, [1.0] + [0.0] * 383, store_id=i, chunk_index=0,
                char_start=0, char_end=1, token_estimate=1,
                identity=embedding_identity,
            )
        assert vs.archive_chunks_for_messages([1]) == 1
        assert vs.connection.execute(
            "SELECT COUNT(*) FROM lcm_chunk_vectors WHERE chunk_id='1:0'"
        ).fetchone()[0] == 1
        for selected, expected in [(["1:0"], []), (["1:0", "0:0", "2:0"], ["0:0", "2:0"])]:
            loaded = (vs._load_chunk_matrix(numpy, identity, 384, selected, "float32")
                      if loader == "numpy" else
                      vs._load_chunk_vectors_for_ids(identity, 384, selected, "float32"))
            assert set(loaded[1]) == set(expected)
            assert loaded[0] == [int(cid.split(":")[0]) + 1 for cid in loaded[1]]
            assert loaded[2] == ["chunk"] * len(expected)
            matrix = numpy.asarray(loaded[3], dtype=numpy.float32)
            assert matrix.size == len(expected) * 384
            if expected:
                numpy.testing.assert_array_equal(matrix[:, 0], 1)
                numpy.testing.assert_array_equal(matrix[:, 1:], 0)
    finally:
        vs.close()
