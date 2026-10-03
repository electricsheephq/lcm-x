#!/usr/bin/env python3
"""Synthetic, reproducible comparison of chunk float32 matrix loading.

Run from the repository root with:

    python benchmarks/benchmark_float32_chunk_loader.py --count 2000 --dim 384

The script creates a temporary SQLite database. It does not read a Hermes
profile, call a provider, or make a performance assertion: elapsed time depends
on the host. It verifies row IDs, chunk IDs, matrix values, and dot-product
scores against the established Python-decoder construction before reporting
median load timings.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy

ROOT = Path(__file__).resolve().parents[1]
MODEL = "synthetic-float32"
PROVIDER = "benchmark"


def load_plugin_module():
    """Load this checkout as a package without a Hermes profile or host runtime."""
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the repository plugin module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_lcm"] = module
    spec.loader.exec_module(module)
    return module


def seed_messages(db_path: Path, count: int) -> None:
    """Create deterministic synthetic message rows used by chunk joins."""
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                source TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO messages VALUES (?, 'synthetic', 'benchmark', 'user', '', ?)",
            ((index + 1, float(index)) for index in range(count)),
        )
        connection.commit()
    finally:
        connection.close()


def decoder_matrix(store, identity_hash: str, dim: int, chunk_ids: list[str]):
    """Build a matrix through the legacy Python-float decoder for comparison."""
    rowids, loaded_ids, kinds, raw_vectors = store._load_chunk_vectors_for_ids(
        identity_hash, dim, chunk_ids, "float32"
    )
    return rowids, loaded_ids, kinds, numpy.asarray(raw_vectors, dtype=numpy.float32)


def median_load_seconds(load, repeats: int) -> float:
    """Return median elapsed load time, excluding fixture setup and validation."""
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        load()
        samples.append(time.perf_counter() - started)
    return float(numpy.median(numpy.asarray(samples)))


def main() -> int:
    """Verify exact synthetic parity and print host-dependent loader timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000, help="synthetic vectors (default: 2000)")
    parser.add_argument("--dim", type=int, default=384, help="vector dimensions (default: 384)")
    parser.add_argument("--repeats", type=int, default=5, help="timed loads per path (default: 5)")
    parser.add_argument("--seed", type=int, default=20261003, help="RNG seed (default: 20261003)")
    args = parser.parse_args()
    if args.count <= 0 or args.dim <= 0 or args.repeats <= 0:
        parser.error("--count, --dim, and --repeats must be positive")

    load_plugin_module()
    vector_store = importlib.import_module("hermes_lcm.vector_store")
    rng = numpy.random.default_rng(args.seed)
    vectors = rng.standard_normal((args.count, args.dim), dtype=numpy.float32)
    vectors /= numpy.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.standard_normal(args.dim, dtype=numpy.float32)
    query /= numpy.linalg.norm(query)

    with tempfile.TemporaryDirectory(prefix="lcm-x-float32-benchmark-") as directory:
        db_path = Path(directory) / "synthetic.db"
        seed_messages(db_path, args.count)
        store = vector_store.VectorStore(db_path, bounded_scan_rows=args.count)
        try:
            identity = vector_store.EmbeddingIdentity.canonical(
                PROVIDER, MODEL, "", args.dim, "float32", "little", "chunk"
            )
            store.register_profile(MODEL, PROVIDER, args.dim, task="chunk")
            for index, vector in enumerate(vectors):
                store.record_chunk_embedding(
                    f"{index + 1}:0",
                    MODEL,
                    vector.tolist(),
                    store_id=index + 1,
                    chunk_index=0,
                    char_start=0,
                    char_end=1,
                    token_estimate=1,
                    identity=identity,
                )
            chunk_ids = [f"{index + 1}:0" for index in range(args.count)]

            expected = decoder_matrix(store, identity.identity_hash, args.dim, chunk_ids)
            actual = store._load_chunk_matrix(
                numpy, identity.identity_hash, args.dim, chunk_ids, "float32"
            )
            assert actual[:3] == expected[:3]
            numpy.testing.assert_array_equal(actual[3], expected[3])
            numpy.testing.assert_array_equal(actual[3] @ query, expected[3] @ query)

            decoder_seconds = median_load_seconds(
                lambda: decoder_matrix(store, identity.identity_hash, args.dim, chunk_ids), args.repeats
            )
            direct_seconds = median_load_seconds(
                lambda: store._load_chunk_matrix(
                    numpy, identity.identity_hash, args.dim, chunk_ids, "float32"
                ),
                args.repeats,
            )
        finally:
            store.close()

    ratio = decoder_seconds / direct_seconds if direct_seconds else float("inf")
    print("synthetic float32 chunk-loader benchmark")
    print(f"vectors={args.count} dim={args.dim} repeats={args.repeats} seed={args.seed}")
    print("verification=row IDs, chunk IDs, matrix values, and dot-product scores match")
    print(f"python_decoder_median_s={decoder_seconds:.6f}")
    print(f"direct_blob_median_s={direct_seconds:.6f}")
    print(f"median_speedup={ratio:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
