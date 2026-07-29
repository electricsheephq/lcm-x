#!/usr/bin/env python3
"""Time cold and warm exact int8 KNN at the frozen #171 synthetic sizes."""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.replay import _ensure_hermes_lcm_package


SIZES = (10_000, 50_000, 185_000)
MODEL = "fast-scan-synthetic"
PROVIDER = "bench"
DIM = 384


def _seed(db_path: Path, count: int, rng: np.random.Generator) -> None:
    _ensure_hermes_lcm_package()
    from hermes_lcm.vector_store import VectorStore

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE messages (
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
        "INSERT INTO messages VALUES (?, 'synthetic', 'bench', 'user', '', ?)",
        ((index, float(index)) for index in range(count)),
    )
    conn.commit()
    conn.close()

    store = VectorStore(db_path)
    try:
        store.ensure_chunk_schema()
        identity_hash = store.register_profile(
            MODEL, PROVIDER, DIM, dtype="int8", task="chunk"
        )
        conn = store.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            for start in range(0, count, 2_000):
                end = min(count, start + 2_000)
                vectors = rng.standard_normal((end - start, DIM)).astype(
                    np.float32
                )
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                max_abs = np.max(np.abs(vectors), axis=1)
                scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(
                    np.float32
                )
                quantized = np.rint(vectors / scales[:, None]).clip(
                    -127, 127
                ).astype(np.int8)
                vector_rows = []
                meta_rows = []
                for offset, index in enumerate(range(start, end)):
                    chunk_id = f"{index}:0"
                    blob = quantized[offset].tobytes() + struct.pack(
                        "<f", float(scales[offset])
                    )
                    vector_rows.append((chunk_id, identity_hash, blob))
                    meta_rows.append(
                        (
                            chunk_id,
                            identity_hash,
                            index,
                            0,
                            0,
                            1,
                            1,
                            "synthetic",
                        )
                    )
                conn.executemany(
                    "INSERT INTO lcm_chunk_vectors VALUES (?, ?, ?)",
                    vector_rows,
                )
                conn.executemany(
                    """
                    INSERT INTO lcm_chunk_meta(
                        chunk_id, identity_hash, store_id, chunk_index,
                        char_start, char_end, token_estimate, embedded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    meta_rows,
                )
            conn.execute(
                "UPDATE lcm_embedding_profile SET data_version = data_version + 1 "
                "WHERE identity_hash = ?",
                (identity_hash,),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        store.close()


def _time_one(
    db_path: Path, count: int, *, resident_max_mb: int, warm_runs: int
) -> dict[str, object]:
    _ensure_hermes_lcm_package()
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.vector_store import VectorStore

    store = VectorStore(
        db_path,
        config=LCMConfig(
            embedding_bounded_scan_rows=2_000,
            knn_resident_max_mb=resident_max_mb,
        ),
    )
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 1.0
    try:
        started = time.perf_counter()
        cold = store.knn_chunks(
            query.tolist(),
            k=50,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        cold_ms = (time.perf_counter() - started) * 1_000.0
        warm_ms = []
        for _ in range(warm_runs):
            started = time.perf_counter()
            warm = store.knn_chunks(
                query.tolist(),
                k=50,
                model=MODEL,
                provider=PROVIDER,
                full_scan=True,
            )
            warm_ms.append((time.perf_counter() - started) * 1_000.0)
        if cold.coverage != "full" or warm.coverage != "full":
            raise RuntimeError(
                f"coverage mismatch: cold={cold.coverage}, warm={warm.coverage}"
            )
        return {
            "n": count,
            "dim": DIM,
            "resident_max_mb": resident_max_mb,
            "cold_ms": round(cold_ms, 3),
            "warm_ms": [round(value, 3) for value in warm_ms],
            "warm_p50_ms": round(float(np.median(warm_ms)), 3),
            "coverage": warm.coverage,
        }
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--resident-max-mb", type=int, default=128)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    invalid = [size for size in args.sizes if size not in SIZES]
    if invalid:
        parser.error(f"--sizes must be selected from {SIZES}: {invalid}")

    def run(root: Path) -> None:
        for count in args.sizes:
            db_path = root / f"fast-scan-{count}.db"
            _seed(db_path, count, np.random.default_rng(171 + count))
            print(
                json.dumps(
                    _time_one(
                        db_path,
                        count,
                        resident_max_mb=max(0, args.resident_max_mb),
                        warm_runs=max(1, args.warm_runs),
                    ),
                    sort_keys=True,
                )
            )

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        run(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="lcm-fast-scan-") as temp_dir:
            run(Path(temp_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
