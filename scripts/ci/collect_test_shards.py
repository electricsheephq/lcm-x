"""Emit four deterministic pytest argument files from the selected collection."""

import hashlib
import os
from pathlib import Path


def pytest_collection_finish(session):
    nodeids = sorted(item.nodeid for item in session.items)
    if not nodeids or len(set(nodeids)) != len(nodeids):
        raise ValueError("expected a nonempty collection of unique test node IDs")

    shards = [nodeids[index::4] for index in range(4)]
    counts = [len(shard) for shard in shards]
    total = sum(counts)
    print(f"Collected total: {len(nodeids)}")
    for index, count in enumerate(counts, 1):
        print(f"Shard {index}/4: {count} tests")
    print(f"Shard sum: {total}")
    if total != len(nodeids) or set().union(*map(set, shards)) != set(nodeids):
        raise ValueError("shards do not cover the collected tests exactly once")
    if not all(counts):
        raise ValueError("every shard must contain tests")

    output = Path(os.environ["PYTEST_SHARD_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards, 1):
        (output / f"shard-{index}.txt").write_text("\n".join(shard) + "\n")

    # Each shard job collects on its own; tests-ran compares these fingerprints so a job that
    # collected a different list (and so split it differently) cannot pass unnoticed.
    fingerprint = os.environ.get("PYTEST_SHARD_FINGERPRINT")
    if fingerprint:
        digest = hashlib.sha256("\n".join(nodeids).encode()).hexdigest()
        Path(fingerprint).parent.mkdir(parents=True, exist_ok=True)
        Path(fingerprint).write_text(f"{len(nodeids)} {digest}\n")
