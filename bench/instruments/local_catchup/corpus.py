"""Deterministic bounded-session corpus and zero-vector / legacy-gap baselines."""
from collections import Counter
from dataclasses import asdict
import shutil
import sqlite3
import time
from unittest.mock import patch

from .audit import connect, preimage
from .common import MODEL, config, dump, parser, product


def build(repo, host, db, rows=250000, shape="short", baseline="empty", seed=1586):
    if rows < 2 or rows % 2 or db.exists():
        raise ValueError("positive even row count and a new DB required")
    mod = product(repo, host)
    from benchmarking.fixtures import make_synthetic_fixture
    store = mod("store").MessageStore(db, hermes_home=str(db.parent))
    dag = mod("dag").SummaryDAG(db)
    vs = mod("vector_store").VectorStore(db, config=config(mod, db, False))
    vs.ensure_chunk_schema()
    identities = {task: vs.register_profile(MODEL, "fastembed", 384, task=task)
                  for task in ("summary", "chunk")}
    identity = mod("vector_store").EmbeddingIdentity.canonical("fastembed", MODEL, "", 384,
                                                             "float32", "little", "chunk")
    expected, tokens, chunks = {"chunks": {}, "leaves": {}}, Counter(), []
    canaries, sessions, long_count = [], [], 0
    for shard, offset in enumerate(range(0, rows // 2, 250)):
        name = f"s{seed}_shard{shard:06d}"
        fixture = make_synthetic_fixture(name=name, message_pairs=min(250, rows // 2-offset),
            canary_count=1, filler_words=20 if shape == "short" else (20, 300, 1200)[shard % 3])
        messages = [dict(m, timestamp=1700000000 + offset * 2 + i)
                    for i, m in enumerate(fixture.messages[1:])]
        if baseline == "upgrade" and shard == 0:
            messages[0]["content"] += " A factual sentence with several details." * 7000
        sessions.append(name)
        canaries.extend(asdict(c) for c in fixture.canaries)
        for i, msg in enumerate(messages):
            msg["content"] += f" [{name}:{i}]"
        counts = [mod("tokens").count_tokens(m["content"]) for m in messages]
        tokens.update(counts)
        with patch.object(time, "time", return_value=1700000000 + offset):
            ids = store.append_batch(name, messages, token_estimates=counts)
        for ident, msg in zip(ids, messages):
            parts = mod("chunking").chunk_message(ident, msg["role"], msg["content"], policy="conversational")
            expected["chunks"].update({c.chunk_id: c.token_estimate for c in parts})
            if baseline == "upgrade" and ((shard == 0 and ident == ids[0]) or ident % 4 == 0):
                selected = parts[:1] if ident == 1 else parts
                if ident == 1:
                    long_count = len(parts)
                chunks.extend(selected)
        leaf = mod("dag").SummaryNode(session_id=name, summary=f"{name}: {fixture.canaries[0].value}",
            source_ids=ids, token_count=20, source_token_count=sum(counts), created_at=1700000000+offset,
            earliest_at=1700000000+offset*2, latest_at=messages[-1]["timestamp"])
        expected["leaves"][str(dag.add_node(leaf))] = leaf.token_count
    for c in chunks:
        values = asdict(c)
        values.pop("text")
        values.pop("store_id")
        vs.record_chunk_embedding(c.chunk_id, MODEL, [1.0]+[0.0]*383,
            store_id=c.store_id, identity=identity, **values)
    if baseline == "upgrade" and long_count <= 32:
        raise RuntimeError("upgrade fixture must have more than 32 chunks")
    store.close()
    dag.close()
    vs.close()
    with connect(db) as c:
        pre = preimage(c)
    with sqlite3.connect(db) as c:
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {"rows": rows, "pairs": rows//2, "system_rows": 0, "tool_rows": 0,
        "stripped_system_rows": len(sessions), "shape": shape, "baseline": baseline, "seed": seed,
        "sessions": sessions, "canaries": canaries, "token_distribution": dict(sorted(tokens.items())),
        "input_tokens": sum(k*v for k, v in tokens.items()), "expected": expected,
        "identities": identities, "dim": 384, "preimage": pre, "database_bytes": db.stat().st_size,
        "upgrade_long_chunks": long_count, "estimator": "product tokens.count_tokens"}


def main():
    p = parser(__doc__)
    p.add_argument("--out", type=__import__("pathlib").Path, required=True)
    p.add_argument("--rows", type=int, default=250000)
    p.add_argument("--shape", choices=("short", "mixed"), default="short")
    p.add_argument("--baseline", choices=("empty", "upgrade"), default="empty")
    p.add_argument("--pilot", type=int, default=1000)
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--peak-disk-gb", type=float, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(a.out).free < max(150, a.peak_disk_gb+20)*2**30:
        raise RuntimeError("disk headroom stop")
    pilot = build(a.lcmx_repo, a.hermes_source, a.out/"pilot.db", min(a.pilot, a.rows), a.shape, a.baseline)
    # Include float32 vectors, SQLite overhead and a conservative WAL/copy margin.
    estimate = 3 * (pilot["database_bytes"] + len(pilot["expected"]["chunks"])*2048) * a.rows/pilot["rows"]
    dump(a.out/"pilot.json", dict(pilot, extrapolated_bytes=estimate))
    if estimate > a.peak_disk_gb*2**30:
        raise RuntimeError("full-build extrapolation exceeds peak-disk budget")
    if not a.pilot_only:
        dump(a.out/"manifest.json", build(a.lcmx_repo, a.hermes_source, a.out/"store.db",
                                         a.rows, a.shape, a.baseline))


if __name__ == "__main__":
    main()
