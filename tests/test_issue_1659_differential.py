"""Small seeded truth oracle; the full sweep reuses this function in scratch."""
import json
import random
import sqlite3

import pytest

from hermes_lcm import command
from tests.test_issue_1652_catchup_disclosure import backfill, engine as engine, record
from tests.test_issue_1536_chunk_revision import (
    LONG, add_rows, chunking, metadata, store as store,
)
from tests.test_issue_1543_coverage_marker import certify, older_walk


@pytest.mark.parametrize("seed", range(60))
def test_no_false_coverage(engine, store, monkeypatch, seed):
    rng = random.Random(seed)
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    configured = rng.choice(["full", "heads", "conversational"])
    engine._config.embedding_content_policy = configured
    backfill(engine, "chunk", limit=100)
    for _ in range(rng.randint(1, 6)):
        engine._store.append("seed", {"role": rng.choice(["user", "assistant", "tool", "system"]),
                                      "content": "cobalt seeded payload. " * rng.randint(1, 90)})
    arrival = rng.choice(["before", "scan", "write", "none"])

    def append():
        engine._store.append("seed", {"role": "user", "content": f"cobalt arrival seed {seed}"})

    with monkeypatch.context() as patch:
        if arrival == "before":
            append()
        elif arrival in {"scan", "write"}:
            seam = "_chunk_pending_rows" if arrival == "scan" else "_record_embedding_catchup"
            original = getattr(command, seam)

            def interleaved(*args, **kwargs):
                if arrival == "write":
                    append()
                result = original(*args, **kwargs)
                if arrival == "scan" and args[3] == 1:
                    append()
                return result

            patch.setattr(command, seam, interleaved)
        result = command._chunk_backfill_run(engine._config, engine._store.db_path, apply=True,
            limit=100, retry_uncertain=False, policy=rng.choice(["full", "heads", "conversational"]))
        assert result["error"] is None, (seed, result)
    with sqlite3.connect(engine._store.db_path) as conn:
        sources = conn.execute("SELECT store_id,role,content FROM messages ORDER BY store_id").fetchall()
        actual = {r[0] for r in conn.execute("SELECT chunk_id FROM lcm_chunk_meta WHERE identity_hash="
            "(SELECT identity_hash FROM lcm_embedding_profile WHERE active=1 AND task='chunk')")}
    truth = {c.chunk_id for source in sources for c in chunking.chunk_message(*source, policy=configured)
             if c.char_end > c.char_start}
    state = record(engine, "chunk")
    assert state["remaining"] or state["operator_retry"] or truth <= actual, (seed, truth - actual, state)
    assert sources == list(engine._store._conn.execute(
        "SELECT store_id,role,content FROM messages ORDER BY store_id"))

    # Disjoint band layouts, small page caps and an upper arrival every call.
    config, vectors, identity, _ = store
    config.ollama_base_url = "http://192.0.2.1:11434"
    n = rng.randint(8, 24)
    first = rng.randint(3, n - 2)
    rows = [(i, rng.choice(["user", "assistant", "tool"]), LONG) for i in range(1, n + 1)]
    policy = rng.choice(["conversational", "heads"])
    config.embedding_content_policy = policy
    add_rows(store, rows, set(range(first, n + 1)))
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (first, n), policy)
    older_walk(store, {1, *rng.sample(range(2, first), rng.randrange(first - 1))})
    monkeypatch.setattr(command, "_CHUNK_HISTORY_SCAN_MESSAGES", rng.randint(1, 5))
    for turn in range(8):
        high = n + turn * 2 + 1
        add_rows(store, [(high, "user", LONG), (high + 1, "user", LONG)],
                 {high + 1, *({high} if rng.random() < .5 else set())})
        before = list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))
        command._validate_cloud_chunk_metadata(config, config.database_path)
        proof = json.loads(metadata(store, "chunk_policy_revision_through"))
        published = {r[0] for r in vectors.connection.execute(
            "SELECT chunk_id FROM lcm_chunk_meta WHERE identity_hash=?", (identity.identity_hash,))}
        expected = {c.chunk_id for row in vectors.connection.execute(
            "SELECT store_id,role,content FROM messages WHERE store_id BETWEEN ? AND ?",
            (proof["from"], proof["through"])) for c in chunking.chunk_message(*row, policy=policy)}
        assert expected <= published, (seed, turn, proof, expected - published)
        assert before == list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))
