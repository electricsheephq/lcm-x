"""Review probe #1843: an unproven upper range while the own history walk is advancing."""
from types import SimpleNamespace

from tests.test_issue_1536_chunk_revision import (  # noqa: F401
    LONG, chunking, command, counts, metadata, store as store,
)



def oracle(store, policy="conversational"):
    _, vectors, identity, _ = store
    have = set(counts(store))
    missing = []
    for sid, role, content in vectors.connection.execute("SELECT store_id, role, content FROM messages"):
        if any(c.char_end > c.char_start for c in chunking.chunk_message(sid, role, content, policy=policy)):
            if sid not in have:
                missing.append(sid)
    return missing


def status(store, policy="conversational"):
    config, vectors, _, _ = store
    config.embedding_content_policy = policy
    dv = vectors.connection.execute("PRAGMA data_version").fetchone()[0]
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
    line = next(x for x in command._embedding_maintenance_status_lines(engine) if "backlog_chunks" in x)
    assert vectors.connection.execute("PRAGMA data_version").fetchone()[0] == dv
    return line


def run(store, *, limit=32, incremental=True, policy="conversational"):
    config, vectors, _, _ = store
    before = list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))
    result = command._chunk_backfill_run(
        config, config.database_path, apply=True, limit=limit,
        retry_uncertain=False, policy=policy, incremental=incremental,
    )
    assert before == list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))
    return result


def test_status_and_remaining_match_oracle_after_manual_limit(store):
    """RED at head a04f16e, GREEN at base 20891a9: default constants, no metadata edits, no older writer."""
    _, vectors, _, _ = store
    vectors.connection.executemany("INSERT INTO messages(store_id, role, content) VALUES(?,?,?)",
                                   [(i, "user", LONG) for i in range(1, 201)])
    vectors.connection.commit()
    for _ in range(3):
        run(store)
    vectors.connection.executemany("INSERT INTO messages(store_id, role, content) VALUES(?,?,?)",
                                   [(i, "user", LONG) for i in range(201, 207)])
    vectors.connection.commit()
    run(store, limit=3, incremental=False)  # `/lcm embed backfill --limit 3`: newest-first, leaves 201-203
    for _ in range(8):
        result = run(store)
        missing = oracle(store)
        assert status(store) in (f"embedding_backlog_chunks: {len(missing)}",
                                 f"embedding_backlog_chunks: {len(missing)} (gap-fill)"), (missing[-5:], status(store))
        assert bool(result["remaining"]) == bool(missing), (result["remaining"], missing[-5:])

    assert not oracle(store)


def test_full_policy_preserves_certificate_and_switch_back_does_not_scan(store, monkeypatch):
    from tests.test_issue_1536_chunk_revision import add_rows

    add_rows(store, [(i, "user", LONG) for i in range(1, 41)], set(range(1, 41)))
    metadata(store, "chunk_policy_revision", 2)
    run(store)
    run(store)
    before = metadata(store, "chunk_policy_revision_through")
    run(store, policy="full")
    assert metadata(store, "chunk_policy_revision_through") == before

    def unexpected(*args, **kwargs):
        raise AssertionError("switching back must retain the conversational proof")

    monkeypatch.setattr(command, "_chunk_gapfill_rows", unexpected)
    run(store)
    assert metadata(store, "chunk_policy_revision_through") == before
