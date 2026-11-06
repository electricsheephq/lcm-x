"""#1490: /lcm status's chunk backlog counts only what the automatic pass will process."""

import hermes_lcm.command as command
from tests.test_issue_1197_short_turn_chunks import FACT, revision, run, store  # noqa: F401  (fixture)


def test_current_revision_uncertain_chunk_is_not_in_backlog(store):  # noqa: F811
    _, vectors, identity, provider = store
    vectors.connection.execute("INSERT INTO messages(store_id, role, content) VALUES(1, 'user', ?)", (FACT,))
    assert run(store)["embedded"] == 1
    assert revision(store) == 2  # identity stamped at the current chunk-policy revision; no gap-fill pending

    vectors.connection.execute("INSERT INTO messages(store_id, role, content) VALUES(2, 'user', 'I work Fridays')")
    command._ensure_inflight_table(vectors.connection)
    vectors.connection.execute(
        "INSERT INTO lcm_embedding_backfill_inflight(embedded_id, identity_hash, state) "
        "VALUES('2:0', ?, 'uncertain')", (identity.identity_hash,),
    )

    assert command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, "conversational") == "0"
    assert run(store)["embedded"] == 0  # the automatic pass skips the uncertain chunk
    assert "I work Fridays" not in provider.texts
    assert command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, "conversational") == "0"

    assert run(store, retry=True)["embedded"] == 1  # an explicit retry still processes it
    assert command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, "conversational") == "0"
