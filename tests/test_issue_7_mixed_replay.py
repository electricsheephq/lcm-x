"""Issue #7: compaction replay must support mixed NEW/REPLAY ordering."""

import json

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _close_without_session_end(engine: LCMEngine) -> None:
    engine._store.close()
    engine._dag.close()
    engine._lifecycle.close()


def _seed_compacted_session(tmp_path, monkeypatch, *, stem: str, keep_open=False):
    session_id = f"{stem}-session"
    config = LCMConfig(
        database_path=str(tmp_path / f"{stem}.db"),
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.on_session_start(
        session_id,
        platform="cli",
        conversation_id=f"{stem}-conversation",
        context_length=200_000,
    )

    def summary(**_kwargs):
        return "Issue seven compacted history.\nExpand for details about: issue seven", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summary)
    history = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "older user turn " + "u" * 200},
        {"role": "assistant", "content": "older assistant turn"},
        {"role": "user", "content": "replay U1"},
        {"role": "assistant", "content": "replay A1"},
        {"role": "user", "content": "replay U2"},
        {"role": "assistant", "content": "replay A2"},
    ]
    compacted = engine.compress(history, current_tokens=10_000)
    original_count = engine._store.get_session_count(session_id)
    assert original_count == len(history)
    assert engine._load_compacted_active_replay_snapshot_digests()
    if not keep_open:
        _close_without_session_end(engine)
    return config, session_id, compacted, original_count, engine


def test_real_compress_rebind_deduplicates_interleaved_replay_occurrences(
    tmp_path,
    monkeypatch,
):
    """A registered compacted snapshot may be replayed around genuinely new rows.

    The host can finish a compression in the middle of a turn. Its next batch
    then contains the provider-visible compacted snapshot plus rows produced
    while compression was running. A scalar prefix cursor cannot represent
    this ordering: the new row at index zero forces cursor zero, which used to
    append every durable snapshot row again.
    """
    _config, session_id, compacted, original_count, after = _seed_compacted_session(
        tmp_path,
        monkeypatch,
        stem="issue-7-mixed-replay",
        keep_open=True,
    )
    for content in ("replay U1", "replay A1", "replay U2", "replay A2"):
        assert sum(message.get("content") == content for message in compacted) == 1

    # The next host batch reuses the exact compacted message objects and adds
    # rows that arrived while compression was completing.  Force the ordinary
    # existing-session reconciliation path that a rebind uses.
    after._ingest_cursor_needs_reconcile = True
    new_rows = [
        {"role": "tool", "tool_call_id": "new-before", "content": "new N0a"},
        {"role": "assistant", "content": "new N0b"},
        {"role": "user", "content": "replay U1"},
        {"role": "assistant", "content": "new N1"},
        {"role": "tool", "tool_call_id": "new-after", "content": "new N2"},
    ]
    split = max(1, len(compacted) // 2)
    mixed = [
        new_rows[0],
        new_rows[1],
        new_rows[2],
        *compacted[:split],
        new_rows[3],
        new_rows[4],
        *compacted[split:],
    ]

    try:
        after._ingest_messages(mixed)

        rows = after._store.get_session_messages(session_id)
        contents = [row["content"] for row in rows]
        assert len(rows) == original_count + len(new_rows)
        assert contents.count("replay U1") == 2
        for content in ("replay A1", "replay U2", "replay A2"):
            assert contents.count(content) == 1
        for content in ("new N0a", "new N0b", "new N1", "new N2"):
            assert contents.count(content) == 1
    finally:
        after.shutdown()


def test_incomplete_registered_snapshot_is_preserved_as_ambiguous(
    tmp_path,
    monkeypatch,
):
    """A partial snapshot is resemblance, not replay proof."""
    config, session_id, compacted, original_count, _closed = _seed_compacted_session(
        tmp_path,
        monkeypatch,
        stem="issue-7-incomplete-replay",
    )
    incomplete = compacted[:-1]
    split = max(1, len(incomplete) // 2)
    incoming = [
        {"role": "assistant", "content": "new before incomplete snapshot"},
        *incomplete[:split],
        {"role": "assistant", "content": "new inside incomplete snapshot"},
        *incomplete[split:],
    ]

    after = LCMEngine(config=config, hermes_home=str(tmp_path))
    after.on_session_start(
        session_id,
        platform="cli",
        conversation_id="issue-7-incomplete-replay-conversation",
        context_length=200_000,
    )
    try:
        after._ingest_messages(incoming)

        assert after._store.get_session_count(session_id) == original_count + len(
            incoming
        )
        assert (
            after._last_ingest_reconciliation.get(
                "replayed_compacted_snapshot_rows", 0
            )
            == 0
        )
    finally:
        after.shutdown()


def test_identity_equal_occurrences_inside_snapshot_span_remain_ambiguous(
    tmp_path,
    monkeypatch,
):
    """A new byte-identical row cannot be mistaken for the replay occurrence."""
    config, session_id, compacted, original_count, _closed = _seed_compacted_session(
        tmp_path,
        monkeypatch,
        stem="issue-7-identity-equal",
    )
    incoming = [
        {"role": "assistant", "content": "new row forcing cursor zero"},
        compacted[0],
        compacted[1],
        {"role": "user", "content": "replay U1"},
        *compacted[2:],
    ]

    after = LCMEngine(config=config, hermes_home=str(tmp_path))
    after.on_session_start(
        session_id,
        platform="cli",
        conversation_id="issue-7-identity-equal-conversation",
        context_length=200_000,
    )
    try:
        after._ingest_messages(incoming)

        rows = after._store.get_session_messages(session_id)
        contents = [row["content"] for row in rows]
        # A restarted engine has no process-local occurrence proof. Preserve
        # the complete copied snapshot plus the equal new row rather than
        # choosing one and potentially losing or reordering real input.
        assert contents.count("replay U1") == 3
        assert contents.count("new row forcing cursor zero") == 1
        assert len(rows) == original_count + len(incoming)
        assert (
            after._last_ingest_reconciliation.get(
                "replayed_compacted_snapshot_rows", 0
            )
            == 0
        )
    finally:
        after.shutdown()


def test_snapshot_proof_cannot_span_a_positive_reconciliation_cursor(
    tmp_path,
    monkeypatch,
):
    """A pre-cursor occurrence cannot prove away an equal post-cursor row."""
    config, session_id, compacted, original_count, _closed = _seed_compacted_session(
        tmp_path,
        monkeypatch,
        stem="issue-7-positive-cursor",
    )
    incoming = [dict(message) for message in compacted]

    after = LCMEngine(config=config, hermes_home=str(tmp_path))
    after.on_session_start(
        session_id,
        platform="cli",
        conversation_id="issue-7-positive-cursor-conversation",
        context_length=200_000,
    )
    # Isolate the cursor/snapshot interaction: row zero has already been
    # reconciled by a separate durable-prefix proof. The remaining rows are
    # only a partial registered snapshot. Even if they are byte-identical to
    # the old snapshot, one can be a genuinely new occurrence, so all must be
    # preserved rather than letting the pre-cursor row complete the proof.
    monkeypatch.setattr(
        after,
        "_reconcile_ingest_cursor_from_store",
        lambda *_args, **_kwargs: 1,
    )
    try:
        after._ingest_messages(incoming)

        assert after._store.get_session_count(session_id) == (
            original_count + len(incoming) - 1
        )
        assert (
            after._last_ingest_reconciliation.get(
                "replayed_compacted_snapshot_rows", 0
            )
            == 0
        )
    finally:
        after.shutdown()


def test_complete_identity_equal_copied_snapshot_is_preserved_as_new(
    tmp_path,
    monkeypatch,
):
    """A complete content match is not proof when every occurrence is new."""
    _config, session_id, compacted, original_count, engine = _seed_compacted_session(
        tmp_path,
        monkeypatch,
        stem="issue-7-copied-complete-snapshot",
        keep_open=True,
    )
    split = max(1, len(compacted) // 2)
    incoming = [
        {"role": "assistant", "content": "new before copied snapshot"},
        *[dict(message) for message in compacted[:split]],
        {"role": "user", "content": "new inside copied snapshot"},
        *[dict(message) for message in compacted[split:]],
    ]
    engine._ingest_cursor_needs_reconcile = True

    try:
        engine._ingest_messages(incoming)

        assert engine._store.get_session_count(session_id) == (
            original_count + len(incoming)
        )
        assert (
            engine._last_ingest_reconciliation.get(
                "replayed_compacted_snapshot_rows", 0
            )
            == 0
        )
    finally:
        engine.shutdown()


def test_out_of_order_exact_alias_keeps_snapshot_ambiguous(
    tmp_path,
    monkeypatch,
):
    """One registered object appearing twice cannot prove exact replay."""
    _config, session_id, compacted, original_count, engine = _seed_compacted_session(
        tmp_path,
        monkeypatch,
        stem="issue-7-out-of-order-alias",
        keep_open=True,
    )
    # Reuse the final registered object before the otherwise ordered snapshot.
    # The second occurrence must make the whole mapping ambiguous.  Counting
    # matches only after the moving lower bound would overlook the first alias
    # and incorrectly suppress the complete suffix as replay.
    incoming = [compacted[-1], *compacted]
    engine._ingest_cursor_needs_reconcile = True
    monkeypatch.setattr(
        engine,
        "_reconcile_ingest_cursor_from_store",
        lambda *_args, **_kwargs: 0,
    )

    try:
        engine._ingest_messages(incoming)

        assert engine._store.get_session_count(session_id) == (
            original_count + len(incoming)
        )
        assert (
            engine._last_ingest_reconciliation.get(
                "replayed_compacted_snapshot_rows", 0
            )
            == 0
        )
    finally:
        engine.shutdown()


def test_snapshot_metadata_migrates_to_bounded_whole_digests(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "snapshot-metadata.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.on_session_start(
        "snapshot-metadata-session",
        platform="cli",
        conversation_id="snapshot-metadata-conversation",
        context_length=200_000,
    )
    prefix = "snapshot-metadata-proof"
    legacy_digest = "a" * 64
    new_digest = "b" * 64
    metadata_key = engine._replay_snapshot_metadata_key(prefix)
    legacy_message_digests = ["c" * 64 for _ in range(8192)]

    try:
        engine._store.write_metadata_json(
            [metadata_key],
            json.dumps(
                {
                    "version": 2,
                    "snapshots": [
                        {
                            "digest": legacy_digest,
                            "message_digests": legacy_message_digests,
                        },
                    ],
                },
                sort_keys=True,
            ),
        )
        assert engine._load_replay_snapshot_digests(prefix) == [legacy_digest]

        engine._remember_replay_snapshot(prefix, new_digest)

        metadata = engine._store.read_metadata_json(metadata_key)
        assert metadata == {
            "version": 1,
            "digests": [legacy_digest, new_digest],
        }
        assert len(json.dumps(metadata)) < 256

        for index in range(20):
            engine._remember_replay_snapshot(prefix, f"{index:064x}")
        metadata = engine._store.read_metadata_json(metadata_key)
        assert metadata["version"] == 1
        assert len(metadata["digests"]) == 16
        assert all(len(digest) == 64 for digest in metadata["digests"])
        assert len(json.dumps(metadata)) < 1200
    finally:
        engine.shutdown()
