"""Issue #7: compaction replay must support mixed NEW/REPLAY ordering."""

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _close_without_session_end(engine: LCMEngine) -> None:
    engine._store.close()
    engine._dag.close()
    engine._lifecycle.close()


def _seed_compacted_session(tmp_path, monkeypatch, *, stem: str):
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
    _close_without_session_end(engine)
    return config, session_id, compacted, original_count


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
    config, session_id, compacted, original_count = _seed_compacted_session(
        tmp_path,
        monkeypatch,
        stem="issue-7-mixed-replay",
    )
    for content in ("replay U1", "replay A1", "replay U2", "replay A2"):
        assert sum(message.get("content") == content for message in compacted) == 1

    after = LCMEngine(config=config, hermes_home=str(tmp_path))
    after.on_session_start(
        session_id,
        platform="cli",
        conversation_id="issue-7-mixed-replay-conversation",
        context_length=200_000,
    )
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
    config, session_id, compacted, original_count = _seed_compacted_session(
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
    config, session_id, compacted, original_count = _seed_compacted_session(
        tmp_path,
        monkeypatch,
        stem="issue-7-identity-equal",
    )
    incoming = [
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
        # Neither equal incoming occurrence is uniquely attributable. Preserve
        # both rather than choosing one and potentially losing or reordering
        # the genuinely new occurrence.
        assert contents.count("replay U1") == 3
        assert len(rows) == original_count + 2
        assert (
            after._last_ingest_reconciliation[
                "replayed_compacted_snapshot_rows"
            ]
            == 3
        )
        assert after._last_ingest_reconciliation["ambiguous_rows_preserved"] == 2
    finally:
        after.shutdown()
