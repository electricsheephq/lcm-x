"""Regression coverage for lossless issue247 lifecycle admission."""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path):
    return LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"),
    )


def test_same_finalized_session_hydrates_frontier_once_and_other_session_cannot_borrow(
    tmp_path,
):
    conversation_id = "issue247-finalized-rebind"
    source_session = "finalized-session"
    source_engine = _engine(tmp_path)
    source_engine.on_session_start(
        source_session,
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_id = source_engine._store.append(
        source_session,
        {"role": "user", "content": "retained source"},
        conversation_id=conversation_id,
    )
    source_engine._lifecycle.advance_frontier(
        conversation_id,
        source_session,
        source_id,
    )
    finalized = source_engine._lifecycle.finalize_session(
        conversation_id,
        source_session,
        frontier_store_id=source_id,
    )
    assert finalized is not None
    assert finalized.current_session_id is None
    assert finalized.last_finalized_session_id == source_session
    assert finalized.current_frontier_store_id == 0
    assert finalized.last_finalized_frontier_store_id == source_id
    source_engine.shutdown()

    resumed = _engine(tmp_path)
    resumed.on_session_start(
        source_session,
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    hydrated = resumed._lifecycle.get_by_conversation(conversation_id)
    assert hydrated is not None
    assert hydrated.current_session_id == source_session
    assert hydrated.current_frontier_store_id == source_id
    assert resumed._last_compacted_store_id == source_id

    resumed.on_session_start(
        source_session,
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    repeated = resumed._lifecycle.get_by_conversation(conversation_id)
    assert repeated is not None
    assert repeated.current_session_id == source_session
    assert repeated.current_frontier_store_id == source_id
    assert resumed._last_compacted_store_id == source_id
    resumed.shutdown()

    unproven = _engine(tmp_path)
    unproven.on_session_start(
        "different-session",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    refused = unproven._lifecycle.get_by_conversation(conversation_id)
    assert refused is not None
    assert refused.current_session_id == "different-session"
    assert refused.current_frontier_store_id == 0
    assert refused.last_finalized_session_id == source_session
    assert refused.last_finalized_frontier_store_id == source_id
    assert unproven._last_compacted_store_id == 0
    unproven.shutdown()
