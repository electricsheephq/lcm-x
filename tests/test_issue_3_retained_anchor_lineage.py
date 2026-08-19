"""Occurrence-lineage foundation for the LCM-X issue #3 retained anchor."""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, name: str) -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            fresh_tail_count=100,
            leaf_chunk_tokens=100_000,
            database_path=str(tmp_path / f"{name}.db"),
        )
    )
    engine.on_session_start(name, platform="cli", context_length=200_000)
    return engine


def _store_row(engine: LCMEngine, content: str) -> dict:
    return next(
        row
        for row in engine._store.get_session_messages(engine._session_id)
        if row["content"] == content
    )


def test_registered_sole_user_maps_below_compaction_frontier(tmp_path) -> None:
    engine = _engine(tmp_path, "retained-lineage")
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "sole real prompt"},
        {"role": "assistant", "content": "initial reply"},
    ]

    try:
        active = engine.compress(messages)
        prompt_row = _store_row(engine, "sole real prompt")
        engine._last_compacted_store_id = max(
            row["store_id"]
            for row in engine._store.get_session_messages(engine._session_id)
        )
        mapped = engine._get_store_id_map_for_messages([active[1]])
    finally:
        engine.shutdown()

    assert mapped[id(active[1])] == prompt_row["store_id"]


def test_scaffold_shaped_real_user_requires_atomic_provenance(tmp_path) -> None:
    engine = _engine(tmp_path, "retained-scaffold-lineage")
    prompt = (
        "[Current user objective preserved from compacted history]\n"
        "This literal marker-shaped text was authored by the user."
    )
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "initial reply"},
    ]

    try:
        active = engine.compress(messages)
        prompt_row = _store_row(engine, prompt)
        assert engine._has_real_user_scaffold_provenance(prompt_row["store_id"])
        engine._last_compacted_store_id = max(
            row["store_id"]
            for row in engine._store.get_session_messages(engine._session_id)
        )
        mapped = engine._get_store_id_map_for_messages([active[1]])
    finally:
        engine.shutdown()

    assert mapped[id(active[1])] == prompt_row["store_id"]


def test_second_real_user_clears_retained_anchor_registration(tmp_path) -> None:
    engine = _engine(tmp_path, "retained-lineage-clear")
    initial = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "first reply"},
    ]
    later = [
        *initial,
        {"role": "user", "content": "second prompt"},
        {"role": "assistant", "content": "second reply"},
    ]

    try:
        engine.compress(initial)
        active = engine.compress(later)
        engine._last_compacted_store_id = max(
            row["store_id"]
            for row in engine._store.get_session_messages(engine._session_id)
        )
        mapped = engine._get_store_id_map_for_messages([active[1]])
        metadata = engine._store.read_metadata_json(
            engine._retained_user_anchor_metadata_key()
        )
    finally:
        engine.shutdown()

    assert metadata == {"store_id": 0, "version": 1}
    assert id(active[1]) not in mapped


def test_duplicate_active_identity_cannot_claim_registered_occurrence(tmp_path) -> None:
    engine = _engine(tmp_path, "retained-lineage-duplicate")
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "repeated prompt"},
        {"role": "assistant", "content": "initial reply"},
    ]

    try:
        engine.compress(messages)
        engine._last_compacted_store_id = max(
            row["store_id"]
            for row in engine._store.get_session_messages(engine._session_id)
        )
        first = {"role": "user", "content": "repeated prompt"}
        second = {"role": "user", "content": "repeated prompt"}
        mapped = engine._get_store_id_map_for_messages([first, second])
    finally:
        engine.shutdown()

    assert id(first) not in mapped
    assert id(second) not in mapped


def test_generated_scaffold_without_provenance_is_not_registered(tmp_path) -> None:
    engine = _engine(tmp_path, "retained-lineage-generated-scaffold")
    generated = {
        "role": "user",
        "content": (
            "[Current user objective preserved from compacted history]\n"
            "Generated context, not a user occurrence."
        ),
    }

    try:
        store_id = engine._store.append(engine._session_id, generated)
        has_provenance = engine._has_real_user_scaffold_provenance(store_id)
        registered = engine._prepare_retained_user_anchor(
            [
                {"role": "system", "content": "stable system"},
                generated,
            ]
        )
        metadata = engine._store.read_metadata_json(
            engine._retained_user_anchor_metadata_key()
        )
    finally:
        engine.shutdown()

    assert not has_provenance
    assert registered is None
    assert metadata == {"store_id": 0, "version": 1}
