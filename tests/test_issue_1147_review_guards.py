"""Round-four proofs: occurrence-bound rescue and per-leaf frontier mapping."""
from pathlib import Path
from typing import Any

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import make_engine, pin_char_counter


Message = dict[str, Any]
SUMMARY = "Earlier turns.\nExpand for details about: turns"


def _row(label: str, size: int = 1200) -> Message:
    return {"role": "user", "content": label + " " + "word " * (size // 5)}


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="pre-existing occurrence remap: follow-up to PR #1147")
def test_rescue_must_not_borrow_retained_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed surplus prefix must not claim the durable suffix's occurrence."""
    pin_char_counter(monkeypatch)  # type: ignore[no-untyped-call]
    engine = make_engine(tmp_path, window=128_000, conversation_id="review-rescue",
                         embeddings_enabled=False, temporal_rollups_enabled=False)
    prefix = _row("duplicate", 40_000)
    suffix = dict(prefix)
    tail = [_row(f"tail-{n}", 80) for n in range(24)]
    view = [prefix, _row("unstored-prefix", 20_000), suffix, *tail]
    calls: list[str] = []

    def summarize(**kwargs: Any) -> tuple[str, int]:
        calls.append(kwargs["text"])
        if len(calls) == 1:
            raise RuntimeError("context length exceeded")
        return SUMMARY, 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    try:
        # A resumed host view can include carry-over occurrences absent from this session's store.
        engine._ingest_messages([suffix, *tail])
        engine._ingest_cursor = len(view)
        full_map = engine._get_store_id_map_for_messages(view)
        assert id(prefix) not in full_map and id(suffix) in full_map
        before = engine._store_complete_frontier()
        rows = engine._store.get_session_messages(engine._session_id)
        # The host count includes prompt/tool overhead, as the ContextEngine contract permits.
        engine.compress(view, current_tokens=100_000)
        assert engine._store.get_session_messages(engine._session_id) == rows
        assert engine._compress_host_rows_consumed == 1  # the later duplicate remains in the working suffix
        assert engine._store_complete_frontier() == before, (
            "rescue borrowed the retained occurrence", len(calls),
            [(node.source_ids, node.summary) for node in engine._dag.get_session_nodes(engine._session_id)],
        )
        assert len(calls) == 1
        assert engine._dag.get_session_nodes(engine._session_id) == []
        assert engine._store.get_session_messages(engine._session_id) == rows
    finally:
        engine.shutdown()


def test_later_leaf_rebuilds_map_above_published_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real sweep rebuilds mapping after each advance, before its next admission."""
    pin_char_counter(monkeypatch)  # type: ignore[no-untyped-call]
    engine = make_engine(tmp_path, window=128_000, conversation_id="review-frontier",
                         embeddings_enabled=False, temporal_rollups_enabled=False)
    view = [_row(f"row-{n}", 8000) for n in range(80)]
    observed: list[tuple[int, set[int]]] = []

    def summarize(**kwargs: Any) -> tuple[str, int]:
        frontier = engine._store_complete_frontier()
        mapped = set((engine._compress_run.store_ids_by_message_id or {}).values())
        if kwargs.get("depth") == 0:
            observed.append((frontier, mapped))
        return SUMMARY, 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    try:
        engine.ingest(view)
        rows = engine._store.get_session_messages(engine._session_id)
        engine.compress(view, current_tokens=count_messages_tokens(view))
        assert len(observed) >= 2
        assert observed[1][0] > observed[0][0]
        assert all(mapped and min(mapped) > frontier for frontier, mapped in observed), observed
        nodes = [node for node in engine._dag.get_session_nodes(engine._session_id) if node.depth == 0]
        assert len(nodes) == len(observed)
        claimed = [store_id for node in nodes for store_id in node.source_ids]
        assert len(claimed) == len(set(claimed))
        assert engine._store.get_session_messages(engine._session_id) == rows
    finally:
        engine.shutdown()
