"""Characterize per-call emission/anchor lifetimes before moving their storage.

The real engine owns assembly, ingestion, publication and cleanup (I2, I4,
I13). Only the summary provider and token counters are scripted.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import pytest

import hermes_lcm.survival_fit as survival_fit  # type: ignore[import-not-found]
import hermes_lcm.tokens as tokens  # type: ignore[import-not-found]
from hermes_lcm.engine import LCMEngine  # type: ignore[import-not-found]
from tests.harness import ScriptedSummarizer, engine_context, pin_char_counter

Rows = list[dict[str, Any]]


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    # The shared harness predates strict typing; keep that boundary explicit.
    cast(Callable[[pytest.MonkeyPatch], None], pin_char_counter)(monkeypatch)
    provider = ScriptedSummarizer()  # type: ignore[no-untyped-call]
    provider.install(monkeypatch)  # type: ignore[no-untyped-call]
    build = cast(Callable[..., AbstractContextManager[Any]], engine_context)
    with build(
        tmp_path, window=200_000, fresh_tail_count=2, leaf_chunk_tokens=1_000,
        fresh_tail_pressure_yield_enabled=False, embeddings_enabled=False,
        temporal_rollups_enabled=False, proactive_recall_enabled=False,
        large_output_externalization_enabled=True,
        large_output_active_replay_stubbing_enabled=True,
    ) as instance:
        instance.threshold_tokens = 12_000
        yield instance


def _compacted(engine: Any) -> Rows:
    rows: Rows = []
    for index in range(4):
        rows.extend([
            {"role": "user", "content": f"Request {index}: " + "source context " * 400},
            {"role": "assistant", "content": f"Reply {index}: " + "record details " * 400},
        ])
    rows.extend([{"role": "user", "content": "Keep the final request."},
                 {"role": "assistant", "content": "Final reply."}])
    result: Rows = engine.compress(rows, force=True)
    assert engine.last_compression_status == "compacted"
    assert len(result) < len(rows)
    assert engine._store.get_session_count(engine.current_session_id) == len(rows)
    return result


def test_direct_assembly_candidates_are_reset_by_next_compress(engine: Any) -> None:
    compacted = _compacted(engine)
    assembled = engine._assemble_context(None, compacted[-2:])
    pending = engine._pending_emission_candidates
    assert pending and all(any(item["span"] in row["content"] for row in assembled) for item in pending)
    before = engine._store.get_session_messages(engine.current_session_id)

    # The empty public call takes the real no-op path without another assembly
    # overwriting the channel and hiding a missing start-of-call reset.
    result = engine.compress([])

    assert result == []
    assert engine._pending_emission_candidates == []
    assert engine._pending_emission_candidates is not pending
    assert engine._compress_commit_proof is None
    assert engine._store.get_session_messages(engine.current_session_id) == before


@pytest.mark.parametrize("rejection", ["trial", "final"])
def test_stub_rejection_preserves_candidate_objects_and_order(
    engine: Any, monkeypatch: pytest.MonkeyPatch, rejection: str,
) -> None:
    rows: Rows = [{"role": "user", "content": "A fresh request."},
                  {"role": "assistant", "content": "A fresh reply."}]
    working = engine._ingest_messages(rows)
    engine._compress_occurrences = None
    # Two distinguishable caller-owned records expose loss, reordering and deep copies.
    pending = [{"kind": "summary", "span": "first"}, {"kind": "objective", "span": "second"}]
    engine._pending_emission_candidates = pending
    before = engine._store.get_session_messages(engine.current_session_id)

    def host_estimate(_rows: Rows) -> int:
        # Trial assembly leaves the channel alone; final assembly publishes a
        # replacement. Model a host token estimate that rejects that final view.
        return 20_000 if rejection == "trial" or engine._pending_emission_candidates is not pending else 0

    monkeypatch.setattr(survival_fit, "_host_estimate", host_estimate)
    result = engine._stub_first_exit(rows, working, list(working), working, 0)

    assert result is None
    restored = engine._pending_emission_candidates
    assert len(restored) == len(pending)
    assert all(actual is expected for actual, expected in zip(restored, pending))
    assert (restored is pending) == (rejection == "trial")
    assert engine._store.get_session_messages(engine.current_session_id) == before


@pytest.mark.parametrize("path", ["committed", "compress_impl"])
def test_anchor_is_cleared_after_assembly_exception_and_retry(
    engine: Any, monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    if path == "committed":
        compacted: Rows = [{"role": "user", "content": "Keep the final request."},
                           {"role": "assistant", "content": "Final reply."}]
        engine.ingest(compacted)
    else:
        compacted = _compacted(engine)
    before = engine._store.get_session_messages(engine.current_session_id)
    count = tokens._count_tokens_cached
    observed: list[Rows] = []

    def fail_anchored_count(text: str, generation: int) -> int:
        anchor = engine._pending_context_anchor_messages
        if anchor is not None:
            observed.append(list(anchor))
            raise RuntimeError("tokenizer rejected anchored assembly")
        return int(count(text, generation))

    def assemble() -> Rows:
        if path == "committed":
            return cast(Rows, engine._assemble_committed_compaction_context(
                compacted[-2:], compacted, None))
        # Replaying the already committed prefix drops its scaffold with no new
        # leaf, reaching _compress_impl's separate anchored-assembly finally.
        return cast(Rows, engine.compress(compacted, force=True))

    assert engine._pending_context_anchor_messages is None
    with monkeypatch.context() as patch:
        patch.setattr(tokens, "_count_tokens_cached", fail_anchored_count)
        with pytest.raises(RuntimeError, match="tokenizer rejected anchored assembly"):
            assemble()
    # Exactly one anchored assembly, anchored on exactly the caller's rows in order.
    assert observed == [compacted]
    assert engine._pending_context_anchor_messages is None
    assert engine._store.get_session_messages(engine.current_session_id) == before

    result = assemble()
    assert result and result[-1]["content"] == "Final reply."
    assert engine._pending_context_anchor_messages is None
    assert engine._store.get_session_messages(engine.current_session_id) == before


@pytest.mark.parametrize("channel", ["_pending_emission_candidates", "_pending_context_anchor_messages"])
def test_uninitialized_engine_channel_reads_raise_attribute_error(channel: str) -> None:
    engine = LCMEngine.__new__(LCMEngine)
    with pytest.raises(AttributeError, match=channel):
        getattr(engine, channel)
    assert getattr(engine, channel, None) is None
