"""Synthetic proof for host-owned LCM invocation cancellation and deadlines."""

from __future__ import annotations

import threading
import time

import pytest

import hermes_lcm.compaction as lcm_compaction
import hermes_lcm.engine as lcm_engine

try:
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
except ImportError:
    AuxiliaryExplicitCancellation = (
        lcm_compaction._StandaloneAuxiliaryExplicitCancellation
    )

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


class _SyntheticFence:
    """Small stand-in for CompressionCommitFence's public worker contract."""

    def __init__(self, deadline_monotonic: float | None = None) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._deadline = deadline_monotonic
        self.begin_calls = 0
        self.finish_calls = 0
        self.held = False

    @property
    def deadline_monotonic(self) -> float | None:
        return self._deadline

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def begin_lock_setup(self) -> bool:
        self._lock.acquire()
        self.begin_calls += 1
        self.held = True
        if self._cancelled:
            self.finish_lock_setup()
            return False
        return True

    def finish_lock_setup(self) -> None:
        self.finish_calls += 1
        self.held = False
        self._lock.release()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True


@pytest.fixture
def engine(tmp_path):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "invocation-fence.db"),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "hermes-home"),
    )
    engine.on_session_start(
        "synthetic-session",
        platform="cli",
        conversation_id="synthetic-conversation",
        context_length=200_000,
    )
    try:
        yield engine
    finally:
        engine.shutdown()


def _install_fence(engine, fence: _SyntheticFence) -> None:
    engine._compression_publication_fence = fence
    engine._compression_cancelled_check = lambda: fence.is_cancelled


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": "synthetic prior turn"},
        {"role": "user", "content": "synthetic current turn"},
        {"role": "user", "content": "synthetic fresh tail"},
    ]


def _base_nodes(engine) -> list[SummaryNode]:
    nodes = []
    for index in range(2):
        node = SummaryNode(
            session_id="synthetic-session",
            depth=0,
            summary=f"synthetic base {index}",
            token_count=2,
            source_token_count=3,
            source_ids=[],
            source_type="messages",
            created_at=float(index + 1),
            earliest_at=float(index + 1),
            latest_at=float(index + 1),
        )
        engine._dag.add_node(node)
        nodes.append(node)
    return nodes


def test_cancel_during_summary_denies_leaf_and_preserves_sources(
    engine, monkeypatch
):
    fence = _SyntheticFence(time.monotonic() + 600)
    _install_fence(engine, fence)

    def summarize(**_kwargs):
        fence.cancel()
        return "synthetic leaf summary", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)

    with pytest.raises(AuxiliaryExplicitCancellation):
        engine.compress(_messages())

    assert engine._dag.get_session_nodes("synthetic-session") == []
    state = engine._lifecycle.get_by_conversation("synthetic-conversation")
    assert state.current_frontier_store_id == 0
    assert len(engine._store.get_session_messages("synthetic-session")) == 3
    assert fence.begin_calls == 1
    assert fence.finish_calls == 1


def test_uncancelled_compress_publishes_inside_fence(engine, monkeypatch):
    fence = _SyntheticFence(time.monotonic() + 600)
    _install_fence(engine, fence)
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **_kwargs: ("synthetic leaf summary", 1),
    )

    result = engine.compress(_messages())

    assert result
    assert len(engine._dag.get_session_nodes("synthetic-session")) == 1
    state = engine._lifecycle.get_by_conversation("synthetic-conversation")
    assert state.current_frontier_store_id > 0
    assert fence.begin_calls == 1
    assert fence.finish_calls == 1
    assert fence.held is False


def test_condensation_cancellation_denies_late_result(engine, monkeypatch):
    base_nodes = _base_nodes(engine)

    fence = _SyntheticFence(time.monotonic() + 600)
    _install_fence(engine, fence)

    def summarize(**_kwargs):
        fence.cancel()
        return "synthetic condensed summary", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)

    token = lcm_compaction._COMPRESSION_INVOCATION.set(
        engine._capture_compression_invocation()
    )
    try:
        with pytest.raises(AuxiliaryExplicitCancellation):
            engine._condense_summary_nodes(base_nodes)
    finally:
        lcm_compaction._COMPRESSION_INVOCATION.reset(token)

    nodes = engine._dag.get_session_nodes("synthetic-session")
    assert [node.node_id for node in nodes] == [node.node_id for node in base_nodes]
    assert fence.begin_calls == 1
    assert fence.finish_calls == 1


def test_one_captured_deadline_reaches_leaf_rescues_and_condensation(
    engine, monkeypatch
):
    deadline = time.monotonic() + 600
    fence = _SyntheticFence(deadline)
    _install_fence(engine, fence)
    captured = engine._capture_compression_invocation()
    token = lcm_compaction._COMPRESSION_INVOCATION.set(captured)
    leaf_deadlines = []
    condensation_deadlines = []
    calls = 0

    def summarize(**kwargs):
        nonlocal calls
        calls += 1
        if kwargs["depth"] == 0:
            leaf_deadlines.append(kwargs["deadline"])
            if len(leaf_deadlines) < 3:
                raise TimeoutError("synthetic timeout")
        else:
            condensation_deadlines.append(kwargs["deadline"])
        return "synthetic summary", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    try:
        engine._summarize_leaf_chunk_with_rescue(_messages() + [
            {"role": "assistant", "content": "synthetic extra turn"},
            {"role": "user", "content": "synthetic second extra"},
            {"role": "assistant", "content": "synthetic third extra"},
        ])
        base_nodes = _base_nodes(engine)
        engine._condense_summary_nodes(base_nodes)
    finally:
        lcm_compaction._COMPRESSION_INVOCATION.reset(token)

    assert calls == 4
    assert len(leaf_deadlines) == 3
    assert all(value == deadline for value in leaf_deadlines)
    assert condensation_deadlines == [deadline]
    assert fence.begin_calls == 1
    assert fence.finish_calls == 1


@pytest.mark.parametrize("host_budget, maximum", [(1000.0, 600.0), (90.0, 90.0)])
def test_threshold_sweep_uses_shared_ceiling_and_honors_lower_host_deadline(
    engine, monkeypatch, host_budget, maximum
):
    engine._config.threshold_full_sweep_enabled = True
    engine.threshold_tokens = 1
    started = time.monotonic()
    fence = _SyntheticFence(started + host_budget)
    _install_fence(engine, fence)
    deadlines = []
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **kwargs: (deadlines.append(kwargs["deadline"]) or "synthetic summary", 1),
    )

    engine.compress(_messages())

    assert deadlines
    assert deadlines[0] - started <= maximum + 0.1
    if maximum == 600.0:
        assert deadlines[0] - started > 120.0


def test_invocation_snapshot_survives_newer_host_generation(engine):
    first_fence = _SyntheticFence(time.monotonic() + 600)
    second_fence = _SyntheticFence(time.monotonic() + 900)
    def first_check():
        return False

    def second_check():
        return True

    engine._compression_publication_fence = first_fence
    engine._compression_cancelled_check = first_check

    captured = engine._capture_compression_invocation()
    engine._compression_publication_fence = second_fence
    engine._compression_cancelled_check = second_check

    assert captured.publication_fence is first_fence
    assert captured.cancel_check is first_check
    assert captured.deadline_monotonic == first_fence.deadline_monotonic
    assert captured.is_cancelled() is False
