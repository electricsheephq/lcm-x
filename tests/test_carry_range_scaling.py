"""Scaling regressions for proof-backed carry-range publication (#495)."""

from __future__ import annotations

import os

import hermes_lcm.engine as lcm_engine_module
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens
from tests.test_compression_boundary import _stub_summarizer


pytestmark = pytest.mark.skipif(
    os.environ.get("LCM_SCALE_TESTS") != "1",
    reason="opt-in: set LCM_SCALE_TESTS=1",
)


def _message(role: str, index: int, padding_words: int) -> dict[str, str]:
    return {
        "role": role,
        "content": f"{role} {index}: note {index}" + (" word" * padding_words),
    }


def _engine(tmp_path, monkeypatch, **config_overrides) -> LCMEngine:
    context_length = config_overrides.pop("context_length", 400_000)
    monkeypatch.setattr(
        lcm_engine_module,
        "summarize_with_escalation",
        _stub_summarizer(),
    )
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "externalized"),
            **config_overrides,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start("S0", platform="acp", context_length=context_length)
    return engine


def test_fresh_tail_1500_publishes_three_in_place_cycles(tmp_path, monkeypatch) -> None:
    engine = _engine(tmp_path, monkeypatch, fresh_tail_count=1_500)
    live = []
    for index in range(1, 2_001):
        live.extend((_message("user", index, 20), _message("assistant", index, 20)))
    live.append(_message("user", 2_001, 20))
    engine.ingest(live)
    next_index = 2_001
    try:
        for _cycle in range(3):
            result = engine.compress(list(live), current_tokens=300_000)
            assert engine._last_compression_status == "compacted"
            assert len(engine._compress_commit_proof["carry_ranges"]) < 200
            engine.on_session_start(
                "S0",
                boundary_reason="compression",
                old_session_id="S0",
                platform="acp",
            )
            live = list(result) + [_message("assistant", next_index, 20)]
            for index in range(next_index + 1, next_index + 601):
                live.extend((_message("user", index, 20), _message("assistant", index, 20)))
            next_index += 601
            engine.ingest(live)
    finally:
        engine.shutdown()


def test_dynamic_40k_leaf_chunk_publishes_three_in_place_cycles(
    tmp_path, monkeypatch
) -> None:
    engine = _engine(
        tmp_path,
        monkeypatch,
        dynamic_leaf_chunk_enabled=True,
        leaf_chunk_tokens=40_000,
        context_length=120_000,
    )
    live = []
    for index in range(1, 1_501):
        live.extend((_message("user", index, 18), _message("assistant", index, 18)))
    live.append(_message("user", 1_501, 18))
    average_tokens = sum(count_message_tokens(row) for row in live) / len(live)
    assert 30 <= average_tokens <= 40
    engine.ingest(live)
    next_index = 1_501
    try:
        for _cycle in range(3):
            result = engine.compress(list(live), current_tokens=110_000)
            assert engine._last_compression_status == "compacted"
            assert len(engine._compress_commit_proof["carry_ranges"]) < 200
            engine.on_session_start(
                "S0",
                boundary_reason="compression",
                old_session_id="S0",
                platform="acp",
            )
            live = list(result) + [_message("assistant", next_index, 18)]
            for index in range(next_index + 1, next_index + 601):
                live.extend((_message("user", index, 18), _message("assistant", index, 18)))
            next_index += 601
            engine.ingest(live)
    finally:
        engine.shutdown()
