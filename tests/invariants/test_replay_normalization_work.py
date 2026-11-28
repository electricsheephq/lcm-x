"""Missing persisted outputs normalize each unique replay identity once per scan (#1090)."""

from collections import Counter
from copy import deepcopy

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize("n", [16, 32])
def test_generationless_normalization_is_linear_and_call_local(tmp_path, monkeypatch, n):
    """Normalize each unique identity once per scan without reusing a prior scan's cache."""
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        embeddings_enabled=False, temporal_rollups_enabled=False,
        empty_lifecycle_gc_enabled=False,
    ), hermes_home=str(tmp_path))
    engine.on_session_start("synthetic-normalization")
    marker = (
        "<persisted-output>\n"
        "This tool result was too large (100 characters, 0.1 KB).\n"
        f"Full output saved to: {tmp_path / 'missing.txt'}\n"
        "Preview (first 7 chars):\npreview\n...\n"
        "</persisted-output>"
    )
    messages = [{"role": "tool", "tool_call_id": "call-missing", "content": marker}] + [
        {"role": "user", "content": f"synthetic row {i}"} for i in range(n - 1)
    ]
    before = deepcopy(messages)
    tail = [("user", "unrelated durable row", "", "", "")]
    calls = Counter()
    original = engine._strip_inline_persisted_output_generation_identity

    def counted(identity):
        calls[identity] += 1
        return original(identity)

    monkeypatch.setattr(engine, "_strip_inline_persisted_output_generation_identity", counted)
    try:
        for _ in range(2):
            calls.clear()
            assert engine._find_reconciled_cursor_for_store_tail(
                messages, tail, allow_empty_prefix=True, session_count=1, raw_session_count=1,
            ) == 0
            assert messages == before
            assert len(calls) == n + 1
            assert max(calls.values()) == 1
    finally:
        engine.shutdown()
