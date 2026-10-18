"""I1/I2: match each shortened-view occurrence once without losing new text."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests import harness

Row = dict[str, Any]
engine_context: Any = harness.engine_context


def history(metadata: str = "none") -> list[Row]:
    rows: list[Row] = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"row-{i}"}
                       for i in range(8)]
    for i, row in enumerate(rows):
        if metadata == "uid":
            row["message_uid"] = f"old-{i}"
        elif metadata == "stamp":
            row["timestamp"] = float(10 + i)
    return rows


def ledger(engine: Any) -> list[tuple[str, str]]:
    return [(row["role"], row["content"]) for row in engine._store.get_session_messages("S")]


def texts(rows: list[Row]) -> list[tuple[str, str]]:
    return [(row["role"], row["content"]) for row in rows]


@pytest.mark.parametrize("shape", ["restore", "append", "compress"])
def test_unstamped_reordered_tail(tmp_path: Path, shape: str) -> None:
    full = history()
    short = full[:6] if shape == "compress" else full[:4]
    restored = short + ([full[7], full[6]] if shape == "compress" else
                        [full[5], full[4], full[6], full[7]])
    fresh: Row = {"role": "user", "content": "new after reorder"}
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        if shape == "compress":
            engine.compress(deepcopy(short), current_tokens=100, force=True)
        else:
            engine.ingest(deepcopy(short))
        if shape == "restore":
            engine.ingest(deepcopy(restored))
            assert ledger(engine) == texts(full)
        engine.ingest(deepcopy(restored + [fresh]))
        assert ledger(engine) == texts(full + [fresh])


def test_equal_uid_different_text_survives(tmp_path: Path) -> None:
    full = history("uid")
    first: Row = {"role": "user", "content": "first new text", "message_uid": "fresh-0"}
    second = {**first, "content": "second new text"}
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        for view in (full, full[:4], full[:4] + [first], full[:4], full, full + [second]):
            engine.ingest(deepcopy(view))
        assert ledger(engine) == texts(full + [first, second])


@pytest.mark.parametrize("keep", [6, 7])
@pytest.mark.parametrize("at_threshold", [False, True])
def test_episode_opened_inside_compress(tmp_path: Path, keep: int, at_threshold: bool) -> None:
    full = history("stamp")
    resend = [{"role": r["role"], "content": r["content"]} for r in full[6:]]
    fresh: Row = {"role": "user", "content": "after rejected compression"}
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        engine.record_rejected_compaction()
        view = full[:6] + resend[:keep - 6]
        output = engine.compress(deepcopy(view), current_tokens=engine.threshold_tokens + 1 if at_threshold else 100)
        final = output + resend[keep - 6:]
        engine.ingest(deepcopy(final))
        engine.ingest(deepcopy(final + [fresh]))
        assert ledger(engine) == texts(full + resend[keep - 6:] + [fresh])


@pytest.mark.parametrize("extra", [1, 2])
def test_unstamped_absorption_cap(tmp_path: Path, extra: int) -> None:
    full = history()
    # The tail contains one occurrence; each extra host occurrence must survive.
    restored = full[:4] + [full[6], full[4], full[5], full[7]] + [full[4]] * extra
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        engine.ingest(deepcopy(full[:4]))
        engine.ingest(deepcopy(restored))
        assert ledger(engine) == texts(full + [full[4]] * extra)


def test_unstamped_prefix_repeat_is_fresh(tmp_path: Path) -> None:
    full = history()
    # This text occurs only before the shrink boundary, never in its old tail.
    fresh = deepcopy(full[0])
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        engine.ingest(deepcopy(full[:4]))
        engine.ingest(deepcopy(full[:4] + [fresh]))
        assert ledger(engine) == texts(full + [fresh])


@pytest.mark.parametrize("shape", ["pruned", "salvage", "literal", "composite"])
def test_uid_rewrite_keeps_one_copy(tmp_path: Path, shape: str) -> None:
    full = history("uid")
    cleared = "[Old tool output cleared to save context space]"
    if shape in {"pruned", "salvage"}:
        full[4] = {**full[4], "role": "assistant", "content": "", "tool_calls": [
            {"id": "older", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}
        full[5] = {**full[5], "role": "tool", "tool_call_id": "older", "tool_name": "lookup",
                   "content": "older output " * 40}
    elif shape == "literal":
        full[4]["content"] = ("[Externalized payload: kind=raw_payload; role=user; chars=143774; "
                              "bytes=143774; ref=raw_payload_literal]")
    else:
        full[4]["content"], full[5]["content"] = "T33 pending\n\nT34 completed", "T34 completed"
    with engine_context(tmp_path, window=128000, conversation_id="conv", embeddings_enabled=False,
                        large_output_externalization_enabled=False,
                        large_output_active_replay_stubbing_enabled=False) as engine:
        engine.ingest(full)
        before = ledger(engine)
        identities = [engine._message_replay_identity(row, strip_carrier=False) for row in full]
        restored = full if shape == "pruned" else deepcopy(full)
        if shape in {"pruned", "salvage"}:
            restored[5]["content"] = cleared
        engine.ingest(deepcopy(full[:4]))
        assert [engine._message_replay_identity(row, strip_carrier=False) for row in restored] == identities
        engine.ingest(restored)
        assert ledger(engine) == before
