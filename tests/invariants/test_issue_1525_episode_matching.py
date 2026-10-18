"""I1/I2: match each shortened-view occurrence once without losing new text."""
from __future__ import annotations

from copy import deepcopy
import sqlite3
import threading
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


@pytest.mark.parametrize("metadata", ["none", "uidmix", "stamphead"])
@pytest.mark.parametrize("steps", [
    [list(range(8)), list(range(4)), list(range(6)), [0, 1, 2, 3, 4, 6, 7, 5]],
    [list(range(8)), list(range(2)), list(range(5)), [0, 1, 2, 3, 5, 6, 7, 4, 8]],
    [list(range(8)), list(range(4)), list(range(6)), list(range(2)), list(range(8))],
    [list(range(8)), list(range(4)), list(range(8)), list(range(4)), list(range(8))],
    [list(range(8)), list(range(2)), list(range(5)), list(range(9))],
], ids=["reordered", "reordered-fresh", "nested", "repeated", "ordered-fresh"])
def test_nested_shrink_lookback(tmp_path: Path, metadata: str, steps: list[list[int]]) -> None:
    full = history()
    for i, row in enumerate(full):
        if metadata == "uidmix" and i % 2 == 0:
            row["message_uid"] = f"old-{i}"
        elif metadata == "stamphead" and i < 4:
            row["timestamp"] = float(10 + i)
    full.append({"role": "user", "content": "new after nested shrink"})
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        for step in steps:
            engine.ingest(deepcopy([full[i] for i in step]))
            assert ledger(engine) == texts(full[:9 if 8 in step else 8])


def test_stale_running_after_inner_bypass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = history("stamp")
    copies = [{"role": row["role"], "content": row["content"]} for row in full[4:]]
    fresh: Row = {"role": "user", "content": "fresh after bypass"}
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        real_count = engine._store.get_session_count
        calls = 0
        auxiliary = threading.local()

        def flaky_count(session_id: str) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return int(real_count(session_id))

        original_generation = type(engine)._in_process_auxiliary_caller_generation

        def generation(self: Any, session_id: str, **kwargs: Any) -> Any:
            return 1 if getattr(auxiliary, "active", False) else original_generation(self, session_id, **kwargs)

        monkeypatch.setattr(engine._store, "get_session_count", flaky_count)
        monkeypatch.setattr(type(engine), "_in_process_auxiliary_caller_generation", generation)
        errors: list[BaseException] = []

        def fork() -> None:
            auxiliary.active = True
            try:
                engine.compress([{"role": "user", "content": "fork row"}], current_tokens=100)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=fork)
        worker.start()
        worker.join()
        assert not errors
        assert calls >= 2
        assert engine._last_compression_status == "running"
        monkeypatch.setattr(engine._store, "get_session_count", real_count)
        engine.ingest(deepcopy(full))
        engine.ingest(deepcopy(full[:4]))
        engine.last_compression_status = ""
        engine.ingest(deepcopy(full[:4] + copies + [fresh]))
        assert ledger(engine) == texts(full + [fresh])


@pytest.mark.parametrize("raises", [False, True])
def test_compress_generation_marker_clears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raises: bool) -> None:
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        before = engine._compress_calls

        def compress_impl(messages: list[Row], **kwargs: Any) -> list[Row]:
            assert engine._active_compress_generation == before
            if raises:
                raise KeyboardInterrupt("synthetic compress exit")
            return messages

        monkeypatch.setattr(engine, "_compress_impl", compress_impl)
        if raises:
            with pytest.raises(KeyboardInterrupt):
                engine.compress([], current_tokens=100)
        else:
            engine.compress([], current_tokens=100)
        assert engine._active_compress_generation is None


@pytest.mark.parametrize(("metadata", "steps", "expected"), [
    ("none", [list(range(8)), list(range(6)), [0, 1, 2, 3, 4, 6, 7, 5]], 8),
    ("none", [list(range(8)), list(range(6)), [0, 1, 2, 3, 4, 6, 5, 7]], 8),
    ("none", [list(range(8)), list(range(10)), list(range(9)),
              [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10]], 11),
    ("stamphead", [list(range(8)), list(range(4)), [0, 1, 2, 3, 5, 4, 7, 6, 8, 9],
                   [0, 1, 2, 3, 5, 4, 7, 6, 8],
                   [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 10]], 11),
], ids=["N2", "N2d", "N6", "H3-stamped-head"])
def test_f5_lookback_all_positions(
    tmp_path: Path, metadata: str, steps: list[list[int]], expected: int,
) -> None:
    full = history()
    if metadata == "stamphead":
        for i in range(4):
            full[i]["timestamp"] = float(10 + i)
    full.extend({"role": "user", "content": f"fresh-{i}"} for i in (8, 9, 10))
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        for step in steps:
            engine.ingest(deepcopy([full[i] for i in step]))
        assert len(ledger(engine)) == expected
        assert ledger(engine) == texts(full[:expected])


@pytest.mark.parametrize("metadata", ["none", "stamp", "uid"])
def test_front_truncation_multiplicity_limitation(tmp_path: Path, metadata: str) -> None:
    full = history(metadata)
    fresh = deepcopy(full[1])
    if metadata == "stamp":
        fresh["timestamp"] = 100.0
    elif metadata == "uid":
        fresh["message_uid"] = "new-occurrence"
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        counts = []
        for view in (full, full[3:], full[3:] + [fresh]):
            engine.ingest(deepcopy(view))
            counts.append(len(ledger(engine)))
        # Documented multiplicity limitation: without occurrence metadata, the
        # new identical row is indistinguishable from the unmatched old row.
        assert counts == ([8, 8, 8] if metadata == "none" else [8, 8, 9])
        assert ledger(engine) == texts(full if metadata == "none" else full + [fresh])


@pytest.mark.parametrize("metadata", ["stamphead", "uidhead", "stamptail", "alt", "s0only"])
@pytest.mark.parametrize("fresh_metadata", ["none", "stamp", "uid"])
@pytest.mark.parametrize("shape", ["X1", "X4", "X4b", "X5", "noappend", "ordered"])
def test_mixed_episode_append_regrow(
    tmp_path: Path, metadata: str, fresh_metadata: str, shape: str,
) -> None:
    """I1/I2: an append must retain replay coverage for an unstamped tail."""
    masks = {"stamphead": "ssssnnnn", "uidhead": "uuuunnnn", "stamptail": "nnnnssss",
             "alt": "sunsunsu", "s0only": "snnnnnnn"}
    full = history()
    for i, (row, kind) in enumerate(zip(full, masks[metadata])):
        if kind == "s":
            row["timestamp"] = float(10 + i)
        elif kind == "u":
            row["message_uid"] = f"old-{i}"
    fresh: Row = {"role": "user", "content": "fresh-F"}
    if fresh_metadata == "stamp":
        fresh["timestamp"] = 500.0
    elif fresh_metadata == "uid":
        fresh["message_uid"] = "new-F"
    short = full[:4]
    appended = short + [fresh]
    views = {
        "X1": [full, short, appended, appended + full[4:]],
        "X4": [full, short, appended, full + [fresh]],
        "X4b": [full, short, appended, full],
        "X5": [full, short, appended, appended + full[4:6]],
        "noappend": [full, short, short, full + [fresh]],
        "ordered": [full, full[:2], full[:5], full + [fresh]],
    }
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        for view in views[shape]:
            engine.ingest(deepcopy(view))
        stored = ledger(engine)
        assert len(stored) == 9
        assert len(set(stored)) == 9
        assert stored == texts(full + [fresh])
