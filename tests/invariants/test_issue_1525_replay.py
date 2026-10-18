"""Independent occurrence-ledger and continuation probes for REV1525."""
from __future__ import annotations

import importlib
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests import harness

engine_context: Any = harness.engine_context



def msg(text: str, stamp: float, role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": text, "timestamp": stamp}


def full_list() -> list[dict[str, Any]]:
    return [msg(f"row-{i}", float(i + 10), "user" if i % 2 == 0 else "assistant") for i in range(8)]


def lineage(root: Path) -> None:
    (root / "home").mkdir(exist_ok=True)
    with sqlite3.connect(root / "home/state.db") as db:
        db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
        db.executemany("INSERT INTO sessions VALUES (?, ?, ?)", [("S", None, "compression"), ("child", "S", None)])


def rows(engine: Any, sid: str = "S") -> list[tuple[Any, ...]]:
    return [(r["role"], r["content"], r["observed_at"]) for r in engine._store.get_session_messages(sid)]


def expected(view: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [(r["role"], r["content"], r.get("timestamp")) for r in view]




@pytest.mark.parametrize("side", ["front", "tail"])
@pytest.mark.parametrize("repeat_short", [False, True])
def test_cold_between_shrink_and_append(tmp_path: Path, side: str, repeat_short: bool) -> None:
    lineage(tmp_path)
    full = full_list()
    short = full[-4:] if side == "front" else full[:4]
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        engine.ingest(deepcopy(short))
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        if repeat_short:
            engine.ingest(deepcopy(short))
        fresh = msg("after cold", 100.)
        engine.ingest(deepcopy(short + [fresh]))
        assert rows(engine) == expected(full + [fresh])


@pytest.mark.parametrize("shape", ["longer", "same", "shorter"])
def test_middle_insert(tmp_path: Path, shape: str) -> None:
    lineage(tmp_path)
    full = full_list()
    inserted = msg("steer in middle", 100.)
    changed = full[:2] + [inserted] + full[2:]
    if shape == "same":
        changed = changed[:-1]
    if shape == "shorter":
        changed = changed[:-4]
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        engine.ingest(deepcopy(changed))
        assert rows(engine) == expected(full + [inserted])
        fresh = msg("after steer", 101.)
        engine.ingest(deepcopy(changed + [fresh]))
        assert rows(engine) == expected(full + [inserted, fresh])
        engine.ingest(deepcopy(changed + [fresh]))
        assert rows(engine) == expected(full + [inserted, fresh])


@pytest.mark.parametrize("restamp", [False, True])
def test_reexpand_stamps(tmp_path: Path, restamp: bool) -> None:
    lineage(tmp_path)
    full = full_list()
    restored = deepcopy(full)
    if restamp:
        for row in restored[4:]:
            row["timestamp"] += 100.
    fresh = msg("after reexpand", 200.)
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        engine.ingest(deepcopy(full[:4]))
        engine.ingest(restored + [fresh])
        ledger = rows(engine)
        # New known stamps are new occurrences; identical stamps are replay.
        assert ledger == expected(full + (restored[4:] if restamp else []) + [fresh])
        engine.ingest(deepcopy(restored + [fresh]))
        assert rows(engine) == ledger


@pytest.mark.parametrize("shrink_first", [False, True])
def test_rotation_after_shrink(tmp_path: Path, shrink_first: bool) -> None:
    lineage(tmp_path)
    full = full_list()
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        if shrink_first:
            engine.ingest(deepcopy(full[:4]))
        parent = rows(engine)
        engine.on_session_start("child", platform="cli", context_length=128000,
                                conversation_id="conv", boundary_reason="compression")
        assert engine._ingest_cursor == 0
        fresh = msg("in child", 100.)
        view = full[:4] + [fresh]
        engine.ingest(deepcopy(view))
        assert rows(engine) == parent
        assert rows(engine, "child") == expected([fresh])
        assert engine._ingest_cursor == len(view)
        engine.ingest(deepcopy(view))
        assert rows(engine, "child") == expected([fresh])


def test_sliding_window_six_turns(tmp_path: Path) -> None:
    lineage(tmp_path)
    full = full_list()
    ledger = deepcopy(full)
    window = deepcopy(full)
    observations = []
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(window)
        for turn in range(6):
            window = window[2:]
            engine.ingest(deepcopy(window))
            assert len(engine._last_active_replay_source_identities) == len(window)
            fresh = [msg(f"sliding-{turn}", 100. + 2 * turn), msg(f"reply-{turn}", 101. + 2 * turn, "assistant")]
            window += fresh
            ledger += fresh
            engine.ingest(deepcopy(window))
            observations.append({"turn": turn, "cursor": engine._ingest_cursor, "rows": rows(engine)})
        assert rows(engine) == expected(ledger)


def test_tool_rows_only_drop(tmp_path: Path) -> None:
    lineage(tmp_path)
    full = full_list()
    full[1] = {**full[1], "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}
    full.insert(2, {**msg("tool result", 11.5, "tool"), "tool_call_id": "call-1", "name": "lookup"})
    short = [r for r in full if r["role"] != "tool"]
    fresh = msg("after tool drop", 100.)
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        engine.ingest(deepcopy(short))
        engine.ingest(deepcopy(short + [fresh]))
        assert rows(engine) == before + expected([fresh])


@pytest.mark.parametrize("mode", ["a-short", "b-full", "c-cold-full", "d-short-full"])
def test_hidden_golden_continuation(tmp_path: Path, monkeypatch: Any, mode: str) -> None:
    from tests.behaviour import compress_trace_harness as h
    original_context: Any = getattr(h, "engine_context")
    full = h.transcript(next(c for c in h.SCENARIOS if c.name == "deadline-hidden-backlog-read"))
    record: dict[str, Any] = {}

    def continue_with(engine: Any) -> None:
        before = rows(engine)
        record["before"] = before
        record["cursor_before"] = engine._ingest_cursor
        if mode == "d-short-full":
            engine.ingest(deepcopy(full[-2:]))
            record["repeated_short_cursor"] = engine._ingest_cursor
        fresh = msg("unique continuation", 1000.)
        view = full[-2:] if mode == "a-short" else full
        engine.ingest(deepcopy(view + [fresh]))
        record["after"] = rows(engine)
        record["cursor_after"] = engine._ingest_cursor
        record["fresh_count"] = rows(engine).count(expected([fresh])[0])
        record["surplus"] = list((Counter(rows(engine)) - Counter(before) - Counter(expected([fresh]))).elements())
        assert len(before) == 17
        assert len(rows(engine)) == 18
        assert record["fresh_count"] == 1
        assert record["surplus"] == []
        assert rows(engine) == before + expected([fresh])

    @contextmanager
    def capture(root: Path, **settings: Any) -> Any:
        with original_context(root, **settings) as engine:
            yield engine
            record["post_golden_cursor"] = engine._ingest_cursor
            record["post_golden_rows"] = rows(engine)
            if mode == "c-cold-full":
                engine.shutdown()
                with original_context(root, **settings) as restarted:
                    continue_with(restarted)
            else:
                continue_with(engine)

    monkeypatch.setattr(h, "engine_context", capture)
    case = next(c for c in h.SCENARIOS if c.name == "deadline-hidden-backlog-read")
    h.observe(case, tmp_path)


def test_ordinary_sql(tmp_path: Path) -> None:
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        full = full_list()
        engine.ingest(full[:2])
        statements: list[str] = []
        engine._store.connection.set_trace_callback(statements.append)
        engine.ingest(full[:4])
        engine._store.connection.set_trace_callback(None)
        assert len(statements) == 50
        assert len(rows(engine)) == 4


def test_unstamped_compressed_prefix_then_shrink(tmp_path: Path, monkeypatch: Any) -> None:
    product = importlib.import_module("hermes_lcm.engine")
    from tests.test_issue_436_identity_anchor import _turns

    lineage(tmp_path)
    monkeypatch.setattr(product, "summarize_with_escalation",
                        lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1))
    original = _turns(1, 8, 0.)
    for row in original:
        row.pop("timestamp")
    with engine_context(tmp_path, window=200000, conversation_id="conv",
                        fresh_tail_count=2, leaf_chunk_tokens=1) as engine:
        engine.ingest(deepcopy(original))
        before = Counter((r[0], r[1]) for r in rows(engine))
        output = engine.compress(deepcopy(original), current_tokens=engine.threshold_tokens + 1, force=True)
        assert engine._last_compression_status == "compacted"
        assert len(output) < len(original)
        engine.on_session_end("S", original)
        engine.on_session_start("S", boundary_reason="compression", old_session_id="S",
                                platform="cli", context_length=200000, conversation_id="conv")
        engine.ingest(deepcopy(output))
        short = output[:-1]
        engine.ingest(deepcopy(short))
        fresh = msg("after unstamped compressed prefix", 1000.)
        engine.ingest(deepcopy(short + [fresh]))
        assert Counter((r[0], r[1]) for r in rows(engine)) == before + Counter([(fresh["role"], fresh["content"])])
        assert rows(engine).count(expected([fresh])[0]) == 1


def test_derive_golden(tmp_path: Path, monkeypatch: Any) -> None:
    from tests.behaviour import compress_trace_harness as h
    states = []
    state = h.state

    def capture(engine: Any, canonical: Any) -> Any:
        value = state(engine, canonical)
        states.append(value)
        return value

    events = {}
    for case in h.SCENARIOS:
        states.clear()
        with monkeypatch.context() as patch:
            if case.name == "deadline-hidden-backlog-read":
                patch.setattr(h, "state", capture)
            events[case.name] = h.observe(case, tmp_path / case.name)
    assert events == json.loads(h.GOLDEN.read_text())


def test_partial_noop_restore_unstamped(tmp_path: Path) -> None:
    """The supported partial-compress caller restores its withheld tail after a no-op."""
    lineage(tmp_path)
    full = full_list()
    for row in full:
        if row["role"] == "assistant":
            row.pop("timestamp")
    fresh = msg("after partial no-op", 1000.)
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        head, tail = deepcopy(full[:-2]), deepcopy(full[-2:])
        returned = engine.compress(head, current_tokens=100, force=True)
        assert returned == head
        assert engine._last_compression_status != "compacted"
        assert engine.current_session_id == "S"
        engine.ingest(returned + tail + [fresh])
        after = rows(engine)
        assert len(before) == 8 and len(after) == 9
        assert after == before + expected([fresh])


def test_actual_compression_rotation_after_shrink(tmp_path: Path, monkeypatch: Any) -> None:
    product = importlib.import_module("hermes_lcm.engine")
    from tests.test_issue_436_identity_anchor import _turns

    lineage(tmp_path)
    monkeypatch.setattr(product, "summarize_with_escalation",
                        lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1))
    full = _turns(1, 8, 0.)
    with engine_context(tmp_path, window=200000, conversation_id="conv",
                        fresh_tail_count=2, leaf_chunk_tokens=1) as engine:
        engine.ingest(deepcopy(full))
        short = deepcopy(full[:12])
        engine.ingest(short)
        before = rows(engine)
        output = engine.compress(short, current_tokens=engine.threshold_tokens + 1, force=True)
        assert engine._last_compression_status == "compacted"
        engine.on_session_end("S", short)
        engine.on_session_start("child", platform="cli", context_length=200000,
                                conversation_id="conv", boundary_reason="compression", old_session_id="S")
        assert engine._ingest_cursor == len(output)
        fresh = msg("after actual rotation", 1000.)
        engine.ingest(deepcopy(output + [fresh]))
        assert rows(engine) == before
        assert rows(engine, "child") == expected([fresh])


@pytest.mark.parametrize("metadata", ["timestamp", "message_uid", "unstamped"])
@pytest.mark.parametrize("together", [False, True])
def test_undo_resend_same_text(tmp_path: Path, metadata: str, together: bool) -> None:
    """Occurrence keys distinguish an identical resend from restored text."""
    full = full_list()
    for index, row in enumerate(full):
        if metadata != "timestamp":
            row.pop("timestamp")
        if metadata == "message_uid":
            row["message_uid"] = f"old-{index}"
    fresh = [msg(full[-2]["content"], 100.), msg("different reply", 101., "assistant")]
    for index, row in enumerate(fresh):
        if metadata != "timestamp":
            row.pop("timestamp")
        if metadata == "message_uid":
            row["message_uid"] = f"new-{index}"
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        engine.ingest(deepcopy(full[:-2]))
        if not together:
            engine.ingest(deepcopy(full[:-2] + fresh[:1]))
            assert rows(engine) == before + expected([] if metadata == "unstamped" else fresh[:1])
        # Identical unstamped text resent after undo is stored once until hosts
        # send occurrence keys; its text survives but its occurrence time is lost.
        stored_fresh = fresh[1:] if metadata == "unstamped" else fresh
        view = full[:-2] + fresh
        engine.ingest(deepcopy(view))
        assert rows(engine) == before + expected(stored_fresh)
        engine.ingest(deepcopy(view))
        assert rows(engine) == before + expected(stored_fresh)


@pytest.mark.parametrize("metadata", ["unstamped", "message_uid"])
@pytest.mark.parametrize("shift", ["insert", "remove", "partial"])
def test_ordered_keyless_regrow(tmp_path: Path, metadata: str, shift: str) -> None:
    full = full_list()
    for index, row in enumerate(full):
        row.pop("timestamp")
        if metadata == "message_uid":
            row["message_uid"] = f"old-{index}"
    fresh = {"role": "user", "content": "new after shrink"}
    view = (full[:4] + [fresh] + full[4:] if shift == "insert" else
            full[:4] + full[5:] + [fresh] if shift == "remove" else full[:6])
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        engine.ingest(deepcopy(full[:4]))
        engine.ingest(deepcopy(view))
        assert rows(engine) == expected(full + ([] if shift == "partial" else [fresh]))
        if shift == "partial":
            engine.ingest(deepcopy(full[:7] + [fresh]))
            assert rows(engine) == expected(full + [fresh])
            engine.ingest(deepcopy(full + [fresh]))
            assert rows(engine) == expected(full + [fresh])


@pytest.mark.parametrize("second", ["same", "shorter", "front"])
@pytest.mark.parametrize("restore", [False, True])
def test_second_short_view(tmp_path: Path, second: str, restore: bool) -> None:
    full = [{"role": r["role"], "content": r["content"]} for r in full_list()]
    short = full[:4] if second == "same" else full[:2] if second == "shorter" else full[-2:]
    fresh = msg("after two short views", 1000.)
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        engine.ingest(deepcopy(full[:4]))
        engine.ingest(deepcopy(short))
        assert rows(engine) == before
        view = (full if restore else short) + [fresh]
        engine.ingest(deepcopy(view))
        assert rows(engine) == before + expected([fresh])
        engine.ingest(deepcopy(view))
        assert rows(engine) == before + expected([fresh])


@pytest.mark.parametrize("stamped", [False, True])
@pytest.mark.parametrize("restore", [False, True])
def test_cold_after_partial_noop(tmp_path: Path, stamped: bool, restore: bool) -> None:
    full = full_list()
    if not stamped:
        for row in full:
            row.pop("timestamp")
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        engine.ingest(deepcopy(full[:-2]))
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        fresh = msg("after restart", 1000.)
        engine.ingest(deepcopy((full if restore else full[:-2]) + [fresh]))
        # A cold, unstamped shorter prefix has no warm positional proof. Keep
        # the base's lossless ambiguous-batch behavior, including its duplicates.
        replayed = full[:-2] if not stamped and not restore else []
        assert rows(engine) == before + expected(replayed + [fresh])


@pytest.mark.parametrize("metadata", ["timestamp", "message_uid", "unstamped"])
@pytest.mark.parametrize("original_stamped", [False, True])
def test_exact_length_identical_resend(tmp_path: Path, metadata: str, original_stamped: bool) -> None:
    """Known stamp conflicts distinguish resends; ambiguous views keep base parity."""
    full = [{"role": row["role"], "content": row["content"]} for row in full_list()]
    if original_stamped and metadata != "unstamped":
        for index, row in enumerate(full):
            row[metadata] = float(index + 10) if metadata == "timestamp" else f"old-{index}"
    resent = deepcopy(full)
    if metadata != "unstamped":
        for index, row in enumerate(resent[-2:]):
            row[metadata] = float(index + 100) if metadata == "timestamp" else f"new-{index}"
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        engine.ingest(deepcopy(full[:-2]))
        engine.ingest(deepcopy(resent))
        # An unstamped identical resend, or transitional legacy rows gaining
        # their first stamps, cannot be distinguished from restored history
        # adopting metadata. Match the base's replay convention in those cases.
        after = before + (expected(resent[-2:]) if original_stamped and metadata != "unstamped" else [])
        assert rows(engine) == after
        engine.ingest(deepcopy(resent))
        assert rows(engine) == after


@pytest.mark.parametrize("field", ["timestamp", "message_uid"])
@pytest.mark.parametrize("extra", [False, True])
def test_legacy_metadata_adoption(tmp_path: Path, field: str, extra: bool) -> None:
    """Restoration can adopt one metadata field while keeping the other known."""
    lineage(tmp_path)
    original = full_list()
    for index, row in enumerate(original):
        if field == "timestamp":
            row.pop("timestamp")
            row["message_uid"] = f"known-{index}"
    restored = deepcopy(original)
    for index, row in enumerate(restored):
        row[field] = float(index + 10) if field == "timestamp" else f"adopted-{index}"
    fresh = [msg("fresh after restore", 1000.)] if extra else []
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(original))
        before = rows(engine)
        assert len(before) == 8
        engine.compress(deepcopy(original[:6]), current_tokens=100, force=True)
        engine.ingest(deepcopy(restored + fresh))
        after = rows(engine)
        assert len(after) == (9 if extra else 8)
        assert Counter((r[0], r[1]) for r in after) == (
            Counter((r[0], r[1]) for r in before)
            + Counter((r["role"], r["content"]) for r in fresh)
        )
        engine.ingest(deepcopy(restored + fresh))
        assert rows(engine) == after


def test_adoption_after_compaction(tmp_path: Path, monkeypatch: Any) -> None:
    """Late timestamps on restored compaction output do not duplicate its tail."""
    product = importlib.import_module("hermes_lcm.engine")
    from tests.test_issue_436_identity_anchor import _turns

    lineage(tmp_path)
    monkeypatch.setattr(product, "summarize_with_escalation",
                        lambda **kwargs: ("Earlier turns.\nExpand for details about: turns", 1))
    original = _turns(1, 8, 0.)
    for row in original:
        row.pop("timestamp")
    with engine_context(tmp_path, window=200000, conversation_id="conv",
                        fresh_tail_count=2, leaf_chunk_tokens=1) as engine:
        engine.ingest(deepcopy(original))
        before = rows(engine)
        assert len(before) == 16
        output = engine.compress(deepcopy(original), current_tokens=engine.threshold_tokens + 1, force=True)
        assert engine._last_compression_status == "compacted"
        engine.on_session_end("S", original)
        engine.on_session_start("S", boundary_reason="compression", old_session_id="S",
                                platform="cli", context_length=200000, conversation_id="conv")
        engine.ingest(deepcopy(output))
        assert engine._compress_commit_proof is None
        engine.ingest(deepcopy(output[:-1]))
        restored = deepcopy(output)
        for index, row in enumerate(restored):
            row["timestamp"] = 1000. + index
        engine.ingest(deepcopy(restored))
        after = rows(engine)
        assert len(after) == 16
        assert Counter((r[0], r[1]) for r in after) == Counter((r[0], r[1]) for r in before)


def test_identical_unstamped_exact_restore(tmp_path: Path) -> None:
    """Two unchanged compressions must not re-store the withheld exchange."""
    full = [{"role": row["role"], "content": row["content"]} for row in full_list()]
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        assert engine.compress(deepcopy(full[:-2]), current_tokens=100, force=True) == full[:-2]
        assert engine.compress(deepcopy(full), current_tokens=100, force=True) == full
        assert rows(engine) == before


def test_partial_then_full_noop_compress(tmp_path: Path) -> None:
    """Legacy rows plus stamped turns survive split/compress/rejoin, full compress, and append."""
    full = [{"role": row["role"], "content": row["content"]} for row in full_list()]
    full += [msg(f"later-{i}", float(i + 100), "user" if i % 2 == 0 else "assistant") for i in range(8)]
    fresh = [msg("after two compressions", 200.), msg("fresh reply", 201., "assistant")]
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        head, tail = deepcopy(full[:6]), deepcopy(full[6:])
        returned = engine.compress(head, current_tokens=100, force=True)
        assert returned == head
        restored = returned + tail
        assert engine.compress(restored, current_tokens=100, force=True) == full
        assert rows(engine) == before
        engine.ingest(deepcopy(restored + fresh))
        assert rows(engine) == before + expected(fresh)
        assert len(rows(engine)) == 18
        engine.ingest(deepcopy(restored + fresh))
        assert rows(engine) == before + expected(fresh)


@pytest.mark.parametrize("transition", ["auxiliary", "failed-append", "anchor-off"])
def test_episode_frontier_lifetime(tmp_path: Path, monkeypatch: Any, transition: str) -> None:
    """I1/I12: preserve the old boundary until a stored occurrence consumes it."""
    full = full_list()
    if transition == "failed-append":
        for row in full:
            row.pop("timestamp")
    if transition == "anchor-off":
        monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "0")
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        engine.ingest(deepcopy(full))
        before = rows(engine)
        engine.ingest(deepcopy(full[:4]))
        if transition == "auxiliary":
            from concurrent.futures import ThreadPoolExecutor
            class Child:
                def __init__(self) -> None:
                    self._parent_session_id, self.session_id = "S", "child"
                    self._subagent_id, self._delegate_depth = "probe", 1
                def start(self) -> None:
                    engine.on_session_start(self.session_id, platform="cli", context_length=128000,
                                            parent_session_id=self._parent_session_id)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(Child().start).result()
            assert engine.current_session_id == "S"
            for row in full:
                row.pop("timestamp")
        elif transition == "failed-append":
            with monkeypatch.context() as patch:
                def fail(*args: Any, **kwargs: Any) -> Any:
                    raise sqlite3.OperationalError("database is locked")
                patch.setattr(engine._store, "_append_protected_batch", fail)
                engine.ingest(deepcopy(full[:4] + [msg("failed fresh", 100.)]))
            assert engine._ingest_failure_count == 1
        else:
            engine.ingest(deepcopy(full[:6]))
        engine.ingest(deepcopy(full + [msg("restored fresh", 101.)]))
        assert rows(engine) == before + expected([msg("restored fresh", 101.)])


@pytest.mark.parametrize("uid", [False, True])
def test_stamped_withheld_row_after_same_text_turn(tmp_path: Path, uid: bool) -> None:
    """I1/#1179: a known old stamp is replay even after a same-text new turn."""
    full = full_list()
    if uid:
        for index, row in enumerate(full):
            row["message_uid"] = f"old-{index}"
    fresh = msg(full[4]["content"], 600.)
    if uid:
        fresh["message_uid"] = "new-turn"
    with engine_context(tmp_path, window=128000, conversation_id="conv") as engine:
        for view in (full, full[:4], full[:4] + [fresh], full + [fresh]):
            engine.ingest(deepcopy(view))
        assert rows(engine) == expected(full + [fresh])
