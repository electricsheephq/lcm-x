"""#2466 review, delta 7: the presence audit runs under the write lock, so its worst case at every window limit is a
user-visible wait. Each shape fills one budget to just inside its limit (the audit must still answer), and one shape
also overfills it (the audit must give main's result, sized in SQL before any row loads).

The bar is 150 ms, measured at load of at most half the cores. Shared CI runners are slower and noisier, so the
assertion multiplies the bar by LCM_PRESENCE_TIMING_SLACK (default 4); run with LCM_PRESENCE_TIMING_SLACK=1 to
check the bar itself.
"""
from __future__ import annotations

import base64
import os
import random
import time

import pytest

from hermes_lcm import identity_anchor
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

from tests.test_anchor_unstored_before_cursor import SESSION, _a, _call, _engine, _u

BAR = 0.150 * float(os.environ.get("LCM_PRESENCE_TIMING_SLACK", "4"))


def _presence(engine, view):
    return engine._identity_anchor_unaccounted(
        view, view, len(view), 0, lambda i: engine._message_replay_identity(view[i]), {}, True,
        set(), {"replayed": set()}, {}, (), [])


def _best(engine, view, reps=3):
    best, out = None, None
    for _ in range(reps):
        start = time.perf_counter()
        out = _presence(engine, view)
        took = time.perf_counter() - start
        best = took if best is None or took < best else best
    print(f"\nWAIT {best * 1000:.1f} ms load1={os.getloadavg()[0]:.1f} result={out}")
    return best, out


def _append(engine, rows, batch=500):
    for i in range(0, len(rows), batch):
        engine._store.append_batch(SESSION, rows[i:i + batch])


def _call_row(i, args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "f", "arguments": args}}]}


ROWS, READ, TEXT = (identity_anchor._PRESENCE_WINDOW_ROWS, identity_anchor._PRESENCE_WINDOW_BYTES,
                    identity_anchor._PRESENCE_TEXT_BYTES)
CALL_ROW_OVERHEAD = 100  # JSON around a call's arguments, generous


@pytest.mark.parametrize("chars,expected", [(READ // ROWS - CALL_ROW_OVERHEAD, [0]), (10_000, [])],
                         ids=["at-row-and-read-limits", "read-limit-overfilled"])
def test_tool_call_json_at_the_read_limit(tmp_path, chars, expected):
    engine = _engine(tmp_path)
    try:
        _append(engine, [_call_row(i, "y" * chars) for i in range(ROWS)])
        took, out = _best(engine, [_call("brand_new", 1)])
        assert out == expected
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()


def test_inline_tool_result_text_at_the_read_limit(tmp_path):
    engine = _engine(tmp_path)
    try:
        pairs, half = [], ROWS // 2
        for i in range(half):
            pairs += [_call_row(i, "{}"), {"role": "tool", "tool_call_id": f"c{i}",
                                           "content": f"r{i} " + "z" * (READ // half - 2 * CALL_ROW_OVERHEAD)}]
        _append(engine, pairs)
        took, out = _best(engine, [_call("brand_new", 1)])
        assert out == [0]
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()


def test_text_rows_at_the_text_limit(tmp_path):
    engine = _engine(tmp_path)
    try:
        _append(engine, [{"role": "assistant", "content": f"reply {i} " + "x" * (TEXT // ROWS - 20), "timestamp": float(i)}
                         for i in range(ROWS)])
        took, out = _best(engine, [_a("a brand new reply", ROWS + 1)])
        assert out == [0]
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()


def test_text_and_tool_call_json_at_both_limits_at_once(tmp_path):
    engine = _engine(tmp_path)
    try:
        rows, half = [], ROWS // 2
        for i in range(half):
            rows += [{"role": "assistant", "content": f"reply {i} " + "x" * (TEXT // half - 20), "timestamp": float(i)},
                     _call_row(i, "y" * ((READ - TEXT) // half - CALL_ROW_OVERHEAD))]
        _append(engine, rows)
        took, out = _best(engine, [_a("a brand new reply", ROWS + 1)])
        assert out == [0]
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()


def _payload_engine(path):
    engine = LCMEngine(config=LCMConfig(database_path=str(path / "lcm.db"), large_output_externalization_enabled=True),
                       hermes_home=str(path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=10_000_000)
    return engine


SMALL_LOADS = TEXT // identity_anchor._PRESENCE_PAYLOAD_MIN_CHARS  # most payload loads one audit may make
LARGE_LOADS = TEXT // (530 * 1024)  # 512K-character payload files that fit the text limit


@pytest.mark.parametrize("raw_bytes,rows,expected", [
    (3_072, SMALL_LOADS - 50, [0]), (3_072, SMALL_LOADS + 50, []), (384 * 1024, LARGE_LOADS, [0]),
    (384 * 1024, 100, [])], ids=["4K-inside", "4K-overfilled", "512K-inside", "512K-overfilled"])
def test_externalized_payloads_are_charged_before_they_load(tmp_path, raw_bytes, rows, expected):
    engine = _payload_engine(tmp_path)
    try:
        rnd = random.Random(raw_bytes + rows)
        view = []
        for i in range(rows):
            blob = base64.b64encode(rnd.randbytes(raw_bytes)).decode()
            view += [{"role": "user", "content": f"look {i} data:image/png;base64,{blob}", "timestamp": float(i)},
                     {"role": "assistant", "content": f"ok {i}"}]
        engine.ingest(view)
        stored = [row for row in engine._store.get_session_messages(SESSION) if row["role"] == "user"]
        assert len(stored) == rows and max(len(row["content"]) for row in stored) < 2_000  # setup: externalized
        took, out = _best(engine, [_u("a brand new prompt", 1e6)])
        assert out == expected
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()
