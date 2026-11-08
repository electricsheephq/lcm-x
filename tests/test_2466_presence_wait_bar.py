"""#2466 review, delta 7: the presence audit runs under the write lock, so its worst case at every window limit is a
user-visible wait. Each shape fills the window to just inside its limits (the audit must still answer), and one shape
overfills it (the audit must give main's result, sized in SQL before any row loads). Only call and result rows are
compared, by their stored text, so the audit loads no payload file.

The bar is 150 ms, measured at load of at most half the cores. Shared CI runners are slower and noisier, so the
assertion multiplies the bar by LCM_PRESENCE_TIMING_SLACK (default 4); run with LCM_PRESENCE_TIMING_SLACK=1 to
check the bar itself.
"""
from __future__ import annotations

import os
import time

import pytest

from hermes_lcm import identity_anchor

from tests.test_anchor_unstored_before_cursor import SESSION, _call, _engine

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


ROWS, READ = identity_anchor._PRESENCE_WINDOW_ROWS, identity_anchor._PRESENCE_WINDOW_BYTES
CALL_ROW_OVERHEAD = 100  # JSON around a call's arguments, generous


@pytest.mark.parametrize("chars,expected", [(READ // ROWS - CALL_ROW_OVERHEAD, [0]), (10_000, None)],
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




def test_a_nul_inside_stored_text_is_still_sized_in_full(tmp_path):
    """SQLite's length() of text stops at the first NUL; Python loads the whole value (#2466 review, delta 7)."""
    engine = _engine(tmp_path)
    try:
        _append(engine, [_call_row(0, "{}"), {"role": "tool", "tool_call_id": "c0", "content": "x\0" + "y" * (READ + 1)}])
        assert _presence(engine, [_call("brand_new", 1)]) is None  # past the read limit: a trip, main's result
    finally:
        engine.shutdown()


@pytest.mark.parametrize("calls", [2_000, 4_000])
def test_pairing_a_wide_parallel_call_row_is_linear(tmp_path, calls):
    """One new row with thousands of parallel calls and their results: pairing is one forward pass (#2466 review)."""
    engine = _engine(tmp_path)
    try:
        row = {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"w{i}", "type": "function", "function": {"name": "f", "arguments": "{}"}} for i in range(calls)]}
        view = [row, *({"role": "tool", "tool_call_id": f"w{i}", "content": f"r{i}"} for i in range(calls))]
        took, out = _best(engine, view)
        assert out == list(range(calls + 1))
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()


DENSE = '{"items":[' + ",".join(["0"] * 900) + "]}"  # about 1.8 KB of valid JSON per call row


@pytest.mark.parametrize("view_kind", ["fresh-only", "history-then-fresh", "history-reserialized-then-fresh"])
def test_dense_json_arguments_at_the_window_limit(tmp_path, view_kind):
    """#2466 review, delta 9: arguments are compared as stored first and parsed only for an id whose stored and
    shown forms differ, within a byte budget; past it the id keeps main's result. Dense valid JSON in every call
    row must not make the audit parse the whole window."""
    import json

    per_row = len(json.dumps(_call_row(ROWS, DENSE)["tool_calls"]).encode())
    filled = min(ROWS, READ // per_row)  # as many dense call rows as fit the window's byte limit
    engine = _engine(tmp_path)
    try:
        stored = [_call_row(i, DENSE) for i in range(filled)]
        _append(engine, stored)
        shown = DENSE.replace(",", ", ")
        history = {"fresh-only": [], "history-then-fresh": stored,
                   "history-reserialized-then-fresh": [_call_row(i, shown) for i in range(filled)]}[view_kind]
        took, out = _best(engine, [*history, _call("brand_new", 1)])
        assert out == [len(history)]
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()


IDS = identity_anchor._PRESENCE_WINDOW_CALL_IDS
PER_ROW = READ // ROWS - 8  # stored bytes per row, just inside the byte limit at 10,000 rows


def _packed(make, per_row):
    """Call entries from ``make(k)`` until the row's stored tool-call JSON reaches ``per_row`` bytes."""
    import json

    calls, k = [], 0
    while True:
        calls.append(make(k))
        if len(json.dumps(calls).encode()) > per_row:
            calls.pop()
            return calls
        k += 1


def _wide_rows(shape):
    """Rows of one shape (the release manager's round-5 cost shapes, plus two dense ids per row) filling the scope to
    just inside the call-id limit, and inside the row and byte limits."""
    def entry(i, k, arguments):
        return {"id": f"call_{i:05d}_{k:04d}", "type": "function", "function": {"name": "f", "arguments": arguments}}

    makers = {
        "wide_hermes": lambda i: (lambda k: entry(i, k, "{}")),
        "wide_tiny": lambda i: (lambda k: {"id": f"{i:05d}{k:04d}"}),
        "args_dict": lambda i: (lambda k: entry(i, k, {f"k{j}": j for j in range(40)})),
        "two_dense_ids": lambda i: (lambda k: entry(i, k, DENSE[:PER_ROW // 2 - 120])),
    }
    rows, total = [], 0
    while len(rows) < ROWS:
        calls = _packed(makers[shape](len(rows)), PER_ROW)
        if shape == "two_dense_ids":
            calls = calls[:2]
        if total + len(calls) > IDS:
            break
        rows.append({"role": "assistant", "content": "", "tool_calls": calls})
        total += len(calls)
    return rows


@pytest.mark.parametrize("view_kind", ["fresh-only", "history-then-fresh"])
@pytest.mark.parametrize("shape", ["wide_hermes", "wide_tiny", "args_dict", "two_dense_ids"])
def test_call_rows_at_the_call_id_limit(tmp_path, shape, view_kind):
    """#2466 review, delta 10: the audit's work grows with stored tool-call entries, so the window has a third limit,
    on those entries, sized in SQL with the rows and bytes. Each shape fills the scope to just inside it. Entries
    without text arguments (wide_tiny's bare ids, args_dict's object arguments) are not the standard tool shape, so
    those scopes give main's result from the SQL sizing pass (#2466 review, scope cut)."""
    engine = _engine(tmp_path)
    try:
        stored = _wide_rows(shape)
        _append(engine, stored, batch=100)
        history = stored if view_kind == "history-then-fresh" else []
        took, out = _best(engine, [*history, _call("brand_new", 1)])
        assert out == ([len(history)] if shape in ("wide_hermes", "two_dense_ids") else None)
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("extra,expected", [(0, [0]), (1, None)], ids=["at-the-limit", "one-entry-past"])
def test_the_call_id_limit_is_counted_in_sql(tmp_path, extra, expected):
    """Three entries per call row, so the row and byte limits stay far away: at the call-id limit the audit answers;
    one entry past it, it gives main's result (nothing beyond its own finds)."""
    full = (IDS - 1) // 3
    assert full + 1 < ROWS
    engine = _engine(tmp_path)
    try:
        rows = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}_{k}", "type": "function", "function": {"name": "f", "arguments": "{}"}} for k in range(3)]}
            for i in range(full)]
        rows.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"tail_{k}", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            for k in range(IDS - 3 * full + extra)]})
        _append(engine, rows)
        took, out = _best(engine, [_call("brand_new", 1)])
        assert out == expected
        assert took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()
