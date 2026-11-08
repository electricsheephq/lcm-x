"""#2466 review, scope cut: the presence check admits only the standard tool shape. Each stored row's tool_calls must be
NULL, or a JSON array of objects whose id and function.arguments are text; the sizing pass checks that in SQL before any
row loads. Any other stored row gives main's result for the whole audit (the audit keeps only its own finds) and logs
one INFO line. Each test stores one row of a non-standard shape beside a fresh call that the check would otherwise
store, so main's result is []."""
from __future__ import annotations

import json
import logging
import os
import time

import pytest

from hermes_lcm import identity_anchor

from tests.test_anchor_unstored_before_cursor import SESSION, _call, _engine, _presence

BAR = 0.150 * float(os.environ.get("LCM_PRESENCE_TIMING_SLACK", "4"))


def _entry(call_id, arguments="{}"):
    return {"id": call_id, "type": "function", "function": {"name": "f", "arguments": arguments}}


def _raw_insert(engine, tool_calls_text):
    """A stored row whose tool_calls column holds exactly this text (a writer other than this store's append)."""
    engine._store.append_batch(SESSION, [{"role": "assistant", "content": "", "tool_calls": [_entry("c_raw")]}])
    conn = engine._store._conn
    store_id = conn.execute("SELECT max(store_id) FROM messages WHERE session_id=?", (SESSION,)).fetchone()[0]
    conn.execute("UPDATE messages SET tool_calls=? WHERE store_id=?", (tool_calls_text, store_id))
    conn.commit()


SHAPES = {
    # Astra, final review, blocker 1: an array stored as a JSON string, which the store unwraps on load.
    "array-stored-as-a-string": lambda e: e._store.append_batch(SESSION, [
        {"role": "assistant", "content": "", "tool_calls": json.dumps([_entry(f"s{k}") for k in range(20)])}]),
    # Astra blocker 1 and a Claude nit: NaN, which Python's json accepts and SQLite's json_valid rejects.
    "nan-in-arguments": lambda e: e._store.append_batch(SESSION, [
        {"role": "assistant", "content": "", "tool_calls": [_entry("n1", {"x": float("nan")})]}]),
    # Astra blocker 2: object arguments (dense), which the window read would decode in full.
    "object-arguments": lambda e: e._store.append_batch(SESSION, [
        {"role": "assistant", "content": "", "tool_calls": [_entry("d1", {"items": [0] * 600})]}]),
    "null-arguments": lambda e: e._store.append_batch(SESSION, [
        {"role": "assistant", "content": "", "tool_calls": [_entry("z1", None)]}]),
    "absent-arguments": lambda e: e._store.append_batch(SESSION, [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a1", "type": "function", "function": {"name": "f"}}]}]),
    "non-text-id": lambda e: e._store.append_batch(SESSION, [
        {"role": "assistant", "content": "", "tool_calls": [_entry(7)]}]),
    "non-object-element": lambda e: _raw_insert(e, json.dumps([_entry("o1"), "o2"])),
    "json-object-not-array": lambda e: _raw_insert(e, json.dumps(_entry("o3"))),
    "invalid-json": lambda e: _raw_insert(e, "[{not json"),
    "empty-text": lambda e: _raw_insert(e, ""),
    # Astra, final review, nit: one row with the same id twice and non-string arguments made sort() raise.
    "same-id-twice-with-null-arguments": lambda e: e._store.append_batch(SESSION, [
        {"role": "assistant", "content": "", "tool_calls": [_entry("t1", None), _entry("t1", None)]}]),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_a_non_standard_stored_tool_shape_gives_mains_result(tmp_path, shape, caplog):
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [{"role": "assistant", "content": "", "tool_calls": [_entry("ok1")]}])
        SHAPES[shape](engine)
        engine._store.append_batch(SESSION, [{"role": "assistant", "content": "", "tool_calls": [_entry("ok2")]}])
        view = [_call("brand_new", 1)]
        with caplog.at_level(logging.INFO, logger="hermes_lcm.identity_anchor"):
            assert _presence(engine, view) == []
            assert _presence(engine, view) == []
        assert sum("non-standard tool-call shape" in r.getMessage() for r in caplog.records) <= 1
    finally:
        engine.shutdown()


def test_standard_shapes_are_admitted(tmp_path):
    """NULL tool_calls, an empty array and text arguments (including text that is not JSON) are admitted."""
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [
            {"role": "user", "content": "a prompt"},
            {"role": "assistant", "content": "", "tool_calls": [_entry("p1"), _entry("p2", "not json")]},
            {"role": "tool", "tool_call_id": "p1", "content": "r1"},
        ])
        _raw_insert(engine, "[]")
        assert _presence(engine, [_call("brand_new", 1)]) == [0]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("where", ["newest", "oldest"])
def test_a_trip_costs_sql_only(tmp_path, where):
    """At the window limits with one non-standard row, the audit stops in the SQL sizing pass. The oldest row is the
    worst case: every admitted row is checked in SQL first."""
    rows_n, per_row = identity_anchor._PRESENCE_WINDOW_ROWS, identity_anchor._PRESENCE_WINDOW_BYTES // 10_000 - 120
    dense = '{"items":[' + ",".join(["0"] * (per_row // 2 - 20)) + "]}"
    rows = [{"role": "assistant", "content": "", "tool_calls": [_entry(f"c{i}", dense)]} for i in range(rows_n - 1)]
    bad = {"role": "assistant", "content": "", "tool_calls": [_entry("bad", {"items": [0] * 600})]}
    rows = [bad, *rows] if where == "oldest" else [*rows, bad]
    engine = _engine(tmp_path)
    try:
        for i in range(0, len(rows), 500):
            engine._store.append_batch(SESSION, rows[i:i + 500])
        best = None
        for _ in range(3):
            start = time.perf_counter()
            out = _presence(engine, [_call("brand_new", 1)])
            took = time.perf_counter() - start
            best = took if best is None or took < best else best
        print(f"\nTRIP {where} {best * 1000:.1f} ms load1={os.getloadavg()[0]:.1f}")
        assert out == []
        assert best < BAR, f"{best * 1000:.0f} ms"
    finally:
        engine.shutdown()
