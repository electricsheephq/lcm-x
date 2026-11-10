"""#2466 review, scope cut: the presence check admits only the standard tool shape. Each stored row's tool_calls must be
NULL, or a JSON array of objects whose id and function.arguments are text; the sizing pass checks that in SQL before any
row loads. Any other stored row gives main's result for the whole audit (the audit keeps only its own finds) and logs
one INFO line. Each test stores one row of a non-standard shape beside a fresh call that the check would otherwise
store, so main's result is a trip (the audit returns None and the caller keeps main's plan)."""
from __future__ import annotations

import json
import os
import time

import pytest

from hermes_lcm import identity_anchor

from tests.test_anchor_unstored_before_cursor import SESSION, _call, _engine, _presence

BAR = 0.150 * float(os.environ.get("LCM_PRESENCE_TIMING_SLACK", "4"))


@pytest.fixture(autouse=True)
def _fresh_presence_notices(monkeypatch):
    """The audit logs each trip reason once per (store path, inode, session) for the process. A deleted test directory
    can hand its path and inode to the next test (tmp_path_retention_policy=failed, inode reuse on CI), so each test
    starts with empty notice sets to see its own line (#2466 CI)."""
    monkeypatch.setattr(identity_anchor, "_PRESENCE_SHAPE_LOGGED", set())
    monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_LOGGED", set())


def _entry(call_id, arguments="{}"):
    return {"id": call_id, "type": "function", "function": {"name": "f", "arguments": arguments}}


def _anchor_lines(monkeypatch):
    """The audit's INFO and WARNING lines, recorded at the logger's own methods: a process-wide logging setup left by
    another test in the same worker cannot hide them, as it can with caplog (CI shards, #2466)."""
    lines = []
    for level in ("info", "warning"):
        monkeypatch.setattr(identity_anchor.logger, level,
                            lambda msg, *args, _level=level, **_: lines.append((_level, msg % args if args else msg)))
    return lines


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
def test_a_non_standard_stored_tool_shape_gives_mains_result(tmp_path, shape, monkeypatch):
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [{"role": "assistant", "content": "", "tool_calls": [_entry("ok1")]}])
        SHAPES[shape](engine)
        engine._store.append_batch(SESSION, [{"role": "assistant", "content": "", "tool_calls": [_entry("ok2")]}])
        view = [_call("brand_new", 1)]
        lines = _anchor_lines(monkeypatch)
        assert _presence(engine, view) is None
        assert _presence(engine, view) is None
        # Every trip reaches the counted reason through the SQL CASE guard, never the exception path.
        assert sum("non-standard tool-call shape" in text for _, text in lines) == 1, lines
        assert not [text for _, text in lines if "presence unavailable" in text]
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
def test_a_trip_costs_sql_only(tmp_path, where, latency_check):
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
        assert out is None
        assert not latency_check or best < BAR, f"{best * 1000:.0f} ms"
    finally:
        engine.shutdown()


def test_one_stored_row_with_many_parallel_calls_is_checked_in_linear_time(tmp_path, latency_check):
    """#2466 terminal review nit: the admission check reads each entry from the entry itself, so one stored row with
    many parallel calls is checked in linear time: four times the entries costs well under eight times as long (a
    lookup from the row's root per entry was quadratic, about sixteen times), and 9,999 calls stay under the bar."""
    def best_of(width):
        engine = _engine(tmp_path / f"w{width}")
        try:
            engine._store.append_batch(SESSION, [{"role": "assistant", "content": "", "tool_calls": [
                _entry(f"w{k}", '{"path": "a"}') for k in range(width)]}])
            best = None
            for _ in range(3):
                start = time.perf_counter()
                out = _presence(engine, [_call("brand_new", 1)])
                took = time.perf_counter() - start
                best = took if best is None or took < best else best
            assert out == [0]
            return best
        finally:
            engine.shutdown()

    wide = identity_anchor._PRESENCE_WINDOW_CALL_IDS - 1
    small, large = best_of(wide // 4), best_of(wide)
    print(f"\nWIDE-ROW {wide // 4}:{small * 1000:.1f} ms {wide}:{large * 1000:.1f} ms x{large / small:.1f} "
          f"load1={os.getloadavg()[0]:.1f}")
    # The ratio is wall time, so it runs only in the nightly latency cell, like the bar: on a loaded CI runner the
    # correctness run measured x8.8 against this x8 bound with linear code (the quadratic form was about x16).
    assert not latency_check or large < 8 * small, f"x{large / small:.1f}"
    assert not latency_check or large < BAR, f"{large * 1000:.0f} ms"


def test_one_stored_row_over_the_call_id_window_trips_before_any_entry_is_checked(tmp_path, monkeypatch, latency_check):
    """#2466 terminal review (bot threads): a row with more entries than the whole call-id window returns its length
    before the per-entry check, so 180,000 tiny calls inside the byte limit trip on the window for about the cost of
    one length read: under four times one admitted row of 9,999 calls (checking each entry first cost over ten)."""
    def best_of(name, width, expect):
        engine = _engine(tmp_path / name)
        try:
            engine._store.append_batch(SESSION, [{"role": "assistant", "content": "", "tool_calls": [
                _entry(f"t{k}", "") for k in range(width)]}])
            best = None
            for _ in range(3):
                start = time.perf_counter()
                out = _presence(engine, [_call("brand_new", 1)])
                took = time.perf_counter() - start
                best = took if best is None or took < best else best
                assert out == expect
            return best
        finally:
            engine.shutdown()

    over = 180_000
    lines = _anchor_lines(monkeypatch)
    admitted = best_of("admitted", identity_anchor._PRESENCE_WINDOW_CALL_IDS - 1, [0])
    tripped = best_of("over", over, None)
    windows = [text for _, text in lines if "presence window incomplete" in text]
    assert len(windows) == 1 and "call_ids=" in windows[0], lines
    assert not [text for _, text in lines if "non-standard tool-call shape" in text]
    print(f"\nOVER-CAP-ROW {over}:{tripped * 1000:.1f} ms vs 9999:{admitted * 1000:.1f} ms "
          f"x{tripped / admitted:.1f} load1={os.getloadavg()[0]:.1f}")
    # Wall-time ratio: nightly latency cell only, as above. The correctness run still checks that the row trips on
    # the call-id window (one log line) and that no entry is checked first (no non-standard-shape line).
    assert not latency_check or tripped < 4 * admitted, f"x{tripped / admitted:.1f}"
    assert not latency_check or tripped < BAR, f"{tripped * 1000:.0f} ms"
