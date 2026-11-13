"""Release-predicate controls plus offline, public-API smoke coverage."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import os
import sqlite3
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bench.instruments.busy_bar import cells, driver, gate, holds, orchestrate


def sqlite_error(code: int, message: str = "SQLite error") -> sqlite3.OperationalError:
    error = sqlite3.OperationalError(message)
    error.sqlite_errorcode = code
    return error


def log(message: str, thread: int = 1, time: float = 2.0) -> dict:
    return {"thread": thread, "time": time, "message": message}


def test_three_lines_count_one_operation() -> None:
    records = [log("SQLite write lock recovery exhausted")] * 3
    result = driver.count_operation(records, 1, 1.0, 3.0, "ingest", 0, 0, True)
    assert result == {"busy": True, "signals": ["write-lock-recovery"]}


def test_other_thread_and_outside_interval_do_not_count() -> None:
    records = [log("SQLite write lock recovery exhausted", thread=2),
               log("SQLite write lock recovery exhausted", time=0.5),
               log("SQLite write lock recovery exhausted", time=3.5)]
    assert driver.count_operation(records, 1, 1, 3, "ingest", 0, 0, True) == {
        "busy": False, "signals": []}


@pytest.mark.parametrize("op,counter", list(driver.COUNTERS.items()))
def test_counter_and_log_are_one_busy_operation(op: str, counter: str) -> None:
    result = driver.count_operation([log("skipped due to SQLite lock")], 1, 1, 3, op, 0, 2, True)
    assert result == {"busy": True, "signals": ["skipped-lock", counter]}


@pytest.mark.parametrize("candidate,op,before,after,busy", [
    (False, "compress", 0, 1, False), (False, "session-end", 0, 1, False),
    (True, "compress", 1, 2, True), (True, "session-end", 0, 1, True),
    (True, "ingest", 0, 1, False), (True, "compress", 2, 2, False),
])
def test_counter_scope(candidate: bool, op: str, before: int, after: int, busy: bool) -> None:
    assert driver.count_operation([], 1, 1, 3, op, before, after, candidate)["busy"] is busy


@pytest.mark.parametrize("message,expected", [
    ("LCM ingest failed (OperationalError): database is locked", "ingest-busy"),
    ("LCM ingest failed (OperationalError): SQLITE_BUSY", "ingest-busy"),
    ("LCM ingest failed (ValueError): invalid row", None),
    ("unrelated busy message", None),
])
def test_pinned_log_signals(message: str, expected: str | None) -> None:
    assert driver.busy_signal(message) == expected


def test_real_operation_combines_counter_and_root_log() -> None:
    import logging

    engine = SimpleNamespace(_ingest_ledger_counters={"ledger.emission_busy": 0})
    capture = driver.Capture()
    operations = driver.Operations(capture, candidate=True)

    def call() -> str:
        engine._ingest_ledger_counters["ledger.emission_busy"] += 1
        logging.warning("SQLite write lock recovery exhausted")
        return "returned"

    logging.getLogger().addHandler(capture)
    try:
        assert operations(engine, "A", "compress", call, 25) == "returned"
    finally:
        logging.getLogger().removeHandler(capture)
    assert len(operations.records) == 1
    assert operations.records[0]["busy"] is True
    assert operations.records[0]["signals"] == ["write-lock-recovery", "ledger.emission_busy"]


def test_success_wall_summary_and_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter((1.0, 1.125))
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    operations = driver.Operations(driver.Capture(), candidate=False)
    assert operations(None, "A", "ingest", lambda: "ok", 0) == "ok"
    assert operations.records[0]["wall_ms"] == 125
    records = [{"op": "ingest", "wall_ms": value} for value in range(1, 21)]
    records += [{"op": "ingest", "wall_ms": 999, flag: True}
                for flag in ("error", "busy", "inconclusive")]
    records.append({"op": "ingest"})  # Older ledgers remain readable, without fabricated timings.
    summary = driver.success_wall_ms(records)
    assert summary["ingest"] == summary["overall"] == {"n": 20, "p50": 10, "p95": 19, "max": 20}
    assert summary["compress"] == {"n": 0, "p50": None, "p95": None, "max": None}
    assert summary["note"] == driver.WALL_NOTE


@pytest.mark.parametrize("busy", [0, 2])
def test_success_wall_is_reported_without_changing_gate(busy: int) -> None:
    baseline, candidate = ledger(0), ledger(busy)
    before = evaluate(baseline, candidate, control(1), control(1))
    for rows, duration in ((baseline, 10), (candidate, 1000)):
        for row in rows:
            for record in row["operations"]:
                record["wall_ms"] = duration
    after = evaluate(baseline, candidate, control(1), control(1))
    wall = after.pop("success_wall_ms")
    before.pop("success_wall_ms")
    assert after == before  # Every existing result field, including verdict and tables, is identical.
    assert wall["p95_ratio"]["overall"]["overall"] == 100
    for cell in cells.CELLS:
        assert wall["baseline"][cell]["ingest"]["p95"] == 10
    assert driver.WALL_NOTE in gate.report({**after, "success_wall_ms": wall})
    for row in baseline:
        for record in row["operations"]:
            record["wall_ms"] = 0
    zero = gate.comparison(baseline, candidate)["success_wall_ms"]
    assert zero["baseline"]["overall"]["overall"]["p95"] == 0
    assert all(value is None for value in zero["p95_ratio"]["overall"].values())
    assert wall["p95_ratio"]["ingest"]["compress"] is None  # No compression in this cell.
    assert wall["baseline"]["overall"]["compress"]["n"] == 1
    assert wall["baseline"]["overall"]["session-end"]["n"] == 1


def ledger(busy: int, k: int = 2) -> list[dict]:
    """One run per cell with the driver's prescribed operations; each engine's first `busy` operations are busy."""
    rows = []
    for cell in cells.CELLS:
        prescribed = driver.prescribed_operations(cell, k)
        operations = []
        for owner in ("A", "B"):
            ops = [op for op in ("ingest", "compress", "session-end") for _ in range(prescribed[(owner, op)])]
            operations += [{"engine": owner, "op": op, "busy": index < busy} for index, op in enumerate(ops)]
        rows.append({"cell": cell, "run": 0, "k": k, "b_view_rows": 2000, "busy_timeout_ms": None, "seed": 2055,
                     "window_ms": 400, "offsets_ms": [1, 2, 3], "errors": [], "operations": operations})
    return rows


CANDIDATE_IDENTITY = {"plugin_version": "candidate", "product_sha256": "0" * 64}


def stamp(rows: list[dict] | None, identity: dict) -> list[dict] | None:
    """The driver records its tree's identity in every ledger row; a fixture's own value wins."""
    return None if rows is None else [{"tree_identity": identity, **row} for row in rows]


def evaluate(baseline: list[dict], candidate: list[dict], control_baseline: list[dict] | None = None,
             control_candidate: list[dict] | None = None, **kwargs: object) -> dict:
    """gate.evaluate on ledgers from the v0.29.2 GA baseline and a distinct candidate tree."""
    return gate.evaluate(stamp(baseline, driver.BASELINE_IDENTITY), stamp(candidate, CANDIDATE_IDENTITY),
                         stamp(control_baseline, driver.BASELINE_IDENTITY),
                         stamp(control_candidate, CANDIDATE_IDENTITY), **kwargs)


def summarize(collected: dict[str, list], **kwargs: object) -> dict:
    """holds.summarize on runs carrying the proof fields holds.probe writes; a fixture's own values win."""
    proven: dict[str, list] = {}
    for name, runs in collected.items():
        proven[name] = []
        for number, run in enumerate(runs):
            calls = [{"view_rows": run["rows"] + 2 if call["shape"] == "t" else run["rows"],
                      "rows_stored": 2 if call["shape"] == "t" else run["rows"], "ingest_failures": 0,
                      "begin_ns": 10 * index + 1, "end_ns": 10 * index + 5, **call}
                     for index, call in enumerate(run["calls"])]
            proven[name].append({
                "run_id": f"{name}-{number}",
                "tree_identity": driver.BASELINE_IDENTITY if name == "baseline" else CANDIDATE_IDENTITY,
                "monitor": {"samples": [[0, 0], [10 * len(calls) + 10, 0]], "intervals": {"max_ms": 0.2}},
                **run, "calls": calls})
    return holds.summarize(proven, **kwargs)


def control(busy: int) -> list[dict]:
    rows = [{**row, "busy_timeout_ms": 100} for row in ledger(0)]
    for row in rows:
        for record in row["operations"]:
            record["busy"] = busy > 0 and (record["engine"] == "A" or busy > 1)
    return rows


def smoke(verdict: str) -> str:
    """Fixtures with fewer than the prescribed runs get the same rule, labelled a smoke."""
    return verdict.replace("GATE", "SMOKE")


@pytest.mark.parametrize("base,candidate,expected", [
    (1, 1, "GATE PASS"), (2, 1, "GATE PASS"), (1, 2, "GATE PASS"), (1, 3, "GATE FAIL"),
    (0, 0, "GATE PASS"), (0, 1, "GATE FAIL"),
])
def test_gate_counts(base: int, candidate: int, expected: str) -> None:
    result = evaluate(ledger(base), ledger(candidate), control(1), control(1))
    assert result["verdict"] == smoke(expected) and not result["gate_session"]
    rendered = gate.report(result)
    assert smoke(expected) in rendered
    total = sum(sum(driver.prescribed_operations(cell, 2).values()) for cell in cells.CELLS)
    assert f"TOTAL total {total}" in rendered


def test_gate_regression_in_one_cell_cannot_hide_in_total() -> None:
    baseline, candidate = ledger(1), ledger(0)
    candidate[0] = ledger(3)[0]
    result = evaluate(baseline, candidate, control(1), control(1))
    assert result["verdict"] == "SMOKE FAIL"
    assert result["failed_cells"] == ["ingest"]


@pytest.mark.parametrize("damage", ["count", "owner", "op", "schedule", "missing", "error", "duplicate"])
def test_gate_mismatched_or_incomplete_exposure_is_inconclusive(damage: str) -> None:
    baseline, candidate = ledger(1), ledger(1)
    if damage == "count":
        candidate[0]["operations"].pop()
    elif damage == "owner":
        candidate[0]["operations"][0]["engine"] = "B"
    elif damage == "op":
        candidate[0]["operations"][0]["op"] = "compress"
    elif damage == "schedule":
        candidate[0]["k"] = 3
    elif damage == "missing":
        candidate.pop()
    elif damage == "error":
        candidate[0]["errors"] = ["B:RuntimeError"]
    else:
        candidate.append(deepcopy(candidate[0]))
    assert evaluate(baseline, candidate, control(1), control(1))["verdict"] == "INCONCLUSIVE"


def test_gate_cli_writes_verdict(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("baseline.jsonl", "candidate.jsonl", "control-base.jsonl", "control-cand.jsonl")]
    for index, path in enumerate(paths):
        rows = stamp(ledger(1) if index < 2 else control(1),
                     driver.BASELINE_IDENTITY if index % 2 == 0 else CANDIDATE_IDENTITY)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = subprocess.run([sys.executable, str(Path(gate.__file__)), *map(str, paths[:2]),
                             "--control-baseline", str(paths[2]), "--control-candidate", str(paths[3])],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 3, result.stderr  # one run per cell: a smoke, never a gate verdict
    assert "SMOKE PASS" in result.stdout and "GATE PASS" not in result.stdout


def test_orchestrator_serial_interleaving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def child(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen.append(Path(command[3]).name)
        cell, output = command[4], Path(command[5])
        row = next(row for row in (control(1) if "--busy-timeout-ms" in command else ledger(1))
                   if row["cell"] == cell)
        row["tree_identity"] = driver.BASELINE_IDENTITY if seen[-1] == "baseline" else CANDIDATE_IDENTITY
        output.write_text(json.dumps(row) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrate.subprocess, "run", child)
    out = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["orchestrate", "baseline", "candidate", str(out), "--runs", "2",
                                       "--work-root", str(tmp_path / "scratch")])
    with pytest.raises(SystemExit) as exc:
        orchestrate.main()
    assert exc.value.code == 3  # R=2 is a smoke
    assert seen == (["baseline", "candidate"] * 4 + ["candidate", "baseline"] * 4) * 2
    assert len(gate.read(out / "baseline.jsonl")) == 8
    assert len(gate.read(out / "candidate.jsonl")) == 8


@pytest.mark.parametrize("cell", cells.CELLS)
def test_current_tree_smoke_totals_only(tmp_path: Path, cell: str) -> None:
    tree = Path(__file__).resolve().parents[1]
    output = tmp_path / "run.jsonl"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", str(Path(driver.__file__)), str(tree), cell, str(output),
         str(tmp_path / "work"), "--runs", "1", "--b-view-rows", "2", "--busy-timeout-ms", "5000"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = gate.read(output)
    assert len(rows) == 1
    result = rows[0]
    assert result["totals"]["A"]["operations"] == cells.A_OPERATIONS[cell]
    assert result["totals"]["B"]["operations"] == 2 * cells.A_OPERATIONS[cell]
    assert result["b_view_sizes"] == [2] * (2 * cells.A_OPERATIONS[cell])
    assert all(row["busy_timeout_ms"] == 5000 for row in result["applied_timeouts"])
    assert result["totals"]["total"]["operations"] == cells.A_OPERATIONS[cell] + len(result["b_view_sizes"])
    assert result["tree_identity"] == driver.tree_identity(tree)


def test_fixed_exposure_ignores_simulated_tree_durations() -> None:
    def simulate(a_delay: float, b_delay: float) -> tuple[list, list]:
        a, b = [], []

        def write(batch: int, index: int) -> None:
            time.sleep(b_delay)
            b.append((batch, index))

        contention = driver.FixedContention(2, write, window_ms=0)
        try:
            for cell in cells.CELLS:
                for number in range(cells.A_OPERATIONS[cell]):
                    def call() -> None:
                        time.sleep(a_delay)
                        a.append((cell, number))
                    contention(call)
        finally:
            contention.close()
        return a, b

    fast_a = simulate(0, 0.002)
    slow_a = simulate(0.004, 0)
    assert fast_a == slow_a
    assert len(fast_a[0]) == sum(cells.A_OPERATIONS.values())
    assert fast_a[1] == [(batch, index) for batch in range(len(fast_a[0])) for index in range(2)]
    views = [cells.writer_view(number, 2000) for number in range(2)]
    assert all(len(view) == 2000 for view in views)
    assert not ({row["content"] for row in views[0]} & {row["content"] for row in views[1]})


@pytest.mark.parametrize("missing", ["baseline", "candidate"])
def test_zero_control_is_inconclusive(missing: str) -> None:
    result = evaluate(ledger(0), ledger(1), control(0 if missing == "baseline" else 1),
                           control(0 if missing == "candidate" else 1))
    assert result["verdict"] == "INCONCLUSIVE"
    assert any(missing + " has 0 busy" in reason for reason in result["reasons"])


def test_control_busy_regression_never_gates() -> None:
    result = evaluate(ledger(0), ledger(0), control(1), control(3))
    assert result["verdict"] == "SMOKE PASS"


@pytest.mark.parametrize("damage", ["seed", "offset", "budget", "error"])
def test_control_must_match_production_exposure(damage: str) -> None:
    scaled = control(1)
    if damage == "seed":
        scaled[0]["seed"] += 1
    elif damage == "offset":
        scaled[0]["offsets_ms"] = [7, 8, 9]
    elif damage == "budget":
        scaled[0]["busy_timeout_ms"] = 99
    else:
        scaled[0]["errors"] = ["driver bug"]
    assert evaluate(ledger(1), ledger(1), control(1), scaled)["verdict"] == "INCONCLUSIVE"


def test_seeded_arrivals_are_identical_across_trees() -> None:
    def simulate(seed: int, delay: float) -> list[float]:
        contention = driver.FixedContention(2, lambda batch, index: time.sleep(delay), seed, 5)
        try:
            for _ in range(4):
                contention(lambda: None)
        finally:
            contention.close()
        return contention.offsets

    offsets = simulate(2055, 0)
    assert offsets == simulate(2055, 0.002)
    assert offsets != simulate(2056, 0)
    assert all(0 <= offset < 5 for offset in offsets)


@pytest.mark.parametrize("code", [5, 1])
@pytest.mark.parametrize("cell", cells.CELLS)
def test_operation_errors_continue_equal_exposure(tmp_path: Path, cell: str, code: int) -> None:
    class Connection:
        def execute(self, sql: str) -> Connection:
            assert sql == "PRAGMA busy_timeout"  # Production mode does not set a PRAGMA.
            return self

        def fetchone(self) -> tuple[int]:
            return (5000,)

    class Engine:
        def __init__(self, **kwargs: object) -> None:
            self._store = SimpleNamespace(_conn=Connection())

        def on_session_start(self, *args: object, **kwargs: object) -> None:
            pass

        def ingest(self, view: list) -> None:
            raise sqlite_error(code, "database is locked")

        def compress(self, view: list, force: bool) -> None:
            raise sqlite_error(code, "database is locked")

        def on_session_end(self, *args: object) -> None:
            raise sqlite_error(code, "database is locked")

        def shutdown(self) -> None:
            pass

    result = driver.run(cell, tmp_path / "run", SimpleNamespace(LCMConfig=lambda **kw: kw),
                        SimpleNamespace(LCMEngine=Engine), driver.Capture(), b_view_rows=2, window_ms=0)
    assert not result["errors"]
    assert result["totals"]["A"] == {"operations": cells.A_OPERATIONS[cell], "busy": cells.A_OPERATIONS[cell] if code == 5 else 0}
    assert result["inconclusive"] is (code != 5)
    assert result["totals"]["B"] == {"operations": 2 * cells.A_OPERATIONS[cell], "busy": 2 * cells.A_OPERATIONS[cell] if code == 5 else 0}
    assert all(record["error"] == "OperationalError" and record["sqlite_message"] == "database is locked"
               for record in result["operations"])
    if cell == "compress":
        # The final ingest retains the original full host view on compression failure.
        a = [record for record in result["operations"] if record["engine"] == "A"]
        assert a[-1]["view_rows"] == cells.TURNS * 2 + 1


@pytest.mark.parametrize("exc,message,busy", [
    (sqlite_error(5, "busy" + "x" * 300), "busy" + "x" * 196, True),
    (sqlite3.OperationalError("syntax error"), "syntax error", False),
    (ValueError("private content locked"), None, False),
])
def test_exception_messages_are_sqlite_only_and_bounded(exc: Exception, message: str | None, busy: bool) -> None:
    operations = driver.Operations(driver.Capture(), False)

    def fail() -> None:
        raise exc

    assert operations(SimpleNamespace(), "A", "ingest", fail, 2) is None
    assert operations.records[0]["sqlite_message"] == message
    assert operations.records[0]["busy"] is busy


@pytest.mark.parametrize("trace,expected", [
    ([(1, 0, "BEGIN"), (2, 0, "INSERT"), (4, 0, "COMMIT")], [(0, 1000, 3000, False)]),
    ([(2, 0, "UPDATE"), (4, 0, "END")], [(0, 2000, 2000, False)]),
    ([(1, 0, "SAVEPOINT"), (2, 0, "INSERT"), (3, 0, "RELEASE"), (4, 0, "COMMIT")],
     [(0, 2000, 2000, False)]),
    ([(1, 0, "BEGIN"), (2, 1, "REPLACE"), (3, 1, "ROLLBACK"), (4, 0, "COMMIT")],
     [(1, 2000, 1000, False), (0, 1000, 3000, False)]),
    ([(2, 0, "DELETE"), (3, 0, "SAVEPOINT"), (4, 0, "RELEASE")], [(0, 2000, 3000, True)]),
])
def test_write_window_extraction(trace: list, expected: list) -> None:
    found = holds.windows(trace, 0, 5)
    assert [(row["conn"], row["start_ms"], row["hold_ms"], row["open_at_end"]) for row in found] == expected


def test_holds_percentiles_and_relative_verdict() -> None:
    def runs(multiplier: float) -> list:
        return [{"journal_mode": "wal", "rows": 2000, "calls": [
            {"shape": "n", "position": position, "longest_ms": n * multiplier,
             "first_start_ms": 2, "call_ms": 30} for position in range(1, 7)] + [
            {"shape": "t", "position": 1, "longest_ms": n * multiplier,
             "first_start_ms": 2, "call_ms": 30}]} for n in range(1, 11)]

    result = summarize({"baseline": runs(1), "candidate": runs(2.25)})
    assert result["verdict"] == "LOCK-TRIPWIRE PASS"
    assert result["tables"]["baseline"]["n"]["p90_longest_ms"] == 9
    assert summarize({"baseline": runs(1), "candidate": runs(2.26)})["verdict"] == "LOCK-TRIPWIRE FAIL"


@pytest.mark.parametrize("platform,size,offsets", [
    ("darwin", 24, {"start": 0, "length": 8, "pid": 16, "type": 20, "whence": 22}),
    ("linux", 32, {"type": 0, "whence": 2, "start": 8, "length": 16, "pid": 24}),
])
def test_monitor_flock_layouts(platform: str, size: int, offsets: dict) -> None:
    import ctypes
    from bench.instruments.busy_bar import monitor

    layout = monitor.flock_type(platform)
    assert ctypes.sizeof(layout) == size
    assert {name: getattr(layout, name).offset for name in offsets} == offsets
    packed = layout()
    packed.start, packed.length, packed.pid = 120, 1, 12345
    packed.type, packed.whence = 3, 0
    unpacked = layout.from_buffer_copy(bytes(packed))
    assert (unpacked.start, unpacked.length, unpacked.pid, unpacked.type, unpacked.whence) == (120, 1, 12345, 3, 0)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        monitor.flock_type("unknown")


def test_monitor_known_hold_and_readers_only(tmp_path: Path) -> None:
    from bench.instruments.busy_bar import monitor

    database = tmp_path / "control.db"
    conn = sqlite3.connect(database)
    assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    conn.execute("CREATE TABLE control (value INTEGER)")
    observer = monitor.Monitor(database)
    writer = subprocess.Popen([sys.executable, "-B", "-c", """
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1])
sys.stdin.readline()
conn.execute('BEGIN IMMEDIATE')
conn.execute('INSERT INTO control VALUES (1)')
begin = time.monotonic_ns()
time.sleep(0.2)
conn.commit()
end = time.monotonic_ns()
print(begin, end, flush=True)
sys.stdin.readline()
conn.close()
""", str(database)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        writer.stdin.write("start\n")
        writer.stdin.flush()
        begin, end = map(int, writer.stdout.readline().split())
        time.sleep(0.003)
        observed = observer.stop()
        periods = monitor.held_periods(observed["samples"], begin - 10_000_000, end + 10_000_000)
        assert len(periods) == 1, "STOP: monitor cannot see a known WAL hold"
        assert {pid for _, pid in observed["samples"] if pid} == {writer.pid}, "STOP: unconfirmed flock layout"
        duration = (periods[0][1] - periods[0][0]) / 1e6
        tolerance = 5 * observed["intervals"]["max_ms"]
        assert abs(duration - (end - begin) / 1e6) <= tolerance
        print("monitor control:", json.dumps({"held_ms": duration, **observed["intervals"]}))
    finally:
        writer.communicate("stop\n", timeout=10)
    assert writer.returncode == 0
    observer = monitor.Monitor(database)
    conn.execute("BEGIN")
    conn.execute("SELECT * FROM control").fetchall()
    time.sleep(0.02)
    conn.rollback()
    observed = observer.stop()
    conn.close()
    assert not any(pid for _, pid in observed["samples"])


@pytest.mark.parametrize("gap,expected", [(149_999, 1), (150_000, 2), (150_001, 2)])
def test_monitor_merges_only_sub_poll_gaps(gap: int, expected: int) -> None:
    from bench.instruments.busy_bar import monitor

    samples = [[0, 0], [100, 123], [1000, 0], [1000 + gap, 123], [500_000, 0]]
    periods = monitor.held_periods(samples, 0, 600_000)
    assert len(periods) == expected
    clipped = monitor.held_periods(samples, 200, 400_000)
    assert clipped[0][0] == 200 and clipped[-1][1] == 400_000


@pytest.mark.parametrize("code,busy", [(5, True), (6, True), (517, True), (1, False)])
@pytest.mark.parametrize("link", [None, "__cause__", "__context__"])
def test_sqlite_code_classification_and_run_inconclusive(code: int, busy: bool, link: str | None) -> None:
    error: Exception = sqlite_error(code, "busy locked text does not decide")
    if link:
        wrapper = RuntimeError("private text")
        setattr(wrapper, link, error)
        error = wrapper
    operations = driver.Operations(driver.Capture(), False)

    def fail() -> None:
        raise error

    operations(SimpleNamespace(), "A", "ingest", fail, 2)
    row = operations.records[0]
    assert row["busy"] is busy
    assert row["inconclusive"] is not busy
    assert row["exceptions"][-1]["sqlite_errorcode"] == code
    # A cycle terminates and classification still finds the SQLite code.
    error.__context__ = error
    assert driver.is_busy(error) is (busy if link != "__context__" else False)


def test_sqlite_words_without_errorcode_are_inconclusive() -> None:
    operations = driver.Operations(driver.Capture(), False)

    def fail() -> None:
        raise sqlite3.OperationalError("busy locked")

    operations(SimpleNamespace(), "A", "compress", fail, 2)
    assert operations.records[0]["busy"] is False
    assert operations.records[0]["inconclusive"] is True


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("other_thread", [False, True])
def test_debug_caught_exception_attribution(wrapped: bool, other_thread: bool) -> None:
    capture = driver.Capture()
    operations = driver.Operations(capture, False)
    logger = logging.getLogger("hermes_lcm.test-busy-bar")
    previous = logging.getLogger().level
    # debug_capture lowers every hermes_lcm logger; restore them all, or later tests on this worker run at DEBUG.
    lcm_levels = {name: logging.getLogger(name).level for name in list(logging.Logger.manager.loggerDict)
                  if name == "hermes_lcm" or name.startswith("hermes_lcm.")}
    driver.debug_capture(capture)

    def caught(code: int) -> None:
        error: Exception = sqlite_error(code)
        if wrapped:
            error = RuntimeError("wrapper")
            error.__cause__ = sqlite_error(code)
        try:
            raise error
        except Exception:
            logger.debug("LCM successful compaction telemetry update failed", exc_info=True)

    def call() -> None:
        if other_thread:
            thread = threading.Thread(target=lambda: caught(517))
            thread.start()
            thread.join()
        else:
            caught(517)
            caught(517)  # Counts once.

    try:
        caught(517)  # Outside the judged call.
        operations(SimpleNamespace(), "A", "compress", call, 2)
        operations(SimpleNamespace(), "A", "session-end", lambda: caught(1), 2)
        operations(SimpleNamespace(), "A", "ingest", lambda: None, 2)
    finally:
        logging.getLogger().removeHandler(capture)
        logging.getLogger().setLevel(previous)
        for name in list(logging.Logger.manager.loggerDict):
            if name == "hermes_lcm" or name.startswith("hermes_lcm."):
                logging.getLogger(name).setLevel(lcm_levels.get(name, logging.NOTSET))
    assert [row["busy"] for row in operations.records] == [not other_thread, False, False]
    assert not any(row["inconclusive"] for row in operations.records)
    assert all(row["level"] == "DEBUG" for row in capture.records)


@pytest.mark.parametrize("base,candidate,expected", [(0, 1, "GATE PASS"), (0, 2, "GATE FAIL"),
                                                      (4, 6, "GATE PASS"), (4, 7, "GATE FAIL")])
def test_gate_exact_tolerance(base: int, candidate: int, expected: str) -> None:
    def rows(count: int) -> list[dict]:
        result = ledger(0)
        # Only the first cell (nine prescribed operations) has busy operations.
        for record in result[0]["operations"][:count]:
            record["busy"] = True
        return result

    cb, cc = rows(0), rows(0)
    for runs in (cb, cc):
        for row in runs:
            row["busy_timeout_ms"] = 100
            for record in row["operations"]:
                record["busy"] = True
    assert evaluate(rows(base), rows(candidate), cb, cc)["verdict"] == smoke(expected)


def test_control_requires_every_class_in_each_tree() -> None:
    scaled = control(1)
    for row in scaled:
        for record in row["operations"]:
            if record["op"] == "session-end":
                record["busy"] = False
    result = evaluate(ledger(0), ledger(0), control(1), scaled)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("candidate has 0 busy session-end" in reason for reason in result["reasons"])


def test_one_nonbusy_raise_makes_gate_inconclusive() -> None:
    candidate = ledger(0)
    candidate[0]["operations"][0]["inconclusive"] = True
    assert evaluate(ledger(0), candidate, control(1), control(1))["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("base,candidate,extended_base,extended_candidate,expected", [
    (1, 3, 4, 12, "GATE FAIL"), (1, 3, 4, 6, "GATE PASS"), (8, 9, 8, 9, "GATE PASS"),
])
def test_small_count_extension(base: int, candidate: int, extended_base: int,
                               extended_candidate: int, expected: str) -> None:
    def runs(count: int, total_runs: int) -> list[dict]:
        rows = []
        for number in range(total_runs):
            for row in ledger(0):
                if number >= 20 and row["cell"] != "ingest":
                    continue
                row["run"], row["seed"] = number, 2055 + number
                if row["cell"] == "ingest" and number < count:
                    row["operations"][0]["busy"] = True
                rows.append(row)
        return rows

    baseline, cand = runs(base, 20), runs(candidate, 20)
    should_extend = base < 8 and candidate > base
    assert gate.extension_cells(baseline, cand) == (["ingest"] if should_extend else [])
    cb, cc = deepcopy(baseline), deepcopy(baseline)
    for scaled in (cb, cc):
        for row in scaled:
            row["busy_timeout_ms"] = 100
            for record in row["operations"]:
                record["busy"] = True
    if should_extend:
        assert evaluate(baseline, cand, cb, cc)["verdict"] == "INCONCLUSIVE"
        # Preserve first twenty counts; place the additional busy records in extension runs.
        def extend(first: list[dict], initial_busy: int, final_busy: int) -> list[dict]:
            extra = runs(0, 80)[80:]
            for row in extra[:final_busy - initial_busy]:
                row["operations"][0]["busy"] = True
            return first + extra
        baseline = extend(baseline, base, extended_base)
        cand = extend(cand, candidate, extended_candidate)
    result = evaluate(baseline, cand, cb, cc)
    assert result["verdict"] == expected
    assert result["initial_tables"]["baseline"]["ingest"]["total"]["busy"] == base
    assert result["tables"]["candidate"]["ingest"]["total"]["busy"] == extended_candidate
    if should_extend:
        assert "20-run busy" in gate.report(result)


def test_hold_position_and_rotating_longest_statistics() -> None:
    def runs(mode: str) -> list[dict]:
        rows = []
        for number in range(10):
            calls = [{"shape": "n", "position": position, "longest_ms":
                      3 if mode == "first" and position == 1 else
                      5 if mode == "rotating" and position == number % 6 + 1 else 1,
                      "first_start_ms": 0, "call_ms": 10} for position in range(1, 7)]
            rows.append({"journal_mode": "wal", "rows": 2000, "calls": calls + [
                {"shape": "t", "longest_ms": 1, "first_start_ms": 0, "call_ms": 10}]})
        return rows

    first = summarize({"baseline": runs("base"), "candidate": runs("first")})
    assert first["verdict"] == "LOCK-TRIPWIRE FAIL"
    assert first["failed_statistics"] == ["position-1", "run-longest"]
    rotating = summarize({"baseline": runs("base"), "candidate": runs("rotating")})
    assert rotating["failed_statistics"] == ["run-longest"]
    assert "position-1" in holds.report(first)
    diagnostic = runs("base")
    diagnostic[0]["trace_diagnostic"] = True
    assert summarize({"baseline": runs("base"), "candidate": diagnostic})["verdict"] == "DIAGNOSTIC ONLY"


def test_isolation_and_shipped_default_readback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    monkeypatch.setenv("HERMES_TEST", "inherited")
    with patch.dict(os.environ):
        driver.isolated_environment(tmp_path)
        assert os.environ["HERMES_HOME"] == str(tmp_path / "isolated-home")
        assert "LCM_HOST_MESSAGE_UID" not in os.environ and "HERMES_TEST" not in os.environ
    values = driver.effective_defaults(SimpleNamespace(host_message_uid_mode=lambda: "shadow"))
    assert values == {"LCM_HOST_MESSAGE_UID": "shadow", "LCM_IDENTITY_ANCHOR": "absent", "LCM_INGEST_LEDGER": "absent"}


def test_orchestrator_extends_only_selected_production_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def child(command: list[str], **kwargs: object) -> SimpleNamespace:
        name, cell, output = Path(command[3]).name, command[4], Path(command[5])
        seed = int(command[command.index("--seed") + 1])
        number = seed - 2055
        scaled = "--busy-timeout-ms" in command
        seen.append((name, cell, number, scaled))
        row = next(row for row in (control(1) if scaled else ledger(0)) if row["cell"] == cell)
        row["seed"] = seed
        row["tree_identity"] = driver.BASELINE_IDENTITY if name == "baseline" else CANDIDATE_IDENTITY
        if not scaled and cell == "ingest":
            first_busy = 1 if name == "baseline" else 3
            row["operations"][0]["busy"] = number < first_busy or 20 <= number < 23
        output.write_text(json.dumps(row) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrate.subprocess, "run", child)
    out = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["orchestrate", "baseline", "candidate", str(out),
                                       "--work-root", str(tmp_path / "scratch")])
    with pytest.raises(SystemExit) as exc:
        orchestrate.main()
    assert exc.value.code == 0
    result = json.loads((out / "verdict.json").read_text())
    assert result["extension_cells"] == ["ingest"]
    assert result["initial_tables"]["baseline"]["ingest"]["total"]["busy"] == 1
    assert result["initial_tables"]["candidate"]["ingest"]["total"]["busy"] == 3
    assert result["tables"]["baseline"]["ingest"]["total"]["busy"] == 4
    assert result["tables"]["candidate"]["ingest"]["total"]["busy"] == 6
    assert len(gate.read(out / "baseline.jsonl")) == 140
    extra = [entry for entry in seen if entry[2] >= 20]
    assert len(extra) == 120
    assert all(cell == "ingest" and not scaled for _, cell, _, scaled in extra)
    assert extra[:4] == [("baseline", "ingest", 20, False), ("candidate", "ingest", 20, False),
                         ("candidate", "ingest", 21, False), ("baseline", "ingest", 21, False)]
    assert {number for _, _, number, scaled in seen if scaled} == set(range(20))


def test_only_a_full_session_gates() -> None:
    def session(runs: int, k: int = 2) -> list[dict]:
        rows = []
        for number in range(runs):
            for row in ledger(0, k):
                row["run"], row["seed"] = number, 2055 + number
                rows.append(row)
        return rows

    def scaled(rows: list[dict]) -> list[dict]:
        rows = deepcopy(rows)
        for row in rows:
            row["busy_timeout_ms"] = 100
            for record in row["operations"]:
                record["busy"] = True
        return rows

    full = evaluate(session(20), session(20), scaled(session(20)), scaled(session(20)))
    assert full["verdict"] == "GATE PASS" and full["gate_session"]
    short = evaluate(session(19), session(19), scaled(session(19)), scaled(session(19)))
    assert short["verdict"] == "SMOKE PASS" and not short["gate_session"]
    assert any("has 19 runs, a gate session needs 20" in reason for reason in short["reasons"])
    assert gate.EXIT[short["verdict"]] == 3
    for key, value in (("b_view_rows", 2), ("k", 1), ("window_ms", 0), ("seed", 7)):
        reduced = [dict(row, **{key: value}) for row in session(20, value if key == "k" else 2)]
        result = evaluate(reduced, reduced, scaled(reduced), scaled(reduced))
        assert result["verdict"] == "SMOKE PASS" and not result["gate_session"], key
        assert any(f"{key}={value!r}, the gate workload is" in reason for reason in result["reasons"]), key


def test_blind_or_short_hold_sessions_never_gate() -> None:
    def runs(multiplier: float, count: int = 10, blind: bool = False) -> list:
        return [{"journal_mode": "wal", "rows": 2000, "calls": [
            {"shape": "n", "position": position, "longest_ms": 0 if blind else n * multiplier,
             "first_start_ms": 2, "call_ms": 30} for position in range(1, 7)] + [
            {"shape": "t", "position": 1, "longest_ms": 0, "first_start_ms": None, "call_ms": 30}]}
            for n in range(1, count + 1)]

    blind = summarize({"baseline": runs(1, blind=True), "candidate": runs(1, blind=True)})
    assert blind["verdict"] == "INCONCLUSIVE"
    assert any("no observed hold" in reason for reason in blind["reasons"])
    assert summarize({"baseline": runs(1), "candidate": runs(1, blind=True)})["verdict"] == "INCONCLUSIVE"
    short = summarize({"baseline": runs(1, 3), "candidate": runs(1, 3)})
    assert short["verdict"] == "LOCK-SMOKE PASS"
    assert summarize({"baseline": runs(1, 3), "candidate": runs(3, 3)})["verdict"] == "LOCK-SMOKE FAIL"
    assert summarize({"baseline": runs(1), "candidate": runs(1)})["verdict"] == "LOCK-TRIPWIRE PASS"
    small = {"baseline": runs(1), "candidate": runs(1)}
    for run in small["candidate"]:
        run["rows"] = 2
    reduced = summarize(small)
    assert reduced["verdict"] == "LOCK-SMOKE PASS"
    assert any("a gate session uses 2000" in reason for reason in reduced["reasons"])


def hold_runs(count: int = 10) -> list[dict]:
    return [{"journal_mode": "wal", "rows": 2000, "calls": [
        {"shape": "n", "position": position, "longest_ms": n, "first_start_ms": 2, "call_ms": 30}
        for position in range(1, 7)] + [
        {"shape": "t", "position": 1, "longest_ms": n, "first_start_ms": 2, "call_ms": 30}]}
        for n in range(1, count + 1)]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("scope", ["one", "all"])
@pytest.mark.parametrize("tree", ["baseline", "candidate"])
def test_nonfinite_holds_are_inconclusive(tree: str, scope: str, value: float) -> None:
    collected = {"baseline": hold_runs(), "candidate": hold_runs()}
    calls = [call for run in collected[tree] for call in run["calls"] if call["shape"] == "n"]
    for call in calls[:1] if scope == "one" else calls:
        call["longest_ms"] = value
    result = summarize(collected)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any(reason.startswith(f"{tree}: ") and "nonfinite" in reason for reason in result["reasons"])


@pytest.mark.parametrize("damage", ["copies", "run-id", "view-rows", "rows-stored", "ingest-failure",
                                    "failure-unrecorded", "monitor-failed", "monitor-short", "call-times"])
def test_hold_summary_needs_distinct_proven_runs(damage: str) -> None:
    collected = {"baseline": hold_runs(), "candidate": hold_runs()}
    run, call = collected["candidate"][0], collected["candidate"][0]["calls"][0]
    if damage == "copies":
        run["run_id"] = "one-run"
        collected["candidate"] = [deepcopy(run) for _ in range(10)]
    elif damage == "run-id":
        run["run_id"] = None
    elif damage == "view-rows":
        call["view_rows"] = 2
    elif damage == "rows-stored":
        call["rows_stored"] = 0
    elif damage == "ingest-failure":
        run["calls"][-1]["ingest_failures"] = 1
    elif damage == "failure-unrecorded":
        call["ingest_failures"] = None
    elif damage == "monitor-failed":
        run["monitor"] = {}
    elif damage == "monitor-short":
        run["monitor"] = {"samples": [[0, 0], [20, 0]], "intervals": {"max_ms": 0.2}}
    else:
        call["begin_ns"] = None
    assert summarize({"baseline": hold_runs(), "candidate": hold_runs()})["verdict"] == "LOCK-TRIPWIRE PASS"
    result = summarize(collected)
    assert result["verdict"] == "INCONCLUSIVE", result["reasons"]


@pytest.mark.parametrize("damage", ["baseline", "missing", "mixed", "same"])
def test_hold_tree_identity(damage: str) -> None:
    collected = {"baseline": hold_runs(), "candidate": hold_runs()}
    if damage == "baseline":
        for run in collected["baseline"]:
            run["tree_identity"] = {"plugin_version": "0.29.1", "product_sha256": "1" * 64}
    elif damage == "missing":
        collected["candidate"][0]["tree_identity"] = None
    elif damage == "mixed":
        collected["candidate"][0]["tree_identity"] = {"plugin_version": "other", "product_sha256": "2" * 64}
    else:
        for run in collected["candidate"]:
            run["tree_identity"] = driver.BASELINE_IDENTITY
    result = summarize(collected)
    assert result["verdict"] == "INCONCLUSIVE"
    if damage == "baseline":
        assert any("not the v0.29.2 GA tree" in reason for reason in result["reasons"])
        override = summarize(collected, baseline_identity_override=True)
        assert override["verdict"] == "LOCK-SMOKE PASS"
        assert any("baseline identity override" in reason for reason in override["reasons"])


def gate_session(k: int = 2) -> list[dict]:
    rows = []
    for number in range(20):
        for row in ledger(0, k):
            row["run"], row["seed"] = number, 2055 + number
            rows.append(row)
    return rows


def busy_control(rows: list[dict]) -> list[dict]:
    rows = deepcopy(rows)
    for row in rows:
        row["busy_timeout_ms"] = 100
        for record in row["operations"]:
            record["busy"] = True
    return rows


@pytest.mark.parametrize("armed_role", range(4))
@pytest.mark.parametrize("fails", [False, True])
def test_storage_arms_never_gate(tmp_path, monkeypatch, capsys, armed_role, fails):
    baseline, candidate = gate_session(), gate_session()
    if fails:
        # Baseline >=8 avoids an extension; candidate exceeds ceil(1.25 * 20) + 1.
        for row in baseline:
            row["operations"][0]["busy"] = True
        for row in candidate:
            for record in row["operations"]:
                record["busy"] = True
    roles = [stamp(baseline, driver.BASELINE_IDENTITY), stamp(candidate, CANDIDATE_IDENTITY),
             stamp(busy_control(baseline), driver.BASELINE_IDENTITY),
             stamp(busy_control(candidate), CANDIDATE_IDENTITY)]
    unarmed = gate.evaluate(*roles)
    assert unarmed["gate_session"]
    assert unarmed["verdict"] == ("GATE FAIL" if fails else "GATE PASS")
    assert gate.EXIT[unarmed["verdict"]] == (1 if fails else 0)
    roles[armed_role][-1]["storage_arms"] = {"record_holds": True, "background_writers": "all"}
    result = gate.evaluate(*roles)
    assert result["verdict"] == ("SMOKE FAIL" if fails else "SMOKE PASS")
    assert not result["gate_session"]
    assert any('"background_writers": "all"' in reason and '"record_holds": true' in reason
               for reason in result["reasons"])
    paths = [tmp_path / f"role-{role}.jsonl" for role in range(4)]
    for path, rows in zip(paths, roles):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(sys, "argv", ["gate", str(paths[0]), str(paths[1]),
                                     "--control-baseline", str(paths[2]), "--control-candidate", str(paths[3])])
    with pytest.raises(SystemExit) as exc:
        gate.main()
    assert exc.value.code == 3
    output = capsys.readouterr().out
    assert result["verdict"] in output and "GATE PASS" not in output and "GATE FAIL" not in output


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_storage_arms_reasons_per_ledger(tmp_path, monkeypatch, capsys, fails, mixed):
    baseline, candidate = gate_session(), gate_session()
    if fails:
        for row in baseline:
            row["operations"][0]["busy"] = True
        for row in candidate:
            for record in row["operations"]:
                record["busy"] = True
    roles = [stamp(baseline, driver.BASELINE_IDENTITY), stamp(candidate, CANDIDATE_IDENTITY),
             stamp(busy_control(baseline), driver.BASELINE_IDENTITY),
             stamp(busy_control(candidate), CANDIDATE_IDENTITY)]
    paths = [tmp_path / f"role-{role}.jsonl" for role in range(4)]

    def cli():
        for path, rows in zip(paths, roles):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        monkeypatch.setattr(sys, "argv", ["gate", str(paths[0]), str(paths[1]),
                                         "--control-baseline", str(paths[2]), "--control-candidate", str(paths[3])])
        with pytest.raises(SystemExit) as exc:
            gate.main()
        return capsys.readouterr().out, exc.value.code

    unarmed, status = cli()
    assert status == (1 if fails else 0)
    # Byte-for-byte round-2 CLI report fingerprints, including tables and verdict.
    expected = ("07fb3b928b9e531631bbd81d0bf26fdef6df32a8aec27e70305131976295f934" if fails else
                "57dc0a9f7e42060adf275e03f597e4ae85b6c7474d277fb193b916ede523d375")
    assert hashlib.sha256(unarmed.encode()).hexdigest() == expected
    for rows in roles:
        for i, row in enumerate(rows):
            row["storage_arms"] = {"wal_autocheckpoint": 0, "checkpointer": "passive:10",
                "background_writers": "stem" if mixed and i % 2 else "all", "stem_backlog": 1001,
                "record_holds": True,
                "configured_connections": [{"wal_autocheckpoint": 0, "synchronous": 2}] * (i % 3 + 1),
                "schedule_start": 123456.125 + i, "schedule_end": 123456.625 + i}
    output, status = cli()
    (tmp_path / "verdict.txt").write_text(output)
    assert status == 3
    assert ("SMOKE FAIL" if fails else "SMOKE PASS") in output
    assert "GATE PASS" not in output and "GATE FAIL" not in output
    reasons = [line for line in output.splitlines() if "storage_arms=" in line]
    assert len(reasons) == 4
    for name in ("baseline", "candidate", "control baseline", "control candidate"):
        reason, = [line for line in reasons if line.startswith(name + ":")]
        assert '"background_writers": "all"' in reason and '"record_holds": true' in reason
        assert ("distinct arm sets" in reason) == mixed
        if mixed:
            assert '"background_writers": "stem"' in reason
    assert "schedule_start" not in output and "schedule_end" not in output
    assert "123456." not in output and "configured_connections" not in output


@pytest.mark.parametrize("value", [None, {}, {"checkpointer": "passive:10"}, {"wal_autocheckpoint": 0}])
def test_storage_arms_presence_is_diagnostic(value):
    rows = gate_session()
    rows[-1]["storage_arms"] = value
    result = evaluate(rows, rows, busy_control(rows), busy_control(rows))
    assert result["verdict"] == "SMOKE PASS" and not result["gate_session"]


@pytest.mark.parametrize("removed", [("B", "ingest"), ("A", "compress"), ("A", "session-end")])
def test_equally_incomplete_workloads_never_pass(removed: tuple) -> None:
    full = gate_session()
    assert evaluate(full, full, busy_control(full), busy_control(full))["verdict"] == "GATE PASS"
    reduced = deepcopy(full)
    for row in reduced:
        row["operations"] = [record for record in row["operations"]
                             if (record["engine"], record["op"]) != removed]
    result = evaluate(reduced, deepcopy(reduced), busy_control(reduced), busy_control(reduced))
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("not the prescribed operation sequence" in reason for reason in result["reasons"])


def test_prescribed_operations_follow_the_driver_schedule() -> None:
    for cell in cells.CELLS:
        for k in (1, 2):
            found = driver.prescribed_operations(cell, k)
            assert sum(count for (owner, _), count in found.items() if owner == "A") == cells.A_OPERATIONS[cell]
            assert found[("B", "ingest")] == k * cells.A_OPERATIONS[cell]
    assert driver.prescribed_operations("compress", 2)[("A", "compress")] == 1
    assert driver.prescribed_operations("session-end", 2)[("A", "session-end")] == 1


@pytest.mark.parametrize("damage", ["baseline", "control", "extension", "missing", "same"])
def test_gate_tree_identity(damage: str) -> None:
    baseline, candidate = gate_session(), gate_session()
    cb, cc = busy_control(baseline), busy_control(candidate)
    other = {"plugin_version": "0.29.1", "product_sha256": "1" * 64}
    if damage == "baseline":
        for row in baseline + cb:
            row["tree_identity"] = other
    elif damage == "control":
        for row in cb:
            row["tree_identity"] = other
    elif damage == "extension":
        baseline[-1]["tree_identity"] = other
    elif damage == "missing":
        candidate[0]["tree_identity"] = None
    else:
        for row in candidate + cc:
            row["tree_identity"] = driver.BASELINE_IDENTITY
    result = evaluate(baseline, candidate, cb, cc)
    assert result["verdict"] == "INCONCLUSIVE"
    if damage == "baseline":
        assert any("not the v0.29.2 GA tree" in reason for reason in result["reasons"])
        override = evaluate(baseline, candidate, cb, cc, baseline_identity_override=True)
        assert override["verdict"] == "SMOKE PASS" and not override["gate_session"]
        assert gate.EXIT[override["verdict"]] == 3


def test_tree_identity_hashes_product_python_only(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    for name, text in (("plugin.yaml", "name: x\nversion: 0.29.2\n"), ("engine.py", "A = 1\n"),
                       ("teams/store.py", "B = 1\n"), ("tests/test_x.py", "C = 1\n"),
                       ("bench/tool.py", "D = 1\n"), ("__pycache__/engine.py", "E = 1\n"),
                       (".hidden/x.py", "F = 1\n")):
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(text)
    first = driver.tree_identity(tree)
    assert first["plugin_version"] == "0.29.2" and len(first["product_sha256"]) == 64
    for name in ("tests/test_x.py", "bench/tool.py", "__pycache__/engine.py", ".hidden/x.py"):
        (tree / name).write_text("changed = 1\n")
    assert driver.tree_identity(tree) == first
    (tree / "teams/store.py").write_text("B = 2\n")
    assert driver.tree_identity(tree)["product_sha256"] != first["product_sha256"]
