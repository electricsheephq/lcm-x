"""Release-predicate controls plus offline, public-API smoke coverage."""

from __future__ import annotations

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


def ledger(busy: int) -> list[dict]:
    return [{"cell": cell, "run": 0, "k": 2, "b_view_rows": 2000, "busy_timeout_ms": None, "seed": 2055, "window_ms": 400, "offsets_ms": [1, 2, 3], "errors": [], "operations": [
        {"engine": owner, "op": ("ingest", "compress", "session-end")[index], "busy": index < busy}
        for owner in ("A", "B") for index in range(3)]} for cell in cells.CELLS]


def control(busy: int) -> list[dict]:
    rows = [{**row, "busy_timeout_ms": 100} for row in ledger(0)]
    for row in rows:
        for record in row["operations"]:
            record["busy"] = busy > 0 and (record["engine"] == "A" or busy > 1)
    return rows


@pytest.mark.parametrize("base,candidate,expected", [
    (1, 1, "GATE PASS"), (2, 1, "GATE PASS"), (1, 2, "GATE PASS"), (1, 3, "GATE FAIL"),
    (0, 0, "GATE PASS"), (0, 1, "GATE FAIL"),
])
def test_gate_counts(base: int, candidate: int, expected: str) -> None:
    result = gate.evaluate(ledger(base), ledger(candidate), control(1), control(1))
    assert result["verdict"] == expected
    rendered = gate.report(result)
    assert expected in rendered
    assert "TOTAL total 24" in rendered


def test_gate_regression_in_one_cell_cannot_hide_in_total() -> None:
    baseline, candidate = ledger(1), ledger(0)
    candidate[0] = ledger(3)[0]
    result = gate.evaluate(baseline, candidate, control(1), control(1))
    assert result["verdict"] == "GATE FAIL"
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
    assert gate.evaluate(baseline, candidate, control(1), control(1))["verdict"] == "INCONCLUSIVE"


def test_gate_cli_writes_verdict(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("baseline.jsonl", "candidate.jsonl", "control-base.jsonl", "control-cand.jsonl")]
    for index, path in enumerate(paths):
        path.write_text("".join(json.dumps(row) + "\n" for row in (ledger(1) if index < 2 else control(1))))
    result = subprocess.run([sys.executable, str(Path(gate.__file__)), *map(str, paths[:2]),
                             "--control-baseline", str(paths[2]), "--control-candidate", str(paths[3])],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "GATE PASS" in result.stdout


def test_orchestrator_serial_interleaving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def child(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen.append(Path(command[3]).name)
        cell, output = command[4], Path(command[5])
        row = next(row for row in (control(1) if "--busy-timeout-ms" in command else ledger(1))
                   if row["cell"] == cell)
        output.write_text(json.dumps(row) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrate.subprocess, "run", child)
    out = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["orchestrate", "baseline", "candidate", str(out), "--runs", "2",
                                       "--work-root", str(tmp_path / "scratch")])
    with pytest.raises(SystemExit) as exc:
        orchestrate.main()
    assert exc.value.code == 0
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
    result = gate.evaluate(ledger(0), ledger(1), control(0 if missing == "baseline" else 1),
                           control(0 if missing == "candidate" else 1))
    assert result["verdict"] == "INCONCLUSIVE"
    assert any(missing + " has 0 busy" in reason for reason in result["reasons"])


def test_control_busy_regression_never_gates() -> None:
    result = gate.evaluate(ledger(0), ledger(0), control(1), control(3))
    assert result["verdict"] == "GATE PASS"


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
    assert gate.evaluate(ledger(1), ledger(1), control(1), scaled)["verdict"] == "INCONCLUSIVE"


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
        return [{"journal_mode": "wal", "calls": [
            {"shape": "n", "position": position, "longest_ms": n * multiplier,
             "first_start_ms": 2, "call_ms": 30} for position in range(1, 7)] + [
            {"shape": "t", "position": 1, "longest_ms": n * multiplier,
             "first_start_ms": 2, "call_ms": 30}]} for n in range(1, 11)]

    result = holds.summarize({"baseline": runs(1), "candidate": runs(2.25)})
    assert result["verdict"] == "LOCK-TRIPWIRE PASS"
    assert result["tables"]["baseline"]["n"]["p90_longest_ms"] == 9
    assert holds.summarize({"baseline": runs(1), "candidate": runs(2.26)})["verdict"] == "LOCK-TRIPWIRE FAIL"


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
    assert [row["busy"] for row in operations.records] == [not other_thread, False, False]
    assert not any(row["inconclusive"] for row in operations.records)
    assert all(row["level"] == "DEBUG" for row in capture.records)


@pytest.mark.parametrize("base,candidate,expected", [(0, 1, "GATE PASS"), (0, 2, "GATE FAIL"),
                                                      (4, 6, "GATE PASS"), (4, 7, "GATE FAIL")])
def test_gate_exact_tolerance(base: int, candidate: int, expected: str) -> None:
    def rows(count: int) -> list[dict]:
        result = ledger(0)
        # Seven attempts in this fixture; only the first cell has busy operations.
        result[0]["operations"] += [{"engine": "A", "op": "ingest", "busy": False}]
        for record in result[0]["operations"][:count]:
            record["busy"] = True
        return result

    cb, cc = rows(0), rows(0)
    for runs in (cb, cc):
        for row in runs:
            row["busy_timeout_ms"] = 100
            for record in row["operations"]:
                record["busy"] = True
    assert gate.evaluate(rows(base), rows(candidate), cb, cc)["verdict"] == expected


def test_control_requires_every_class_in_each_tree() -> None:
    scaled = control(1)
    for row in scaled:
        for record in row["operations"]:
            if record["op"] == "session-end":
                record["busy"] = False
    result = gate.evaluate(ledger(0), ledger(0), control(1), scaled)
    assert result["verdict"] == "INCONCLUSIVE"
    assert any("candidate has 0 busy session-end" in reason for reason in result["reasons"])


def test_one_nonbusy_raise_makes_gate_inconclusive() -> None:
    candidate = ledger(0)
    candidate[0]["operations"][0]["inconclusive"] = True
    assert gate.evaluate(ledger(0), candidate, control(1), control(1))["verdict"] == "INCONCLUSIVE"


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
        assert gate.evaluate(baseline, cand, cb, cc)["verdict"] == "INCONCLUSIVE"
        # Preserve first twenty counts; place the additional busy records in extension runs.
        def extend(first: list[dict], initial_busy: int, final_busy: int) -> list[dict]:
            extra = runs(0, 80)[80:]
            for row in extra[:final_busy - initial_busy]:
                row["operations"][0]["busy"] = True
            return first + extra
        baseline = extend(baseline, base, extended_base)
        cand = extend(cand, candidate, extended_candidate)
    result = gate.evaluate(baseline, cand, cb, cc)
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
            rows.append({"journal_mode": "wal", "calls": calls + [
                {"shape": "t", "longest_ms": 1, "first_start_ms": 0, "call_ms": 10}]})
        return rows

    first = holds.summarize({"baseline": runs("base"), "candidate": runs("first")})
    assert first["verdict"] == "LOCK-TRIPWIRE FAIL"
    assert first["failed_statistics"] == ["position-1", "run-longest"]
    rotating = holds.summarize({"baseline": runs("base"), "candidate": runs("rotating")})
    assert rotating["failed_statistics"] == ["run-longest"]
    assert "position-1" in holds.report(first)
    diagnostic = runs("base")
    diagnostic[0]["trace_diagnostic"] = True
    assert holds.summarize({"baseline": runs("base"), "candidate": diagnostic})["verdict"] == "DIAGNOSTIC ONLY"


def test_isolation_and_shipped_default_readback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    monkeypatch.setenv("HERMES_TEST", "inherited")
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
