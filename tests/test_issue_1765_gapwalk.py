"""#1765: SQL-eligible whitespace consumes budget without stranding history."""
import ast
import hashlib
import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from hermes_lcm import command
from hermes_lcm.db_bootstrap import ensure_chunk_tables
from tests import _issue_1748_base_gapfill as frozen
from tests.test_issue_1748_gapwalk import discover

MAIN = "5dd67c0dab62f409beec5f86951181f3d2affb1c"


class FetchedCursor(sqlite3.Cursor):
    def fetchall(self):
        rows = super().fetchall()
        self.connection.fetched += len(rows)
        return rows


class FetchedConnection(sqlite3.Connection):
    fetched = 0

    def execute(self, sql, parameters=()):
        if "SELECT store_id, role, content FROM messages AS m " in sql:
            return self.cursor(factory=FetchedCursor).execute(sql, parameters)
        return super().execute(sql, parameters)


def store(path, high, blank_count=0):
    conn = sqlite3.connect(path, factory=FetchedConnection)
    conn.execute("CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT)")
    ensure_chunk_tables(conn)
    conn.executemany("INSERT INTO messages VALUES(?, 'user', ?)",
                     ((i, "\n" if 13_001 <= i < 13_001 + blank_count else "Synthetic fact.")
                      for i in range(1, high + 1)))
    publish(conn, [(1,), (high,)])
    return conn


def publish(conn, rows):
    # No IGNORE: duplicate selection/publication must fail the regression.
    conn.executemany("INSERT INTO lcm_chunk_meta(chunk_id, identity_hash, store_id) "
                     "VALUES(?, 'synthetic', ?)", ((f"{r[0]}:0", r[0]) for r in rows))


@pytest.mark.parametrize("blank_count", [1999, 2000, 2001])
def test_long_whitespace_run_drains_with_published_arrivals(tmp_path, blank_count):
    with closing(store(tmp_path / "drain.db", 20_000, blank_count)) as conn, conn:
        expected = set(range(2, 20_000)) - set(range(13_001, 13_001 + blank_count))
        remaining = []
        # Ten ordinary pages plus at most two whitespace/boundary pages.
        for turn in range(1, 13):
            high = 20_000 + turn
            conn.execute("INSERT INTO messages VALUES(?, 'user', 'Published arrival.')", (high,))
            publish(conn, [(high,)])
            rows = discover(conn, "synthetic", high)
            ids = [r[0] for r in rows]
            assert len(ids) == len(set(ids)) and set(ids) <= expected
            publish(conn, rows)
            expected.difference_update(ids)
            remaining.append(len(expected))
            if not expected:
                break
        print("WHITESPACE_DRAIN " + json.dumps(dict(blank_count=blank_count, remaining=remaining)))
        assert not expected, remaining
        have = {r[0] for r in conn.execute("SELECT store_id FROM lcm_chunk_meta")}
        assert have == set(range(1, high + 1)) - set(range(13_001, 13_001 + blank_count))


@pytest.mark.parametrize("page_size", [32, 2000])
def test_whitespace_arrival_interval_counts_fetched_rows(tmp_path, page_size):
    with closing(store(tmp_path / "top.db", 4000)) as conn, conn:
        publish(conn, discover(conn, "synthetic", 4000, limit=32))
        conn.executemany("INSERT INTO messages VALUES(?, 'user', '\n')",
                         ((i,) for i in range(4001, 6002)))
        conn.execute("INSERT INTO messages VALUES(6002, 'user', 'Published arrival.')")
        publish(conn, [(6002,)])
        conn.fetched = 0
        state = {}
        rows = list(command._chunk_gapfill_rows(
            conn, "synthetic", (1, 6002), "conversational", 2000,
            record_progress=True, gap_page_size=page_size, gap_scan_state=state))
        command._record_chunk_gapfill_walk(conn, "synthetic", state)
        assert conn.fetched == 2000
        assert rows == []
        # This partial top segment must itself resume under subsequent arrivals.
        saved = json.loads(conn.execute("SELECT value FROM metadata WHERE key='chunk_gapfill_scan:synthetic'")
                           .fetchone()[0])
        assert saved["walk"]["below"] == 4002 and saved["walk"]["pending"] == []
        conn.execute("INSERT INTO messages VALUES(6003, 'user', 'Next published arrival.')")
        publish(conn, [(6003,)])
        rows = discover(conn, "synthetic", 6003)
        assert rows and rows[0][0] == 3967


@pytest.mark.parametrize("drift", ["revision", "eligibility"])
def test_frozen_base_rejects_live_policy_drift(tmp_path, monkeypatch, drift):
    with closing(store(tmp_path / "frozen.db", 4)) as conn, conn:
        if drift == "revision":
            monkeypatch.setattr(frozen, "CHUNK_POLICY_REVISION", 3)
        else:
            monkeypatch.setattr(frozen, "chunk_message", lambda *a, **kw: [])
        with pytest.raises(AssertionError, match="frozen base"):
            list(frozen._chunk_gapfill_rows(conn, "synthetic", (1, 4), "conversational", 2000))


def pinned_main_gapfill():
    source = subprocess.run(["git", "show", f"{MAIN}:command.py"],
                            cwd=Path(__file__).resolve().parents[1],
                            check=True, capture_output=True, text=True).stdout
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "_chunk_gapfill_rows")
    namespace = dict(vars(command))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_main_gapfill", "exec"), namespace)
    return namespace["_chunk_gapfill_rows"]


@pytest.mark.parametrize("shape", ["prefix250k", "live100k"])
def test_ordinary_shapes_match_main_output_and_vm_steps(tmp_path, monkeypatch, shape):
    """#1758's generated prefix and live-arrival shapes; no provider or wall clock."""
    baseline, candidate = pinned_main_gapfill(), command._chunk_gapfill_rows
    high = 250_000 if shape == "prefix250k" else 100_000
    with closing(store(tmp_path / "ordinary.db", high)) as conn, conn:
        filled = 0
        cells = [0, 50_000, 240_000] if shape == "prefix250k" else [60_000] * 21
        for index, target in enumerate(cells):
            while filled < target:
                rows = discover(conn, "synthetic", high, min(2000, target - filled))
                assert rows
                publish(conn, rows)
                filled += len(rows)
            if shape == "live100k" and index:
                high += 1
                conn.execute("INSERT INTO messages VALUES(?, 'user', 'Published arrival.')", (high,))
                publish(conn, [(high,)])
            arms = {}
            outputs = {}
            for arm, implementation in (("main", baseline), ("candidate", candidate)):
                conn.execute("SAVEPOINT compare")
                steps = 0

                def step():
                    nonlocal steps
                    steps += 1
                    return 0

                conn.set_progress_handler(step, 1)
                try:
                    with monkeypatch.context() as patch:
                        patch.setattr(command, "_chunk_gapfill_rows", implementation)
                        rows = discover(conn, "synthetic", high)
                finally:
                    conn.set_progress_handler(None, 0)
                outputs[arm] = [(c.chunk_id, c.text.encode("utf-8")) for row in rows
                                for c in command.chunk_message(*row, policy="conversational")]
                digest = hashlib.sha256()
                for chunk_id, content in outputs[arm]:
                    digest.update(json.dumps([chunk_id, content.decode("utf-8")]).encode("utf-8"))
                arms[arm] = dict(vm_steps=steps, chunks=len(outputs[arm]), sha256=digest.hexdigest())
                conn.execute("ROLLBACK TO compare")
                conn.execute("RELEASE compare")
            print("ORDINARY_IDENTITY " + json.dumps(dict(shape=shape, pass_index=index, filled=target, arms=arms)))
            assert outputs["candidate"] == outputs["main"]
            assert arms["candidate"]["vm_steps"] <= arms["main"]["vm_steps"]
            # Keep a real persistent walk across the twenty arrival passes.
            if shape == "live100k":
                publish(conn, discover(conn, "synthetic", high))
