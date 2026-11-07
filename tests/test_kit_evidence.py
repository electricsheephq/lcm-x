"""Synthetic fixtures for public-clean release-kit evidence records."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from bench.instruments.release_gauntlet import kit_evidence as kit


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def _log(logger, text, level="WARNING"):
    return f"2026-01-01 01:02:03,123 [{level}] {logger}: {text}\n"


def _trace(final):
    return 'Traceback (most recent call last):\n  File "synthetic.py", line 1\n    synthetic()\n' + final + "\n"


def _run(count):
    return (_log("agent.background_review", "thread=bg-review OpenAI client created", "INFO")
            + _trace(kit.REFUSAL) * count
            + _log("agent.background_review", "Background review complete: thread=bg-review", "INFO"))


@pytest.mark.parametrize("text,counts,outside,holds", [
    (_run(1) * 3, {"1": 3}, 0, True), (_run(2), {"2": 1}, 0, False),
    (_run(0) + kit.REFUSAL + "\n", {"0": 1}, 1, False),
], ids=["one-per-run", "two-in-run", "outside-run"])
def test_detach_runs(tmp_path, text, counts, outside, holds):
    result = kit.trace_events([_write(tmp_path / "host.log", text)])
    assert result["bg_review_runs"] == sum(counts.values())
    assert result["runs_by_refusal_count"] == counts
    assert result["refusals_total"] == sum(int(n) * count for n, count in counts.items()) + outside
    assert result["refusals_outside_runs"] == outside
    assert result["per_event_rule_holds"] is holds
    assert result["other_tracebacks"] == {}


def test_import_events_and_privacy(tmp_path):
    marker = "SYNTHETIC_MESSAGE_MUST_NOT_ESCAPE"
    text = (_log("agent.relay_runtime", marker)
            + _trace("ModuleNotFoundError: No module named 'optional.module'") * 2
            + _log("agent.optional", marker, "ERROR")
            + _trace("ImportError: " + marker) + _trace("ValueError: " + marker))
    result = kit.trace_events([_write(tmp_path / "host.log", text)])
    assert result["optional_import_traces_total"] == 3
    assert result["by_event"] == [
        {"logger": "agent.optional", "missing_module": None, "count": 1},
        {"logger": "agent.relay_runtime", "missing_module": "optional.module", "count": 2}]
    assert result["other_tracebacks"] == {"ValueError": 1}
    assert result["unattributed"] == 0
    assert marker not in json.dumps(result)
    isolated = _write(tmp_path / "isolated.log", _trace("ModuleNotFoundError: No module named 'alone'"))
    result = kit.trace_events([tmp_path / "host.log", isolated])
    assert result["optional_import_traces_total"] == 4
    assert result["unattributed"] == 1


def _messages(path, order, rows):
    with sqlite3.connect(path) as conn:
        conn.execute(f"CREATE TABLE messages ({order} INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
                     "content TEXT, tool_calls TEXT, tool_call_id TEXT)")
        conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                         [(i, *row) for i, row in enumerate(rows, 1)])
    return path


def test_lifecycle_doctor_rule(tmp_path):
    path = _messages(tmp_path / "lcm.db", "store_id", [("stored", "user", "synthetic content", None, None)])
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE summary_nodes (session_id TEXT)")
        conn.execute("INSERT INTO summary_nodes VALUES ('summarized')")
        conn.execute("CREATE TABLE lcm_lifecycle_state (conversation_id TEXT, current_session_id TEXT, "
                     "last_finalized_session_id TEXT)")
        conn.executemany("INSERT INTO lcm_lifecycle_state VALUES (?, ?, ?)", [
            ("empty", None, None), ("missing", "absent", "also-absent"),
            ("has-message", "stored", "absent"), ("has-node", "absent", "summarized"),
            ("same-ref", "stored", "stored")])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = kit.lifecycle_rows(path)
    assert result["lifecycle_rows"] == 5
    assert result["empty_lifecycle_rows"] == 2
    assert result["affected_stored_rows"] == 0
    assert result["identity_order_check"] == "PASS"
    assert result["empty_rows"][0] == {
        "conversation_id_h12": hashlib.sha256(b"empty").hexdigest()[:12],
        "current_session_h12": None, "last_finalized_session_h12": None,
        "messages_in_referenced_sessions": 0, "nodes_in_referenced_sessions": 0}
    assert result["empty_rows"][1]["current_session_h12"] == hashlib.sha256(b"absent").hexdigest()[:12]
    assert all(row["messages_in_referenced_sessions"] == row["nodes_in_referenced_sessions"] == 0
               for row in result["empty_rows"])
    assert "synthetic content" not in json.dumps(result)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with kit._db(path) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE forbidden (id INTEGER)")
    with pytest.raises(sqlite3.OperationalError):
        kit.lifecycle_rows(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()


@pytest.mark.parametrize("left,right,matched,reordered", [
    (["a", "b", "c"], ["a", "b", "c"], 3, 0),
    (["a", "b", "c"], ["b", "a", "c"], 3, 1),
    (["a", "a", "b", "c"], ["c", "b", "a"], 2, 1),
])
def test_reorder_unique_rows(tmp_path, left, right, matched, reordered):
    paths = [_messages(tmp_path / name, order, [("session", "user", text, None, None) for text in texts])
             for name, order, texts in (("lcm.db", "store_id", left), ("state.db", "id", right))]
    result = kit.reorder(*paths)
    assert result == {"sessions": 1, "matched": matched, "reordered_rows": reordered, "per_session": [
        {"session_h12": hashlib.sha256(b"session").hexdigest()[:12], "matched": matched,
         "reordered_rows": reordered}]}


def test_reorder_tool_identity_and_session_isolation(tmp_path):
    paths = []
    for name, order, arg, call_id in (("lcm.db", "store_id", "one", "call-a"), ("state.db", "id", "two", "call-b")):
        calls = json.dumps([{"id": "call", "function": {"name": "synthetic", "arguments": arg}}])
        paths.append(_messages(tmp_path / name, order, [
            ("shared", "assistant", "", calls, None), ("shared", "tool", "result", None, call_id),
            ("shared", "user", "matched", None, None), (name, "user", "isolated", None, None)]))
    result = kit.reorder(*paths)
    assert (result["sessions"], result["matched"], result["reordered_rows"]) == (1, 1, 0)
    assert all(value not in json.dumps(result) for value in ("one", "two", "call-a", "result", "isolated"))


def test_reorder_without_optional_host_columns(tmp_path):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", "text", None, None)])
    host = tmp_path / "state.db"
    with sqlite3.connect(host) as conn:
        conn.execute("CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, id INTEGER)")
        conn.execute("INSERT INTO messages VALUES ('s', 'user', 'text', 1)")
    assert kit.reorder(lcm, host)["matched"] == 1


@pytest.mark.parametrize("changed,verdict", [(False, "PATHS_ONLY"), (True, "OTHER")])
def test_config_diff(tmp_path, changed, verdict):
    a = {"files": ["/root-a/cache"], "knob": 912345}
    b = {"files": ["/root-b/cache"], "knob": 998877 if changed else 912345}
    yaml = pytest.importorskip("yaml")
    paths = [_write(tmp_path / name, yaml.safe_dump(value)) for name, value in (("a.yaml", a), ("b.yaml", b))]
    result = kit.config_diff(paths[0], "/root-a", paths[1], "/root-b")
    assert result == {"differing_keys": ["files.0", "knob"] if changed else ["files.0"],
                      "path_only_keys": ["files.0"], "other_keys": ["knob"] if changed else [], "verdict": verdict}
    assert all(value not in json.dumps(result) for value in ("912345", "998877", "/root-a", "/root-b"))


@pytest.mark.parametrize("candidate,previous,verdict,unmatched,failed", [
    (["warn"], ["warn"], "MATCHED", [], []),
    (["warn"], [], "UNMATCHED", ["shape"], []),
    (["warn", "warn"], ["warn"], "UNMATCHED", ["shape"], []),
    (["fail"], [], "UNMATCHED", [], ["shape"]),
    ([], ["fail"], "MATCHED", [], ["shape"]),
])
def test_doctor_match(tmp_path, candidate, previous, verdict, unmatched, failed):
    paths = []
    for name, statuses in (("candidate.json", candidate), ("previous.json", previous)):
        value = {"checks": [{"check": "shape", "status": status, "detail": "PRIVATE_DETAIL"} for status in statuses],
                 "overall": "ignored"}
        paths.append(_write(tmp_path / name, json.dumps([{}, value] if name == "previous.json" else value)))
    result = kit.doctor_match(*paths)
    assert (result["verdict"], result["unmatched"], result["failed_checks"]) == (verdict, unmatched, failed)
    assert result["warnings"] == ([{"check": "shape", "candidate": candidate.count("warn"),
                                   "previous": previous.count("warn"), "matched": not unmatched}] if "warn" in candidate else [])
    assert "PRIVATE_DETAIL" not in json.dumps(result)


def test_cli(tmp_path):
    log = _write(tmp_path / "host.log", _run(1))
    out = tmp_path / "evidence.json"
    proc = subprocess.run([sys.executable, "-m", "bench.instruments.release_gauntlet.kit_evidence", "trace-events",
                           "--log", str(log), "--out", str(out)], cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True, check=True)
    assert proc.stdout == out.read_text(encoding="utf-8")
    assert json.loads(proc.stdout) == kit.trace_events([log])
    assert proc.stderr == ""
    invalid = _write(tmp_path / "PRIVATE_INPUT_MARKER.json", "[]")
    proc = subprocess.run([sys.executable, "-m", "bench.instruments.release_gauntlet.kit_evidence", "doctor-match",
                           "--candidate", str(invalid), "--previous", str(invalid), "--out", str(out)],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert proc.stderr == "Evidence input/output error\n"
