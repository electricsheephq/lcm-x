"""Synthetic fixtures for public-clean release-kit evidence records."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from bench.instruments.release_gauntlet import kit_evidence as kit


def _h(name):
    return "h:" + kit._sha(name)[:12]


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def _log(logger, text, level="WARNING"):
    return f"2026-01-01 01:02:03,123 [{level}] {logger}: {text}\n"


def _trace(final):
    return 'Traceback (most recent call last):\n  File "synthetic.py", line 1\n    synthetic()\n' + final + "\n"


def _run(count):
    return (_log("agent.background_review", "thread=bg-review OpenAI client created", "INFO")
            + _log("agent.background_review", "background-review compressor detachment failed") * count
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
    assert result["rule_determined"] is True
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
        {"logger": "agent.optional", "missing_module": "f:" + kit._fingerprint("ImportError: " + marker), "count": 1,
         "per_event_counts": [1]},
        {"logger": "agent.relay_runtime", "missing_module": "optional.module", "count": 2, "per_event_counts": [2]}]
    assert result["other_tracebacks"] == {"ValueError": 1}
    assert result["unattributed"] == 0
    assert result["events_by_logger"] == {"agent.optional": 1, "agent.relay_runtime": 1}
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
        conn.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
        conn.execute("UPDATE messages SET conversation_id = 'has-message'")
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
        "messages_in_referenced_sessions": 0, "nodes_in_referenced_sessions": 0, "affected_stored_rows": 0}
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
    (["a", "a", "b", "c"], ["c", "b", "a"], 3, 2),
])
def test_reorder_unique_rows(tmp_path, left, right, matched, reordered):
    paths = [_messages(tmp_path / name, order, [("session", "user", text, None, None) for text in texts])
             for name, order, texts in (("lcm.db", "store_id", left), ("state.db", "id", right))]
    result = kit.reorder(*paths)
    assert (result["sessions"], result["matched"], result["reordered_rows"]) == (1, matched, reordered)
    row = result["per_session"][0]
    assert (row["lcm_rows"], row["host_rows"], row["matched"], row["reordered_rows"]) == (
        len(left), len(right), matched, reordered)
    assert row["session_h12"] == hashlib.sha256(b"session").hexdigest()[:12]


def test_reorder_tool_identity_and_session_isolation(tmp_path):
    paths = []
    for name, order, arg, call_id in (("lcm.db", "store_id", "one", "call-a"), ("state.db", "id", "two", "call-b")):
        calls = json.dumps([{"id": "call", "function": {"name": "synthetic", "arguments": arg}}])
        paths.append(_messages(tmp_path / name, order, [
            ("shared", "assistant", "", calls, None), ("shared", "tool", "result", None, call_id),
            ("shared", "user", "matched", None, None), (name, "user", "isolated", None, None)]))
    result = kit.reorder(*paths)
    assert (result["sessions"], result["matched"], result["reordered_rows"]) == (3, 1, 0)
    assert all(value not in json.dumps(result) for value in ("one", "two", "call-a", "result", "isolated"))


def test_reorder_without_optional_host_columns(tmp_path):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", "text", None, None)])
    host = tmp_path / "state.db"
    with sqlite3.connect(host) as conn:
        conn.execute("CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, id INTEGER)")
        conn.execute("INSERT INTO messages VALUES ('s', 'user', 'text', 1)")
    result = kit.reorder(lcm, host)
    assert result["matched"] == 1
    assert result["identity_degraded"] is True
    assert result["missing_columns"] == ["tool_call_id", "tool_calls"]
    assert result["per_session"][0]["columns_compared"]["host"] == ["session_id", "role", "content"]


@pytest.mark.parametrize("changed,verdict", [(False, "PATHS_ONLY"), (True, "OTHER")])
def test_config_diff(tmp_path, changed, verdict):
    a = {"files": ["/root-a/cache"], "knob": 912345}
    b = {"files": ["/root-b/cache"], "knob": 998877 if changed else 912345}
    yaml = pytest.importorskip("yaml")
    paths = [_write(tmp_path / name, yaml.safe_dump(value)) for name, value in (("a.yaml", a), ("b.yaml", b))]
    result = kit.config_diff(paths[0], "/root-a", paths[1], "/root-b")
    files, knob = ".h:" + kit._sha("files")[:8] + "[0]", ".h:" + kit._sha("knob")[:8]
    assert result == {"differing_keys": sorted([files, knob]) if changed else [files],
                      "path_only_keys": [files], "other_keys": [knob] if changed else [], "verdict": verdict}
    assert all(value not in json.dumps(result) for value in ("912345", "998877", "/root-a", "/root-b"))


@pytest.mark.parametrize("candidate,previous,verdict,unmatched,failed", [
    (["warn"], ["warn"], "MATCHED", [], []),
    (["warn"], [], "UNMATCHED", ["shape"], []),
    (["warn", "warn"], ["warn"], "UNMATCHED", ["shape"], []),
    (["fail"], [], "UNMATCHED", [], ["shape"]),
    (["pass"], ["fail"], "MATCHED", [], ["shape"]),
])
def test_doctor_match(tmp_path, candidate, previous, verdict, unmatched, failed):
    paths = []
    for name, statuses in (("candidate.json", candidate), ("previous.json", previous)):
        value = {"checks": [{"check": "shape", "status": status, "detail": "PRIVATE_DETAIL"} for status in statuses],
                 "overall": "ignored"}
        paths.append(_write(tmp_path / name, json.dumps([{"checks": []}, value] if name == "previous.json" else value)))
    result = kit.doctor_match(*paths)
    assert (result["verdict"], result["unmatched"], result["failed_checks"]) == (verdict, unmatched, failed)
    assert result["warnings"] == ([{"check": "shape", "candidate": candidate.count("warn"),
                                   "previous": previous.count("warn"), "matched": not unmatched}] if "warn" in candidate else [])
    assert "PRIVATE_DETAIL" not in json.dumps(result)


@pytest.mark.parametrize("left,right,lcm_only,host_only,reordered", [
    (["a", "b"], ["a", "b", "c"], 0, 1, 0),
    (["a", "b", "b"], ["a", "b"], 1, 0, 0),
    (["continue", "A1", "continue", "A2"], ["continue", "continue", "A1", "A2"], 0, 0, 1),
])
def test_reorder_occurrences(tmp_path, left, right, lcm_only, host_only, reordered):
    paths = [_messages(tmp_path / name, order, [("s", "user", text, None, None) for text in texts])
             for name, order, texts in (("lcm.db", "store_id", left), ("host.db", "id", right))]
    result = kit.reorder(*paths)
    for row in (result, result["per_session"][0]):
        assert (row["lcm_rows"], row["host_rows"]) == (len(left), len(right))
        assert (row["lcm_only"], row["host_only"], row["reordered_rows"]) == (lcm_only, host_only, reordered)
        assert row["matched"] == min(len(left), len(right))


def test_reorder_wrong_session(tmp_path):
    paths = [_messages(tmp_path / name, order, [(sid, "user", "x", None, None)])
             for name, order, sid in (("lcm.db", "store_id", "s"), ("host.db", "id", "t"))]
    result = kit.reorder(*paths)
    assert (result["matched"], result["lcm_only"], result["host_only"]) == (0, 1, 1)
    assert result["lcm_only_sessions"] == result["host_only_sessions"] == 1
    assert result["lcm_only_sessions_h12"] == [kit._sha("s")[:12]]
    assert result["host_only_sessions_h12"] == [kit._sha("t")[:12]]


def test_lifecycle_affected_conversation(tmp_path):
    path = _messages(tmp_path / "lcm.db", "store_id", [("old", "user", "private", None, None)])
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
        conn.execute("UPDATE messages SET conversation_id = 'conv'")
        conn.execute("CREATE TABLE summary_nodes (session_id TEXT)")
        conn.execute("CREATE TABLE lcm_lifecycle_state "
                     "(conversation_id TEXT, current_session_id TEXT, last_finalized_session_id TEXT)")
        conn.execute("INSERT INTO lcm_lifecycle_state VALUES ('conv', 'new', NULL)")
    result = kit.lifecycle_rows(path)
    assert result["identity_order_check"] == "NOT_EVALUATED"
    assert result["affected_stored_rows"] == result["empty_rows"][0]["affected_stored_rows"] == 1
    assert result["empty_rows"][0]["messages_in_referenced_sessions"] == 0


@pytest.mark.parametrize("candidate,previous,verdict", [
    ({"checks": [{"check": "shape", "status": "WARN", "detail": {"a": 1}}]},
     {"checks": [{"check": "shape", "status": "warn", "detail": {"b": 1}}]}, "UNMATCHED"),
    ({"checks": [{"check": "shape", "status": "WARN", "detail": {"a": 2}}]},
     {"checks": [{"check": "shape", "status": "warn", "detail": {"a": 1}}]}, "MATCHED"),
    ({"checks": [{"check": "shape", "status": "warn", "detail": []}]},
     {"checks": [{"check": "shape", "status": "warn", "detail": "private"}]}, "UNMATCHED"),
    ({"checks": [{"check": "lifecycle_fragmentation", "status": "warn",
                  "detail": {"empty_lifecycle_rows": 1, "state_db_error": False}}]},
     {"checks": [{"check": "lifecycle_fragmentation", "status": "warn",
                  "detail": {"empty_lifecycle_rows": 0, "state_db_error": True}}]}, "UNMATCHED"),
    ({"checks": [{"check": "shape", "status": "unexpected"}]}, {"checks": []}, "UNMATCHED"),
    ([{"checks": [{"check": "shape", "status": "FAIL"}]}, {"checks": []}], {"checks": []}, "UNMATCHED"),
    ([{"checks": [{"check": "shape", "status": "warn"}]}] * 2,
     {"checks": [{"check": "shape", "status": "warn"}]}, "UNMATCHED"),
])
def test_doctor_signatures(tmp_path, candidate, previous, verdict):
    paths = [_write(tmp_path / name, json.dumps(value))
             for name, value in (("candidate.json", candidate), ("previous.json", previous))]
    result = kit.doctor_match(*paths)
    assert result["verdict"] == verdict
    assert "candidate_signatures" in result
    if isinstance(candidate, dict) and any(row["status"] == "unexpected" for row in candidate["checks"]):
        assert result["unknown_status"] == ["shape"]
    assert "private" not in json.dumps(result)


def test_doctor_signature_causes_hold_no_values(tmp_path):
    paths = [_write(tmp_path / name, json.dumps({"checks": [
        {"check": "lifecycle_fragmentation", "status": "WARN",
         "detail": {"empty_lifecycle_rows": count, "state_db_error": "PRIVATE_STATE"}}]}))
             for name, count in (("candidate.json", 2), ("previous.json", 1))]
    result = kit.doctor_match(*paths)
    assert result["verdict"] == "MATCHED"
    assert result["candidate_signatures"] == [{
        "check": "lifecycle_fragmentation", "status": "warn", "detail_type": "dict",
        "detail_keys": [_h("empty_lifecycle_rows"), _h("state_db_error")],
        "causes": {_h("empty_lifecycle_rows"): True, _h("state_db_error"): "f:" + kit._fingerprint("PRIVATE_STATE")}}]
    assert "PRIVATE_STATE" not in json.dumps(result)


@pytest.mark.parametrize("text,holds,overlap,ambiguous,warnings,refusals", [
    (_log("agent", "thread=bg-review:A OpenAI client created", "INFO") + _trace(kit.REFUSAL)
     + _log("agent", "thread=bg-review:B OpenAI client created", "INFO") + _trace(kit.REFUSAL)
     + _log("agent", "Background review complete: thread=bg-review", "INFO") * 2,
     None, 1, 1, 0, 2),
    (_run(1).replace(_trace(kit.REFUSAL),
                    _log("agent", "background-review compressor detachment failed") + _trace(kit.REFUSAL)),
     False, 0, 0, 2, 1),
    (_run(1).replace(kit.REFUSAL, "  | " + kit.REFUSAL), True, 0, 0, 1, 1),
    (_log("agent", "background-review compressor detachment failed"), False, 0, 0, 1, 0),
    (_log("agent", "thread=bg-review:A OpenAI client created", "INFO") + _trace(kit.REFUSAL)
     + _log("agent", "Background review complete: thread=bg-review:A", "INFO")
     + _trace(kit.REFUSAL), False, 0, 0, 0, 2),
], ids=["overlap", "warning-mismatch", "exception-group", "outside-warning", "explicit-completion"])
def test_trace_interleaving_and_warnings(tmp_path, text, holds, overlap, ambiguous, warnings, refusals):
    result = kit.trace_events([_write(tmp_path / "host.log", text)])
    assert result["per_event_rule_holds"] is holds
    assert result["rule_determined"] is (holds is not None)
    assert result["overlapping_runs"] == overlap
    assert result["ambiguous_completions"] == ambiguous
    assert result["detach_warnings_total"] == warnings
    assert result["refusals_total"] == refusals


def test_import_event_counts_and_trace_boundaries(tmp_path):
    text = (_log("agent.one", "private").replace("01-01 ", "01-01T").replace(",123", ".123")
            + _trace("ModuleNotFoundError: No module named 'one'") * 2
            + _log("agent.one", "private", "ERROR") + _trace("ImportError: private")
            + _log("agent.debug", "private", "DEBUG") + _trace("ImportError: private")
            + _log("agent.critical", "private", "CRITICAL") + _trace("KeyboardInterrupt")
            + "ModuleNotFoundError: No module named 'not_a_trace'\n")
    result = kit.trace_events([_write(tmp_path / "host.log", text)])
    assert result["events_by_logger"] == {"agent.one": 2}
    assert result["optional_import_traces_total"] == 4
    assert {row["logger"] for row in result["by_event"]} == {"agent.one", "agent.debug"}


@pytest.mark.parametrize("a,b,root_a,root_b,verdict", [
    ({"a.b": 1}, {"a": {"b": 1}}, "", "", "OTHER"),
    ({"a": [1]}, {"a": {"0": 1}}, "", "", "OTHER"),
    ({"model": "/root-a-extra"}, {"model": "/root-b-extra"}, "/root-a", "/root-b", "OTHER"),
    ({"model": "/root-a"}, {"model": "/root-b"}, "/root-a", "/root-b", "PATHS_ONLY"),
    ({"model": "/root-a/cache"}, {"model": "<ROOT>/cache"}, "/root-a", "/root-b", "OTHER"),
    ({"model": {"private_name": 1}}, {"model": {"private_name": 2}}, "", "", "OTHER"),
])
def test_config_shape_and_privacy(tmp_path, a, b, root_a, root_b, verdict):
    yaml = pytest.importorskip("yaml")
    paths = [_write(tmp_path / name, yaml.safe_dump(value)) for name, value in (("a.yaml", a), ("b.yaml", b))]
    result = kit.config_diff(paths[0], root_a, paths[1], root_b)
    assert result["verdict"] == verdict
    assert "private_name" not in json.dumps(result)
    if isinstance(a.get("model"), dict):
        assert result["other_keys"] == [".model.h:" + kit._sha("private_name")[:8]]


def test_config_recursive_alias(tmp_path):
    pytest.importorskip("yaml")
    path = _write(tmp_path / "recursive.yaml", "model: &loop\n  nested: *loop\n")
    with pytest.raises(ValueError, match="^unreadable config$"):
        kit.config_diff(path, "", path, "")


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


def test_unclosed_run_with_events_is_undetermined(tmp_path):
    text = (_run(1) + _log("agent", "thread=bg-review OpenAI client created", "INFO")
            + _log("host.fork", "background-review compressor detachment failed") + _trace(kit.REFUSAL))
    result = kit.trace_events([_write(tmp_path / "host.log", text)])
    assert result["unclosed_runs_with_events"] == 1
    assert result["rule_determined"] is False
    assert result["per_event_rule_holds"] is None


def test_lifecycle_without_conversation_column_is_not_evaluated(tmp_path):
    path = _messages(tmp_path / "lcm.db", "store_id", [("stored", "user", "synthetic content", None, None)])
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE summary_nodes (session_id TEXT)")
        conn.execute("CREATE TABLE lcm_lifecycle_state (conversation_id TEXT, current_session_id TEXT, "
                     "last_finalized_session_id TEXT)")
        conn.execute("INSERT INTO lcm_lifecycle_state VALUES ('empty', NULL, NULL)")
    result = kit.lifecycle_rows(path)
    assert result["conversation_column"] is False
    assert result["identity_order_check"] == "NOT_EVALUATED"


def test_doctor_detail_keys_from_data_are_hashed(tmp_path):
    key = "/private/store/PRIVATE_KEY_NAME.db"
    paths = [_write(tmp_path / name, json.dumps({"checks": [
        {"check": "storage", "status": "warn", "detail": {key: 1, "rows": 2}}]}))
             for name in ("candidate.json", "previous.json")]
    result = kit.doctor_match(*paths)
    assert result["candidate_signatures"][0]["detail_keys"] == [
        "h:" + hashlib.sha256(key.encode()).hexdigest()[:12], _h("rows")]
    assert "PRIVATE_KEY_NAME" not in json.dumps(result)


@pytest.mark.parametrize("final,name", [("Exception: failed", "Exception"), ("SystemExit: 2", "SystemExit"),
                                        ("KeyboardInterrupt", "KeyboardInterrupt"),
                                        ("pkg.mod.CustomError: x", _h("pkg.mod.CustomError"))])
def test_bare_and_dotted_traceback_terminators_are_counted(tmp_path, final, name):
    result = kit.trace_events([_write(tmp_path / "host.log", _log("agent", "start", "INFO") + _trace(final))])
    assert result["other_tracebacks"] == {name: 1}


@pytest.mark.parametrize("line", [
    "2026-01-01 01:02:03,123 WARNING agent.optional: optional import failed\n",
    "2026-01-01 01:02:03,123 WARNING [sess-1] agent.optional: optional import failed\n",
    "01:02:03 - agent.optional - WARNING - optional import failed\n",
    "01:02:03 - agent.optional - WARNING [sess-1] - optional import failed\n",
], ids=["file", "file-tagged", "verbose", "verbose-tagged"])
def test_host_log_formats_attribute_import_events(tmp_path, line):
    text = line + _trace("ModuleNotFoundError: No module named 'opt'")
    result = kit.trace_events([_write(tmp_path / "host.log", text)])
    assert result["events_by_logger"] == {"agent.optional": 1}
    assert result["unattributed"] == 0


def test_import_counts_kept_per_triggering_event(tmp_path):
    def events(*counts):
        return "".join(_log("agent.optional", "warn") + _trace("ModuleNotFoundError: No module named 'opt'") * n
                       for n in counts)
    a = kit.trace_events([_write(tmp_path / "a.log", events(2, 2))])["by_event"]
    b = kit.trace_events([_write(tmp_path / "b.log", events(1, 3))])["by_event"]
    assert a[0]["count"] == b[0]["count"] == 4
    assert (a[0]["per_event_counts"], b[0]["per_event_counts"]) == ([2, 2], [1, 3])


@pytest.mark.parametrize("lcm,host,reordered,lcm_only,host_only", [
    ("abab", "baba", 1, 0, 0), ("aba", "ba", 0, 1, 0), ("abc", "cab", 1, 0, 0), ("aab", "aab", 0, 0, 0)])
def test_reorder_is_true_lcs_shortfall_with_repeats(tmp_path, lcm, host, reordered, lcm_only, host_only):
    paths = [_messages(tmp_path / name, order, [("s", "user", ch, None, None) for ch in rows])
             for name, order, rows in (("lcm.db", "store_id", lcm), ("state.db", "id", host))]
    result = kit.reorder(*paths)
    assert (result["reordered_rows"], result["lcm_only"], result["host_only"]) == (reordered, lcm_only, host_only)


def test_lifecycle_counts_legacy_padded_and_blank_conversation_ids(tmp_path):
    path = _messages(tmp_path / "lcm.db", "store_id", [("old", "user", "synthetic", None, None)])
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
        conn.execute("UPDATE messages SET conversation_id = ' conv '")
        conn.execute("CREATE TABLE summary_nodes (session_id TEXT)")
        conn.execute("CREATE TABLE lcm_lifecycle_state (conversation_id TEXT, current_session_id TEXT, "
                     "last_finalized_session_id TEXT)")
        conn.execute("INSERT INTO lcm_lifecycle_state VALUES ('conv', NULL, NULL)")
    result = kit.lifecycle_rows(path)
    assert (result["affected_stored_rows"], result["identity_order_check"]) == (1, "NOT_EVALUATED")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE messages SET conversation_id = NULL")
    result = kit.lifecycle_rows(path)
    assert (result["affected_stored_rows"], result["blank_conversation_rows"]) == (1, 1)
    assert result["identity_order_check"] == "NOT_EVALUATED"


def test_doctor_list_detail_new_cause_is_unmatched(tmp_path):
    old = "ignored config key 'lcm.foo' in /root-a/config.yaml"
    def doctor(name, entries):
        return _write(tmp_path / name, json.dumps({"checks": [
            {"check": "config_validation", "status": "warn", "detail": entries}]}))
    same = kit.doctor_match(doctor("c1.json", [old.replace("/root-a", "/root-b")]), doctor("p.json", [old]))
    assert same["verdict"] == "MATCHED"
    new = kit.doctor_match(doctor("c2.json", [old, "incremental_max_depth=0 disables condensation"]),
                           doctor("p2.json", [old]))
    assert new["verdict"] == "UNMATCHED"
    assert len(new["candidate_signatures"][0]["detail_fingerprints"]) == 2
    assert "condensation" not in json.dumps(new) and "root-a" not in json.dumps(new)


def test_doctor_fingerprint_keeps_status_codes_and_url_hosts(tmp_path):
    def doctor(name, text):
        return _write(tmp_path / name, json.dumps({"checks": [{"check": "inactive_process", "status": "warn",
                                                                "detail": text}]}))
    for a, b in (("HTTP 401 from provider", "HTTP 500 from provider"),
                 ("calls https://api.a.example/v1", "calls https://other.example/x")):
        assert kit.doctor_match(doctor("c.json", b), doctor("p.json", a))["verdict"] == "UNMATCHED"
    assert kit.doctor_match(doctor("c.json", "pid 48213 at /x/a"), doctor("p.json", "pid 51002 at /y/b")
                            )["verdict"] == "MATCHED"


def test_indented_bare_name_does_not_end_a_traceback(tmp_path):
    trace = ("Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n    raise (\n"
             "    ValueError\n    )\nModuleNotFoundError: No module named 'opt'\n")
    result = kit.trace_events([_write(tmp_path / "host.log", _log("agent.optional", "warn") + trace)])
    assert result["optional_import_traces_total"] == 1
    assert result["other_tracebacks"] == {}


def test_doctor_dict_detail_new_cause_is_unmatched(tmp_path):
    def doctor(name, missing):
        return _write(tmp_path / name, json.dumps({"checks": [{"check": "payload_storage", "status": "warn",
            "detail": {"suspicious_rows": 3, "externalized_payload_refs_missing": missing, "check": "x"}}]}))
    assert kit.doctor_match(doctor("c.json", 0), doctor("p.json", 0))["verdict"] == "MATCHED"
    result = kit.doctor_match(doctor("c1.json", 1), doctor("p1.json", 0))
    assert result["verdict"] == "UNMATCHED"
    assert result["candidate_signatures"][0]["check"] == "payload_storage"  # A detail key cannot overwrite it.


@pytest.mark.parametrize("candidate", [{"checks": []}, [{"checks": []}, {"checks": []}]])
def test_empty_candidate_checks_are_refused(tmp_path, candidate):
    previous = _write(tmp_path / "p.json", json.dumps({"checks": [{"check": "x", "status": "fail"}]}))
    with pytest.raises(ValueError):
        kit.doctor_match(_write(tmp_path / "c.json", json.dumps(candidate)), previous)


@pytest.mark.parametrize("candidate,previous,verdict", [
    ({"state_db_error": "no such table: sessions"}, {"state_db_error": "database is locked"}, "UNMATCHED"),
    ({"status": "degraded"}, {"status": "ok"}, "UNMATCHED"),
    ({"sub": {"new_err": 1}}, {"sub": {"old": 1}}, "UNMATCHED"),
    ({"count": -1}, {"count": 0}, "UNMATCHED"),
    ({"state_db_error": "locked at /a/x.db pid 48213"}, {"state_db_error": "locked at /b/y.db pid 51002"}, "MATCHED"),
], ids=["string-cause", "status-string", "nested-dict", "negative", "same-cause"])
def test_doctor_dict_string_causes(tmp_path, candidate, previous, verdict):
    def doctor(name, detail):
        return _write(tmp_path / name, json.dumps({"checks": [
            {"check": "lifecycle_fragmentation", "status": "warn", "detail": detail}]}))
    result = kit.doctor_match(doctor("c.json", candidate), doctor("p.json", previous))
    assert result["verdict"] == verdict
    encoded = json.dumps(result)
    assert all(word not in encoded for word in ("sessions", "locked", "degraded", "/a/x.db", "48213"))


@pytest.mark.parametrize("final,name", [("ProviderCircuitOpen: open", _h("ProviderCircuitOpen")),
                                        ("pkg.ProviderRateLimited", _h("pkg.ProviderRateLimited"))])
def test_terminators_without_a_conventional_suffix_are_counted(tmp_path, final, name):
    result = kit.trace_events([_write(tmp_path / "host.log", _log("agent", "start", "INFO") + _trace(final))])
    assert result["other_tracebacks"] == {name: 1}


def test_a_traceback_without_a_final_line_is_counted(tmp_path):
    cut = 'Traceback (most recent call last):\n  File "synthetic.py", line 1\n'
    text = _log("agent", "start", "INFO") + cut + _log("agent", "next", "INFO") + cut
    result = kit.trace_events([_write(tmp_path / "host.log", text)])
    assert result["other_tracebacks"] == {"(no final line)": 2}


def test_exception_group_source_line_does_not_end_a_sub_traceback(tmp_path):
    group = ("  + Exception Group Traceback (most recent call last):\n"
             "  |   File \"x.py\", line 1, in <module>\n"
             "  | ExceptionGroup: two (2 sub-exceptions)\n"
             "  +-+---------------- 1 ----------------\n"
             "    | Traceback (most recent call last):\n"
             "    |   File \"x.py\", line 2, in f\n"
             "    |     marker\n"
             "    | ProviderRateLimited: slow\n"
             "    +------------------------------------\n")
    result = kit.trace_events([_write(tmp_path / "host.log", _log("agent", "start", "INFO") + group)])
    assert result["other_tracebacks"] == {"ExceptionGroup": 1, _h("ProviderRateLimited"): 1}


def _host_generations(path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, "
                     "tool_calls TEXT, tool_call_id TEXT, timestamp REAL, active INTEGER, compacted INTEGER)")
        conn.executemany("INSERT INTO messages VALUES (?, 's', 'user', ?, NULL, NULL, ?, ?, ?)",
                         [(i, *row) for i, row in enumerate(rows, 1)])
    return path


def test_host_with_generations_but_no_stamp_is_unresolved(tmp_path):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", "A", None, None)])
    host = tmp_path / "state.db"
    with sqlite3.connect(host) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT, active INTEGER)")
        conn.execute("INSERT INTO messages VALUES (1, 's', 'user', 'A', 1)")
    assert kit.reorder(lcm, host)["host_copies_resolved"] is False


def test_repeated_identities_are_counted(tmp_path):
    rows = [("s", "user", ch, None, None) for ch in ("continue", "x", "continue")]
    paths = [_messages(tmp_path / n, o, rows) for n, o in (("lcm.db", "store_id"), ("state.db", "id"))]
    result = kit.reorder(*paths)
    assert result["repeated_identity_rows"] == 2
    assert result["per_session"][0]["repeated_identity_rows"] == 2


@pytest.mark.parametrize("word", ["ghp_SECRETTOKEN123", "WARNING", "lowercase_word", "CustomerSecretToken",
                                  "sk_live_abc123.ProviderError"])
def test_a_non_exception_word_after_a_cut_traceback_is_hashed(tmp_path, word):
    cut = 'Traceback (most recent call last):\n  File "synthetic.py", line 1\n'
    result = kit.trace_events([_write(tmp_path / "host.log", _log("agent", "start", "INFO") + cut + word + "\n")])
    assert word not in json.dumps(result)
    assert sum(result["other_tracebacks"].values()) == 1


def test_different_import_failures_do_not_share_a_signature(tmp_path):
    def cell(name, final):
        return kit.trace_events([_write(tmp_path / name, _log("agent.optional", "warn") + _trace(final))])["by_event"]
    a = cell("a.log", "ImportError: cannot import name 'optional_helper' from 'optional_pkg'")
    b = cell("b.log", "ImportError: cannot import name 'ContextEngine' from 'hermes'")
    assert a[0]["missing_module"] != b[0]["missing_module"]
    assert "optional_helper" not in json.dumps(a) and "ContextEngine" not in json.dumps(b)


def test_data_shaped_lowercase_doctor_keys_are_hashed(tmp_path):
    paths = [_write(tmp_path / name, json.dumps({"checks": [
        {"check": "storage", "status": "warn", "detail": {"customer_secret_token": 1}}]}))
             for name in ("candidate.json", "previous.json")]
    result = kit.doctor_match(*paths)
    assert "customer_secret_token" not in json.dumps(result)
    assert result["verdict"] == "MATCHED"


def test_host_generation_copies_make_the_session_not_definitive(tmp_path):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", ch, None, None) for ch in "ABC"])
    host = _host_generations(tmp_path / "state.db", [("A", 1.0, 0, 1), ("B", 2.0, 0, 1),
                                                     ("A", 1.0, 1, 0), ("B", 2.0, 1, 0), ("C", 3.0, 1, 0)])
    result = kit.reorder(lcm, host)
    assert result["host_stamp_repeat_rows"] == 2 and result["per_session"][0]["counts_definitive"] is False
    assert (result["host_copies_resolved"], result["sessions_not_definitive"]) == (False, 1)
    assert result["host_only"] == 2  # Copies are kept, never folded away: the record cannot read clean.


@pytest.mark.parametrize("rows", [
    [("A", 1.0, 0, 1), ("B", 2.0, 0, 1), ("B", 2.0, 1, 0), ("A", 1.0, 1, 0)],          # a reordered compaction list
    [("A", 1.0, 0, 1), ("A", 1.0, 1, 0), ("A", 1.0, 1, 0), ("B", 2.0, 1, 0)],          # a doubled live copy
    [("A", 1.0, 1, 0), ("A", 1.0, 1, 0)]])                                             # a doubled live row
def test_any_stamp_repeat_is_not_definitive(tmp_path, rows):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", ch, None, None) for ch in "AB"])
    result = kit.reorder(lcm, _host_generations(tmp_path / "state.db", rows))
    assert result["host_copies_resolved"] is False and result["sessions_not_definitive"] == 1


@pytest.mark.parametrize("rows,generation_rows,definitive", [
    ([("go", 1.0, 1, 0), ("go", 1.000001, 1, 0)], 0, True),     # two sends, all active, none compacted
    ([("go", 1.0, 0, 0), ("go", 2.0, 1, 0)], 1, False),         # an inactive copy, then a fresh-stamped active one
    ([("go", 1.0, 1, 1), ("go", 2.0, 1, 0)], 1, False)])        # a compacted row whose identity repeats
def test_distinct_stamps_stay_definitive_only_without_generation_copies(tmp_path, rows, generation_rows, definitive):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", "go", None, None)])
    result = kit.reorder(lcm, _host_generations(tmp_path / "state.db", rows))
    assert (result["host_only"], result["host_stamp_repeat_rows"]) == (1, 0)
    assert (result["per_session"][0]["counts_definitive"], result["host_copies_resolved"]) == (definitive, definitive)
    assert result["host_generation_rows"] == result["per_session"][0]["host_generation_rows"] == generation_rows


def _rewrites(path, overrides):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany("INSERT INTO metadata VALUES (?, ?)", [
            (f"host_rewrite_identity:{store_id}", json.dumps({"version": 1, "content": content, "stripped": ["", "\n"],
                                                              "stored_sha256": hashlib.sha256(stored.encode()).hexdigest()}))
            for store_id, content, stored in overrides])
    return path


@pytest.mark.parametrize("overrides,credited,lcm_only", [
    ([(2, "go", "go\n")], 1, 0),            # a valid recorded rewrite: the host form's identity
    ([], 0, 1),                              # no override: edge whitespace is never stripped
    ([(2, "go", "other\n")], 0, 1),         # the recorded hash does not match the stored content
    ([(2, "stop", "go\n")], 0, 1)])         # a valid hash, but the recorded form is not an edge-whitespace rewrite
def test_recorded_host_rewrites_are_credited(tmp_path, overrides, credited, lcm_only):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", "hi", None, None), ("s", "user", "go\n", None, None),
                                                      ("s", "assistant", "ok", None, None)])
    host = _messages(tmp_path / "state.db", "id", [("s", "user", "hi", None, None), ("s", "user", "go", None, None),
                                                   ("s", "assistant", "ok", None, None)])
    result = kit.reorder(_rewrites(lcm, overrides), host)
    row = result["per_session"][0]
    assert (row["matched"], row["lcm_only"], row["host_only"]) == (3 - lcm_only, lcm_only, lcm_only)
    assert row["host_rewrites_credited"] == result["host_rewrites_credited"] == credited


def test_a_reorder_among_credited_rows_is_counted(tmp_path):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", "A\n", None, None), ("s", "user", " B", None, None)])
    host = _messages(tmp_path / "state.db", "id", [("s", "user", "B", None, None), ("s", "user", "A", None, None)])
    result = kit.reorder(_rewrites(lcm, [(1, "A", "A\n"), (2, "B", " B")]), host)
    assert (result["matched"], result["lcm_only"], result["host_only"]) == (2, 0, 0)
    assert (result["reordered_rows"], result["host_rewrites_credited"]) == (1, 2)


@pytest.mark.parametrize("root_a,root_b,a,b,verdict", [
    ("/root-a/", "/root-b/", "/root-a/cache", "/root-b/cache", "PATHS_ONLY"),
    ("/", "/root-b", "/cache", "/root-b/cache", "PATHS_ONLY"),
    ("/", "/root-b/", "/cache", "/root-b/other", "OTHER"),
    ("/a/", "/b", "/a/", "/b/", "PATHS_ONLY")])
def test_config_roots_with_trailing_separators(tmp_path, root_a, root_b, a, b, verdict):
    yaml = pytest.importorskip("yaml")
    paths = [_write(tmp_path / name, yaml.safe_dump({"model": value})) for name, value in (("a.yaml", a), ("b.yaml", b))]
    assert kit.config_diff(paths[0], root_a, paths[1], root_b)["verdict"] == verdict


def test_only_the_exact_override_key_is_read(tmp_path):
    lcm = _messages(tmp_path / "lcm.db", "store_id", [("s", "user", "go\n", None, None)])
    host = _messages(tmp_path / "state.db", "id", [("s", "user", "go", None, None)])
    with sqlite3.connect(lcm) as conn:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany("INSERT INTO metadata VALUES (?, ?)", [(key, json.dumps(
            {"content": "go", "stored_sha256": hashlib.sha256(b"go\n").hexdigest()})) for key in (
                "HOST_REWRITE_IDENTITY:1", "hostXrewriteXidentity:1")])
    assert kit.reorder(lcm, host)["host_rewrites_credited"] == 0


@pytest.mark.parametrize("value,missing,expected", [
    (0, True, False), (1, False, True), ("0", True, False), ("false", True, False), (" 1 ", False, True),
    (None, True, True), (None, False, False)])
def test_generation_flags_read_any_storage_class(value, missing, expected):
    assert kit._flag(value, missing) is expected
