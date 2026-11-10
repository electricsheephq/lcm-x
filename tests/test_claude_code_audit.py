"""Claude Code capture audit gate: synthetic writer stores and deliberate ledger damage.

Design: docs/claude-code-adapter.md ("The audit (the gate)").
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import weakref
from contextlib import closing
from pathlib import Path

import pytest

from hermes_lcm import claude_code_audit as ca
from hermes_lcm import claude_code_store as cc
from hermes_lcm.config import LCMConfig
from hermes_lcm.externalize import extract_externalized_refs
from tests.test_claude_code_store import CONVERSATION, _entry, _fork, _open, _transcript


def _close(store):
    cc.to_rollback_journal(store.connection)
    store.close()


@pytest.mark.parametrize("blocking", [False, True])
def test_hook_blocking_latency(tmp_path, blocking):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [{"event": "Stop", "session": "s1", "ms": value,
             **({"blocking_ms": value / 10} if blocking else {})} for value in (10, 20)]
    (logs / "hook.log").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = ca._home_report(tmp_path)["hooks_blocking_ms"]
    assert result == ({"first": {"n": 1, "max": 1.0},
                       "Stop": {"n": 1, "p50": 2.0, "p95": 2.0, "max": 2.0}} if blocking else {})


def _tamper(home, sql, args=()):
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute(sql, args)


def _cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "lcm_x_capture.py"
    spec = importlib.util.spec_from_file_location("capture_audit_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def captured(tmp_path, monkeypatch):
    home = tmp_path / "home"
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    saved = folder / "saved.txt"
    saved.write_text("complete saved output", encoding="utf-8")
    (folder / "binary.bin").write_bytes(b"\x00saved binary chunks\xff")
    marker = f"<persisted-output>\nFull output saved to: {saved}\nPreview\n</persisted-output>"
    result = _entry("user", "r1", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}])
    entries = [*CONVERSATION[:2], result, *CONVERSATION[3:], CONVERSATION[0],
               _entry("assistant", "thinking", [{"type": "thinking", "thinking": "private"}]),
               {"type": "future-kind"}]
    path = _transcript(tmp_path, entries)
    lineage = cc.Lineage("s1", None, path)
    monkeypatch.setattr(cc, "FILE_CHUNK_BYTES", 4)
    store = _open(home, tmp_path / "hermes")
    try:
        assert cc.drain(store, lineage) == "drained"
        assert cc.capture_files(store, lineage) == 1
    finally:
        _close(store)
    return home, lineage


def _failed(home, counter, expected=1):
    report = ca.audit(home)
    assert report["failures"][counter] == expected
    assert report["lineages"][0]["failures"][counter] == expected
    assert report["ok"] is False
    return report


def test_clean_capture(captured, capsys):
    home, lineage = captured
    report = ca.audit(home)
    assert report["ok"] is True
    assert report["verified"] is True and report["unverified"] == []
    assert report["failures"] == dict.fromkeys(ca.FAILURES, 0)
    row = report["lineages"][0]
    assert row["lineage"] == lineage.key and row["lines"] == 10
    assert row["classes"] == {"projected": 6, "raw_only": 1, "shared_raw": 2, "unknown_type": 1}
    assert row["unknown_types"] == {"future-kind": 1}
    assert row["fork_copies"] == row["surplus"] == row["unverifiable"] == row["unanswered"] == 0
    assert row["files"] == {"held": 1, "kept": 1, "uncaptured": 0}
    assert report["lag_s"]["p50"] > 0
    assert report["growth"]["transcript_bytes"] == lineage.path.stat().st_size
    assert str(home) not in json.dumps(report) and "complete saved output" not in json.dumps(report)
    assert _cli().main(["audit", "--home", str(home)]) == 0
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("counter", ["lost_lines", "lost"])
def test_missing_line_and_identity(captured, counter):
    home, _lineage = captured
    _tamper(home, "DELETE FROM capture_events WHERE source_kind = 'line' AND position = 1")
    _failed(home, counter)


def test_digest_mismatch(captured):
    home, _lineage = captured
    _tamper(home, "UPDATE capture_events SET line_digest = 'wrong' WHERE source_kind = 'line' AND position = 1")
    _failed(home, "digest_mismatch")


@pytest.mark.parametrize("sql", [
    "UPDATE capture_events SET raw = x'00' WHERE source_kind = 'line' AND position = 1",
    "UPDATE capture_events SET raw = x'00' WHERE source_kind = 'file' AND disposition = 'kept' AND position = 0",
    "DELETE FROM capture_events WHERE source_kind = 'file' AND disposition = 'kept' AND position = 1",
    "UPDATE capture_events SET content_digest = 'wrong' WHERE source_kind = 'file' AND disposition = 'kept'",
    "UPDATE capture_events SET raw_from = -1 WHERE raw_from IS NOT NULL",
    "UPDATE capture_events SET session_id = 'other' WHERE entry_class = 'shared_raw' AND raw IS NOT NULL",
])
def test_corrupt_bytes_or_shared_reference(captured, sql):
    home, _lineage = captured
    _tamper(home, sql)
    _failed(home, "corrupt")


@pytest.mark.parametrize("target", ["different_digest", "other_session"])
def test_raw_reference_must_match_digest_and_session(captured, tmp_path, target):
    home, _lineage = captured
    if target == "other_session":
        path = _transcript(tmp_path, [CONVERSATION[4]], session="s2")
        store = _open(home, tmp_path / "hermes")
        try:
            cc.drain(store, cc.Lineage("s2", None, path))
        finally:
            _close(store)
        source = "lineage_key = 'claude-code:s2' AND raw IS NOT NULL"
    else:
        source = "lineage_key = 'claude-code:s1' AND source_kind = 'line' AND position = 0"
    _tamper(home, "UPDATE capture_events SET raw_from = (SELECT event_id FROM capture_events WHERE "
            + source + ") WHERE raw_from IS NOT NULL")
    _failed(home, "corrupt")


def test_snapshot_keeps_no_raw_and_decompresses_shared_bytes_once(captured, monkeypatch):
    home, lineage = captured
    with closing(sqlite3.connect(f"{(home / 'lcm.db').as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        shared = conn.execute("SELECT raw FROM capture_events WHERE entry_class = 'shared_raw' "
                              "AND raw IS NOT NULL").fetchone()[0]
        decompress = ca.zlib.decompress
        shared_reads = []

        def counted(data):
            if data == shared:
                shared_reads.append(conn.in_transaction)
            return decompress(data)

        monkeypatch.setattr(ca.zlib, "decompress", counted)
        snapshot = ca._snapshot(conn, cc.CURSOR_PREFIX + lineage.key)
        assert shared_reads == [True]
        assert snapshot[5]["corrupt"] == 0 and not conn.in_transaction
        assert all(set(event) <= set(ca.EVENT_FIELDS) for event in snapshot[2])
        assert all(not isinstance(value, bytes) for event in snapshot[2] for value in event.values())


def test_audit_never_reads_whole_files(captured, monkeypatch):
    home, lineage = captured
    read_text = Path.read_text

    def refused(*_args, **_kwargs):
        pytest.fail("audit used a whole-file read")

    def config_only(path, *args, **kwargs):
        assert path != lineage.path and not path.is_relative_to(cc.session_dir(lineage.path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", refused)
    monkeypatch.setattr(Path, "read_text", config_only)
    assert ca.audit(home)["ok"] is True


@pytest.mark.parametrize("held", [False, True])
def test_missing_message_rows(captured, held):
    home, _lineage = captured
    position = 2 if held else 0
    _tamper(home, "DELETE FROM messages WHERE store_id = (SELECT json_extract(store_ids, '$[0]') "
            "FROM capture_events WHERE source_kind = 'line' AND position = ?)", (position,))
    _failed(home, "missing_rows", 2 if held else 1)


def test_duplicate_identity(captured):
    home, _lineage = captured
    _tamper(home, "UPDATE capture_events SET disposition = 'new' WHERE disposition = 'duplicate'")
    _failed(home, "duplicates")


@pytest.mark.parametrize(("counter", "call"), [("orphans", "unknown"), ("orphans", ""), ("doubles", "toolu_1")])
def test_tool_results(captured, counter, call):
    home, _lineage = captured
    _tamper(home, "INSERT INTO messages(session_id, role, content, tool_call_id, timestamp) "
            "VALUES('s1', 'tool', 'extra result', ?, 0)", (call,))
    _failed(home, counter)


def test_call_in_two_assistant_rows(captured):
    home, _lineage = captured
    _tamper(home, "INSERT INTO messages(session_id, role, tool_calls, timestamp) "
            "SELECT session_id, role, tool_calls, timestamp FROM messages WHERE tool_calls IS NOT NULL")
    _failed(home, "doubles")


def test_surplus_line(captured):
    home, _lineage = captured
    columns = ("session_id, lineage_key, source_kind, generation, position, uid, entry_type, entry_class, "
               "line_digest, content_digest, disposition, store_ids, notes, raw, raw_from, projection_version, captured_at")
    source = columns.replace("position,", "10,", 1)
    _tamper(home, f"INSERT INTO capture_events({columns}) SELECT {source} FROM capture_events "
            "WHERE source_kind = 'line' AND position = 0")
    _failed(home, "surplus_lines")


def test_round5_uncaptured_old_file_does_not_fail(captured):
    home, lineage = captured
    path = cc.session_dir(lineage.path) / "tool-results" / "binary.bin"
    os.utime(path, (0, 0))
    _tamper(home, "DELETE FROM capture_events WHERE source_kind = 'file' AND disposition = 'kept'")
    report = ca.audit(home)
    assert report["ok"] is True and report["verified"] is False
    assert report["unverified"] == [lineage.key]
    assert report["lineages"][0]["files"]["uncaptured"] == 1
    assert "pending" not in report["lineages"][0]["files"]
    assert "files_lost" not in report["failures"]


def test_pending_lines_and_files_are_not_lost(captured):
    home, lineage = captured
    with lineage.path.open("ab") as handle:
        handle.write(json.dumps(_entry("user", "later", "pending")).encode() + b"\npartial")
    path = cc.session_dir(lineage.path) / "tool-results" / "pending.txt"
    path.write_text("not captured yet", encoding="utf-8")
    os.utime(path, (2_000_000_000, 2_000_000_000))
    report = ca.audit(home)
    assert report["ok"] is True
    assert report["lineages"][0]["pending_lines"] == 1
    assert report["lineages"][0]["files"]["uncaptured"] == 1
    assert report["verified"] is False


def test_revision_restatement_is_not_a_double(tmp_path):
    home = tmp_path / "home"
    revised = {**CONVERSATION[1], "message": {**CONVERSATION[1]["message"], "id": "revision"}}
    path = _transcript(tmp_path, [*CONVERSATION[:3], revised])
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, cc.Lineage("s1", None, path))
    finally:
        _close(store)
    report = ca.audit(home)
    assert report["ok"] is True
    assert report["failures"]["doubles"] == 0


def test_fork_copies_have_their_own_column(tmp_path):
    home = tmp_path / "home"
    parent = _transcript(tmp_path, CONVERSATION[:3])
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, cc.Lineage("s1", None, parent))
        fork = _fork(tmp_path, store, "f1", CONVERSATION[:3], [])
        cc.drain(store, fork)
    finally:
        _close(store)
    report = ca.audit(home, session="f1")
    assert report["ok"] is True
    assert len(report["lineages"]) == 1
    row = report["lineages"][0]
    assert row["fork_copies"] == 3 and row["surplus"] == 0
    assert row["failures"]["duplicates"] == 0
    assert report["sessions"]["forks"] == 1 and report["sessions"]["fork_copy_without_origin"] == 0


def test_two_fork_copies_of_one_identity_are_duplicates(tmp_path):
    home = tmp_path / "home"
    parent = _transcript(tmp_path, CONVERSATION[:3])
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, cc.Lineage("s1", None, parent))
        fork = _fork(tmp_path, store, "f1", [*CONVERSATION[:3], CONVERSATION[0]], [])
        cc.drain(store, fork)
    finally:
        _close(store)
    _tamper(home, "UPDATE capture_events SET disposition = 'fork_copy' "
            "WHERE lineage_key = 'claude-code:f1' AND disposition = 'duplicate'")
    report = ca.audit(home, session="f1")
    assert report["ok"] is False and report["failures"]["duplicates"] == 1
    row = report["lineages"][0]
    assert row["failures"]["duplicates"] == 1 and row["fork_copies"] == 4
    assert row["surplus"] == 0


@pytest.mark.parametrize("keep_copy", [False, True])
def test_generations_with_and_without_set_aside_copy(tmp_path, keep_copy):
    home = tmp_path / "home"
    path = _transcript(tmp_path, CONVERSATION[:1])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, lineage)
        if keep_copy:
            path.with_name(path.name + ".superseded-1").write_bytes(path.read_bytes())
        path.write_text(json.dumps(_entry("user", "new", "rewritten")) + "\n", encoding="utf-8")
        cc.drain(store, lineage)
    finally:
        _close(store)
    report = ca.audit(home)
    assert report["ok"] is True
    row = report["lineages"][0]
    assert row["generation"] == 1 and row["generations_gone"] == (0 if keep_copy else 1)
    assert row["unverifiable"] == (0 if keep_copy else 1)
    assert row["lines"] == (2 if keep_copy else 1)
    assert report["unverified"] == ([] if keep_copy else [lineage.key])
    assert report["verified"] is keep_copy


def test_cli_failures_and_invalid_homes(captured, tmp_path, capsys, monkeypatch):
    home, _lineage = captured
    monkeypatch.delitem(sys.modules, "hermes_lcm.mcp_reader", raising=False)
    cli = _cli()
    _tamper(home, "UPDATE capture_events SET raw = x'00' WHERE source_kind = 'line' AND position = 1")
    assert cli.main(["audit", "--home", str(home)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    plain = tmp_path / "unstamped"
    plain.mkdir()
    with closing(sqlite3.connect(plain / "lcm.db")) as conn:
        conn.execute("CREATE TABLE metadata(key, value)")
    for invalid in ("relative", str(tmp_path / "absent"), str(plain)):
        assert cli.main(["audit", "--home", invalid]) == 2
        assert json.loads(capsys.readouterr().out)["ok"] is False
    assert "hermes_lcm.mcp_reader" not in sys.modules


def _home_state(home):
    return {str(path.relative_to(home)): (path.stat().st_size, path.stat().st_mtime_ns,
                                        path.read_bytes() if path.is_file() else None)
            for path in [home, *home.rglob("*")]}


@pytest.mark.parametrize("immutable", [False, True])
def test_audit_leaves_home_bytes_sizes_mtimes_and_names_unchanged(captured, immutable):
    home, _lineage = captured
    database = home / "lcm.db"
    if immutable:
        with database.open("r+b") as handle:
            handle.seek(18)
            handle.write(b"\x02\x02")
    before = _home_state(home)
    report = ca.audit(home)
    assert report["ok"] is True and report["opened_immutable"] is immutable
    assert _home_state(home) == before
    assert not Path(f"{database}-wal").exists() and not Path(f"{database}-shm").exists()


def test_round5_hook_latency_worker_log_spool_and_growth(captured):
    home, lineage = captured
    logs = home / "logs"
    logs.mkdir()
    (logs / "hook.log.1").write_text(json.dumps({"event": "PostToolBatch", "ms": 1}) + "\n", encoding="utf-8")
    records = [{"event": "PostToolBatch", "ms": n} for n in range(2, 21)]
    (logs / "hook.log").write_text("bad\n[]\n" + "\n".join(map(json.dumps, records)) + "\n" +
                                 json.dumps({"event": "PostToolBatch", "ms": "bad"}), encoding="utf-8")
    (logs / "worker.log").write_text('{"stage":"sweep","error":"OSError"}\nbad\n' * 2, encoding="utf-8")
    spool = home / "spool"
    spool.mkdir()
    for name in ("one.json", "two.claimed", "three.failed", "ignore.txt"):
        (spool / name).write_text("{}", encoding="utf-8")
    (home / "lcm-large-outputs").mkdir()
    (home / "lcm-large-outputs" / "one").write_bytes(b"abc")
    report = ca.audit(home)
    assert report["hooks_in_process_ms"]["PostToolBatch"] == {"n": 20, "p50": 10, "p95": 19, "max": 20}
    assert "hook_budget_ok" not in report and "hooks" not in report
    assert report["worker_log"] == {"sweep:OSError": 2}
    assert report["spool"] == {"failed": 1, "waiting": 2}
    assert report["growth"]["store_bytes"] == (home / "lcm.db").stat().st_size + 3
    assert report["growth"]["ratio"] == report["growth"]["store_bytes"] / lineage.path.stat().st_size
    (logs / "hook.log").write_text('{"event":"PostToolBatch","ms":75}\n', encoding="utf-8")
    assert ca.audit(home)["hooks_in_process_ms"]["PostToolBatch"]["max"] == 75


def test_projection_errors_are_report_only(tmp_path):
    home = tmp_path / "home"
    path = _transcript(tmp_path, [_entry("user", "broken-projection", 7)])
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, cc.Lineage("s1", None, path))
    finally:
        _close(store)
    report = ca.audit(home)
    assert report["ok"] is True and report["failures"]["lost"] == 0
    assert report["lineages"][0]["projection_errors"] == 1


def test_session_selection_and_markers(tmp_path):
    home = tmp_path / "home"
    store = _open(home, tmp_path / "hermes")
    try:
        for session, agent in (("s1", None), ("s1", "a7"), ("s2", None)):
            path = _transcript(tmp_path, [_entry("user", "u", "hello", session=session)], session=session,
                               name=f"s1/subagents/agent-{agent}.jsonl" if agent else None)
            cc.drain(store, cc.Lineage(session, agent, path))
        store.update_metadata_json(cc.LOCAL_GC_PREFIX + "s1", lambda _current: True)
        store.update_metadata_json(cc.FORGOTTEN_PREFIX + "gone", lambda _current: {"at": 0})
        store.update_metadata_json(cc.FORK_PREFIX + "s1", lambda _current: True)
        cursor = store.read_metadata_json(cc.CURSOR_PREFIX + "claude-code:s1")
        store.update_metadata_json(cc.CURSOR_PREFIX + "claude-code:s1",
                                   lambda _current: {**cursor, "fork_linked": True, "fork_parent": ""})
    finally:
        _close(store)
    report = ca.audit(home, session="s1")
    assert report["ok"] is True
    assert {row["session"] for row in report["lineages"]} == {"s1", "s1/agent:a7"}
    assert report["sessions"]["local_gc"] == ["s1"]
    assert report["sessions"]["fork_parent_unknown"] == ["s1"]
    assert report["sessions"]["forgotten"] == 0 and ca.audit(home)["sessions"]["forgotten"] == 1


@pytest.mark.parametrize("state", ["missing", "replaced", "truncated", "rewritten"])
def test_unreadable_current_generation_is_report_only(captured, monkeypatch, state):
    home, lineage = captured
    monkeypatch.setattr(ca, "cursor_state", lambda *_args: state)
    report = ca.audit(home)
    assert report["ok"] is True
    assert report["lineages"][0]["transcript_state"] == state
    assert report["lineages"][0]["unverifiable"] == 5
    assert report["unverified"] == [lineage.key] and report["verified"] is False


@pytest.mark.parametrize("state", ["missing", "replaced"])
def test_missing_or_replaced_transcript_is_unverified(captured, state):
    home, lineage = captured
    if state == "missing":
        lineage.path.rename(lineage.path.with_suffix(".set-aside"))
    else:
        replacement = lineage.path.with_suffix(".replacement")
        replacement.write_bytes(lineage.path.read_bytes())
        replacement.replace(lineage.path)
    report = ca.audit(home)
    assert report["ok"] is True and report["verified"] is False
    assert report["lineages"][0]["transcript_state"] == state
    assert report["unverified"] == [lineage.key]


def test_round5_stray_rows(captured):
    home, _lineage = captured
    _tamper(home, "INSERT INTO messages(session_id, role, content, timestamp) VALUES('s1', 'user', 'stray', 0)")
    _failed(home, "stray_rows")


@pytest.mark.parametrize("case", ["empty_ids", "projection_raises"])
def test_round5_projection_row_count(captured, monkeypatch, case):
    home, _lineage = captured
    if case == "empty_ids":
        _tamper(home, "UPDATE capture_events SET store_ids = '[]' WHERE source_kind = 'line' AND position = 0")
    else:
        project = cc.project_entry

        def failed(entry, **kwargs):
            if entry.get("uuid") == "u1":
                raise ValueError("synthetic projection failure")
            return project(entry, **kwargs)

        monkeypatch.setattr(ca, "project_entry", failed, raising=False)
    _failed(home, "row_count")


def test_round5_other_projection_is_report_only(captured):
    home, _lineage = captured
    # An older projection gave this line one row more than the current one does.
    _tamper(home, "INSERT INTO messages(session_id, role, content, timestamp) VALUES('s1', 'user', 'older', 0)")
    _tamper(home, "UPDATE capture_events SET projection_version = 'older', "
            "store_ids = json_insert(store_ids, '$[#]', (SELECT MAX(store_id) FROM messages)) "
            "WHERE source_kind = 'line' AND position = 0")
    report = ca.audit(home)
    assert report["ok"] is True
    assert report["lineages"][0]["other_projection"] == 1
    assert report["failures"]["row_count"] == 0


def _held_store(tmp_path, external):
    home = tmp_path / "home"
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    saved = folder / "saved.txt"
    saved.write_text("complete output\n" * 200, encoding="utf-8")
    marker = f"<persisted-output>\nFull output saved to: {saved}\npreview"
    result = _entry("user", "r1", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}])
    path = _transcript(tmp_path, [*CONVERSATION[:2], result])
    config = LCMConfig(large_output_externalization_threshold_chars=100 if external else 12_000)
    store = cc.open_store(cc.admit(home, hermes_home=tmp_path / "hermes"), home, config)
    try:
        cc.drain(store, cc.Lineage("s1", None, path))
        assert cc.capture_files(store, cc.Lineage("s1", None, path)) == 0
    finally:
        _close(store)
    return home


@pytest.mark.parametrize("external", [False, True])
def test_round5_held_text_must_match(capsys, tmp_path, external):
    home = _held_store(tmp_path, external)
    assert ca.audit(home)["ok"] is True
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        content = conn.execute("SELECT content FROM messages WHERE role = 'tool'").fetchone()[0]
    if external:
        [ref] = extract_externalized_refs(content)
        payload_path = home / "lcm-large-outputs" / ref
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["content"] = "damaged output"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        _tamper(home, "UPDATE messages SET content = 'damaged output' WHERE role = 'tool'")
    before = _home_state(home)
    _failed(home, "held_mismatch")
    assert _cli().main(["audit", "--home", str(home)]) == 1
    assert json.loads(capsys.readouterr().out)["failures"]["held_mismatch"] == 1
    assert _home_state(home) == before


def test_growth_counts_the_externalized_payloads(tmp_path):
    """REGRESSION: the growth counted a ``payloads`` folder, while ingest externalizes large
    outputs into the home's ``lcm-large-outputs``, so the store read small by their bytes."""
    home = _held_store(tmp_path, external=True)
    payloads = [path for path in (home / "lcm-large-outputs").rglob("*") if path.is_file()]
    assert payloads
    assert ca.audit(home)["growth"]["store_bytes"] == (home / "lcm.db").stat().st_size + sum(
        path.stat().st_size for path in payloads)


@pytest.mark.parametrize("session", [None, "absent"])
def test_round5_no_vacuous_pass(tmp_path, session):
    home = tmp_path / "home"
    _close(_open(home, tmp_path / "hermes"))
    report = ca.audit(home, session=session)
    assert report["ok"] is True and report["verified"] is False
    assert report["unverified"] == ([cc.lineage_key(session)] if session else [])


def test_round5_subagent_without_cursor(captured):
    home, lineage = captured
    _transcript(lineage.path.parents[2], [_entry("user", "child", "child")],
                name="s1/subagents/agent-a7.jsonl")
    report = ca.audit(home)
    assert report["ok"] is True and report["verified"] is False
    assert report["unverified"] == [cc.lineage_key("s1", "a7")]


@pytest.mark.parametrize("pending", ["line", "file", "spool"])
def test_round5_pending_is_not_verified(captured, pending):
    home, lineage = captured
    if pending == "line":
        with lineage.path.open("ab") as handle:
            handle.write(json.dumps(_entry("user", "later", "later")).encode() + b"\n")
    elif pending == "file":
        (cc.session_dir(lineage.path) / "tool-results" / "later.txt").write_text("later", encoding="utf-8")
    else:
        (home / "spool").mkdir()
        (home / "spool" / "waiting.claimed").write_text("{}", encoding="utf-8")
    report = ca.audit(home)
    assert report["ok"] is True and report["verified"] is False
    assert report["unverified"] == ([] if pending == "spool" else [lineage.key])


@pytest.mark.parametrize("matching_copy", [False, True])
def test_round5_older_generation_requires_full_match(tmp_path, matching_copy):
    home = tmp_path / "home"
    original = [CONVERSATION[0], _entry("user", "u2", "original second line")]
    path = _transcript(tmp_path, original)
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, lineage)
        if matching_copy:
            path.with_name(path.name + ".superseded-z").write_bytes(path.read_bytes())
        wrong = [original[0], _entry("user", "u2", "different second line")]
        path.with_name(path.name + ".superseded-a").write_text(
            "".join(json.dumps(entry) + "\n" for entry in wrong), encoding="utf-8")
        path.write_text(json.dumps(_entry("user", "new", "new generation")) + "\n", encoding="utf-8")
        cc.drain(store, lineage)
    finally:
        _close(store)
    report = ca.audit(home)
    assert report["ok"] is True
    assert report["failures"]["digest_mismatch"] == report["failures"]["lost_lines"] == 0
    assert report["lineages"][0]["generations_gone"] == (0 if matching_copy else 1)
    assert report["verified"] is matching_copy


@pytest.mark.parametrize("marker", [{"at": 0}, False])
def test_round5_forgotten_is_skipped(captured, tmp_path, monkeypatch, marker):
    home, _lineage = captured
    store = _open(home, tmp_path / "hermes")
    try:
        store.delete_session_messages("s1")
        store.update_metadata_json(cc.FORGOTTEN_PREFIX + "s1", lambda _current: marker)
    finally:
        _close(store)

    def refused(*_args, **_kwargs):
        pytest.fail("forgotten lineage was checked")

    monkeypatch.setattr(ca, "_ledger", refused)
    monkeypatch.setattr(ca, "cursor_state", refused)
    report = ca.audit(home)
    assert report["ok"] is True and report["verified"] is False and report["unverified"] == []
    assert report["lineages"][0]["forgotten"] is True
    assert not any(report["failures"].values())


def test_round5_releases_each_lineage_before_next_snapshot(captured, tmp_path, monkeypatch):
    home, _lineage = captured
    path = _transcript(tmp_path, [_entry("user", "other", "other")], session="s2")
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, cc.Lineage("s2", None, path))
    finally:
        _close(store)
    original = ca._snapshot
    refs = []

    class Tracked(list):
        pass

    def tracked(*args, **kwargs):
        assert all(ref() is None for ref in refs), "previous lineage ledger is still retained"
        snapshot = list(original(*args, **kwargs))
        snapshot[2] = Tracked(snapshot[2])
        refs.append(weakref.ref(snapshot[2]))
        return tuple(snapshot)

    monkeypatch.setattr(ca, "_snapshot", tracked)
    assert ca.audit(home)["ok"] is True
    assert len(refs) == 2 and all(ref() is None for ref in refs)


def test_round5_messages_keep_only_call_ids(captured):
    home, lineage = captured
    with closing(sqlite3.connect(f"{(home / 'lcm.db').as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        messages = ca._snapshot(conn, cc.CURSOR_PREFIX + lineage.key)[3]
    assert all(set(row) == {"store_id", "role", "tool_call_id", "call_ids"} for row in messages)
    assert [row["call_ids"] for row in messages if row["role"] == "assistant"] == [["toolu_1"], []]


def test_round5_projected_without_uuid_is_reported(tmp_path):
    home = tmp_path / "home"
    entry = _entry("user", "", "no uuid")
    path = _transcript(tmp_path, [entry])
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, cc.Lineage("s1", None, path))
    finally:
        _close(store)
    report = ca.audit(home)
    assert report["ok"] is True and report["lineages"][0]["projected_without_uuid"] == 1


def test_each_sessions_first_hook_and_precompact_paths_are_reported_apart(captured):
    """For #2141: a session's first hook (a cold interpreter start) stays out of its event's
    percentiles, PreCompact keeps its own event, and its runs are counted by path."""
    home, _lineage = captured
    logs = home / "logs"
    logs.mkdir()
    rows = [{"event": "SessionStart", "session": "a", "ms": 400, "outcome": "worker_started"},
            {"event": "PostToolBatch", "session": "a", "ms": 3, "outcome": "worker_running"},
            {"event": "PostToolBatch", "session": "b", "ms": 300, "outcome": "worker_started"},
            {"event": "PostToolBatch", "session": "b", "ms": 5, "outcome": "worker_running"},
            {"event": "PreCompact", "session": "a", "ms": 900, "outcome": "drained"},
            {"event": "PreCompact", "session": "a", "ms": 2, "outcome": "worker_running"},
            {"event": "PreCompact", "session": "b", "ms": 10000, "outcome": "drain_bound+worker"},
            {"event": "PreCompact", "session": "b", "ms": 10400, "outcome": "drain_incomplete+holder"},
            {"event": "PreCompact", "session": "b", "ms": 10100, "outcome": "drain_bound+holder"},
            {"event": "PreCompact", "session": "c", "ms": 40, "outcome": "runtime_missing"}]
    (logs / "hook.log").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = ca.audit(home)
    hooks = report["hooks_in_process_ms"]
    assert hooks["first"] == {"n": 3, "max": 400}
    assert "SessionStart" not in hooks and hooks["PostToolBatch"]["n"] == 2 and hooks["PostToolBatch"]["max"] == 5
    assert hooks["PreCompact"]["n"] == 5 and hooks["PreCompact"]["max"] == 10400
    # Only the worker knows its deadline ended the run (drain_bound): an incomplete drain that took
    # as long (a failed record, a store left in WAL mode, the lock taken) did not reach it.
    assert report["precompact"] == {"drain": 4, "poke": 1, "bound": 2, "other": 1}


@pytest.mark.parametrize("case", ["listed_twice", "listed_by_two_entries"])
def test_a_projected_row_id_listed_twice_fails_row_count(tmp_path, case):
    """REGRESSION: a lost row hid behind an id listed twice, by one projected entry or by two:
    the count still matched, and every listed id existed."""
    home = tmp_path / "home"
    two_rows = _entry("user", "r2", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "the output"},
                                     {"type": "text", "text": "and a note"}])
    path = _transcript(tmp_path, [CONVERSATION[0], CONVERSATION[1], two_rows])
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, cc.Lineage("s1", None, path))
    finally:
        _close(store)
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        ids = {uid: json.loads(store_ids) for uid, store_ids in conn.execute(
            "SELECT uid, store_ids FROM capture_events WHERE source_kind = 'line'")}
    assert len(ids["r2"]) == 2 and ca.audit(home)["ok"] is True
    if case == "listed_twice":  # the note's row is lost; the tool row's id is listed in its place
        lost, uid, listed = ids["r2"][1], "r2", [ids["r2"][0]] * 2
    else:  # the first prompt's row is lost; it lists the note's row instead
        lost, uid, listed = ids["u1"][0], "u1", [ids["r2"][1]]
    _tamper(home, "DELETE FROM messages WHERE store_id = ?", (lost,))
    _tamper(home, "UPDATE capture_events SET store_ids = ? WHERE uid = ?", (json.dumps(listed), uid))
    _failed(home, "row_count")


@pytest.mark.parametrize("case", ["no_cursor", "subagent_without_cursor", "subagent_not_on_disk", "with_cursor",
                                  "unreadable"])
def test_a_failed_record_without_a_cursor_is_unverified(captured, tmp_path, case):
    """REGRESSION: a record set aside as ``.failed`` before its transcript had a cursor names a
    session nothing else shows, and a whole-home audit verified the store without it. A subagent
    transcript not on disk still counts: one removed before capture was dropped, and verified."""
    home, _lineage = captured
    (home / "spool").mkdir()
    session, agent = {"no_cursor": ("s9", None), "subagent_without_cursor": ("s1", "a3"),
                      "subagent_not_on_disk": ("s1", "a3")}.get(case, ("s1", None))
    agent_path = tmp_path / "elsewhere" / "agent-a3.jsonl"  # outside subagents/: only the record names it
    if case == "subagent_without_cursor":
        agent_path.parent.mkdir()
        agent_path.write_text("{}\n", encoding="utf-8")
    text = "{" if case == "unreadable" else json.dumps({
        "host": "claude-code", "event": "SubagentStop" if agent else "Stop", "session_id": session,
        "agent_id": agent, "agent_transcript_path": str(agent_path) if agent else None})
    (home / "spool" / "k.failed").write_text(text, encoding="utf-8")
    report = ca.audit(home)
    expected = {"no_cursor": [cc.lineage_key("s9")], "subagent_without_cursor": [cc.lineage_key("s1", "a3")],
                "subagent_not_on_disk": [cc.lineage_key("s1", "a3")], "with_cursor": [],
                "unreadable": ["spool:unreadable_failed_record"]}[case]
    assert report["ok"] is True and report["spool"]["failed"] == 1
    assert report["unverified"] == expected and report["verified"] is (not expected)
    if case == "no_cursor":  # another session's record does not touch this one's verdict
        assert ca.audit(home, session="s1")["unverified"] == []


def test_a_forgotten_sessions_subagent_without_a_cursor_is_unverified(captured, tmp_path):
    """REGRESSION: the writer still captures a forgotten session's subagents (their rows are their
    own), but the audit skipped them, so one not captured yet left nothing unverified."""
    home, lineage = captured
    store = _open(home, tmp_path / "hermes")
    try:
        store.delete_session_messages("s1")
        store.update_metadata_json(cc.FORGOTTEN_PREFIX + "s1", lambda _current: {"at": 0})
    finally:
        _close(store)
    _transcript(lineage.path.parents[2], [_entry("user", "child", "child")], name="s1/subagents/agent-a7.jsonl")
    report = ca.audit(home)
    assert report["unverified"] == [cc.lineage_key("s1", "a7")] and report["verified"] is False


def test_an_empty_generation_never_matches_a_set_aside_copy(tmp_path):
    """A generation with no ledger rows may have lost them all, so no set-aside copy, an empty
    one included, can stand in for it: it stays gone (unverified, never failed). An empty
    transcript replaced later is the one case this leaves unverified for nothing."""
    home = tmp_path / "home"
    path = _transcript(tmp_path, [])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, tmp_path / "hermes")
    try:
        cc.drain(store, lineage)
        assert store.read_metadata_json(cc.CURSOR_PREFIX + lineage.key)["generation"] == 0
        os.rename(path, path.with_name(path.name + ".superseded-1"))
        _transcript(tmp_path, [CONVERSATION[0]])
        cc.drain(store, lineage)
        assert store.read_metadata_json(cc.CURSOR_PREFIX + lineage.key)["generation"] == 1
    finally:
        _close(store)
    report = ca.audit(home)
    assert report["ok"] is True and report["lineages"][0]["generations_gone"] == 1
    assert report["verified"] is False and report["unverified"] == [lineage.key]
