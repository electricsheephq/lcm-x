"""Synthetic outside-in capture proofs: hook -> worker CLI -> contract SQL checker."""
import ast
from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

from tests.test_claude_code_store import _KILLED_DRIVER

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/capture_outside_in.py"
CAPTURE = ROOT / "scripts/lcm_x_capture.py"
FIXTURES = sorted((ROOT / "tests/fixtures/capture_sessions").glob("*.jsonl"))


def run(script, *args, env, payload=None, accepted=(0,)):
    done = subprocess.run([sys.executable, "-I", "-S", str(script), *map(str, args)],
                          input=payload, env=env, capture_output=True, text=True, timeout=30)
    assert done.returncode in accepted, {"exit": done.returncode}
    return done


def hook(home, env, session, path, event="SessionStart", **extra):
    # A real holder prevents detached workers; the explicit CLI invocation owns the drain.
    with (home / "capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(ROOT / "adapters/claude-code/bin/lcmx_hook.py", env=env, payload=json.dumps({
            "hook_event_name": event, "session_id": session, "transcript_path": str(path), **extra}))
    assert list((home / "spool").glob("*.json"))


def read_sql(home, sql):
    with closing(sqlite3.connect((home / "lcm.db").as_uri() + "?mode=ro", uri=True)) as conn:
        return conn.execute(sql).fetchall()


def capture_session(base, fixture):
    home = base / "store"
    home.mkdir(parents=True)
    env = {"PATH": os.defpath, "HOME": str(base / "user"), "HERMES_HOME": str(base / "hermes"),
           "CLAUDE_CONFIG_DIR": str(base / "claude"), "LCMX_CC_HOME": str(home),
           "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(ROOT)}
    rows = [json.loads(line) for line in fixture.read_text().splitlines()]
    sessions = {}
    for row in rows:
        session = row["sessionId"] + ("/agent:" + row["agentId"] if row.get("agentId") else "")
        sessions.setdefault(session, []).append(row)
    inputs = []
    for session, entries in sessions.items():
        parent, _, agent = session.partition("/agent:")
        folder = base / "claude/projects/synthetic"
        path = folder / parent / "subagents" / f"agent-{agent}.jsonl" if agent else folder / f"{parent}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        inputs.append((session, path))
        split = {"02": len(entries) - 1, "03": 1, "05": 4, "09": 2}.get(fixture.stem[:2], len(entries))
        path.write_text("".join(json.dumps(row) + "\n" for row in entries[:split]))
        if not agent:
            hook(home, env, parent, path, "SessionStart",
                 **({"source": "fork"} if parent.startswith("fork-") else {}))
            run(CAPTURE, "worker", "--home", home, env=env)
        if split < len(entries):
            before = read_sql(home, "SELECT count(*) FROM messages")[0][0]
            with path.open("a") as handle:
                handle.write("".join(json.dumps(row) + "\n" for row in entries[split:]))
            hook(home, env, parent, path, "SessionStart", source="resume")
            run(CAPTURE, "worker", "--home", home, env=env)
            if fixture.stem.startswith("02"):
                assert read_sql(home, "SELECT count(*) FROM messages")[0][0] == before
    # The public worker sweep discovers subagent transcripts after the parent was registered.
    run(CAPTURE, "worker", "--home", home, env=env)
    return home, env, inputs


def verdict(home, env, inputs):
    args = [arg for session, path in inputs for arg in ("--transcript", f"{session}={path}")]
    checked = run(CHECKER, "--home", home, *args, env=env, accepted=(0, 1))
    audited = run(CAPTURE, "audit", "--home", home, env=env, accepted=(0, 1))
    return checked, json.loads(checked.stdout), json.loads(audited.stdout)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return capture_session(tmp_path_factory.mktemp("outside-in-clean"), FIXTURES[0])


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_fixture(tmp_path_factory, fixture, record_property):
    home, env, inputs = capture_session(tmp_path_factory.mktemp(fixture.stem), fixture)
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    record_property("home", str(home))
    assert report["pass"] == audit["ok"], "STOP: clean fixture disagreement; both outputs recorded in JUnit"
    assert checked.returncode == 0 and audit["ok"] and audit["verified"]
    assert {r["lineage"] for r in report["lineages"]} == {r["lineage"] for r in audit["lineages"]}
    assert all(r["pass"] for r in report["lineages"])
    if fixture.stem.startswith("06"):
        assert read_sql(home, "SELECT count(*) FROM message_relations WHERE kind='fork_copy_of'")[0][0] == 4
        assert dict(read_sql(home, "SELECT child_session_id,parent_session_id FROM session_lineage")) == {
            "fork-a": "parent", "fork-b": "parent"}
    if fixture.stem.startswith("08"):
        assert read_sql(home, "SELECT child_session_id,parent_session_id FROM session_lineage") == [
            ("subparent/agent:child", "subparent")]
    if fixture.stem.startswith("09"):
        assert read_sql(home, "SELECT related_store_id,store_id FROM message_relations WHERE kind='supersedes'") == [(1, 3)]
        assert read_sql(home, "SELECT store_id FROM messages WHERE role='tool'") == [(2,)]


def test_generated_batch(tmp_path_factory, record_property):
    base = tmp_path_factory.mktemp("generated-batch")
    fixture = base / "batch.jsonl"
    run(ROOT / "tests/fixtures/capture_sessions/generate_batch.py", fixture, env={"PATH": os.defpath})
    home, env, inputs = capture_session(base, fixture)
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    assert report["pass"] == audit["ok"], "STOP: clean batch disagreement; outputs recorded in JUnit"
    assert checked.returncode == 0 and audit["ok"] and audit["verified"]


def test_worker_killed_mid_batch_then_rerun(tmp_path, record_property):
    home = tmp_path / "store"
    home.mkdir()
    env = {"PATH": os.defpath, "HOME": str(tmp_path / "user"), "HERMES_HOME": str(tmp_path / "hermes"),
           "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"), "LCMX_CC_HOME": str(home),
           "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(ROOT)}
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()]
    # Two turns with paired tools: each two-line batch holds a whole call/result pair.
    entries = [rows[index] for index in (1, 2, 5, 6)]
    path = tmp_path / "claude/projects/synthetic/plain.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in entries))
    hook(home, env, "plain", path)
    stalled = tmp_path / "stalled"
    worker = subprocess.Popen([sys.executable, "-I", "-S", "-c", _KILLED_DRIVER, str(ROOT), str(home),
                               env["HERMES_HOME"], str(stalled)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        end = time.monotonic() + 30
        while not stalled.exists() and worker.poll() is None and time.monotonic() < end:
            time.sleep(0.05)
    finally:
        worker.kill()
        _out, error = worker.communicate(timeout=10)
    assert stalled.exists(), error[-2000:]
    assert worker.returncode == -signal.SIGKILL
    assert list((home / "spool").glob("*.claimed"))
    assert read_sql(home, "SELECT position FROM capture_events WHERE source_kind='line' ORDER BY position") == [(0,), (1,)]
    assert read_sql(home, "SELECT role,tool_call_id FROM messages ORDER BY store_id") == [
        ("assistant", None), ("tool", "call1"), ("user", None)]

    # Mutant: omit the rerun. Only the second batch's two projected rows are missing.
    checked = run(CHECKER, "--home", home, "--transcript", f"plain={path}", env=env, accepted=(1,))
    audited = run(CAPTURE, "audit", "--home", home, env=env)
    report, audit = json.loads(checked.stdout), json.loads(audited.stdout)
    record_property("mutant_checker", checked.stdout)
    record_property("mutant_checker_exit", checked.returncode)
    record_property("mutant_audit", audited.stdout)
    record_property("mutant_audit_exit", audited.returncode)
    assert checked.returncode == 1 and not report["pass"]
    # The audit counts the claimed record as pending work, not loss: consistent so far, never verified.
    assert audit["ok"] and not audit["verified"] and audit["spool"]["waiting"] == 1
    [lineage] = report["lineages"]
    assert (lineage["expected"], lineage["actual"], lineage["lost"], lineage["duplicated"],
            lineage["order_kept"], lineage["unpaired"]) == (5, 3, 2, 0, True, 0)

    run(CAPTURE, "worker", "--home", home, env=env)
    checked = run(CHECKER, "--home", home, "--transcript", f"plain={path}", env=env)
    audited = run(CAPTURE, "audit", "--home", home, env=env)
    report, audit = json.loads(checked.stdout), json.loads(audited.stdout)
    record_property("checker", checked.stdout)
    record_property("checker_exit", checked.returncode)
    record_property("audit", audited.stdout)
    record_property("audit_exit", audited.returncode)
    assert checked.returncode == 0 and report["pass"] and audit["ok"] and audit["verified"]
    [lineage] = report["lineages"]
    assert (lineage["expected"], lineage["actual"], lineage["lost"], lineage["duplicated"],
            lineage["order_kept"], lineage["unpaired"]) == (5, 5, 0, 0, True, 0)
    assert read_sql(home, "SELECT position FROM capture_events WHERE source_kind='line' ORDER BY position") == [
        (0,), (1,), (2,), (3,)]
    assert not any(path.suffix in {".json", ".claimed", ".failed"} for path in (home / "spool").iterdir())


@pytest.mark.parametrize("mutant, expected, audit_asserted", [
    ("loss", (1, 0, True, 0), True), ("duplicate", (0, 1, True, 0), True),
    ("reorder", (0, 0, False, 0), False), ("tool", (1, 0, True, 1), True),
    ("double-binding", (0, 1, True, 0), True), ("same-uid-twice", (0, 1, True, 0), False),
    ("fork-rebound", (1, 1, True, 0), True), ("moved-binding", (1, 2, True, 0), False),
    ("cross-lineage", (0, 1, True, 0), False), ("stray-lineage", (0, 1, True, 0), False),
])
def test_mutants(clean, tmp_path_factory, mutant, expected, audit_asserted, record_property):
    source, env, inputs = clean
    base = tmp_path_factory.mktemp("mutant-" + mutant)
    if mutant == "double-binding":
        fixture = base / "same-text.jsonl"
        fixture.write_text("".join(json.dumps(dict(type="user", uuid=uid, sessionId="same-text",
                           message=dict(role="user", content=text))) + "\n"
                           for uid, text in [("u1", "ok"), ("u2", "ok"), ("u3", "done")]))
        source, env, inputs = capture_session(base / "source", fixture)
    if mutant in {"cross-lineage", "fork-rebound"}:
        source, env, inputs = capture_session(base / "source", FIXTURES[4])
    home = base / "store"
    home.mkdir()
    with closing(sqlite3.connect((source / "lcm.db").as_uri() + "?mode=ro", uri=True)) as src, \
            closing(sqlite3.connect(home / "lcm.db")) as dst:
        src.backup(dst)  # One database snapshot, no copied store/home trees.
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        if mutant in {"loss", "tool"}:
            row_id = conn.execute("SELECT store_id FROM messages WHERE role=? LIMIT 1",
                                  ("user" if mutant == "loss" else "tool",)).fetchone()[0]
            conn.execute("DELETE FROM host_uid_bindings WHERE store_id=?", (row_id,))
            conn.execute("DELETE FROM messages WHERE store_id=?", (row_id,))
        elif mutant == "duplicate":
            columns = [r[1] for r in conn.execute("PRAGMA table_info(messages)") if r[1] != "store_id"]
            names = ",".join(columns)
            new = conn.execute(f"INSERT INTO messages({names}) SELECT {names} FROM messages WHERE store_id=1").lastrowid
            conn.execute("INSERT INTO host_uid_bindings(store_id,uid,lineage_key,kind,binding_version) "
                         "SELECT ?,uid,lineage_key,'version',binding_version FROM host_uid_bindings WHERE store_id=1", (new,))
        elif mutant == "double-binding":
            conn.execute("DELETE FROM host_uid_bindings WHERE store_id=2")
            conn.execute("DELETE FROM messages WHERE store_id=2")
            conn.execute("INSERT INTO host_uid_bindings(store_id,uid,lineage_key,kind,binding_version) "
                         "VALUES(1,'u2','claude-code:same-text','version',1)")
        elif mutant in {"same-uid-twice", "cross-lineage", "stray-lineage"}:
            lineage = {"cross-lineage": "claude-code:fork-a", "stray-lineage": "claude-code:zzz"}.get(mutant)
            conn.execute("INSERT INTO host_uid_bindings(store_id,uid,lineage_key,kind,binding_version) "
                         "SELECT store_id,uid,COALESCE(?,lineage_key),'version',binding_version "
                         "FROM host_uid_bindings WHERE store_id=1", (lineage,))
        elif mutant == "fork-rebound":
            conn.execute("DELETE FROM messages WHERE store_id=3")
            conn.execute("UPDATE host_uid_bindings SET store_id=1 WHERE store_id=3")
        elif mutant == "moved-binding":
            conn.execute("UPDATE host_uid_bindings SET lineage_key='claude-code:zzz' WHERE store_id=1")
        else:
            # Move identities with rows, so only chronology changes; ledger references remain stale.
            for table in ("messages", "host_uid_bindings"):
                conn.execute(f"UPDATE {table} SET store_id=-store_id WHERE store_id IN (1,2)")
                conn.execute(f"UPDATE {table} SET store_id=CASE store_id WHEN -1 THEN 2 ELSE 1 END WHERE store_id<0")
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    record_property("home", str(home))
    assert checked.returncode == 1 and not report["pass"]
    observed = (sum(r["lost"] for r in report["lineages"]), sum(r["duplicated"] for r in report["lineages"]),
                all(r["order_kept"] for r in report["lineages"]), sum(r["unpaired"] for r in report["lineages"]))
    record_property("expectation", json.dumps(expected))
    record_property("audit_asserted", json.dumps(audit_asserted))
    assert observed == expected
    for row in report["lineages"]:
        assert len(row["store_ids"]) == len(set(row["store_ids"])) == row["actual"]
        own_expected = expected
        if mutant in {"cross-lineage", "fork-rebound"}:
            own_expected = {"claude-code:parent": (0, 1, True, 0),
                            "claude-code:fork-a": (int(mutant == "fork-rebound"), 0, True, 0)}.get(
                                row["lineage"], (0, 0, True, 0))
        assert (row["lost"], row["duplicated"], row["order_kept"], row["unpaired"]) == own_expected
    if audit_asserted:
        assert not audit["ok"]


def test_reader_stdio(clean):
    home, env, _ = clean
    requests = [dict(jsonrpc="2.0", id=1, method="initialize", params={
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "outside-in", "version": "1"}}),
        dict(jsonrpc="2.0", method="notifications/initialized", params={}),
        dict(jsonrpc="2.0", id=2, method="tools/call", params={
            "name": "lcm_grep", "arguments": {"query": "synthetic amber compass beacon", "mode": "full_text"}})]
    done = run(ROOT / "scripts/lcm_x_mcp.py", "--store", home / "lcm.db", "--hermes-home", home,
               env=env, payload="".join(json.dumps(r) + "\n" for r in requests))
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2] and all("error" not in r for r in replies)
    found = json.loads(replies[1]["result"]["content"][0]["text"])
    assert any("synthetic amber compass beacon" in r.get("snippet", "").replace(">>>", "").replace("<<<", "")
               for r in found["results"])


def test_stdlib_import_boundary():
    tree = ast.parse(CHECKER.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not any(alias.name in {"import_module", "__import__", "eval", "exec"} for alias in node.names)
            imports.append(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec"})
            assert not (isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"import_module", "__import__"})
    assert imports and set(imports) <= sys.stdlib_module_names


@pytest.mark.parametrize("shape", ["empty", "payload", "persisted", "revision", "unexpected",
                                  "blank", "undecodable", "utf8", "uidless", "docs", "error",
                                  "store-marker", "digest", "bogus", "persisted-payload"])
def test_projection_fidelity(tmp_path, shape, record_property):
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()]
    base = tmp_path / "capture"
    if shape == "empty":
        rows[0]["message"]["content"] = ""
    if shape in {"payload", "digest"}:
        rows[2]["message"]["content"][0]["content"] = "synthetic " * 7500
        assert len(rows[2]["message"]["content"][0]["content"]) == 75000
    if shape.startswith("persisted"):
        saved = base / "claude/projects/synthetic/plain/tool-results/output.txt"
        saved.parent.mkdir(parents=True)
        saved.write_text("synthetic recovered full output" if shape == "persisted" else "synthetic " * 7500)
        rows[2]["message"]["content"][0]["content"] = (
            f"<persisted-output>\nFull output saved to: {saved}\n\nPreview:\nsynthetic\n</persisted-output>")
    if shape == "revision":
        rows = [rows[1], rows[2], rows[5], json.loads(json.dumps(rows[2]))]
        rows[1]["message"]["content"] = rows[1]["message"]["content"][:1]
        rows[-1]["message"]["content"] = [rows[1]["message"]["content"][0],
                                             json.loads(FIXTURES[0].read_text().splitlines()[6])["message"]["content"][0]]
    fixture = tmp_path / ("09_projection.jsonl" if shape == "revision" else "01_projection.jsonl")
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows))
    home, env, inputs = capture_session(base, fixture)
    if shape in {"blank", "undecodable", "utf8", "uidless"}:
        path = inputs[0][1]
        lines = path.read_text().splitlines(keepends=True)
        lines.insert(len(lines) if shape == "blank" else 1, {"blank": " \t\n", "undecodable": "{broken\n", "utf8": "BAD_UTF8\n",
                         "uidless": json.dumps(dict(type="user", message=dict(content="raw"))) + "\n"}[shape])
        path.write_bytes("".join(lines).encode().replace(b"BAD_UTF8", b"\xff"))
        hook(home, env, "plain", path)
        run(CAPTURE, "worker", "--home", home, env=env)
    if shape == "unexpected":
        extra = inputs[0][1].with_name("extra.jsonl")
        extra.write_text(json.dumps(dict(type="user", uuid="extra", sessionId="extra", message=dict(content="extra"))) + "\n")
        hook(home, env, "extra", extra)
        run(CAPTURE, "worker", "--home", home, env=env)
    if shape in {"store-marker", "digest", "bogus"}:
        content = "<persisted-output>junk" if shape == "store-marker" else "[Externalized tool output: bogus]"
        if shape == "digest":
            content = read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call1'")[0][0]
            digest = content.rsplit("_", 2)[1]
            content = content.replace(digest, "0" * 12 if digest != "0" * 12 else "1" * 12)
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content=? WHERE tool_call_id='call1'", (content,))
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    if shape in {"store-marker", "digest", "bogus"}:
        assert checked.returncode == 1 and not report["pass"]
        assert [(r["lost"], r["duplicated"], r["order_kept"], r["unpaired"], r["unverified"], r["unprojected"])
                for r in report["lineages"]] == [(1, 1, True, 0, 0, 0)]
        return
    if shape == "empty":
        clean_verdict = checked, report, audit
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("DELETE FROM host_uid_bindings WHERE store_id=1")
            conn.execute("DELETE FROM messages WHERE store_id=1")
        checked, report, audit = verdict(home, env, inputs)
        record_property("mutant", checked.stdout)
        record_property("mutant_audit", json.dumps(audit, sort_keys=True))
        assert checked.returncode == 1 and not report["pass"] and not audit["ok"]
        assert [(r["lost"], r["duplicated"], r["order_kept"], r["unpaired"]) for r in report["lineages"]] == [(1, 0, True, 0)]
        checked, report, audit = clean_verdict
    assert audit["ok"]
    assert checked.returncode == int(shape == "unexpected") and report["pass"] == (shape != "unexpected")
    assert all((r["lost"], r["duplicated"], r["order_kept"], r["unpaired"]) == (0, 0, True, 0) for r in report["lineages"])
    if shape == "unexpected":
        assert report["unexpected_sessions"] == [{"session_id": "extra", "rows": 1}]
    if shape in {"persisted", "persisted-payload", "undecodable", "utf8", "uidless"}:
        assert sum(r["unverified" if shape.startswith("persisted") else "unprojected"] for r in report["lineages"]) == 1
    if shape == "payload":
        assert sum(r["unverified"] for r in report["lineages"]) == 0
        assert read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call1'")[0][0].startswith("[Externalized tool output:")
    if shape == "persisted-payload":
        assert read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call1'")[0][0].startswith("[Externalized tool output:")
    if shape == "persisted":
        assert read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call1'") == [("synthetic recovered full output",)]
    if shape == "docs":
        paragraph = next(line for line in (ROOT / "docs/claude-code-adapter.md").read_text().splitlines() if "An independent row checker" in line)
        assert all(word in paragraph for word in ("unverified", "unprojected", "unexpected_sessions", "every captured transcript"))
    if shape == "error":
        checked = run(CHECKER, "--home", home, "--transcript", "plain=" + str(tmp_path / "PRIVATE_missing"), env=env, accepted=(1,))
        assert json.loads(checked.stdout)["error_type"] == "FileNotFoundError" and "PRIVATE" not in checked.stdout


@pytest.mark.parametrize("preview_first", [True, False], ids=["preview-first", "preview-last"])
@pytest.mark.parametrize("corrupt", [False, True], ids=["clean", "corrupted"])
def test_preview_occurrence(tmp_path, preview_first, corrupt, record_property):
    base = tmp_path / "capture"
    saved = base / "claude/projects/synthetic/plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("synthetic recovered output")
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[-1]["message"]["content"] = rows[-1]["message"]["content"][:1]
    rows.append(json.loads(json.dumps(rows[-1])))
    rows[1 if preview_first else 2]["message"]["content"][0]["content"] = (
        f"<persisted-output>\nFull output saved to: {saved}\n\nPreview:\nsynthetic\n</persisted-output>")
    fixture = tmp_path / "09_preview.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows))
    home, env, inputs = capture_session(base, fixture)
    assert read_sql(home, "SELECT store_id,related_store_id FROM message_relations WHERE kind='supersedes'") == [(3, 2)]
    if corrupt:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content='CORRUPTED' WHERE store_id=?", (3 if preview_first else 2,))
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    assert checked.returncode == int(corrupt) and report["pass"] == (not corrupt)
    assert [(r["lost"], r["duplicated"], r["order_kept"], r["unpaired"], r["unverified"], r["unprojected"])
            for r in report["lineages"]] == [(int(corrupt), int(corrupt), True, 0, 1, 0)]
    assert not report["unexpected_sessions"]
    if not corrupt:
        assert audit["ok"]


@pytest.mark.parametrize("changed", [False, True], ids=["retained-id", "changed-id"])
def test_call_revision_ancestry(tmp_path, changed, record_property):
    rows = [json.loads(line) for line in next(p for p in FIXTURES if p.stem.startswith("09")).read_text().splitlines()]
    if changed:
        rows[0]["message"]["content"][0]["id"] = "old-tool"
    fixture = tmp_path / "09_call.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows))
    home, env, inputs = capture_session(tmp_path / "capture", fixture)
    assert read_sql(home, "SELECT store_id,role FROM messages ORDER BY store_id") == [(1, "assistant"), (2, "tool"), (3, "assistant")]
    assert read_sql(home, "SELECT store_id,related_store_id FROM message_relations WHERE kind='supersedes'") == [(3, 1)]
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    assert checked.returncode == int(changed) and report["pass"] == (not changed)
    assert [(r["lost"], r["duplicated"], r["order_kept"], r["unpaired"], r["unverified"], r["unprojected"])
            for r in report["lineages"]] == [(0, 0, True, int(changed), 0, 0)]
    assert not report["unexpected_sessions"] and audit["ok"]


def crafted_revision(conn, new, old):
    # Plain public captures need not create this optional contracted table.
    conn.execute("CREATE TABLE IF NOT EXISTS message_relations (relation_id INTEGER PRIMARY KEY, "
                 "store_id INTEGER NOT NULL, kind TEXT NOT NULL, related_store_id INTEGER, "
                 "ordinal INTEGER, observed_at REAL, created_at REAL)")
    conn.execute("INSERT INTO message_relations(store_id,kind,related_store_id) VALUES(?,'supersedes',?)", (new, old))


@pytest.mark.parametrize("uids", [("A", "B"), ("A#0", "B#1")], ids=["plain", "split"])
def test_cross_uid_revision(tmp_path, uids, record_property):
    rows = [dict(type="assistant", uuid=uids[0], sessionId="crafted", message=dict(content=[
                dict(type="tool_use", id="c", name="synthetic", input={})])),
            dict(type="assistant", uuid=uids[1], sessionId="crafted", message=dict(content="final"))]
    fixture = tmp_path / "01_cross_uid.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows))
    home, env, inputs = capture_session(tmp_path / "capture", fixture)
    assert read_sql(home, "SELECT store_id,uid FROM host_uid_bindings ORDER BY store_id") == list(enumerate(uids, 1))
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        crafted_revision(conn, 2, 1)
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    assert checked.returncode == 1
    assert report == {"pass": False, "errors": 1, "error_type": "ValueError"}


@pytest.mark.parametrize("unbound", [1, 3], ids=["old", "new"])
def test_unbound_revision(tmp_path, unbound, record_property):
    home, env, inputs = capture_session(tmp_path / "capture", next(p for p in FIXTURES if p.stem.startswith("09")))
    assert read_sql(home, "SELECT store_id,related_store_id FROM message_relations WHERE kind='supersedes'") == [(3, 1)]
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("DELETE FROM host_uid_bindings WHERE store_id=?", (unbound,))
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    assert checked.returncode == 1
    assert report == {"pass": False, "errors": 1, "error_type": "ValueError"}


@pytest.mark.parametrize("literal", ["reference", "marker"])
@pytest.mark.parametrize("duplicate", [False, True], ids=["once", "twice"])
def test_literal_tool_output(tmp_path, literal, duplicate, record_property):
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()]
    rows[2]["message"]["content"] = rows[2]["message"]["content"][:1]
    rows[2]["message"]["content"][0]["content"] = "synthetic " * 7500
    fixture = tmp_path / "01_literal.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows[:3]))
    home, env, inputs = capture_session(tmp_path / "capture", fixture)
    reference = read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call1'")[0][0]
    assert reference.startswith("[Externalized tool output:") and "chars=75000; bytes=75000;" in reference
    text = reference if literal == "reference" else "[Externalized tool output: bogus]"
    rows[6]["message"]["content"][0]["content"] = text
    with inputs[0][1].open("a") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows[3:]))
    hook(home, env, "plain", inputs[0][1])
    run(CAPTURE, "worker", "--home", home, env=env)
    assert read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call2'") == [(text,)]
    if duplicate:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            old = conn.execute("SELECT store_id FROM messages WHERE tool_call_id='call2'").fetchone()[0]
            names = ",".join(r[1] for r in conn.execute("PRAGMA table_info(messages)") if r[1] != "store_id")
            new = conn.execute(f"INSERT INTO messages({names}) SELECT {names} FROM messages WHERE store_id=?", (old,)).lastrowid
            conn.execute("INSERT INTO host_uid_bindings(store_id,uid,lineage_key,kind,binding_version) "
                         "SELECT ?,uid,lineage_key,'version',binding_version FROM host_uid_bindings WHERE store_id=?", (new, old))
            crafted_revision(conn, new, old)
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    assert checked.returncode == int(duplicate) and report["pass"] == (not duplicate)
    assert [(r["lost"], r["duplicated"], r["order_kept"], r["unpaired"], r["unverified"], r["unprojected"])
            for r in report["lineages"]] == [(0, int(duplicate), True, 0, 0, 0)]
    assert not report["unexpected_sessions"]
    if not duplicate:
        assert audit["ok"]


def test_preview_then_verbatim_revision(tmp_path, record_property):
    base = tmp_path / "capture"
    saved = base / "claude/projects/synthetic/plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    recovered = "synthetic recovered output"
    saved.write_text(recovered)
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[-1]["message"]["content"] = rows[-1]["message"]["content"][:1]
    rows.append(json.loads(json.dumps(rows[-1])))
    rows[1]["message"]["content"][0]["content"] = (
        f"<persisted-output>\nFull output saved to: {saved}\n\nPreview:\nsynthetic\n</persisted-output>")
    rows[2]["message"]["content"][0]["content"] = recovered
    fixture = tmp_path / "09_verbatim.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows))
    home, env, inputs = capture_session(base, fixture)
    assert read_sql(home, "SELECT content FROM messages WHERE role='tool' ORDER BY store_id") == [(recovered,), (recovered,)]
    assert read_sql(home, "SELECT store_id,related_store_id FROM message_relations WHERE kind='supersedes'") == [(3, 2)]
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    assert audit["ok"]
    #2777: a recovered text equal to a later verbatim revision is reported out of order; a fix must make this pass while the corrupted-pair tests keep failing.
    assert checked.returncode == 1 and not report["pass"]
    assert [(r["lost"], r["duplicated"], r["order_kept"], r["unpaired"], r["unverified"], r["unprojected"])
            for r in report["lineages"]] == [(0, 0, False, 0, 1, 0)]
    assert not report["unexpected_sessions"]


@pytest.mark.parametrize("preview_first", [True, False], ids=["corrupted-pair", "mirror"])
def test_corrupted_preview_verbatim_pair(tmp_path, preview_first, record_property):
    base = tmp_path / "capture"
    saved = base / "claude/projects/synthetic/plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("A")
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[-1]["message"]["content"] = rows[-1]["message"]["content"][:1]
    rows.append(json.loads(json.dumps(rows[-1])))
    rows[1 if preview_first else 2]["message"]["content"][0]["content"] = (
        f"<persisted-output>\nFull output saved to: {saved}\n\nPreview:\nsynthetic\n</persisted-output>")
    rows[2 if preview_first else 1]["message"]["content"][0]["content"] = "A"
    fixture = tmp_path / "09_corrupted_verbatim.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows))
    home, env, inputs = capture_session(base, fixture)
    assert read_sql(home, "SELECT store_id,content FROM messages WHERE role='tool' ORDER BY store_id") == [(2, "A"), (3, "A")]
    assert read_sql(home, "SELECT store_id,related_store_id FROM message_relations WHERE kind='supersedes'") == [(3, 2)]
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("UPDATE messages SET content='garbage' WHERE store_id=?", (3 if preview_first else 2,))
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit, sort_keys=True))
    assert checked.returncode == 1 and not report["pass"]
    assert [(r["lost"], r["duplicated"], r["order_kept"], r["unpaired"], r["unverified"], r["unprojected"])
            for r in report["lineages"]] == [(0, 0, False, 0, 1, 0)]
    assert not report["unexpected_sessions"]
