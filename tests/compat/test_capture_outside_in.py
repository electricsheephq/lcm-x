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
        assert sum(r["unverified" if shape.startswith("persisted") else "unprojected"] for r in report["lineages"]) == (0 if shape.startswith("persisted") else 1)
    if shape == "payload":
        assert sum(r["unverified"] for r in report["lineages"]) == 0
        assert read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call1'")[0][0].startswith("[Externalized tool output:")
    if shape == "persisted-payload":
        assert read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call1'")[0][0].startswith("[Externalized tool output:")
    if shape == "persisted":
        assert read_sql(home, "SELECT content FROM messages WHERE tool_call_id='call1'") == [("synthetic recovered full output",)]
    if shape == "docs":
        paragraph = next(line for line in (ROOT / "docs/claude-code-adapter.md").read_text().splitlines() if "An independent row checker" in line)
        assert all(word in paragraph for word in ("unverified", "unprojected", "unexpected_sessions", "every transcript file of the session"))
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
            for r in report["lineages"]] == [(int(corrupt), int(corrupt), True, 0, 0, 0)]
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
    assert checked.returncode == 0 and report["pass"]
    assert [(r["lost"], r["duplicated"], r["order_kept"], r["unpaired"], r["unverified"], r["unprojected"])
            for r in report["lineages"]] == [(0, 0, True, 0, 0, 0)]
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
            for r in report["lineages"]] == [(1, 1, True, 0, 0, 0)]
    assert not report["unexpected_sessions"]


def write_entries(path, entries):
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def check_only(home, env, inputs):
    args = [arg for session, path in inputs for arg in ("--transcript", f"{session}={path}")]
    done = run(CHECKER, "--home", home, *args, env=env, accepted=(0, 1))
    return done.returncode, json.loads(done.stdout)


@pytest.mark.parametrize("corrupt", [False, True], ids=["healthy", "corrupted"])
def test_verified_preview(tmp_path, corrupt, record_property):
    base = tmp_path / "capture"
    saved = base / "claude/projects/synthetic/plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_bytes("synthetic café\r\nfull output\n".encode())
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[1]["message"]["content"] = rows[1]["message"]["content"][:1]
    rows[1]["message"]["content"][0]["content"] = (
        f"<persisted-output>\nFull output saved to: {saved}\n\nPreview:\nsynthetic\n</persisted-output>")
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_preview.jsonl", rows))
    if corrupt:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content='corrupted' WHERE role='tool'")
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    row = report["lineages"][0]
    assert (code, report["pass"]) == (int(corrupt), not corrupt)
    assert (row["lost"], row["duplicated"], row["unverified"]) == (int(corrupt), int(corrupt), 0)
    saved.unlink()
    code, report = check_only(home, env, inputs)
    row = report["lineages"][0]
    assert code == 0 and row["unverified"] == 1
    assert row["unverified_reasons"] == [{"uid": "r1", "reason": "saved_output_missing"}]


def preview_rows(saved, session="plain", named="marker"):
    """Two fixture rows, the second a tool result whose content is a preview naming `saved` in its marker line, or
    only in the entry's toolUseResult.persistedOutputPath."""
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[1]["message"]["content"] = rows[1]["message"]["content"][:1]
    marker = f"Full output saved to: {saved}\n\n" if named == "marker" else ""
    preview = f"<persisted-output>\n{marker}Preview:\nfull saved\n</persisted-output>"
    rows[1]["message"]["content"][0]["content"] = preview
    if named != "marker":
        rows[1]["toolUseResult"] = {"persistedOutputPath": saved}
    return [dict(row, sessionId=session) for row in rows], preview


def counts(report):
    return {r["lineage"].split(":", 1)[1]: (r["lost"], r["duplicated"], r["unverified"]) for r in report["lineages"]}


def preview_copies(report):
    return {r["lineage"].split(":", 1)[1]: r["preview_copies"] for r in report["lineages"]}


@pytest.mark.parametrize("named", ["marker", "metadata", "marker_and_own_metadata", "two_markers"])
@pytest.mark.parametrize("own_results", ["absent", "present"])
@pytest.mark.parametrize("corrupt", [False, True], ids=["healthy", "corrupted"])
def test_fork_copy_of_parent_preview(tmp_path, named, own_results, corrupt, record_property):
    """A fork's copy of its parent's preview names the parent's saved file, outside the fork's own session folder.
    Capture keeps that preview as the fork's row, so the check compares the row with the preview exactly."""
    base = tmp_path / "capture"
    projects = base / "claude/projects/synthetic"
    saved = projects / "plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("full saved output\ncafé\n")
    if own_results == "present":  # the fork saved an output of its own, so its folder has tool-results/
        (projects / "fork-plain/tool-results").mkdir(parents=True)
    rows, preview = preview_rows(str(saved), named="metadata" if named == "metadata" else "marker")
    fork, _ = preview_rows(str(saved), "fork-plain", "metadata" if named == "metadata" else "marker")
    if named in ("marker_and_own_metadata", "two_markers"):  # a name in the fork's own folder, read second
        own = projects / "fork-plain/tool-results/own.txt"
        own.parent.mkdir(parents=True, exist_ok=True)
        own.write_text("the fork's own output\n")
        if named == "two_markers":
            fork[1]["message"]["content"][0]["content"] = preview.replace(
                "Preview:", f"Full output saved to: {own}\nPreview:")
        else:
            fork[1]["toolUseResult"] = {"persistedOutputPath": str(own)}
    fork_preview = fork[1]["message"]["content"][0]["content"]
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_fork_preview.jsonl", rows + fork))
    stored = dict(read_sql(home, "SELECT session_id, content FROM messages WHERE role='tool'"))
    assert stored == {"plain": "full saved output\ncafé\n", "fork-plain": fork_preview}
    if corrupt:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content=? WHERE role='tool' AND session_id='fork-plain'",
                         (fork_preview.replace("full saved\n", "full saveD\n"),))
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(corrupt), not corrupt)
    assert counts(report) == {"plain": (0, 0, 0), "fork-plain": (int(corrupt), int(corrupt), 0)}
    assert preview_copies(report) == {"plain": 0, "fork-plain": int(not corrupt)}


@pytest.mark.parametrize("case", ["owner_altered", "owner_not_supplied", "other_file_in_parent_folder",
                                  "outside_every_session"])
def test_preview_copy_needs_a_verified_owner(tmp_path, case, record_property):
    """A preview naming a path outside its own folders verifies as a copy only when a row of another lineage
    verified against that same file. Without one (the owner row altered or not supplied, a file in the parent's
    folder that no parent row names, a path outside every session) it takes the saved-file path and stays
    unverified."""
    base = tmp_path / "capture"
    projects = base / "claude/projects/synthetic"
    saved = projects / "plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("full saved output\n")
    named = {"other_file_in_parent_folder": projects / "plain/tool-results/other.txt",
             "outside_every_session": tmp_path / "elsewhere/output.txt"}.get(case, saved)
    named.parent.mkdir(parents=True, exist_ok=True)
    named.write_text("full saved output\n")
    rows, _ = preview_rows(saved)
    fork, fork_preview = preview_rows(named, "fork-plain")
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_owner.jsonl", rows + fork))
    stored = dict(read_sql(home, "SELECT session_id, content FROM messages WHERE role='tool'"))
    assert stored == {"plain": "full saved output\n", "fork-plain": fork_preview}
    if case == "owner_altered":
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content='altered' WHERE role='tool' AND session_id='plain'")
    if case == "owner_not_supplied":
        inputs = [item for item in inputs if item[0] == "fork-plain"]
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    fork_report = next(r for r in report["lineages"] if r["lineage"] == "claude-code:fork-plain")
    assert counts(report)["fork-plain"] == (0, 0, 1) and fork_report["preview_copies"] == 0
    assert fork_report["unverified_reasons"] == [{"uid": "r1", "reason": "saved_output_outside_session"}]
    if case == "owner_altered":
        assert (code, counts(report)["plain"]) == (1, (1, 1, 0))
    elif case == "owner_not_supplied":
        assert code == 1 and [s["session_id"] for s in report["unexpected_sessions"]] == ["plain"]
    else:
        assert (code, counts(report)["plain"], preview_copies(report)["plain"]) == (0, (0, 0, 0), 0)


def test_preview_copy_needs_another_lineage(tmp_path, record_property):
    """A row never owns its own copy. A main transcript directly inside a folder named subagents/ belongs to that
    folder's parent (capture's session_dir()), while saved_preview() reads <session>/ beside it; a row verified there
    names a path outside its own folders, and keeps the verdict main gives it."""
    base = tmp_path / "capture"
    saved = base / "claude/projects/synthetic/plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("full saved output\n")
    rows, _ = preview_rows(saved)
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_self.jsonl", rows))
    assert read_sql(home, "SELECT content FROM messages WHERE role='tool'") == [("full saved output\n",)]
    moved = tmp_path / "moved/subagents"
    (moved / "plain/tool-results").mkdir(parents=True)
    (moved / "plain/tool-results/output.txt").write_text("full saved output\n")
    (moved / "plain.jsonl").write_text(
        dict(inputs)["plain"].read_text().replace(str(saved), str(moved / "plain/tool-results/output.txt")))
    code, report = check_only(home, env, [("plain", moved / "plain.jsonl")])
    record_property("checker", json.dumps(report))
    assert (code, report["pass"], counts(report), preview_copies(report)) == (0, True, {"plain": (0, 0, 0)}, {"plain": 0})


@pytest.mark.parametrize("spelling", ["fork_names_alias", "parent_names_alias", "supplied_through_alias"])
def test_preview_copy_compares_resolved_paths(tmp_path, spelling, record_property):
    """Paths are compared resolved on both sides. A copy whose marker names its parent's file through a linked
    folder, or a parent whose marker does, is still that file's copy; and transcripts supplied through a linked folder
    keep a row's own saved output inside its own folder, so it is checked against the file, never as a copy."""
    base = tmp_path / "capture"
    projects = base / "claude/projects/synthetic"
    saved = projects / "plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("full saved output\n")
    alias = tmp_path / "alias"
    alias.symlink_to(projects, target_is_directory=True)
    aliased = alias / "plain/tool-results/output.txt"
    rows, _ = preview_rows(aliased if spelling == "parent_names_alias" else saved)
    fork, fork_preview = preview_rows(aliased if spelling == "fork_names_alias" else saved, "fork-plain")
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_resolved.jsonl", rows + fork))
    stored = dict(read_sql(home, "SELECT session_id, content FROM messages WHERE role='tool'"))
    assert stored == {"plain": "full saved output\n", "fork-plain": fork_preview}
    if spelling == "supplied_through_alias":
        inputs = [(session, alias / path.relative_to(projects)) for session, path in inputs]
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"], counts(report)) == (0, True, {"plain": (0, 0, 0), "fork-plain": (0, 0, 0)})
    assert preview_copies(report) == {"plain": 0, "fork-plain": 1}


@pytest.mark.parametrize("shape", ["file_linked_elsewhere", "quoted_marker"])
@pytest.mark.parametrize("corrupt", [False, True], ids=["healthy", "corrupted"])
def test_preview_named_outside_own_folder(tmp_path, shape, corrupt, record_property):
    """Other names that resolve outside the row's own tool-results/, with no owner row: a saved file linked to a file
    elsewhere, and a quoted marker (capture reads the path as written, quotes included). Capture keeps the preview;
    the checker does not verify it. The linked file stays unverified; the quoted marker is read with its quotes
    stripped, so the stored preview fails against the file's text (as on main)."""
    base = tmp_path / "capture"
    own = base / "claude/projects/synthetic/plain"
    (own / "tool-results").mkdir(parents=True)
    saved = own / "tool-results/output.txt"
    if shape == "file_linked_elsewhere":
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere/real.txt").write_text("full saved output\n")
        saved.symlink_to(tmp_path / "elsewhere/real.txt")
        rows, preview = preview_rows(saved)
    else:
        saved.write_text("full saved output\n")
        rows, preview = preview_rows(f'"{saved}"')
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_elsewhere.jsonl", rows))
    assert read_sql(home, "SELECT content FROM messages WHERE role='tool'") == [(preview,)]
    if corrupt:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content='corrupted' WHERE role='tool'")
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    if shape == "file_linked_elsewhere":  # accepted unverified, whatever its content
        assert (code, report["pass"], counts(report)) == (0, True, {"plain": (0, 0, 1)})
        assert report["lineages"][0]["unverified_reasons"] == [{"uid": "r1", "reason": "saved_output_outside_session"}]
    else:
        assert (code, report["pass"], counts(report)) == (1, False, {"plain": (1, 1, 0)})
    assert preview_copies(report) == {"plain": 0}


@pytest.mark.parametrize("shape,reason", [
    ("no_results_folder", "saved_output_missing"), ("results_is_a_file", "saved_output_missing"),
    ("linked_session_folder", "saved_output_outside_session"), ("linked_results_folder", "saved_output_outside_session"),
    ("file_linked_inside", "saved_output_outside_session")])
def test_preview_inside_own_folder_capture_kept_stays_unverified(tmp_path, shape, reason, record_property):
    """Capture also keeps the preview when a path inside the row's own folder cannot be read safely: tool-results
    missing or not a folder, the session folder or tool-results symlinked, or the saved file a symlink. The check
    covers only names outside that folder, so these stay unverified with their reason, as before."""
    base = tmp_path / "capture"
    projects = base / "claude/projects/synthetic"
    own, real = projects / "plain", tmp_path / "elsewhere"
    if shape == "no_results_folder":
        own.mkdir(parents=True)
    elif shape == "results_is_a_file":
        own.mkdir(parents=True)
        (own / "tool-results").write_text("not a folder\n")
    elif shape == "linked_session_folder":
        (real / "tool-results").mkdir(parents=True)
        (real / "tool-results/output.txt").write_text("full saved output\n")
        projects.mkdir(parents=True)
        own.symlink_to(real, target_is_directory=True)
    elif shape == "linked_results_folder":
        real.mkdir(parents=True)
        (real / "output.txt").write_text("full saved output\n")
        own.mkdir(parents=True)
        (own / "tool-results").symlink_to(real, target_is_directory=True)
    else:  # a link inside the session's own tool-results to a real file beside it
        (own / "tool-results").mkdir(parents=True)
        (own / "tool-results/real.txt").write_text("full saved output\n")
        (own / "tool-results/output.txt").symlink_to(own / "tool-results/real.txt")
    rows, preview = preview_rows(own / "tool-results/output.txt")
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_kept.jsonl", rows))
    assert read_sql(home, "SELECT content FROM messages WHERE role='tool'") == [(preview,)]
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"], counts(report)) == (0, True, {"plain": (0, 0, 1)})
    assert report["lineages"][0]["unverified_reasons"] == [{"uid": "r1", "reason": reason}]


@pytest.mark.parametrize("named", ["unknown_user", "not_text", "blank_marker", "nul_byte"])
def test_preview_path_error_stays_unverified(tmp_path, named, record_property):
    """A path capture cannot use (an unknown ~user, a persistedOutputPath that is not text, or a marker that names
    nothing) keeps the preview, but capture reports it as missing or unnamed, not outside the session: it stays
    unverified."""
    base = tmp_path / "capture"
    (base / "claude/projects/synthetic/plain/tool-results").mkdir(parents=True)
    rows, preview = (preview_rows("~lcmx-no-such-user/output.txt") if named == "unknown_user"
                     else preview_rows("/nul\x00/output.txt") if named == "nul_byte"
                     else preview_rows(7, named="metadata"))
    if named == "blank_marker":  # the marker's path is only spaces, with nothing after it
        preview = "<persisted-output>\nFull output saved to:   "
        rows[1]["message"]["content"][0]["content"] = preview
        del rows[1]["toolUseResult"]
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_user.jsonl", rows))
    assert read_sql(home, "SELECT content FROM messages WHERE role='tool'") == [(preview,)]
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"], counts(report)) == (0, True, {"plain": (0, 0, 1)})
    reason = "saved_output_unnamed" if named == "blank_marker" else "saved_output_unreadable"
    assert report["lineages"][0]["unverified_reasons"] == [{"uid": "r1", "reason": reason}]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a folder without permissions")
def test_preview_unsearchable_session_folder(tmp_path, record_property):
    """A session folder capture cannot search keeps the preview, but the named path lies inside the row's own
    folder, so the row stays unverified with its reason, as before."""
    base = tmp_path / "capture"
    own = base / "claude/projects/synthetic/plain"
    (own / "tool-results").mkdir(parents=True)
    own.chmod(0)
    try:
        rows, preview = preview_rows(own / "tool-results/output.txt")
        home, env, inputs = capture_session(base, write_entries(tmp_path / "01_closed.jsonl", rows))
        assert read_sql(home, "SELECT content FROM messages WHERE role='tool'") == [(preview,)]
        code, report = check_only(home, env, inputs)
        record_property("checker", json.dumps(report))
        assert (code, report["pass"], counts(report)) == (0, True, {"plain": (0, 0, 1)})
        reasons = report["lineages"][0]["unverified_reasons"]  # the folder check raises on Python 3.11-3.13
        assert reasons in ([{"uid": "r1", "reason": "saved_output_unreadable"}],
                           [{"uid": "r1", "reason": "saved_output_missing"}])
    finally:
        own.chmod(0o755)


@pytest.mark.parametrize("shape,copy_name", [
    ("subagent", None), ("workflow_subagent", None), ("deepest_subagent", None), ("dotted_session", None),
    ("set_aside", "plain.jsonl.superseded-1"), ("set_aside", "plain.jsonl.orphaned-1"),
    ("set_aside", "plain.superseded-1.jsonl"), ("dotted_set_aside", "plain.v1.jsonl.superseded-1"),
    ("dotted_set_aside", "plain.v1.orphaned-1.jsonl")])
@pytest.mark.parametrize("stored_preview", [False, True], ids=["healthy", "preview_stored"])
def test_preview_own_folder(tmp_path, shape, copy_name, stored_preview, record_property):
    """The row's own session folder is found as capture's session_dir() finds it for the lineage's live transcript:
    <session>/ beside <session>.jsonl, even when the session id holds dots; a subagent's saved output is in its
    session's tool-results/, up to four folders below subagents/; and a set-aside copy is drained under its
    replacement's lineage, so it belongs to the replacement's folder. Capture recovers each saved output, so the
    check verifies the row against the file, and a row holding only the preview fails."""
    base = tmp_path / "capture"
    projects = base / "claude/projects/synthetic"
    session = "plain.v1" if shape.startswith("dotted") else "plain"
    saved = projects / session / "tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("full saved output\n")
    rows, preview = preview_rows(saved, session)
    head = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][:1]
    agent = [dict(row, uuid="s" + row["uuid"], agentId="x") for row in rows]
    nested = {"workflow_subagent": "workflows/run-1", "deepest_subagent": "a/b/c/d"}.get(shape)
    if shape == "subagent":
        rows = head + agent
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_own.jsonl", head if nested else rows))
    if nested:  # a workflow agent sits up to four folders below its session's subagents/ (#2562)
        path = projects / "plain/subagents" / nested / "agent-x.jsonl"
        path.parent.mkdir(parents=True)
        write_entries(path, agent)
        run(CAPTURE, "worker", "--home", home, env=env)
        inputs = [*inputs, ("plain/agent:x", path)]
    stored = [content for (content,) in read_sql(home, "SELECT content FROM messages WHERE role='tool'")]
    assert stored == ["full saved output\n"] and preview not in stored  # recovered from the row's own session folder
    if copy_name:  # the checker reads the row from the set-aside copy first
        live = dict(inputs)[session]
        copy = live.with_name(copy_name)
        copy.write_bytes(live.read_bytes())
        inputs = [(session, copy), (session, live)]
    if stored_preview:  # what capture would have stored had it looked in the wrong folder
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content=? WHERE role='tool'", (preview,))
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(stored_preview), not stored_preview)
    lineage = "plain/agent:x" if shape.endswith("subagent") else session
    assert counts(report)[lineage] == (int(stored_preview), int(stored_preview), 0)
    assert all(row == (0, 0, 0) for key, row in counts(report).items() if key != lineage)


def test_own_session_folders_match_capture(tmp_path):
    """The checker's folder rule is capture's session_dir() for live transcript names, read from both modules."""
    import importlib.util
    specs = {name: importlib.util.spec_from_file_location(name, path)
             for name, path in (("checker_under_test", CHECKER), ("capture_rule", ROOT / "claude_code_capture.py"))}
    modules = {}
    for name, spec in specs.items():
        modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modules[name])
    checker, capture = modules["checker_under_test"], modules["capture_rule"]
    session = tmp_path / "projects/p/s"
    names = [tmp_path / "projects/p/s.jsonl", tmp_path / "projects/p/s.v1.jsonl", tmp_path / "projects/p/.hidden.jsonl",
             tmp_path / "projects/p/agent-a.b.jsonl", tmp_path / "projects/p/noext",
             tmp_path / "subagents/p/s.jsonl"]  # a main transcript never searches for subagents/
    names += [session / "subagents" / Path(*["d"] * depth) / "agent-x.jsonl" for depth in range(6)]
    for path in names:
        assert checker.session_folder(path) == capture.session_dir(path), path
        assert checker.own_session_folders(path) == [capture.session_dir(path)], path
    assert checker.session_folder(names[6]) == session and checker.session_folder(names[11]) != session
    for copy, live in (("s.v1.jsonl.superseded-1", "s.v1.jsonl"), ("s.v1.orphaned-1.jsonl", "s.v1.jsonl")):
        copy = tmp_path / "projects/p" / copy
        assert checker.own_session_folders(copy) == [capture.session_dir(copy), capture.session_dir(copy.with_name(live))]
    assert checker.outside_named_path("Full output saved to: /x/tool-results/y", {}, []) is None


@pytest.mark.parametrize("shape", ["base64", "image", "tool-media", "wrapped", "multiple"])
@pytest.mark.parametrize("corrupt", [False, True], ids=["healthy", "corrupted"])
def test_inline_payload(tmp_path, shape, corrupt, record_property):
    import base64
    payload = base64.b64encode(bytes(range(256)) * 16).decode()
    text = "café prefix\n" + payload + "\nsuffix"
    if shape == "wrapped":
        text = "café prefix\n" + "\n".join(payload[i:i + 76] for i in range(0, len(payload), 76)) + "\nsuffix"
    if shape == "multiple":
        text += " data:image/png;base64," + payload
    rows = [dict(type="user", uuid="media", sessionId="media", message=dict(content=text))]
    if shape == "image":
        rows[0]["message"]["content"] = [dict(type="text", text="café"), dict(
            type="image", source=dict(type="base64", media_type="image/png", data=payload))]
    if shape == "tool-media":
        rows = [dict(type="assistant", uuid="call", sessionId="media", message=dict(content=[
                    dict(type="tool_use", id="c", name="synthetic", input={})])),
                dict(type="user", uuid="result", sessionId="media", message=dict(content=[
                    dict(type="tool_result", tool_use_id="c", content=text)]))]
    home, env, inputs = capture_session(tmp_path / "capture", write_entries(tmp_path / "01_media.jsonl", rows))
    stored = read_sql(home, "SELECT content FROM messages ORDER BY store_id")[-1][0]
    assert "[Externalized LCM ingest payload:" in stored
    if corrupt:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content=? WHERE store_id=(SELECT max(store_id) FROM messages)",
                         (stored.replace("chars=", "chars=9", 1),))
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(corrupt), not corrupt)
    assert [(r["lost"], r["duplicated"], r["unverified"]) for r in report["lineages"]] == [
        (int(corrupt), int(corrupt), 0)]


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("corrupt", [False, True], ids=["healthy", "corrupted"])
def test_non_tool_payload(tmp_path, role, corrupt, record_property):
    rows = [dict(type=role, uuid="large", sessionId="large", message=dict(content="".join(f"Synthetic observation {i}: café sample number {i*i}.\n" for i in range(2500))))]
    home, env, inputs = capture_session(tmp_path / "capture", write_entries(tmp_path / "01_large.jsonl", rows))
    stored = read_sql(home, "SELECT content FROM messages")[0][0]
    assert stored.startswith("[Externalized payload:")
    if corrupt:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content=?", (stored.replace("bytes=", "bytes=9", 1),))
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(corrupt), not corrupt)
    assert [(r["lost"], r["duplicated"]) for r in report["lineages"]] == [(int(corrupt), int(corrupt))]


@pytest.mark.parametrize("corrupt", [False, True], ids=["healthy", "corrupted"])
def test_reference_then_literal_revision(tmp_path, corrupt, record_property):
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[1]["message"]["content"] = rows[1]["message"]["content"][:1]
    rows[1]["message"]["content"][0]["content"] = "synthetic " * 7500
    home, env, inputs = capture_session(tmp_path / "capture", write_entries(tmp_path / "01_reference.jsonl", rows))
    marker = read_sql(home, "SELECT content FROM messages WHERE role='tool'")[0][0]
    assert marker.startswith("[Externalized tool output:")
    rows[1]["message"]["content"][0]["content"] = marker
    with inputs[0][1].open("a") as handle:
        handle.write(json.dumps(rows[1]) + "\n")
    hook(home, env, "plain", inputs[0][1])
    run(CAPTURE, "worker", "--home", home, env=env)
    assert read_sql(home, "SELECT content FROM messages WHERE role='tool' ORDER BY store_id") == [(marker,), (marker,)]
    if corrupt:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content='corrupted' WHERE store_id=2")
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(corrupt), not corrupt)


@pytest.mark.parametrize("dangling", [False, True], ids=["healthy", "corrupted"])
def test_dangling_binding(clean, tmp_path, dangling, record_property):
    source, env, inputs = clean
    home = tmp_path / "store"
    home.mkdir()
    with closing(sqlite3.connect(source / "lcm.db")) as src, closing(sqlite3.connect(home / "lcm.db")) as dst:
        src.backup(dst)
        if dangling:
            dst.execute("INSERT INTO host_uid_bindings(store_id,uid,lineage_key,kind,binding_version) "
                        "VALUES(999999,'dangling','claude-code:plain','canonical',1)")
            dst.commit()
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(dangling), not dangling)
    assert report["dangling_bindings"] == ([dict(store_id=999999, uid="dangling", lineage_key="claude-code:plain",
                                                 kind="canonical")] if dangling else [])


@pytest.mark.parametrize("corrupt", [False, True], ids=["healthy", "corrupted"])
def test_thousand_references(tmp_path, corrupt, record_property, monkeypatch):
    import hashlib
    import importlib.util
    home = tmp_path / "store"
    home.mkdir()
    entries, stored = [], []
    for i in range(1000):
        text = f"synthetic {i} café " * 100
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        marker = (f"[Externalized tool output: tool=synthetic; tool_call_id=c{i}; chars={len(text)}; "
                  f"bytes={len(text.encode())}; read it with lcm_expand; ref=20000101_000000_c{i}_{digest}_abcd.json]")
        entries.extend([dict(type="assistant", uuid=f"a{i}", sessionId="batch", message=dict(content=[
                           dict(type="tool_use", id=f"c{i}", name="synthetic", input={})])),
                        dict(type="user", uuid=f"t{i}", sessionId="batch", message=dict(content=[
                           dict(type="tool_result", tool_use_id=f"c{i}", content=text)]))])
        calls = json.dumps([dict(id=f"c{i}", type="function", function=dict(name="synthetic", arguments="{}"))])
        stored.extend([(2*i+1, "batch", "assistant", "", None, calls, f"a{i}"),
                       (2*i+2, "batch", "tool", marker, f"c{i}", None, f"t{i}")])
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("CREATE TABLE messages(store_id INTEGER, session_id TEXT, role TEXT, content TEXT, "
                     "tool_call_id TEXT, tool_calls TEXT)")
        conn.execute("CREATE TABLE host_uid_bindings(store_id INTEGER, uid TEXT, lineage_key TEXT, kind TEXT)")
        conn.executemany("INSERT INTO messages VALUES(?,?,?,?,?,?)", [row[:-1] for row in stored])
        conn.executemany("INSERT INTO host_uid_bindings VALUES(?,?,'claude-code:batch','canonical')",
                         [(row[0], row[-1]) for row in stored])
        if corrupt:
            conn.execute("UPDATE messages SET content=replace(content,'bytes=','bytes=9') WHERE store_id=2000")
    inputs = [("batch", write_entries(tmp_path / "batch.jsonl", entries))]
    code, report = check_only(home, {"PATH": os.defpath}, inputs)
    record_property("checker", json.dumps(report))
    record_property("home", str(home))
    assert (code, report["pass"]) == (int(corrupt), not corrupt)
    assert [(r["lost"], r["duplicated"]) for r in report["lineages"]] == [(int(corrupt), int(corrupt))]
    spec = importlib.util.spec_from_file_location("outside_in_parse_probe", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original_loads, parsed = json.loads, []
    def count_loads(*args, **kwargs):
        parsed.append(None)
        return original_loads(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(checker.json, "loads", count_loads)
        measured = checker.check(home, inputs)
    record_property("json_loads", len(parsed))
    assert measured["pass"] == (not corrupt)
    assert len(parsed) <= len(entries) * 8
    assert report["elapsed_ms"] > 0


@pytest.mark.parametrize("path_key", [True, False], ids=["path-key", "id-key"])
def test_reused_workflow_agent_id(tmp_path, path_key, record_property):
    entries = [dict(type="user", uuid="parent", sessionId="parent", message=dict(content="parent")),
               dict(type="user", uuid="first", sessionId="parent", agentId="child", message=dict(content="first"))]
    home, env, inputs = capture_session(tmp_path / "capture", write_entries(tmp_path / "01_workflow.jsonl", entries))
    later = inputs[1][1].parent / "workflows/run2/agent-child.jsonl"
    later.parent.mkdir(parents=True)
    write_entries(later, [dict(type="user", uuid="later", sessionId="parent", message=dict(content="later"))])
    run(CAPTURE, "worker", "--home", home, env=env)
    key = "parent/agent:/workflows/run2/agent-child.jsonl"
    inputs.append((key if path_key else "parent/agent:child", later))
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(not path_key), path_key)
    assert report["unexpected_sessions"] == ([] if path_key else [dict(session_id=key, rows=1)])


@pytest.mark.parametrize("reverse", [False, True], ids=["healthy", "reversed"])
def test_revision_slot_order(tmp_path, reverse, record_property):
    entries = [dict(type="user", uuid="u", sessionId="order", message=dict(content=text))
               for text in ("first revision", "second revision")]
    home, env, inputs = capture_session(tmp_path / "capture", write_entries(tmp_path / "09_order.jsonl", entries))
    if reverse:
        with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
            conn.execute("UPDATE messages SET content=CASE store_id WHEN 1 THEN 'second revision' ELSE 'first revision' END")
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(reverse), not reverse)
    assert [(r["lost"], r["duplicated"], r["order_kept"]) for r in report["lineages"]] == [(0, 0, not reverse)]


@pytest.mark.parametrize("quote", ["", "'", '"'], ids=["unquoted", "single-quoted", "double-quoted"])
def test_preview_supplied_paths(tmp_path, quote, record_property):
    base = tmp_path / "capture"
    saved = base / "claude/projects/synthetic/plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("full saved output\ncafé\n")
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[1]["message"]["content"] = rows[1]["message"]["content"][:1]
    rows[1]["message"]["content"][0]["content"] = (
        f"<persisted-output>\nFull output saved to: {quote}{saved}{quote}\n\nPreview:\nx\n</persisted-output>")
    historical = write_entries(tmp_path / "historical.jsonl", rows)
    home, env, inputs = capture_session(base, historical)
    orders = (inputs, [("plain", historical), *inputs], [*inputs, ("plain", historical)])
    for order in orders:  # a copy supplied from another folder never moves the row's own folder
        code, report = check_only(home, env, order)
        record_property("checker", json.dumps(report))
        if quote:  # capture keeps a quoted preview; the checker strips the quotes and reads the file (as on main)
            assert (code, report["pass"], counts(report)) == (1, False, {"plain": (1, 1, 0)})
        else:
            assert (code, report["pass"], counts(report)) == (0, True, {"plain": (0, 0, 0)})
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("UPDATE messages SET content='' WHERE role='tool'")
    for order in orders:
        code, report = check_only(home, env, order)
        record_property("checker", json.dumps(report))
        assert (code, report["pass"]) == (1, False)
        assert [(r["lost"], r["duplicated"], r["unverified"]) for r in report["lineages"]] == [(1, 1, 0)]


@pytest.mark.parametrize("quoted", [
    "[Externalized LCM ingest payload: kind=x; field=content; chars=1; bytes=1; "
    "ref=20000101_000000_a_b_0123456789ab_ff.json]",
    "[Externalized LCM ingest payload: note]",
], ids=["wellformed", "short"])
def test_quoted_ingest_marker(tmp_path, quoted, record_property):
    import base64
    payload = base64.b64encode(bytes(range(256)) * 16).decode()
    text = f"docs say {quoted} ok\n{payload}\nsuffix"
    rows = [dict(type="user", uuid="q", sessionId="quote", message=dict(content=text))]
    home, env, inputs = capture_session(tmp_path / "capture", write_entries(tmp_path / "01_quote.jsonl", rows))
    stored = read_sql(home, "SELECT content FROM messages")[0][0]
    assert quoted in stored and stored.count("[Externalized LCM ingest payload:") == 2
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    assert audit["ok"] and checked.returncode == 0 and report["pass"]
    # Preserve the quoted marker, but corrupt the real replacement's fingerprint.
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("UPDATE messages SET content=?", (stored[:len(quoted) + 9] +
                     stored[len(quoted) + 9:].replace("bytes=", "bytes=9", 1),))
    code, report = check_only(home, env, inputs)
    record_property("corrupted", json.dumps(report))
    assert code == 1 and not report["pass"]
    assert [(r["lost"], r["duplicated"]) for r in report["lineages"]] == [(1, 1)]


@pytest.mark.parametrize("guard", ["cross_occurrence", "corrupt_prefix", "dropped_suffix"])
def test_ingest_position_guards(tmp_path, guard, record_property):
    import hashlib
    def marker(text):
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        return (f"[Externalized LCM ingest payload: kind=media_payload; field=content; chars={len(text)}; "
                f"bytes={len(text.encode())}; ref=20000101_000000_media_payload_content_{digest}_abcd.json]")
    def image(label, data):
        return [dict(type="text", text=label), dict(type="image", source=dict(
            type="base64", media_type="image/png", data=data))]
    a, b = "AAAA" * 500, "BBBB" * 500
    if guard == "cross_occurrence":
        sources = [image("a", a), image("b", b)]
        stored = [json.dumps(image(label, marker(a)), ensure_ascii=False, sort_keys=True) for label in ("a", "b")]
    else:
        sources = ["prefix\n" + b + "\nsuffix"]
        stored = [("PREFIX\n" if guard == "corrupt_prefix" else "prefix\n") + marker(b) +
                  ("\nsuffix" if guard == "corrupt_prefix" else "")]
    entries = [dict(type="user", uuid=f"u{i}", sessionId="s", message=dict(content=source))
               for i, source in enumerate(sources)]
    home = tmp_path / "store"
    home.mkdir()
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("CREATE TABLE messages(store_id INTEGER, session_id TEXT, role TEXT, content TEXT, "
                     "tool_call_id TEXT, tool_calls TEXT)")
        conn.execute("CREATE TABLE host_uid_bindings(store_id INTEGER, uid TEXT, lineage_key TEXT, kind TEXT)")
        for i, text in enumerate(stored):
            conn.execute("INSERT INTO messages VALUES(?,'s','user',?,NULL,NULL)", (i + 1, text))
            conn.execute("INSERT INTO host_uid_bindings VALUES(?,?,'claude-code:s','canonical')", (i + 1, f"u{i}"))
    code, report = check_only(home, {"PATH": os.defpath}, [("s", write_entries(tmp_path / "s.jsonl", entries))])
    record_property("checker", json.dumps(report))
    assert code == 1 and not report["pass"]
    assert [(r["lost"], r["duplicated"]) for r in report["lineages"]] == [(1, 1)]


def test_preview_size_cap(tmp_path, monkeypatch, record_property):
    import importlib.util
    base = tmp_path / "capture"
    saved = base / "claude/projects/synthetic/plain/tool-results/output.txt"
    saved.parent.mkdir(parents=True)
    with saved.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)  # Sparse scratch file, never a repository fixture.
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[1]["message"]["content"] = rows[1]["message"]["content"][:1]
    preview = f"<persisted-output>\nFull output saved to: {saved}\n\nPreview:\nx\n</persisted-output>"
    rows[1]["message"]["content"][0]["content"] = preview
    home, env, inputs = capture_session(base, write_entries(tmp_path / "01_huge.jsonl", rows))
    assert read_sql(home, "SELECT content FROM messages WHERE role='tool'") == [(preview,)]
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert code == 0 and report["pass"]
    row = report["lineages"][0]
    assert row["unverified"] == 1
    assert row["unverified_reasons"] == [{"uid": "r1", "reason": "saved_output_too_large"}]
    spec = importlib.util.spec_from_file_location("outside_in_size_probe", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path != saved, "oversized saved output must not be opened"
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        assert checker.check(home, inputs)["lineages"][0]["unverified_reasons"] == row["unverified_reasons"]


@pytest.mark.parametrize("metadata", ["Error: x", ["Error: x"]], ids=["str", "list"])
def test_preview_non_mapping_metadata(tmp_path, metadata, record_property):
    rows = [json.loads(line) for line in FIXTURES[0].read_text().splitlines()][1:3]
    rows[1]["message"]["content"] = rows[1]["message"]["content"][:1]
    rows[1]["message"]["content"][0]["content"] = "<persisted-output>\nPreview:\nx\n</persisted-output>"
    rows[1]["toolUseResult"] = metadata
    home, env, inputs = capture_session(tmp_path / "capture", write_entries(tmp_path / "01_metadata.jsonl", rows))
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert code == 0 and report["pass"]
    assert report["lineages"][0]["unverified_reasons"] == [{"uid": "r1", "reason": "saved_output_unnamed"}]


def test_checker_release_instructions():
    for name in ("docs/claude-code-adapter.md", "docs/process/claude-code-unit-release.md",
                 "bench/specs/RELEASE-READINESS-V1.md"):
        text = (ROOT / name).read_text()
        assert "oldest first, historical versions before the live file" in text
        assert "with any unverified row cannot pass" in text
        assert "and justified" not in text
    for name in ("scripts/capture_outside_in.py", "docs/claude-code-adapter.md"):
        assert "ingest-payload replacements inside tool-call arguments" in (ROOT / name).read_text()


# The user's own text may quote the marker prefix (code that handles markers does). Each literal sits
# before the escape-letter shape, so the stored arguments are not JSON and take the text-only path.
LITERAL_MARKERS = {
    "literal-marker": 'prefix [Externalized LCM ingest payload: field="quoted"] café\n',
    "code-marker": 'x = r"\\[Externalized LCM ingest payload: assistant output quarantined; "\ny = [1]\n',
}


def capture_tool_call_payload(tmp_path, shape):
    import base64
    payload = base64.b64encode(bytes(range(256)) * 18).decode()[:6000]
    uri = "data:image/png;base64," + payload
    text = f'<html>café<img src="{uri}"></html>'
    if shape == "bare":
        text = "prefix café\n" + payload[:5000] + "\nsuffix"
    elif shape == "raw-bare":
        text = "prefix " + payload[:5000]
    elif shape == "escape-letter":
        # The raw JSON run includes the 'n' from \\n: 1 + 4999 is accepted mod 4.
        text = "prefix café\n" + payload[:4999]
    elif shape in LITERAL_MARKERS:
        text = LITERAL_MARKERS[shape] + payload[:4999]
    elif shape == "wrapped":
        text = "prefix café\n" + "\n".join(payload[i:i + 76] for i in range(0, 6000, 76)) + "\nsuffix"
    elif shape == "two-payloads":
        text += "\nsecond: " + payload[:5000] + "\nend"
    arguments = dict(file_path="/synthetic/a.html", content=text)
    if shape == "nested":
        arguments = dict(edits=[dict(new=text)])
    calls = [dict(type="tool_use", id="write", name="Write", input=arguments)]
    if shape == "two-calls":
        calls.insert(0, dict(type="tool_use", id="read", name="Read", input=dict(file_path="/synthetic/a.html")))
    entries = [dict(type="assistant", uuid="a", sessionId="tool-payload", message=dict(content=calls)),
               dict(type="user", uuid="r", sessionId="tool-payload", message=dict(content=[
                   dict(type="tool_result", tool_use_id=call["id"], content="ok") for call in calls]))]
    return capture_session(tmp_path / "capture", write_entries(tmp_path / "01_calls.jsonl", entries))


@pytest.mark.parametrize("shape", ["uri", "bare", "wrapped", "nested", "two-payloads", "two-calls",
                                  "raw-bare", "escape-letter", *LITERAL_MARKERS])
def test_tool_call_payload(tmp_path, shape, record_property):
    home, env, inputs = capture_tool_call_payload(tmp_path, shape)
    stored = read_sql(home, "SELECT tool_calls FROM messages WHERE role='assistant'")[0][0]
    twice = shape == "two-payloads" or shape in LITERAL_MARKERS
    assert stored.count("[Externalized LCM ingest payload:") == (2 if twice else 1)
    if shape == "two-calls":
        assert "[Externalized" not in json.loads(stored)[0]["function"]["arguments"]
    arguments = json.loads(stored)[-1]["function"]["arguments"]
    if shape == "raw-bare":
        assert json.loads(arguments)["content"].startswith("prefix [Externalized LCM ingest payload:")
        assert arguments.count("[Externalized LCM ingest payload:") == 1
        assert arguments.startswith('{"file_path": "/synthetic/a.html", "content": ')
    elif shape == "escape-letter" or shape in LITERAL_MARKERS:
        assert "\\[Externalized LCM ingest payload:" in arguments
        assert "chars=5000; bytes=5000;" in arguments
        with pytest.raises(json.JSONDecodeError):
            json.loads(arguments)
        if shape in LITERAL_MARKERS:
            # The literal is kept as the user's text, and the product marker follows the escape's backslash.
            literal = json.dumps(LITERAL_MARKERS[shape][:-1], ensure_ascii=False)[1:-1]
            assert literal + "\\[Externalized LCM ingest payload: " in arguments
        record_property("base64_run_chars", 4999)
        record_property("stored_arguments", arguments)
    checked, report, audit = verdict(home, env, inputs)
    record_property("checker", checked.stdout)
    record_property("audit", json.dumps(audit))
    assert [(r["lost"], r["duplicated"]) for r in report["lineages"]] == [(0, 0)]
    assert checked.returncode == 0 and report["pass"] and audit["ok"] and audit["verified"]


@pytest.mark.parametrize("corruption", ["sha12", "chars", "extra-key", "duplicate-key", "name", "swap",
                                       "surrounding", "escape-sha12"])
def test_tool_call_payload_corrupted(tmp_path, corruption, record_property):
    import re
    shape = "escape-letter" if corruption == "escape-sha12" else "two-calls" if corruption == "swap" else "uri"
    home, env, inputs = capture_tool_call_payload(tmp_path, shape)
    calls = json.loads(read_sql(home, "SELECT tool_calls FROM messages WHERE role='assistant'")[0][0])
    function = calls[-1]["function"]
    arguments = function["arguments"]
    if corruption in {"sha12", "escape-sha12"}:
        arguments = re.sub(r"_([0-9a-f]{12})_", lambda m: "_" + ("1" if m[1][0] == "0" else "0") + m[1][1:] + "_",
                           arguments, count=1)
    elif corruption == "chars":
        arguments = arguments.replace("chars=", "chars=9", 1)
    elif corruption == "extra-key":
        arguments = arguments[:-1] + ', "extra": 1}'
    elif corruption == "duplicate-key":
        # Last-wins parsing would hide the injected value and accept this row.
        arguments = '{"content": "lost original", ' + arguments[1:]
    elif corruption == "name":
        function["name"] = "Wrong"
    elif corruption == "swap":
        calls.reverse()
    else:
        arguments = arguments.replace("<html>", "<htmL>", 1)
    function["arguments"] = arguments
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("UPDATE messages SET tool_calls=? WHERE role='assistant'", (json.dumps(calls),))
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert code == 1 and not report["pass"]
    assert [(r["lost"], r["duplicated"]) for r in report["lineages"]] == [(1, 1)]


def test_tool_call_formatting_without_placeholder(tmp_path, record_property):
    entries = [dict(type="assistant", uuid="a", sessionId="format", message=dict(content=[
                   dict(type="tool_use", id="c", name="Write", input=dict(file_path="/synthetic/a", content="plain"))])),
               dict(type="user", uuid="r", sessionId="format", message=dict(content=[
                   dict(type="tool_result", tool_use_id="c", content="ok")]))]
    home, env, inputs = capture_session(tmp_path / "capture", write_entries(tmp_path / "01_format.jsonl", entries))
    original = read_sql(home, "SELECT tool_calls FROM messages WHERE role='assistant'")[0][0]
    assert "[Externalized" not in original
    calls = json.loads(original)
    function = calls[0]["function"]
    function["arguments"] = json.dumps(json.loads(function["arguments"]), separators=(",", ":"))
    assert json.dumps(calls) != original
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("UPDATE messages SET tool_calls=? WHERE role='assistant'", (json.dumps(calls),))
    code, report = check_only(home, env, inputs)
    record_property("checker", json.dumps(report))
    assert code == 1 and not report["pass"]
    assert [(r["lost"], r["duplicated"]) for r in report["lineages"]] == [(1, 1)]


@pytest.mark.parametrize("stored, source", [
    ('{"v":true}', '{"v":1}'), ('{"v":1}', '{"v":1.0}'),
    pytest.param('{"v":-0.0}', '{"v":0.0}', id="zero-sign"),
    ('{"v":null}', '{"v":false}'), ('{"v":2}', '{"v":1}'),
    ('{"v":"1"}', '{"v":1}'), ('{"v":[]}', '{"v":{}}'),
    ('{"v":[1]}', '{"v":[1,2]}'), ('{"v":1}', '{"other":1}'),
    ('{"nested":{"v":0,"v":1}}', '{"nested":{"v":1}}'),
    ('invalid', '{}'), ('{}', 'invalid'),
])
def test_tool_call_argument_mismatches(stored, source):
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_arguments", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    assert not checker.tool_calls_match(calls(stored), calls(source), defaultdict(set))


@pytest.mark.parametrize("field, stored, source", [
    ("id", True, 1), ("name", 1.0, 1), ("type", 1, True),
], ids=["id", "name", "type"])
def test_tool_call_identity_types(field, stored, source):
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_identity_types", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    left = dict(id="c", type="function", function=dict(name="Write", arguments="{}"))
    right = json.loads(json.dumps(left))
    if field == "name":
        left["function"][field], right["function"][field] = stored, source
    else:
        left[field], right[field] = stored, source
    assert not checker.tool_calls_match([left], [right], defaultdict(set))


def test_tool_call_placeholder_at_leaf_start():
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_leaf_start", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    source = json.dumps(dict(content="payload café tail"))
    stored = json.dumps(dict(content=reference_for("payload café") + " tail"), separators=(",", ":"))
    assert checker.tool_calls_match(calls(stored), calls(source), defaultdict(set))



@pytest.mark.parametrize("smuggled", ['"content": "', '"extra": "'], ids=["duplicate-key", "extra-key"])
def test_tool_call_marker_metadata_cannot_add_keys(smuggled):
    # A marker whose metadata holds JSON syntax can verify at its position and still add a key around it.
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_marker_keys", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    honest = reference_for("payload")
    crafted = honest.replace("chars=", f'field=x", {smuggled}x; chars=', 1)
    assert checker.REFERENCE.fullmatch(crafted) and checker.INGEST.fullmatch(crafted)
    source = '{"content": "payload"}'
    stored = '{"content": "' + crafted + '"}'
    assert checker.content_matches(stored, source, None, defaultdict(set))
    assert not checker.tool_calls_match(calls(stored), calls(source), defaultdict(set))
    assert checker.tool_calls_match(calls('{"content": "' + honest + '"}'), calls(source), defaultdict(set))


def test_tool_call_invalid_json_needs_verified_text():
    # Invalid stored JSON (#2792's escape letter) passes only on the text check; a wrong digest fails.
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_invalid_json", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    run = "QUJD" * 300
    source = json.dumps(dict(content="line\n" + run))
    stored = source.replace("n" + run, reference_for("n" + run))
    try:
        json.loads(stored)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("expected invalid stored JSON")
    assert checker.tool_calls_match(calls(stored), calls(source), defaultdict(set))
    corrupted = stored.replace(reference_for("n" + run).split("_")[-2], "0" * 12)
    assert corrupted != stored
    assert not checker.tool_calls_match(calls(corrupted), calls(source), defaultdict(set))


@pytest.mark.parametrize("shape", ["extra-key", "duplicate-key", "hidden-key"])
def test_tool_call_invalid_json_restores_exactly(shape):
    # Pins the accepted limit of the text-only path: stored text that is not JSON passes when every source
    # character is exact text or lies inside a placeholder verified at its own position, whatever the
    # marker's metadata holds or which characters its span covers. No character is lost; the product
    # cannot write these markers. When #2792 is fixed this path goes, and these cases fail.
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_marker_invalid", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    if shape == "hidden-key":
        source = json.dumps(dict(a="x\n", b="y"))
        span = source[source.index("n"):source.rindex('"')]
        assert span == 'n", "b": "y'
        stored = source.replace(span, reference_for(span))
    else:
        smuggled = '"extra": 1' if shape == "extra-key" else '"content": 1'
        crafted = reference_for("payload").replace("chars=", f'field=x", {smuggled}, "tail": "\\q; chars=', 1)
        source, stored = '{"content": "payload"}', '{"content": "' + crafted + '"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(stored)
    assert checker.tool_calls_match(calls(stored), calls(source), defaultdict(set))
    lossy = stored.replace("}", "!}") if shape == "hidden-key" else stored.replace('"}', 'X"}')
    assert lossy != stored and not checker.tool_calls_match(calls(lossy), calls(source), defaultdict(set))


def test_tool_call_source_duplicate_key_fails_closed():
    # The source is json.dumps output; a duplicate key there cannot come from capture and fails closed.
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_source_duplicate", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    source = '{"content": "lost", "content": "payload"}'
    stored = '{"content": "' + reference_for("payload") + '"}'
    assert not checker.tool_calls_match(calls(stored), calls(source), defaultdict(set))


def test_tool_call_source_parse_error_fails_closed():
    # Only the stored side may take the text-only path; a source that does not parse fails closed.
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_source_invalid", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    crafted = reference_for("pay\tload").replace("chars=", 'field=x", "extra": "x; chars=', 1)
    source, stored = '{"content": "pay\tload"}', '{"content": "' + crafted + '"}'
    assert sorted(json.loads(stored)) == ["content", "extra"]
    assert not checker.tool_calls_match(calls(stored), calls(source), defaultdict(set))

@pytest.mark.parametrize("side", ["stored", "source"])
def test_tool_call_argument_parse_recursion(side, monkeypatch):
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_argument_recursion", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original_loads = json.loads
    def guarded_loads(text, **kwargs):
        if text == side:
            raise RecursionError("nested JSON")
        return original_loads(text, **kwargs)
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    monkeypatch.setattr(checker.json, "loads", guarded_loads)
    assert not checker.tool_calls_match(calls(side if side == "stored" else "{}"),
                                        calls(side if side == "source" else "{}"), defaultdict(set))


@pytest.mark.parametrize("container", ["list", "dict"])
def test_tool_call_argument_walk_iterative(container, monkeypatch):
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_argument_walk", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    source, stored = "payload café", reference_for("payload café")
    for _ in range(sys.getrecursionlimit() * 3):
        source = [source] if container == "list" else dict(child=source)
        stored = [stored] if container == "list" else dict(child=stored)
    def decoded(text, **kwargs):
        return stored if text == "stored" else source
    def calls(arguments):
        return [dict(id="c", type="function", function=dict(name="Write", arguments=arguments))]
    monkeypatch.setattr(checker.json, "loads", decoded)
    index = defaultdict(set)
    assert checker.tool_calls_match(calls("stored"), calls("source"), index)
    assert set(index) == {checker.fingerprint("payload café")}


def reference_for(text, kind="LCM ingest payload"):
    import hashlib
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return (f"[Externalized {kind}: chars={len(text)}; bytes={len(text.encode())}; "
            f"ref=20000101_000000_payload_{digest}_abcd.json]")


def checker_store(tmp_path, entries, rows):
    home = tmp_path / "store"
    home.mkdir()
    with closing(sqlite3.connect(home / "lcm.db")) as conn, conn:
        conn.execute("CREATE TABLE messages(store_id INTEGER, session_id TEXT, role TEXT, content TEXT, "
                     "tool_call_id TEXT, tool_calls TEXT)")
        conn.execute("CREATE TABLE host_uid_bindings(store_id INTEGER, uid TEXT, lineage_key TEXT, kind TEXT)")
        for i, (uid, role, content, call_id, calls) in enumerate(rows, 1):
            conn.execute("INSERT INTO messages VALUES(?,'s',?,?,?,?)", (i, role, content, call_id, calls))
            conn.execute("INSERT INTO host_uid_bindings VALUES(?,?,'claude-code:s','canonical')", (i, uid))
    return home, [("s", write_entries(tmp_path / "s.jsonl", entries))]


@pytest.mark.parametrize("shape", ["duplicate-key", "reformatted"])
@pytest.mark.parametrize("lossy", [True, False], ids=["lossy", "healthy"])
def test_json_text_preservation(tmp_path, shape, lossy, record_property):
    payload = "<payload>"
    source = ('{"data":"important original text","data":"<payload>"}' if shape == "duplicate-key"
              else '{"data":"<payload>","keep":"same"}')
    if lossy:
        stored = json.dumps(dict(data=reference_for(payload)) if shape == "duplicate-key"
                            else dict(keep="same", data=reference_for(payload)))
    else:
        stored = source.replace(payload, reference_for(payload))
    entries = [dict(type="user", uuid="u", sessionId="s", message=dict(content=source))]
    home, inputs = checker_store(tmp_path, entries, [("u", "user", stored, None, None)])
    code, report = check_only(home, {"PATH": os.defpath}, inputs)
    record_property("checker", json.dumps(report))
    assert (code, report["pass"]) == (int(lossy), not lossy)
    assert [(r["lost"], r["duplicated"], r["unverified"]) for r in report["lineages"]] == [
        (int(lossy), int(lossy), 0)]


def test_deep_output_reference(tmp_path, record_property):
    depth = sys.getrecursionlimit() * 3
    text = "[" * depth + '"important original text"' + "]" * depth
    entries = [dict(type="assistant", uuid="a", sessionId="s", message=dict(content=[
                   dict(type="tool_use", id="c", name="synthetic", input={})])),
               dict(type="user", uuid="t", sessionId="s", message=dict(content=[
                   dict(type="tool_result", tool_use_id="c", content=text)]))]
    calls = json.dumps([dict(id="c", type="function", function=dict(name="synthetic", arguments="{}"))])
    home, inputs = checker_store(tmp_path, entries, [("a", "assistant", "", None, calls),
                                ("t", "tool", reference_for(text, "tool output"), "c", None)])
    code, report = check_only(home, {"PATH": os.defpath}, inputs)
    record_property("checker", json.dumps(report))
    assert code == 0 and report["pass"]
    assert [(r["lost"], r["duplicated"], r["unpaired"], r["unverified"]) for r in report["lineages"]] == [
        (0, 0, 0, 0)]


@pytest.mark.parametrize("operation", ["index", "compare"])
def test_deep_structured_parse_guard(operation, monkeypatch):
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_deep_parse", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    depth = sys.getrecursionlimit() * 3
    text = "[" * depth + json.dumps(reference_for("payload")) + "]" * depth
    index = defaultdict(set)
    def recursion_error(_):
        raise RecursionError("nested JSON")
    # Decoder nesting limits vary across Python versions; exercise its failure explicitly.
    monkeypatch.setattr(checker.json, "loads", recursion_error)
    if operation == "index":
        assert checker.index_source(text, index, True) is None
        assert text in index[checker.fingerprint(text)]
    else:
        assert not checker.content_matches(text, "[]", [], index)


@pytest.mark.parametrize("container", ["list", "dict"])
def test_deep_structured_walks(monkeypatch, container):
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location("outside_in_deep_walks", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    original, stored = "payload", reference_for("payload")
    for _ in range(sys.getrecursionlimit() * 3):
        original = [original] if container == "list" else dict(child=original)
        stored = [stored] if container == "list" else dict(child=stored)
    index = defaultdict(set)
    # Isolate the walks from the JSON decoder's independent nesting limit.
    with monkeypatch.context() as patch:
        patch.setattr(checker.json, "loads", lambda _: original)
        assert checker.index_source("[]", index, True) is original
        assert "payload" in index[checker.fingerprint("payload")]
        patch.setattr(checker.json, "loads", lambda _: stored)
        assert checker.content_matches(reference_for("payload") + " ", "[]", original, index)
        wrong = "lost"
        for _ in range(sys.getrecursionlimit() * 3):
            wrong = [wrong] if container == "list" else dict(child=wrong)
        patch.setattr(checker.json, "loads", lambda _: wrong)
        assert not checker.content_matches(reference_for("payload") + " ", "[]", original, index)
