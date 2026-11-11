"""Tests for the LCM-X dev-agent hub. From the repo root:

    uv run --no-project --python 3.12.16 --with pytest==9.1.1 pytest -c adapters/eve/writer/pytest.ini adapters/hub
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcmx_hub as hub_module  # noqa: E402

hub_module.bootstrap()
NODE = HERE.parent / "node"


# -- fixtures: transcripts with the real shapes ------------------------------------------------------------

def cc_user(uid: str, text: str, session: str = "s-1") -> dict:
    return {"type": "user", "uuid": uid, "sessionId": session, "timestamp": "2026-11-09T10:00:00.000Z",
            "message": {"role": "user", "content": text}}


def cc_assistant(uid: str, text: str = "", tool: tuple[str, str, dict] | None = None, session: str = "s-1") -> dict:
    content = ([{"type": "text", "text": text}] if text else []) + (
        [{"type": "tool_use", "id": tool[0], "name": tool[1], "input": tool[2]}] if tool else [])
    return {"type": "assistant", "uuid": uid, "sessionId": session, "timestamp": "2026-11-09T10:00:01.000Z",
            "message": {"role": "assistant", "content": content}}


def cc_result(uid: str, call: str, output: str, session: str = "s-1") -> dict:
    return {"type": "user", "uuid": uid, "sessionId": session, "timestamp": "2026-11-09T10:00:02.000Z",
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call, "content": output}]}}


def codex_lines(texts: list[str]) -> list[dict]:
    lines = [{"type": "session_meta", "ordinal": 0, "payload": {"id": "c-1", "cwd": "/w"}},
             {"type": "response_item", "ordinal": 1, "payload": {"type": "message", "role": "developer",
                                                                  "content": [{"type": "input_text", "text": "rules"}]}}]
    ordinal = 2
    for text in texts:
        lines.append({"type": "response_item", "ordinal": ordinal, "timestamp": "2026-11-09T11:00:00Z",
                      "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
        lines.append({"type": "response_item", "ordinal": ordinal + 1, "payload": {"type": "reasoning", "summary": []}})
        lines.append({"type": "response_item", "ordinal": ordinal + 2, "payload": {
            "type": "custom_tool_call", "call_id": f"k{ordinal}", "name": "apply_patch", "input": "*** patch"}})
        lines.append({"type": "response_item", "ordinal": ordinal + 3, "payload": {
            "type": "custom_tool_call_output", "call_id": f"k{ordinal}", "output": f"patched for {text}"}})
        lines.append({"type": "response_item", "ordinal": ordinal + 4, "payload": {
            "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"done: {text}"}]}})
        lines.append({"type": "event_msg", "ordinal": ordinal + 5, "payload": {"type": "task_complete"}})
        ordinal += 6
    return lines


def write(path: Path, lines: list[dict], mode: str = "w", partial: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
        handle.write(partial)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.delenv("LCMX_LLM_CLI", raising=False)
    monkeypatch.delenv("LCMX_LLM_BASE_URL", raising=False)
    return tmp_path / "hub"


def cc_file(root: Path, node: str = "laptop", session: str = "s-1") -> Path:
    return root / "mirror" / node / "claude" / "projects" / "-Users-mia-work" / f"{session}.jsonl"


def codex_file(root: Path, node: str = "server") -> Path:
    return root / "mirror" / node / "codex" / "sessions" / "2026" / "11" / "09" / \
        "rollout-2026-11-09T11-00-00-0199aaaa-bbbb-cccc-dddd-eeeeffff0000.jsonl"


def rows(root: Path) -> list[tuple[str, str, str]]:
    with closing(sqlite3.connect(root / "store" / "lcm.db")) as conn:
        return conn.execute("SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()


def scan(root: Path, **kwargs) -> dict:
    hub = hub_module.Hub(root, **kwargs)
    try:
        return hub.scan()
    finally:
        hub.close()


# -- capture ------------------------------------------------------------------------------------------------

def test_claude_and_codex_sessions_are_stored_once_and_apart(root: Path) -> None:
    write(cc_file(root), [cc_user("u1", "fix the parser"), cc_assistant("a1", "looking", ("t1", "Read", {"p": "x.py"})),
                          cc_result("r1", "t1", "def parse(): ..."), cc_assistant("a2", "fixed")])
    write(cc_file(root, session="s-2"), [cc_user("v1", "other task", "s-2"), cc_assistant("w1", "ok", session="s-2")])
    write(codex_file(root), codex_lines(["add tests"]))
    report = scan(root)
    assert report["failed"] == 0
    stored = rows(root)
    by_session: dict[str, list[tuple[str, str]]] = {}
    for session, role, content in stored:
        by_session.setdefault(session, []).append((role, content))
    assert [r for r, _ in by_session["cc:laptop:s-1"]] == ["user", "assistant", "tool", "assistant"]
    assert by_session["cc:laptop:s-2"] == [("user", "other task"), ("assistant", "ok")]
    codex = by_session["codex:server:0199aaaa-bbbb-cccc-dddd-eeeeffff0000"]
    assert [r for r, _ in codex] == ["system", "user", "assistant", "tool", "assistant"]
    assert codex[-1][1] == "done: add tests"
    assert scan(root)["changed"] == 0  # nothing new: nothing stored again
    result = hub_module.audit(root)
    assert result["ok"], result
    assert result["sessions"] == 3


def test_appends_partial_lines_and_rewrites(root: Path) -> None:
    path = cc_file(root)
    write(path, [cc_user("u1", "first")], partial='{"type": "user", "uuid": "u2", "mess')  # a line being written
    scan(root)
    assert [c for _s, _r, c in rows(root)] == ["first"]
    with open(path, "w") as handle:  # the writer finished the line and added another
        handle.write(json.dumps(cc_user("u1", "first")) + "\n" + json.dumps(cc_user("u2", "second")) + "\n")
    scan(root)
    assert [c for _s, _r, c in rows(root)] == ["first", "second"]
    write(path, [cc_user("u0", "a new head"), cc_user("u1", "first"), cc_user("u2", "second")])  # file replaced
    scan(root)
    assert [c for _s, _r, c in rows(root)] == ["first", "second", "a new head"]  # only the new entry
    assert hub_module.audit(root)["ok"]


def test_subagent_transcript_is_its_own_session(root: Path) -> None:
    write(cc_file(root), [cc_user("u1", "main task")])
    write(cc_file(root).with_suffix("") / "subagents" / "agent-abc.jsonl", [cc_user("x1", "sub task")])
    scan(root)
    sessions = {s for s, _r, _c in rows(root)}
    assert sessions == {"cc:laptop:s-1", "cc:laptop:s-1/agent:abc"}


def test_summaries_through_an_llm_route(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import re

    import agent.auxiliary_client as auxiliary

    calls = []

    class Model:
        def complete(self, params: dict, timeout: float) -> dict:
            calls.append(params)
            nonce = re.search(r'<lcm-summary nonce="([0-9a-f]+)">', params["messages"][0]["content"]).group(1)
            return {"content": f'<lcm-summary nonce="{nonce}">\nThe user filed numbered notes.\n'
                               f'Expand for details about: notes\n</lcm-summary>'}

    monkeypatch.setattr(auxiliary, "BRIDGE", Model())
    lines = []
    for index in range(40):
        lines += [cc_user(f"u{index}", f"Note {index}: " + "words " * 250), cc_assistant(f"a{index}", f"noted {index}")]
    write(cc_file(root), lines)
    scan(root, context_length=8_000, summarize=True)
    with closing(sqlite3.connect(root / "store" / "lcm.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] >= 1
    assert calls
    assert hub_module.audit(root)["ok"]


# -- node pushes ----------------------------------------------------------------------------------------------

def frames(*items: tuple[dict, bytes]) -> io.BufferedReader:
    data = b"".join(json.dumps(header).encode() + b"\n" + body for header, body in items) + b'{"done": true}\n'
    return io.BufferedReader(io.BytesIO(data))


def test_push_appends_replaces_and_refuses(tmp_path: Path) -> None:
    node_root = tmp_path / "mirror" / "laptop"
    rel = "claude/projects/p/s.jsonl"
    report = hub_module.receive_push(node_root, frames(({"path": rel, "offset": 0, "length": 4, "replace": True}, b"abc\n")))
    assert report == {"files": 1, "bytes": 4, "refused": 0}
    hub_module.receive_push(node_root, frames(({"path": rel, "offset": 4, "length": 4, "replace": False}, b"def\n")))
    assert (node_root / rel).read_bytes() == b"abc\ndef\n"
    wrong_offset = hub_module.receive_push(node_root, frames(({"path": rel, "offset": 2, "length": 1, "replace": False}, b"x")))
    assert wrong_offset["refused"] == 1 and (node_root / rel).read_bytes() == b"abc\ndef\n"
    for bad in ("../laptop2/claude/projects/p/s.jsonl", "claude/projects/../../x.jsonl", "/etc/passwd",
                "claude/settings.json", "codex/auth.json", "codex/sessions/x.txt"):
        refused = hub_module.receive_push(node_root, frames(({"path": bad, "offset": 0, "length": 1, "replace": True}, b"x")))
        assert refused["refused"] == 1, bad
    assert hub_module.allowed("claude/projects/p/s/tool-results/out.txt")
    state = hub_module.node_state(node_root)
    assert state[rel]["size"] == 8


def load_sync():
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader("lcmx_sync", str(NODE / "lcmx_sync"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("lcmx_sync", loader))
    loader.exec_module(module)
    return module


def test_sync_plan_sends_only_what_the_hub_lacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sync = load_sync()
    claude = tmp_path / "claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    project = claude / "projects" / "p"
    project.mkdir(parents=True)
    (project / "same.jsonl").write_bytes(b"line1\n")
    (project / "grown.jsonl").write_bytes(b"line1\nline2\n")
    (project / "changed.jsonl").write_bytes(b"NEW\n")
    (project / "new.jsonl").write_bytes(b"x\n")
    (project / "notes.md").write_bytes(b"not a transcript")
    digest = lambda data: __import__("hashlib").sha256(data).hexdigest()  # noqa: E731
    remote = {"claude/projects/p/same.jsonl": {"size": 6, "head": digest(b"line1\n")},
              "claude/projects/p/grown.jsonl": {"size": 6, "head": digest(b"line1\n")},
              "claude/projects/p/changed.jsonl": {"size": 4, "head": digest(b"OLD\n")}}
    todo = {rel: (base, offset, replace) for rel, _path, base, offset, replace in sync.plan(remote)}
    assert todo == {"claude/projects/p/grown.jsonl": (0, 6, False), "claude/projects/p/changed.jsonl": (0, 0, True),
                    "claude/projects/p/new.jsonl": (0, 0, True)}
    # The plugins ship the same scripts as node/.
    for plugin in ("claude-code", "codex"):
        for name in ("lcmx_sync", "lcmx_recall_remote", "lcmx_context"):
            assert (HERE.parent / "plugins" / plugin / "bin" / name).read_bytes() == (NODE / name).read_bytes()
        skill = HERE.parent / "plugins" / plugin / "skills" / "lcmx" / "SKILL.md"
        assert skill.read_bytes() == (HERE.parent / "skills" / "lcmx" / "SKILL.md").read_bytes()


def node_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync):
    claude = tmp_path / "claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.delenv("LCMX_HUB_HISTORY", raising=False)
    monkeypatch.setattr(sync, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(sync, "BASELINE", tmp_path / "state" / "baseline-v2.json")
    monkeypatch.setattr(sync, "BASELINE_V1", tmp_path / "state" / "baseline.json")
    return claude / "projects" / "p"


def push_to(node_root: Path, sync, todo: list) -> dict:
    data = b"".join(sync.push_frames(todo)) + b'{"done": true}\n'
    return hub_module.receive_push(node_root, io.BufferedReader(io.BytesIO(data)))


def test_only_turns_after_the_baseline_reach_the_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Version 2 baseline (DX D-13): an old session keeps its history back but sends its new turns; new files go
    whole; the hub's copy of a file is its bytes after the baseline offset."""
    sync = load_sync()
    project = node_home(tmp_path, monkeypatch, sync)
    project.mkdir(parents=True)
    old = project / "old.jsonl"
    old.write_bytes(b'{"a":1}\n{"a":2}\n{"partial"')  # a half-written line at the install
    (project / "old" / "subagents").mkdir(parents=True)
    (project / "old" / "subagents" / "agent-1.jsonl").write_bytes(b'{"s":1}\n')
    (project / "old" / "tool-results").mkdir()
    (project / "old" / "tool-results" / "t1.txt").write_bytes(b"old output")
    bases = sync.baseline()
    assert bases["claude/projects/p/old.jsonl"] == 16  # after the last complete line
    assert bases["claude/projects/p/old/tool-results/t1.txt"] == 10
    assert list(sync.plan({}, bases)) == [("claude/projects/p/old.jsonl", old, 16, 0, True)]  # only the partial line
    with open(old, "ab") as handle:
        handle.write(b':1}\n{"a":3}\n')
    (project / "old" / "subagents" / "agent-2.jsonl").write_bytes(b'{"s":2}\n')
    (project / "later.jsonl").write_bytes(b'{"new":1}\n')
    node_root = tmp_path / "hub" / "mirror" / "laptop"
    todo = list(sync.plan({}, sync.baseline()))
    assert {rel for rel, *_ in todo} == {"claude/projects/p/old.jsonl", "claude/projects/p/later.jsonl",
                                          "claude/projects/p/old/subagents/agent-2.jsonl"}
    push_to(node_root, sync, todo)
    assert (node_root / "claude/projects/p/old.jsonl").read_bytes() == b'{"partial":1}\n{"a":3}\n'
    # The next turn appends only the new bytes, and an unchanged node sends nothing.
    with open(old, "ab") as handle:
        handle.write(b'{"a":4}\n')
    remote = hub_module.node_state(node_root)
    todo = list(sync.plan(remote, sync.baseline()))
    assert [(rel, offset, replace) for rel, _path, _base, offset, replace in todo] == [
        ("claude/projects/p/old.jsonl", 22, False)]
    push_to(node_root, sync, todo)
    assert (node_root / "claude/projects/p/old.jsonl").read_bytes() == b'{"partial":1}\n{"a":3}\n{"a":4}\n'
    assert list(sync.plan(hub_module.node_state(node_root), sync.baseline())) == []
    # A file rewritten below its baseline offset sends nothing (the old history stays on the node).
    old.write_bytes(b'{"a":1}\n')
    assert [rel for rel, *_ in sync.plan(hub_module.node_state(node_root), sync.baseline())] == []
    monkeypatch.setenv("LCMX_HUB_HISTORY", "all")
    assert sync.baseline() is None


def test_exclude_keeps_agent_workspaces_off_the_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sync = load_sync()
    project = node_home(tmp_path, monkeypatch, sync)
    agents = project.parent / "-Users-mia-agents-bot-work"
    for folder, name in ((project, "dev.jsonl"), (agents, "agent.jsonl"), (agents / "s" / "subagents", "a-1.jsonl")):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b'{"x":1}\n')
    patterns = sync.exclude_patterns({"LCMX_HUB_EXCLUDE": " claude/projects/-Users-mia-agents-* ,,"})
    assert patterns == ["claude/projects/-Users-mia-agents-*"]
    assert [rel for rel, *_ in sync.plan({}, {}, patterns)] == ["claude/projects/p/dev.jsonl"]
    assert len(list(sync.plan({}, {}))) == 3


def test_a_version_1_baseline_is_converted_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-11-09 nodes have a list of sessions; their files start at the first line written after that time."""
    sync = load_sync()
    project = node_home(tmp_path, monkeypatch, sync)
    created = 1_794_236_800.0  # 2026-11-09T15:06:40Z
    before = json.dumps({"type": "user", "timestamp": "2026-11-09T15:00:00.123456789Z"}).encode() + b"\n"
    no_time = b'{"type": "summary"}\n'
    after = json.dumps({"type": "user", "timestamp": "2026-11-09T22:10:00+07:00"}).encode() + b"\n"
    write_bytes = lambda path, data: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(data))  # noqa
    write_bytes(project / "s-old.jsonl", before + no_time + after)
    write_bytes(project / "s-quiet.jsonl", before)
    write_bytes(project / "s-old" / "tool-results" / "old.txt", b"x")
    os.utime(project / "s-old" / "tool-results" / "old.txt", (created - 60, created - 60))
    write_bytes(project / "s-old" / "tool-results" / "new.txt", b"y")
    write_bytes(project / "s-new.jsonl", before)
    sync.BASELINE.parent.mkdir(parents=True)
    version_1 = json.dumps({"created": created, "sessions": ["claude/projects/p/s-old", "claude/projects/p/s-quiet"]})
    sync.BASELINE_V1.write_text(version_1)
    bases = sync.baseline()
    assert bases == {"claude/projects/p/s-old.jsonl": len(before + no_time),
                     "claude/projects/p/s-quiet.jsonl": len(before),
                     "claude/projects/p/s-old/tool-results/old.txt": 1}
    assert json.loads(sync.BASELINE.read_text())["version"] == 2 and sync.baseline() == bases
    assert sync.BASELINE_V1.read_text() == version_1  # an 0.1.0 timer that still runs keeps its session list
    todo = {rel: base for rel, _path, base, _offset, _replace in sync.plan({}, bases)}
    assert todo == {"claude/projects/p/s-old.jsonl": len(before + no_time), "claude/projects/p/s-new.jsonl": 0,
                    "claude/projects/p/s-old/tool-results/new.txt": 0}


# -- daemon and recall ------------------------------------------------------------------------------------------

def test_daemon_serves_recall_over_its_socket(root: Path) -> None:
    write(cc_file(root), [cc_user("u1", "the deploy key lives in the vault item ZEBRA-17"), cc_assistant("a1", "noted")])
    env = {**os.environ, "LCM_EMBEDDINGS_ENABLED": "false"}
    daemon = subprocess.Popen([sys.executable, "-I", str(HERE / "lcmx_hub.py"), "serve", "--root", str(root)], env=env)
    try:
        for _ in range(100):
            if (root / hub_module.SOCKET).exists():
                break
            time.sleep(0.1)
        hub_module._request(root, {"method": "scan"})
        requests = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                     "params": {"name": "lcm_grep", "arguments": {"query": "ZEBRA-17"}}}]
        out = subprocess.run([sys.executable, "-I", str(HERE / "lcmx_hub.py"), "recall", "--root", str(root)],
                             input="\n".join(json.dumps(r) for r in requests) + "\n", capture_output=True, text=True,
                             env=env, timeout=120)
        replies = {r["id"]: r for r in map(json.loads, out.stdout.splitlines())}
        assert replies[1]["result"]["serverInfo"]["name"] == "lcmx-hub"
        names = {tool["name"] for tool in replies[2]["result"]["tools"]}
        assert {"lcm_grep", "lcm_recall", "lcm_expand", "lcm_load_session"} <= names
        found = json.loads(replies[3]["result"]["content"][0]["text"])
        assert found["total_results"] >= 1 and found["results"][0]["session_id"] == "cc:laptop:s-1"
        capsule = hub_module._request(root, {"method": "context", "node": "laptop", "harness": "claude",
                                             "session": "s-1", "event": "compact"})
        assert capsule["session"] == "cc:laptop:s-1" and "lcm_load_session" in capsule["text"]
    finally:
        daemon.terminate()
        daemon.wait(30)


def test_codex_cli_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """call_llm through a coding CLI: a fake ``codex`` that answers in the file named by -o."""
    import agent.auxiliary_client as auxiliary

    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "codex").write_text("#!/bin/sh\nout=''\nwhile [ $# -gt 0 ]; do [ \"$1\" = -o ] && out=$2; shift; done\n"
                                "cat > /dev/null\nprintf 'summary from codex' > \"$out\"\n")
    (fake / "codex").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    monkeypatch.setenv("LCMX_LLM_CLI", "codex")
    monkeypatch.setattr(auxiliary, "BRIDGE", None)
    response = auxiliary.call_llm(messages=[{"role": "system", "content": "be brief"},
                                            {"role": "user", "content": "x"}], task="compression")
    assert response.choices[0].message.content == "summary from codex"


def test_backup_is_consistent(root: Path) -> None:
    write(cc_file(root), [cc_user("u1", "hello"), cc_assistant("a1", "hi")])
    scan(root)
    buffer = io.BytesIO()
    manifest = hub_module.backup(root, buffer)
    assert manifest["files"]["lcm.db"]["integrity"] == "ok"
    assert manifest["files"]["hub-ledger.db"]["rows"]["entries"] == 2
    import tarfile

    archive = tarfile.open(fileobj=io.BytesIO(buffer.getvalue()))
    assert set(archive.getnames()) == {"lcm.db", "hub-ledger.db", "manifest.json"}
    copy = root / "restored-ledger.db"
    copy.write_bytes(archive.extractfile("hub-ledger.db").read())
    with closing(sqlite3.connect(copy)) as conn:  # D-22: no raw (unredacted) text in the backup
        assert conn.execute("SELECT COUNT(*) FROM entries WHERE length(raw) > 0").fetchone()[0] == 0


def test_a_new_hub_has_a_store_to_back_up(root: Path) -> None:
    scan(root)  # no transcripts yet
    assert (root / "store" / "lcm.db").exists()
    manifest = hub_module.backup(root, io.BytesIO())
    assert manifest["files"]["lcm.db"]["integrity"] == "ok" and manifest["files"]["lcm.db"]["rows"]
    assert hub_module.audit(root)["ok"]


def test_ssh_command_refuses_other_verbs(root: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "bash -c id")
    assert hub_module.ssh_command(root, "laptop") == 2
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "state")
    assert hub_module.ssh_command(root, "laptop") == 0
    assert json.loads(capsys.readouterr().out) == {}
    assert hub_module.ssh_command(root, "../x") == 2


# -- context hooks ------------------------------------------------------------------------------------------------

def summarizing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import re

    import agent.auxiliary_client as auxiliary

    class Model:
        def complete(self, params: dict, timeout: float) -> dict:
            nonce = re.search(r'<lcm-summary nonce="([0-9a-f]+)">', params["messages"][0]["content"]).group(1)
            return {"content": f'<lcm-summary nonce="{nonce}">\nThe user filed numbered notes about the pier.\n'
                               f'Expand for details about: notes\n</lcm-summary>'}

    monkeypatch.setattr(auxiliary, "BRIDGE", Model())


def test_idle_summary_keeps_the_harness_and_stores_no_row_twice(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle summary once bound the session as platform "cc" (its id prefix), not "claude-code"; the engine then
    stored the session's tail again, and every later turn of it too."""
    import sqlite3

    summarizing_model(monkeypatch)
    lines = []
    for index in range(40):
        lines += [cc_user(f"u{index}", f"Note {index}: " + "words " * 250), cc_assistant(f"a{index}", f"noted {index}")]
    write(cc_file(root), lines)
    quiet = hub_module.Hub(root, context_length=8_000, summarize=False)
    try:
        quiet.scan()  # over the threshold, no summary yet: the idle backlog summarizes it
    finally:
        quiet.close()
    hub = hub_module.Hub(root, context_length=8_000, summarize=True)
    try:
        assert hub.summarize_backlog() == {"session": "cc:laptop:s-1"}
        write(cc_file(root), lines + [cc_user("u-late", "a later turn about the pier"), cc_assistant("a-late", "ok")])
        hub.scan()
    finally:
        hub.close()
    report = hub_module.audit(root)
    assert report["ok"], report
    with closing(sqlite3.connect(root / "store" / "lcm.db")) as db:
        assert db.execute("SELECT DISTINCT source FROM messages").fetchall() == [("claude-code",)]


def test_hub_context_capsule_and_prompt_recall(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summarizing_model(monkeypatch)
    lines = []
    for index in range(40):
        lines += [cc_user(f"u{index}", f"Note {index}: " + "words " * 250), cc_assistant(f"a{index}", f"noted {index}")]
    write(cc_file(root), lines)
    write(codex_file(root), codex_lines(["rotate the deploy key ZEBRA-17 in the vault"]))
    write(cc_file(root, "server", "s-noise"), [cc_user("n1", "<system-reminder>Linked sessions: ZEBRA-17 lead</system-reminder>",
                                                     "s-noise"), cc_assistant("n2", "ok", session="s-noise")])
    hub = hub_module.Hub(root, context_length=8_000, summarize=True)
    try:
        hub.scan()
        capsule = hub.context("laptop", "claude", "s-1", "compact")
        assert capsule["session"] == "cc:laptop:s-1"
        assert "lcm_load_session(session_id='cc:laptop:s-1')" in capsule["text"]
        assert "Hub summaries of this session" in capsule["text"] and "numbered notes about the pier" in capsule["text"]
        assert "lcm_expand(node_id=" in capsule["text"]
        assert "not synced yet" in hub.context("laptop", "claude", "s-2", "compact")["text"]
        found = hub.context("laptop", "claude", "s-1", "prompt", "where is the deploy key ZEBRA-17")
        assert "codex:server:0199aaaa-bbbb-cccc-dddd-eeeeffff0000" in found["text"] and "ZEBRA-17" in found["text"]
        assert "s-noise" not in found["text"]  # harness reminders are not memories
        own = hub.context("laptop", "claude", "s-1", "prompt", "Note 3 words")["text"]
        assert "cc:laptop:s-1" not in own  # the agent has its own session; the hits come from others
        many = hub.context("laptop", "claude", "other", "prompt", "Note words")["text"]
        assert many.count("cc:laptop:s-1") == 2  # at most 2 hits per session
        assert hub.context("laptop", "codex", "0199aaaa-bbbb-cccc-dddd-eeeeffff0000", "start")["session"] == \
            "codex:laptop:0199aaaa-bbbb-cccc-dddd-eeeeffff0000"
        for bad in ("", "../x", "a b", "x" * 200):
            with pytest.raises(ValueError):
                hub.context("laptop", "claude", bad, "compact")
    finally:
        hub.close()


def test_ssh_context_names_the_session_without_the_daemon(root: Path, monkeypatch: pytest.MonkeyPatch,
                                                          capsys) -> None:
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "context")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"event": "id", "harness": "claude",
                                                              "session": "0a1b2c3d-4e5f"}) + "\n"))
    assert hub_module.ssh_command(root, "laptop") == 0
    assert json.loads(capsys.readouterr().out) == {"session": "cc:laptop:0a1b2c3d-4e5f"}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"event": "compact", "harness": "claude",
                                                              "session": "s-1"}) + "\n"))
    assert hub_module.ssh_command(root, "laptop") == 0  # no daemon: an error line, and the hook prints nothing
    assert "error" in json.loads(capsys.readouterr().out)


FAKE_SSH = """#!/bin/sh
last=""; for a in "$@"; do last=$a; done
case "$last" in
  state) echo '{}' ;;
  push) cat > /dev/null; echo '{}' ;;
  context) cat >> "$FAKE_LOG"; printf '%s\\n' "$FAKE_REPLY" ;;
esac
"""


def run_context(tmp_path: Path, payload: dict, reply: dict, harness: str = "claude", **env: str):
    fake = tmp_path / "fakebin"
    fake.mkdir(exist_ok=True)
    (fake / "ssh").write_text(FAKE_SSH)
    (fake / "ssh").chmod(0o755)
    (tmp_path / "node.env").write_text("LCMX_HUB_HOST=hubhost\nLCMX_HUB_KEY=~/.ssh/lcmx_node\n")
    log = tmp_path / "requests.log"
    full = {"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(tmp_path), "LCMX_HUB_CONFIG": str(tmp_path / "node.env"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"), "CODEX_HOME": str(tmp_path / "codex"),
            "FAKE_LOG": str(log), "FAKE_REPLY": json.dumps(reply), **env}
    done = subprocess.run([sys.executable, "-I", "-S", str(NODE / "lcmx_context"), "--harness", harness],
                          input=json.dumps(payload), capture_output=True, text=True, env=full, timeout=60)
    requests = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return done, requests


def test_context_hook_script(tmp_path: Path) -> None:
    start = {"hook_event_name": "SessionStart", "source": "startup", "session_id": "s-9"}
    done, requests = run_context(tmp_path, start, {})
    assert done.returncode == 0 and requests == []  # the note is local
    out = json.loads(done.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart" and "lcm_recall" in out["additionalContext"]
    compact = dict(start, source="compact")
    done, requests = run_context(tmp_path, compact, {"session": "codex:laptop:s-9", "text": "CAPSULE"}, "codex")
    assert requests == [{"event": "compact", "harness": "codex", "session": "s-9"}]
    assert json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"].endswith("CAPSULE")
    (tmp_path / "requests.log").unlink()
    prompt = {"hook_event_name": "UserPromptSubmit", "session_id": "s-9", "prompt": "how did we fix the relay timeout?"}
    done, requests = run_context(tmp_path, prompt, {"text": "HITS"})
    assert done.stdout == "" and requests == []  # proactive recall is off by default
    done, requests = run_context(tmp_path, prompt, {"text": "HITS"}, LCMX_HUB_PROACTIVE="on")
    assert requests[0]["query"] == prompt["prompt"]
    assert json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"] == "HITS"
    (tmp_path / "requests.log").unlink()
    precompact = {"hook_event_name": "PreCompact", "session_id": "s-9", "trigger": "auto"}
    done, requests = run_context(tmp_path, precompact, {"session": "cc:laptop:s-9"})
    assert requests == [{"event": "id", "harness": "claude", "session": "s-9"}]
    assert "cc:laptop:s-9" in done.stdout and "word for word" in done.stdout
    assert (tmp_path / ".cache" / "lcmx-hub" / "last-sync.json").exists()  # it synced first
    done, _ = run_context(tmp_path, precompact, {"session": "codex:laptop:s-9"}, "codex")
    assert done.stdout == ""  # Codex: sync only
    # A hub error never breaks the session: no output, exit 0, the error is kept for the operator.
    done, _ = run_context(tmp_path, compact, {"error": "down"})
    assert done.returncode == 0 and "CAPSULE" not in done.stdout
    assert (tmp_path / ".cache" / "lcmx-hub" / "context-error.txt").read_text().strip().endswith("hub: down")
    done, _ = run_context(tmp_path, {"hook_event_name": "Stop"}, {})
    assert done.returncode == 0 and done.stdout == ""
