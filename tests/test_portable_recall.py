"""Synthetic preservation/isolation gates; no native-host qualification claims."""
import io
import base64
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_lcm.portable import PortableError, PortableRecall, project_event, serve


def portable(tmp_path, host="manual", **kwargs):
    return PortableRecall(tmp_path / "corpora", project="synthetic-project",
                          instance=kwargs.get("instance", "synthetic-instance"), host=host)


def claude(text, event="event-1", session="session-a", **extra):
    return {"type": "user", "uuid": event, "sessionId": session,
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": text}, **extra}


def write_jsonl(path, events, trailing=""):
    path.write_text("".join(json.dumps(event) + "\n" for event in events) + trailing)


def rows(client, session="session-a"):
    context = client.open(session)
    try:
        return context._store.load_session_page(session, limit=1000)
    finally:
        context.close()


def references(value):
    if isinstance(value, dict):
        if "portable_ref" in value:
            yield value
        for item in value.values():
            yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def test_identity_not_content_dedup_and_changed_event_preserved(tmp_path):
    client = portable(tmp_path)
    first = client.ingest("session-a", event_id="event-1", messages=[{"role": "user", "content": "same"}])
    same = client.ingest("session-a", event_id="event-1", messages=[{"role": "user", "content": "same"}])
    distinct = client.ingest("session-a", event_id="event-2", messages=[{"role": "user", "content": "same"}])
    changed = client.ingest("session-a", event_id="event-1", messages=[{"role": "user", "content": "corrected"}])
    assert same["status"] == "duplicate" and same["store_ids"] == first["store_ids"]
    assert distinct["status"] == "appended" and distinct["store_ids"] != first["store_ids"]
    assert changed["status"] == "conflict" and changed["version"] == 2
    assert [row["content"] for row in rows(client)] == ["same", "same", "corrected"]


def test_atomic_rows_ledger_and_checkpoint_on_failure(tmp_path):
    client = portable(tmp_path)
    context = client.open("session-a", create=True)
    try:
        conn = context._store.connection
        conn.execute("CREATE TEMP TRIGGER fail_receipt BEFORE INSERT ON metadata "
                     "WHEN NEW.key='portable:event:synthetic' BEGIN SELECT RAISE(ABORT,'synthetic crash'); END")
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            context._store.append_source_event(
                "session-a", [{"role": "user", "content": "not committed"}],
                identity="synthetic", digest="digest", source_metadata={"synthetic": True},
                checkpoint_key="portable:checkpoint:synthetic", checkpoint={"offset": 50})
        assert context._store.get_session_count("session-a") == 0
        assert context._store.read_metadata_json("portable:event:synthetic") is None
        assert context._store.read_metadata_json("portable:row:1") is None
        assert context._store.read_metadata_json("portable:checkpoint:synthetic") is None
    finally:
        context.close()
    assert rows(portable(tmp_path)) == []


def test_rows_and_original_envelope_both_use_ingest_protection(tmp_path):
    client = portable(tmp_path)
    blob = base64.b64encode(bytes(range(256)) * 32).decode()
    content = "data:image/png;base64," + blob
    receipt = client.ingest("session-a", event_id="media", messages=[{"role": "user", "content": content}],
                            envelope={"synthetic_media": content})
    context = client.open("session-a")
    try:
        row = context._store.get(receipt["store_ids"][0])
        provenance = context._store.read_metadata_json(f"portable:row:{row['store_id']}")
        assert blob not in row["content"]
        assert blob not in provenance["source_metadata"]
        assert provenance["identity"] == receipt["event_identity"]
    finally:
        context.close()


def test_production_recall_and_exact_unicode_pagination(tmp_path):
    client = portable(tmp_path)
    text = "nebula " + "é 東京 👋 quote \" punctuation. " * 100
    receipt = client.ingest("session-a", event_id="event-1", messages=[{"role": "user", "content": text}])
    result = client.call("lcm_recall", {"session": "session-a", "query": "nebula"})
    assert "error" not in result
    assert result["total_results"] == 1
    hits = list(references(result))
    assert hits and all(hit["corpus_id"] == receipt["corpus_id"] for hit in hits)
    for hit in hits:
        page = client.call("lcm_expand", {"session": "session-a", "reference": hit["portable_ref"], "max_tokens": 2000})
        assert page["content"] == hit["content"][:len(page["content"])]
    # The full stored span is independently pageable, without normalizing text.
    reference = f"lcmx:{receipt['corpus_id']}:lcm:{receipt['store_ids'][0]}:0-{len(text)}"
    offset, parts = 0, []
    for _ in range(1000):
        page = client.call("lcm_expand", {"session": "session-a", "reference": reference,
                                          "offset": offset, "max_tokens": 10})
        parts.append(page["content"])
        assert text[page["content_offset"]:page["content_offset"] + len(page["content"])] == page["content"]
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    else:
        pytest.fail("pagination did not terminate")
    assert "".join(parts) == text
    assert client.call("lcm_describe", {"session": "session-a"})["store_message_count"] == 1


def test_namespace_and_session_isolation(tmp_path):
    client = portable(tmp_path)
    receipt = client.ingest("session-a", event_id="a", messages=[{"role": "user", "content": "nebula alpha"}])
    client.ingest("session-b", event_id="b", messages=[{"role": "user", "content": "other evidence"}])
    ref = f"lcmx:{receipt['corpus_id']}:lcm:{receipt['store_ids'][0]}:0-6"
    with pytest.raises(PortableError, match="reference_corpus_mismatch"):
        client.call("lcm_expand", {"session": "session-b", "reference": ref})
    result = client.call("lcm_recall", {"session": "session-b", "query": "nebula"})
    assert not list(references(result))
    other = portable(tmp_path, instance="other-instance")
    assert other.corpus_id("session-a") != client.corpus_id("session-a")
    with pytest.raises(PortableError, match="session_not_captured"):
        other.call("lcm_describe", {"session": "session-a"})
    with pytest.raises(PortableError, match="required_argument_missing"):
        client.call("lcm_describe", {})
    with pytest.raises(PortableError, match="unsupported_argument"):
        client.call("lcm_describe", {"session": "session-a", "root": str(tmp_path)})


def test_capture_restart_incomplete_and_source_unchanged(tmp_path):
    client = portable(tmp_path, "claude")
    path = tmp_path / "synthetic.jsonl"
    write_jsonl(path, [claude("first")], json.dumps(claude("second", "event-2")))
    original = path.read_bytes()
    result = client.capture("session-a", transcript=path, transcript_root=tmp_path)
    assert result["incomplete"] and len(rows(client)) == 1
    assert path.read_bytes() == original
    with path.open("a") as stream:
        stream.write("\n")
    restarted = portable(tmp_path, "claude")
    result = restarted.capture("session-a", transcript=path, transcript_root=tmp_path)
    assert result["coverage_complete"] and [row["content"] for row in rows(restarted)] == ["first", "second"]
    assert restarted.capture("session-a", transcript=path, transcript_root=tmp_path)["appended"] == 0


def test_capture_crash_after_event_commit_preserves_cursor_and_gap(tmp_path, monkeypatch):
    client = portable(tmp_path, "claude")
    path = tmp_path / "synthetic.jsonl"
    write_jsonl(path, [{"type": "future_event", "uuid": "gap", "sessionId": "session-a"}, claude("after gap")])
    original_ingest = client.ingest
    def commit_then_crash(*args, **kwargs):
        original_ingest(*args, **kwargs)
        raise RuntimeError("synthetic postcommit crash")
    monkeypatch.setattr(client, "ingest", commit_then_crash)
    with pytest.raises(RuntimeError):
        client.capture("session-a", transcript=path, transcript_root=tmp_path)
    restarted = portable(tmp_path, "claude")
    report = restarted.capture("session-a", transcript=path, transcript_root=tmp_path)
    assert report["unsupported_total"] == 1 and not report["coverage_complete"]
    assert [row["content"] for row in rows(restarted)] == ["after gap"]
    assert restarted.capture("session-a", transcript=path, transcript_root=tmp_path)["unsupported_total"] == 1


def test_capture_rewrite_and_rotation_keep_old_generation(tmp_path):
    client = portable(tmp_path, "claude")
    path = tmp_path / "synthetic.jsonl"
    write_jsonl(path, [claude("first"), claude("original", "event-2")])
    client.capture("session-a", transcript=path, transcript_root=tmp_path)
    # Same first line, same inode, same byte length: consumed-prefix validation matters.
    write_jsonl(path, [claude("first"), claude("modified", "event-2")])
    report = client.capture("session-a", transcript=path, transcript_root=tmp_path)
    assert report["generation_changed"]
    assert [row["content"] for row in rows(client)] == ["first", "original", "first", "modified"]
    write_jsonl(path, [claude("rotated", "event-3")])
    assert client.capture("session-a", transcript=path, transcript_root=tmp_path)["generation_changed"]
    assert len(rows(client)) == 5


def test_distinct_source_files_and_repeated_uuid_conflicts(tmp_path):
    client = portable(tmp_path, "claude")
    for name in ("one", "two"):
        path = tmp_path / (name + ".jsonl")
        write_jsonl(path, [claude("same")])
        client.capture("session-a", transcript=path, transcript_root=tmp_path)
    assert len(rows(client)) == 2
    path = tmp_path / "one.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(claude("correction")) + "\n")
    assert client.capture("session-a", transcript=path, transcript_root=tmp_path)["conflicts"] == 1
    assert [row["content"] for row in rows(client)] == ["same", "same", "correction"]
    with path.open("a") as stream:
        stream.write(json.dumps(claude("correction")) + "\n")
    assert client.capture("session-a", transcript=path, transcript_root=tmp_path)["duplicates"] == 1
    assert len(rows(client)) == 3


def test_capture_scope_and_branch_identity(tmp_path):
    client = portable(tmp_path, "claude")
    path = tmp_path / "synthetic.jsonl"
    write_jsonl(path, [claude("wrong", session="session-b")])
    with pytest.raises(PortableError, match="transcript_session_mismatch"):
        client.capture("session-a", transcript=path, transcript_root=tmp_path)
    assert rows(client) == []
    with pytest.raises(PortableError, match="message_branch_identity_missing"):
        project_event("claude", "session-a#agent:branch-a", claude("main"))
    event = claude("fork", agentId="branch-a", parentUuid="parent-event")
    projected, supported = project_event("claude", "session-a#agent:branch-a", event)
    assert supported and projected[0]["content"] == "fork"
    client.ingest("session-a#agent:branch-a", event_id="fork", messages=projected, envelope=event)
    assert client.corpus_id("session-a#agent:branch-a") != client.corpus_id("session-a")
    with pytest.raises(PortableError, match="transcript_outside_configured_root"):
        client.capture("session-a", transcript=path, transcript_root=tmp_path / "corpora")


def test_codex_finalized_tools_and_unmatched_chronology(tmp_path):
    client = portable(tmp_path, "codex")
    path = tmp_path / "synthetic.jsonl"
    events = [{"type": "session_meta", "payload": {"id": "session-a", "forked_from_id": "parent-synthetic"}},
              {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "early", "output": "early result"}},
              {"type": "response_item", "payload": {"type": "function_call", "call_id": "paired", "name": "synthetic", "arguments": "{}"}},
              {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "paired", "output": "paired result"}},
              {"type": "response_item", "payload": {"type": "function_call", "call_id": "early", "name": "synthetic", "arguments": "{}"}}]
    write_jsonl(path, events)
    report = client.capture("session-a", transcript=path, transcript_root=tmp_path)
    assert report["coverage_complete"] and len(rows(client)) == 4
    status = client.call("lcm_status", {"session": "session-a"})
    assert status["tool_pairing"]["unmatched_results"] == 1
    assert status["tool_pairing"]["unresolved_calls"] == 1
    assert not status["native_context_engine"] and not status["native_host_qualified"]
    assert status["embeddings"] is False and status["scorer"] == "baseline"


def test_claude_tool_blocks_keep_call_identity():
    assistant = {"type": "assistant", "sessionId": "session-a", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "using tool"}, {"type": "tool_use", "id": "call-a", "name": "synthetic", "input": {}}]}}
    result = claude([{ "type": "tool_result", "tool_use_id": "call-a", "content": "result"}])
    calls, _ = project_event("claude", "session-a", assistant)
    results, _ = project_event("claude", "session-a", result)
    assert calls[0]["tool_calls"][0]["id"] == results[0]["tool_call_id"] == "call-a"
    assert results[0]["role"] == "tool" and results[0]["content"] == "result"


def test_portable_capture_never_dereferences_hermes_file_markers(tmp_path, monkeypatch):
    from hermes_lcm import ingest_protection
    def forbidden_read(*args, **kwargs):
        pytest.fail("portable ingestion attempted Hermes file recovery")
    monkeypatch.setattr(ingest_protection, "recover_hermes_persisted_output_with_file_stat", forbidden_read)
    marker = "<persisted-output>\nOutput too large (100 characters).\nFull output saved to: /synthetic/no-read.txt\n</persisted-output>"
    manual = portable(tmp_path)
    with pytest.raises(PortableError, match="unsupported_host_file_marker"):
        manual.ingest("session-a", event_id="a", messages=[{"role": "tool", "content": marker}])
    client = portable(tmp_path, "claude")
    path = tmp_path / "synthetic.jsonl"
    write_jsonl(path, [claude([{"type": "tool_result", "tool_use_id": "call-a", "content": marker}])])
    report = client.capture("session-a", transcript=path, transcript_root=tmp_path)
    assert report["unsupported_total"] == 1 and not report["coverage_complete"]
    assert rows(client) == []


def test_capsule_four_facets_bounded_exact_and_uncertain(tmp_path):
    client = portable(tmp_path)
    text = "Goal: finish synthetic work.\nMust preserve source.\nDecision: approved local test.\nPending: next step needs proof."
    client.ingest("session-a", event_id="a", messages=[{"role": "user", "content": text}])
    capsule = client.capsule("session-a")
    assert capsule["token_upper_bound"] <= 2000 and capsule["omitted_facets"] == []
    assert "Session: session-a" in capsule["content"] and "state_not_verified" in capsule["content"]
    for line in capsule["content"].splitlines():
        if line.startswith("source: "):
            page = client.call("lcm_expand", {"session": "session-a", "reference": line[8:]})
            assert page["content"] in text


def test_hooks_fail_open_and_emit_only_sessionstart_context(tmp_path):
    client = portable(tmp_path, "claude")
    path = tmp_path / "synthetic.jsonl"
    write_jsonl(path, [claude("Goal: keep native compaction.")])
    payload = {"session_id": "session-a", "transcript_path": str(path), "hook_event_name": "PreCompact"}
    output, receipt = client.hook(payload, transcript_root=tmp_path)
    assert output == {} and receipt["status"] == "observed"
    for count, event in enumerate(("UserPromptSubmit", "PostToolUse"), start=2):
        with path.open("a") as stream:
            stream.write(json.dumps(claude("incremental hook evidence", "incremental-" + event)) + "\n")
        payload["hook_event_name"] = event
        output, receipt = client.hook(payload, transcript_root=tmp_path)
        assert output == {} and receipt["status"] == "observed"
        assert len(rows(client)) == count
    payload["hook_event_name"] = "SessionStart"
    output, receipt = client.hook(payload, transcript_root=tmp_path)
    assert output["hookSpecificOutput"]["additionalContext"] and receipt["capsule_emitted"]
    assert "decision" not in output
    assert client.hook(payload, transcript_root=tmp_path, assistance=False)[0] == {}
    payload["session_id"] = "wrong-session"
    output, receipt = client.hook(payload, transcript_root=tmp_path)
    assert output == {} and receipt["status"] == "failed_open"
    status = client.call("lcm_status", {"session": "session-a"})
    assert status["hooks"]["SessionStart"]["qualified"] is False
    assert status["hooks"]["SessionStart"]["injection_confirmed"] is False


def test_stdio_protocol_read_only_tools_and_invalid_call(tmp_path):
    client = portable(tmp_path)
    requests = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "lcm_describe", "arguments": {}}},
                {"jsonrpc": "2.0", "id": 4, "method": "unknown"}]
    output = io.StringIO()
    serve(client, io.StringIO("".join(json.dumps(item) + "\n" for item in requests)), output)
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(replies) == 4 and replies[0]["result"]["protocolVersion"] == "2025-11-25"
    assert len(replies[1]["result"]["tools"]) == 4
    assert all(tool["annotations"]["readOnlyHint"] for tool in replies[1]["result"]["tools"])
    assert replies[2]["result"]["isError"] is True and replies[3]["error"]["code"] == -32601


def test_cli_normal_package_import_without_hermes_and_fail_open(tmp_path):
    # A fresh interpreter exercises actual __init__, which conftest does not.
    script = Path(__file__).resolve().parents[1] / "scripts" / "lcm_portable.py"
    command = [sys.executable, "-B", str(script), "--root", str(tmp_path / "corpora"),
               "--project", "synthetic-project", "--instance", "synthetic-instance", "--host", "claude",
               "hook", "--transcript-root", str(tmp_path)]
    result = subprocess.run(command, input="invalid json", text=True, capture_output=True, timeout=20)
    assert result.returncode == 0 and result.stdout == ""
    assert json.loads(result.stderr)["status"] == "failed_open"
    code = ("import importlib.util,sys;from pathlib import Path;"
            f"p=Path({str(script.parents[1])!r});"
            "s=importlib.util.spec_from_file_location('hermes_lcm',p/'__init__.py',submodule_search_locations=[str(p)]);"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "from hermes_lcm.portable import PortableRecall;assert 'agent.context_engine' not in sys.modules")
    imported = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True, timeout=20)
    assert imported.returncode == 0, imported.stderr
