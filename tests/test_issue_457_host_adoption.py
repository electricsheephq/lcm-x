"""Existing worker reproduction extended through the real Hermes commit path."""
import json
import importlib.util
from pathlib import Path
import re
import socket
import sys
from copy import deepcopy
from types import SimpleNamespace

from agent import auxiliary_client as aux
from agent.conversation_compression import CompressionCommitFence
# Prefer the verified host's tools package over this plugin's tools.py.
sys.path.insert(0, str(Path(aux.__file__).resolve().parents[1]))
from hermes_state import SessionDB
from run_agent import AIAgent

_fixture_spec = importlib.util.spec_from_file_location(
    "issue457_fixture", Path(__file__).with_name("test_issue_457_normal_cancel.py")
)
_fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)
_engine = _fixture._engine


def test_cancel_cold_retry_commits_host_without_duplicate_end_flush(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-6-astra\n  provider: openai-codex\n"
        "context:\n  engine: compressor\ncompression:\n  in_place: true\n"
        "  context_timeout_seconds: 150\n  context_total_ceiling_seconds: 600\n"
    )
    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden in synthetic host acceptance")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    calls = []
    def provider(**kwargs):
        calls.append(1)
        match = re.search(r'<lcm-summary nonce="[0-9a-f]+">', kwargs["messages"][0]["content"])
        assert match
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=
            match.group(0) + "\nSynthetic retained decisions and constraints.\n"
            "Expand for details about: host cancellation test\n</lcm-summary>"))])
    monkeypatch.setattr(aux, "call_llm", provider)
    sid = "synthetic-cancel"
    db = SessionDB(home / "state.db")
    db.create_session(sid, source="operator-test", model="gpt-6-astra")
    for i in range(96):
        db.append_message(sid, role="user" if i % 2 == 0 else "assistant",
                          content=f"host turn {i}: " + "retained detail " * 80)
    def original_rows():
        return [tuple(row) for row in db._conn.execute(
            "SELECT id,role,content,tool_call_id,tool_calls FROM messages WHERE id<=96 ORDER BY id")]
    original = original_rows()
    old = db.get_messages_as_conversation(sid, include_row_ids=True, repair_alternation=True)
    original_input = deepcopy(old)
    engine = _engine(tmp_path, tail=24)
    def build_agent():
        a = AIAgent(model="gpt-6-astra", provider="openai-codex", api_mode="codex_responses",
                    base_url="https://chatgpt.com/backend-api/codex", api_key="synthetic-disabled",
                    enabled_toolsets=[], session_id=sid, session_db=db, quiet_mode=True,
                    skip_context_files=True, skip_memory=True, skip_background_review=True,
                    platform="telegram")
        a.context_compressor = engine
        a.compression_in_place = True
        a._compression_feasibility_checked = True
        a._print_fn = lambda *args, **kwargs: None
        return a
    try:
        agent = build_agent()
        engine.ingest(old)
        raw_before = engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        fence = CompressionCommitFence()
        compress = engine.compress
        def late_cancel(*args, **kwargs):
            result = compress(*args, **kwargs)
            assert fence.try_cancel_before_commit()
            return result
        engine.compress = late_cancel
        agent._compress_context(old, "Synthetic host recovery.", approx_tokens=90_000,
                                force=True, commit_fence=fence)
        assert old == original_input
        assert original_rows() == original
        assert len(db.get_messages_as_conversation(sid)) == 96
        assert len(calls) == 1
        engine.shutdown()
        db.close()
        db = SessionDB(home / "state.db")
        engine = _engine(tmp_path, tail=24)
        agent = build_agent()
        old = db.get_messages_as_conversation(sid, include_row_ids=True, repair_alternation=True)
        agent._compress_context(old, "Synthetic host recovery.", approx_tokens=90_000, force=True)
        active = db.get_messages_as_conversation(sid, include_row_ids=True, repair_alternation=True)
        count_after_commit = engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        receipt = {"in_place": bool(getattr(agent, "_last_compaction_in_place", False)),
                   "before_active": 96, "after_active": len(active), "provider_calls": len(calls),
                   "raw_before": raw_before, "raw_after_commit": count_after_commit,
                   "original_host_rows_preserved": original_rows() == original,
                   "tail_preserved": [engine._message_replay_identity(m) for m in active[-24:]] ==
                                     [engine._message_replay_identity(m) for m in original_input[-24:]]}
        engine.shutdown()
        engine = _engine(tmp_path, tail=24)
        engine.ingest(active)
        receipt["raw_after_cold_replay"] = engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        print("HOST_ADOPTION_RECEIPT=" + json.dumps(receipt, sort_keys=True))
        assert receipt["in_place"] and receipt["after_active"] < 96
        assert receipt["provider_calls"] == 1
        assert receipt["original_host_rows_preserved"] and receipt["tail_preserved"]
        assert receipt["raw_after_commit"] == receipt["raw_after_cold_replay"] == raw_before
        continued = active + deepcopy(original_input[-2:])
        engine.ingest(continued)
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == raw_before + 2
        engine.ingest(continued)
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == raw_before + 2
    finally:
        engine.shutdown()
        db.close()
