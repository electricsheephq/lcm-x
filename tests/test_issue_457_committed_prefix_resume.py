"""#457 under DEFAULT LCMConfig tuning: a late cancel after the leaf commit, then retry.

The host discards a cancelled compress() result and replays its own copy. The
retry must resume from the committed frontier: no provider call, no publication
conflict, and the committed summary plus the protected tail come back.
"""
import json
import logging
import re
import socket
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest

aux = pytest.importorskip("agent.auxiliary_client", reason="requires real Hermes worker hooks")
_host = pytest.importorskip("agent.conversation_compression")
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

SID, CID = "issue-457", "issue-457-conv"
SUMMARY_HEADER = "Summary (d0, node "  # engine._assemble_context's DAG summary header


def _engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "t457.db")),
                       hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SID, platform="cli", conversation_id=CID, context_length=200_000)
    return engine


def _transcript():
    plain, i = [], 0
    while len(plain) < 40 or count_messages_tokens(plain[:-30]) < 24_000:
        plain.append({"role": "user" if i % 2 == 0 else "assistant",
                      "content": f"owned turn {i} " + " ".join(f"w{i}x{j}" for j in range(120))})
        i += 1
    return plain + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
         "function": {"name": "synthetic", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "synthetic result"},
    ]


def _tail_kept(out, tail):
    """Every tail row's content and tool calls survive, in order, in the returned rows."""
    text = "\n".join(str(m.get("content") or "") for m in out)
    calls = [c for m in out for c in (m.get("tool_calls") or [])]
    pos = 0
    for row in tail:
        if row.get("content"):
            pos = text.find(str(row["content"]), pos)
            if pos < 0:
                return False
        if any(call not in calls for call in row.get("tool_calls") or []):
            return False
    return True


@pytest.mark.parametrize("restart", [False, True], ids=["same-engine", "cold-engine"])
def test_default_tuning_late_cancel_then_retry_resumes(tmp_path, monkeypatch, caplog, restart):
    def deny(*a, **k):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", deny)
    calls = []

    def provider(**kwargs):
        calls.append(1)
        m = re.search(r'<lcm-summary nonce="[0-9a-f]+">', str(kwargs["messages"][0].get("content") or ""))
        head = m.group(0) if m else "<lcm-summary>"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=(
            f"{head}\nDurable synthetic summary of ordered decisions and active work.\n"
            "Expand for details about: triage\n</lcm-summary>")))])

    monkeypatch.setattr(aux, "call_llm", provider)
    resumes = []
    real_prefix_len = LCMEngine._committed_replay_prefix_len

    def spy_prefix_len(self, working, start):
        found = real_prefix_len(self, working, start)
        resumes.append({"k": found[0], "start": start, "v4": self._compress_occurrences is not None})
        return found

    monkeypatch.setattr(LCMEngine, "_committed_replay_prefix_len", spy_prefix_len)
    messages = _transcript()
    original = deepcopy(messages)
    engine = _engine(tmp_path)
    leaves = lambda: [n for n in engine._dag.get_session_nodes(SID) if n.source_type == "messages"]
    frontier = lambda: engine._lifecycle.get_by_conversation(CID).current_frontier_store_id

    def dispatch(fn, fence, event, gen):
        return _host._run_summary_dispatch(
            SimpleNamespace(context_compressor=engine, session_id=engine._session_id), messages, fn,
            {"current_tokens": count_messages_tokens(messages), "force": True},
            commit_fence=fence, attempt_generation=gen, hard_cancel_event=event)

    fence, stop = _host.CompressionCommitFence(), Event()

    committed = {}

    def cancel_after_compress(*a, **k):
        result = committed["out"] = deepcopy(engine.compress(*a, **k))
        assert fence.try_cancel_before_commit() is True
        stop.set()
        return result

    r = {"input_rows": len(messages)}
    try:
        engine.ingest(messages)
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            dispatch(cancel_after_compress, fence, stop, 1)
        r.update(first_attempt_calls=len(calls), leaves_after_cancel=len(leaves()),
                 frontier_after_cancel=frontier())
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path)
        caplog.set_level(logging.WARNING)
        outs = {}
        for n in (1, 2):
            resumes.clear()
            before = len(calls)
            outs[n] = out = dispatch(engine.compress, _host.CompressionCommitFence(), None, 1 + n)
            r[f"retry{n}"] = dict(
                provider_calls=len(calls) - before, returned_rows=len(out),
                summary_rows=sum(SUMMARY_HEADER in str(m.get("content") or "") for m in out),
                same_as_committed_output=out == committed["out"], tail_kept=_tail_kept(out, original[-32:]),
                frontier=frontier(), leaves=len(leaves()), store_rows=engine._store.get_session_count(SID),
                status=engine._last_compression_status, noop=engine._last_compression_noop_reason,
                resumes=list(resumes))
        r["conflict_log_lines"] = sum("publication_invariant_conflict" in rec.getMessage()
                                      for rec in caplog.records)
        r["input_preserved"] = messages == original
        print("ISSUE457_DEFAULTS=" + json.dumps(r, sort_keys=True, default=str))
        assert r["first_attempt_calls"] == 1
        assert r["leaves_after_cancel"] == 1 and r["frontier_after_cancel"] == 110
        for n in (1, 2):
            got = r[f"retry{n}"]
            assert got["provider_calls"] == 0, "FREEZE: the retry re-summarized committed rows"
            assert [(x["k"], x["start"]) for x in got["resumes"]] == [(110, 0)]
            assert got["status"] == "compacted"
            assert (got["frontier"], got["leaves"], got["store_rows"]) == (110, 1, 142)
            # Output = the committed attempt's own output: the summary + the whole 32-row
            # protected tail (the assembler folds the summary into the leading user turn and
            # the tool-call assistant into the assistant turn before it: 31 rows).
            assert got["same_as_committed_output"] and got["tail_kept"]
            assert got["returned_rows"] == 31 and got["summary_rows"] == 1
            assert SUMMARY_HEADER in str(outs[n][0].get("content") or "")
        assert r["conflict_log_lines"] == 0
        assert r["input_preserved"]
    finally:
        engine.shutdown()
