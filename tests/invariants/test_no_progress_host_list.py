"""#1880: automatic representation-only output preserves the host snapshot (I1/I3/I8)."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as lcm_engine
from tests.harness import engine_context, pin_char_counter


@pytest.mark.parametrize("kind", ["think", "internal", "redaction", "quarantine", "ignored", "stub",
                                 "recall", "summary", "old-summary", "missing-stub", "projection", "objective-prefix", "recovery-prefix", "repair", "drop", "merge", "leaf", "progress",
                                 "stub-first", "force", "bypass", "overflow", "fit"])
def test_no_progress_host_contract(tmp_path, monkeypatch, kind):
    if kind == "recall" and not hasattr(lcm_engine, "_PROACTIVE_RECALL_HEADER"):
        pytest.skip("v0.29.1 has no proactive recall header")
    pin_char_counter(monkeypatch)
    with engine_context(tmp_path, window=200 if kind == "fit" else 200_000, conversation_id="conv", survival_fit=kind == "fit") as engine:
        host = [{"role": "user", "content": "question " * 100},
                {"role": "assistant", "content": "<think>internal</think>answer " * 100},
                {"role": "user", "content": "latest question"}, {"role": "assistant", "content": "latest answer"}]
        if kind == "recall":
            host[1]["content"] = "<relevant-memories>\n" + lcm_engine._PROACTIVE_RECALL_HEADER + " old memory"
        if kind == "internal":
            host[1]["content"] = [{"type": "thinking", "thinking": "internal"}, {"type": "text", "text": "answer"}]
        if kind == "old-summary":
            host[1]["content"] = "CONTEXT SUMMARY old"
        if kind == "missing-stub":
            host[1]["role"], host[1]["tool_call_id"] = "tool", "earlier-result"
        snapshot = deepcopy(host)
        emitted = engine._sanitize_active_context_messages(host)
        contents = {"redaction": "[LCM sensitive redaction: synthetic]", "quarantine":
                    "[LCM active replay placeholder: assistant output quarantined; synthetic]",
                    "ignored": "[LCM active replay placeholder: message ignored; synthetic]",
                    "stub": "[Externalized tool output: synthetic]", "summary": "CONTEXT SUMMARY new",
                    "old-summary": "new representation",
                    "missing-stub": engine._missing_tool_result_stub("")["content"],
                    "projection": "[LCM survival fit: this assistant message (900 tokens) is stored verbatim as store id 1; "
                                  "lcm_expand / lcm_grep reach the full text. (projection 1200/600)]",
                    "objective-prefix": "[Current user objective preserved from compacted history]\nobjective",
                    "recovery-prefix": "[LCM overflow recovery] synthetic"}
        if kind in contents:
            emitted[1] = {**emitted[1], "content": contents[kind]}
        if kind == "recall":
            emitted[1] = {**emitted[1], "content": host[1]["content"] + " changed"}
        if kind == "repair":
            emitted[1] = {**emitted[1], "tool_calls": [{"id": "repair"}]}
        if kind in {"drop", "merge"}:
            emitted.pop(1)
        def impl(messages, **kwargs):
            engine.ingest(messages)
            engine._ingest_cursor = len(emitted)
            engine._last_compression_status = "sanitized"
            engine._foreground_budget.leaves = int(kind == "leaf")
            engine._compress_host_rows_consumed = 1
            engine._foreground_budget.progress = "condensation" if kind == "progress" else ""
            engine._stub_first_exit_now = {} if kind == "stub-first" else None
            engine._compress_forced_overflow = kind == "overflow"
            return emitted
        monkeypatch.setattr(engine, "_compress_impl", impl)
        fit = Mock(wraps=engine._survival_fit)
        monkeypatch.setattr(engine, "_survival_fit", fit)
        result = engine.compress(host, force=kind == "force", bypass_cooldown=kind == "bypass")
        assert host == snapshot
        if kind in {"think", "internal", "fit"}:
            assert fit.call_args.args[1] is host
            if kind == "fit":
                assert len(result) < len(host)
            else:
                assert result is host and result == snapshot
                assert engine._compress_commit_proof is None and engine._compress_commit_reason == "identity_noop"
        else:
            assert result is emitted
