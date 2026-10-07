"""#957 r4: reserve repair stubs without stripping ordinary tail content."""

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def test_default_prefix_bound_counts_ordinary_tagged_tail_and_repair_stubs(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(engine_module, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)

    def no_model_calls(**kwargs):
        pytest.fail("tail measurement must not call a model")

    monkeypatch.setattr(engine_module, "summarize_with_escalation", no_model_calls)
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), survival_fit=False)
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    original = "Discuss this literal block: <relevant-memories>" + "abcd" * 1_000 + "</relevant-memories>"
    calls = [{"id": f"missing-{index}", "type": "function",
              "function": {"name": "read_file", "arguments": "{}"}} for index in range(30)]
    tail = [{"role": "user", "content": original},
            {"role": "assistant", "content": "Running tools.", "tool_calls": calls}]
    system = {"role": "system", "content": "Synthetic system prompt."}
    try:
        instance.on_session_start("issue-957-r4", context_length=4_000)
        assert config.max_assembly_tokens == 0 and not config.survival_fit
        assert instance._effective_assembly_token_cap() is None
        node_ids = [instance._dag.add_node(SummaryNode(
            session_id=instance.current_session_id, depth=1,
            summary=f"Stored history {index}: " + "abcd" * 190, token_count=200,
            source_token_count=400, source_ids=[], source_type="nodes",
            created_at=float(index + 1), expand_hint="synthetic history",
        )) for index in range(20)]
        assembled = instance._assemble_context(system, tail, include_lcm_note=False)
        assert any("Summary (d1, node " in row["content"] for row in assembled)
        assert any(row.get("role") == "user" and row["content"] == original for row in assembled)
        assert next(row for row in assembled if row.get("tool_calls"))["tool_calls"] == calls
        assert [row["tool_call_id"] for row in assembled if row.get("role") == "tool"] == [
            call["id"] for call in calls
        ]
        assert tail[0]["content"] == original
        assert all(instance._dag.get_node(node_id) is not None for node_id in node_ids)
        overhead = instance._survival_host_overhead(
            [system, *tail], max(int(instance.last_prompt_tokens or 0), instance._last_gate_tokens),
        )
        ceiling = instance._survival_ceiling()
        emitted_tokens = tokens.count_messages_tokens(assembled)
        print(f"emitted tokens: {emitted_tokens}; host overhead: {overhead}; ceiling: {ceiling}")
        assert emitted_tokens + overhead <= ceiling
    finally:
        instance.shutdown()
