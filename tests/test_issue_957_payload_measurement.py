"""#957: budget-only tail measurement must not externalize payloads."""

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import load_externalized_payload
from hermes_lcm.ingest_protection import (
    externalized_payload_stats,
    extract_all_externalized_payload_refs,
)


def test_repeated_uncapped_assembly_writes_only_emitted_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(engine_module, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)

    def no_model_calls(**kwargs):
        pytest.fail("tail measurement must not call a model")

    monkeypatch.setattr(engine_module, "summarize_with_escalation", no_model_calls)
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    home = str(tmp_path / "home")
    instance = LCMEngine(config=config, hermes_home=home)
    payload = "data:image/png;base64," + "QUJD" * 2_048
    original = engine_module._PRESERVED_OBJECTIVE_CONTEXT_PREFIX + "\nInspect " + payload
    tail = [{"role": "user", "content": original}]
    try:
        instance.on_session_start("issue-957-payload", context_length=128_000)
        assert config.max_assembly_tokens == 0 and config.survival_fit
        assert instance._effective_assembly_token_cap() is None
        node_id = instance._dag.add_node(SummaryNode(
            session_id=instance.current_session_id, depth=0,
            summary="Synthetic stored history.", token_count=10,
            source_token_count=20, source_ids=[], source_type="messages",
            created_at=1.0, expand_hint="synthetic history",
        ))
        counts = [externalized_payload_stats(config, home)["externalized_payload_count"]]
        for _ in range(2):
            assembled = instance._assemble_context(
                {"role": "system", "content": "Synthetic system prompt."}, tail,
                include_lcm_note=False,
            )
            assert any(f"node {node_id})" in row["content"] for row in assembled)
            refs = extract_all_externalized_payload_refs(assembled[-1]["content"])
            assert len(refs) == 1
            assert load_externalized_payload(refs[0], config=config, hermes_home=home)["content"] == payload
            counts.append(externalized_payload_stats(config, home)["externalized_payload_count"])
        print(f"payload counts (initial, assembly 1, assembly 2): {counts}")
        assert counts[0] == 0
        assert [after - before for before, after in zip(counts, counts[1:])] == [1, 1]
        assert tail == [{"role": "user", "content": original}]
        assert instance._dag.get_node(node_id) is not None
    finally:
        instance.shutdown()
