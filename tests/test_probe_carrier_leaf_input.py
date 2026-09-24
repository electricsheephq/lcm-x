"""Regression for compacted carrier prefixes entering leaf summarization."""

import re

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _turn(i):
    return (
        {"role": "user", "content": f"[T{i:02d}] user turn {i}: " + ("alpha beta gamma delta " * 40)},
        {"role": "assistant", "content": f"reply to T{i:02d}: noted item {i}."},
    )


def test_carrier_prefix_never_reaches_leaf_summarizer(tmp_path, monkeypatch):
    seen = []
    calls = {"n": 0}

    def summarize(*args, **kwargs):
        calls["n"] += 1
        text = kwargs.get("text") if "text" in kwargs else args[0]
        seen.append(text)
        tags = sorted(set(re.findall(r"\[T(\d\d)\] user", text or "")))
        return (f"UNIQUE-SUMMARY-{calls['n']} covers " + " ".join(tags) + ".", 1)

    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", summarize)
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=3,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host = []
        for i in range(1, 9):
            host.extend(_turn(i))
            engine.ingest(host)
        host.append(_turn(9)[0])
        compressed = engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"
        carrier = compressed[0]
        remainder = engine._generated_context_carrier_remainder(carrier)
        assert remainder is not None
        assert "UNIQUE-SUMMARY-1" in carrier["content"]
        carrier_source_ids = set(engine._get_store_ids_for_messages([carrier]))
        assert carrier_source_ids

        host2 = list(compressed)
        host2.append(_turn(9)[1])
        for i in range(10, 16):
            host2.extend(_turn(i))
        host2.append(_turn(16)[0])
        engine.ingest(host2)
        existing_node_ids = {node.node_id for node in engine._dag.get_session_nodes("S0")}
        before = len(seen)
        engine.compress(list(host2), force=True)
        new_inputs = seen[before:]
        new_nodes = [
            node
            for node in engine._dag.get_session_nodes("S0")
            if node.node_id not in existing_node_ids
        ]

        assert engine._last_compression_status == "compacted"
        assert new_inputs
        assert all("Summary (d" not in text for text in new_inputs)
        assert all("UNIQUE-SUMMARY-1" not in text for text in new_inputs)
        assert carrier_source_ids <= {
            source_id for node in new_nodes for source_id in node.source_ids
        }
    finally:
        engine.shutdown()
