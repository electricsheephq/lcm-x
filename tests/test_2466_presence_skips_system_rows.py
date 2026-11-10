"""verify-1: a rewritten system prompt in the same ingest as a class-A miss. Synthetic strings only."""
from __future__ import annotations

from collections import Counter

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("s-v1", platform="cli", context_length=1_000_000)
    return engine


def _m(role, text, ts=None):
    m = {"role": role, "content": text + PAD}
    if ts is not None:
        m["timestamp"] = float(ts)
    return m


def _roles(engine):
    rows = engine._store.get_session_messages("s-v1")
    return Counter((r["role"], str(r.get("content") or "")[:30]) for r in rows)


def test_rewritten_system_prompt_with_class_a(tmp_path):
    engine = _engine(tmp_path)
    try:
        sys0 = {"role": "system", "content": "You are a helpful agent."}
        view = [sys0, _m("user", "U1", 1), _m("assistant", "A1"), _m("user", "U2", 2), _m("assistant", "A2"),
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                {"role": "tool", "content": "r1", "tool_call_id": "c1"}, _m("assistant", "A3")]
        engine.ingest(view)
        changed = list(view)
        changed[0] = {"role": "system", "content": "You are a helpful agent. Today is Monday."}
        changed[1] = _m("user", "U1 replaced", 5)
        changed += [_m("user", "U3", 6), _m("assistant", "A4")]
        engine.ingest(changed)
        after = _roles(engine)
        again = changed + [_m("user", "U4", 7), _m("assistant", "A5")]
        engine.ingest(again)
        final = _roles(engine)
        systems = sum(v for k, v in final.items() if k[0] == "system")
        dup = {k: v for k, v in final.items() if v > 1}
        print(f"\nVERIFY1 systems_after_classA={sum(v for k, v in after.items() if k[0] == 'system')} "
              f"systems_final={systems} dups={len(dup)} total={sum(final.values())}")
        assert systems == 1, "rewritten system prompt stored by the presence check"
    finally:
        engine.shutdown()
