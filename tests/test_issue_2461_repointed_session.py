"""#2461: routing re-points preserve durable ownership (I1, I2)."""
from copy import deepcopy

import pytest

import hermes_lcm.engine as module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.engine_registry import resolve_active_lcm_engine, use_active_lcm_engine
from hermes_lcm.tokens import count_messages_tokens


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "summarize_with_escalation", lambda *a, **k: (
        "Synthetic summary of ordered turns.\nExpand for details about: turns", 1))
    engines = []

    def open_engine():
        engine = LCMEngine(config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2, leaf_chunk_tokens=100,
            large_output_externalization_path=str(tmp_path / "payloads")),
            hermes_home=str(tmp_path / "home"))
        engines.append(engine)
        return engine

    yield open_engine
    for engine in engines:
        engine.shutdown()


def start(engine, sid="S", cid="K1", **kwargs):
    engine.on_session_start(sid, conversation_id=cid, platform="cli", context_length=200_000, **kwargs)


def history(n=12):
    return [{"role": "user" if i % 2 == 0 else "assistant", "timestamp": float(i + 1),
             "content": f"synthetic turn {i} " + "alpha beta gamma delta " * 40} for i in range(n)]


def compact(engine, rows):
    output = engine.compress(deepcopy(rows), current_tokens=count_messages_tokens(rows), force=True)
    assert engine._last_compacted_store_id > 0
    assert engine._dag.get_session_nodes("S")
    return output


@pytest.mark.parametrize("restart", [False, True], ids=["resident", "reopened"])
@pytest.mark.parametrize("compacted", [False, True], ids=["T1", "T2"])
def test_repoint(host, compacted, restart):
    engine = host()
    start(engine)
    rows = history()
    engine.ingest(deepcopy(rows))
    replay = compact(engine, rows) if compacted else rows
    frontier = engine._last_compacted_store_id
    before = engine._store.get_session_messages("S")
    if restart:  # T3: derive ownership again from the reopened store.
        engine.shutdown()
        engine = host()
    start(engine, cid="K2")
    engine.ingest(deepcopy(replay + history(14)[12:]))
    stored = engine._store.get_session_messages("S")
    assert len(stored) == len(rows) + 2
    assert stored[:len(before)] == before
    assert {row["conversation_id"] for row in stored} == {"K1"}
    assert engine._lifecycle.get_by_conversation("K2") is None
    assert engine._last_compacted_store_id == frontier
    assert engine._store.get_range("S", conversation_id="K1") == stored
    assert engine._store.get_range("S", conversation_id="unrelated") == []
    assert resolve_active_lcm_engine(conversation_id="K2") is engine
    assert use_active_lcm_engine(lambda selected: selected, conversation_id="K2").value is engine


def test_T4_compression_handoff(host):
    engine = host()
    start(engine)
    rows = history()
    engine.ingest(deepcopy(rows))
    compact(engine, rows)
    nodes = [node.node_id for node in engine._dag.get_session_nodes("S")]
    start(engine, cid="K2")
    start(engine, sid="child", cid="K2", boundary_reason="compression", old_session_id="S")
    assert engine._conversation_id == "K1"
    assert [node.node_id for node in engine._dag.get_session_nodes("child")] == nodes
    assert engine._store.get_session_count("S") == len(rows)
    assert engine._store.get_session_count("child") == 0


@pytest.mark.parametrize("case", ["N1_separate", "N2_split", "N3_live"])
def test_controls(host, case):
    engine = host()
    start(engine)
    rows = history(2)
    engine.ingest(deepcopy(rows))
    if case == "N2_split":
        engine._store.append_batch("S", rows, conversation_id="K2")
    elif case == "N3_live":
        other = host()
        start(other, sid="other")
    before = engine._store.get_session_count("S")
    start(engine, sid="other" if case == "N1_separate" else "S", cid="K2")
    engine.ingest(deepcopy(history(4) if case == "N2_split" else history(4)[2:]))
    assert engine._conversation_id == "K2"
    assert engine._store.get_session_count("S") == before + (0 if case == "N1_separate" else 2)
    if case == "N3_live":
        assert engine._lifecycle.get_by_conversation("K1").current_session_id == "other"
        assert engine._ingest_ledger_counters.get("ledger.repoint_unadopted", 0) == 1
