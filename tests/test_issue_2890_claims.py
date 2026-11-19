"""Round 6 regressions for hidden input claims and externalized tool results."""
import json
import random
import time
from copy import deepcopy

import pytest

from hermes_lcm import escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import engine_context, host_call
from tests.lineage_guard_sessions import PAD, SESSION, simulate
from tests.test_issue_2890_lineage_guard import _content_digest, _covered_digests, _nodes, _stored, _summary


def assert_progress(record, leaves):
    detail = {key: record[key] for key in ("step", "leaves", "refused", "lineage_refusal", "noop")}
    assert not record["publication_error"], detail
    assert not (record["refused"] or record["lineage_refusal"]), detail
    assert record["leaves"] >= leaves, detail


@pytest.mark.issue(2890)
@pytest.mark.parametrize(("seed", "index", "leaves"), [
    (598, 6, 3), (130, 5, 1), (152, 9, 1), (143, 2, 12), (135, 6, 1), (145, 6, 1),
    (15, 2, 12),
])
def test_edited_claims(seed, index, leaves, tmp_path, monkeypatch):
    records = simulate(seed, tmp_path, monkeypatch, _summary, (_content_digest, _covered_digests, _nodes, _stored),
                       edits=True, stop_after=index + 1)
    assert_progress(records[index], leaves)


@pytest.mark.issue(2890)
@pytest.mark.parametrize(("seed", "turn", "leaves"), [(0, 26, 2), (1, 8, 1), (3, 21, 2), (7, 9, 1),
                                                     (20, 17, 1), (41, 40, 2), (95, 5, 2)])
def test_review_claims(seed, turn, leaves, tmp_path, monkeypatch):
    # Seeds from the reviewer (rv1), plus verify-2 seed 95 (all results name-only).
    externalized = seed == 95
    rng = random.Random(1000003 * seed + 77 if externalized else seed * 7919 + 13)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", _summary)
    stamp = clock = 1_900_000_000.0 if externalized else 1_800_000_000.0

    def now():
        nonlocal clock
        clock += 0.5
        return clock

    monkeypatch.setattr(time, "time", now)
    if externalized:
        window, tail, budget, big_p, style, restart_p = 16, 4, 2000, 0.3, "name", 0
        pad = " lorem ipsum dolor sit amet"
    else:
        window = rng.choice([16, 24, 32, 64, 128])
        tail = rng.choice([2, 4, 6, 8])
        budget = rng.choice([400, 800, 2000, 6000])
        big_p = rng.choice([0.1, 0.3, 0.6])
        style = rng.choice(["name", "none", "mixed"])
        restart_p = rng.choice([0.0, 0.1])
        pad = PAD

    def bind():
        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), embeddings_enabled=False,
                                           temporal_rollups_enabled=False, fresh_tail_count=tail,
                                           leaf_chunk_tokens=budget), hermes_home=str(tmp_path))
        engine.on_session_start(SESSION, platform="cli", context_length=window * 1024, conversation_id="conv")
        return engine

    view = [{"role": "system", "content": "You are a helpful agent."}] if not externalized and rng.random() < 0.6 else []

    def add(**row):
        nonlocal stamp
        stamp += 2
        view.append({**row, "timestamp": stamp})

    engine = bind()
    turns = 30 if externalized else rng.randint(20, 40)
    try:
        for n in range(1, turn + 1):
            add(role="user", content=f"question {n}" + pad * rng.randint(1, 25))
            if rng.random() < 0.6:
                count = rng.choice([1, 1, 2, 3])
                calls = [{"id": f"call_{n}_{i}", "type": "function",
                          "function": {"name": "read_file", "arguments": json.dumps({"path": f"/f{n}_{i}"})}}
                         for i in range(count)]
                calling = rng.choice(["", f"calling {n}"]) if externalized else ("" if rng.random() < 0.5 else f"calling {n}")
                add(role="assistant", content=calling, tool_calls=calls)
                for call in calls:
                    call_id = call["id"]
                    current_style = rng.choice(["name", "none", "both"]) if style == "mixed" else style
                    names = {"name": "read_file"} if current_style != "none" else {}
                    if current_style == "both":
                        names["tool_name"] = "read_file"
                    body = (f"BIG {call_id} " + "".join(f"line {j} of {call_id} data data data\n" for j in range(420))
                            if rng.random() < big_p else f"result {call_id}" + pad * rng.randint(0, 40))
                    add(role="tool", tool_call_id=call_id, content=body, **names)
            add(role="assistant", content=f"reply {n}" + pad * rng.randint(1, 60))
            engine.ingest(deepcopy(view))
            if not externalized and rng.random() < restart_p:
                engine.shutdown()
                engine = bind()
            if rng.random() < 0.35 or n == turns:
                nodes_before = len(_nodes(engine))
                forced = rng.choice([True, False]) if externalized else rng.random() < 0.5
                before = json.loads(json.dumps(view))
                view = (engine.compress(before, current_tokens=count_messages_tokens(before), focus_topic=None, force=True)
                        if forced else host_call(engine, before).view)
                assert engine._last_compression_status != "error", engine._last_compression_noop_reason
                if n == turn:
                    assert len(_nodes(engine)) - nodes_before >= leaves, engine._last_compression_noop_reason
                    return
                if rng.random() < 0.7:
                    engine.on_session_start(SESSION, old_session_id=SESSION, boundary_reason="compression",
                                            platform="cli", conversation_id="conv")
        pytest.fail(f"seed {seed} did not call compress on turn {turn}")
    finally:
        engine.shutdown()


@pytest.mark.issue(2890)
def test_orphan_id_is_scoped_to_bound_session(tmp_path):
    with engine_context(tmp_path, window=16_384, session_id=SESSION) as engine:
        engine._retained_anchor_orphan_id = 9
        engine.on_session_start(SESSION, old_session_id=SESSION, boundary_reason="compression", platform="cli")
        assert engine._retained_anchor_orphan_id == 9
        engine.on_session_start("new-session", platform="cli")
        assert engine.current_session_id == "new-session"
        assert engine._retained_anchor_orphan_id == 0
