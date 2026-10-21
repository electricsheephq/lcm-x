"""Third-round parity regressions on generated stores and production copies."""

import copy
import sqlite3

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import ScriptedSummarizer, make_engine
from tests.test_compression_boundary import _stub_summarizer, _turn
from tests.test_issue_1742_fix_r1 import _operator_copy
from tests.test_issue_1742_fix_r2 import _inspect
from tests.test_issue_1742_inspector import inspector
from tests.test_store_wedge_inspect import _copy, counter  # noqa: F401


@pytest.mark.parametrize("repeat,yield_limit", [(500, 0), (400, 0), (400, 3000)])
def test_context_cap_admission_matches_production(tmp_path, monkeypatch, repeat, yield_limit):
    ScriptedSummarizer().install(monkeypatch)
    backlog = [{"role": "user" if i % 2 == 0 else "assistant",
                "content": f"private payload {i} " * repeat, "timestamp": 1000.0 + i}
               for i in range(10)]
    tail = [{"role": "user", "content": "tail user " * 20, "timestamp": 2000.0},
            {"role": "assistant", "content": "tail reply " * 20, "timestamp": 2001.0}]
    view = backlog + tail
    raw = count_messages_tokens(backlog)
    assert (20_000 <= raw < 24_000) if repeat == 500 else raw < 20_000
    config = dict(leaf_chunk_tokens=24000, fresh_tail_count=2, fresh_tail_max_tokens=3000)
    origin = tmp_path / "origin"
    engine = make_engine(origin, window=50000, session_id="S", conversation_id="conv", **config)
    try:
        engine.ingest(copy.deepcopy(view))
        threshold = engine.threshold_tokens
        cap = engine._working_leaf_chunk_tokens(raw)
        assert cap == 20000
    finally:
        engine.shutdown()
    path = tmp_path / "copy.db"
    _operator_copy(origin / "lcm.db", path)
    result = _inspect_result(path, view, **config, context_window=50000,
                             selection_budget=cap, effective_trigger=threshold,
                             pressure_yield_limit=yield_limit)
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=50000, session_id="S", conversation_id="conv", **config)
    try:
        # Capture the already-engaged waiver without asking compress to infer
        # its observation history. The ordinary arm exercises public compress.
        if yield_limit or raw < cap:
            engine._pressure_yield_tail_token_limit = yield_limit
            eligible, _, _ = engine._leaf_compaction_candidate_status_once(view)
            assert eligible == bool(yield_limit)
        else:
            engine.compress(copy.deepcopy(view), current_tokens=threshold + 1, force=False)
            assert engine._last_compacted_store_id > 0
    finally:
        engine.shutdown()
    if raw >= cap or yield_limit:
        assert result.get("evidence_code") is None
        assert result["view_state"] == "complete" and result["healthy"]
        assert result["leaf_budget"] == 24000
    else:
        assert result.get("evidence_code") == "leaf_admission_unassessed"


def _inspect_result(path, view, **config):
    # Unlike round two's helper, this also permits the declared below-cap refusal.
    from tests.test_store_wedge_inspect import _inputs
    inputs = _inputs(path, view, **config)
    inputs["effective_config"]["lcm_config"].update(
        {k: v for k, v in config.items() if k in inspector.LCMConfig.__dataclass_fields__})
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    before = path.read_bytes()
    siblings = set(path.parent.iterdir())
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert path.read_bytes() == before and set(path.parent.iterdir()) == siblings
    return result


def test_mapped_composite_replay_matches_production(tmp_path, monkeypatch):
    ScriptedSummarizer().install(monkeypatch)
    a, b = "alpha words here " * 4, "bravo words here " * 4
    view = [{"role": role, "content": text, "timestamp": 500.0 + i * 5}
            for i, (role, text) in enumerate([
                ("user", a), ("assistant", "reply one " * 6),
                ("user", b), ("assistant", "reply two " * 6),
                ("user", a + "\n\n" + b), ("assistant", "reply three " * 6),
                ("user", "tail question " * 6), ("assistant", "tail answer " * 6)])]
    path = _copy(tmp_path, copy.deepcopy(view))
    result = _inspect(path, view, leaf_chunk_tokens=60, selection_budget=200,
                      effective_trigger=10000, fresh_tail_count=2)
    assert result["view_state"] == "complete" and result["healthy"]
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=10000, session_id="S", conversation_id="conv",
                         leaf_chunk_tokens=60, fresh_tail_count=2, fresh_tail_max_tokens=0)
    try:
        replay = copy.deepcopy(view)
        assert len(engine._get_store_id_map_for_messages(replay)) == len(replay)
        before = engine._store._conn.execute("SELECT count(*) FROM message_relations").fetchone()[0]
        engine.compress(replay, current_tokens=count_messages_tokens(replay), force=True)
        assert engine._last_compacted_store_id > 0
        after = engine._store._conn.execute("SELECT count(*) FROM message_relations").fetchone()[0]
        assert before == after == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("donor", ["different_stamp", "foreign", "alias", "ancestor_unknown"])
def test_unmapped_composite_requires_scoped_same_stamp_donor(tmp_path, donor):
    view = [{"role": "user", "content": "R\n\nU", "timestamp": 500.0}]
    path = _copy(tmp_path, [{"role": "user", "content": "R", "timestamp": 510.0},
                            {"role": "user", "content": "U", "timestamp": 520.0}])
    with sqlite3.connect(path) as connection:
        if donor == "foreign":
            connection.execute("UPDATE messages SET conversation_id='other',observed_at=500 WHERE store_id=1")
        if donor == "alias":
            connection.execute("INSERT INTO message_relations "
                               "(store_id,kind,related_store_id,ordinal,observed_at,created_at) "
                               "VALUES (1,'alt_stamp',1,0,500,1)")
        if donor == "ancestor_unknown":
            connection.execute("UPDATE messages SET session_id='P',observed_at=500 WHERE store_id=1")
    result = _inspect_result(path, view, leaf_chunk_tokens=1)
    if donor == "alias":
        assert result.get("evidence_code") == "identity_anchor_unsupported"
    elif donor == "ancestor_unknown":
        assert result.get("evidence_code") == "unverified_ancestry"
    else:
        assert result.get("evidence_code") != "identity_anchor_unsupported"


@pytest.mark.parametrize("owner", [None, "", " \t ", "mixed"])
def test_blank_owned_rotation_predecessor(tmp_path, monkeypatch, owner):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())
    root = tmp_path / "origin"
    engine = make_engine(root, window=4000, session_id="S0", conversation_id="conv",
                         leaf_chunk_tokens=400, fresh_tail_count=6)
    host = []
    try:
        for turn in range(1, 13):
            user, reply = _turn(turn)
            host.append(user)
            engine.ingest(host)
            host.append(reply)
            engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        compacted = engine.compress(list(host), force=True)
        engine.on_session_end("S0", host)
        engine.on_session_start("S1", boundary_reason="compression", old_session_id="S0",
                                platform="acp", context_length=4000)
        assert engine._last_compacted_store_id == 19
    finally:
        engine.shutdown()
    path = tmp_path / "copy.db"
    _operator_copy(root / "lcm.db", path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET conversation_id=? WHERE session_id='S0'",
                           (None if owner == "mixed" else owner,))
        if owner == "mixed":
            connection.execute("UPDATE messages SET conversation_id='conv' WHERE store_id=1")
    before = path.read_bytes()
    result = inspector.inspect_store(path, session_id="S1", conversation_id="conv")
    assert path.read_bytes() == before
    assert (result["covered_rows"], result["frontier"]) == (19, 19)
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=4000, session_id="S1", conversation_id="conv",
                         leaf_chunk_tokens=400, fresh_tail_count=6)
    try:
        assert engine._rotation_predecessor_session_ids("S1") == ["S0"]
        view = list(compacted)
        for turn in range(14, 26):
            view.extend(_turn(turn))
        engine.ingest(view)
        engine.compress(view, force=True)
        assert engine._last_compacted_store_id > 19
    finally:
        engine.shutdown()
