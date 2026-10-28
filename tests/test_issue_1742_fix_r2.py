"""Final-round inspector parity probes, using generated stores only."""

import copy
import json
import sqlite3

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import make_engine
from tests.harness.provider import ScriptedSummarizer
from tests.test_compression_boundary import _stub_summarizer, _turn
from tests.test_issue_1742_fix_r1 import CALL, _operator_copy
from tests.test_issue_1742_inspector import inspector
from tests.test_store_wedge_inspect import _copy, _inputs, _messages, counter  # noqa: F401


def _inspect(path, view, **config):
    inputs = _inputs(path, view, **config)
    inputs["effective_config"]["lcm_config"].update(
        {key: value for key, value in config.items() if key in LCMConfig.__dataclass_fields__})
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    before = path.read_bytes()
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert path.read_bytes() == before
    assert result.get("evidence_code") is None
    assert "private payload" not in json.dumps(result)
    return result


def test_empty_rotation_hop_matches_inplace(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())
    findings = []
    for rotated in (False, True):
        root = tmp_path / ("rotation" if rotated else "inplace")
        root.mkdir()
        engine = LCMEngine(
            config=LCMConfig(database_path=str(root / "lcm.db"), fresh_tail_count=6,
                             leaf_chunk_tokens=400, large_output_externalization_path=str(root / "x")),
            hermes_home=str(root / "home"),
        )
        s0, s1, s2 = ("S0", "S1", "S2") if rotated else ("S0", "S0", "S0")
        engine.on_session_start(s0, platform="acp", context_length=4000)
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
            first = engine.compress(list(host), force=True)
            engine.on_session_end(s0, host)
            engine.on_session_start(s1, boundary_reason="compression", old_session_id=s0,
                                    platform="acp", context_length=4000)
            second = engine.compress(list(first), current_tokens=7000, bypass_cooldown=True)
            assert second != first
            engine.on_session_end(s1, first)
            engine.on_session_start(s2, boundary_reason="compression", old_session_id=s1,
                                    platform="acp", context_length=4000)
            engine.ingest(list(second) + [_turn(13)[1]])
            if rotated:
                assert engine._rotation_predecessor_session_ids(s2) == [s1, s0]
        finally:
            engine.shutdown()
        path = root / "copy.db"
        _operator_copy(root / "lcm.db", path)
        with sqlite3.connect(path) as connection:
            conversation, = connection.execute(
                "SELECT conversation_id FROM lcm_lifecycle_state WHERE current_session_id=?", (s2,)).fetchone()
            if rotated:
                assert connection.execute("SELECT count(*) FROM messages WHERE session_id=?", (s1,)).fetchone()[0] == 0
                for child, parent in ((s1, s0), (s2, s1)):
                    assert connection.execute("SELECT value FROM metadata WHERE key=?",
                                              (f"rotation_predecessor_session:{child}",)).fetchone()[0] == json.dumps(parent)
        before = path.read_bytes()
        result = inspector.inspect_store(path, session_id=s2, conversation_id=conversation)
        assert path.read_bytes() == before
        findings.append((result["covered_rows"], result["frontier"]))
    assert findings == [(19, 19), (19, 19)]


def _reordered_store(tmp_path, order, config, *, repeat=8, window=10000):
    messages = [{**m, "content": f"private payload {i} " * repeat, "timestamp": 1000.0 + i}
                for i, m in enumerate(_messages(6))]
    view = [copy.deepcopy(messages[i]) for i in order]
    root = tmp_path / "origin"
    engine = make_engine(root, window=window, session_id="S", conversation_id="conv", **config)
    try:
        engine.ingest(copy.deepcopy(messages))
        engine.ingest(copy.deepcopy(view))
    finally:
        engine.shutdown()
    path = tmp_path / "copy.db"
    _operator_copy(root / "lcm.db", path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 6
    return path, view


@pytest.mark.parametrize("order", [(1, 0, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5), (2, 0, 1, 3, 4, 5),
                                   (0, 1, 3, 2, 4, 5), (3, 0, 1, 2, 4, 5), (0, 1, 2, 3, 4, 5)])
def test_shown_retained_backlog(tmp_path, monkeypatch, order):
    ScriptedSummarizer().install(monkeypatch)
    config = dict(leaf_chunk_tokens=60, fresh_tail_count=6, fresh_tail_max_tokens=0)
    path, view = _reordered_store(tmp_path, order, config)
    result = _inspect(path, view, **config)
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=10000, session_id="S", conversation_id="conv", **config)
    blocked = order.index(0) != 0
    try:
        backlog = engine._store_complete_backlog(view, 0)
        assert tuple(backlog) == (order[0], False)
        engine.compress(copy.deepcopy(view), current_tokens=count_messages_tokens(view), force=True)
        reason = engine._last_compression_noop_reason or ""
        assert "unresolved retained occurrence" not in reason  # #1162: product escapes; inspector still reports the stop.
        assert (engine._last_compacted_store_id > 0) == blocked
    finally:
        engine.shutdown()
    assert result["empty_input_kind"] == ("retained" if blocked else "none")
    assert result["blocker_source_ids"] == ([1] if blocked else [])
    assert result["blocker_in_fresh_tail"] == blocked
    assert result["selection_failed"] == blocked
    assert result["healthy"] == (not blocked)


@pytest.mark.parametrize("order", [(1, 0, 2, 3, 4, 5), (3, 0, 1, 2, 4, 5)])
@pytest.mark.parametrize("min_observations", [3, 1000])
@pytest.mark.parametrize("window", [8500, 9000])
def test_shown_retained_nonforced(tmp_path, monkeypatch, order, min_observations, window):
    calls = ScriptedSummarizer().install(monkeypatch)
    config = dict(leaf_chunk_tokens=60, fresh_tail_count=6, fresh_tail_max_tokens=0,
                  fresh_tail_pressure_yield_min_observations=min_observations)
    path, view = _reordered_store(tmp_path, order, config, repeat=300, window=window)
    result = _inspect(path, view, **config)
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=window, session_id="S", conversation_id="conv", **config)
    try:
        observed = count_messages_tokens(view) + 200
        assert observed >= engine.threshold_tokens
        # The low-observation arm may yield its tail on pass three. The captured
        # snapshot describes the initial stop, before that separate host state.
        for attempt in range(5 if min_observations == 1000 else 2):
            engine.compress(copy.deepcopy(view), current_tokens=observed, force=False)
            assert "unresolved retained occurrence" not in engine._last_compression_noop_reason  # #1162: escape progresses.
            assert engine._last_compacted_store_id >= 1
        assert calls.calls  # #1162: the stored retained row is summarized instead of repeating the zero-leaf stop.
    finally:
        engine.shutdown()
    assert result["empty_input_kind"] == "retained"
    assert result["blocker_source_ids"] == [1] and result["blocker_in_fresh_tail"]
    assert result["selection_failed"] and not result["healthy"]


def test_attribution_budget_floor(tmp_path, monkeypatch):
    ScriptedSummarizer().install(monkeypatch)
    config = dict(leaf_chunk_tokens=60, fresh_tail_count=4, fresh_tail_max_tokens=0)
    path, view = _reordered_store(tmp_path, (1, 0, 2, 3, 4, 5), config)
    result = _inspect(path, view, **config)
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=10000, session_id="S", conversation_id="conv", **config)
    try:
        chunk = view[:2]
        assert count_messages_tokens(chunk) > 60
        mapping = engine._get_store_id_map_for_messages(view)
        attributed = engine._identity_anchor_summary_input(chunk, mapping, view, chunk, budget=60)
        assert [ids for _message, ids in attributed] == [[], [1]]
        engine.compress(copy.deepcopy(view), current_tokens=count_messages_tokens(view), force=True)
        assert engine._last_compacted_store_id == 2
    finally:
        engine.shutdown()
    assert result["empty_input_kind"] == "none"
    assert not result["selection_failed"] and result["healthy"]


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("filler", [10, 2100])
def test_copy_subtraction_requires_complete_scan(tmp_path, monkeypatch, filler, force):
    ScriptedSummarizer().install(monkeypatch)
    messages = [{"role": "user", "content": "q1 question"},
                {"role": "assistant", "content": "", "tool_calls": CALL},
                {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "RESULT body"},
                {"role": "user", "content": "copy slot 4"},
                {"role": "assistant", "content": "copy slot 5"}]
    messages += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"filler {i}"}
                 for i in range(filler)]
    tail = [{"role": "user", "content": "tail user " * 10}, {"role": "assistant", "content": "tail reply " * 10}]
    path = _copy(tmp_path, messages + tail)
    with sqlite3.connect(path) as connection:
        for dst, src in ((4, 2), (5, 3)):
            connection.execute("UPDATE messages SET (role,content,tool_call_id,tool_calls,tool_name)="
                               "(SELECT role,content,tool_call_id,tool_calls,tool_name FROM messages WHERE store_id=?) "
                               "WHERE store_id=?", (src, dst))
        for ids in ([1, 2, 3], list(range(6, 6 + filler))):
            connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                               "VALUES ('S','summary',?,'messages',0)", (json.dumps(ids),))
    result = _inspect(path, tail, leaf_chunk_tokens=60, fresh_tail_count=2, effective_trigger=20)
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=10000, session_id="S", conversation_id="conv", leaf_chunk_tokens=60,
                         fresh_tail_count=2, fresh_tail_max_tokens=0)
    truncated = filler == 2100
    try:
        assert tuple(engine._store_complete_backlog(tail, 0)) == (2 if truncated else 0, truncated)
        engine.compress(copy.deepcopy(tail), current_tokens=count_messages_tokens(tail), force=force)
        assert (engine._last_compression_status == "compacted") == truncated
    finally:
        engine.shutdown()
    with sqlite3.connect(product / "lcm.db") as connection:
        sources = [json.loads(r[0]) for r in connection.execute("SELECT source_ids FROM summary_nodes ORDER BY node_id")]
        assert len(sources) == (3 if truncated else 2)
        if truncated:
            assert sources[-1] == [4, 5]
    assert result["empty_input_kind"] == "none"
    assert result["selection_failed"] == truncated
    assert not result["healthy"]
