"""Inspector follow-ups on generated, closed copies; never live operator data."""

import copy
import json
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm import message_patterns
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import ScriptedSummarizer, make_engine
from tests.test_issue_1742_fix_r1 import PatternEngine, _operator_copy
from tests.test_issue_1742_fix_r2 import _reordered_store
from tests.test_issue_1742_fix_r3 import _inspect_result
from tests.test_issue_1742_inspector import ROOT, inspector
from tests.test_store_wedge_inspect import _copy, _inputs, _messages, counter  # noqa: F401


def test_unpersisted_ignored_message_matches_production(tmp_path, monkeypatch):
    monkeypatch.setattr(message_patterns, "_regex_engine", PatternEngine)
    ScriptedSummarizer().install(monkeypatch)
    view = [{"role": "user", "content": "IGNORE this prompt"}, *_messages(4)]
    config = dict(ignore_message_patterns=["^IGNORE"], leaf_chunk_tokens=1, fresh_tail_count=1,
                  fresh_tail_max_tokens=0)
    origin = tmp_path / "origin"
    engine = make_engine(origin, window=10000, session_id="S", conversation_id="conv", **config)
    try:
        engine.ingest(copy.deepcopy(view))
        assert len(engine._store.get_session_messages("S")) == len(view) - 1
    finally:
        engine.shutdown()
    path = tmp_path / "copy.db"
    _operator_copy(origin / "lcm.db", path)
    before = path.read_bytes()
    result = _inspect_result(path, view, **config)
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=10000, session_id="S", conversation_id="conv", **config)
    try:
        engine.compress(copy.deepcopy(view), current_tokens=count_messages_tokens(view), force=True)
        assert engine._last_compacted_store_id > 0
    finally:
        engine.shutdown()
    assert inspector.matches_message_pattern is message_patterns.matches_message_pattern
    assert result["view_state"] == "complete" and result["healthy"]
    assert result["unreachable_source_ids"] == [] and not result["selection_failed"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("enabled,tail", [(False, 1), (True, 0)])
def test_contradictory_pressure_yield_capture_is_config_incomplete(tmp_path, enabled, tail):
    view = _messages(3)
    path = _copy(tmp_path, view)
    result = _inspect_result(path, view, leaf_chunk_tokens=10000, pressure_yield_limit=1,
                             fresh_tail_count=tail, fresh_tail_pressure_yield_enabled=enabled)
    assert result.get("evidence_code") == "config_incomplete"
    assert result["view_state"] == "unknown" and not result["healthy"]
    assert result["prefix_over_leaf_budget"] is None and "selection_failed" not in result


def test_public_compress_arms_yield_after_hold_and_min_observations(tmp_path, monkeypatch):
    calls = ScriptedSummarizer().install(monkeypatch)
    config = dict(leaf_chunk_tokens=60, fresh_tail_count=6, fresh_tail_max_tokens=0,
                  fresh_tail_pressure_yield_min_observations=3, survival_fit=False)
    # Keep no-progress results unchanged so they arm the real hold; a survival
    # fit would shorten the view and avoid that hold before the next compress.
    window = 8500
    path, view = _reordered_store(tmp_path, (0, 1, 2, 3, 4, 5), config, repeat=300, window=window)
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=window, session_id="S", conversation_id="conv", **config)
    real_time, offset = engine_module.time, [0.0]
    clock = SimpleNamespace(**vars(real_time))
    clock.monotonic = lambda: real_time.monotonic() + offset[0]
    monkeypatch.setattr(engine_module, "time", clock)
    armed = []
    original = engine._maybe_engage_fresh_tail_pressure_yield

    def observe(*args, **kwargs):
        engaged = original(*args, **kwargs)
        if engaged:
            armed.append((engine._pressure_yield_blocked_streak,
                          engine._pressure_yield_tail_token_limit))
        return engaged

    monkeypatch.setattr(engine, "_maybe_engage_fresh_tail_pressure_yield", observe)
    try:
        observed = count_messages_tokens(view) + 200
        assert observed >= engine.threshold_tokens
        for attempt in range(1, 4):
            engine.compress(copy.deepcopy(view), current_tokens=observed, force=False)
            assert engine._pressure_yield_tail_token_limit == 0  # invocation scoped
            if attempt < 3:
                assert not armed and not calls.calls
                assert engine._pressure_yield_blocked_streak == attempt
                assert engine._last_compacted_store_id == 0
                assert engine._no_progress_hold_active()
                offset[0] += engine._no_progress_hold[0] - clock.monotonic() + 1
                assert not engine._no_progress_hold_active()
        assert len(armed) == 1 and armed[0][0] == 3 and armed[0][1] > 0
        assert engine._last_compacted_store_id > 0 and calls.calls
        inputs = _inputs(path, view, **{k: config[k] for k in
                         ("leaf_chunk_tokens", "fresh_tail_count", "fresh_tail_max_tokens")},
                         selection_budget=60, context_window=window,
                         effective_trigger=engine.threshold_tokens, pressure_yield_limit=armed[0][1])
        inputs["effective_config"]["lcm_config"] = asdict(engine._config)
        inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    finally:
        engine.shutdown()
    before = path.read_bytes()
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert result.get("evidence_code") is None
    assert result["view_state"] == "complete" and not result["selection_failed"]
    assert result["empty_input_kind"] == "none" and result["blocker_source_ids"] == []
    assert result["healthy"] and path.read_bytes() == before


@pytest.mark.parametrize("end,code", [(2**63 - 1, "unverified_ancestry"), (2**63, "invalid_input")])
def test_api_carry_end_has_sqlite_integer_bounds(tmp_path, capsys, end, code):
    path = _copy(tmp_path, _messages(2))
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO metadata(key,value) VALUES (?,?)",
                           ("identity_anchor_carry:S", json.dumps([["P", 0, end]])))
    before = path.read_bytes()
    with pytest.raises(inspector.InspectionRefused, match=f"^{code}$"):
        inspector.inspect_store(path, session_id="S", conversation_id="conv")
    assert path.read_bytes() == before
    assert inspector.main(["inspect", "--copy", str(path), "--session-id", "S",
                           "--conversation-id", "conv"]) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {"error": code} and output.err == ""


def test_foreign_lifecycle_predecessor_documentation_matches_production(tmp_path, monkeypatch):
    ScriptedSummarizer().install(monkeypatch)
    path = _copy(tmp_path, _messages(3))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET session_id='P' WHERE store_id IN (1,2)")
        connection.execute("INSERT INTO metadata(key,value) VALUES (?,?)",
                           ("rotation_predecessor_session:S", json.dumps("P")))
        connection.execute("INSERT INTO lcm_lifecycle_state(conversation_id,current_session_id) VALUES ('other','P')")
        connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                           "VALUES ('S','summary','[1,2]','messages',0)")
    before = path.read_bytes()
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv")
    assert (result["source_rows"], result["covered_rows"]) == (1, 2)
    assert path.read_bytes() == before
    product = tmp_path / "product"
    product.mkdir()
    _operator_copy(path, product / "lcm.db")
    engine = make_engine(product, window=10000, session_id="S", conversation_id="conv")
    try:
        assert engine._rotation_predecessor_session_ids("S") == ["P"]
        assert engine._store_complete_node_covered([1, 2]) == {1, 2}
    finally:
        engine.shutdown()
    doc = (ROOT / "docs/operations/store-wedge-inspect.md").read_text()
    assert "foreign to the conversation" not in doc
    assert "another conversation's lifecycle" in doc
