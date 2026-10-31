"""Review-round regressions adapted from the #1752 generated production probes."""

import copy
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hermes_lcm import message_patterns
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import make_engine
from tests.harness.provider import ScriptedSummarizer
from tests.test_compression_boundary import _assert_clean_commit_sequence, _run_host_commit_sequence
from tests.test_issue_1742_inspector import inspector
from tests.test_store_wedge_inspect import _copy, _inputs, _messages, counter  # noqa: F401


class Pattern:
    def __init__(self, pattern):
        self.pattern = pattern
        self.compiled = re.compile(pattern)

    def search(self, text, *, timeout):
        return self.compiled.search(text)


class PatternEngine:
    error = re.error
    compile = Pattern


def _operator_copy(source, target):
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(target)) as destination:
            original.backup(destination)
            assert destination.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    assert not any(Path(str(target) + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))


def test_empty_lineage_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(message_patterns, "_regex_engine", PatternEngine)
    ScriptedSummarizer().install(monkeypatch)
    origin = tmp_path / "origin"
    origin.mkdir()
    engine = make_engine(origin, window=200000, session_id="S", conversation_id="conv",
                         ignore_message_patterns=["SECRET"])
    messages = [{"role": "user", "content": "SECRET cron prompt"}]
    messages += [{"role": "assistant", "content": f"dependent {i} " + "d" * 11000} for i in range(20)]
    messages += [{"role": "user", "content": "visible objective " + "y" * 200}]
    messages += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"tail {i} " + "t" * 400}
                 for i in range(30)]
    try:
        engine.ingest(messages)
        engine.compress(messages, current_tokens=count_messages_tokens(messages), force=True)
        frontier = engine._last_compacted_store_id
        assert frontier > 0
    finally:
        engine.shutdown()
    path = tmp_path / "copy.db"
    _operator_copy(origin / "lcm.db", path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM summary_nodes WHERE source_type='messages' "
                                  "AND source_ids='[]'").fetchone()[0] > 0
    before = path.read_bytes()
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv")
    assert result["covered_rows"] == 0 and result["frontier"] == frontier
    assert path.read_bytes() == before


CALL = [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]


def _e1_copy(tmp_path):
    big, small = " filler" * 60, " f" * 4
    messages = [
        {"role": "user", "content": "q1" + small},
        {"role": "assistant", "content": "", "tool_calls": CALL},
        {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "RESULT" + big},
        {"role": "user", "content": "q2" + small},
        {"role": "assistant", "content": "x5" + small},
        {"role": "user", "content": "x6" + small},
        {"role": "user", "content": "h7" + small},
        {"role": "user", "content": "q8" + small},
        {"role": "assistant", "content": "a9" + big},
        {"role": "user", "content": "q10" + big},
    ]
    path = _copy(tmp_path, messages, frontier=4)
    with sqlite3.connect(path) as connection:
        for dst, src in ((5, 2), (6, 3)):
            connection.execute("UPDATE messages SET (role,content,tool_call_id,tool_calls,tool_name)="
                               "(SELECT role,content,tool_call_id,tool_calls,tool_name FROM messages WHERE store_id=?) "
                               "WHERE store_id=?", (src, dst))
        connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                           "VALUES ('S','summary','[1,2,3,4]','messages',0)")
    return path, messages


@pytest.mark.parametrize("empty_bindings", [False, True], ids=["absent", "empty"])
def test_e1_prefix(tmp_path, empty_bindings):
    path, messages = _e1_copy(tmp_path)
    if empty_bindings:
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE host_uid_bindings "
                               "(store_id INTEGER,uid TEXT,lineage_key TEXT,binding_version INTEGER,kind TEXT)")
    inputs = _inputs(path, [messages[8], messages[9], messages[7]],
                     leaf_chunk_tokens=60, selection_budget=2000, fresh_tail_count=1)
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **copy.deepcopy(inputs))
    # Frozen oracle: covered tool twins 5/6 are excluded; the prefix contains
    # hidden row 7 and retained row 8 (14 fallback tokens). Do not call the
    # inspector a second time through the same inherited helper as an oracle.
    assert result == dict(
        polluted_copies="suspected", polluted_source_ids=[2, 3, 5, 6], confirmed_source_ids=[],
        source_rows=10, covered_rows=4, frontier=4, view_state="complete", healthy=False,
        prefix_over_leaf_budget=False, blocker_in_fresh_tail=True, blocker_source_ids=[8],
        empty_input_kind="none", unreachable_source_ids=[5, 6, 7], prefix_tokens=14, prefix_rows=2,
        leaf_budget=60, selection_budget=2000, selection_failed=False,
    )


def test_e1_only_hidden_copies_are_a_noop(tmp_path):
    path, messages = _e1_copy(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM messages WHERE store_id=7")
    inputs = _inputs(path, messages[7:], leaf_chunk_tokens=60, fresh_tail_count=3)
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert result["empty_input_kind"] == "none" and not result["selection_failed"]
    assert result["blocker_source_ids"] == [] and result["prefix_rows"] == 0


@pytest.mark.parametrize("merge,tail", [(False, 6), (False, 7), (True, 6), (True, 7)])
def test_rotated_store_matches_inplace(tmp_path, monkeypatch, merge, tail):
    findings = []
    for in_place in (True, False):
        root = tmp_path / ("inplace" if in_place else "rotation")
        root.mkdir()
        with monkeypatch.context() as patch:
            sequence = _run_host_commit_sequence(root, patch, in_place=in_place, merge=merge, tail=tail)
        _assert_clean_commit_sequence(sequence)
        path = root / "copy.db"
        _operator_copy(root / "lcm.db", path)
        with sqlite3.connect(path) as connection:
            conversation, session, frontier = connection.execute(
                "SELECT conversation_id,current_session_id,current_frontier_store_id FROM lcm_lifecycle_state "
                "WHERE current_session_id IS NOT NULL").fetchone()
            sources = set()
            for (ids,) in connection.execute("SELECT source_ids FROM summary_nodes WHERE source_type='messages'"):
                sources.update(json.loads(ids))
        before = path.read_bytes()
        result = inspector.inspect_store(path, session_id=session, conversation_id=conversation)
        assert result["covered_rows"] == len(sources) > 0
        assert result["frontier"] == frontier > 0
        assert path.read_bytes() == before
        findings.append((result["covered_rows"], result["frontier"]))
    assert findings[0] == findings[1]


@pytest.mark.parametrize("stored_note,view_note", [(True, False), (False, True), (False, False)])
def test_composite_identity_forms(tmp_path, stored_note, view_note):
    note = "[Note: model was just switched from A to B]\n"
    stored = [{"role": "user", "content": (note if stored_note else "") + "alpha words here", "timestamp": 500.0},
              {"role": "user", "content": "bravo words here", "timestamp": 510.0}]
    view = [{"role": "user", "content": (note if view_note else "") + "alpha words here\n\nbravo words here",
             "timestamp": 500.0}]
    engine = make_engine(tmp_path / "product", window=10000, session_id="S", conversation_id="conv")
    try:
        engine.ingest(stored)
        claims = engine._identity_anchor_summary_input(view, engine._get_store_id_map_for_messages(view),
                                                       view, view, budget=1000)
        assert [ids for _message, ids in claims] == [[1, 2]]
    finally:
        engine.shutdown()
    path = _copy(tmp_path, stored)
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **_inputs(path, view))
    assert result.get("evidence_code") == "identity_anchor_unsupported"
    assert result["unreachable_source_ids"] is None


@pytest.mark.parametrize("stored_note", [False, True], ids=["active-filter", "stored-filter"])
def test_missing_regex_leaves_view_unknown(tmp_path, monkeypatch, stored_note):
    monkeypatch.setattr(message_patterns, "_regex_engine", None)
    messages = [{**m, "role": "user"} for m in _messages(4)]
    if stored_note:
        messages[0]["content"] = "[Note: model was just switched from A to B]\n" + messages[0]["content"]
    path = _copy(tmp_path, messages)
    view = copy.deepcopy([messages[0], messages[2], messages[1]])
    if stored_note:
        view[0]["content"] = view[0]["content"].split("\n", 1)[1]
    inputs = _inputs(path, view, leaf_chunk_tokens=1, selection_budget=1)
    inputs["effective_config"]["lcm_config"]["ignore_message_patterns"] = [
        "model was just switched" if stored_note else "private payload 0"]
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert result.get("evidence_code") == "ignore_patterns_unavailable"
    assert result["view_state"] == "unknown" and result["prefix_over_leaf_budget"] is None
    assert result["unreachable_source_ids"] is None and not result["healthy"]


@pytest.mark.parametrize("fault", ["foreign", "missing", "malformed", "cycle", "outside", "contradictory"])
def test_rotation_ancestry_refusals(tmp_path, fault):
    path = _copy(tmp_path, _messages(3))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET session_id='P' WHERE store_id IN (1,2)")
        connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                           "VALUES ('S','summary','[1,2]','messages',0)")
        if fault != "missing":
            connection.execute("INSERT INTO metadata(key,value) VALUES ('rotation_predecessor_session:S',?)",
                               ('123' if fault == "malformed" else json.dumps('Q' if fault == "outside" else 'P'),))
        if fault == "foreign":
            connection.execute("UPDATE messages SET conversation_id='other' WHERE session_id='P'")
        if fault == "contradictory":
            connection.execute("UPDATE messages SET conversation_id='other' WHERE store_id=1")
        if fault == "cycle":
            connection.execute("INSERT INTO metadata(key,value) VALUES (?,?)",
                               ("rotation_predecessor_session:P", json.dumps("S")))
    with pytest.raises(inspector.InspectionRefused, match="^unverified_ancestry$"):
        inspector.inspect_store(path, session_id="S", conversation_id="conv")


@pytest.mark.parametrize("sources,valid", [("[2,1]", True), ("[1,2]", False)])
def test_node_lineage_uses_created_at_order(tmp_path, sources, valid):
    path = _copy(tmp_path, _messages(2))
    with sqlite3.connect(path) as connection:
        for node, created in ((1, 20), (2, 10)):
            connection.execute("INSERT INTO summary_nodes(node_id,session_id,summary,source_ids,source_type,created_at) "
                               "VALUES (?,'S','summary',?,'messages',?)", (node, f"[{node}]", created))
        connection.execute("INSERT INTO summary_nodes(node_id,session_id,summary,source_ids,source_type,created_at) "
                           "VALUES (3,'S','summary',?,'nodes',30)", (sources,))
    if valid:
        assert inspector.inspect_store(path, session_id="S", conversation_id="conv")["covered_rows"] == 2
    else:
        with pytest.raises(inspector.InspectionRefused, match="^invalid_dag$"):
            inspector.inspect_store(path, session_id="S", conversation_id="conv")


def test_dynamic_leaf_admission_is_unassessed(tmp_path):
    messages = _messages(3)
    path = _copy(tmp_path, messages)
    inputs = _inputs(path, messages, leaf_chunk_tokens=1)
    inputs["effective_config"]["lcm_config"]["dynamic_leaf_chunk_enabled"] = True
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert result.get("evidence_code") == "leaf_admission_unassessed"
    assert result["prefix_over_leaf_budget"] is None and "selection_failed" not in result


def test_carry_foreign_conversation_is_unverified(tmp_path):
    path = _copy(tmp_path, _messages(3))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET session_id='P',conversation_id='other' WHERE store_id=1")
        connection.execute("INSERT INTO metadata(key,value) VALUES (?,?)",
                           ("identity_anchor_carry:S", json.dumps([["P", 0, 1]])))
        connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                           "VALUES ('S','summary','[1]','messages',0)")
    with pytest.raises(inspector.InspectionRefused, match="^unverified_ancestry$"):
        inspector.inspect_store(path, session_id="S", conversation_id="conv")


@pytest.mark.parametrize("real_compiler", [False, True], ids=["double", "production-compiler"])
def test_filtered_view_completeness_uses_original_snapshot(tmp_path, monkeypatch, real_compiler):
    if real_compiler:
        if message_patterns._regex_engine is None:
            pytest.skip("production ignore compiler requires optional regex; regex is absent")
    else:
        monkeypatch.setattr(message_patterns, "_regex_engine", PatternEngine)
    messages = _messages(3)
    messages[0]["content"] = "[Note: model was just switched from A to B]\n" + messages[0]["content"]
    path = _copy(tmp_path, messages)
    view = copy.deepcopy(messages)
    view[0]["content"] = view[0]["content"].split("\n", 1)[1]
    inputs = _inputs(path, view, leaf_chunk_tokens=1, selection_budget=1)
    inputs["effective_config"]["lcm_config"]["ignore_message_patterns"] = ["model was just switched"]
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    selected = []
    original = inspector.select_oldest_leaf_chunk

    def select(raw, budget):
        chunk = original(raw, budget)
        selected.append([view.index(message) + 1 for message in chunk])
        return chunk

    monkeypatch.setattr(inspector, "select_oldest_leaf_chunk", select)
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert result.get("evidence_code") is None and selected[0] == [2]
    assert result["view_state"] == "complete"
    assert result["unreachable_source_ids"] == []
