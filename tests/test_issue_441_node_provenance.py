"""#441: every new summary node records the escalation level and the model that produced it.

The record lives in a sidecar table written in the node's own transaction, not in a summary_nodes
column: rows there are decoded by position, here and by every older build after a plugin rollback.
The schema version does not change."""

from __future__ import annotations

import json
import logging
import sqlite3

import pytest

import hermes_lcm.db_bootstrap as db_bootstrap
import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore

PAD = " alpha beta gamma delta" * 30
ACCEPTED = "Earlier turns.\nExpand for details about: turns"


def _node(session_id: str = "S", depth: int = 0, idx: int = 1) -> SummaryNode:
    return SummaryNode(session_id=session_id, depth=depth, summary=f"summary {idx}", token_count=10,
                       source_ids=[idx], source_type="messages", expand_hint=f"hint {idx}")


@pytest.fixture
def dag(tmp_path):
    instance = SummaryDAG(tmp_path / "lcm.db")
    try:
        yield instance
    finally:
        instance.close()


# -- T1: the write API records level and model; a node written without them has no record -------------------

def test_t1_add_node_records_level_and_model(dag):
    recorded = dag.add_node(_node(idx=1), escalation_level=2, model="m")
    plain = dag.add_node(_node(idx=2))

    assert dag.get_node_provenance(recorded) == {"escalation_level": 2, "model": "m"}
    assert dag.get_node_provenance(plain) is None
    assert [n.expand_hint for n in dag.get_session_nodes("S")] == ["hint 1", "hint 2"]


# -- T2: the provenance row and the node commit or roll back together ----------------------------------------

def test_t2_failed_provenance_insert_writes_no_node(dag):
    dag.connection.execute(
        "CREATE TRIGGER fail_provenance BEFORE INSERT ON summary_node_provenance "
        "BEGIN SELECT RAISE(ABORT, 'provenance insert failed'); END"
    )
    dag.connection.commit()

    with pytest.raises(sqlite3.DatabaseError, match="provenance insert failed"):
        dag.add_node(_node(idx=1), escalation_level=1, model="m")

    assert dag.get_session_node_count("S") == 0
    assert dag.connection.execute("SELECT COUNT(*) FROM summary_node_provenance").fetchone()[0] == 0
    assert dag.search("summary") == []
    # A node written without provenance never touches the table.
    assert dag.add_node(_node(idx=2)) > 0
    assert dag.get_session_node_count("S") == 1


# -- T3: counts by level, with unrecorded nodes counted apart ------------------------------------------------

def test_t3_counts_by_escalation_level(dag):
    for idx, level in enumerate((1, 3, 3), start=1):
        dag.add_node(_node(idx=idx), escalation_level=level, model="m")
    dag.add_node(_node(idx=4))
    dag.add_node(_node(session_id="other", idx=5), escalation_level=2, model="m")

    assert dag.count_nodes_by_escalation_level("S") == {1: 1, 3: 2, "unrecorded": 1}
    assert dag.count_nodes_by_escalation_level("other") == {2: 1, "unrecorded": 0}
    assert dag.count_nodes_by_escalation_level(None) == {1: 1, 2: 1, 3: 2, "unrecorded": 1}
    assert dag.count_nodes_by_escalation_level("empty") == {"unrecorded": 0}


# -- T4: the leaf and condensation paths record what the summarizer reports ----------------------------------

@pytest.fixture
def _pin_lcm_char_counter(monkeypatch):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


def _summarizer(level: int, model: str):
    def fake(**kwargs):
        provenance = kwargs.get("provenance")
        if provenance is not None:
            provenance["model"] = model
        return (kwargs["text"] if level == 3 else ACCEPTED), level  # #652: level 3 only verbatim
    return fake


def test_t4_leaf_records_level_and_model(tmp_path, monkeypatch, caplog, _pin_lcm_char_counter):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summarizer(3, "stub-model"))
    engine = LCMEngine(config=LCMConfig(
        fresh_tail_count=2, leaf_chunk_tokens=400, context_threshold=0.001, max_assembly_tokens=100_000,
        threshold_full_sweep_enabled=False,  # #1013: this cell counts one leaf per compaction
        database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    try:
        view = [{"role": "system", "content": "system prompt"}]
        for i in range(6):
            view += [{"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
                     {"role": "assistant", "content": f"reply to T{i}{PAD}"}]
        engine.ingest(view)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(view, current_tokens=engine.threshold_tokens + 1)

        leaves = [n for n in engine._dag.get_session_nodes("S") if n.depth == 0]
        assert leaves
        for leaf in leaves:
            assert engine._dag.get_node_provenance(leaf.node_id) == {
                "escalation_level": 3, "model": "stub-model"}
        described = json.loads(lcm_tools.lcm_describe({"node_id": leaves[0].node_id}, engine=engine))
        assert (described["escalation_level"], described["model"]) == (3, "stub-model")
        status = json.loads(lcm_tools.lcm_status({}, engine=engine))
        assert status["dag"]["nodes_by_escalation_level"] == {"3": len(leaves), "unrecorded": 0}
    finally:
        engine.shutdown()


def test_t4_condensation_records_level_and_model(tmp_path, monkeypatch):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm.db")
    engine = LCMEngine(config=config)
    engine._session_id = "test-session"
    try:
        for index in range(engine._config.condensation_fanin):
            engine._dag.add_node(_node(session_id="test-session", idx=index))
        monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summarizer(1, "cond-model"))
        engine._maybe_condense()

        parent = next(n for n in engine._dag.get_session_nodes("test-session") if n.depth == 1)
        assert engine._dag.get_node_provenance(parent.node_id) == {"escalation_level": 1, "model": "cond-model"}
        children = [n for n in engine._dag.get_session_nodes("test-session") if n.depth == 0]
        assert all(engine._dag.get_node_provenance(n.node_id) is None for n in children)
        described = json.loads(lcm_tools.lcm_describe({"node_id": children[0].node_id}, engine=engine))
        assert "escalation_level" not in described and "model" not in described
    finally:
        engine.shutdown()


# -- T5: the chain reports the model that answered; recording does not change the calls ----------------------

class _Llm:
    def __init__(self, answers: dict[str, str | None]):
        self.answers = answers
        self.calls: list[tuple] = []

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        self.calls.append((prompt, max_tokens, model, timeout, reasoning_effort))
        return self.answers.get(model)


def _escalate(monkeypatch, answers, **kwargs):
    llm = _Llm(answers)
    monkeypatch.setattr(escalation, "_invoke_summary_llm", llm)
    result = escalation.summarize_with_escalation(
        text="source text " * 200, source_tokens=1000, token_budget=100, **kwargs)
    return result, llm.calls


def test_t5_fallback_model_that_answered_is_recorded(monkeypatch):
    provenance: dict = {}
    (summary, level), calls = _escalate(
        monkeypatch, {"a": None, "b": "short summary"}, model="a", fallback_models=["b"], provenance=provenance)
    assert (summary, level) == ("short summary", 1)
    assert provenance == {"model": "b"}
    assert [call[2] for call in calls] == ["a", "b"]


def test_t5_deterministic_and_host_default_routes(monkeypatch):
    provenance: dict = {}
    (_summary, level), _calls = _escalate(monkeypatch, {}, model="a", provenance=provenance)
    assert level == 3 and provenance == {"model": "deterministic"}

    provenance = {}
    (_summary, level), _calls = _escalate(monkeypatch, {"": "short summary"}, provenance=provenance)
    assert level == 1 and provenance == {"model": ""}


def test_t5_recording_leaves_the_calls_and_result_unchanged(monkeypatch):
    answers = {"a": None, "b": None}
    plain, plain_calls = _escalate(monkeypatch, answers, model="a", fallback_models=["b"])
    recorded, recorded_calls = _escalate(
        monkeypatch, answers, model="a", fallback_models=["b"], provenance={})
    assert plain == recorded and plain_calls == recorded_calls and plain[1] == 3


# -- T6: no schema version change; a store holding the table is still the v5 shape ---------------------------

def test_t6_schema_version_unchanged_and_store_classified_v5(tmp_path):
    assert db_bootstrap.SCHEMA_VERSION == 5
    db_path = tmp_path / "lcm.db"
    dag = SummaryDAG(db_path)
    store = MessageStore(db_path)
    try:
        dag.add_node(_node(), escalation_level=1, model="m")
        conn = store._conn
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "summary_node_provenance" in tables
        columns = {row[1] for row in conn.execute("PRAGMA table_info(summary_nodes)")}
        assert columns == db_bootstrap._V5_CORE_TABLE_COLUMNS["summary_nodes"]
        assert db_bootstrap.read_existing_schema_version(conn) == db_bootstrap.SCHEMA_VERSION
        db_bootstrap.refuse_schema_version_too_new(conn)
        assert db_bootstrap.classify_version_mismatch(conn) == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
    finally:
        store.close()
        dag.close()
    # Reopening the store keeps the recorded row and the stamp.
    reopened = SummaryDAG(db_path)
    try:
        assert reopened.get_node_provenance(1) == {"escalation_level": 1, "model": "m"}
        assert db_bootstrap.read_existing_schema_version(reopened.connection) == 5
    finally:
        reopened.close()
