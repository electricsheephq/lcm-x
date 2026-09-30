"""#667: `/lcm doctor repair level3` finds level 3 fragments and their condensed ancestors, read-only."""

import hashlib
import json
from pathlib import Path

import pytest

from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import _L3_TRUNCATION_MARKER, _deterministic_truncate
from hermes_lcm.tokens import count_tokens

LONG_SOURCE = "\n".join(f"[user]: step {i} of the deploy: checked host {i} and restarted worker {i}." for i in range(400))


@pytest.fixture
def engine(tmp_path):
    e = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm_667.db")), hermes_home=str(tmp_path / "hermes_home"))
    e._session_id = "s1"
    e._session_platform = "telegram"
    e._conversation_id = "s1"
    yield e
    e.shutdown()


def _node(engine, session_id, depth, summary, source_ids, source_type="messages"):
    return engine._dag.add_node(SummaryNode(
        session_id=session_id, depth=depth, summary=summary, token_count=count_tokens(summary),
        source_token_count=1000, source_ids=source_ids, source_type=source_type, created_at=float(len(source_ids)),
    ))


def _rows(engine, session_id, count):
    return [engine._store.append(session_id, {"role": "user", "content": f"row {i} {session_id}"}, token_estimate=4)
            for i in range(count)]


def _build_store(engine):
    """A truncated leaf under a parent and a grandparent, a verbatim level 3 leaf, and a healthy branch."""
    rows = _rows(engine, "s1", 6)
    fragment = _deterministic_truncate(LONG_SOURCE, 512)
    assert _L3_TRUNCATION_MARKER in fragment and count_tokens(fragment) <= 512
    truncated = _node(engine, "s1", 0, fragment, rows[0:2])
    verbatim = _node(engine, "s1", 0, "[user]: row 2 s1", rows[2:3])  # level 3 of a source that fit: no marker
    healthy = _node(engine, "s1", 0, "The user checked rows 3 and 4.", rows[3:5])
    parent = _node(engine, "s1", 1, "Arc: deploy steps, hosts checked.", [truncated, verbatim], "nodes")
    sibling = _node(engine, "s1", 1, "Arc: rows checked.", [healthy], "nodes")
    grandparent = _node(engine, "s1", 2, "Durable: the deploy.", [parent, sibling], "nodes")
    other = _rows(engine, "s2", 2)
    other_leaf = _node(engine, "s2", 0, "Healthy summary of s2.", other)
    other_parent = _node(engine, "s2", 1, "Healthy arc of s2.", [other_leaf], "nodes")
    return {"truncated": truncated, "verbatim": verbatim, "parent": parent, "grandparent": grandparent,
            "sibling": sibling, "other_parent": other_parent}


def _file_state(db_path: Path) -> dict:
    state = {}
    for path in (db_path, Path(f"{db_path}-wal")):
        if path.exists():
            state[path.name] = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return state


def test_dry_run_counts_fragments_and_ancestors(engine):
    ids = _build_store(engine)

    result = handle_lcm_command("doctor repair level3", engine)

    assert "LCM doctor repair level3" in result
    assert "status: repair-needed" in result
    assert "detection: truncation marker, at most 512 tokens" in result
    assert "flagged_leaves: 1" in result
    assert "flagged_nodes: 0" in result
    assert "affected_ancestors: 2" in result
    assert "session s1: flagged_leaves=1 flagged_nodes=0 ancestors_by_depth=d1=1, d2=1" in result
    assert f"node {ids['truncated']} (session s1, d0): raw_rows_stored=yes (2/2)" in result
    assert "session s2" not in result
    assert f"node {ids['verbatim']} " not in result
    assert result.endswith("note: read-only scan only — nothing was changed")


def test_tool_action_matches_the_command(engine):
    ids = _build_store(engine)

    payload = json.loads(engine.handle_tool_call("lcm_doctor", {"action": "repair_level3"}))

    assert payload["status"] == "repair-needed" and payload["read_only"] is True
    assert [item["node_id"] for item in payload["flagged"]] == [ids["truncated"]]
    assert [item["node_id"] for item in payload["ancestors"]] == [ids["parent"], ids["grandparent"]]
    assert payload["sessions"] == {
        "s1": {"flagged_leaves": 1, "flagged_nodes": 0, "ancestors_by_depth": {"1": 1, "2": 1}}}
    assert "error" in json.loads(engine.handle_tool_call("lcm_doctor", {"action": "nope"}))


def test_dry_run_writes_nothing(engine):
    _build_store(engine)
    db_path = Path(engine._store.db_path)
    before = _file_state(db_path)

    handle_lcm_command("doctor repair level3", engine)
    engine.handle_tool_call("lcm_doctor", {"action": "repair_level3"})

    assert before and _file_state(db_path) == before
    assert not engine._store.connection.in_transaction and not engine._dag.connection.in_transaction
    _rows(engine, "s1", 1)  # positive control: a write does change the files the proof compares
    assert _file_state(db_path) != before


def test_clean_store_reports_ok(engine):
    rows = _rows(engine, "s1", 2)
    _node(engine, "s1", 0, "[user]: row 0 s1\n[user]: row 1 s1", rows)  # verbatim level 3 only

    result = handle_lcm_command("doctor repair level3", engine)

    assert "status: ok" in result and "flagged_leaves: 0" in result and "affected_ancestors: 0" in result


def test_condensed_fragment_missing_rows_and_long_marker_quote(engine):
    rows = _rows(engine, "s1", 3)
    leaf = _node(engine, "s1", 0, _deterministic_truncate(LONG_SOURCE, 512), [rows[0], 999_999])
    other_leaf = _node(engine, "s1", 0, "Healthy leaf.", rows[1:3])
    condensed = _node(engine, "s1", 1, _deterministic_truncate(LONG_SOURCE, 512), [other_leaf], "nodes")
    top = _node(engine, "s1", 2, "Durable.", [condensed], "nodes")
    quote = LONG_SOURCE + _L3_TRUNCATION_MARKER + "the model quoted the marker in a long summary"
    _node(engine, "s1", 0, quote, rows[1:2])  # above the bound: a model summary, not a fragment

    result = handle_lcm_command("doctor repair level3", engine)

    assert "flagged_leaves: 1" in result and "flagged_nodes: 1" in result and "affected_ancestors: 1" in result
    assert f"node {leaf} (session s1, d0): raw_rows_stored=no (1/2)" in result
    assert f"node {condensed} (session s1, d1): source_nodes_stored=yes (1/1)" in result
    assert "ancestors_by_depth=d2=1" in result and f"node {top} " not in result


def test_provenance_level_decides_where_recorded(engine):
    rows = _rows(engine, "s1", 3)
    short_quote = _node(engine, "s1", 0, "Summary." + _L3_TRUNCATION_MARKER + "tail", rows[0:1])
    long_fragment = _node(engine, "s1", 0, LONG_SOURCE + _L3_TRUNCATION_MARKER + LONG_SOURCE, rows[1:2])
    unrecorded = _node(engine, "s1", 0, _deterministic_truncate(LONG_SOURCE, 512), rows[2:3])
    conn = engine._dag.connection
    conn.execute("CREATE TABLE summary_node_provenance (node_id INTEGER PRIMARY KEY, escalation_level INTEGER "
                 "NOT NULL, model TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)")
    conn.executemany("INSERT INTO summary_node_provenance VALUES (?, ?, '', 0)", [(short_quote, 1), (long_fragment, 3)])
    conn.commit()

    payload = json.loads(engine.handle_tool_call("lcm_doctor", {"action": "repair_level3"}))

    assert payload["provenance_table"] is True
    assert [item["node_id"] for item in payload["flagged"]] == [long_fragment, unrecorded]
