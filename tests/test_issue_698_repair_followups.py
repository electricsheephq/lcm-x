"""#698: bounded diagnostics and backup-first, per-group level 3 repair."""

import json
import sqlite3

import pytest

from hermes_lcm import command, level3_repair
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import _L3_TRUNCATION_MARKER
from hermes_lcm.rollup_builder import initialize_rollup_invalidation_outbox
from hermes_lcm.schemas import LCM_DOCTOR
from hermes_lcm.tokens import count_tokens


@pytest.fixture
def engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "repair.db")),
                       hermes_home=str(tmp_path / "home"))
    engine._session_id = "s1"
    engine._conversation_id = "s1"
    yield engine
    engine.shutdown()


def _node(engine, session, sources, *, depth=0, fragment=True):
    text = "Preserved fragment " + _L3_TRUNCATION_MARKER if fragment else "Healthy ancestor"
    return engine._dag.add_node(SummaryNode(
        session_id=session, depth=depth, summary=text, token_count=count_tokens(text),
        source_ids=sources, source_type="nodes" if depth else "messages", created_at=123.0,
    ))


def _leaf(engine, session="s1"):
    row = engine._store.append(session, {"role": "user", "content": "Stored source"}, token_estimate=3)
    return _node(engine, session, [row]), row


def _route(monkeypatch, on_call=None):
    def summarize(**kwargs):
        if on_call:
            on_call()
        kwargs["provenance"]["model"] = "repair-model"
        return "Repaired model summary", 1

    monkeypatch.setattr(level3_repair, "summarize_with_escalation", summarize)
    monkeypatch.setattr(level3_repair, "summary_route_available", lambda *args, **kwargs: True)


def test_sqlite_error_rolls_back_one_group_and_continues(engine, monkeypatch):
    failed, _ = _leaf(engine)
    repaired, _ = _leaf(engine)
    old_text = engine._dag.get_node(failed).summary
    conn = engine._dag.connection
    conn.execute(f"""CREATE TRIGGER reject_one_repair BEFORE UPDATE OF summary ON summary_nodes
                     WHEN OLD.node_id = {failed}
                     BEGIN SELECT RAISE(ABORT, 'injected repair failure'); END""")
    conn.commit()
    _route(monkeypatch)

    result = level3_repair.repair_level3_fragments(engine)

    assert result["status"] == "partial"
    assert [group["outcome"] for group in result["groups"]] == ["error", "repaired"]
    assert result["groups"][0]["reason"] == "IntegrityError: injected repair failure"
    assert engine._dag.get_node(failed).summary == old_text
    assert engine._dag.get_node(repaired).summary == "Repaired model summary"
    assert not conn.in_transaction
    with sqlite3.connect(result["backup"]["backup_path"]) as backup:
        assert backup.execute("SELECT summary FROM summary_nodes WHERE node_id = ?", (repaired,)).fetchone()[0] \
            != "Repaired model summary"
    monkeypatch.setattr(command, "repair_level3_fragments", lambda engine: result)
    rendered = command._doctor_repair_level3_apply_text(engine)
    assert "error — IntegrityError: injected repair failure" in rendered
    assert "groups_error: 1" in rendered


@pytest.mark.parametrize("source_type", ["messages", "nodes"])
def test_sources_are_rechecked_inside_commit_transaction(engine, monkeypatch, source_type):
    leaf, row = _leaf(engine)
    group = [leaf]
    if source_type == "nodes":
        sibling_row = engine._store.append("s1", {"role": "user", "content": "Sibling"}, token_estimate=2)
        sibling = _node(engine, "s1", [sibling_row], fragment=False)
        group.append(_node(engine, "s1", [leaf, sibling], depth=1, fragment=False))
    old_texts = [engine._dag.get_node(node_id).summary for node_id in group]
    calls = 0

    def remove_source():
        nonlocal calls
        calls += 1
        if calls != len(group):  # delete after the final summariser has read all its inputs
            return
        conn = engine._dag.connection
        if source_type == "messages":
            conn.execute("DELETE FROM messages WHERE store_id = ?", (row,))
        else:
            conn.execute("DELETE FROM summary_nodes WHERE node_id = ?", (sibling,))
        conn.commit()

    _route(monkeypatch, remove_source)
    real_stored_sources = level3_repair._stored_sources
    transaction_checks = []

    def stored_sources(conn, *args):
        transaction_checks.append(conn is engine._dag.connection and conn.in_transaction)
        return real_stored_sources(conn, *args)

    monkeypatch.setattr(level3_repair, "_stored_sources", stored_sources)
    result = level3_repair.repair_level3_fragments(engine)

    assert result["groups"][0]["outcome"] == "rolled back"
    assert result["groups"][0]["reason"] == "a source was removed during the repair"
    assert result["status"] == "partial" and any(transaction_checks)
    assert [engine._dag.get_node(node_id).summary for node_id in group] == old_texts
    assert not engine._dag.connection.in_transaction


def test_repair_inserts_missing_provenance_and_preserves_existing_timestamp(engine, monkeypatch):
    legacy, _ = _leaf(engine)
    recorded, _ = _leaf(engine)
    conn = engine._dag.connection
    conn.execute("INSERT INTO summary_node_provenance VALUES (?, 3, 'old-model', 456)", (recorded,))
    conn.commit()
    _route(monkeypatch)

    result = level3_repair.repair_level3_fragments(engine)

    assert result["status"] == "ok"
    assert conn.execute("SELECT escalation_level, model, created_at FROM summary_node_provenance WHERE node_id = ?",
                        (legacy,)).fetchone() == (1, "repair-model", 123.0)
    assert conn.execute("SELECT escalation_level, model, created_at FROM summary_node_provenance WHERE node_id = ?",
                        (recorded,)).fetchone() == (1, "repair-model", 456.0)


def test_repair_schedules_rollups_once_per_affected_session_after_commits(engine, monkeypatch):
    engine._config.temporal_rollups_enabled = True
    initialize_rollup_invalidation_outbox(engine._dag)
    nodes = [_leaf(engine, session)[0] for session in ("s1", "s1", "s2")]
    _route(monkeypatch)
    scheduled = []

    def schedule(session):
        assert all(engine._dag.get_node(node_id).summary == "Repaired model summary" for node_id in nodes)
        assert not engine._dag.connection.in_transaction
        scheduled.append(session)

    monkeypatch.setattr(engine, "_schedule_rollup_maintenance", schedule)
    monkeypatch.setattr(engine, "_invalidate_rollups_for_published_node",
                        lambda node: pytest.fail("repair must not drain rollups synchronously"))
    result = level3_repair.repair_level3_fragments(engine)

    assert scheduled == ["s1", "s2"] and result["rollups_scheduled"] == 2
    assert engine._dag.connection.execute("SELECT COUNT(*) FROM lcm_rollup_invalidations").fetchone()[0] > 0
    monkeypatch.setattr(command, "repair_level3_fragments", lambda engine: result)
    assert "rollups_scheduled: 2" in command._doctor_repair_level3_apply_text(engine)


def test_overlapping_condensations_keep_reservation_until_last_exit(engine, monkeypatch):
    leaf, _ = _leaf(engine)
    node = engine._dag.get_node(leaf)
    first = engine._condensation_in_flight([node])
    second = engine._condensation_in_flight([node])
    first.__enter__()
    second.__enter__()
    try:
        first.__exit__(None, None, None)
        _route(monkeypatch)
        result = level3_repair.repair_level3_fragments(engine)
        assert result["groups"][0]["outcome"] == "rolled back"
        assert leaf in engine._condensation_inflight_ids
    finally:
        second.__exit__(None, None, None)
    assert not engine._condensation_inflight_ids
    assert level3_repair.repair_level3_fragments(engine)["status"] == "ok"


def test_tool_pages_both_lists_with_totals_and_cursor(engine):
    leaves, ancestors = [], []
    for _ in range(51):
        leaf, _ = _leaf(engine)
        leaves.append(leaf)
        parent = _node(engine, "s1", [leaf], depth=1, fragment=False)
        ancestors.extend([parent, _node(engine, "s1", [parent], depth=2, fragment=False)])
    seen_leaves, seen_ancestors = [], []
    cursor = 0
    while True:
        page = json.loads(engine.handle_tool_call("lcm_doctor", {"action": "repair_level3", "cursor": cursor}))
        assert len(page["flagged"]) <= 50 and len(page["ancestors"]) <= 50
        assert page["total_flagged"] == 51 and page["total_ancestors"] == 102
        seen_leaves.extend(item["node_id"] for item in page["flagged"])
        seen_ancestors.extend(item["node_id"] for item in page["ancestors"])
        if "next_cursor" not in page:
            break
        assert page["next_cursor"] == cursor + 50
        cursor = page["next_cursor"]
    assert seen_leaves == leaves and seen_ancestors == sorted(ancestors, key=lambda i: (engine._dag.get_node(i).depth, i))
    assert LCM_DOCTOR["parameters"]["properties"]["cursor"]["type"] == "integer"


def test_tool_scopes_to_foreground_but_slash_scan_remains_store_wide(engine):
    foreground, _ = _leaf(engine, "s1")
    other, _ = _leaf(engine, "s2")
    local_parent = _node(engine, "s1", [foreground], depth=1, fragment=False)
    other_parent = _node(engine, "s2", [other], depth=1, fragment=False)
    cross_parent = _node(engine, "s2", [foreground], depth=1, fragment=False)
    engine._foreground_session_id = "s1"
    engine._session_id = "s2"  # a side-channel bind must not change diagnostic scope

    payload = json.loads(engine.handle_tool_call("lcm_doctor", {"action": "repair_level3"}))

    assert [item["node_id"] for item in payload["flagged"]] == [foreground]
    assert [item["node_id"] for item in payload["ancestors"]] == [local_parent]
    assert set(payload["sessions"]) == {"s1"}
    slash = command._doctor_repair_level3_text(engine)
    assert f"node {other} (session s2" in slash and "affected_ancestors: 3" in slash
    assert [item["node_id"] for item in level3_repair.scan_level3_fragments(engine)["ancestors"]] \
        == [local_parent, other_parent, cross_parent]
    engine._foreground_session_id = engine._session_id = ""
    empty = json.loads(engine.handle_tool_call("lcm_doctor", {"action": "repair_level3"}))
    assert empty["flagged"] == [] and empty["ancestors"] == []
