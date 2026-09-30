"""#667: `/lcm doctor repair level3` finds level 3 fragments and their condensed ancestors, read-only."""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import hermes_lcm.engine as engine_mod
import hermes_lcm.escalation as escalation_mod
import hermes_lcm.level3_repair as level3_repair
from hermes_lcm.maintenance import flush_engine_connections
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode, build_nodes_fts_spec
from hermes_lcm.db_bootstrap import check_external_content_fts_integrity
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
    conn.execute("CREATE TABLE IF NOT EXISTS summary_node_provenance (node_id INTEGER PRIMARY KEY, escalation_level INTEGER "
                 "NOT NULL, model TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)")
    conn.executemany("INSERT INTO summary_node_provenance VALUES (?, ?, '', 0)", [(short_quote, 1), (long_fragment, 3)])
    conn.commit()

    payload = json.loads(engine.handle_tool_call("lcm_doctor", {"action": "repair_level3"}))

    assert payload["provenance_table"] is True
    assert [item["node_id"] for item in payload["flagged"]] == [long_fragment, unrecorded]


# -- Stage 2: `/lcm doctor repair level3 apply` -------------------------------------------------------------------


def _fake_route(monkeypatch, level=1, on_call=None):
    calls = []

    def fake(*, text, depth, **_kwargs):
        calls.append((depth, text))
        if on_call is not None:
            on_call(len(calls))
        return f"Model summary {len(calls)} at d{depth}.\nExpand for details about: repaired {len(calls)}", level

    monkeypatch.setattr(level3_repair, "summarize_with_escalation", fake, raising=False)  # red on behaviour
    return calls


def _texts(engine):
    return {row[0]: row[1:] for row in engine._dag.connection.execute(
        "SELECT node_id, summary, source_ids, source_type, depth, created_at FROM summary_nodes")}


def _raw_rows(engine):
    return engine._store.connection.execute("SELECT * FROM messages ORDER BY store_id").fetchall()


def test_apply_repairs_leaf_and_ancestors_in_place(engine, monkeypatch):
    ids = _build_store(engine)
    before, raw_before = _texts(engine), _raw_rows(engine)
    calls = _fake_route(monkeypatch)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    group = [ids["truncated"], ids["parent"], ids["grandparent"]]
    assert "status: ok" in result
    assert (f"group 1 (session s1, top node {ids['grandparent']}): repaired — node {ids['truncated']} d0 fragment "
            f"-> level 1, node {ids['parent']} d1 ancestor -> level 1, node {ids['grandparent']} d2 ancestor -> level 1"
            ) in result
    assert "groups_repaired: 1" in result and "nodes_repaired: 3" in result and "summariser_calls: 3" in result
    assert "second_scan: flagged_leaves=0 flagged_nodes=0 affected_ancestors=0" in result
    backup = Path(next(line for line in result.splitlines() if line.startswith("backup_path: ")).split(": ", 1)[1])
    with sqlite3.connect(backup) as copy:
        assert copy.execute("SELECT summary FROM summary_nodes WHERE node_id = ?", (ids["truncated"],)).fetchone()[0] \
            == before[ids["truncated"]][0]
    after = _texts(engine)
    assert set(after) == set(before) and _raw_rows(engine) == raw_before
    for node_id, row in after.items():
        assert row[1:] == before[node_id][1:]  # links, depth and created_at kept
        assert (row[0] != before[node_id][0]) == (node_id in group)
    assert "row 0 s1" in calls[0][1]  # the leaf is re-summarised from its stored rows
    assert after[ids["truncated"]][0] in calls[1][1] and "[user]: row 2 s1" in calls[1][1]  # repaired child + sibling
    assert after[ids["parent"]][0] in calls[2][1]
    assert engine._dag.get_node(ids["truncated"]).expand_hint == "repaired 1"
    conn = engine._dag.connection
    assert {node.node_id for node in engine._dag.search("Model summary", session_id="s1")} == set(group)
    assert check_external_content_fts_integrity(conn, build_nodes_fts_spec())["status"] == "pass"
    expanded = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": ids["truncated"]}))
    assert "row 0 s1" in json.dumps(expanded)
    conn.execute("UPDATE summary_nodes SET summary = 'drift' WHERE node_id = ?", (ids["other_parent"],))
    assert check_external_content_fts_integrity(conn, build_nodes_fts_spec())["status"] == "fail"  # the oracle bites


def test_apply_skips_a_group_with_missing_raw_rows(engine, monkeypatch):
    rows = _rows(engine, "s1", 1)
    leaf = _node(engine, "s1", 0, _deterministic_truncate(LONG_SOURCE, 512), [rows[0], 999_999])
    before = _texts(engine)
    calls = _fake_route(monkeypatch)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert "status: partial" in result and "summariser_calls: 0" in result and "backup_path" not in result
    assert f"skipped — sources missing: node {leaf} (session s1) 1/2 message rows stored" in result
    assert calls == [] and _texts(engine) == before


def test_apply_refuses_before_backup_while_no_route(engine, monkeypatch):
    _build_store(engine)
    db_path = Path(engine._store.db_path)
    state = _file_state(db_path)
    calls = _fake_route(monkeypatch)
    monkeypatch.setattr(level3_repair, "summary_route_available", lambda *_args, **_kwargs: False, raising=False)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert "status: refused" in result and "every summary route is refused; nothing was changed" in result
    assert calls == [] and _file_state(db_path) == state
    assert not engine.backup_dir().exists() or not any(engine.backup_dir().iterdir())


def test_a_level3_result_leaves_the_group_untouched(engine, monkeypatch):
    ids = _build_store(engine)
    before = _texts(engine)
    _fake_route(monkeypatch, level=3)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert f"skipped — summary route refused at node {ids['truncated']} (level 3)" in result
    assert "status: partial" in result and "summariser_calls: 1" in result and "backup_path: " in result
    assert _texts(engine) == before
    assert "second_scan: flagged_leaves=1 flagged_nodes=0 affected_ancestors=2" in result


def test_a_concurrent_parent_rolls_the_group_back(engine, monkeypatch):
    ids = _build_store(engine)
    added = []

    def live_condensation(call_number):
        if call_number == 1:
            added.append(_node(engine, "s1", 3, "A live parent.", [ids["grandparent"]], "nodes"))

    _fake_route(monkeypatch, on_call=live_condensation)
    before = _texts(engine)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert "rolled back — a node changed or a new parent linked in during the repair" in result
    assert "groups_rolled_back: 1" in result and "status: partial" in result
    assert {k: v for k, v in _texts(engine).items() if k not in added} == before
    assert "affected_ancestors=3" in result  # the rerun sees the new parent too


def test_tool_path_never_writes(engine):
    _build_store(engine)
    db_path = Path(engine._store.db_path)
    state = _file_state(db_path)

    payload = json.loads(engine.handle_tool_call("lcm_doctor", {"action": "repair_level3", "apply": True}))

    assert "operator-only" in payload.get("error", "")
    assert _file_state(db_path) == state and not engine.backup_dir().exists()


def test_a_crash_between_group_commits_leaves_a_consistent_store_and_a_rerun_completes(engine, monkeypatch):
    ids = _build_store(engine)
    rows = _rows(engine, "s1", 1)
    lone = _node(engine, "s1", 0, _deterministic_truncate(LONG_SOURCE, 512), rows)  # a second group
    _fake_route(monkeypatch)
    real_commit, commits = getattr(level3_repair, "_commit_group", None), []

    def crash_on_second(*args):
        commits.append(1)
        if len(commits) == 2:
            raise RuntimeError("injected crash")
        return real_commit(*args)

    monkeypatch.setattr(level3_repair, "_commit_group", crash_on_second, raising=False)
    with pytest.raises(RuntimeError):
        handle_lcm_command("doctor repair level3 apply", engine)

    conn = engine._dag.connection
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert check_external_content_fts_integrity(conn, build_nodes_fts_spec())["status"] == "pass"
    assert [item["node_id"] for item in level3_repair.scan_level3_fragments(engine)["flagged"]] == [lone]
    monkeypatch.setattr(level3_repair, "_commit_group", real_commit, raising=False)
    result = handle_lcm_command("doctor repair level3 apply", engine)
    assert f"top node {lone}): repaired" in result and "second_scan: flagged_leaves=0" in result
    assert ids["truncated"] not in {item["node_id"] for item in level3_repair.scan_level3_fragments(engine)["flagged"]}


# -- Round 3: the #689 review findings ---------------------------------------------------------------------------


def test_m1_an_ancestor_with_a_missing_child_skips_the_whole_group(engine, monkeypatch):
    rows = _rows(engine, "s1", 1)
    fragment = _node(engine, "s1", 0, _deterministic_truncate(LONG_SOURCE, 512), rows)
    parent = _node(engine, "s1", 1, "Arc: the fragment and a sibling that is gone.", [fragment, 999_999], "nodes")
    before = _texts(engine)
    calls = _fake_route(monkeypatch)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert f"skipped — sources missing: node {parent} (session s1) 1/2 child nodes stored" in result
    assert calls == [] and _texts(engine) == before and "backup_path" not in result


def test_m2_a_group_a_live_condensation_selected_is_rolled_back(engine, monkeypatch):
    rows = _rows(engine, "s1", 4)
    fragment = _node(engine, "s1", 0, _deterministic_truncate(LONG_SOURCE, 512), rows[:1])
    for row in rows[1:]:
        _node(engine, "s1", 0, f"Healthy leaf {row}.", [row])
    old_text = engine._dag.get_node(fragment).summary
    _fake_route(monkeypatch)
    repair_output = []

    def live_condensation_summary(**_kwargs):  # the repair runs while the condensation waits on its summariser
        repair_output.append(handle_lcm_command("doctor repair level3 apply", engine))
        return "Arc summary of four leaves.", 1

    monkeypatch.setattr(engine_mod, "summarize_with_escalation", live_condensation_summary)
    assert engine._maybe_condense() == 1

    assert (f"rolled back — a condensation in flight uses node {fragment}; run apply again when the agent is idle"
            in repair_output[0])
    assert engine._dag.get_node(fragment).summary == old_text
    assert not engine._condensation_inflight_ids
    rerun = handle_lcm_command("doctor repair level3 apply", engine)  # the new parent is found and repaired too
    assert "groups_repaired: 1" in rerun and "nodes_repaired: 2" in rerun and "second_scan: flagged_leaves=0" in rerun


def test_m3_the_backup_flush_waits_for_an_open_dag_transaction(engine):
    done = threading.Event()
    with engine._dag._db_lock:
        engine._dag.connection.execute("BEGIN IMMEDIATE")
        engine._dag.connection.execute("INSERT INTO metadata(key, value) VALUES('lcm_667_probe', 'x')")
        flusher = threading.Thread(target=lambda: (flush_engine_connections(engine), done.set()))
        flusher.start()
        assert not done.wait(0.3)  # the flush does not commit the transaction this thread owns
        engine._dag.connection.rollback()
    flusher.join(5)
    assert done.is_set()
    assert engine._dag.connection.execute("SELECT 1 FROM metadata WHERE key = 'lcm_667_probe'").fetchone() is None


def test_n2_a_repaired_ancestor_carries_its_children_token_counts(engine, monkeypatch):
    ids = _build_store(engine)
    _fake_route(monkeypatch)

    handle_lcm_command("doctor repair level3 apply", engine)

    for parent in (ids["parent"], ids["grandparent"]):
        node = engine._dag.get_node(parent)
        assert node.source_token_count == sum(engine._dag.get_node(i).token_count for i in node.source_ids)


def test_n3_the_provenance_level_follows_the_repair(engine, monkeypatch):
    ids = _build_store(engine)
    conn = engine._dag.connection
    conn.execute("CREATE TABLE IF NOT EXISTS summary_node_provenance (node_id INTEGER PRIMARY KEY, escalation_level INTEGER "
                 "NOT NULL, model TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL)")
    conn.executemany("INSERT INTO summary_node_provenance VALUES (?, ?, 'm', 7)",
                     [(ids["truncated"], 3), (ids["parent"], 2), (ids["other_parent"], 2)])
    conn.commit()
    _fake_route(monkeypatch)

    assert "status: ok" in handle_lcm_command("doctor repair level3 apply", engine)

    assert dict((r[0], r[1:]) for r in conn.execute("SELECT * FROM summary_node_provenance")) == {
        ids["truncated"]: (1, "", 7.0), ids["parent"]: (1, "", 7.0), ids["other_parent"]: (2, "m", 7.0)}


def test_n4_repair_rejections_do_not_touch_the_live_route_keys(engine, monkeypatch):
    _build_store(engine)
    monkeypatch.setattr(escalation_mod, "_invoke_summary_llm", lambda *_a, **_k: LONG_SOURCE * 2)  # never shorter

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert "summary route refused at node" in result
    breaker = engine._summary_circuit_breaker
    assert breaker._rejections == {"repair:": 2} and not breaker._open_until


def test_a_failed_backup_writes_nothing(engine, monkeypatch):
    _build_store(engine)
    before = _texts(engine)
    calls = _fake_route(monkeypatch)
    monkeypatch.setattr(level3_repair, "backup_database",
                        lambda _engine: {"ok": False, "db_path": "x", "error": "disk full"}, raising=False)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert "status: error" in result and "backup failed: disk full; nothing was changed" in result
    assert calls == [] and _texts(engine) == before


def test_a_text_changed_between_read_and_commit_rolls_back(engine, monkeypatch):
    ids = _build_store(engine)

    def edit(call_number):
        if call_number == 2:
            engine._dag.connection.execute(
                "UPDATE summary_nodes SET summary = 'edited elsewhere' WHERE node_id = ?", (ids["parent"],))
            engine._dag.connection.commit()

    _fake_route(monkeypatch, on_call=edit)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert "rolled back — a node changed or a new parent linked in during the repair" in result
    assert _L3_TRUNCATION_MARKER in engine._dag.get_node(ids["truncated"]).summary


# -- Round 4: a repair commit between a condensation's selection and its registration ----------------------------


def _frontier_with_fragment(engine):
    rows = _rows(engine, "s1", 4)
    fragment = _node(engine, "s1", 0, _deterministic_truncate(LONG_SOURCE, 512), rows[:1])
    others = [_node(engine, "s1", 0, f"Healthy leaf {row}.", [row]) for row in rows[1:]]
    return fragment, others


def _capture_condensation(monkeypatch):
    inputs = []

    def live_summary(*, text, **_kwargs):
        inputs.append(text)
        return f"Arc summary {len(inputs)}.", 1

    monkeypatch.setattr(engine_mod, "summarize_with_escalation", live_summary)
    return inputs


def _after_first_call(monkeypatch, owner, name, action):
    original, fired = getattr(owner, name), []

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        if not fired:
            fired.append(1)
            action()
        return result

    monkeypatch.setattr(owner, name, wrapped)


def test_r4a_the_sweep_condenses_the_text_a_repair_committed_after_selection(engine, monkeypatch):
    fragment, _others = _frontier_with_fragment(engine)
    _fake_route(monkeypatch)
    inputs = _capture_condensation(monkeypatch)
    repairs = []
    _after_first_call(monkeypatch, engine, "_select_threshold_sweep_condensation_group",
                      lambda: repairs.append(handle_lcm_command("doctor repair level3 apply", engine)))

    passes, _reason = engine._run_threshold_sweep_condensation(
        target_tokens=0, pass_budget=1, deadline=time.monotonic() + 120)

    assert passes == 1 and "groups_repaired: 1" in repairs[0]
    repaired = engine._dag.get_node(fragment).summary
    assert _L3_TRUNCATION_MARKER not in repaired and repaired in inputs[0]
    assert _L3_TRUNCATION_MARKER not in inputs[0]


def test_r4b_maybe_condense_condenses_the_text_a_repair_committed_after_selection(engine, monkeypatch):
    fragment, _others = _frontier_with_fragment(engine)
    _fake_route(monkeypatch)
    inputs = _capture_condensation(monkeypatch)
    repairs = []
    _after_first_call(monkeypatch, engine._dag, "get_uncondensed_at_depth",
                      lambda: repairs.append(handle_lcm_command("doctor repair level3 apply", engine)))

    assert engine._maybe_condense() == 1

    assert "groups_repaired: 1" in repairs[0]
    repaired = engine._dag.get_node(fragment).summary
    assert repaired in inputs[0] and _L3_TRUNCATION_MARKER not in inputs[0]


def test_r4c_a_node_deleted_before_registration_skips_the_pass(engine, monkeypatch):
    fragment, others = _frontier_with_fragment(engine)
    inputs = _capture_condensation(monkeypatch)
    conn = engine._dag.connection

    def delete_one():
        conn.execute("DELETE FROM summary_nodes WHERE node_id = ?", (others[0],))
        conn.commit()

    _after_first_call(monkeypatch, engine, "_select_threshold_sweep_condensation_group", delete_one)

    passes, reason = engine._run_threshold_sweep_condensation(
        target_tokens=0, pass_budget=1, deadline=time.monotonic() + 120)

    parents = [node for node in engine._dag.get_session_nodes("s1") if node.depth == 1]
    assert passes == 1 and reason == "pass_budget_exhausted" and len(inputs) == 1  # skipped, then re-selected
    assert len(parents) == 1 and others[0] not in parents[0].source_ids
    assert sorted(parents[0].source_ids) == sorted([fragment, *others[1:]])


def test_a_repaired_leaf_externalizes_tool_output_under_its_own_session(engine, monkeypatch):
    """Repair of another session's leaf must not label that session's tool output with the bound session."""
    rows = [engine._store.append("s2", {"role": "user", "content": "run the report"}, token_estimate=4),
            engine._store.append("s2", {"role": "tool", "tool_call_id": "call-s2", "content": "report body " * 40},
                                 token_estimate=120)]
    leaf = _node(engine, "s2", 0, _deterministic_truncate(LONG_SOURCE, 512), rows)
    sessions = []

    def capture(content, *, tool_call_id, session_id, **_kwargs):
        sessions.append((tool_call_id, session_id))
        return None

    monkeypatch.setattr(engine_mod, "maybe_externalize_tool_output", capture)
    _fake_route(monkeypatch)
    assert engine._session_id == "s1"

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert "groups_repaired: 1" in result and f"top node {leaf}" in result
    assert sessions == [("call-s2", "s2")]


# -- Round 6 (bot threads at fda0d60c): the repair's own call budget, its own route keys, the prompt version, the model --


def test_r6a_the_repair_never_uses_the_live_spend_budget(engine, monkeypatch):
    _build_store(engine)
    seen = []

    def fake(*, text, depth, spend_guard=None, **_kwargs):
        seen.append(spend_guard)
        assert spend_guard is not None and spend_guard.try_record_call()
        return f"Model summary at d{depth}.\nExpand for details about: repaired", 1

    monkeypatch.setattr(level3_repair, "summarize_with_escalation", fake, raising=False)

    assert "status: ok" in handle_lcm_command("doctor repair level3 apply", engine)

    assert seen and all(guard is not engine._summary_spend_guard for guard in seen)
    assert engine._summary_spend_guard._calls == [] and engine._summary_spend_guard.allows()


def test_r6b_the_route_check_reads_the_repair_keys(engine, monkeypatch):
    _build_store(engine)
    _fake_route(monkeypatch)
    breaker, model = engine._summary_circuit_breaker, engine._config.summary_model
    for _ in range(max(1, breaker.failure_threshold)):
        breaker.record_failure(model)  # the live route is open; the repair keys are healthy
    assert not breaker.allows(model)

    assert "status: ok" in handle_lcm_command("doctor repair level3 apply", engine)


def test_r6b_an_open_repair_route_refuses_before_the_backup(engine, monkeypatch):
    _build_store(engine)
    calls = _fake_route(monkeypatch)
    breaker, model = engine._summary_circuit_breaker, engine._config.summary_model
    for _ in range(max(1, breaker.failure_threshold)):
        breaker.record_failure("repair:" + model)  # the repair keys are open; the live route is healthy
    assert breaker.allows(model)

    result = handle_lcm_command("doctor repair level3 apply", engine)

    assert "status: refused" in result and calls == []
    assert not engine.backup_dir().exists() or not any(engine.backup_dir().iterdir())


def test_r6c_the_repair_uses_the_configured_prompt_version(engine, monkeypatch):
    _build_store(engine)
    engine._config.summary_prompt_version = 2
    versions = []

    def fake(*, text, depth, prompt_version=1, **_kwargs):
        versions.append(prompt_version)
        return f"Model summary at d{depth}.\nExpand for details about: repaired", 1

    monkeypatch.setattr(level3_repair, "summarize_with_escalation", fake, raising=False)

    assert "status: ok" in handle_lcm_command("doctor repair level3 apply", engine)
    assert versions and set(versions) == {2}


def test_r6d_the_provenance_row_names_the_repair_model(engine, monkeypatch):
    ids = _build_store(engine)
    conn = engine._dag.connection
    conn.execute("INSERT OR REPLACE INTO summary_node_provenance VALUES (?, 3, 'old-model', 7)", (ids["truncated"],))
    conn.commit()

    def fake(*, text, depth, provenance=None, **_kwargs):
        provenance["model"] = "repair-model"
        return f"Model summary at d{depth}.\nExpand for details about: repaired", 1

    monkeypatch.setattr(level3_repair, "summarize_with_escalation", fake, raising=False)

    assert "status: ok" in handle_lcm_command("doctor repair level3 apply", engine)
    assert conn.execute("SELECT escalation_level, model, created_at FROM summary_node_provenance WHERE node_id = ?",
                        (ids["truncated"],)).fetchone() == (1, "repair-model", 7.0)
