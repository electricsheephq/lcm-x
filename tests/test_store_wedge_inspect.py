"""#1598 acceptance 4: synthetic copies, no input writes or payload output."""

import copy
import hashlib
import importlib.util
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from tests.harness import make_engine, pin_char_counter

spec = importlib.util.spec_from_file_location(
    "store_wedge_inspect", Path(__file__).resolve().parents[1] / "scripts/store_wedge_inspect.py")
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


@pytest.fixture(autouse=True)
def counter(monkeypatch):
    pin_char_counter(monkeypatch)


def _messages(n=4):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"private payload {i} " * 8}
            for i in range(n)]


def _copy(tmp_path, messages, *, frontier=0):
    origin = tmp_path / "fixture"
    origin.mkdir()
    engine = make_engine(origin, window=10000, session_id="S", conversation_id="conv")
    try:
        engine.ingest(messages)
    finally:
        engine.shutdown()
    path = tmp_path / "operator-copy.db"
    with closing(sqlite3.connect((origin / "lcm.db").as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(path)) as destination:
            source.backup(destination)
            destination.execute("PRAGMA journal_mode=DELETE")
            destination.execute("UPDATE lcm_lifecycle_state SET current_frontier_store_id=?", (frontier,))
            destination.commit()
    return path


def _inputs(path, messages, **overrides):
    config = dict(leaf_chunk_tokens=1000, selection_budget=1000, effective_trigger=2000,
                  fresh_tail_count=1, fresh_tail_max_tokens=0, pressure_yield_limit=0,
                  context_window=10000, token_counter="fallback-v1",
                  lcm_config=asdict(LCMConfig(leaf_chunk_tokens=1000)))
    config.update(overrides)
    config["lcm_config"]["leaf_chunk_tokens"] = config["leaf_chunk_tokens"]
    snapshot = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), session_id="S",
                    conversation_id="conv", capture_id="synthetic-capture", messages=copy.deepcopy(messages),
                    effective_config=copy.deepcopy(config))
    return dict(view_snapshot=snapshot, effective_config=config)


def _inspect(path, **kwargs):
    before = path.read_bytes()
    files = set(path.parent.rglob("*"))
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **kwargs)
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(before).digest()
    assert set(path.parent.rglob("*")) <= files
    output = json.dumps(result)
    assert "private payload" not in output and "synthetic-capture" not in output and "conv" not in output
    return result


def test_healthy_matching_view_and_input_unchanged(tmp_path):
    messages = _messages()
    path = _copy(tmp_path, messages)
    result = _inspect(path, **_inputs(path, messages))
    assert result["healthy"] and result["view_state"] == "complete"
    assert result["unreachable_source_ids"] == []
    assert not result["prefix_over_leaf_budget"] and not result["blocker_in_fresh_tail"]
    assert result["empty_input_kind"] == "none"


@pytest.mark.parametrize("missing", ["snapshot", "config", "digest", "capture", "scope", "settings", "counter"])
def test_missing_or_mismatched_evidence_is_unknown(tmp_path, missing):
    messages = _messages()
    path = _copy(tmp_path, messages)
    inputs = _inputs(path, messages)
    if missing in {"snapshot", "config"}:
        inputs.pop("view_snapshot" if missing == "snapshot" else "effective_config")
    elif missing == "settings":
        inputs["effective_config"].pop("context_window")
        inputs["view_snapshot"]["effective_config"] = inputs["effective_config"]
    elif missing == "counter":
        inputs["effective_config"]["token_counter"] = "unavailable"
        inputs["view_snapshot"]["effective_config"] = inputs["effective_config"]
    else:
        inputs["view_snapshot"][{"digest": "source_sha256", "capture": "capture_id", "scope": "session_id"}[missing]] = ""
    result = _inspect(path, **inputs)
    assert result["view_state"] == "unknown" and not result["healthy"]
    assert result["unreachable_source_ids"] is None


def test_known_old_writer_pollution_is_suspected_without_occurrence_proof(tmp_path):
    from tests.test_issue_773_hermes_pruned_tool_replay import CLEARED

    messages = [{"role": "tool", "tool_call_id": "call", "content": "private payload original"},
                {"role": "tool", "tool_call_id": "call", "content": CLEARED}]
    path = _copy(tmp_path, messages)
    result = _inspect(path)
    assert result["polluted_copies"] == "suspected" and result["polluted_source_ids"] == [1, 2]
    assert result["view_state"] == "unknown" and not result["healthy"]


def test_equal_real_user_turns_are_never_confirmed_by_text_or_timestamp(tmp_path):
    messages = [{"role": "user", "content": "private payload repeated", "timestamp": 100.0},
                {"role": "user", "content": "private payload repeated", "timestamp": 200.0}]
    path = _copy(tmp_path, messages)
    result = _inspect(path, **_inputs(path, messages))
    assert result["polluted_copies"] == "suspected" and result["confirmed_source_ids"] == []
    assert result["source_rows"] == 2 and result["unreachable_source_ids"] == []


def test_same_uid_version_and_exact_identity_can_confirm_copies(tmp_path):
    messages = [{"role": "user", "content": "private payload repeated"}] * 2
    path = _copy(tmp_path, messages)
    with sqlite3.connect(path) as connection:
        # Existing optional table shape, synthetic observed occurrence bindings.
        connection.execute("CREATE TABLE IF NOT EXISTS host_uid_bindings "
                           "(store_id INTEGER,uid TEXT,lineage_key TEXT,binding_version INTEGER,kind TEXT)")
        connection.executemany("INSERT INTO host_uid_bindings(store_id,uid,lineage_key,binding_version,kind) "
                               "VALUES (?,?,?,?,?)", [(1, "same", "lineage", 1, "canonical"),
                                                       (2, "same", "lineage", 1, "version")])
    result = _inspect(path)
    assert result["polluted_copies"] == "confirmed" and result["confirmed_source_ids"] == [1, 2]


def test_over_budget_prefix_through_named_blocker_and_tail_are_independent(tmp_path):
    messages = _messages(5)
    path = _copy(tmp_path, messages)
    view = messages[2:] + messages[1:2]  # source 1 hidden; source 2 retained out of order
    result = _inspect(path, **_inputs(path, view, leaf_chunk_tokens=20, selection_budget=2000))
    assert result["blocker_source_ids"] == [2]
    assert result["prefix_over_leaf_budget"] and result["blocker_in_fresh_tail"]
    assert result["prefix_rows"] == 2 and result["prefix_tokens"] > result["leaf_budget"]
    assert result["unreachable_source_ids"] == [1] and not result["healthy"]


def test_first_oversized_source_is_not_a_budget_wedge(tmp_path):
    messages = _messages(3)
    path = _copy(tmp_path, messages)
    result = _inspect(path, **_inputs(path, messages[1:] + messages[:1], leaf_chunk_tokens=1, selection_budget=1000))
    assert result["blocker_source_ids"] == [1] and result["blocker_in_fresh_tail"]
    assert not result["prefix_over_leaf_budget"]
    assert result["empty_input_kind"] == "retained"


def test_reachability_checks_below_and_above_frontier_and_dag(tmp_path):
    messages = _messages(4)
    path = _copy(tmp_path, messages, frontier=2)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                           "VALUES ('S','private summary','[2]','messages',0)")
    result = _inspect(path, **_inputs(path, messages[3:]))
    assert result["covered_rows"] == 1 and result["unreachable_source_ids"] == [1, 3]


def test_no_source_branch(tmp_path):
    path = _copy(tmp_path, _messages(1), frontier=1)
    result = _inspect(path, **_inputs(path, [{"role": "user", "content": "unpersisted"}], fresh_tail_count=0))
    assert result["empty_input_kind"] == "no_source"
    assert result["view_state"] == "unknown" and not result["healthy"]


def test_contiguity_branch_uses_real_attribution_and_never_invents_blocker(tmp_path):
    messages = _messages(3)
    path = _copy(tmp_path, messages)
    before, files = path.read_bytes(), set(tmp_path.rglob("*"))
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        rows = [dict(r) for r in connection.execute("SELECT * FROM messages ORDER BY store_id")]
        probe = inspector._Probe(connection, rows, "S", 0, asdict(LCMConfig()), set())
        chunk = [messages[2], messages[0]]
        result = probe._store_complete_input(chunk, {}, {id(messages[2]): 3, id(messages[0]): 1},
                                              chunk, chunk, 0, [], 1, scanned=(rows, False))
        assert result == [] and probe._store_complete_blocked is None
        assert probe._store_complete_empty_reason == "leaf input cannot form a contiguous stored prefix"
        assert inspector._empty_input_kind(result, None, probe._store_complete_empty_reason, chunk, {}) == "contiguity"
    assert path.read_bytes() == before and set(tmp_path.rglob("*")) == files


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecars_refused_before_open(tmp_path, monkeypatch, suffix):
    path = _copy(tmp_path, _messages())
    Path(str(path) + suffix).write_bytes(b"private payload")
    before, files = path.read_bytes(), set(tmp_path.rglob("*"))
    monkeypatch.setattr(inspector.sqlite3, "connect", lambda *_a, **_k: pytest.fail("opened sidecar store"))
    with pytest.raises(inspector.InspectionRefused, match="^sidecar_present$"):
        inspector.inspect_store(path, session_id="S", conversation_id="conv")
    assert path.read_bytes() == before and set(tmp_path.rglob("*")) == files


@pytest.mark.parametrize("case", ["schema", "wal_header", "cycle", "foreign_source"])
def test_schema_wal_header_and_dag_refusals_preserve_input(tmp_path, case):
    path = _copy(tmp_path, _messages())
    with sqlite3.connect(path) as connection:
        if case == "schema":
            connection.execute("UPDATE metadata SET value='999' WHERE key='schema_version'")
        elif case == "wal_header":
            connection.execute("PRAGMA journal_mode=WAL")
        else:
            connection.execute("INSERT INTO summary_nodes(node_id,session_id,summary,source_ids,source_type,created_at) "
                               "VALUES (1,'S','private summary',?,?,0)",
                               ("[1]" if case == "cycle" else "[999]", "nodes" if case == "cycle" else "messages"))
    before, files = path.read_bytes(), set(tmp_path.rglob("*"))
    with pytest.raises(inspector.InspectionRefused):
        inspector.inspect_store(path, session_id="S", conversation_id="conv")
    assert path.read_bytes() == before and set(tmp_path.rglob("*")) == files


def test_cli_emits_fixed_errors_without_payload_or_paths(tmp_path, capsys):
    bad = tmp_path / "private-payload.json"
    bad.write_text('{"sensitive":')
    code = inspector.main(["inspect", "--copy", str(tmp_path / "missing"), "--session-id", "private-scope",
                           "--conversation-id", "private-conversation", "--effective-config", str(bad), "--json"])
    assert code == 2 and capsys.readouterr().out == '{"error": "invalid_input"}\n'
