"""#1598 acceptance 4: synthetic copies, no input writes or payload output."""

import copy
import hashlib
import importlib.util
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

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
    files = _copy_files(path)
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **kwargs)
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(before).digest()
    assert _copy_files(path) == files
    output = json.dumps(result)
    assert "private payload" not in output and "synthetic-capture" not in output and "conv" not in output
    return result


def _copy_files(path):
    # The CLI can create siblings of the copy, never files under the fixture's
    # engine. Its background shutdown can still remove its own WAL/SHM files.
    return {p for p in path.parent.iterdir() if p.name != "fixture"}


def test_v5_optional_time_columns_leave_view_unknown(tmp_path):
    messages = _messages()
    path = _copy(tmp_path, messages)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX IF EXISTS idx_msg_conversation_observed")
        for column in ("ingested_at", "observed_at", "observed_at_source"):
            connection.execute(f"ALTER TABLE messages DROP COLUMN {column}")
    result = _inspect(path, **_inputs(path, messages))
    assert result["source_rows"] == len(messages)
    assert result["view_state"] == "unknown" and not result["healthy"]
    assert result["evidence_code"] == "source_time_unavailable"
    assert result["unreachable_source_ids"] is None


@pytest.mark.parametrize("owner", [None, "", " \t ", " conv ", "\tconv\t"])
def test_copy_groups_use_normalized_conversation_ownership(tmp_path, owner):
    path = _copy(tmp_path, [{"role": "user", "content": "private payload repeated"}] * 2)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET conversation_id=? WHERE store_id=1", (owner,))
    result = _inspect(path)
    assert result["source_rows"] == 2
    assert result["polluted_copies"] == "suspected" and result["polluted_source_ids"] == [1, 2]


@pytest.mark.parametrize("names", [("terminal", ""), ("", "terminal")])
def test_salvage_pair_accepts_missing_tool_name(tmp_path, names):
    path = _copy(tmp_path, _messages(2))
    with sqlite3.connect(path) as connection:
        for store_id, name in enumerate(names, 1):
            connection.execute("UPDATE messages SET role='tool',tool_call_id='call',tool_name=?,content=? "
                               "WHERE store_id=?", (name, "private payload" if store_id == 1
                                                     else inspector._HERMES_PRUNED_TOOL_PLACEHOLDER, store_id))
    result = _inspect(path)
    assert result["polluted_copies"] == "suspected" and result["polluted_source_ids"] == [1, 2]


def test_ignored_view_candidate_is_filtered_before_leaf_selection(tmp_path, monkeypatch):
    # Exercise selection even in minimal installs without the optional regex
    # package; this fixture pattern is bounded and uses the timeout interface.
    class Pattern:
        def __init__(self, pattern):
            self.compiled = re.compile(pattern)

        def search(self, text, *, timeout):
            return self.compiled.search(text)

    monkeypatch.setattr(inspector, "compile_message_patterns", lambda patterns: [Pattern(p) for p in patterns])
    messages = [{**m, "role": "user"} for m in _messages(4)]
    path = _copy(tmp_path, messages)
    inputs = _inputs(path, [messages[0], messages[2], messages[1]], selection_budget=1)
    inputs["effective_config"]["lcm_config"]["ignore_message_patterns"] = ["private payload 0"]
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    result = _inspect(path, **inputs)
    assert result["empty_input_kind"] == "retained" and result["selection_failed"]
    assert result["blocker_source_ids"] == [2] and result["blocker_in_fresh_tail"]


def test_reused_session_mapping_keeps_foreign_replay_candidates(tmp_path):
    messages = _messages(3)
    path = _copy(tmp_path, [messages[0], *messages])
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET conversation_id='other' WHERE store_id=1")
    result = _inspect(path, **_inputs(path, messages))
    assert result["source_rows"] == 3
    assert result["unreachable_source_ids"] == [2]


@pytest.mark.parametrize("shape", ["foreign", "mixed", "missing"])
def test_reused_session_dag_uses_conversation_lineage(tmp_path, shape):
    messages = _messages(3)
    path = _copy(tmp_path, messages)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET conversation_id='other' WHERE store_id=1")
        connection.execute("INSERT INTO summary_nodes(node_id,session_id,summary,source_ids,source_type,created_at) "
                           "VALUES (1,'S','private summary','[1]','messages',0)")
        sources = {"foreign": "[1]", "mixed": "[1,2]", "missing": "[1,999]"}[shape]
        connection.execute("INSERT INTO summary_nodes(node_id,session_id,summary,source_ids,source_type,created_at) "
                           "VALUES (2,'S','private summary',?,'messages',0)", (sources,))
        connection.execute("INSERT INTO summary_nodes(node_id,session_id,summary,source_ids,source_type,created_at) "
                           "VALUES (3,'S','private summary','[1,2]','nodes',0)")
    if shape != "foreign":
        with pytest.raises(inspector.InspectionRefused, match="^invalid_dag$"):
            _inspect(path)
    else:
        result = _inspect(path, **_inputs(path, messages[1:]))
        assert result["source_rows"] == 2 and result["covered_rows"] == 0
        assert result["unreachable_source_ids"] == [] and result["healthy"]


def test_witnessed_composite_view_leaves_attribution_unknown(tmp_path):
    messages = [{"role": "user", "content": "private payload R", "timestamp": 500.0},
                {"role": "user", "content": "private payload U", "timestamp": 510.0}]
    path = _copy(tmp_path, messages)
    with sqlite3.connect(path) as connection:
        connection.executemany("INSERT INTO message_relations "
                               "(store_id,kind,related_store_id,ordinal,observed_at,created_at) "
                               "VALUES (1,'composite',?,?,500,1)", [(1, 0), (2, 1)])
    view = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"],
             "timestamp": 500.0}, {"role": "assistant", "content": "private payload reply"}]
    result = _inspect(path, **_inputs(path, view))
    assert result["view_state"] == "unknown" and not result["healthy"]
    assert result["evidence_code"] == "identity_anchor_unsupported"
    assert result["unreachable_source_ids"] is None and result["prefix_over_leaf_budget"] is None
    assert "blocker_source_ids" not in result and "selection_failed" not in result


@pytest.mark.parametrize("sources", ["{", '"private lineage"', "null", "{}", "[true]", "[0]"])
def test_malformed_summary_sources_raise_invalid_dag(tmp_path, sources):
    path = _copy(tmp_path, _messages())
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                           "VALUES ('S','private summary',?,'messages',0)", (sources,))
    with pytest.raises(inspector.InspectionRefused, match="^invalid_dag$"):
        _inspect(path)


@pytest.mark.parametrize("frontier", ["private frontier marker", 2.5, b"private frontier marker"])
def test_noninteger_frontier_refused_without_echo(tmp_path, capsys, frontier):
    path = _copy(tmp_path, _messages(), frontier=frontier)
    code = inspector.main(["inspect", "--copy", str(path), "--session-id", "S",
                           "--conversation-id", "conv"])
    captured = capsys.readouterr()
    assert code == 2 and captured.out == '{"error": "invalid_store"}\n'
    assert captured.err == ""


@pytest.mark.parametrize("stage", ["inspect", "serialize"])
def test_cli_guards_unexpected_exceptions_and_serialization(tmp_path, monkeypatch, capsys, stage):
    path = _copy(tmp_path, _messages())

    def fail(*_args, **_kwargs):
        raise AttributeError("private payload and path")

    if stage == "inspect":
        monkeypatch.setattr(inspector, "inspect_store", fail)
    else:
        monkeypatch.setattr(inspector, "inspect_store", lambda *_a, **_k: {"healthy": False})
        # A module-local double: patching json.dumps itself would also break
        # unrelated background threads that serialize while this test runs.
        monkeypatch.setattr(inspector, "json", SimpleNamespace(dumps=fail, loads=json.loads))
    code = inspector.main(["inspect", "--copy", str(path), "--session-id", "S",
                           "--conversation-id", "conv"])
    captured = capsys.readouterr()
    assert code == 2 and captured.out == '{"error": "invalid_input"}\n'
    assert captured.err == ""


@pytest.mark.parametrize("count", [0, 1, 4])
def test_entire_view_in_fresh_tail_matches_product_noop(tmp_path, count):
    messages = _messages(count)
    engine = make_engine(tmp_path / "product", window=10000, session_id="S",
                         conversation_id="conv", fresh_tail_count=4)
    try:
        engine.ingest(copy.deepcopy(messages))
        engine.compress(copy.deepcopy(messages), force=True)
        expected = "no eligible raw backlog outside fresh tail" if count else "empty message list"
        assert engine._last_compression_noop_reason == expected
    finally:
        engine.shutdown()
    path = _copy(tmp_path, messages)
    result = _inspect(path, **_inputs(path, messages, fresh_tail_count=4))
    assert result["empty_input_kind"] == "none"
    assert not result["selection_failed"] and result["healthy"]


def test_empty_raw_view_with_hidden_backlog_is_not_a_noop(tmp_path):
    messages = _messages()
    path = _copy(tmp_path, messages)
    result = _inspect(path, **_inputs(path, messages[1:], fresh_tail_count=4))
    assert not result["healthy"] and result["unreachable_source_ids"] == [1]


def test_first_tool_content_is_pruned_placeholder_not_pollution(tmp_path):
    messages = [{"role": "user", "content": "private payload"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call", "name": "terminal",
                 "content": inspector._HERMES_PRUNED_TOOL_PLACEHOLDER},
                {"role": "assistant", "content": "done"}]
    result = _inspect(_copy(tmp_path, messages))
    assert result["polluted_copies"] == "none" and result["polluted_source_ids"] == []


@pytest.mark.parametrize("case", ["missing_messages", "bad_messages", "config_extra", "config_type"])
def test_malformed_view_evidence_keeps_database_findings(tmp_path, case):
    messages = _messages()
    path = _copy(tmp_path, messages)
    inputs = _inputs(path, messages)
    if case == "missing_messages":
        inputs["view_snapshot"].pop("messages")
    elif case == "bad_messages":
        inputs["view_snapshot"]["messages"] = [{"role": "assistant", "tool_calls": [{"function": "bad"}]}]
    elif case == "config_extra":
        inputs["effective_config"]["lcm_config"]["unexpected"] = True
    else:
        inputs["effective_config"]["lcm_config"]["ignore_message_patterns"] = 42
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    result = _inspect(path, **inputs)
    assert result["view_state"] == "unknown" and not result["healthy"]
    assert result["source_rows"] == 4 and result["polluted_copies"] == "none"
    assert result["evidence_code"] in {"snapshot_invalid", "config_incomplete"}


@pytest.mark.parametrize("option", ["--view-snapshot", "--effective-config"])
def test_malformed_evidence_file_keeps_database_findings(tmp_path, capsys, option):
    path = _copy(tmp_path, _messages())
    bad = tmp_path / "evidence.json"
    bad.write_text('{"private payload":')
    code = inspector.main(["inspect", "--copy", str(path), "--session-id", "S",
                           "--conversation-id", "conv", option, str(bad)])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert code == 0 and captured.err == ""
    assert result["view_state"] == "unknown" and result["source_rows"] == 4
    assert result["polluted_copies"] == "none" and not result["healthy"]


def test_documented_backup_conversion_is_accepted(tmp_path, monkeypatch):
    messages = _messages()
    _copy(tmp_path, messages)
    # Execute the operator procedure verbatim on a synthetic WAL-mode source.
    with closing(sqlite3.connect(tmp_path / "fixture/lcm.db")) as source:
        assert source.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        with closing(sqlite3.connect(tmp_path / "SOURCE.db")) as destination:
            source.backup(destination)
    doc = (Path(__file__).resolve().parents[1] / "docs/operations/store-wedge-inspect.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", doc, re.DOTALL)
    assert len(blocks) == 1, "document the executable backup and journal-mode conversion"
    monkeypatch.chdir(tmp_path)
    exec(compile(blocks[0], "documented-copy-procedure", "exec"), {"__name__": "__main__"})
    path = tmp_path / "COPY.db"
    assert path.read_bytes()[18:20] == b"\x01\x01"
    assert not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))
    assert _inspect(path)["source_rows"] == len(messages)


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
    before, files = path.read_bytes(), _copy_files(path)
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
    assert path.read_bytes() == before and _copy_files(path) == files


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecars_refused_before_open(tmp_path, monkeypatch, suffix):
    path = _copy(tmp_path, _messages())
    Path(str(path) + suffix).write_bytes(b"private payload")
    before, files = path.read_bytes(), _copy_files(path)
    monkeypatch.setattr(inspector, "sqlite3", SimpleNamespace(
        connect=lambda *_a, **_k: pytest.fail("opened sidecar store"),
        Row=sqlite3.Row, Error=sqlite3.Error,
    ))
    # Rollup maintenance can open its own schema-check connection while the
    # inspector's refusal double is installed. That unrelated read must work.
    def background_read():
        with closing(sqlite3.connect(":memory:")) as connection:
            return connection.execute("SELECT 1").fetchone()

    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(background_read).result(timeout=5) == (1,)
    with pytest.raises(inspector.InspectionRefused, match="^sidecar_present$"):
        inspector.inspect_store(path, session_id="S", conversation_id="conv")
    assert path.read_bytes() == before and _copy_files(path) == files


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
    before, files = path.read_bytes(), _copy_files(path)
    with pytest.raises(inspector.InspectionRefused):
        inspector.inspect_store(path, session_id="S", conversation_id="conv")
    assert path.read_bytes() == before and _copy_files(path) == files


def test_cli_emits_fixed_errors_without_payload_or_paths(tmp_path, capsys):
    bad = tmp_path / "private-payload.json"
    bad.write_text('{"sensitive":')
    code = inspector.main(["inspect", "--copy", str(tmp_path / "missing"), "--session-id", "private-scope",
                           "--conversation-id", "private-conversation", "--effective-config", str(bad), "--json"])
    assert code == 2 and capsys.readouterr().out == '{"error": "invalid_input"}\n'
