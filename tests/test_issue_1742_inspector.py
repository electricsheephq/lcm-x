"""Generated-fixture acceptance cases for #1742; no live stores or host files."""

import importlib.util
import copy
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

import pytest

from tests.test_store_wedge_inspect import _copy, _inputs, _messages, counter  # noqa: F401

ROOT = Path(os.environ.get("LCMX_INSPECTOR_ROOT", Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("inspector_1742", ROOT / "scripts/store_wedge_inspect.py")
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


@pytest.fixture(autouse=True)
def patterns(monkeypatch):
    class Pattern:
        def __init__(self, text):
            self.pattern = re.compile(text)

        def search(self, text, *, timeout):
            return self.pattern.search(text)

    monkeypatch.setattr(inspector, "compile_message_patterns", lambda values: [Pattern(p) for p in values])


def inspect(path, messages=None, **overrides):
    before = path.read_bytes()
    siblings = set(path.parent.iterdir())
    inputs = {} if messages is None else _inputs(path, messages, **overrides)
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert path.read_bytes() == before
    assert set(path.parent.iterdir()) == siblings
    return result


def test_item01_unwitnessed_composite_refuses_before_attribution(tmp_path):
    messages = [{"role": "user", "content": "R", "timestamp": 500.0},
                {"role": "user", "content": "U", "timestamp": 510.0}]
    path = _copy(tmp_path, messages)
    view = [{"role": "user", "content": "R\n\nU", "timestamp": 500.0}]
    result = inspect(path, view)
    assert result.get("evidence_code") == "identity_anchor_unsupported"
    assert result["unreachable_source_ids"] is None


def test_item02_dependent_reply_stays_in_selection(tmp_path, monkeypatch):
    messages = _messages(4)
    path = _copy(tmp_path, messages)
    inputs = _inputs(path, messages, leaf_chunk_tokens=1, selection_budget=1)
    inputs["effective_config"]["lcm_config"]["ignore_message_patterns"] = ["payload 0"]
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    selected = []
    original = inspector.select_oldest_leaf_chunk

    def select(raw, budget):
        chunk = original(raw, budget)
        selected.append([messages.index(m) + 1 for m in chunk])
        return chunk

    monkeypatch.setattr(inspector, "select_oldest_leaf_chunk", select)
    inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    expected = original(messages[1:3], 1)
    assert selected[0] == [messages.index(m) + 1 for m in expected] == [2]


def test_item03_foreign_only_fresh_tail_is_unassessed(tmp_path):
    messages = _messages(1)
    path = _copy(tmp_path, messages)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET conversation_id='other'")
    result = inspect(path, messages)
    assert not result["healthy"] and result["view_state"] == "unknown"
    assert result["unreachable_source_ids"] is None


@pytest.mark.parametrize("sources", ["[]", "[2,1]", "[1,1]"])
@pytest.mark.parametrize("kind", ["messages", "nodes"])
def test_item04_malformed_lineage_order_is_refused(tmp_path, sources, kind):
    path = _copy(tmp_path, _messages(2))
    with sqlite3.connect(path) as connection:
        for node_id in (1, 2):
            connection.execute("INSERT INTO summary_nodes(node_id,session_id,summary,source_ids,source_type,created_at) "
                               "VALUES (?,'S','summary',?,'messages',?)", (node_id, f"[{node_id}]", node_id))
        connection.execute("INSERT INTO summary_nodes(node_id,session_id,summary,source_ids,source_type,created_at) "
                           "VALUES (3,'S','summary',?,?,0)", (sources, kind))
    if sources == "[]" and kind == "messages":
        assert inspect(path)["covered_rows"] == 2
    else:
        with pytest.raises(inspector.InspectionRefused, match="^invalid_dag$"):
            inspect(path)


def test_item05_persisted_dependent_reply_is_unassessed(tmp_path):
    messages = _messages(3)
    path = _copy(tmp_path, messages)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES "
                           "('ignored_dependent_reply_hashes:S', ?)",
                           (json.dumps([{"store": hashlib.sha256(b"S\x002").hexdigest()[:16]}]),))
    result = inspect(path, messages, leaf_chunk_tokens=1)
    assert result.get("evidence_code") == "unsupported_view_evidence"
    assert result["unreachable_source_ids"] is None


def test_item06_below_threshold_admission_is_unassessed(tmp_path):
    messages = _messages(3)
    path = _copy(tmp_path, messages)
    result = inspect(path, messages, leaf_chunk_tokens=10000)
    assert result.get("evidence_code") == "leaf_admission_unassessed"
    assert result["prefix_over_leaf_budget"] is None
    assert result["unreachable_source_ids"] is None
    assert "selection_failed" not in result


@pytest.mark.parametrize("source", [None, "", " \t "])
def test_item07_legacy_source_spelling_does_not_split_copies(tmp_path, source):
    path = _copy(tmp_path, [_messages(1)[0]] * 2)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET source=? WHERE store_id=1", (source,))
        connection.execute("UPDATE messages SET source='unknown' WHERE store_id=2")
    result = inspect(path)
    assert result["polluted_copies"] == "suspected"
    assert result["polluted_source_ids"] == [1, 2]


@pytest.mark.parametrize("key", ["message_uid", "_lcm", "store_id"])
def test_item08_special_identity_refuses_before_findings(tmp_path, monkeypatch, key):
    messages = _messages(3)
    path = _copy(tmp_path, messages)
    view = copy.deepcopy(messages)
    view[0][key] = 1 if key == "store_id" else "synthetic"
    calls = []
    original = inspector._Probe._get_store_id_map_for_messages

    def mapping(self, view):
        calls.append(True)
        return original(self, view)

    monkeypatch.setattr(inspector._Probe, "_get_store_id_map_for_messages", mapping)
    result = inspect(path, view, leaf_chunk_tokens=1)
    assert result.get("evidence_code") == "unsupported_view_evidence"
    assert not calls
    assert result["unreachable_source_ids"] is None
    assert result["prefix_over_leaf_budget"] is None


def test_item09_parent_carry_lineage_is_not_invalid_dag(tmp_path):
    path = _copy(tmp_path, _messages(3))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE messages SET session_id='P' WHERE store_id IN (1,2)")
        connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES "
                           "('identity_anchor_carry:S','[[\"P\",0,2]]')")
        connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                           "VALUES ('S','summary','[1,2]','messages',0)")
    result = inspect(path)
    assert result["source_rows"] == 1 and not result["healthy"]


def test_item10_mapped_stored_ignore_filter(tmp_path, monkeypatch):
    messages = _messages(3)
    messages[0]["content"] = "[Note: model was just switched from A to B]\n" + messages[0]["content"]
    path = _copy(tmp_path, messages)
    view = copy.deepcopy(messages)
    view[0]["content"] = view[0]["content"].split("\n", 1)[1]
    inputs = _inputs(path, view, leaf_chunk_tokens=1, selection_budget=1)
    inputs["effective_config"]["lcm_config"]["ignore_message_patterns"] = ["model was just switched"]
    inputs["view_snapshot"]["effective_config"] = copy.deepcopy(inputs["effective_config"])
    selections = []
    original = inspector.select_oldest_leaf_chunk

    def select(raw, budget):
        selected = original(raw, budget)
        selections.append([view.index(m) + 1 for m in selected])
        return selected

    monkeypatch.setattr(inspector, "select_oldest_leaf_chunk", select)
    result = inspector.inspect_store(path, session_id="S", conversation_id="conv", **inputs)
    assert result.get("evidence_code") is None
    assert selections[0] == [2]


def test_item11_bound_twin_leaves_view_unassessed(tmp_path):
    messages = [{"role": "assistant", "content": "reply"}] * 2 + _messages(1)
    path = _copy(tmp_path, messages)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO summary_nodes(session_id,summary,source_ids,source_type,created_at) "
                           "VALUES ('S','summary','[1]','messages',0)")
        connection.execute("CREATE TABLE host_uid_bindings "
                           "(store_id INTEGER,uid TEXT,lineage_key TEXT,binding_version INTEGER,kind TEXT)")
        connection.execute("INSERT INTO host_uid_bindings"
                           "(store_id,uid,lineage_key,binding_version,kind) VALUES (1,'uid','lineage',1,'canonical')")
    result = inspect(path, messages[-1:], leaf_chunk_tokens=1)
    assert result.get("evidence_code") == "copy_exclusions_unassessed"
    assert result["unreachable_source_ids"] is None


@pytest.mark.parametrize("table,column", [("summary_nodes", "created_at"),
                                         ("summary_nodes", "expand_hint"),
                                         ("lcm_lifecycle_state", "debt_kind"),
                                         ("lcm_lifecycle_state", "updated_at"),
                                         ("lcm_migration_state", "completed_at")])
def test_item12_complete_v5_core_shape(tmp_path, table, column):
    path = _copy(tmp_path, _messages(1))
    with sqlite3.connect(path) as connection:
        for (name,) in connection.execute("SELECT name FROM sqlite_schema WHERE type='trigger' AND tbl_name=?",
                                          (table,)).fetchall():
            connection.execute(f'DROP TRIGGER "{name}"')
        # Remove fixture indexes referring to the selected column, then drop it.
        for name, sql in connection.execute("SELECT name,sql FROM sqlite_schema WHERE type='index' AND tbl_name=?",
                                             (table,)).fetchall():
            if sql and column in sql:
                connection.execute(f'DROP INDEX "{name}"')
        connection.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')
    with pytest.raises(inspector.InspectionRefused, match="^schema_mismatch$"):
        inspect(path)


def test_item13_documented_header_check_reads_twenty_bytes(tmp_path, monkeypatch):
    path = _copy(tmp_path, _messages(1))
    source = tmp_path / "SOURCE.db"
    source.write_bytes(path.read_bytes())
    doc = (ROOT / "docs/operations/store-wedge-inspect.md").read_text()
    block = re.findall(r"```python\n(.*?)```", doc, re.DOTALL)[0]
    original_open = Path.open
    reads = []

    class Reader:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def read(self, size=-1):
            reads.append(size)
            assert size == 20
            return self.file.read(size)

    class FixturePath(type(Path())):
        def open(self, *args, **kwargs):
            file = original_open(self, *args, **kwargs)
            return Reader(file) if args and args[0] == "rb" else file

    # Substitute pathlib only in the executable documentation's import, never
    # process-wide: SQLite backup and other threads retain their own references.
    import builtins
    from types import SimpleNamespace
    original_import = builtins.__import__

    def local_import(name, *args, **kwargs):
        return SimpleNamespace(Path=FixturePath) if name == "pathlib" else original_import(name, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    exec(compile(block, "documented-copy-procedure", "exec"),
         {"__builtins__": {**vars(builtins), "__import__": local_import}})
    assert reads == [20]


def test_item14_missing_metadata_is_schema_mismatch(tmp_path):
    path = _copy(tmp_path, _messages(1))
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE metadata")
    with pytest.raises(inspector.InspectionRefused, match="^schema_mismatch$"):
        inspect(path)
