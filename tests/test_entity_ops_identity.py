"""W1a no-effect guards: run unchanged on the base and on the candidate."""

import ast
from contextlib import closing
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.entity_store import EntityStore
from hermes_lcm.store import MessageStore


def test_flag_off_schema_and_turn_path(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
                       hermes_home=str(tmp_path / "home"))
    try:
        schema = [tuple(r) for r in engine._store.connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
        assert not any("lcm_entit" in str(r) or "lcm_memory_" in str(r) for r in schema)
        assert engine._entities is None
    finally:
        engine.shutdown()
    root = Path(__file__).resolve().parents[1]
    for path in root.glob("*.py"):
        if path.name == "entity_ops.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "entity_ops", path
            elif isinstance(node, ast.Import):
                assert not any(n.name.endswith("entity_ops") for n in node.names), path


def test_s_r4_event_only_kind_requires_trusted_reduction(tmp_path):
    """W1-r1: trusted reduction resolves kind without changing S's source contract."""
    from hermes_lcm.entity_ops import resolved_view
    path = tmp_path / "lcm.db"
    with closing(MessageStore(path)), closing(EntityStore(path, profile_key="named-profile")) as store:
        entity = store.create_entity("Alex", "person")
        alias = store.add_alias(entity, "Alex")
        access = dict(access_scope="P1", principal_is_owner=True)
        before = (store.get_entity(entity, **access), store.get_alias(alias, **access))
        assert resolved_view(store, **access)["cards"][entity]["kind"] == "person"
        store.append_event("change_kind", {"kind": {"entity_id": entity, "before": "person", "after": "org"}})
        after = (store.get_entity(entity, **access), store.get_alias(alias, **access))
        assert before == after
        assert after[0]["kind"] == "person"
        assert resolved_view(store, **access)["cards"][entity]["kind"] == "org"
        with pytest.raises(ValueError, match="trusted replay only"):
            store.events_after(**access)
        assert store.events_after()[-1]["op"] == "change_kind"
