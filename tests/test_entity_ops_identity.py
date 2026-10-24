"""W1a no-effect guards: run unchanged on the base and on the candidate."""

import ast
from contextlib import closing
import os
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
    assert_no_production_imports(Path(__file__).resolve().parents[1])


def assert_no_production_imports(root):
    """Recursively reject every static/dynamic production import of the dark API."""
    excluded = {"tests", "venv", "build", "dist", "site-packages", "node_modules", "__pycache__"}
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [name for name in subdirs if not name.startswith(".") and name not in excluded]
        for name in files:
            if not name.endswith(".py") or name.startswith("."):
                continue
            path = Path(directory) / name
            tree = ast.parse(path.read_text())
            _assert_no_import_nodes(tree, path)


def _assert_no_import_nodes(tree, path):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "entity_ops" not in (node.module or "").split("."), path
            assert not any(n.name == "entity_ops" for n in node.names), path
        elif isinstance(node, ast.Import):
            assert not any("entity_ops" in n.name.split(".") for n in node.names), path
        elif isinstance(node, ast.Call) and (getattr(node.func, "attr", None) == "import_module"
                                            or getattr(node.func, "id", None) == "import_module"):
            assert not any(isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                           and "entity_ops" in arg.value.split(".")
                           for arg in node.args + [keyword.value for keyword in node.keywords]), path


@pytest.mark.parametrize("form", [
    "import hermes_lcm.entity_ops", "from hermes_lcm.entity_ops import EntityOps",
    "from hermes_lcm import entity_ops", "from . import entity_ops",
    "from .entity_ops import EntityOps", "importlib.import_module('hermes_lcm.entity_ops')",
    "import_module('.entity_ops', package='hermes_lcm')",
    "import_module(name='hermes_lcm.entity_ops')",
    "importlib.import_module(name='.entity_ops', package='hermes_lcm')",
])
@pytest.mark.parametrize("nested", [False, True])
def test_every_production_import_form_is_killed(tmp_path, form, nested):
    module = tmp_path / ("nested/module.py" if nested else "module.py")
    module.parent.mkdir(exist_ok=True)
    production = Path(__file__).resolve().parents[1] / "entity_store.py"
    module.write_text(production.read_text() + "\n" + form + "\n")
    with pytest.raises(AssertionError):
        assert_no_production_imports(tmp_path)


def test_import_guard_excludes_test_tree(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_consumer.py").write_text("from hermes_lcm.entity_ops import EntityOps\n")
    assert_no_production_imports(tmp_path)


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
