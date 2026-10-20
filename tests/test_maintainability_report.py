"""Deterministic acceptance checks for the report-only maintainability inventory."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/maintainability/report.py"
BASELINE = SCRIPT.with_name("baseline.json")
spec = importlib.util.spec_from_file_location("maintainability_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def run(root, *arguments):
    # The subprocess timeout is an execution budget, not a timing assertion.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments, "--format", "json"],
        cwd=root, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_repository_baseline():
    result = run(ROOT, "--baseline", str(BASELINE))
    assert result["metrics"]["warnings"] == []
    assert result["metrics"]["headline"]["engine_attributes"] > 0
    # Ruff complexity was collected for the real tree (the synthetic fixtures pin the exact values).
    assert any("cc" in metrics for metrics in result["metrics"]["functions"].values())
    assert len(report.markdown(result).splitlines()) <= 60


def test_growth_and_baseline_round_trip(tmp_path):
    shutil.copyfile(ROOT / "reset_state.py", tmp_path / "reset_state.py")
    declarations = tmp_path / "scripts/maintainability/attr_lifetimes.json"
    declarations.parent.mkdir(parents=True)
    declarations.write_text("{}\n")
    baseline = tmp_path / "baseline.json"
    original = run(tmp_path, "--baseline", str(baseline), "--write-baseline")
    assert original["metrics"]["warnings"] == []
    assert json.loads(baseline.read_text()) == original["metrics"]
    assert all(not value for value in original["changes"].values())
    assert run(tmp_path, "--baseline", str(baseline)) == original
    path = tmp_path / "reset_state.py"
    path.write_text(path.read_text() + "\ndef added_long_function():\n" + "    value = 1\n" * 81)
    grown = run(tmp_path, "--baseline", str(baseline))
    assert grown["changes"]["new_large_functions"] == [{
        "function": "reset_state.py::added_long_function",
        **grown["metrics"]["functions"]["reset_state.py::added_long_function"],
    }]
    assert json.loads(baseline.read_text()) == original["metrics"]
    rewritten = run(tmp_path, "--baseline", str(baseline), "--write-baseline")
    assert all(not value for value in rewritten["changes"].values())
    assert run(tmp_path, "--baseline", str(baseline)) == rewritten


def test_metric_contracts_and_growth(tmp_path):
    (tmp_path / "engine.py").write_text('''from .mix import StateMixin
class LCMEngine(StateMixin):
    def work(self, config):
        self._state = 1
        self._pair, self._other = 2, 3
        return getattr(config, "limit", 1)
''')
    (tmp_path / "mix.py").write_text('''class StateMixin:
    def reset(self):
        self._state: int = 0
class Unrelated:
    def reset(self):
        self._unrelated = 1
''')
    (tmp_path / "store.py").write_text('''from .engine import LCMEngine
from hermes_lcm import compaction
import hermes_lcm.tools
from . import command
SQL = "CREATE TABLE sample (id); ALTER TABLE sample ADD COLUMN name; CREATE UNIQUE INDEX idx ON sample(id)"
''')
    (tmp_path / "db_bootstrap.py").write_text('SQL = "CREATE TABLE schema_table (id)"\n')
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_example.py").write_text('''from hermes_lcm.store import _private
engine._state
self._store._db
unrelated._field
''')
    old = report.collect(tmp_path)
    assert old["warnings"] == []
    assert old["attributes"] == {
        "_state": ["engine.py", "mix.py"], "_pair": ["engine.py"], "_other": ["engine.py"],
    }
    assert len(old["imports"]) == 4
    assert len(old["config_getattr"]) == 1
    assert len(old["private_refs"]) == 3
    assert len(old["ddl_outside_schema"]) == 3
    assert old["headline"]["ddl_modules"] == 2
    (tmp_path / "mix.py").write_text('''class StateMixin:
    def reset(self):
        self._state = 0
        self._pair = 1
        self._new = 2
        if self._pair:
            self._new = 3
''')
    new = report.collect(tmp_path)
    changes = report.compare(new, old)
    assert changes["new_attributes"] == ["_new"]
    assert changes["new_writers"] == {"_pair": ["mix.py"]}
    assert changes["function_growth"] == [{
        "function": "mix.py::StateMixin.reset",
        "delta": {"loc": 4, "cc": 1, "branches": 1, "statements": 4},
    }]
    assert not changes["new_config_getattr"]
    # Moving source lines must not manufacture new sites.
    path = tmp_path / "engine.py"
    path.write_text("\n\n" + path.read_text())
    assert not report.compare(report.collect(tmp_path), new)["new_config_getattr"]


def test_new_complex_short_function(tmp_path):
    source = "def branchy(value):\n" + "    if value:\n        pass\n" * 16
    (tmp_path / "branchy.py").write_text(source)
    changes = report.compare(report.collect(tmp_path), {})
    assert changes["new_large_functions"][0]["cc"] == 17
    assert changes["new_large_functions"][0]["loc"] < 80


def test_incomplete_collection_is_report_only(tmp_path, monkeypatch):
    (tmp_path / "engine.py").write_text("def broken(:\n")
    result = run(tmp_path, "--baseline", str(tmp_path / "baseline.json"), "--write-baseline")
    assert result["metrics"]["warnings"]
    assert not (tmp_path / "baseline.json").exists()
    (tmp_path / "engine.py").write_text("def valid():\n    pass\n")
    monkeypatch.setenv("PATH", "")
    result = run(tmp_path, "--baseline", str(tmp_path / "baseline.json"), "--write-baseline")
    assert "Ruff metrics unavailable" in result["metrics"]["warnings"][0]
    assert not (tmp_path / "baseline.json").exists()


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root can list a mode-000 directory")
def test_unreadable_directory_blocks_the_baseline_write(tmp_path):
    (tmp_path / "engine.py").write_text("def valid():\n    pass\n")
    hidden = tmp_path / "pkg"
    hidden.mkdir()
    (hidden / "store.py").write_text("def other():\n    pass\n")
    hidden.chmod(0)
    try:
        result = run(tmp_path, "--baseline", str(tmp_path / "baseline.json"), "--write-baseline")
    finally:
        hidden.chmod(0o755)
    assert "Could not read pkg" in result["metrics"]["warnings"]
    assert not (tmp_path / "baseline.json").exists()


def test_standalone_f_string_is_not_ddl(tmp_path):
    (tmp_path / "store.py").write_text('''def note(name):
    f"CREATE TABLE {name} (id) is documented here, not executed"
    return f"ALTER TABLE {name} ADD COLUMN x"
''')
    result = run(tmp_path)["metrics"]
    assert [site["detail"] for site in result["ddl_outside_schema"].values()] == ["ALTER TABLE"]


def test_write_replaces_an_incompatible_baseline(tmp_path):
    (tmp_path / "engine.py").write_text("def valid():\n    pass\n")
    lifetimes = tmp_path / "scripts/maintainability/attr_lifetimes.json"
    lifetimes.parent.mkdir(parents=True)
    lifetimes.write_text("{}")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"version": 0}))
    result = run(tmp_path, "--baseline", str(baseline), "--write-baseline")
    assert json.loads(baseline.read_text())["version"] == result["metrics"]["version"]
    assert result["metrics"]["warnings"] == []


def test_ddl_evaluated_inside_a_standalone_f_string_is_reported(tmp_path):
    (tmp_path / "store.py").write_text('''def hidden(conn):
    f"{conn.execute('CREATE TABLE hidden (id)')}"
''')
    result = run(tmp_path)["metrics"]
    assert [site["detail"] for site in result["ddl_outside_schema"].values()] == ["CREATE TABLE"]


def lifetime_tree(root, declarations):
    (root / "engine.py").write_text('''class LCMEngine:
    def __init__(self):
        self._existing = 0
        self._new = 1
''')
    baseline = root / "baseline.json"
    baseline.write_text(json.dumps({"version": 1, "attributes": {"_existing": ["engine.py"]}}))
    path = root / "scripts/maintainability/attr_lifetimes.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(declarations))
    return baseline, path


def test_new_attribute_without_lifetime(tmp_path):
    baseline, _ = lifetime_tree(tmp_path, {})
    result = run(tmp_path, "--baseline", str(baseline))
    assert result["changes"]["new_attributes_without_lifetime"] == ["_new"]
    assert result["changes"]["invalid_lifetimes"] == []
    assert result["changes"]["stale_lifetimes"] == []
    assert result["metrics"]["warnings"] == []


@pytest.mark.parametrize("lifetime", ["process", "profile", "conversation", "session", "invocation"])
@pytest.mark.parametrize("with_reason", [False, True])
def test_new_attribute_with_lifetime(tmp_path, lifetime, with_reason):
    entry = {"lifetime": lifetime, "why": "Cleared at the declared boundary."} if with_reason else lifetime
    baseline, _ = lifetime_tree(tmp_path, {"_new": entry})
    result = run(tmp_path, "--baseline", str(baseline))
    assert result["changes"]["new_attributes"] == ["_new"]
    assert result["changes"]["new_attributes_without_lifetime"] == []
    assert result["changes"]["invalid_lifetimes"] == []
    assert result["changes"]["stale_lifetimes"] == []


@pytest.mark.parametrize("entry", [
    "forever", None, [], {}, {"lifetime": []}, {"lifetime": "forever"},
    {"lifetime": "session", "why": 1}, {"lifetime": "session", "why": "two\nlines"},
    {"lifetime": "session", "typo": "ignored?"},
])
def test_invalid_and_stale_lifetimes(tmp_path, entry):
    baseline, _ = lifetime_tree(tmp_path, {"_new": entry, "_removed": "process"})
    result = run(tmp_path, "--baseline", str(baseline))
    assert result["changes"]["invalid_lifetimes"] == ["_new"]
    assert result["changes"]["stale_lifetimes"] == ["_removed"]
    assert result["changes"]["new_attributes_without_lifetime"] == ["_new"]


def test_missing_lifetime_file_warns_once(tmp_path):
    baseline, path = lifetime_tree(tmp_path, {})
    path.unlink()
    result = run(tmp_path, "--baseline", str(baseline))
    assert result["metrics"]["warnings"] == ["Attribute lifetime declarations missing; using no declarations."]
    assert result["changes"]["new_attributes_without_lifetime"] == ["_new"]
    assert result["changes"]["invalid_lifetimes"] == []
    assert result["changes"]["stale_lifetimes"] == []


def test_lifetime_markdown_keeps_all_categories_with_many_changes(tmp_path):
    baseline, _ = lifetime_tree(tmp_path, {"_new": None, "_removed": "process"})
    result = run(tmp_path, "--baseline", str(baseline))
    result["changes"] = {group: [f"entry_{i}" for i in range(20)] for group in result["changes"]}
    result["metrics"]["warnings"] = ["An advisory warning."]
    rendered = report.markdown(result)
    assert len(rendered.splitlines()) <= 60
    for group in result["changes"]:
        assert f"### {group}\n- entry_0\n" in rendered
    assert rendered.count("more; see --format json") == len(result["changes"])
    assert "An advisory warning." in rendered
