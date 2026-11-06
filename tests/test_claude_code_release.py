"""LCM-X for Claude Code release mechanics: the unit's dependency set and its bundle.

Design: docs/claude-code-adapter.md ("Release mechanics").
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def test_the_unit_depends_on_the_import_closure_of_its_entry_points():
    delta = _script("claude_code_unit_delta")
    dependencies = delta.closure()
    assert {"claude_code_store.py", "claude_code_capture.py", "store.py", "dag.py", "db_bootstrap.py",
            "scripts/lcm_x_capture.py"} <= dependencies
    # The entry points install the package as a bare module, so its __init__.py never runs.
    assert "__init__.py" not in dependencies
    assert not any(path.startswith(("tests/", "bench/", "docs/")) for path in dependencies)
    changed = ["store.py", "docs/claude-code-adapter.md", "tests/test_claude_code_store.py",
               "adapters/claude-code/bin/lcmx_hook.py", "bench/aml/aml_server.py"]
    assert delta.relevant(changed, dependencies) == ["adapters/claude-code/bin/lcmx_hook.py", "store.py"]


def test_the_unit_asset_bundles_the_exact_core_commit(tmp_path):
    build = _script("build_claude_code_unit")
    asset = build.build("HEAD", tmp_path / "one")
    again = build.build("HEAD", tmp_path / "two")
    assert asset.read_bytes() == again.read_bytes()  # the same commit gives the same archive
    digest, name = (tmp_path / "one" / f"{asset.name}.sha256").read_text(encoding="utf-8").split()
    assert name == asset.name and digest == hashlib.sha256(asset.read_bytes()).hexdigest()
    version = json.loads(_git("show", "HEAD:adapters/claude-code/.claude-plugin/plugin.json"))["version"]
    prefix = f"lcm-x-claude-code-{version}/"
    with tarfile.open(asset) as tar:
        names = {member.name[len(prefix):] for member in tar.getmembers()}
        bundle = json.loads(tar.extractfile(prefix + "bundle.json").read())
    assert bundle == {"unit": version, "ref": "HEAD", "core_commit": _git("rev-parse", "HEAD").strip(),
                      "core_version": bundle["core_version"]}
    # The plugin at the root, and core where the plugin's runtime looks for it (lcmx_runtime.package_root),
    # with every module the entry points import and the files core reads at run time.
    committed = set(_git("ls-tree", "-r", "--name-only", "HEAD").split())
    imported = _script("claude_code_unit_delta").closure() & committed
    assert {"lcm-x/" + path for path in imported} <= names
    assert {".claude-plugin/plugin.json", "hooks/hooks.json", ".grok-plugin/plugin.json", ".grok-plugin/hooks.json",
            "bin/lcmx_hook.py", "lcm-x/plugin.yaml", "lcm-x/LICENSE",
            "lcm-x/skills/hermes-lcm/references/recall-policy.md", "lcm-x/access_context/inventory.json"} <= names
    assert not any(name.startswith(("lcm-x/tests/", "lcm-x/bench/", "lcm-x/adapters/", "lcm-x/.github/"))
                   for name in names)
    assert {name for name in names if name.startswith("lcm-x/scripts/")} <= {"lcm-x/scripts/lcm_x_capture.py",
                                                                            "lcm-x/scripts/lcm_x_mcp.py"}
    assert "lcm-x/CLAUDE.md" not in names and "lcm-x/AGENTS.md" not in names


def test_a_unit_tag_names_the_version_it_builds():
    build = _script("build_claude_code_unit")
    build.check_tag("claude-code-v0.1.0", "0.1.0")
    build.check_tag("HEAD", "0.1.0-dev")  # a commit, not a release
    with pytest.raises(ValueError):
        build.check_tag("claude-code-v0.1.0", "0.1.0-dev")
