"""#631: the plugin checkout must not be a Hermes package-manager workspace member.

Hermes' package manager (``pm/``, after v2026.9.24) makes any plugin checkout
with a ``pyproject.toml`` a uv workspace member. A member without a
``[build-system]`` gets ``[project].name`` and no version, ``uv lock`` rejects
it, and the update adds ``hermes-lcm-x`` to ``plugins.disabled``. Lint settings
therefore live in ``ruff.toml`` and the repo has no ``pyproject.toml``.
"""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"


def test_lint_settings_live_in_ruff_toml_not_pyproject():
    assert not (REPO_ROOT / "pyproject.toml").exists()

    config = tomllib.loads((REPO_ROOT / "ruff.toml").read_text(encoding="utf-8"))

    assert config["target-version"] == "py311"
    assert config["lint"]["select"] == ["E4", "E7", "E9", "F"]
    assert config["lint"]["ignore"] == ["E402"]


def test_checkout_is_not_a_package_manager_member():
    # Mirrors PythonDeclaration.is_member in Hermes pm/plugin_declarations.py, used by
    # pm/workspace.py::_is_member_candidate: not external and (a pyproject.toml exists
    # or the manifest declares python/pip dependencies). Nothing is imported from Hermes.
    manifest = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    top_level_keys = set(re.findall(r"^([A-Za-z_]+):", manifest, re.MULTILINE))
    declares_dependencies = bool(top_level_keys & {"pip_dependencies", "python_dependencies"})
    has_pyproject = (REPO_ROOT / "pyproject.toml").exists()

    # Without python_runtime: external, membership is exactly "pyproject or dependencies".
    assert "python_runtime" not in top_level_keys
    assert not has_pyproject and not declares_dependencies


_STAGE_PROBE = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
try:
    from pm import workspace
    from pm.plugin_declarations import read_python_declaration
except ImportError as exc:
    print(json.dumps({"skip": f"Hermes package manager not importable: {exc}"}))
    raise SystemExit(0)
tree, checkout, home = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
member = read_python_declaration(checkout).is_member
result = {"is_member": member}
if not member:
    root = home / "generation"
    identity = home / "plugins" / "hermes-lcm-x"
    workspace._generate_pyproject({identity: checkout}, root, source=tree)
    core = (tree / "pyproject.toml").read_text(encoding="utf-8-sig").rstrip("\n") + "\n"
    result["generation_is_core"] = (root / "pyproject.toml").read_text(encoding="utf-8") == core
    result["plugin_sources"] = (root / "plugin-sources").exists()
print(json.dumps(result))
"""


def test_installed_package_manager_stages_no_member(tmp_path):
    if not (HERMES_AGENT / "pm" / "workspace.py").is_file():
        pytest.skip("no Hermes package manager installed")
    candidates = [HERMES_AGENT / "venv" / "bin" / "python", Path(sys.executable)]
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "HERMES_HOME": str(tmp_path / "hermes-home"), "PATH": "/usr/bin:/bin"}
    result = None
    for python in candidates:
        if not python.is_file():
            continue
        # -B -I: import the installed tree read-only; every write goes under tmp_path.
        proc = subprocess.run(
            [str(python), "-B", "-I", "-c", _STAGE_PROBE, str(HERMES_AGENT), str(REPO_ROOT), str(home)],
            capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        if "skip" not in result:
            break
    if result is None or "skip" in result:
        pytest.skip((result or {}).get("skip", "no interpreter for the Hermes package manager"))

    assert result == {"is_member": False, "generation_is_core": True, "plugin_sources": False}
