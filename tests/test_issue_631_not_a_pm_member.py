"""#631 guardrails for lint configuration and package-manager membership.

A lint-only ``pyproject.toml`` is invalid in Hermes' generated uv workspace: PM
adds a project name but no version, then ``uv lock`` rejects the member and the
plugin can be disabled. LCM-X therefore keeps lint configuration in ``ruff.toml``
and never adds ``pyproject.toml`` merely for tooling. Its runtime NumPy declaration
is intentionally different: Hermes stages that manifest-only dependency as a valid
workspace member; ``test_packaging_install.py`` exercises that supported path.
"""

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_lint_settings_live_in_ruff_toml_not_a_lint_only_pyproject():
    """Keep #631's lint-only-pyproject failure mode out of the checkout."""
    assert not (REPO_ROOT / "pyproject.toml").exists()

    config = tomllib.loads((REPO_ROOT / "ruff.toml").read_text(encoding="utf-8"))

    assert config["target-version"] == "py311"
    assert config["lint"]["select"] == ["E4", "E7", "E9", "F"]
    assert config["lint"]["ignore"] == ["E402"]


def test_manifest_keeps_lint_configuration_out_of_runtime_metadata():
    """Keep lint settings in ruff.toml rather than runtime or project metadata."""
    has_pyproject = (REPO_ROOT / "pyproject.toml").exists()
    runtime_dependencies = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")

    assert not has_pyproject
    assert "[tool.ruff]" not in runtime_dependencies
    assert (REPO_ROOT / "ruff.toml").is_file()
