"""Keep core activation independent of the host-owned dependency consent."""
import ast
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAME = "hermes-lcm-x-local-embeddings"
SOURCE = f"electricsheephq/lcm-x/companions/{NAME}"


def test_dependency_is_only_declared_by_stable_companion():
    core = {line.split(":", 1)[0] for line in (ROOT / "plugin.yaml").read_text().splitlines()}
    assert not {"python_dependencies", "pip_dependencies"} & core
    assert not (ROOT / "pyproject.toml").exists()
    carrier = dict(line.split(": ", 1) for line in
                   (ROOT / "companions" / NAME / "plugin.yaml").read_text().splitlines())
    requirements = [line for line in (ROOT / "requirements-semantic.txt").read_text().splitlines()
                    if line and not line.startswith("#")]
    assert carrier["name"] == NAME
    assert carrier["version"] == "0.1.0"
    assert ast.literal_eval(carrier["python_dependencies"]) == requirements == ["fastembed>=0.8.0,<1.0"]
    assert not {"provides_tools", "provides_hooks", "config"} & carrier.keys()


@pytest.mark.parametrize("existing", [None, "directory", "symlink"])
def test_installer_hands_consent_to_native_install_without_forcing(tmp_path, existing):
    home = tmp_path / "hermes-home"
    target = home / "plugins" / NAME
    target.parent.mkdir(parents=True)
    if existing == "directory":
        target.mkdir()
        (target / "keep").write_text("preserve me")
    elif existing == "symlink":
        target.symlink_to(tmp_path / "earlier-companion")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.jsonl"
    hermes = bindir / "hermes"
    hermes.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if sys.argv[2] == 'install': print('HOST: installed but not enabled')\n"
    )
    hermes.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path / "user"), "HERMES_HOME": str(home),
           "PATH": f"{bindir}:{os.environ['PATH']}", "CALL_LOG": str(log)}
    env.pop("HERMES_PROFILE", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/install.sh")], env=env,
                            input="", text=True, capture_output=True, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == ["plugins", "enable", "hermes-lcm-x"]
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if existing is None:
        assert calls[1:] == [["plugins", "install", SOURCE, "--ref", sha]]
        assert "HOST: installed but not enabled" in result.stdout
        assert not target.is_symlink()
    else:
        assert len(calls) == 1
        assert f"hermes plugins install {SOURCE} --force --ref {sha}" in result.stdout
        if existing == "directory":
            assert (target / "keep").read_text() == "preserve me"
        else:
            assert target.is_symlink()
    assert f"hermes plugins enable {NAME}" in result.stdout
    assert (home / "plugins/hermes-lcm-x").resolve() == ROOT


def test_missing_dependency_remediation_names_host_companion(monkeypatch):
    from hermes_lcm.config import LCMConfig
    from hermes_lcm import embedding_provider as provider

    monkeypatch.setattr(provider.importlib.util, "find_spec", lambda name: None)
    probe = provider.probe_provider_availability(LCMConfig(
        embeddings_enabled=True, embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5"))
    assert not probe["available"]
    assert f"hermes plugins install {SOURCE}" in probe["detail"]
    assert f"hermes plugins enable {NAME}" in probe["detail"]


def test_doctor_action_uses_host_companion():
    from hermes_lcm.diagnostics import doctor_guidance_for_check

    guidance = doctor_guidance_for_check({"check": "embedding_provider_health", "status": "warn"})
    assert f"hermes plugins install {SOURCE}" in guidance["operator_action"]
    assert f"hermes plugins enable {NAME}" in guidance["operator_action"]
