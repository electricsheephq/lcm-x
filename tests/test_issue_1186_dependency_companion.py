"""Keep core activation independent of the host-owned dependency consent."""
import ast
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys

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
        "if 'install' in sys.argv: print('HOST: installed but not enabled')\n"
    )
    hermes.chmod(0o755)
    env = {**os.environ, "HOME": str(tmp_path / "user"), "HERMES_HOME": str(home),
           "PATH": f"{bindir}:{os.environ['PATH']}", "CALL_LOG": str(log)}
    env.pop("HERMES_PROFILE", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/install.sh")], env=env,
                            input="", text=True, capture_output=True, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == ["-p", "default", "plugins", "enable", "hermes-lcm-x"]
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if existing is None:
        assert calls[1:] == [["-p", "default", "plugins", "install", SOURCE, "--ref", sha]]
        assert "HOST: installed but not enabled" in result.stdout
        assert not target.is_symlink()
    else:
        assert len(calls) == 1
        if existing == "directory":
            assert f"hermes -p default plugins install {SOURCE} --force --ref {sha}" in result.stdout
            assert (target / "keep").read_text() == "preserve me"
        else:
            assert "Move only this link aside by hand" in result.stdout
            assert f"hermes -p default plugins install {SOURCE} --ref {sha}" in result.stdout
            assert "--force" not in result.stdout
            assert target.is_symlink()
    assert f"hermes -p default plugins enable {NAME}" in result.stdout
    assert (home / "plugins/hermes-lcm-x").resolve() == ROOT


@pytest.mark.parametrize("profile,sticky,custom", [
    ("p1", "p2", False), (None, "p2", False), (None, None, True), ("default", "p2", False),
])
@pytest.mark.parametrize("existing", [False, True])
def test_installer_targets_explicit_profile_and_root(tmp_path, profile, sticky, custom, existing):
    result, home, target, calls = run_resolving_installer(
        tmp_path, profile=profile, sticky=sticky, custom=custom, existing=existing)
    assert result.returncode == 0, result.stderr
    assert (target / "plugins/hermes-lcm-x").resolve() == ROOT
    assert all(call["home"] == str(target) for call in calls)
    assert calls[0]["args"] == ["plugins", "enable", "hermes-lcm-x"]
    assert len(calls) == (1 if existing else 2)
    if existing:
        assert (target / "plugins" / NAME / "keep").read_text() == "preserve"
    # Every printed host command is executable against the same selected home.
    for line in result.stdout.splitlines():
        if "hermes " in line and "plugins " in line and not line.startswith("Plugin installed but not enabled."):
            assert "hermes -p " in line, line
            command = line[line.index("HERMES_HOME="):]
            assert shlex.split(command)[0] == f"HERMES_HOME={home}", line


def test_fake_hermes_runs_with_interpreter_path_containing_spaces(tmp_path, monkeypatch):
    interpreter = tmp_path / "python with spaces" / "python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    monkeypatch.setenv("LCM_REAL_HERMES_PYTHON", str(interpreter))
    result, _, target, calls = run_resolving_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (target / "plugins/hermes-lcm-x").resolve() == ROOT
    assert [call["args"][:3] for call in calls] == [
        ["plugins", "enable", "hermes-lcm-x"], ["plugins", "install", SOURCE]]


def run_resolving_installer(tmp_path, *, profile=None, sticky=None, custom=True,
                            existing=False, failure=None, legacy=False, profile_home=False, resolution_source=None):
    """Use real host profile resolution when requested; stub native verbs."""
    home = tmp_path / ("custom root" if custom else "user/.hermes")
    canonical = profile.strip().casefold() if profile else "default"
    target = home / "profiles/work" if profile_home else (
        home / "profiles" / canonical if canonical != "default" else home)
    target.mkdir(parents=True)
    if sticky:
        (home / "active_profile").write_text(sticky)
    if failure:
        (target / "config.yaml").write_text("context:\n  engine: lcm\n")
    if existing:
        carrier = target / "plugins" / NAME
        carrier.mkdir(parents=True)
        (carrier / "keep").write_text("preserve")
    if legacy:
        old = target / "plugins/hermes-lcm"
        old.mkdir(parents=True)
        (old / "keep").write_text("legacy")
        (target / "config.yaml").write_text("plugins:\n  enabled: [hermes-lcm]\n")
        (target / "plugins/hermes-lcm-x").symlink_to(ROOT)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.jsonl"
    hermes = bindir / "hermes"
    python = os.environ.get("LCM_REAL_HERMES_PYTHON", sys.executable)
    launcher = f'#!/bin/sh\n""":"\nexec {shlex.quote(python)} -B "$0" "$@"\n":"""\n'
    hermes.write_text(launcher + """import json, os, sys
from pathlib import Path
root = Path(os.environ['HERMES_HOME'])
args = sys.argv[1:]
if args[:1] == ['-p']:
    profile, args = args[1].strip().casefold(), args[2:]
else:
    sticky = root / 'active_profile'
    profile = sticky.read_text().strip() if sticky.exists() else 'default'
source = os.environ.get('LCM_1186_PROFILE_SOURCE')
if source:
    # Execute the real pin's resolution; only validation/liveness and verbs are stubbed.
    # Import the whole module so each pin retains its own helper dependencies.
    sys.path.insert(0, str(Path(source).resolve().parents[1]))
    from hermes_cli import profiles
    profiles._canon_valid = lambda name: name.strip().casefold()
    profiles._get_default_hermes_home = lambda: Path.home() / '.hermes'
    profiles.named_profile_is_live = lambda path: path.is_dir()
    home = Path(profiles.resolve_profile_env(profile))
else:
    root = root.parent.parent if root.parent.name == 'profiles' else root
    home = root if profile == 'default' else root / 'profiles' / profile
with open(os.environ['CALL_LOG'], 'a') as f:
    f.write(json.dumps({'home': str(home), 'args': args}) + '\\n')
if args[1] == 'enable' and not (home / 'plugins' / args[2]).exists():
    sys.exit(1)
if args[1] == 'install':
    if os.environ.get('FAIL_COMPANION') == '1':
        sys.exit(1)
    print('Plugin installed but not enabled. Run `hermes plugins enable ' + args[2].rsplit('/', 1)[-1] + '` to activate.')
""")
    hermes.chmod(0o755)
    script = ROOT / "scripts/install.sh"
    if failure == "non-git":
        checkout = tmp_path / "archive"
        (checkout / "scripts").mkdir(parents=True)
        (checkout / "skills/hermes-lcm").mkdir(parents=True)
        script = checkout / "scripts/install.sh"
        script.write_text((ROOT / "scripts/install.sh").read_text())
    env = {**os.environ, "HOME": str(tmp_path / "user"), "HERMES_HOME": str(target if profile_home else home),
           "PATH": f"{bindir}:{os.environ['PATH']}", "CALL_LOG": str(log),
           "FAIL_COMPANION": "1" if failure == "install" else "0"}
    env.pop("HERMES_PROFILE", None)
    if resolution_source is not None:
        env["LCM_1186_PROFILE_SOURCE"] = str(resolution_source)
    if profile:
        env["HERMES_PROFILE"] = profile
    result = subprocess.run(["bash", str(script)], env=env, input="", text=True, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, home, target, calls


@pytest.mark.parametrize("profile", [None, "work", "p1"])
@pytest.mark.parametrize("existing", [False, True])
def test_profile_shaped_home_targets_one_profile_or_refuses_conflict(tmp_path, profile, existing):
    result, home, target, calls = run_resolving_installer(
        tmp_path, profile=profile, profile_home=True, existing=existing)
    if profile == "p1":
        assert result.returncode != 0
        assert "work" in result.stderr and "p1" in result.stderr
        assert f"HERMES_HOME={home}" in shlex.split(result.stderr)
        assert "HERMES_PROFILE=p1" in result.stderr
        assert f"HERMES_HOME={target}" in shlex.split(result.stderr)
        assert not calls
        assert not list(home.rglob("hermes-lcm-x"))
        assert not list(home.rglob("skills"))
        return
    assert result.returncode == 0, result.stderr
    assert (target / "plugins/hermes-lcm-x").resolve() == ROOT
    assert (target / "skills/hermes-lcm-x").resolve() == ROOT / "skills/hermes-lcm"
    assert all(call["home"] == str(target) for call in calls)
    assert calls[0]["args"] == ["plugins", "enable", "hermes-lcm-x"]
    assert len(calls) == (1 if existing else 2)
    if existing:
        assert (target / "plugins" / NAME / "keep").read_text() == "preserve"
        assert "--force --ref" in result.stdout
    else:
        assert calls[1]["args"][:3] == ["plugins", "install", SOURCE]
    for line in result.stdout.splitlines():
        if "HERMES_HOME=" in line:
            command = shlex.split(line[line.index("HERMES_HOME="):])
            assert command[:4] == [f"HERMES_HOME={home}", "hermes", "-p", "work"]
    assert not (home / "plugins").exists()


@pytest.mark.parametrize("profile", ["Default", " default ", "P1", " p1 ", "\u2003P1\u2003", "p/1", " ", "p" * 65])
def test_noncanonical_profile_is_refused_before_links_or_host_calls(tmp_path, profile):
    result, home, target, calls = run_resolving_installer(tmp_path, profile=profile)
    assert result.returncode == 2, result.stderr
    assert "canonical" in result.stderr
    assert "HERMES_PROFILE=default ./scripts/install.sh" in result.stderr
    assert not calls
    assert not (target / "plugins/hermes-lcm-x").exists()
    assert not (target / "skills/hermes-lcm-x").exists()
    assert not (home / "profiles" / profile / "plugins/hermes-lcm-x").exists()


@pytest.mark.parametrize("profile", ["Default", " default ", "default", " P1 ", "p1"])
def test_canonical_legacy_target_cannot_be_bypassed_by_profile_spelling(tmp_path, profile):
    result, _, target, calls = run_resolving_installer(tmp_path, profile=profile, legacy=True)
    assert result.returncode == (0 if profile in {"default", "p1"} else 2), result.stderr
    assert not any(call["args"][:3] == ["plugins", "enable", "hermes-lcm-x"] for call in calls)
    assert (target / "plugins/hermes-lcm-x").resolve() == ROOT
    assert (target / "plugins/hermes-lcm/keep").read_text() == "legacy"
    assert (target / "config.yaml").read_text() == "plugins:\n  enabled: [hermes-lcm]\n"
    if result.returncode == 0:
        assert "BREAKING" in result.stdout
    else:
        assert not calls


@pytest.mark.parametrize("failure", ["install", "non-git"])
def test_companion_failure_preserves_core_success_and_migration_text(tmp_path, failure):
    result, home, target, calls = run_resolving_installer(tmp_path, failure=failure)
    assert result.returncode == 0, result.stderr
    assert calls[0]["args"] == ["plugins", "enable", "hermes-lcm-x"]
    assert (target / "plugins/hermes-lcm-x").exists()
    assert "Activation requires both:" in result.stdout
    assert "BREAKING" in result.stdout
    assert "Retry" in result.stdout
    retry = next(line.split("Retry: ", 1)[1] for line in result.stdout.splitlines() if "Retry: " in line)
    assert shlex.split(retry)[:7] == [f"HERMES_HOME={home}", "hermes", "-p", "default", "plugins", "install", SOURCE]


def test_separate_legacy_copy_is_preserved_without_auto_enable(tmp_path):
    result, _, target, calls = run_resolving_installer(tmp_path, legacy=True)
    assert result.returncode == 0, result.stderr
    assert not any(call["args"][:3] == ["plugins", "enable", "hermes-lcm-x"] for call in calls)
    assert (target / "plugins/hermes-lcm/keep").read_text() == "legacy"
    assert (target / "config.yaml").read_text() == "plugins:\n  enabled: [hermes-lcm]\n"
    assert "BREAKING" in result.stdout and "Do NOT keep both names enabled" in result.stdout


def test_install_documentation_matches_host_owned_activation_and_live_anchor():
    for path in [ROOT / "README.md", ROOT / "docs/operator-guide.md",
                 ROOT / "skills/hermes-lcm/references/configuration.md"]:
        text = " ".join(path.read_text().split())
        assert "never edits `config.yaml`" not in text
        assert "without editing config" not in text
        assert "a-host-update-can-remove-fastembed" not in text
        assert "companion" in text
        assert "separate older copy" in text
    assert "#install-consent-updates-and-recovery" in (ROOT / "README.md").read_text()
    assert "#install-consent-updates-and-recovery" in (ROOT / "docs/operator-guide.md").read_text()


def test_missing_dependency_remediation_names_host_companion(monkeypatch):
    from hermes_lcm.config import LCMConfig
    from hermes_lcm import embedding_provider as provider

    monkeypatch.setattr(provider.importlib.util, "find_spec", lambda name: None)
    probe = provider.probe_provider_availability(LCMConfig(
        embeddings_enabled=True, embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5"))
    assert not probe["available"]
    assert f"hermes plugins install {SOURCE}" in probe["detail"]
    assert f"hermes -p default plugins enable {NAME}" in probe["detail"]


def test_doctor_action_uses_host_companion():
    from hermes_lcm.diagnostics import doctor_guidance_for_check

    guidance = doctor_guidance_for_check({"check": "embedding_provider_health", "status": "warn"})
    assert f"hermes plugins install {SOURCE}" in guidance["operator_action"]
    assert f"hermes -p default plugins enable {NAME}" in guidance["operator_action"]
