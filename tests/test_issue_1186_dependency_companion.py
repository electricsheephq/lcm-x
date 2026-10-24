"""Keep core activation independent of the host-owned dependency consent."""
import ast
import json
import os
from pathlib import Path
import subprocess
import shlex

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


def run_resolving_installer(tmp_path, *, profile=None, sticky=None, custom=True,
                            existing=False, failure=None, legacy=False):
    """Stub the two pinned hosts' -p/root/sticky resolution and native verbs."""
    home = tmp_path / ("custom root" if custom else "user/.hermes")
    target = home / "profiles" / profile if profile and profile != "default" else home
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
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.jsonl"
    hermes = bindir / "hermes"
    hermes.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root = Path(os.environ['HERMES_HOME'])
args = sys.argv[1:]
if args[:1] == ['-p']:
    profile, args = args[1], args[2:]
else:
    sticky = root / 'active_profile'
    profile = sticky.read_text().strip() if sticky.exists() else 'default'
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
    env = {**os.environ, "HOME": str(tmp_path / "user"), "HERMES_HOME": str(home),
           "PATH": f"{bindir}:{os.environ['PATH']}", "CALL_LOG": str(log),
           "FAIL_COMPANION": "1" if failure == "install" else "0"}
    env.pop("HERMES_PROFILE", None)
    if profile:
        env["HERMES_PROFILE"] = profile
    result = subprocess.run(["bash", str(script)], env=env, input="", text=True, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return result, home, target, calls


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
        text = path.read_text()
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
    assert f"hermes plugins enable {NAME}" in probe["detail"]


def test_doctor_action_uses_host_companion():
    from hermes_lcm.diagnostics import doctor_guidance_for_check

    guidance = doctor_guidance_for_check({"check": "embedding_provider_health", "status": "warn"})
    assert f"hermes plugins install {SOURCE}" in guidance["operator_action"]
    assert f"hermes plugins enable {NAME}" in guidance["operator_action"]
