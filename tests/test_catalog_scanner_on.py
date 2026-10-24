"""Real catalog install with the default scanner; fixture transport only (#1946)."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not os.environ.get("LCM_REAL_HERMES_PYTHON"), reason="requires prepared real host")


def run(args, cwd, env=None):
    completed = subprocess.run([str(a) for a in args], cwd=cwd, env=env, capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout.strip()


def test_catalog_scanner_on(tmp_path):
    spec = importlib.util.spec_from_file_location("payload", ROOT / "scripts/build_plugin_payload.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    receipts = ROOT / "catalog-scanner-1946"
    receipts.mkdir(exist_ok=True)
    repo = tmp_path / "repository"
    repo.mkdir()
    run(["git", "init", "-b", "dist"], repo)
    env = dict(os.environ, GIT_AUTHOR_NAME="Fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="Fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid")
    commits = {}
    for label, source in [("baseline", "77b2b4f795fb6a4f84c55abf6d0441da4469971e"), ("candidate", str(ROOT))]:
        out = tmp_path / label / "plugin"
        builder.build(source, out, receipts / f"{label}-manifest.json")
        index_env = dict(env, GIT_INDEX_FILE=str(tmp_path / f"{label}.index"))
        run(["git", "read-tree", "--empty"], repo, index_env)
        for path in sorted(out.rglob("*")):
            if path.is_file():
                blob = run(["git", "hash-object", "-w", str(path)], repo)
                run(["git", "update-index", "--add", "--cacheinfo", "100644", blob,
                     "plugin/" + path.relative_to(out).as_posix()], repo, index_env)
        tree = run(["git", "write-tree"], repo, index_env)
        parent = ["-p", commits["baseline"]] if commits else []
        commits[label] = run(["git", "commit-tree", tree, *parent, "-m", label], repo, env)
    run(["git", "update-ref", "refs/heads/dist", commits["candidate"]], repo)
    bare = tmp_path / "payload.git"
    run(["git", "clone", "--bare", repo, bare], tmp_path)
    transport = tmp_path / "transport"
    transport.mkdir()
    real_git = shutil.which("git")
    wrapper = transport / "git"
    wrapper.write_text(f"#!{sys.executable}\nimport os,sys\n"
                       f"args=[{('file://' + str(bare))!r} if x=='https://payload.invalid/lcm-x.git' else x for x in sys.argv[1:]]\n"
                       f"os.execv({real_git!r},[{real_git!r},*args])\n")
    wrapper.chmod(0o755)
    catalog = {label: {"name": "hermes-lcm-x", "repo": "https://payload.invalid/lcm-x.git", "sha": sha,
                      "subdir": "plugin", "version": "0.28.1" if label == "baseline" else "0.29.0-rc1",
                      "description": "Isolated scanner test", "maintainer": "fixture", "tier": "community",
                      "category": "memory"} for label, sha in commits.items()}
    config = tmp_path / "catalog.json"
    config.write_text(json.dumps(catalog))
    host_env = {"PATH": str(transport) + ":/usr/bin:/bin", "HOME": str(tmp_path / "home"),
                "PYTHONPATH": os.environ["LCM_REAL_HERMES_SRC"],
                "HERMES_HOME": str(tmp_path / "hermes"), "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1",
                "HERMES_DISABLE_LAZY_INSTALLS": "1", "PROBE_ENGINE": "lcm-x"}
    for phase in ("candidate", "consent"):
        host_env["HERMES_HOME"] = str(tmp_path / ("hermes-" + phase))
        command = [os.environ["LCM_REAL_HERMES_PYTHON"], ROOT / "tests/fixtures/catalog_scanner_probe.py",
                   phase, config, receipts / f"{phase}.json", ROOT / "tests/fixtures/catalog-scanner-findings.json",
                   ROOT / "tests/test_real_plugin_manager_identity.py"]
        completed = subprocess.run([str(a) for a in command], cwd=os.environ["LCM_REAL_HERMES_SRC"],
                                   env=host_env, capture_output=True, text=True, timeout=240)
        (receipts / f"{phase}.stdout").write_text(completed.stdout)
        (receipts / f"{phase}.stderr").write_text(completed.stderr)
        assert completed.returncode == 0, completed.stdout + completed.stderr
