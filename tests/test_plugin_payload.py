"""Payload contract, including mutations of imports, assets and shipped paths (#1946)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder():
    path = ROOT / "scripts/build_plugin_payload.py"
    assert path.is_file(), "payload builder is required"
    spec = importlib.util.spec_from_file_location("payload_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    files = {
        "__init__.py": "from . import core\n",
        "core.py": "from .package.child import VALUE\n",
        "package/__init__.py": "",
        "package/child.py": "VALUE = 1\n",
        "plugin.yaml": "name: fixture\n",
        "LICENSE": "fixture license\n",
        "skills/hermes-lcm/SKILL.md": "fixture skill\n",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    git(root, "init", "-b", "main")
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-m", "fixture")
    return root


def build(builder, source, tmp_path, assets=None):
    allowlist = tmp_path / "assets.txt"
    allowlist.write_text(assets or "plugin.yaml\nLICENSE\nskills/hermes-lcm/**\n")
    return builder.build(str(source), tmp_path / "payload", tmp_path / "manifest.json", allowlist)


def test_import_closure_and_manifest(builder, source, tmp_path):
    result = build(builder, source, tmp_path)
    paths = [row["path"] for row in result["files"]]
    assert paths == sorted(paths)
    assert {"__init__.py", "core.py", "package/__init__.py", "package/child.py"} <= set(paths)
    assert result["source_sha"] == git(source, "rev-parse", "HEAD")
    assert result["builder_version"]
    assert result["file_count"] == len(paths)
    assert result == json.loads((tmp_path / "manifest.json").read_text())
    for row in result["files"]:
        data = (tmp_path / "payload" / row["path"]).read_bytes()
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert row["bytes"] == len(data)
    assert result["total_bytes"] == sum(row["bytes"] for row in result["files"])


@pytest.mark.parametrize("text", ["from .missing import value", "from . import missing",
                                        "import hermes_lcm.missing"])
def test_unresolved_local_import_is_refused(builder, source, tmp_path, text):
    (source / "core.py").write_text(text)
    with pytest.raises(ValueError, match="unresolved local import"):
        build(builder, source, tmp_path)
    assert not (tmp_path / "payload").exists()


@pytest.mark.parametrize("assets", ["plugin.yaml\nLICENSE\n", "plugin.yaml\nLICENSE\nmissing.md\n"])
def test_missing_or_unlisted_assets_refused(builder, source, tmp_path, assets):
    with pytest.raises(ValueError, match="asset"):
        build(builder, source, tmp_path, assets)


@pytest.mark.parametrize("prefix", ["tests", "bench", "scripts", ".github"])
def test_development_import_mutation_refused(builder, source, tmp_path, prefix):
    (source / prefix).mkdir()
    (source / prefix / "__init__.py").write_text("")
    (source / "core.py").write_text(f"from . import {prefix}" if prefix != ".github" else "")
    git(source, "add", ".")
    assets = "plugin.yaml\nLICENSE\nskills/hermes-lcm/**\n" + prefix + "/**\n"
    with pytest.raises(ValueError, match="forbidden payload path"):
        build(builder, source, tmp_path, assets)


def test_external_symlink_refused(builder, source, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    (source / "skills/hermes-lcm/escape.md").symlink_to(outside)
    git(source, "add", ".")
    with pytest.raises(ValueError, match="symlink"):
        build(builder, source, tmp_path)


@pytest.mark.parametrize("limit,value", [("MAX_FILE_BYTES", 8), ("MAX_FILES", 2), ("MAX_TOTAL_BYTES", 20)])
def test_limits_refused_before_writing(builder, source, tmp_path, monkeypatch, limit, value):
    monkeypatch.setattr(builder, limit, value)
    with pytest.raises(ValueError, match="limit"):
        build(builder, source, tmp_path)
    assert not (tmp_path / "payload").exists()


def test_git_sha_source_matches_checkout(builder, source, tmp_path, monkeypatch):
    monkeypatch.chdir(source)
    result = build(builder, git(source, "rev-parse", "HEAD"), tmp_path)
    assert result["source_sha"] == git(source, "rev-parse", "HEAD")
    assert "package/child.py" in {row["path"] for row in result["files"]}


def test_checkout_drift_gate(builder, tmp_path):
    result = builder.build(str(ROOT), tmp_path / "payload", tmp_path / "manifest.json")
    paths = {row["path"] for row in result["files"]}
    # Re-derive using only the staged tree: a missing import cannot use the source checkout.
    builder.closure(tmp_path / "payload", paths, lambda p: (tmp_path / "payload" / p).read_bytes())
    assert result["file_count"] <= builder.MAX_FILES
    assert result["total_bytes"] <= builder.MAX_TOTAL_BYTES
    assert all(row["bytes"] <= builder.MAX_FILE_BYTES for row in result["files"])
    assert not any(p.split("/")[0] in {"tests", "bench", "scripts", ".github"} for p in paths)
    # Reviewed file-read sites from the #1946 measure lane: standalone config, skill
    # registration and guidance recall policy. engine/runtime_identity __file__ sites
    # only report provenance; bypass/host_uid_emit import host-owned modules.
    assert "config.py" in paths
    assert (tmp_path / "payload/skills/hermes-lcm/references/recall-policy.md").read_text().strip()
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT / "skills/hermes-lcm").rglob("*") if p.is_file()} <= paths
