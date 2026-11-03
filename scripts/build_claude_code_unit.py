#!/usr/bin/env python3
"""Build the LCM-X for Claude Code release asset from one commit.

The asset is the plugin folder with core vendored at ``lcm-x/`` (where the plugin's runtime looks
for it) and a ``bundle.json`` that names the exact core commit it bundles (docs/claude-code-adapter.md,
"Release mechanics"). The build reads only the commit, never the working tree, and the same commit
gives the same archive.

    python scripts/build_claude_code_unit.py claude-code-v0.1.0 --out dist/
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = "adapters/claude-code/"
TAG_PREFIX = "claude-code-v"
# Core is the commit minus repository tooling. It keeps what the worker and the reader read at run
# time besides their modules (plugin.yaml, the recall policy under skills/) and the license.
NOT_CORE_DIRS = (".agents/", ".github/", "adapters/", "bench/", "benchmarking/", "benchmarks/", "companions/",
                 "docs/", "release/", "tests/")
NOT_CORE_FILES = {".gitignore", "AGENTS.md", "CLAUDE.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "pytest.ini",
                  "ruff.toml"}
ENTRY_SCRIPTS = {"scripts/lcm_x_capture.py", "scripts/lcm_x_mcp.py"}  # the worker; the reader (with #1216)


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def is_core(path: str) -> bool:
    if path.startswith("scripts/"):
        return path in ENTRY_SCRIPTS
    return not path.startswith(NOT_CORE_DIRS) and path not in NOT_CORE_FILES


def check_tag(ref: str, version: str) -> None:
    """A unit tag names the plugin version it builds: claude-code-v0.1.0 never ships 0.1.0-dev."""
    if ref.startswith(TAG_PREFIX) and ref != TAG_PREFIX + version:
        raise ValueError(f"{ref} does not match the plugin version {version}")


def _core_version(plugin_yaml: bytes) -> str:
    for line in plugin_yaml.decode("utf-8").splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip()
    raise ValueError("plugin.yaml has no version")


def build(ref: str, out: Path) -> Path:
    """The asset for ``ref`` in ``out``, with its ``.sha256`` beside it."""
    commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    stamp = int(_git("show", "-s", "--format=%ct", commit).decode())
    with tarfile.open(fileobj=io.BytesIO(_git("archive", "--format=tar", commit))) as source:
        files = {member.name: (source.extractfile(member).read(), member.mode)
                 for member in source.getmembers() if member.isfile()}
    version = json.loads(files[PLUGIN_DIR + ".claude-plugin/plugin.json"][0])["version"]
    check_tag(ref, version)
    entries: dict[str, tuple[bytes, int]] = {}
    for path, item in files.items():
        if path.startswith(PLUGIN_DIR):
            entries[path[len(PLUGIN_DIR):]] = item
        elif is_core(path):
            entries["lcm-x/" + path] = item
    bundle = {"unit": version, "ref": ref, "core_commit": commit, "core_version": _core_version(files["plugin.yaml"][0])}
    entries["bundle.json"] = ((json.dumps(bundle, indent=2, sort_keys=True) + "\n").encode(), 0o644)
    name = f"lcm-x-claude-code-{version}"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(entries):
            data, mode = entries[path]
            info = tarfile.TarInfo(f"{name}/{path}")
            info.size, info.mtime, info.mode = len(data), stamp, mode
            tar.addfile(info, io.BytesIO(data))
    out.mkdir(parents=True, exist_ok=True)
    asset = out / f"{name}.tar.gz"
    asset.write_bytes(gzip.compress(archive.getvalue(), mtime=stamp))
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    (out / f"{asset.name}.sha256").write_text(f"{digest}  {asset.name}\n", encoding="utf-8")
    return asset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref", help=f"the tag ({TAG_PREFIX}<version>) or commit to build")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    args = parser.parse_args(argv)
    print(build(args.ref, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
