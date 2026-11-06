#!/usr/bin/env python3
"""Which files the LCM-X for Claude Code unit depends on changed since its last release.

The unit bundles the whole core, so its gate re-runs whenever a core file the worker or the
reader imports changes (docs/claude-code-adapter.md, "Release mechanics"). The set is computed,
not kept by hand: the import closure of the unit's entry points, plus the plugin folder.

    python scripts/claude_code_unit_delta.py            # since the last claude-code-v* tag
    python scripts/claude_code_unit_delta.py --since REF --check   # exit 1 when anything changed
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hermes_lcm"  # the name the entry points import the package under
ENTRY_POINTS = ("scripts/lcm_x_capture.py", "scripts/lcm_x_mcp.py")  # the worker; the reader (with #1216)
PLUGIN_DIR = "adapters/claude-code/"
TAG_PATTERN = "claude-code-v*"


def _module_file(root: Path, parts: list[str]) -> Optional[Path]:
    """The file of a module inside the package, given its dotted parts below the package."""
    if not parts:
        return None  # the entry points install the package as a bare module: its __init__.py never runs
    for candidate in (root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts, "__init__.py")):
        if candidate.is_file():
            return candidate
    return None


def _imported(path: Path, root: Path) -> set[Path]:
    """Package modules ``path`` imports anywhere in its body, a lazy import included."""
    package = [] if path.parent == root or root not in path.parents else list(path.parent.relative_to(root).parts)
    if path.parent.name == "scripts" and path.parent.parent == root:
        package = []  # entry scripts import the package by name, never relatively
    found: set[Path] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        targets: list[list[str]] = []
        if isinstance(node, ast.Import):
            targets = [alias.name.split(".")[1:] for alias in node.names if alias.name.split(".")[0] == PACKAGE]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)] if node.level > 1 else package
                module = base + (node.module.split(".") if node.module else [])
            elif node.module and node.module.split(".")[0] == PACKAGE:
                module = node.module.split(".")[1:]
            else:
                continue
            targets = [module] + [module + [alias.name] for alias in node.names]
        for parts in targets:
            target = _module_file(root, parts)
            if target is not None:
                found.add(target)
    return found


def closure(root: Path = ROOT, entry_points: Iterable[str] = ENTRY_POINTS) -> set[str]:
    """Repository paths of every package module the entry points import, directly or not."""
    pending = [root / entry for entry in entry_points if (root / entry).is_file()]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        pending.extend(_imported(path, root) - seen)
    return {path.relative_to(root).as_posix() for path in seen}


def relevant(changed: Iterable[str], dependencies: set[str]) -> list[str]:
    """The changed paths the unit depends on: its import closure and its plugin folder."""
    return sorted({path for path in changed if path in dependencies or path.startswith(PLUGIN_DIR)})


def last_release(root: Path = ROOT) -> Optional[str]:
    tags = subprocess.run(["git", "-C", str(root), "tag", "--list", TAG_PATTERN, "--sort=-v:refname"],
                          check=True, capture_output=True, text=True).stdout.split()
    return tags[0] if tags else None


def changed_since(ref: str, root: Path = ROOT) -> list[str]:
    return subprocess.run(["git", "-C", str(root), "diff", "--name-only", f"{ref}..HEAD"],
                          check=True, capture_output=True, text=True).stdout.split()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", help="compare against this ref instead of the last unit tag")
    parser.add_argument("--check", action="store_true", help="exit 1 when a dependency changed")
    args = parser.parse_args(argv)
    since = args.since or last_release()
    dependencies = closure()
    # Without an earlier release every dependency counts as changed: the first release runs the gate.
    changed = relevant(changed_since(since), dependencies) if since else sorted(dependencies)
    print(json.dumps({"since": since, "dependencies": len(dependencies), "changed": changed}, indent=2))
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
