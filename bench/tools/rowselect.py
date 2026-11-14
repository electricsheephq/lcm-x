"""Select release rows from a committed diff and static product import graph."""
from __future__ import annotations

import argparse
import ast
from collections import deque
import fnmatch
import hashlib
from importlib.util import resolve_name
import json
from pathlib import Path, PurePosixPath
import subprocess

MEMORY = frozenset("ingest storage retry compaction summariser recall delivery".split())
CLASSES = MEMORY | {"host", "cli", "doctor", "install", "instrument"}
ROWS = {"phase-a", "d1", "d4-d8", "d2-d3", "d7", "phase-c-cells"}
PHASE_D = frozenset({"d1", "d2-d3", "d4-d8", "d7"})
NON_PRODUCT = ("bench/", "docs/", "tests/", ".github/release-notes/")
DEFAULT_TABLE = Path(__file__).resolve().parents[1] / "specs/row-classes.json"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def load_table(path):
    raw = Path(path).read_bytes()
    table = json.loads(raw)
    if (table["version"] != 1 or set(table["memory_classes"]) != MEMORY
            or not isinstance(table["exempt"], dict)
            or not all(isinstance(p, str) and isinstance(r, str) and r.strip()
                       for p, r in table["exempt"].items())
            or not isinstance(table["hubs"], list)
            or not all(isinstance(p, str) for p in table["hubs"])
            or not isinstance(table["standalone"], list)
            or not all(isinstance(g, str) for g in table["standalone"])
            or not isinstance(table["rows"], dict)
            or not set(table["rows"]) <= CLASSES
            or not all(isinstance(v, list) and set(v) <= ROWS for v in table["rows"].values())
            or not isinstance(table["entries"], list)
            or not all(isinstance(e["glob"], str) and isinstance(e["classes"], list)
                       and set(e["classes"]) <= CLASSES for e in table["entries"])):
        raise ValueError("invalid class table")
    return table, hashlib.sha256(raw).hexdigest()


def module_name(path):
    if path.startswith(NON_PRODUCT) or not path.endswith(".py"):
        return None
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not all(p.isidentifier() for p in parts):
        return None
    return ".".join(["hermes_lcm", *parts])


def standalone(path, table):
    """Scripts and benchmark entry points: never importers, never unplaceable."""
    return any(fnmatch.fnmatchcase(path, glob) for glob in table["standalone"])


def import_graph(repo, head):
    paths = git(repo, "ls-tree", "-r", "--name-only", "-z", head).decode().split("\0")
    modules = {module_name(p): p for p in paths if module_name(p)}
    graph = {p: set() for p in modules.values()}
    for name, path in sorted(modules.items()):
        tree = ast.parse(git(repo, "show", f"{head}:{path}"), filename=path)
        package = name if path.endswith("__init__.py") else name.rpartition(".")[0]

        def add(target):
            if not target.startswith("hermes_lcm"):
                target = "hermes_lcm." + target
            parts = target.split(".")
            for end in range(2, len(parts) + 1):  # importing a.b runs a/__init__.py too
                name = ".".join(parts[:end])
                if name in modules:
                    graph[path].add(modules[name])

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = resolve_name("." * node.level + target, package)
                add(target)
                for alias in node.names:
                    add(target + "." + alias.name)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("hermes_lcm."):
                    add(node.value)
            elif isinstance(node, ast.Call) and node.args:
                func = node.func
                func_name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if func_name not in {"import_module", "__import__"}:
                    continue
                arg = node.args[0]
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    continue
                target = arg.value
                if target.startswith("."):
                    pkg = node.args[1] if len(node.args) > 1 else next(
                        (k.value for k in node.keywords if k.arg == "package"), None)
                    if not isinstance(pkg, ast.Constant) or not isinstance(pkg.value, str):
                        continue
                    target = resolve_name(target, pkg.value)
                add(target)
    return graph


def path_classes(path, table):
    if path.startswith(NON_PRODUCT):
        return set(), "non-product"
    for entry in table["entries"]:
        if fnmatch.fnmatchcase(path, entry["glob"]):
            return set(entry["classes"]), "matched " + entry["glob"]
    return set(CLASSES), "unmatched"


def placement(table, graph):
    """Return the reverse import graph (standalone importers dropped) and the hub-reachable set.

    A module no hub reaches cannot be placed: nothing shows how the plugin would load it.
    """
    hubs = set(table["hubs"])
    if not hubs <= graph.keys():
        raise ValueError("table hub missing at HEAD")
    reverse = {p: set() for p in graph}
    for importer, targets in graph.items():
        if not standalone(importer, table):
            for target in targets:
                reverse[target].add(importer)
    reachable, pending = set(), list(sorted(hubs))
    while pending:
        path = pending.pop()
        if path not in reachable:
            reachable.add(path)
            pending.extend(sorted(graph[path] - reachable))
    return reverse, reachable


def classify(path, table, graph, reverse, reachable):
    """Return one changed path's classes, its reasons, and whether it failed closed."""
    classes, reason = path_classes(path, table)
    reasons = [{"reason": reason, "classes": sorted(classes)}]

    def include(extra, why):
        classes.update(extra)
        reasons.append({"reason": why, "classes": sorted(extra)})

    closed = reason == "unmatched"
    hubs = set(table["hubs"])
    loose = standalone(path, table)
    if not module_name(path):
        if path.endswith(".py") and not path.startswith(NON_PRODUCT) and not loose:
            include(MEMORY, "not importable")
            closed = True
    elif path in hubs:
        include(MEMORY, "hub change")
    elif path not in graph:
        if not loose:
            include(MEMORY, "absent at head")
            closed = True
    else:
        seen, queue = {path}, deque([(path, [])])
        while queue:
            target, chain = queue.popleft()
            for importer in sorted(reverse[target] - seen - hubs):
                seen.add(importer)
                via = [*chain, importer]
                extra, source_reason = path_classes(importer, table)
                include(extra, "via " + " -> ".join(via))
                closed |= source_reason == "unmatched"
                queue.append((importer, via))
        selected = {r for c in classes for r in table["rows"].get(c, [])}
        if path in reachable and not selected & PHASE_D and path not in table["exempt"]:
            include(MEMORY, "hub closure")
            closed = True
        if path not in reachable and not loose:
            include(MEMORY, "not in import graph")
            closed = True
    return classes, reasons, closed


def select(repo, base, head, table_path=DEFAULT_TABLE):
    table, digest = load_table(table_path)
    base, head = (git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
                  for ref in (base, head))
    fields = iter(git(repo, "diff", "--no-ext-diff", "--name-status", "-z", "-M",
                      base, head, "--").decode().split("\0"))
    changed = set()
    for status in fields:
        if status:
            changed.add(next(fields))
            if status.startswith(("R", "C")):
                changed.add(next(fields))
    graph = import_graph(repo, head)
    reverse, reachable = placement(table, graph)
    result, rows, fail_closed = {}, set(), []
    for path in sorted(changed):
        classes, reasons, closed = classify(path, table, graph, reverse, reachable)
        result[path] = {"classes": sorted(classes),
                        "reasons": sorted(reasons, key=lambda r: r["reason"])}
        rows.update(r for c in classes for r in table["rows"].get(c, []))
        if closed:
            fail_closed.append(path)
    return dict(base=base, head=head, table_sha256=digest, paths=result,
                rows=sorted(rows), always=["L1", "R"], fail_closed=fail_closed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = select(args.repo, args.base, args.head, args.table)
    except (OSError, subprocess.CalledProcessError, ValueError, SyntaxError,
            KeyError, TypeError, StopIteration, ImportError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        detail = " ".join(detail.split())[:300]
        parser.exit(2, f"rowselect: {type(error).__name__}: {detail}\n")
    if args.text:
        for key in ("base", "head", "table_sha256"):
            print(f"{key}: {result[key]}")
        print("PATH\tCLASSES\tREASONS")
        for path, item in result["paths"].items():
            print(f"{path}\t{','.join(item['classes']) or '-'}\t"
                  + "; ".join(f"{r['reason']} ({','.join(r['classes']) or '-'})"
                              for r in item["reasons"]))
        for key in ("rows", "always", "fail_closed"):
            print(f"{key}: {', '.join(result[key]) or '-'}")
    else:
        print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
