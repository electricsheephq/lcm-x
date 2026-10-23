"""Report-only structural metrics; see README.md for scope and counting rules."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

STORAGE = {"db_bootstrap", "store", "vector_store", "lifecycle_state", "dag", "trajectory_store"}
ORCHESTRATION = {"engine", "compaction", "tools", "command", "survival_fit", "escalation"}
PRODUCTION_PACKAGES = {"access_context", "access_policy", "teams"}
SCHEMA_MODULES = {"db_bootstrap.py"}
SKIP = {"venv", "__pycache__", "node_modules"}
# This lexical pattern IS the metric: config/_config, optionally dotted, with a default.
CONFIG_GETATTR = re.compile(
    r'''\bgetattr\(\s*(?:[A-Za-z_]\w*\.)*_?config\s*,\s*(["'])(?P<field>\w+)\1\s*,'''
)
DDL = re.compile(r"\b(?:CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)|ALTER\s+TABLE)\b", re.I)
RULES = {"C901": "cc", "PLR0912": "branches", "PLR0915": "statements"}
SITE_GROUPS = ("imports", "config_getattr", "private_refs", "ddl_outside_schema")
LIFETIMES = {"process", "profile", "conversation", "session", "invocation"}


def python_files(root, warnings=None):
    def unreadable(error):
        # os.walk skips a directory it cannot list; record it so an incomplete scan never replaces the baseline.
        if warnings is not None:
            warnings.append(f"Could not read {Path(error.filename).relative_to(root).as_posix()}")

    for directory, dirs, files in os.walk(root, onerror=unreadable):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP)
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(directory) / name


def is_production(path):
    return len(path.parts) == 1 or path.parts[0] in PRODUCTION_PACKAGES


class Scan(ast.NodeVisitor):
    def __init__(self, path, result, classes):
        self.path, self.result, self.classes = path, result, classes
        self.stack = []
        self.class_key = None
        self.occurrences = Counter()
        self.functions = []

    def site(self, group, line, identity, detail=None):
        prefix = f"{self.path}::{'.'.join(self.stack) or '<module>'}::{identity}"
        self.occurrences[(group, prefix)] += 1
        key = f"{prefix}#{self.occurrences[(group, prefix)]}"
        self.result[group][key] = {"line": line, "detail": detail or identity}

    def visit_ClassDef(self, node):
        previous = self.class_key
        self.stack.append(node.name)
        self.class_key = f"{self.path}::{'.'.join(self.stack)}"
        self.classes[self.class_key] = {
            "bases": [ast.unparse(b) for b in node.bases], "writes": defaultdict(set),
        }
        self.generic_visit(node)
        self.class_key = previous
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        key = f"{self.path}::{'.'.join(self.stack)}"
        self.result["functions"][key] = {
            "line": node.lineno, "loc": node.end_lineno - node.lineno + 1,
            "cc": None, "branches": None, "statements": None,
        }
        self.functions.append((node.lineno, node.end_lineno, tuple(self.stack)))
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):
        if (self.class_key and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name) and node.value.id == "self"):
            self.classes[self.class_key]["writes"][node.attr].add(self.path)
        if self.path.startswith("tests/") and node.attr.startswith("_"):
            receiver = node.value
            name = receiver.id if isinstance(receiver, ast.Name) else (
                receiver.attr if isinstance(receiver, ast.Attribute) else ""
            )
            if re.search(r"(?:^|_)(?:engine|store)(?:_?\d+)?$", name):
                self.site("private_refs", node.lineno, ast.unparse(node))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        if self.path.startswith("tests/") and module.startswith("hermes_lcm."):
            for alias in node.names:
                if alias.name.startswith("_"):
                    self.site("private_refs", node.lineno, f"from {module} import {alias.name}")
        if module in {"", "hermes_lcm"}:
            targets = [a.name.split(".")[0] for a in node.names]
        elif node.level or module.startswith("hermes_lcm."):
            targets = [module.removeprefix("hermes_lcm.").split(".")[0]]
        else:
            targets = [module.split(".")[0]]
        self.imports(node, targets)

    def visit_Import(self, node):
        self.imports(node, [a.name.removeprefix("hermes_lcm.").split(".")[0] for a in node.names])

    def imports(self, node, targets):
        if self.path in {f"{name}.py" for name in STORAGE}:
            for target in targets:
                if target in ORCHESTRATION:
                    self.site("imports", node.lineno, target)

    def visit_Expr(self, node):
        # Standalone string expressions (plain or f-strings) are documentation, not SQL literals in use.
        value = node.value
        if isinstance(value, ast.JoinedStr):
            # Skip the literal fragments, but still visit evaluated expressions (they may execute DDL).
            for part in value.values:
                if isinstance(part, ast.FormattedValue):
                    self.visit(part)
        elif not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            self.generic_visit(node)

    def visit_Constant(self, node):
        if not is_production(Path(self.path)) or not isinstance(node.value, str):
            return
        for match in DDL.finditer(node.value):
            operation = " ".join(match.group().upper().split())
            self.result["ddl_operations"][operation] += 1
            self.result["ddl_modules"].add(self.path)
            if self.path not in SCHEMA_MODULES:
                digest = hashlib.sha256(node.value.encode()).hexdigest()[:16]
                self.site("ddl_outside_schema", node.lineno, f"{operation}:{digest}", operation)


def ruff_metrics(root, files, result):
    command = [
        "ruff", "check", "--select", ",".join(RULES), "--output-format", "json",
        "--no-cache", "--ignore-noqa", "--config", "lint.per-file-ignores={}",
        "--config", "lint.mccabe.max-complexity=0",
        "--config", "lint.pylint.max-branches=0", "--config", "lint.pylint.max-statements=0",
        *files,
    ]
    try:
        proc = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=15,
                              env={**os.environ, "RAYON_NUM_THREADS": "1"})
        if proc.returncode not in (0, 1):
            raise ValueError("ruff did not produce metrics")
        diagnostics = json.loads(proc.stdout)
        lookup = {(k.split("::")[0], v["line"]): v for k, v in result["functions"].items()}
        for metric in lookup.values():
            metric.update(cc=0, branches=0, statements=0)
        for item in diagnostics:
            path = Path(item["filename"]).relative_to(root).as_posix()
            metric = lookup.get((path, item["location"]["row"]))
            value = re.search(r"\((\d+) > \d+\)", item["message"])
            if metric is not None and item["code"] in RULES and value:
                metric[RULES[item["code"]]] = int(value[1])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        result["warnings"].append("Ruff metrics unavailable; install Ruff and rerun.")


def collect(root):
    result = {group: {} for group in SITE_GROUPS}
    result.update(version=1, functions={}, file_loc={}, attributes={}, warnings=[],
                  ddl_operations=Counter(), ddl_modules=set())
    classes, files = {}, []
    for path in python_files(root, result["warnings"]):
        relative = path.relative_to(root)
        name = relative.as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeError, SyntaxError):
            result["warnings"].append(f"Could not parse {name}")
            continue
        files.append(name)
        result["file_loc"][name] = len(source.splitlines())
        visitor = Scan(name, result, classes)
        visitor.visit(tree)
        if is_production(relative):
            for match in CONFIG_GETATTR.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                scopes = [s for start, end, s in visitor.functions if start <= line <= end]
                visitor.stack = list(max(scopes, key=len, default=()))
                visitor.site("config_getattr", line, re.sub(r"\s+", "", match.group()))
    # Follow repository-defined base classes, without importing product code.
    selected, pending = set(), ["engine.py::LCMEngine"]
    by_name = {key.split("::")[1]: key for key in classes if "tests/" not in key}
    while pending:
        key = pending.pop()
        if key in selected or key not in classes:
            continue
        selected.add(key)
        pending.extend(by_name[b] for b in classes[key]["bases"] if b in by_name)
    writers = defaultdict(set)
    for key in sorted(selected):
        for attr, modules in classes[key]["writes"].items():
            writers[attr].update(modules)
    result["attributes"] = {attr: sorted(modules) for attr, modules in sorted(writers.items())}
    result["engine_classes"] = sorted(selected)
    result["ddl_modules"] = sorted(result["ddl_modules"])
    if files:
        ruff_metrics(root, files, result)
    result["headline"] = {
        "engine_loc": result["file_loc"].get("engine.py", 0),
        "c901_over_10": sum((m["cc"] or 0) > 10 for m in result["functions"].values()),
        "engine_attributes": len(writers), "config_getattr": len(result["config_getattr"]),
        "private_refs": len(result["private_refs"]), "import_violations": len(result["imports"]),
        "ddl_modules": len(result["ddl_modules"]),
        "alter_table": result["ddl_operations"]["ALTER TABLE"],
    }
    return result


def load_lifetimes(path, warnings):
    try:
        declarations = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(declarations, dict):
            raise ValueError("expected an object")
        return declarations
    except FileNotFoundError:
        warnings.append("Attribute lifetime declarations missing; using no declarations.")
    except (OSError, UnicodeError, ValueError):
        warnings.append("Attribute lifetime declarations unreadable or not an object; using no declarations.")
    return {}


def valid_lifetime(entry):
    if isinstance(entry, dict):
        if set(entry) - {"lifetime", "why"}:
            return False
        why = entry.get("why", "")
        if not isinstance(why, str) or len(why.splitlines()) > 1:
            return False
        entry = entry.get("lifetime")
    return isinstance(entry, str) and entry in LIFETIMES


def compare(current, baseline, lifetimes=None):
    lifetimes = lifetimes or {}
    invalid = sorted(name for name, entry in lifetimes.items() if not valid_lifetime(entry))
    declared = set(lifetimes) - set(invalid)
    new_attributes = sorted(set(current["attributes"]) - set(baseline.get("attributes", {})))
    growth, new_functions = [], []
    for key, metric in current["functions"].items():
        old = baseline.get("functions", {}).get(key)
        if old is None:
            if metric["loc"] > 80 or (metric["cc"] or 0) > 15:
                new_functions.append({"function": key, **metric})
        else:
            delta = {field: metric[field] - old[field] for field in ("loc", *RULES.values())
                     if metric[field] is not None and old.get(field) is not None
                     and metric[field] > old[field]}
            if delta:
                growth.append({"function": key, "delta": delta})
    return {
        "function_growth": sorted(growth, key=lambda row: (-row["delta"].get("loc", 0),
                                                           -row["delta"].get("cc", 0), row["function"])),
        "new_large_functions": sorted(new_functions, key=lambda row: (-row["loc"], row["function"])),
        "new_attributes": new_attributes,
        "new_attributes_without_lifetime": sorted(set(new_attributes) - declared),
        "invalid_lifetimes": invalid,
        "stale_lifetimes": sorted(set(lifetimes) - set(current["attributes"])),
        "new_writers": {attr: sorted(set(modules) - set(baseline["attributes"][attr]))
                        for attr, modules in current["attributes"].items()
                        if attr in baseline.get("attributes", {})
                        and set(modules) - set(baseline["attributes"][attr])},
        **{f"new_{group}": sorted(set(current[group]) - set(baseline.get(group, {})))
           for group in SITE_GROUPS},
    }


def markdown(report):
    lines = ["# Maintainability report (report-only)",
             "Growth is advisory; it never requires extraction or blocks a small fix."]
    lines.extend(f"- {key}: {value}" for key, value in report["metrics"]["headline"].items())
    changes = report["changes"]
    lines.append("## Changes from baseline")
    lines.extend(f"- {key}: {len(value)}" for key, value in changes.items())
    warnings = report["metrics"]["warnings"]
    # Reserve a heading and bounded details for every category, plus warnings.
    entry_budget = max(1, min(3, (60 - len(lines) - len(changes) - bool(warnings)) // len(changes)))
    for group, rows in changes.items():
        lines.append(f"### {group}")
        entries = list(rows.items()) if isinstance(rows, dict) else rows
        shown = entry_budget - 1 if len(entries) > entry_budget else len(entries)
        lines.extend(f"- {entry}" for entry in entries[:shown])
        if len(entries) > shown:
            lines.append(f"- {len(entries) - shown} more; see --format json")
    if warnings:
        lines.append(f"Warnings ({len(warnings)}): {'; '.join(warnings[:2])}")
    return "\n".join(lines[:60])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("scripts/maintainability/baseline.json"))
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    args = parser.parse_args()
    try:
        metrics = collect(Path.cwd().resolve())
        lifetimes = load_lifetimes(Path("scripts/maintainability/attr_lifetimes.json"), metrics["warnings"])
        # Only an incomplete collection blocks a write; an unreadable or older baseline is what a write repairs.
        collection_warnings = list(metrics["warnings"])
        baseline = {}
        if args.baseline.exists():
            try:
                baseline = json.loads(args.baseline.read_text())
                if baseline.get("version") != metrics["version"]:
                    raise ValueError("incompatible baseline")
            except (OSError, ValueError, AttributeError):
                metrics["warnings"].append("Baseline unreadable or incompatible; comparing to empty baseline.")
                baseline = {}
        elif not args.write_baseline:
            metrics["warnings"].append("Baseline missing; comparing to empty baseline.")
        if args.write_baseline:
            if collection_warnings:
                metrics["warnings"].append("Baseline not written because collection is incomplete.")
            else:
                metrics["warnings"] = []
                args.baseline.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
                baseline = metrics
        report = {"metrics": metrics, "changes": compare(metrics, baseline, lifetimes)}
        print(json.dumps(report, sort_keys=True) if args.format == "json" else markdown(report))
    except Exception as error:
        # Monitoring must not become a build gate, including on an operational error.
        warning = f"Maintainability report incomplete ({type(error).__name__}); rerun after checking inputs."
        print(json.dumps({"warnings": [warning]}) if args.format == "json" else warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
