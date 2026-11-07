"""Source pins; build writes d5-clone-pins.json and returns its contents."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import symtable


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _path(directory, name):
    if not isinstance(name, str) or not name or name == ".." or Path(name).name != name:
        raise ValueError("expected a file basename")
    return Path(directory) / name


def _source_sha256(path, name, assignment=False):
    source = Path(path).read_text(encoding="utf-8")
    nodes = ast.parse(source).body
    for part in name.split("."):
        matches = [n for n in nodes if getattr(n, "name", None) == part]
        if assignment:
            matches = [n for n in nodes if isinstance(n, (ast.Assign, ast.AnnAssign))
                       and any(isinstance(t, ast.Name) and t.id == part
                               for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
        if len(matches) != 1:
            raise ValueError("source name missing or ambiguous")
        node = matches[0]
        nodes = getattr(node, "body", [])
    if not assignment and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("expected a function")
    if getattr(node, "decorator_list", []):
        node.lineno = node.decorator_list[0].lineno
        node.col_offset = node.decorator_list[0].col_offset - 1
    return _sha(ast.get_source_segment(source, node).encode("utf-8"))


def function_source_sha256(path, qualname):
    return _source_sha256(path, qualname)


def assign_source_sha256(path, name):
    return _source_sha256(path, name, assignment=True)


def call_closure(path, roots):
    """Return (function names, assignment names), following module-resolved loads."""
    source = Path(path).read_text(encoding="utf-8")
    body = ast.parse(source).body
    scopes = {s.get_name(): s for s in symtable.symtable(source, "instrument", "exec").get_children()}
    functions = {n.name: n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assignments = {t.id: n for n in body if isinstance(n, (ast.Assign, ast.AnnAssign))
                   for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                   if isinstance(t, ast.Name)}
    pending, seen = list(roots), set()

    def globals_read(scope):
        names = {s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced()}
        return names | {n for child in scope.get_children() for n in globals_read(child)}

    while pending:
        name = pending.pop()
        if name in seen:
            continue
        node = functions.get(name, assignments.get(name))
        if node is None:
            raise ValueError("closure name missing")
        seen.add(name)
        expressions = ([node.args, *node.decorator_list, *([node.returns] if node.returns else [])]
                       if name in functions else [node])
        loads = {n.id for expr in expressions for n in ast.walk(expr)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if name in functions:
            loads |= globals_read(scopes[name])
        pending.extend(loads & (functions.keys() | assignments.keys()))
    return seen & functions.keys(), seen & assignments.keys()


def build(instrument_dir, files, roots_by_file, loads_by_file, interpreter):
    manifest = {"schema": "d5-clone-pins/1", "interpreter": dict(interpreter),
                "files": [], "functions": [], "extra_hashes": []}
    for name in files:
        path = _path(instrument_dir, name)
        manifest["files"].append({"name": name, "sha256": _sha(path.read_bytes()), "loads": [
            {"name": load, "sha256": _sha(_path(instrument_dir, load).read_bytes())}
            for load in loads_by_file.get(name, [])]})
        functions, assignments = call_closure(path, roots_by_file.get(name, []))
        manifest["functions"].extend(
            {"file": name, "qualname": f, "sha256": function_source_sha256(path, f)}
            for f in sorted(functions))
        manifest["extra_hashes"].extend(
            {"file": name, "kind": "assign", "name": a, "sha256": assign_source_sha256(path, a)}
            for a in sorted(assignments))
    (Path(instrument_dir) / "d5-clone-pins.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(manifest_path, instrument_dir, python_version=None):
    mismatches = []
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not manifest["files"]:
            mismatches.append("empty")  # A manifest that pins nothing proves nothing.
        if manifest["schema"] != "d5-clone-pins/1":
            mismatches.append("schema")
        if manifest["interpreter"]["path_basename"] != Path(sys.executable).name:
            mismatches.append("interpreter")
        if not isinstance(manifest["interpreter"]["version"], str):
            mismatches.append("interpreter")
        if python_version is not None and manifest["interpreter"]["version"] != python_version:
            mismatches.append("interpreter")
        entries = [(f["name"], f["sha256"], None) for f in manifest["files"]]
        entries += [(m["name"], m["sha256"], None) for f in manifest["files"] for m in f["loads"]]
        entries += [(f["file"], f["sha256"], f["qualname"]) for f in manifest["functions"]]
        entries += [(a["file"], a["sha256"], (a["name"],)) for a in manifest["extra_hashes"]
                    if a["kind"] == "assign"]
        if any(a["kind"] != "assign" for a in manifest["extra_hashes"]):
            mismatches.append("extra_hashes")
        for name, expected, member in entries:
            try:
                path = _path(instrument_dir, name)
                actual = (_sha(path.read_bytes()) if member is None else
                          assign_source_sha256(path, member[0]) if isinstance(member, tuple) else
                          function_source_sha256(path, member))
                if actual == expected:
                    continue
            except (OSError, ValueError, SyntaxError):
                pass
            mismatches.append(Path(name).name)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        mismatches.append("manifest")
    return {"verdict": "FAIL" if mismatches else "PASS", "mismatches": sorted(set(mismatches))}


def require(*args, **kwargs):
    if verify(*args, **kwargs)["verdict"] == "FAIL":
        raise SystemExit(2)
