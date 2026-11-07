"""Source pins; build writes d5-clone-pins.json and returns its contents."""

import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import symtable


def _sha(data):
    return hashlib.sha256(data).hexdigest()


MANIFEST = "d5-clone-pins.json"


def _load_path(directory, name):
    """A pinned file by its relative POSIX name under the instrument directory (no '..', no absolute names)."""
    parts = Path(name).parts if isinstance(name, str) else ()
    if not parts or Path(name).is_absolute() or any(part in ("..", ".") for part in parts):
        raise ValueError("expected a relative file name")
    return Path(directory).joinpath(*parts)


def directory_files(instrument_dir):
    """Every file under the instrument directory except the manifest and bytecode caches. The manifest must pin
    all of them, so a sibling file cannot change unnoticed however it is loaded (import, importlib, runpy)."""
    root = Path(instrument_dir)
    found = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or relative.as_posix() == MANIFEST:
            continue
        if path.is_symlink():
            found.add("symlink:" + relative.as_posix())  # Never pinnable: a link can point anywhere.
        elif path.is_file():
            found.add(relative.as_posix())
    return found


def _path(directory, name):
    if not isinstance(name, str) or not name or name == ".." or Path(name).name != name:
        raise ValueError("expected a file basename")
    return Path(directory) / name


_PINNABLE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Import,
             ast.ImportFrom)
_DYNAMIC = {"globals", "vars", "locals", "eval", "exec", "__import__", "importlib", "import_module"}


def _bindings(body):
    """Each module-level name -> the top-level statements that bind or touch it.

    A name with exactly one pinnable binding statement can be pinned by that statement's source. A name also
    touched by any other top-level statement (a second def, an augmented assignment, a subscript store, or a
    binding inside if/try/for/with) has more than one entry, so the closure refuses it instead of dropping it."""
    out = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [(alias.asname or alias.name).split(".")[0] for alias in node.names]
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
        else:  # Any other top-level statement (if/try/for/with/...): every name it binds anywhere inside.
            names = _stored_names(node)
        for name in dict.fromkeys(names):
            out.setdefault(name, []).append(node)
    return out


def _stored_names(node):
    names = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.append(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(n.name)
        elif isinstance(n, ast.alias):
            names.append((n.asname or n.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.append(n.name)
    return names


def _segment(source, node):
    if getattr(node, "decorator_list", []):
        first = node.decorator_list[0]
        line = source.splitlines(keepends=True)[first.lineno - 1]
        node.lineno, node.col_offset = first.lineno, line.rindex("@", 0, first.col_offset + 1)
    return ast.get_source_segment(source, node)


def _source_sha256(path, name, assignment=False):
    source = Path(path).read_text(encoding="utf-8")
    body = ast.parse(source).body
    if assignment:
        nodes = _bindings(body).get(name, [])
        if len(nodes) != 1 or not isinstance(nodes[0], _PINNABLE) or isinstance(
                nodes[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError("source name missing or ambiguous")
        return _sha(_segment(source, nodes[0]).encode("utf-8"))
    nodes = body
    for part in name.split("."):
        matches = [n for n in nodes if getattr(n, "name", None) == part]
        if len(matches) != 1:
            raise ValueError("source name missing or ambiguous")
        node = matches[0]
        nodes = getattr(node, "body", [])
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("expected a function")
    return _sha(_segment(source, node).encode("utf-8"))


def function_source_sha256(path, qualname):
    return _source_sha256(path, qualname)


def assign_source_sha256(path, name):
    """The source of the one top-level statement binding *name*: an assignment, class or import."""
    return _source_sha256(path, name, assignment=True)


def call_closure(path, roots, data=()):
    """Return (function names, other pinned names) reachable from *roots*.

    Names in *data* are run inputs (for example, constants loaded from a recorded constants file): they are
    neither pinned nor followed. Raises ValueError when a reachable module-level name cannot be pinned by exactly one top-level statement,
    or when a reachable function reads globals dynamically (globals(), vars(), eval, exec, __import__)."""
    source = Path(path).read_text(encoding="utf-8")
    body = ast.parse(source).body
    top = symtable.symtable(source, "instrument", "exec")
    tables = {s.get_name(): s for s in top.get_children()}
    bindings = _bindings(body)
    # Every name the module binds anywhere (a def under if, an except target, ...); a reached one must be pinnable.
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    pending, seen = list(roots), set()

    def globals_read(scope):
        names = {s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced()}
        return names | {n for child in scope.get_children() for n in globals_read(child)}

    while pending:
        name = pending.pop()
        if name in data:
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
                   for n in bindings.get(name, [])):
                raise ValueError(f"data names must be plain values, not code: {name}")
            continue
        if name in seen:
            continue
        nodes = bindings.get(name, [])
        if len(nodes) != 1 or not isinstance(nodes[0], _PINNABLE):
            raise ValueError(f"closure name missing or not pinnable: {name}")
        node = nodes[0]
        seen.add(name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if any((isinstance(n, ast.Name) and n.id in _DYNAMIC) or (
                    isinstance(n, ast.Attribute) and n.attr in ("modules", "import_module")) for n in ast.walk(node)):
                raise ValueError("closure reads globals or modules dynamically")
            # Outside the body (evaluated in module scope): decorators, defaults, annotations, bases.
            outer = [*node.decorator_list, *((node.bases + [k.value for k in node.keywords])
                                             if isinstance(node, ast.ClassDef) else
                                             [node.args, *([node.returns] if node.returns else [])])]
            loads = {n.id for expr in outer for n in ast.walk(expr)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            loads |= globals_read(tables[name])  # Inside the body: the symbol table's free globals only.
        else:
            loads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        pending.extend(loads & (bindings.keys() | bound))
    functions = {n for n in seen if isinstance(bindings[n][0], (ast.FunctionDef, ast.AsyncFunctionDef))}
    return functions, seen - functions


def _module_file(directory, parts):
    base = Path(directory).joinpath(*parts) if parts else None
    for candidate in ((base.with_suffix(".py"), base / "__init__.py") if base else ()):
        if candidate.is_file():
            return candidate
    return None


def local_imports(path, instrument_dir):
    """Files under instrument_dir that *path* imports: relative imports, and absolute dotted names whose tail
    (after the instrument directory's own name, or from the start) resolves to a module or package there."""
    directory = Path(instrument_dir)
    found = set()
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            dotted = [alias.name.split(".") for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".") if node.module else []
            dotted = [module] + [module + [alias.name] for alias in node.names if alias.name != "*"]
            if node.level:  # Relative: resolved against the importing file's directory, which is instrument_dir.
                for parts in dotted:
                    hit = _module_file(directory, parts)
                    if hit:
                        found.add(hit)
                continue
        else:
            continue
        for parts in dotted:
            for start in range(len(parts)):
                if start and parts[start - 1] != directory.name:
                    continue
                hit = _module_file(directory, parts[start:])
                if hit:
                    found.add(hit)
    return {f.relative_to(directory).as_posix() for f in found}


def build(instrument_dir, files, roots_by_file, loads_by_file, interpreter, data_by_file=None):
    manifest = {"schema": "d5-clone-pins/1", "interpreter": dict(interpreter),
                "files": [], "functions": [], "extra_hashes": []}
    for name in files:
        path = _path(instrument_dir, name)
        listed = set(loads_by_file.get(name, []))
        pending, scanned = [name], set()
        while pending:  # Imports of the file and, transitively, of every module it loads must all be listed.
            current = pending.pop()
            if current in scanned:
                continue
            scanned.add(current)
            imported = local_imports(Path(instrument_dir) / current, instrument_dir) - {current}
            if imported - listed:
                raise ValueError("a local import is not pinned in loads")
            pending.extend(imported)
        manifest["files"].append({"name": name, "sha256": _sha(path.read_bytes()), "loads": [
            {"name": load, "sha256": _sha(_load_path(instrument_dir, load).read_bytes())}
            for load in loads_by_file.get(name, [])]})
        data = sorted((data_by_file or {}).get(name, []))
        manifest["files"][-1]["data"] = data
        functions, assignments = call_closure(path, roots_by_file.get(name, []), data)
        manifest["functions"].extend(
            {"file": name, "qualname": f, "sha256": function_source_sha256(path, f)}
            for f in sorted(functions))
        manifest["extra_hashes"].extend(
            {"file": name, "kind": "assign", "name": a, "sha256": assign_source_sha256(path, a)}
            for a in sorted(assignments))
    unpinned = directory_files(instrument_dir) - set(files) - {
        load for name in files for load in loads_by_file.get(name, [])}
    if unpinned:
        raise ValueError("every file under the instrument directory must be pinned")
    (Path(instrument_dir) / MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def _interpreter_version(interpreter):
    if interpreter is None:
        return platform.python_version()
    try:
        probe = subprocess.run([str(interpreter), "-I", "-c", "import platform; print(platform.python_version())"],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return probe.stdout.strip() if probe.returncode == 0 else None


def verify(manifest_path, instrument_dir, python_version=None, interpreter=None):
    """Check every pin. The interpreter is the one the instruments run under (default: this one); its basename
    and its version (probed, unless *python_version* is given) must equal the manifest's."""
    mismatches = []
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not manifest["files"]:
            mismatches.append("empty")  # A manifest that pins nothing proves nothing.
        if manifest["schema"] != "d5-clone-pins/1":
            mismatches.append("schema")
        if manifest["interpreter"]["path_basename"] != Path(interpreter or sys.executable).name:
            mismatches.append("interpreter")
        version = python_version if python_version is not None else _interpreter_version(interpreter)
        if not isinstance(manifest["interpreter"]["version"], str) or manifest["interpreter"]["version"] != version:
            mismatches.append("interpreter")
        names = [f["name"] for f in manifest["files"]]
        listed = set(names)
        if len(listed) != len(names) or any(
                entry["file"] not in listed for entry in manifest["functions"] + manifest["extra_hashes"]):
            mismatches.append("manifest")  # Duplicate files, or a member of a file the manifest does not pin.
        entries = [(f["name"], f["sha256"], None) for f in manifest["files"]]
        entries += [(m["name"], m["sha256"], None) for f in manifest["files"] for m in f["loads"]]
        entries += [(f["file"], f["sha256"], f["qualname"]) for f in manifest["functions"]]
        entries += [(a["file"], a["sha256"], (a["name"],)) for a in manifest["extra_hashes"]
                    if a["kind"] == "assign"]
        if any(a["kind"] != "assign" for a in manifest["extra_hashes"]):
            mismatches.append("extra_hashes")
        for name, expected, member in entries:
            try:
                path = _load_path(instrument_dir, name)
                actual = (_sha(path.read_bytes()) if member is None else
                          assign_source_sha256(path, member[0]) if isinstance(member, tuple) else
                          function_source_sha256(path, member))
                if actual == expected:
                    continue
            except (OSError, ValueError, SyntaxError):
                pass
            safe = isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
            mismatches.append(name if safe else "manifest")  # Only a relative name under the instrument dir.
        pinned = {f["name"] for f in manifest["files"]} | {m["name"] for f in manifest["files"] for m in f["loads"]}
        if directory_files(instrument_dir) - pinned:
            mismatches.append("unpinned_files")  # A file added beside the instruments, by name only in the count.
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        mismatches.append("manifest")
    return {"verdict": "FAIL" if mismatches else "PASS", "mismatches": sorted(set(mismatches))}


def require(*args, **kwargs):
    if verify(*args, **kwargs)["verdict"] == "FAIL":
        raise SystemExit(2)
