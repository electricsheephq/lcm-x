"""store.py must not reach command.py through its imports (#2138).

Walks the edges the release row selector (#2090) follows: module-level and
function-level imports, `hermes_lcm.*` string constants and import_module
literals. While store.py reaches command.py, a change to almost any module
selects every release row.
"""
import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module_file(name):
    parts = name.split(".")[1:]
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    for path in (ROOT.joinpath(*parts, "__init__.py"), ROOT.joinpath(*parts).with_suffix(".py")):
        if path.is_file():
            return path
    return None


def _targets(path):
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    name = ".".join(["hermes_lcm", *parts])
    package = name if is_package else name.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ""
            if node.level:
                target = resolve_name("." * node.level + target, package)
            yield target
            yield from (f"{target}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("hermes_lcm."):
                yield node.value
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            func_name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            arg = node.args[0]
            if func_name in {"import_module", "__import__"} and isinstance(arg, ast.Constant) \
                    and isinstance(arg.value, str):
                if not arg.value.startswith("."):
                    yield arg.value
                else:
                    package_arg = node.args[1] if len(node.args) > 1 else next(
                        (k.value for k in node.keywords if k.arg == "package"), None)
                    if isinstance(package_arg, ast.Constant) and isinstance(package_arg.value, str):
                        yield resolve_name(arg.value, package_arg.value)


def _imported_files(target):
    if not target.startswith("hermes_lcm"):
        target = "hermes_lcm." + target
    parts = target.split(".")
    for end in range(2, len(parts) + 1):  # importing a.b runs a/__init__.py too
        path = _module_file(".".join(parts[:end]))
        if path is not None:
            yield path


def _closure(start):
    seen, pending = set(), [start]
    while pending:
        path = pending.pop()
        if path not in seen:
            seen.add(path)
            for target in _targets(path):
                pending.extend(_imported_files(target))
    return seen


def test_store_import_closure_excludes_command():
    closure = _closure(ROOT / "store.py")
    assert ROOT / "config.py" in closure  # store.py imports config at module level
    assert ROOT / "command.py" not in closure
