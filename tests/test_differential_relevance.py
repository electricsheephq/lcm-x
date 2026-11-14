"""Keep differential relevance in step with the engine's static local imports."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Computed imports resolved by hand, keyed by file and import expression (not by
# line, so unrelated edits above a site do not move it). Do not infer the target
# of any other computed import: inventory it here after an audit.
COMPUTED_IMPORTS = {
    # recorder.py: importlib.import_module(package + ".escalation")
    ("recorder.py", "package + '.escalation'"): "escalation.py",
    # recorder.py: importlib.import_module(package + ".ingest_persisted")
    ("recorder.py", "package + '.ingest_persisted'"): "ingest_persisted.py",
}
# __init__.py:68 also loads the explicitly named config.py with
# spec_from_file_location (exec_module at :77). Ordinary imports already reach it.


def _workflow_paths() -> list[str]:
    lines = (ROOT / ".github/workflows/differential.yml").read_text(encoding="utf-8").splitlines()
    keys = [index for index, line in enumerate(lines)
            if line.strip() == "TURN_PATH_FILES: |-"]
    assert len(keys) == 1, "Expected one TURN_PATH_FILES block"
    start = keys[0]
    indentation = len(lines[start]) - len(lines[start].lstrip())
    paths = []
    for line in lines[start + 1:]:
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= indentation:
            break
        paths.append(line.strip())
    assert paths, "Empty TURN_PATH_FILES block"
    assert len(paths) == len(set(paths)), "Duplicate differential relevance paths"
    return paths


def _gate_paths(name: str = "TURN_PATH_FILES") -> set[str]:
    source = (ROOT / "scripts/maintainer_gate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    values = [node.value for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name
                      for target in node.targets)]
    assert len(values) == 1, f"Expected one maintainer {name} assignment"
    if name == "NON_TURN_PATH_FILES":
        # Exclusions are policy decisions, recorded beside each path on main.
        # A bare set member must not silently exempt a newly reached module.
        assert isinstance(values[0], ast.Set), "Expected literal documented exclusions"
        lines = source.splitlines()
        undocumented = [ast.literal_eval(node) for node in values[0].elts
                        if not lines[node.end_lineno - 1].partition("#")[2].strip()]
        assert not undocumented, "Exclusions need written reasons:\n" + "\n".join(sorted(undocumented))
    return ast.literal_eval(values[0])


def _module_path(root: Path, parts: tuple[str, ...]) -> str | None:
    """Resolve only repository modules, including the installed plugin name."""
    if parts[:1] == ("hermes_lcm",):
        parts = parts[1:]
    for end in range(1, len(parts)):
        if not root.joinpath(*parts[:end], "__init__.py").is_file():
            return None
    base = root.joinpath(*parts)
    if (base / "__init__.py").is_file():
        return (base / "__init__.py").relative_to(root).as_posix()
    if parts and base.with_suffix(".py").is_file():
        return base.with_suffix(".py").relative_to(root).as_posix()
    return None


def _import_closure(
    root: Path, computed: dict[tuple[str, str], str],
) -> tuple[set[str], Counter[tuple[str, str]]]:
    pending = ["engine.py"]
    reached: set[str] = set()
    unresolved: Counter[tuple[str, str]] = Counter()
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        path = Path(current)
        # Loading a member executes every enclosing package initializer too.
        for directory in (Path(), *reversed(path.parent.parents), path.parent):
            initializer = directory / "__init__.py"
            if (root / initializer).is_file():
                pending.append(initializer.as_posix())
        tree = ast.parse((root / current).read_text(encoding="utf-8"), filename=current)
        importlib_names = {"importlib"}
        import_module_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                importlib_names.update(alias.asname or alias.name for alias in node.names
                                       if alias.name == "importlib")
            elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
                import_module_names.update(alias.asname or alias.name for alias in node.names
                                           if alias.name == "import_module")
        for node in ast.walk(tree):  # Includes functions and TYPE_CHECKING branches.
            targets = []
            if isinstance(node, ast.Import):
                targets = [tuple(alias.name.split(".")) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = path.parent.parts if node.level else ()
                assert not node.level or node.level - 1 <= len(package), (
                    f"Import escapes repository package: {current}:{node.lineno}"
                )
                if node.level:
                    package = package[:len(package) - node.level + 1]
                target = package + tuple(node.module.split(".") if node.module else ())
                targets = [target] + [target + (alias.name,) for alias in node.names
                                      if alias.name != "*"]
            elif isinstance(node, ast.Call):
                function = node.func
                dynamic = (
                    isinstance(function, ast.Name)
                    and function.id in import_module_names | {"__import__"}
                ) or (
                    isinstance(function, ast.Attribute) and function.attr == "import_module"
                    and isinstance(function.value, ast.Name)
                    and function.value.id in importlib_names
                )
                if not dynamic:
                    continue
                name = node.args[0] if node.args else next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "name"), None,
                )
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    module = name.value
                    if module.startswith("."):
                        # Relative import_module names are relative to its package
                        # argument, normally __package__ in a plugin module.
                        package_arg = node.args[1] if len(node.args) > 1 else next(
                            (keyword.value for keyword in node.keywords if keyword.arg == "package"), None,
                        )
                        package = ("hermes_lcm",) + path.parent.parts
                        if isinstance(package_arg, ast.Constant) and isinstance(package_arg.value, str):
                            package = tuple(package_arg.value.split("."))
                        elif not (isinstance(package_arg, ast.Name) and package_arg.id == "__package__"):
                            unresolved[current, ast.unparse(node)] += 1
                            continue
                        level = len(module) - len(module.lstrip("."))
                        assert level <= len(package), f"Invalid relative import: {current}:{node.lineno}"
                        targets = [package[:len(package) - level + 1]
                                   + tuple(filter(None, module.lstrip(".").split(".")))]
                    else:
                        targets = [tuple(module.split("."))]
                else:
                    key = (current, ast.unparse(name) if name is not None else "<missing name>")
                    unresolved[key] += 1
                    if key in computed:
                        pending.append(computed[key])
            for parts in targets:
                # import pkg.child executes pkg/__init__.py as well as child.
                for end in range(1, len(parts) + 1):
                    resolved = _module_path(root, parts[:end])
                    if resolved is not None:
                        pending.append(resolved)
                if not parts:
                    resolved = _module_path(root, ())
                    if resolved is not None:
                        pending.append(resolved)
    return reached, unresolved


def test_differential_and_maintainer_paths_match() -> None:
    assert set(_workflow_paths()) == _gate_paths()
    assert not _gate_paths() & _gate_paths("NON_TURN_PATH_FILES"), "Listed paths cannot also be excluded"


def test_differential_paths_exist() -> None:
    paths = set(_workflow_paths()) | _gate_paths() | _gate_paths("NON_TURN_PATH_FILES")
    missing = sorted(path for path in paths if not (ROOT / path).is_file())
    assert not missing, "Nonexistent differential relevance paths:\n" + "\n".join(missing)


def test_engine_import_closure_is_listed_or_documented_excluded() -> None:
    closure, unresolved = _import_closure(ROOT, COMPUTED_IMPORTS)
    assert unresolved == Counter(COMPUTED_IMPORTS.keys()), (
        f"Computed import inventory needs an audit: {dict(unresolved)!r}"
    )
    missing = sorted(closure - set(_workflow_paths()) - _gate_paths("NON_TURN_PATH_FILES"))
    assert not missing, (
        "Engine import modules neither in TURN_PATH_FILES nor documented as excluded:\n"
        + "\n".join(missing)
    )


def test_static_closure_follows_packages_and_deferred_imports(tmp_path: Path) -> None:
    sources = {
        "__init__.py": "from . import policy\nraise RuntimeError('never execute')\n",
        "engine.py": (
            "import importlib as loader\nfrom importlib import import_module as load\n"
            "import pkg.child\nfrom pkg import member\nfrom . import sibling\n"
            "from hermes_lcm.named import value\nfrom external_dependency import ignored\n"
            "if TYPE_CHECKING:\n    import checked\n"
            "def deferred():\n    from .deferred import value\n"
            "loader.import_module('dynamic')\nload(name='.relative', package=__package__)\n"
            "load('.member', 'hermes_lcm.pkg')\n"
        ),
        "pkg/__init__.py": "from . import initialized\n",
        "pkg/child.py": "from .. import parent\n",
    }
    leaves = {"policy", "pkg/member", "pkg/initialized", "sibling", "named", "checked",
              "deferred", "dynamic", "relative", "parent"}
    sources.update({f"{name}.py": "" for name in leaves})
    for name, source in sources.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source, encoding="utf-8")
    closure, unresolved = _import_closure(tmp_path, {})
    assert closure == set(sources)
    assert not unresolved


def test_computed_imports_are_inventoried_without_guessing(tmp_path: Path) -> None:
    (tmp_path / "engine.py").write_text('import importlib\nimportlib.import_module(package + ".target")\n')
    (tmp_path / "target.py").write_text("")
    key = ("engine.py", "package + '.target'")
    closure, unresolved = _import_closure(tmp_path, {})
    assert closure == {"engine.py"}
    assert unresolved == Counter({key: 1})
    closure, _ = _import_closure(tmp_path, {key: "target.py"})
    assert closure == {"engine.py", "target.py"}
