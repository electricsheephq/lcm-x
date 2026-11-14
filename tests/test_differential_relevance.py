"""Keep differential relevance in step with the engine's static local imports."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import subprocess

import pytest


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

# Engine-reached NON_TURN_PATH_FILES exclusions, reviewed with this PR on 2026-11-14.
# Newly reached exclusions need their written reason re-audited before adding them;
# remove exclusions that are no longer reached.
ENGINE_REACHED_EXCLUSIONS = frozenset({
    "adaptive_retrieval.py", "answer_contract.py", "assertion_extraction.py",
    "assertion_rebuild.py", "assertion_state.py", "assertion_store.py", "chunking.py",
    "command.py", "command_auth.py", "command_doctor.py", "command_router.py", "command_status.py",
    "cpu_capacity.py", "embedding_maintenance.py", "embedding_provider.py", "entity_store.py",
    "evidence_compiler.py", "evidence_pack.py", "evidence_tools.py", "expand_query_tools.py",
    "expand_tools.py", "expansion_evidence.py", "guidance.py", "inactive_record.py", "inspect_tools.py",
    "level3_repair.py", "maintenance.py", "memory_tools.py", "occurrence_time.py", "presets.py",
    "query_view_store.py", "reasoning.py", "recall_delivery.py", "recall_session.py", "recent.py",
    "recorder.py", "requirements_compiler.py", "retrieval_context.py", "retrieval_core.py",
    "runtime_identity.py", "schema_families.py", "search_query.py", "selective_compiler.py",
    "selective_recall.py", "sqlite_session.py", "tools.py", "trajectory_store.py", "vector_store.py",
})


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


def _tracked_python_files(root: Path) -> set[str]:
    """Repository modules are tracked files. CI writes an untracked agent/ host
    stub at the root (ci.yml), which must not count as an engine module."""
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
                            capture_output=True, check=True).stdout.decode("utf-8")
    return {path for path in listed.split("\0") if path}


def _module_path(root: Path, parts: tuple[str, ...], tracked: set[str] | None) -> str | None:
    """Resolve only repository modules, including the installed plugin name."""
    if parts[:1] == ("hermes_lcm",):
        parts = parts[1:]

    def known(relative: Path) -> bool:
        return (root / relative).is_file() and (tracked is None or relative.as_posix() in tracked)

    def package(directory: Path) -> bool:
        if known(directory / "__init__.py"):
            return True
        if tracked is not None:
            prefix = directory.as_posix() + "/"
            return any(file.startswith(prefix) and file.endswith(".py") and (root / file).is_file()
                       for file in tracked)
        return any(file.is_file() for file in (root / directory).rglob("*.py"))

    for end in range(1, len(parts)):
        if not package(Path(*parts[:end])):
            return None
    base = Path(*parts)
    if known(base / "__init__.py"):
        return (base / "__init__.py").as_posix()
    if parts and known(base.with_suffix(".py")):
        return base.with_suffix(".py").as_posix()
    return None


def _import_closure(
    root: Path, computed: dict[tuple[str, str], str], tracked: set[str] | None = None,
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
            if (root / initializer).is_file() and (tracked is None or initializer.as_posix() in tracked):
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
                if isinstance(function, ast.Name) and function.id == "__import__":
                    fromlist = node.args[3] if len(node.args) > 3 else next(
                        (item.value for item in node.keywords if item.arg == "fromlist"), None,
                    )
                    level = node.args[4] if len(node.args) > 4 else next(
                        (item.value for item in node.keywords if item.arg == "level"), ast.Constant(0),
                    )
                    static_fromlist = isinstance(fromlist, (ast.List, ast.Tuple)) and all(
                        isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in fromlist.elts
                    )
                    supported = len(node.args) + len(node.keywords) == 1 or (
                        static_fromlist and isinstance(level, ast.Constant)
                        and type(level.value) is int and level.value == 0
                        and len(node.args) <= 5 and all(item.arg in {
                            "name", "globals", "locals", "fromlist", "level",
                        } for item in node.keywords)
                    )
                    if isinstance(name, ast.Constant) and isinstance(name.value, str) and supported:
                        target = tuple(name.value.split("."))
                        targets = [target] + [target + (item.value,) for item in
                                              (fromlist.elts if static_fromlist else [])
                                              if item.value != "*"]
                    else:
                        key = (current, ast.unparse(node))
                        unresolved[key] += 1
                        if key in computed:
                            pending.append(computed[key])
                elif isinstance(name, ast.Constant) and isinstance(name.value, str):
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
                            key = (current, ast.unparse(node))
                            unresolved[key] += 1
                            if key in computed:
                                pending.append(computed[key])
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
                    resolved = _module_path(root, parts[:end], tracked)
                    if resolved is not None:
                        pending.append(resolved)
                if not parts:
                    resolved = _module_path(root, (), tracked)
                    if resolved is not None:
                        pending.append(resolved)
    return reached, unresolved


def _reached_exclusion_changes(
    closure: set[str], excluded: set[str], pinned: frozenset[str],
) -> tuple[set[str], set[str]]:
    reached = closure & excluded
    return reached - pinned, pinned - reached


def test_engine_import_closure_is_listed_or_documented_excluded() -> None:
    closure, unresolved = _import_closure(ROOT, COMPUTED_IMPORTS, _tracked_python_files(ROOT))
    assert unresolved == Counter(COMPUTED_IMPORTS.keys()), (
        f"Computed import inventory needs an audit: {dict(unresolved)!r}"
    )
    excluded = _gate_paths("NON_TURN_PATH_FILES")
    newly_reached, no_longer_reached = _reached_exclusion_changes(
        closure, excluded, ENGINE_REACHED_EXCLUSIONS,
    )
    assert not newly_reached, (
        "Newly reached exclusions: re-audit their written reason in scripts/maintainer_gate.py, "
        "then add them to ENGINE_REACHED_EXCLUSIONS:\n" + "\n".join(sorted(newly_reached))
    )
    assert not no_longer_reached, (
        "No longer reached: remove from ENGINE_REACHED_EXCLUSIONS:\n"
        + "\n".join(sorted(no_longer_reached))
    )
    missing = sorted(closure - _gate_paths() - excluded)
    hint = ("\nBoth lists hold top-level modules only "
            "(tests/test_maintainer_gate.py::test_top_level_modules_are_classified). "
            "A reached subpackage module needs path support in both lists and in the relevance "
            "step of .github/workflows/differential.yml." if any("/" in path for path in missing) else "")
    assert not missing, (
        "Engine import modules neither in TURN_PATH_FILES nor documented as excluded:\n"
        + "\n".join(missing) + hint
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


@pytest.mark.parametrize(("call", "expression"), [
    ("loader.import_module(package + '.target')", "package + '.target'"),
    ("loader.import_module('.target', package=package)", None),
])
def test_computed_imports_are_inventoried_without_guessing(
    tmp_path: Path, call: str, expression: str | None,
) -> None:
    (tmp_path / "engine.py").write_text(f"import importlib as loader\n{call}\n")
    (tmp_path / "target.py").write_text("")
    key = ("engine.py", expression or ast.unparse(ast.parse(call).body[0].value))
    closure, unresolved = _import_closure(tmp_path, {})
    assert closure == {"engine.py"}
    assert unresolved == Counter({key: 1})
    closure, _ = _import_closure(tmp_path, {key: "target.py"})
    assert closure == {"engine.py", "target.py"}


def test_namespace_package_respects_tracked_files(tmp_path: Path) -> None:
    (tmp_path / "engine.py").write_text("from .features import handler\n")
    (tmp_path / "features").mkdir()
    (tmp_path / "features/handler.py").write_text("")
    for tracked in (None, {"engine.py", "features/handler.py"}, {"engine.py"}):
        closure, unresolved = _import_closure(tmp_path, {}, tracked)
        assert closure == ({"engine.py"} if tracked == {"engine.py"}
                           else {"engine.py", "features/handler.py"})
        assert not unresolved


@pytest.mark.parametrize(("call", "expected"), [
    ("__import__('pkg.member')", {"pkg/__init__.py", "pkg/member.py"}),
    ("__import__('pkg', fromlist=['member'])", {"pkg/__init__.py", "pkg/member.py"}),
    ("__import__('pkg', None, None, ['member'])", {"pkg/__init__.py", "pkg/member.py"}),
    ("__import__('pkg', globals=g, locals=l, fromlist=('member', '*'), level=0)",
     {"pkg/__init__.py", "pkg/member.py"}),
    ("__import__('pkg', fromlist=[])", {"pkg/__init__.py"}),
    ("__import__('x', level=1)", None),
    ("__import__(name)", None),
    ("__import__('pkg', fromlist=members)", None),
    ("__import__('pkg', fromlist=['member'], level=1)", None),
])
def test_builtin_imports(tmp_path: Path, call: str, expected: set[str] | None) -> None:
    (tmp_path / "engine.py").write_text(call + "\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/member.py").write_text("")
    closure, unresolved = _import_closure(tmp_path, {})
    assert closure == {"engine.py"} | (expected or set())
    key = ("engine.py", ast.unparse(ast.parse(call).body[0].value))
    assert unresolved == (Counter({key: 1}) if expected is None else Counter())
    if expected is None:
        closure, unresolved = _import_closure(tmp_path, {key: "pkg/member.py"})
        assert closure == {"engine.py", "pkg/__init__.py", "pkg/member.py"}
        assert unresolved == Counter({key: 1})


def test_reached_exclusion_changes() -> None:
    assert _reached_exclusion_changes(
        {"new.py", "kept.py", "listed.py"}, {"new.py", "kept.py", "old.py"},
        frozenset({"kept.py", "old.py"}),
    ) == ({"new.py"}, {"old.py"})
    assert _reached_exclusion_changes({"kept.py"}, {"kept.py"}, frozenset({"kept.py"})) == (set(), set())


def test_untracked_host_stub_is_not_a_repository_module(tmp_path: Path) -> None:
    # The CI test job writes an untracked agent/ stub of the Hermes host at the root.
    sources = {
        "engine.py": "from agent.context_engine import ContextEngine\nfrom . import store\n",
        "store.py": "",
        "agent/__init__.py": "",
        "agent/context_engine.py": "",
    }
    for name, source in sources.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source, encoding="utf-8")
    closure, _ = _import_closure(tmp_path, {}, {"engine.py", "store.py"})
    assert closure == {"engine.py", "store.py"}
    closure, _ = _import_closure(tmp_path, {})  # Without a tracked set the stub would count.
    assert {"agent/__init__.py", "agent/context_engine.py"} <= closure
