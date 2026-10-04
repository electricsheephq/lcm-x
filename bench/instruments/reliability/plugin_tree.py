"""Export an lcm-x git ref with ``git archive`` and read its plugin identity FROM the exported tree.

v0.23.x trees install as ``hermes-lcm`` / engine ``lcm``; v0.24.x as ``hermes-lcm-x`` / ``lcm-x``. Nothing
here is hard-coded: the dir name and ``plugins.enabled`` entry come from plugin.yaml ``name``, the engine
from plugin_identity.py ``ENGINE_NAME`` (or, before that file existed, engine.py's ``name`` property).
"""
from __future__ import annotations

import ast
import io
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

from bench.instruments.reliability.hosts import tree_hash


def resolve(repo: Path, ref: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def identity(tree: Path) -> dict:
    name = re.search(r"^name:\s*[\"']?([\w.-]+)", (tree / "plugin.yaml").read_text(), re.M).group(1)
    ident = tree / "plugin_identity.py"
    engine = None
    if ident.exists():
        m = re.search(r"^ENGINE_NAME\s*=\s*[\"']([\w.-]+)[\"']", ident.read_text(), re.M)
        engine = m and m.group(1)
    if engine is None:
        m = re.search(r"def name\(self\)[^:]*:\s*\n\s*return\s+[\"']([\w.-]+)[\"']", (tree / "engine.py").read_text())
        engine = m and m.group(1)
    if not engine:
        raise ValueError(f"cannot read the context-engine name from {tree}")
    return {"dir": name, "enabled": name, "engine": engine, "module": "hermes_plugins." + name.replace("-", "_")}


def export(repo: Path, ref: str, plugins_root: Path, out: Path | None = None) -> dict:
    """``out`` (the run's output dir): the plugins root must be a plain dir resolving under it, checked before any
    cache deletion or marker write; default: the plugins root's own parent."""
    out = Path(out) if out is not None else plugins_root.parent
    if plugins_root.is_symlink() or out.resolve() not in plugins_root.resolve().parents:
        raise ValueError(f"plugins root {plugins_root} is a symlink or does not resolve under {out}; refused")
    sha = resolve(repo, ref)
    dest, marker = plugins_root / sha[:12], plugins_root / f"{sha[:12]}.export.json"
    try:  # reuse a cached export only if it is the same sha AND its tree still hashes as it did at export time
        rec = json.loads(marker.read_text())
        reuse = rec.get("sha") == sha and not dest.is_symlink() and tree_hash(dest)[0] == rec.get("tree_sha256")
    except (OSError, ValueError):
        reuse = False
    if not reuse:
        if dest.is_symlink() or plugins_root.resolve() not in dest.resolve().parents:
            raise ValueError(f"plugin export dir {dest} is not a plain dir under {plugins_root}")
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        blob = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", sha], capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(dest, filter="data")
        marker.write_text(json.dumps({"sha": sha, "tree_sha256": tree_hash(dest)[0]}) + "\n")
    return {"ref": ref, "sha": sha, "tree": str(dest), "reused": reuse, **identity(dest)}


def parse_bool(tree: Path, key: str, value: str) -> tuple[bool, str]:
    """The plugin's OWN env boolean parser at this tree (config.py ``_parse_bool_env``), applied to ``value``;
    an unparseable value keeps the field default False (config.py ``native_recovery: bool = False``)."""
    src = (Path(tree) / "config.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_parse_bool_env")
    ns = {"os": SimpleNamespace(environ={key: value})}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "config.py", "exec"), ns)  # noqa: S102 - the plugin's parser
    return bool(ns["_parse_bool_env"](key, False)), f"config.py:{fn.lineno}"


def recovery_prefixes(tree: Path) -> tuple[str, ...]:
    """Literal prefixes before formatting fields, from the scored tree's recovery constants."""
    return tuple(ast.literal_eval(n.value).split("{", 1)[0]
                 for n in ast.walk(ast.parse((Path(tree) / "engine.py").read_text()))
                 if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in {"_OVERFLOW_RECOVERY_PLACEHOLDER", "_OVERFLOW_RECOVERY_OVERCAP_NOTE"})


def carrier_markers(tree: Path) -> tuple[re.Pattern, tuple[str, ...]]:
    """The plugin's OWN generated-carrier markers at this tree: engine.py ``_LCM_SUMMARY_PART_HEADER_RE`` (the
    ``[Recent|Session Arc|Durable|Depth-N Summary (dN, node N)]`` part header its carrier detection verifies) and every
    ``_PRESERVED_*_PREFIX`` string constant in engine.py/reconcile.py (objective/todo carriers). Raises if absent."""
    header, prefixes = None, []
    for name in ("engine.py", "reconcile.py"):
        for node in ast.walk(ast.parse((Path(tree) / name).read_text())):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target, value = node.targets[0].id, node.value
                if target == "_LCM_SUMMARY_PART_HEADER_RE" and isinstance(value, ast.Call) and value.args:
                    header = ast.literal_eval(value.args[0])
                elif re.fullmatch(r"_PRESERVED_\w+_PREFIX", target) and isinstance(value, ast.Constant):
                    prefixes.append(value.value)
    if not header or not prefixes:
        raise ValueError(f"no generated-carrier markers in {tree}")
    return re.compile(header), tuple(prefixes) + recovery_prefixes(tree)
