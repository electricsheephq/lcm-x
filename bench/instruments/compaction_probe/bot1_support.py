"""Offline extraction helpers for the seven PR 1689 regressions."""
import ast
import importlib.util
from pathlib import Path

HERE = Path(__file__).parent

S2 = HERE / "track_s/s2"

DECISION = HERE / "track_s/decision"

def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def function(path, name, namespace, owner=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]
