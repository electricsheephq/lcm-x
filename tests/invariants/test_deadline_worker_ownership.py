"""Keep retrieval deadline workers bound to their registration's unload gate."""

import ast
from pathlib import Path


def test_retrieval_deadline_workers_forward_an_engine():
    path = Path(__file__).resolve().parents[2] / "tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    missing = []
    # This launcher in tools.py handles only engine-scoped retrieval. The
    # provider module's standalone timeout helper is outside this contract.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_run_within_deadline":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        engine = keywords.get("engine")
        if engine is None or (isinstance(engine, ast.Constant) and engine.value is None):
            worker = ast.unparse(keywords["name"]) if "name" in keywords else "unnamed worker"
            missing.append(f"{path.name}:{node.lineno}: {worker}")
    assert not missing, "Deadline workers must pass their retrieval engine:\n" + "\n".join(missing)
