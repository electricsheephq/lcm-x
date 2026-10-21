"""Whole-source continuity receipt rejects value-only paraphrases and echoes."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("role,content,exact", [
    ("user", "[synthetic-id] Audit the synthetic fixture.\n", True),
    ("user", "A paraphrase: Audit the synthetic fixture.", False),
    ("assistant", "[synthetic-id] Audit the synthetic fixture.\n", False),
])
def test_continuity_requires_whole_source_request_bytes(role, content, exact):
    path = Path(__file__).resolve().parents[1] / "bench/instruments/compaction_probe/track_s/s2/run_s_lcmx.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "continuity")
    namespace = {"R": SimpleNamespace(flatten=lambda _, rows: "\n".join(r["content"] for r in rows))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    source = [{"role": "user", "content": "[synthetic-id] Audit the synthetic fixture.\n"}]
    man = {"continuity": [{"id": "S1-CONT-current_request", "row_index": 0, "value": "Audit the synthetic fixture."}]}
    receipt = namespace["continuity"](man, "", [{"role": role, "content": content}], source)[0]
    assert receipt["present"] is True  # the old value-substring check alone passes all three
    assert receipt["whole_source_request_in_view"] is exact
