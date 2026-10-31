"""Regression for PR 1689 dispatch."""
import ast
import sys
import pytest
from .bot1_support import DECISION, load


@pytest.mark.parametrize("status,expected", [("FAIL", 1), ("INCOMPLETE", 2), ("COMPLETE", 0)])
def test_deep_dispatch_exit_status(tmp_path, monkeypatch, status, expected):
    sys.path.insert(0, str(DECISION))
    import score_deep
    monkeypatch.setattr(score_deep, "write_deep", lambda _: dict(per_arm={"H": dict(result=status)}))
    mat = tmp_path / "seed-1"
    mat.mkdir()
    (mat / "deep_metadata.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["score_decision", "--material", str(tmp_path), "--run-root", str(tmp_path), "--out", str(tmp_path / "scores")])
    mod = load(DECISION / "score_decision.py")
    assert mod.main() == expected
    tree = ast.parse((DECISION / "score_decision.py").read_text())
    assert isinstance(tree.body[-1].body[0], ast.Raise)
