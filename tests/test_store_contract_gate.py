"""Exercise the version decision table through the gate entry point, without git writes."""
import pytest

from scripts import store_contract_gate as gate


def document(version=1, digest="a", row=True):
    digest = "sha256:" + digest * 64
    return (f"Contract version: {version}\nDigest: {digest}\n"
            + (f"| {version} | {digest} | 2026-11-16 | Synthetic reason. |\n" if row else ""))


@pytest.mark.parametrize("base, version, digest, row, expected", [
    (document(), 1, "a", True, 0), (document(), 1, "b", True, 1),
    (document(), 2, "b", True, 0), (document(), 2, "b", False, 1),
    (document(), 2, "a", True, 1), (document(), 3, "b", True, 1),
    (None, 1, "a", True, 0), (None, 2, "a", True, 1),
    (document() + document(2, "b").splitlines()[-1] + "\n", 2, "b", True, 1),
])
def test_gate_decisions(tmp_path, monkeypatch, base, version, digest, row, expected):
    path = tmp_path / gate.DOC
    path.parent.mkdir()
    path.write_text(document(version, digest, row), encoding="utf-8")
    replies = iter([str(tmp_path), "base-sha", gate.DOC if base else "", *([base] if base else [])])
    monkeypatch.setattr(gate, "git", lambda *_args: next(replies))
    assert gate.main(["--base-ref", "HEAD^1"]) == expected
