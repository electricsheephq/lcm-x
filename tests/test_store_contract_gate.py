"""Exercise gate decisions through the entry point, without git writes."""
import pytest

from scripts import store_contract_gate as gate


def document(version=1, digest="a", row=True, *, history="",
             reason="Synthetic reason."):
    digest = "sha256:" + digest * 64
    current_row = f"| {version} | {digest} | 2026-11-16 | {reason} |\n"
    return (f"Contract version: {version}\nDigest: {digest}\n" + history
            + (current_row if row else ""))


FIRST_ROW = document().splitlines(keepends=True)[-1]
SECOND_ROW = document(2, "b").splitlines(keepends=True)[-1]
SECOND_VERSION = document(2, "b", history=FIRST_ROW)


@pytest.mark.parametrize("base, current, expected", [
    pytest.param(document(), document(), 0, id="unchanged"),
    pytest.param(document(), document(digest="b", history=FIRST_ROW), 1,
                 id="digest-without-version-bump"),
    pytest.param(document(), SECOND_VERSION, 0,
                 id="iii-changed-digest-history-kept"),
    pytest.param(document(), document(2, "b", False, history=FIRST_ROW), 1,
                 id="missing-current-row"),
    pytest.param(document(), document(2, history=FIRST_ROW), 1,
                 id="version-without-digest"),
    pytest.param(document(), document(3, "b", history=FIRST_ROW), 1,
                 id="version-skipped"),
    pytest.param(None, document(), 0, id="initial-version-one"),
    pytest.param(None, document(2), 1, id="initial-version-two"),
    pytest.param(document() + SECOND_ROW,
                 SECOND_VERSION, 1, id="reused-current-row"),
    pytest.param(document(), document(2, "b"), 1,
                 id="i-changed-digest-history-removed"),
    pytest.param(document(), document(reason="Edited reason."), 1,
                 id="ii-unchanged-digest-history-edited"),
    pytest.param(SECOND_VERSION, document(1, "b", history=FIRST_ROW + SECOND_ROW),
                 1, id="iv-version-down"),
    pytest.param(document(), document(2, "b", history=FIRST_ROW, reason=" "), 1,
                 id="v-blank-current-reason"),
    pytest.param(document(), document() + "Contract version: 1\n", 1,
                 id="vi-two-version-lines"),
    pytest.param(document(), "\n".join(
        line for line in document().splitlines()
        if not line.startswith("Digest:")), 1, id="vii-missing-digest"),
    pytest.param(None, document(digest="g"), 1,
                 id="vii-invalid-digest"),
    pytest.param(None, document().replace("a" * 64, "a" * 63), 1,
                 id="vii-short-digest"),
    pytest.param(SECOND_VERSION, document(2, "b"), 1,
                 id="unchanged-digest-earlier-row-removed"),
    pytest.param(SECOND_VERSION, document(2, "b", history=FIRST_ROW.replace(
        "| 1 |", "| 3 |")), 1, id="earlier-row-version-edited"),
    pytest.param(SECOND_VERSION, document(2, "b", history=FIRST_ROW.replace(
        "a" * 64, "c" * 64)), 1, id="earlier-row-digest-edited"),
    pytest.param(SECOND_VERSION, document(2, "b", history=FIRST_ROW.replace(
        "2026-11-16", "2026-11-15")), 1, id="earlier-row-date-edited"),
    pytest.param(SECOND_VERSION, document(2, "b", history=FIRST_ROW.replace(
        "Synthetic reason.", "Edited reason.")), 1,
                 id="earlier-row-reason-edited"),
])
def test_gate_decisions(tmp_path, monkeypatch, base, current, expected):
    path = tmp_path / gate.DOC
    path.parent.mkdir()
    path.write_text(current, encoding="utf-8")
    replies = iter([str(tmp_path), "base-sha", gate.DOC if base else "",
                    *([base] if base else [])])
    monkeypatch.setattr(gate, "git", lambda *_args: next(replies))
    assert gate.main(["--base-ref", "HEAD^1"]) == expected
