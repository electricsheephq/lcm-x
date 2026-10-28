"""docs/invariants.md stays mechanically consistent: unique IDs, real test files, proofs with receipts."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "invariants.md"


def _rows():
    rows = []
    for line in DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| I") and re.fullmatch(r"I\d+", cells[0]):
            rows.append(cells)
    return rows


def test_catalogue_ids_are_unique_and_rows_complete():
    rows = _rows()
    ids = [r[0] for r in rows]
    assert ids, "no invariant rows parsed"
    assert len(ids) == len(set(ids)), ids
    assert all(len(r) == 6 for r in rows), [r[0] for r in rows if len(r) != 6]


def test_every_named_test_file_exists():
    names = set(re.findall(r"\btest_[A-Za-z0-9_]+\.py\b", DOC.read_text(encoding="utf-8")))
    missing = sorted(n for n in names if not (ROOT / "tests" / n).exists())
    assert not missing, missing


def test_proven_rows_name_a_receipt():
    # A row is enforced only with a recorded red/green seed proof (docs/invariants.md, "Recording a seed proof").
    for row in _rows():
        proof = row[5]
        assert proof == "gap" or re.fullmatch(r"proven: .*(#\d+|https://\S+).*", proof), (row[0], proof)
        links = re.findall(r"\]\((invariants/[^)#]+\.md)#([^)]+)\)", proof)
        assert proof == "gap" or links, (row[0], "a proven row links its receipt", proof)
        for target, anchor in links:
            receipt = DOC.parent / target
            assert receipt.is_file(), (row[0], target)
            headings = re.findall(r"^## (.+)$", receipt.read_text(encoding="utf-8"), re.MULTILINE)
            slugs = {re.sub(r"[^a-z0-9 _-]", "", heading.lower()).replace(" ", "-") for heading in headings}
            assert anchor in slugs, (row[0], target, anchor)


def test_receipts_link_existing_inverse_patches():
    receipts = DOC.parent / "invariants" / "receipts.md"
    patches = re.findall(r"\]\((reverts/[^)]+\.patch)\)", receipts.read_text(encoding="utf-8"))
    assert patches
    for patch in patches:
        assert (receipts.parent / patch).is_file(), patch
