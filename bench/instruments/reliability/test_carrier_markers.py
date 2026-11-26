"""carrier_markers finds the summary-part header in summary_prefix.py when a tree has that file, and in engine.py otherwise.

v0.30.4 moved `_LCM_SUMMARY_PART_HEADER_RE` from engine.py to summary_prefix.py. A reader that looked only in engine.py
and reconcile.py raised on such a tree. The scorer then refused host-merge credit, and crash and restart cells read FAIL
on stores identical to the previous release's (release decisions T1-619 and T1-621).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.instruments.reliability import plugin_tree

HEADER = r"^\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]"
HEADER_LINE = f"_LCM_SUMMARY_PART_HEADER_RE = re.compile({HEADER!r})\n"


def _tree(root: Path, header_in: str | None) -> Path:
    """A plugin tree with the objective prefix in engine.py, the todo prefix in reconcile.py, and the header in one file."""
    (root / "engine.py").write_text(
        "import re\n" + '_PRESERVED_OBJECTIVE_PREFIX = "[Objective] "\n' + (HEADER_LINE if header_in == "engine.py" else ""))
    (root / "reconcile.py").write_text('_PRESERVED_TODO_PREFIX = "[Todo] "\n')
    if header_in == "summary_prefix.py":
        (root / "summary_prefix.py").write_text("import re\n" + HEADER_LINE)
    return root


def test_header_in_engine_py_on_a_tree_without_summary_prefix_py(tmp_path: Path) -> None:
    header, prefixes = plugin_tree.carrier_markers(_tree(tmp_path, "engine.py"))
    assert header.pattern == HEADER
    assert prefixes == ("[Objective] ", "[Todo] ")


def test_header_moved_to_summary_prefix_py(tmp_path: Path) -> None:
    header, prefixes = plugin_tree.carrier_markers(_tree(tmp_path, "summary_prefix.py"))
    assert header.pattern == HEADER
    assert header.match("[Depth-2 Summary (d2, node 7)]")
    assert prefixes == ("[Objective] ", "[Todo] ")


def test_summary_prefix_py_without_the_header_keeps_engine_py_header(tmp_path: Path) -> None:
    tree = _tree(tmp_path, "engine.py")
    (tree / "summary_prefix.py").write_text("LIMIT = 1\n")
    header, prefixes = plugin_tree.carrier_markers(tree)
    assert header.pattern == HEADER
    assert prefixes == ("[Objective] ", "[Todo] ")


def test_no_header_in_any_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no generated-carrier markers"):
        plugin_tree.carrier_markers(_tree(tmp_path, None))


def test_this_repository_tree_still_reads() -> None:
    header, prefixes = plugin_tree.carrier_markers(Path(__file__).resolve().parents[3])
    assert header.pattern and prefixes
