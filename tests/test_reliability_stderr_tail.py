"""#985: an error reason keeps both ends of the host's stderr, so the command it asks for survives."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.instruments.reliability import process_cell as PC  # noqa: E402


def _cell(tmp_path: Path, text: str | None) -> PC.ProcessCell:
    cell = object.__new__(PC.ProcessCell)
    cell.d, cell.phase = tmp_path, 1
    if text is not None:
        (tmp_path / "host-1.stderr").write_text(text)
    return cell


def test_long_stderr_keeps_the_first_and_last_thousand_characters(tmp_path):
    text = "Installing dependencies. If that fails, run manually: pip install x" + "." * 5000 + "TAIL"
    out = _cell(tmp_path, text).stderr_tail()
    assert out.startswith("Installing dependencies. If that fails, run manually: pip install x")
    assert out.endswith("TAIL")
    assert " … " in out and len(out) == 2003


def test_short_and_missing_stderr_are_returned_whole(tmp_path):
    assert _cell(tmp_path, "short").stderr_tail() == "short"
    assert _cell(tmp_path / "missing", None).stderr_tail() == ""


def test_cutoff_is_exact_at_two_edges(tmp_path):
    at = "a" * 1000 + "b" * 1000
    assert _cell(tmp_path, at).stderr_tail() == at
    (tmp_path / "over").mkdir()
    over = "a" * 1000 + "x" + "b" * 1000
    assert _cell(tmp_path / "over", over).stderr_tail() == "a" * 1000 + " … " + "b" * 1000


def test_capped_reason_keeps_both_ends_within_its_budget(tmp_path):
    text = "HEAD" + "." * 5000 + "TAIL"
    out = _cell(tmp_path, text).stderr_tail(150)
    assert out.startswith("HEAD") and out.endswith("TAIL") and len(out) == 303
