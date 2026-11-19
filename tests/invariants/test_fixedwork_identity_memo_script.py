"""#2856: scripts/identity_row_classes.py counts equal the gate's verdicts row by row, and print only counts."""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.reconcile import _freeze_replay_identity_value, _replay_identity_memo_gate
from tests.invariants.test_fixedwork_identity_memo import real_artifacts

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "identity_row_classes.py"
PURE = [{"role": "user", "content": "plain question"},
        {"role": "assistant", "content": "checking", "tool_calls": [
            {"id": "c9", "type": "function", "function": {"name": "read_file", "arguments": '{"p": 1}'}}]},
        {"role": "tool", "tool_call_id": "c9", "name": "read_file", "content": "a result"},
        {"role": "user", "content": [{"type": "text", "text": "parts"}]}]


def _expected(rows, stored):
    counts = Counter(total=len(rows), pure=0, multiple_kinds=0, unkeyable=0)
    for row in rows:
        kinds = _replay_identity_memo_gate(row["role"], normalize_content_value(row.get("content")) or "",
                                           _freeze_replay_identity_value(row.get("tool_calls")))
        counts["pure"] += not kinds
        counts["multiple_kinds"] += len(kinds) > 1
        counts.update(f"kind_{kind}" for kind in kinds)
    counts["stored" if stored else "live"] = len(rows)
    return counts


def _run(path, *flags):
    result = subprocess.run([sys.executable, str(SCRIPT), str(path), *flags], capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True)
    return result.stdout, {key: int(value) for key, value in (line.split("\t") for line in result.stdout.splitlines())}


def test_a9_counts_equal_the_gate_row_by_row(tmp_path):
    rows = [row for _kind, row in real_artifacts(tmp_path / "artifacts")] + PURE
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    for flags, stored in (((), False), (("--stored",), True)):
        stdout, counts = _run(path, *flags)
        expected = _expected(rows, stored)
        assert {k: v for k, v in counts.items() if v} == {k: v for k, v in expected.items() if v}
        assert counts["pure"] == len(PURE) and counts["total"] == len(rows)
        assert all(k.startswith("kind_") or k in {"total", "pure", "multiple_kinds", "unkeyable", "live", "stored"}
                   for k in counts)
        # Only counts: no content, ids or tool names reach the output.
        for needle in ("plain question", "c9", "read_file", "write_file", "call_spill", "Externalized", "summary"):
            assert needle not in stdout.replace("kind_externalized", "").replace("kind_summary", "")
