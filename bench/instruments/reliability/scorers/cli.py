"""Standalone scoring of an lcm.db copy.

    python -m bench.instruments.reliability.scorers.cli --db <lcm.db> --gauntlet-run <run dir>   # a live gauntlet run
    python -m bench.instruments.reliability.scorers.cli --cell-dir <harness cell dir>           # re-score one cell

The DB (with any -wal/-shm) is first copied into a private temp dir and only the copy is opened, read-only:
the source file is never opened, so a positive-control DB cannot be modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from bench.instruments.reliability.scorers import bars, dupes, multiset  # noqa: E402


def gauntlet_transcript(run: Path) -> list[tuple[str, str]]:
    """The gauntlet driver's run dir: material (or actual-material) + probes are the prompts sent, in order;
    results.jsonl holds each turn's ``raw_answer`` (same loader as lossless_bar_multiset.py)."""
    def rows(p):
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    def text(r):
        for key in ("text", "prompt", "content", "message", "question"):
            if key in r:
                v = r[key]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
        raise ValueError("prompt has no text field")
    material = run / "actual-material.jsonl" if (run / "actual-material.jsonl").is_file() else run / "material/turns.jsonl"
    sent = [text(r) for r in rows(material) + rows(run / "probes.jsonl")]
    items = []
    results = rows(run / "results.jsonl")
    if len(sent) != len(results):
        raise ValueError(f"prompt/result count mismatch: prompts={len(sent)}, results={len(results)}, "
                         f"first misaligned index={min(len(sent), len(results))}")
    for i, r in enumerate(results):
        if "input_sha256" in r and r["input_sha256"] != hashlib.sha256(sent[i].encode("utf-8")).hexdigest():
            raise ValueError(f"input_sha256 mismatch at index {i}")
        items.append(("user", sent[i]))
        if r.get("raw_answer"):
            items.append(("assistant", r["raw_answer"]))
    return items


def private_copy(db: Path, tmp: Path) -> Path:
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(db) + suffix)
        if src.exists():
            shutil.copy2(src, tmp / (db.name + suffix))
    return tmp / db.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path)
    ap.add_argument("--gauntlet-run", type=Path)
    ap.add_argument("--cell-dir", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cell_dir:
        cell = json.loads((a.cell_dir / "cell.json").read_text())
        report = bars.score(cell, a.cell_dir)
    else:
        if not (a.db and a.gauntlet_run):
            ap.error("--db and --gauntlet-run are required without --cell-dir")
        with tempfile.TemporaryDirectory() as tmp:
            copy = private_copy(a.db, Path(tmp))
            import sqlite3
            con = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
            try:
                stored = con.execute("select store_id, session_id, role, coalesce(content,''), conversation_id "
                                     "from messages order by store_id").fetchall()
            finally:
                con.close()
            try:
                ms = multiset.phase_c_score(gauntlet_transcript(a.gauntlet_run), stored)
            except ValueError as exc:
                ms = {"verdict": "INCONCLUSIVE", "reason": str(exc)}
            dup = dupes.count(copy)
        report = {"db": str(a.db), "multiset": {k: v for k, v in ms.items() if k not in ("missing", "duplicated", "extra")},
                  "dupes": {k: v for k, v in dup.items() if k != "per_session"}}
    rendered = json.dumps(report, indent=1, default=str) + "\n"
    if a.out:
        a.out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
