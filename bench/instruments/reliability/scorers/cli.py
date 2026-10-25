"""Standalone scoring of an lcm.db copy.

    python -m bench.instruments.reliability.scorers.cli --db <lcm.db> --gauntlet-run <run dir>   # a live gauntlet run
    python -m bench.instruments.reliability.scorers.cli --cell-dir <harness cell dir> --plugin-tree <checkout>  # re-score one cell

The DB (with any -wal/-shm) is first copied into a private temp dir and only the copy is opened, read-only:
the source file is never opened, so a positive-control DB cannot be modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from bench.instruments.reliability.scorers import bars, continuity, dupes, multiset  # noqa: E402
from bench.instruments.reliability import stop_turn  # noqa: E402
from bench.instruments.reliability.process_cell import apply_f4, read_jsonl  # noqa: E402
from bench.instruments.reliability.run_matrix import unfired_reason  # noqa: E402


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
    ap.add_argument("--plugin-tree", type=Path, help="exact-SHA git checkout for --cell-dir scoring")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.plugin_tree is not None and not a.cell_dir:
        ap.error("--plugin-tree requires --cell-dir")
    if a.cell_dir:
        if not (a.cell_dir / "cell.json").is_file():
            ap.exit(2, "cannot re-score: no cell.json (pre-run cells have no scoring inputs)\n")
        cell = json.loads((a.cell_dir / "cell.json").read_text())
        uploaded = json.loads((a.cell_dir / "verdict.json").read_text()) if (a.cell_dir / "verdict.json").is_file() else {}
        def refuse(reason):
            ap.exit(2, f"cannot re-score: {reason}\n")

        if uploaded.get("verdict") == "ERROR":
            refuse("uploaded verdict is ERROR (a harness decision)")
        if cell.get("drain") and cell.get("transport") != "acp-process":
            refuse("drain cells need the R2 observer's per-compaction counters (acp-process only)")
        if not (a.cell_dir / "phase-A.json").is_file():
            refuse("no phase-A.json (no probe scoring inputs)")
        if not (a.cell_dir / "db/lcm.db").is_file():
            refuse("no retained database (db/lcm.db absent)")
        if cell.get("transport") == "acp-process":
            if not uploaded.get("acp_session"):
                refuse("acp-process verdict.json has no acp_session")
            cell["chat_root"] = uploaded["acp_session"]
        f4 = cell.get("id", "").startswith("continuity/")
        if f4 and (cell.get("transport") != "acp-process" or any(
                not (a.cell_dir / name).is_file() for name in ("provider-requests.jsonl", "observer.jsonl"))):
            refuse("continuity F4 requires acp-process provider-requests.jsonl and observer.jsonl")
        if cell.get("stop_after_tool") and any(not (a.cell_dir / name).is_file() for name in (
                "phase-B.json", "observer.jsonl", "transcript.jsonl", "provider-requests.jsonl",
                "stop-immediate.json", "stop-end.json", "stop-catchup.json")):
            refuse("stop_after_tool disk scoring inputs are incomplete")
        if a.plugin_tree is not None:
            expected = cell["plugin"]["sha"]
            tree = a.plugin_tree.resolve()
            env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")}
            try:
                identity = subprocess.check_output(
                    ["git", "-C", str(tree), "rev-parse", "--is-inside-work-tree", "--show-toplevel", "HEAD"],
                    env=env, text=True, stderr=subprocess.DEVNULL).splitlines()
                dirty = subprocess.check_output(
                    ["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"],
                    env=env, text=True, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.CalledProcessError):
                ap.error(f"--plugin-tree must be a git checkout with HEAD equal to {expected}")
            if identity != ["true", str(tree), expected] or dirty:
                ap.error(f"--plugin-tree must be a clean checkout root with HEAD equal to {expected}; got {identity}")
            cell["plugin"]["tree"] = str(tree)
        elif not Path(cell["plugin"]["tree"]).exists():
            refuse("recorded plugin tree is missing; supply --plugin-tree <checkout>")
        fired_path, phase_path = a.cell_dir / "faults-fired.jsonl", a.cell_dir / "phase-A.json"
        fired = {json.loads(line)["kind"] for line in fired_path.read_text().splitlines() if line.strip()} if fired_path.is_file() else set()
        citations = json.loads(phase_path.read_text()).get("citations", {}) if phase_path.is_file() else {}
        reason = unfired_reason({**cell, "faults": cell.get("faults", [])}, fired, citations)
        if cell.get("stop_after_tool"):
            run = SimpleNamespace(d=a.cell_dir, transcript=a.cell_dir / "transcript.jsonl",
                                  notes=lambda kind: [n for n in read_jsonl(a.cell_dir / "observer.jsonl")
                                                      if n["kind"] == kind])
            last = json.loads((a.cell_dir / "phase-B.json").read_text())
            report = stop_turn.score(run, last)
        else:
            report = {"verdict": "UNSUPPORTED", "reason": reason} if reason else bars.score(cell, a.cell_dir)
            if f4:
                requests = read_jsonl(a.cell_dir / "provider-requests.jsonl")
                candidates = [r for r in requests if r.get("role") == "main" and r.get("unexpected")]
                if any("rid" not in r for r in candidates):
                    refuse("continuity F4 scenario findings are unavailable on disk")
                unique = {r["rid"]: r for r in candidates}
                if any(not r.get("current_user_missing") or any(k not in r for k in (
                        "turn", "current_user_projected")) for r in unique.values()):
                    refuse("continuity F4 scenario findings are unavailable on disk")
                unexpected = [{k: r[k] for k in ("turn", "current_user_missing", "current_user_projected")}
                              for r in unique.values()]
                report["continuity"] = continuity.score_dir(cell, a.cell_dir)
                apply_f4(report, cell, unexpected)
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
