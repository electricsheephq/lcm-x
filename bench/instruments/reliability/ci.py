"""Nightly CI helpers for .github/workflows/reliability-nightly.yml: host prep from hosts.ci.json, and the gate.

    python -m bench.instruments.reliability.ci prep --host eva-0.21.5 --root "$RUNNER_TEMP/hosts" --out hosts.json
    python -m bench.instruments.reliability.ci gate r1/results.jsonl r2/results.jsonl --open-issues open-issues.txt

The gate fails on any ERROR, and on a FAIL in the G-REL-1 cell set unless the cell targets an open issue. It also
fails on an empty results file, a host missing from any file, an empty set and, per (host, transport, plugin sha) present, on any missing, duplicate or unexpected cell
row against the ``--cells all`` list run_matrix.py uses for that transport (UNSUPPORTED rows count as present).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CI_HOSTS = Path(__file__).with_name("hosts.ci.json")
# G-REL-1 item 3 cell families; "crash-" covers every crash cell, gateway-second-restart the restart cells.
G_REL_1 = ("baseline", "acp-trailing", "preflight-continue", "repeat-identical-replies", "long-80", "window-1m",
           "pressure-disagreement", "lcm-tool-mid-turn", "parallel-tool-group", "crash-", "gateway-second-restart",
           "cancel-retry", "publication-failure")
# Data cells inside a gate family that never gate (still expected rows for completeness).
NON_GATE = ("publication-failure/rotation-child-persistent", "drain/hidden-backlog/in-place", "drain/hidden-backlog/rotation",
            "drain/hidden-backlog-large/in-place", "drain/hidden-backlog-large/rotation")


def in_gate_set(cell_id: str) -> bool:
    family = cell_id.split("/")[0]
    return cell_id not in NON_GATE and any(family == g or (g.endswith("-") and family.startswith(g)) for g in G_REL_1)


def expected_cells(transport: str | None) -> list[str]:
    """The cell ids run_matrix.py runs for ``--cells all`` on this transport (None: R1 in-process)."""
    from bench.instruments.reliability import cells as C
    extra = ()
    if transport:
        from bench.instruments.reliability import process_cell
        extra = process_cell.R2_CELLS
    return [c["id"] for c in C.select("all", extra=extra)]


def completeness(results: list[dict]) -> list[str]:
    """A non-empty set with exactly one row per expected cell for every (host, transport, plugin sha) present."""
    if not results:
        return ["empty result set"]
    groups, problems = {}, []
    for r in results:
        groups.setdefault((r.get("host"), r.get("transport"), r.get("plugin_sha")), []).append(r.get("cell"))
    for (host, transport, sha), got in sorted(groups.items(), key=str):
        where = f"{host} {transport or 'in-process'} {str(sha)[:12]}"
        want = expected_cells(transport)
        missing = sorted(set(want) - set(got))
        dupes = sorted({c for c in got if got.count(c) > 1})
        unexpected = sorted({str(c) for c in got} - set(want))
        for label, cells in (("missing", missing), ("duplicate", dupes), ("unexpected", unexpected)):
            if cells:
                problems.append(f"INCOMPLETE {where}: {len(cells)} {label} cell row(s): {cells[:5]}")
    return problems


def file_coverage(per_file: dict[str, list[dict]]) -> list[str]:
    """Every input results file (one lane: R1, R2) is non-empty and covers every host any file has: a wholly
    missing lane leaves no group for completeness() to check."""
    problems = [f"empty result file: {name}" for name, rows in per_file.items() if not rows]
    hosts = {r.get("host") for rows in per_file.values() for r in rows}
    for name, rows in per_file.items():
        if rows and (missing := sorted(hosts - {r.get("host") for r in rows}, key=str)):
            problems.append(f"result file {name} has no rows for host(s) {missing}")
    return problems


def gate(results: list[dict], open_issues: set[int], per_file: dict[str, list[dict]] | None = None) -> list[str]:
    problems = (file_coverage(per_file) if per_file is not None else []) + completeness(results)
    for r in results:
        where = f"{r['verdict']} {r['host']} {r['cell']} ({r.get('transport', 'in-process')})"
        if r["verdict"] == "ERROR":
            problems.append(f"{where}: {str(r.get('reason', ''))[:200]}")
        elif r["verdict"] == "FAIL" and in_gate_set(r["cell"]) and not set(r.get("targets") or []) & open_issues:
            problems.append(f"{where}: G-REL-1 cell fails and targets no open issue (targets {r.get('targets')})")
    return problems


def prep(name: str, root: Path, out: Path) -> dict:
    """Clone the pinned sha (the harness verifies git HEAD and cites the source) and install it editable."""
    spec = json.loads(CI_HOSTS.read_text())["hosts"][name]
    src, venv = root / name / "src", root / name / "venv"  # the venv stays outside the verified tree
    for cmd in (["git", "init", "-q", str(src)],
                ["git", "-C", str(src), "fetch", "-q", "--depth", "1", f"https://github.com/{spec['repo']}.git", spec["sha"]],
                ["git", "-C", str(src), "checkout", "-q", "--detach", "FETCH_HEAD"],
                ["uv", "venv", "-q", "--python", spec["python_version"], str(venv)],
                ["uv", "pip", "install", "-q", "-p", str(venv / "bin" / "python"), "-e", f"{src}[{spec['extras']}]", *spec["pins"]]):
        subprocess.run(cmd, check=True)
    hosts = {"hosts": {name: {"python": str(venv / "bin" / "python"), "src": str(src), "sha": spec["sha"],
                              "hermes_version": spec["hermes_version"]}}}
    out.write_text(json.dumps(hosts, indent=1) + "\n")
    return hosts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--host", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("results", nargs="+")
    g.add_argument("--open-issues", required=True, help="file with one open issue number per line")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        prep(a.host, Path(a.root).resolve(), Path(a.out))
        return 0
    per_file = {f: [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()] for f in a.results}
    results = [r for rows in per_file.values() for r in rows]
    open_issues = {int(n) for n in Path(a.open_issues).read_text().split()}
    problems = gate(results, open_issues, per_file)
    for line in problems:
        print(f"GATE: {line}")
    print(f"GATE {'FAILS' if problems else 'PASSES'}: {len(results)} cells, {len(problems)} gating")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
