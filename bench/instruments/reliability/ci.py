"""Prepare pinned hosts, enforce the absolute gate, and compare reliability matrices.

    python -m bench.instruments.reliability.ci prep --host ref-0.21.5 --root "$RUNNER_TEMP/hosts" --out hosts.json
    python -m bench.instruments.reliability.ci gate r1/results.jsonl r2/results.jsonl --open-issues open-issues.txt

The gate fails on any ERROR, and on a FAIL in the G-REL-1 cell set unless every failed bar is declared by an
open target in cells.ISSUES on that host (cells.ISSUE_HOSTS; empty or missing failed_bars always gates). It also
fails on an empty results file, a host missing from any file, an empty set and, per (host, transport, plugin sha) present, on any missing, duplicate or unexpected cell
row against the ``--cells all`` list run_matrix.py uses for that transport (UNSUPPORTED rows count as present).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path

from bench.instruments.reliability import report

CI_HOSTS = Path(__file__).with_name("hosts.ci.json")
# G-REL-1 item 3 cell families; "crash-" covers every crash cell, gateway-second-restart the restart cells.
G_REL_1 = ("baseline", "acp-trailing", "preflight-continue", "repeat-identical-replies", "long-80", "window-1m",
           "pressure-disagreement", "lcm-tool-mid-turn", "parallel-tool-group", "crash-", "gateway-second-restart",
           "cancel-retry", "publication-failure", "reply-held-crash")
# Data cells inside a gate family that never gate (still expected rows for completeness).
NON_GATE = ("drain/hidden-backlog/in-place", "drain/hidden-backlog/rotation",
            "stop-after-tool/no-retry/in-place",
            "drain/hidden-backlog-large/in-place", "drain/hidden-backlog-large/rotation",
            *(f"continuity/{shape}/{mode}" for shape in
              ("sole-user-tool-loop", "survival-fit", "survival-fit-older-turns")
              for mode in ("in-place", "rotation")))
# #1588: exact cells, transport and bars proven by nightly 39539195029; never whole families.
CELL_GATE = {
    # #1900 / #1588: permanent loss gate; accepted surplus and B1 multiplicity never gate.
    "publication-failure/rotation-child-persistent": ("in-process", ("B2D",)),
    **{f"continuity/forced-compaction-then-turn/{m}": ("acp-process", ("B1", "B2", "B3", "B4", "B8", "F4"))
       for m in ("in-place", "rotation")},
    **{f"native-on-off/{v}/in-place": ("in-process", ("B3", "B4", "B8")) for v in ("v0.23.3", "v0.24.3")},
    **{c: ("in-process", ("B1", "B2")) for c in ("overflow-recovery-restart/in-place",
       "overflow-recovery-restart/rotation", "upgrade-crash-after-rotation/rotation")},
}


def in_gate_set(cell_id: str) -> bool:
    family = cell_id.split("/")[0]
    return cell_id in CELL_GATE or (cell_id not in NON_GATE and
           any(family == g or (g.endswith("-") and family.startswith(g)) for g in G_REL_1))


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
    from bench.instruments.reliability import cells as C
    from bench.instruments.reliability import controls as CT

    problems = (file_coverage(per_file) if per_file is not None else []) + completeness(results)
    groups = {}
    for r in results:
        groups.setdefault((r["host"], r.get("transport")), []).append(r)
    for (host, transport), rows in groups.items():
        required = host in CT.P8_MUST_SUPPORT.get(transport, ())
        for cell, want in CT.P8_CONTROLS.items():
            got = [r for r in rows if r["cell"] == cell]
            where = f"{host} {cell} ({transport or 'in-process'}): P8 control"
            if not got and required:
                problems.append(f"{where}: missing row")
            for r in got:
                if r["verdict"] == "UNSUPPORTED" and not required:
                    continue
                if r["verdict"] != want:
                    problems.append(f"{where}: {r['verdict']}, expected {want}")
                elif want == "FAIL" and "B9" not in (r.get("failed_bars") or {}):
                    problems.append(f"{where}: FAIL without B9")
    for r in results:
        where = f"{r['verdict']} {r['host']} {r['cell']} ({r.get('transport', 'in-process')})"
        if r["verdict"] == "ERROR":
            problems.append(f"{where}: {report.edges(str(r.get('reason', '')))}")
        elif (r["verdict"] == "FAIL" or "B2S" in CELL_GATE.get(r["cell"], ("", ()))[1]) and in_gate_set(r["cell"]):
            targets = set(r.get("targets") or [])
            open_targets = targets & open_issues
            transport = r.get("transport", "in-process")
            declared = {bar for target in open_targets if target in C.ISSUES
                        and r["host"].startswith(C.ISSUE_HOSTS.get(target, ("",)))
                        and transport in C.ISSUE_TRANSPORTS.get(target, (transport,)) for bar in C.ISSUES[target][0]}
            failed = set(r.get("failed_bars") or {})
            if r["cell"] in CELL_GATE:
                lane, bars = CELL_GATE[r["cell"]]
                if transport != lane:
                    continue
                if "B2D" in bars and "B2" in failed:
                    b2 = (r.get("numbers") or {}).get("B2") or {}
                    deficit = b2.get("deficit_rows")
                    if deficit is None or deficit > 0:
                        failed.add("B2D")
                if "B2S" in bars:
                    b2 = (r.get("numbers") or {}).get("B2") or {}
                    n = {k: b2.get(k) for k in ("deficit_rows", "silent_surplus_rows", "visible_surplus_rows",
                                                "surplus_rows", "tool_surplus_rows", "tool_missing_rows",
                                                "split_keys")}
                    ext = b2.get("externalized")
                    n["externalized"] = ext.get("deficit_rows") if isinstance(ext, dict) else None
                    # Fail closed: a proven verdict (PASS, or FAIL with its bars), every count present and a
                    # non-negative int, and the split covers all surplus. An UNSUPPORTED or INCONCLUSIVE row
                    # did not prove its scenario, so clean counts never satisfy B2S. A split assistant reply
                    # fails too: the scorer counts no deficit for a split key whatever its expected count, so
                    # two expected copies matched by one split pair hide a lost copy.
                    if (r["verdict"] not in ("PASS", "FAIL")
                            or any(type(v) is not int or v < 0 for v in n.values())
                            or n["visible_surplus_rows"] + n["silent_surplus_rows"]
                            != n["surplus_rows"] + n["tool_surplus_rows"]
                            or n["deficit_rows"] or n["silent_surplus_rows"] or n["tool_missing_rows"]
                            or n["externalized"] or n["split_keys"]):
                        failed.add("B2S")
                if not failed and r["verdict"] != "FAIL":
                    continue
                if failed:
                    failed &= set(bars) | ({"B9"} if lane == "acp-process" and r["host"] == "upstream-uid" else set())
                    if not failed:
                        continue
            uncovered = sorted(failed - declared)
            if not failed or uncovered:
                detail = f"uncovered bars {uncovered}" if failed else "empty or missing failed_bars; nothing attributable"
                problems.append(f"{where}: G-REL-1 cell fails: {detail} "
                                f"(targets {sorted(targets)}, open targets {sorted(open_targets)})")
    return problems


def prep(name: str, root: Path, out: Path) -> dict:
    """Clone the pinned sha (the harness verifies git HEAD and cites the source) and install it editable."""
    spec = json.loads(CI_HOSTS.read_text())["hosts"][name]
    src, venv = root / name / "src", root / name / "venv"  # the venv stays outside the verified tree
    for cmd in (["git", "init", "-q", str(src)],
                ["git", "-C", str(src), "fetch", "-q", "--depth", "1", f"https://github.com/{spec['repo']}.git", spec["sha"]],
                ["git", "-C", str(src), "checkout", "-q", "--detach", "FETCH_HEAD"],
                ["uv", "venv", "-q", "--python", spec["python_version"], str(venv)]):
        subprocess.run(cmd, check=True)
    # uv pip does not inherit project-level overrides from an editable dependency.
    # Use the pinned host's own policy (e.g. its Pillow/browser-harness override).
    project = tomllib.loads((src / "pyproject.toml").read_text())
    overrides = project.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    override_args = []
    if overrides:
        override_file = root / name / "overrides.txt"
        override_file.write_text("\n".join(overrides) + "\n")
        override_args = ["--overrides", str(override_file)]
    subprocess.run(["uv", "pip", "install", "-q", "-p", str(venv / "bin" / "python"),
                    "-e", f"{src}[{spec['extras']}]", *spec["pins"], *override_args], check=True)
    hosts = {"hosts": {name: {"python": str(venv / "bin" / "python"), "src": str(src), "sha": spec["sha"],
                              "hermes_version": spec["hermes_version"]}}}
    out.write_text(json.dumps(hosts, indent=1) + "\n")
    return hosts


ROUNDS = ("r1", "r2", "default-window-r1")
FAILING = {"FAIL", "ERROR", "INCONCLUSIVE"}
FLAKE_WINDOW = 7
TABLE_HEADER = "| Cell | Baseline verdict | Candidate verdict | Class |\n|---|---|---|---|"
KEY_PATTERN = re.compile(r"[a-zA-Z0-9_.-]+@[0-9a-f]{40}/(?:r1|r2|default-window-r1)/[a-zA-Z0-9_./-]+")
ALARM_LABEL = "nightly-regression"
ALARM_TITLE = "Reliability nightly regression"
ALARM_MARKER = "### Tracked cells\n\n```json\n"
TABLES_ARTIFACT = "reliability-tables"


def load_tables(root: Path) -> tuple[dict, set]:
    rows, flakes = {}, set()
    for file in sorted(root.rglob("results.jsonl")):
        round_name = file.parent.name
        if round_name not in ROUNDS:
            continue
        keys = {}
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cell = row["cell"]
            key = f"{row['host']}@{row['host_sha']}/{round_name}/{cell}"
            if not KEY_PATTERN.fullmatch(key) or row["verdict"] not in report.VERDICTS:
                raise ValueError(f"invalid cell record in {file}")
            if key in rows:
                raise ValueError(f"duplicate cell {key} in {file}")
            rows[key] = row["verdict"]
            keys[cell] = key
        metadata = file.with_name("run.json")
        if metadata.is_file():
            for cell in json.loads(metadata.read_text()).get("comparison_flakes", []):
                if cell not in keys or rows[keys[cell]] not in FAILING:
                    raise ValueError(f"invalid retry record in {metadata}")
                flakes.add(keys[cell])
    if not rows:
        raise ValueError(f"no results.jsonl rows in {root}")
    return rows, flakes


def load_side(name: str, root: Path) -> tuple[dict, set]:
    try:
        return load_tables(root)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{name}: {exc}") from exc


def compare(candidate: tuple[dict, set], baseline: tuple[dict, set], known: set, past_flakes: set,
            *, nightly: bool) -> list[tuple[str, str, str, str]]:
    current, current_flakes = candidate
    previous, previous_flakes = baseline
    rows = []
    for key in sorted(current.keys() | previous.keys()):
        before, after = previous.get(key, "—"), current.get(key, "—")
        kind = "MATCHED"
        if key not in current:
            kind = "MISSING"
        elif after in FAILING and (key in known or before not in FAILING or (nightly and key in previous_flakes)):
            kind = "NEW"
            if nightly and key in current_flakes and key not in known and key not in past_flakes:
                kind = "FLAKE"
        elif before in FAILING and after not in FAILING:
            kind = "FIXED"
        rows.append((key, before, after, kind))
    return rows


def table_lines(rows):
    return [f"| `{key}` | {before} | {after} | {kind} |" for key, before, after, kind in rows]


def read_known(data: list) -> set:
    if not isinstance(data, list):
        raise ValueError("known-new must contain a JSON list")
    keys = set()
    for entry in data:
        if isinstance(entry, str):
            keys.add(entry)
        elif (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str)
              and isinstance(entry[1], list) and all(isinstance(cell, str) for cell in entry[1])):
            prefix, cells = entry
            keys.update(f"{prefix}/{cell}" for cell in cells)
        else:
            raise ValueError("invalid known-new entry")
    if any(not KEY_PATTERN.fullmatch(key) for key in keys):
        raise ValueError("invalid known-new cell key")
    return keys


def pack_known(keys: set) -> list:
    groups = defaultdict(list)
    for key in sorted(keys):
        build, round_name, cell = key.split("/", 2)
        groups[f"{build}/{round_name}"].append(cell)
    return list(groups.items())


def retry_new(candidate: Path, baseline: tuple[dict, set], known: set, scratch: Path) -> None:
    """Retry each fresh failing cell once on the same pinned host and transport."""
    current = load_side("candidate", candidate)
    groups = defaultdict(list)
    for key, _, _, kind in compare(current, baseline, known, set(), nightly=True):
        if kind == "NEW" and key not in known:
            build, round_name, cell = key.split("/", 2)
            host, sha = build.split("@")
            groups[(host, sha, round_name)].append(cell)
    if not groups:
        return
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.11.26"], check=True, stdout=sys.stderr)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"cell retries unavailable: {exc}", file=sys.stderr)
        return
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        scratch = Path(tmp)
        host_files = {}
        for (host, sha, round_name), cells in groups.items():
            try:
                if host not in host_files:
                    hosts_file = scratch / f"{host}.json"
                    spec = prep(host, scratch / "hosts", hosts_file)
                    host_files[host] = (hosts_file, spec["hosts"][host]["sha"])
                hosts_file, prepared_sha = host_files[host]
                if sha != prepared_sha:
                    raise ValueError("retry requires the original pinned host SHA")
                results = scratch / f"reliability-{host}" / round_name
                transport = ["--transport", "acp-process"] if round_name == "r2" else []
                subprocess.run(
                    [sys.executable, "-m", "bench.instruments.reliability.run_matrix", *transport,
                     "--hosts-file", str(hosts_file), "--hosts", host, "--plugin-ref", "HEAD",
                     "--cells", ",".join(cells), "--jobs", "4", "--out", str(results)],
                    check=True, stdout=sys.stderr,
                )
                retried, _ = load_tables(results.parent)
                build = f"{host}@{sha}/{round_name}"
                passed = [cell for cell in cells if retried.get(f"{build}/{cell}") == "PASS"]
                if passed:
                    metadata = candidate / f"reliability-{host}" / round_name / "run.json"
                    run = json.loads(metadata.read_text())
                    run["comparison_flakes"] = passed
                    pending = metadata.with_name(".run.json.pending")
                    pending.write_text(json.dumps(run, indent=1) + "\n")
                    pending.replace(metadata)
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
                print(f"retry unavailable for {host}/{round_name}: {exc}", file=sys.stderr)


def compare_command(args) -> int:
    nightly = args.mode == "nightly"
    try:
        baseline = load_side("baseline", args.baseline)
        known = read_known(json.loads(args.known_new.read_text())) if args.known_new else set()
        # The current night, baseline and five older comparable nights make seven.
        past_flakes = baseline[1].copy()
        if nightly:
            included = {args.candidate.resolve(), args.baseline.resolve()}
            history = dict.fromkeys(path.resolve() for path in args.history)
            older_nights = [path for path in history if path not in included]
            for older in older_nights[:FLAKE_WINDOW - 2]:
                past_flakes.update(load_side("history", older)[1])
            if args.retry_scratch:
                retry_new(args.candidate, baseline, known, args.retry_scratch)
        candidate = load_side("candidate", args.candidate)
    except (OSError, ValueError) as exc:
        print(f"compare: {exc}", file=sys.stderr)
        return 2
    rows = compare(candidate, baseline, known, past_flakes, nightly=nightly)
    print(TABLE_HEADER, *table_lines(rows), sep="\n")
    new = [row for row in rows if row[-1] == "NEW"]
    missing = any(row[-1] == "MISSING" for row in rows)
    if args.alarm_result:
        tracked = {row[0] for row in new} | (known - candidate[0].keys())
        result = {"new": new, "missing": missing, "known": pack_known(tracked)}
        args.alarm_result.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    return int(bool(new or missing))


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True)


def baseline_ids() -> list[str]:
    repo = os.environ["GITHUB_REPOSITORY"]
    available = set(gh(
        "api", "--method", "GET", "--paginate", f"repos/{repo}/actions/artifacts",
        "-f", f"name={TABLES_ARTIFACT}", "-f", "per_page=100",
        "--jq", ".artifacts[] | select(.expired == false) | .workflow_run.id // empty",
    ).splitlines())
    if not available:
        return []
    runs = gh(
        "api", "--method", "GET", "--paginate", f"repos/{repo}/actions/workflows/reliability-nightly.yml/runs",
        "-f", f"branch={os.environ['GITHUB_REF_NAME']}", "-f", "event=schedule", "-f", "status=completed",
        "-f", "per_page=100", "--jq", ".workflow_runs[] | select(.conclusion != \"cancelled\") | .id",
    ).splitlines()
    return [run for run in runs if run in available and run != os.environ["GITHUB_RUN_ID"]][:FLAKE_WINDOW - 1]


def open_alarm() -> dict | None:
    repo = os.environ["GITHUB_REPOSITORY"]
    issues = gh(
        "api", "--method", "GET", "--paginate", f"repos/{repo}/issues",
        "-f", f"labels={ALARM_LABEL}", "-f", "state=open", "-f", "per_page=100",
        "--jq", '.[] | select(.pull_request == null) | {number, title, body} | @json',
    )
    alarms = [issue for line in issues.splitlines() if (issue := json.loads(line))["title"] == ALARM_TITLE]
    if len(alarms) > 1:
        raise ValueError("multiple open reliability nightly alarms")
    return alarms[0] if alarms else None


def alarm_state() -> set:
    issue = open_alarm()
    if issue is None:
        return set()
    repo = os.environ["GITHUB_REPOSITORY"]
    comments = gh(
        "api", "--paginate", f"repos/{repo}/issues/{issue['number']}/comments?per_page=100",
        "--jq", '.[] | select(.user.login == "github-actions[bot]") | .body | @json',
    ).splitlines()
    for body in [*(json.loads(comment) for comment in reversed(comments)), issue["body"]]:
        if ALARM_MARKER in body:
            return read_known(json.loads(body.split(ALARM_MARKER, 1)[1].split("\n```", 1)[0]))
    raise ValueError("open nightly alarm has no tracked cells block")


def alarm_messages(result: dict, run_url: str) -> list[str]:
    intro = f"[Reliability nightly run]({run_url})"
    state = ALARM_MARKER + json.dumps(result["known"], separators=(",", ":")) + "\n```"
    if not result["new"]:
        return [f"{intro}\n\nComparison clean.\n\n{state}\n"]
    lines = table_lines(result["new"])
    messages = []
    body = f"{intro}\n\n{state}\n\n{TABLE_HEADER}\n"
    for line in lines:
        if len(body) + len(line) > 60_000:
            messages.append(body)
            body = f"{intro}\n\n{TABLE_HEADER}\n"
        body += line + "\n"
    return [*messages, body]


def post_alarm(result: dict, scratch: Path) -> None:
    if not result["new"] and result["missing"]:
        return
    repo = os.environ["GITHUB_REPOSITORY"]
    run_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    issue = open_alarm()
    if issue is None and not result["new"]:
        return
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        body_file = Path(tmp) / "body.md"
        for body in alarm_messages(result, run_url):
            body_file.write_text(body)
            if issue is None:
                labels = json.loads(gh("label", "list", "--search", ALARM_LABEL, "--json", "name"))
                if not any(label["name"] == ALARM_LABEL for label in labels):
                    gh("label", "create", ALARM_LABEL, "--color", "B60205",
                       "--description", "Reliability nightly found a new failing cell")
                issue = open_alarm()
                if issue is None:
                    print(gh("issue", "create", "--title", ALARM_TITLE, "--label", ALARM_LABEL,
                             "--body-file", str(body_file)).strip())
                    issue = open_alarm()
                    if issue is None:
                        raise ValueError("created nightly alarm could not be read back")
                    continue
            comment = json.loads(gh("api", "--method", "POST", f"repos/{repo}/issues/{issue['number']}/comments",
                                    "-F", f"body=@{body_file}", "--jq", "{id, html_url}"))
            receipt = gh("api", f"repos/{repo}/issues/comments/{comment['id']}", "--jq", ".body | @json")
            if json.loads(receipt).rstrip("\n") != body.rstrip("\n"):
                raise ValueError("nightly alarm comment could not be read back")
            print(comment["html_url"])


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
    c = sub.add_parser("compare")
    c.add_argument("--candidate", required=True, type=Path)
    c.add_argument("--baseline", required=True, type=Path)
    c.add_argument("--mode", required=True, choices=("nightly", "respin"))
    c.add_argument("--known-new", type=Path)
    c.add_argument("--history", type=Path, action="append", default=[], help="older comparable nights, newest first")
    c.add_argument("--retry-scratch", type=Path, help="in CI, retry fresh failures under this scratch directory")
    c.add_argument("--alarm-result", type=Path, help="write the nightly issue payload as JSON")
    sub.add_parser("baselines", help="list up to six completed scheduled runs with table artifacts")
    alarm = sub.add_parser("alarm", help="read or post the scheduled nightly alarm")
    operations = alarm.add_mutually_exclusive_group(required=True)
    operations.add_argument("--read", type=Path)
    operations.add_argument("--post", type=Path)
    alarm.add_argument("--scratch", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        prep(a.host, Path(a.root).resolve(), Path(a.out))
        return 0
    if a.cmd == "compare":
        return compare_command(a)
    if a.cmd == "baselines":
        for run in baseline_ids():
            print(run)
        return 0
    if a.cmd == "alarm":
        if a.read:
            a.read.write_text(json.dumps(pack_known(alarm_state())) + "\n")
        else:
            post_alarm(json.loads(a.post.read_text()), a.scratch)
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
