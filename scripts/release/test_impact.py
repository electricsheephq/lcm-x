"""Decide which test suites a Git diff affects, failing closed on unknown paths."""

import argparse
from collections import Counter, namedtuple
from functools import cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "bench/specs/test-impact-map.json"
KEYS = "version glob_rules inputs path_sets fail_closed_paths ignored_paths mechanisms suites".split()
Decision = namedtuple("Decision", "changed_mechanisms fail_closed unmapped verdicts")


class MapError(Exception):
    pass


def load_map(path):
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        if missing := set(KEYS) - m.keys():
            raise MapError(f"missing keys: {', '.join(sorted(missing))}")
        mechanism_ids = {row["id"] for row in m["mechanisms"]}
        if len(mechanism_ids) != len(m["mechanisms"]) or len({row["id"] for row in m["suites"]}) != len(m["suites"]):
            raise MapError("duplicate mechanism or suite id")
        path_lists = [m["fail_closed_paths"], m["ignored_paths"], *m["path_sets"].values()]
        path_lists.extend(row["paths"] for row in m["mechanisms"] + m["suites"])
        if any(not isinstance(paths, list) or not all(isinstance(p, str) for p in paths) for paths in path_lists):
            raise MapError("paths must be lists of strings")
        for suite in m["suites"]:
            suite_id, kind = suite["id"], suite["kind"]
            mechanisms, paths, inputs = (suite[key] for key in ("mechanisms", "paths", "inputs"))
            if kind not in ("mapped", "manual") or (kind == "mapped" and not (mechanisms or paths or inputs)):
                raise MapError(f"{suite_id}: invalid kind {kind!r} or missing mapped suite triggers")
            unknown = ((set(mechanisms) - mechanism_ids) | (set(inputs) - m["inputs"].keys())
                       | {p for p in paths if p.startswith("@") and p[1:] not in m["path_sets"]})
            if unknown:
                raise MapError(f"{suite_id}: unknown references {', '.join(sorted(unknown))}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise MapError(str(exc)) from exc
    return m


@cache
def compile_glob(glob):
    wildcards = {"**/": "(?:[^/]+/)*", "**": ".*", "*": "[^/]*", "?": "[^/]"}
    tokens = re.findall(r"\*\*/|\*\*|\*|\?|.", glob, re.DOTALL)
    return re.compile("".join(wildcards.get(token, re.escape(token)) for token in tokens), re.DOTALL)


def literal_globs(m):
    own_paths = {glob for row in m["suites"] for glob in row["paths"] if not glob.startswith("@")}
    return own_paths.union(m["fail_closed_paths"], m["ignored_paths"], *(row["paths"] for row in m["mechanisms"]))


def decide(m, paths, inputs=()):
    paths, inputs = list(paths), list(inputs)
    if unknown := set(inputs) - m["inputs"].keys():
        raise MapError(f"unknown input: {', '.join(sorted(unknown))}")

    def matching(globs):
        patterns = tuple(map(compile_glob, globs))
        return [path for path in paths if any(pattern.fullmatch(path) for pattern in patterns)]

    changed = {row["id"]: hits for row in m["mechanisms"] if (hits := matching(row["paths"]))}
    fail_closed = matching(m["fail_closed_paths"])
    mapped = set(matching(literal_globs(m)))
    unmapped = [path for path in paths if path not in mapped]
    verdicts = []
    for suite in m["suites"]:
        name = suite["name"]
        if suite["kind"] == "manual":
            verdicts.append((suite["id"], "MANUAL", f"{name}: not decided by the map ({suite['note']})"))
            continue
        reasons = [f"{mid}: {changed[mid][0]}" for mid in suite["mechanisms"] if mid in changed]
        globs = [glob for p in suite["paths"] for glob in (m["path_sets"][p[1:]] if p.startswith("@") else [p])]
        reasons.extend(f"path {path}" for path in matching(globs)[:1])
        reasons.extend(f"input {input_name}" for input_name in inputs if input_name in suite["inputs"])
        if closed := fail_closed or unmapped:
            reasons.append(f"fail closed: {closed[0]}")
        verdict = ("RUN", "; ".join(reasons)) if reasons else ("SKIP", f"suite {name} skipped: nothing it measures changed")
        verdicts.append((suite["id"], *verdict))
    return Decision(changed, fail_closed, unmapped, verdicts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--map", type=Path, default=MAP)
    try:
        args = parser.parse_args()
        revision = f"{args.base}..{args.head}"
        m = load_map(args.map)
        git = ["git", "diff", "--no-renames", "--name-only", "-z", "--end-of-options", revision, "--"]
        paths = os.fsdecode(subprocess.check_output(git, cwd=ROOT)).split("\0")[:-1]
        decision = decide(m, paths, args.input)
    except MapError as exc:
        print(f"BAD MAP: {exc}", file=sys.stderr)
        return 2
    except (subprocess.CalledProcessError, SystemExit) as exc:
        return 0 if isinstance(exc, SystemExit) and exc.code == 0 else 1
    print(f"test-impact {revision}: {len(paths)} changed paths")
    for path in decision.unmapped:
        print(f"UNMAPPED {path}: no row covers it; every mapped suite runs (fail closed)")
    for path in decision.fail_closed:
        print(f"FAIL-CLOSED {path}: hub or unassigned product file; every mapped suite runs")
    order = ("RUN", "SKIP", "MANUAL")
    for sid, verdict, reason in sorted(decision.verdicts, key=lambda row: order.index(row[1])):
        print(f"{verdict} {sid}: {reason}")
    counts = Counter(verdict for _, verdict, _ in decision.verdicts)
    print("counts:", *(f"{status.lower()}={counts[status]}" for status in order), f"unmapped_paths={len(decision.unmapped)}")


if __name__ == "__main__":
    sys.exit(main())
