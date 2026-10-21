"""Require executed JUnit evidence for changed tests; fail closed on report gaps."""
import argparse
from collections import Counter
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def changed_tests(event, payload, root):
    if event == "pull_request":
        base, head = (payload["pull_request"][key]["sha"] for key in ("base", "head"))
    elif event == "merge_group":
        base, head = (payload["merge_group"][key] for key in ("base_sha", "head_sha"))
    else:
        return []
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", "--diff-filter=AMR", f"{base}...{head}"],
        cwd=root,
    ).decode().split("\0")
    return [p for p in paths if fnmatch.fnmatch(Path(p).name, "test_*.py")
            or fnmatch.fnmatch(Path(p).name, "*_test.py")]


def check(root, reports, changed, allowlist):
    executed, cases, collected, errors, allowed = Counter(), Counter(), set(), [], {}
    for line in allowlist.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        path, sep, reason = line.partition("#")
        if not sep or not path.strip() or not reason.strip():
            errors.append(f"invalid allowlist entry (reason required): {line}")
        else:
            allowed[path.strip()] = reason.strip()

    def relative(value):
        return Path(os.path.relpath((root / value).resolve(), root.resolve())).as_posix()

    if not reports:
        errors.append("missing JUnit reports")
    for report in reports:
        try:
            tree = ET.parse(report).getroot()
            if tree.tag not in {"testsuites", "testsuite"}:
                raise ValueError("not a JUnit testsuite")
            ran = 0
            for suite in tree.iter("testsuite"):
                if suite.get("file"):
                    collected.add(relative(suite.get("file")))
            for case in tree.iter("testcase"):
                file = case.get("file")
                if not file:
                    raise ValueError("testcase missing file attribute (requires xunit1)")
                path = relative(file)
                collected.add(path)
                cases[path] += 1
                if case.find("skipped") is None:
                    executed[path] += 1
                    ran += 1
            if not ran:
                errors.append(f"{report}: zero executed tests")
        except (OSError, ET.ParseError, ValueError) as exc:
            errors.append(f"{report}: missing or malformed JUnit: {exc}")
    summary = ["## Tests ran", "", "Changed test file | Executed | Status", "--- | ---: | ---"]
    for path, reason in sorted(allowed.items()):
        print(f"::notice file={path}::allowlisted: {reason}")
    for path in sorted(set(changed)):
        why = ("executed" if executed[path] else "all skipped" if path in collected
               else "not collected by any CI job")
        if path in collected and not executed[path]:
            why = "all skipped" if cases[path] else "collected with zero tests"
        summary.append(f"{path} | {executed[path]} | {why}")
        if not executed[path] and path not in allowed:
            errors.append(f"{path}: {why}")
    summary.extend(["", *(f"Allowlisted: {path} — {reason}" for path, reason in sorted(allowed.items())), *errors])
    return errors, "\n".join(summary) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", nargs="+", type=Path, required=True)
    parser.add_argument("--changed-files", type=Path)
    args = parser.parse_args()
    try:
        root = Path.cwd()
        changed = (args.changed_files.read_text().splitlines() if args.changed_files else
                   changed_tests(os.environ.get("GITHUB_EVENT_NAME", "push"),
                                 json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
                                 if os.environ.get("GITHUB_EVENT_PATH") else {}, root))
        errors, summary = check(root, args.junit, changed, root / ".github/tests-ran-allowlist.txt")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        errors, summary = [str(exc)], f"Tests-ran error: {exc}\n"
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(summary)
    print(summary)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
