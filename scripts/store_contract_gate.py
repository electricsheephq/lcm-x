"""Enforce the version and changelog of the documented store surface."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

DOC = "docs/store-contract.md"


def contract(text):
    version, = re.findall(r"^Contract version: ([1-9][0-9]*)$", text, re.M)
    digest, = re.findall(r"^Digest: (sha256:[0-9a-f]{64})$", text, re.M)
    rows = re.findall(r"^\| ([1-9][0-9]*) \| (sha256:[0-9a-f]{64}) \| (\d{4}-\d{2}-\d{2}) \| ([^|\r\n]+) \|$", text, re.M)
    if sum(v == version and d == digest and bool(reason.strip()) for v, d, _, reason in rows) != 1:
        raise ValueError("current version needs one changelog row with its digest and a one-line reason")
    return int(version), digest


def check(base, current):
    version, digest = contract(current)
    if base is None:
        if version != 1:
            raise ValueError("initial store contract must be version 1")
    else:
        old_version, old_digest = contract(base)
        if version != old_version + (digest != old_digest):
            raise ValueError("changed digest requires exactly version+1; unchanged digest requires the same version")
        if digest != old_digest and re.search(rf"^\| {version} \| {digest} \|", base, re.M):
            raise ValueError("changed digest needs a new changelog row")


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True)
    args = parser.parse_args(argv)
    try:
        root = Path(git(None, "rev-parse", "--show-toplevel").strip())
        base = git(root, "rev-parse", "--verify", "--end-of-options", args.base_ref + "^{commit}").strip()
        old = git(root, "show", f"{base}:{DOC}") if git(root, "ls-tree", "--name-only", base, "--", DOC) else None
        check(old, (root / DOC).read_text(encoding="utf-8"))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"store contract gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
