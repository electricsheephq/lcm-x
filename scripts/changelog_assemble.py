"""Fold changelog.d/ fragments into the Unreleased section of CHANGELOG.md.

Usage: python3 scripts/changelog_assemble.py [--check] [--keep] [--repo PATH]

Exit codes: 0 success or no fragments; 1 invalid fragment; 2 CHANGELOG.md has no `## Unreleased` heading.
Nothing is written unless every fragment is valid and the heading exists.
"""
import argparse
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

HEADING = b"## Unreleased"
FRAGMENT_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*\.md$")


def validate(path):
    """Return (lines, error) for one fragment; lines are the non-blank lines, each ending in a newline."""
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "empty fragment"
    if not lines[0].startswith("- "):
        return None, "first non-blank line must start with '- '"
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip() and not (line.startswith("- ") or line.startswith("  ")):
            return None, f"line {number} must start with '- ' or two spaces: {line!r}"
    return [line + "\n" for line in lines], None


def insert_at(changelog):
    """Return the byte offset just after `## Unreleased` and its following blank line, or None."""
    lines = changelog.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        offset += len(line)
        if line.rstrip(b"\r\n") == HEADING:
            if index + 1 < len(lines) and not lines[index + 1].strip():
                offset += len(lines[index + 1])
            return offset
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate and print what would be added; change nothing")
    parser.add_argument("--keep", action="store_true", help="assemble without deleting the fragment files")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent, help="repository root")
    args = parser.parse_args(argv)

    fragment_dir = args.repo / "changelog.d"
    entries = sorted(p for p in fragment_dir.iterdir() if p.name != "README.md") if fragment_dir.is_dir() else []
    errors = [f"{p.name}: file name must match {FRAGMENT_NAME.pattern}" for p in entries if not FRAGMENT_NAME.match(p.name)]
    fragments = [p for p in entries if FRAGMENT_NAME.match(p.name)]

    added = []
    for path in fragments:
        lines, error = validate(path)
        if error:
            errors.append(f"{path.name}: {error}")
        else:
            added.extend(lines)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    if not fragments:
        print("no changelog fragments")
        return 0

    changelog_path = args.repo / "CHANGELOG.md"
    changelog = changelog_path.read_bytes()
    offset = insert_at(changelog)
    if offset is None:
        print("error: CHANGELOG.md has no '## Unreleased' heading", file=sys.stderr)
        return 2

    block = "".join(added).encode("utf-8")
    rest = changelog[offset:]
    if rest.startswith(b"#"):
        block += b"\n"
    if args.check:
        print(f"would add {len(added)} line(s) from {len(fragments)} fragment(s):")
        sys.stdout.write("".join(added))
        return 0

    handle, temp = tempfile.mkstemp(dir=changelog_path.parent, prefix=".CHANGELOG.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(changelog[:offset] + block + rest)
        os.chmod(temp, stat.S_IMODE(os.stat(changelog_path).st_mode))
        os.replace(temp, changelog_path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise
    if not args.keep:
        undeleted = []
        for path in fragments:
            try:
                path.unlink()
            except OSError as exc:
                undeleted.append(f"{path.name} ({exc})")
        if undeleted:
            print(
                "error: CHANGELOG.md is already updated, but these fragments could not be deleted; delete them by hand "
                "before any rerun or their entries will be added twice: " + ", ".join(undeleted),
                file=sys.stderr,
            )
            return 1
    print(f"added {len(added)} line(s) from {len(fragments)} fragment(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
