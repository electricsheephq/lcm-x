#!/usr/bin/env python3
"""Read-only post-merge close-out check for one merged PR.

usage: postmerge_issue_check.py <pr-number> [owner/repo]

Lists every issue the PR body or its commits reference (#N), and reports each
one's state. An issue that is still open after the merge is flagged only when
the PR named it after a closing keyword. GitHub closes only the issue that
directly follows each closing keyword ("Fixes #1, #2" closes #1 only), so this
catches the rest. "Part of #N" and issues only mentioned (follow-ups, context)
are listed, not flagged. It writes nothing. Exit status is 1 when any issue is
flagged.

Requires Python 3.11+ and an authenticated `gh` CLI.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import NamedTuple

DEFAULT_REPO = "electricsheephq/lcm-x"

_REF = re.compile(r"#(\d+)")
_PART_OF = re.compile(r"[Pp]art of #(\d+)")
_CLOSING = re.compile(
    r"(?i)\b(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+((?:#\d+[\s,]*(?:and\s+)?)+)"
)


class IssueRefs(NamedTuple):
    refs: list[int]
    closing: set[int]
    part_of: set[int]


def parse_issue_refs(text: str, pr_number: int | None = None) -> IssueRefs:
    """Return every #N reference, the closing-keyword targets, and "Part of" targets."""
    refs = {int(n) for n in _REF.findall(text)}
    if pr_number is not None:
        refs.discard(pr_number)
    closing: set[int] = set()
    for group in _CLOSING.findall(text):
        closing |= {int(n) for n in _REF.findall(group)}
    part_of = {int(n) for n in _PART_OF.findall(text)}
    return IssueRefs(sorted(refs), closing, part_of)


def _gh(*args: str) -> dict:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    pr = argv[1]
    repo = argv[2] if len(argv) > 2 else DEFAULT_REPO

    info = _gh("pr", "view", pr, "-R", repo, "--json", "state,body,commits,mergedAt,title")
    if info["state"] != "MERGED":
        print(f"PR {pr} is {info['state']}, not merged", file=sys.stderr)
        return 2
    text = info["body"] + "\n" + "\n".join(
        c["messageHeadline"] + "\n" + c.get("messageBody", "") for c in info["commits"]
    )
    parsed = parse_issue_refs(text, int(pr))

    flagged = 0
    print(f"PR #{pr} merged {info['mergedAt']}: {info['title']}")
    for n in parsed.refs:
        try:
            issue = _gh("issue", "view", str(n), "-R", repo, "--json", "state,title,url")
        except subprocess.CalledProcessError:
            continue  # a PR number or a missing issue
        if "/pull/" in issue["url"]:
            continue
        if n in parsed.closing:
            kind = "closing keyword"
        elif n in parsed.part_of:
            kind = "part of"
        else:
            kind = "mentioned"
        note = ""
        if issue["state"] == "OPEN" and n in parsed.closing:
            flagged += 1
            note = "  <-- the PR says it fixes this, but it is still open: close it or comment what remains"
        print(f"  #{n} [{issue['state']}] ({kind}) {issue['title'][:70]}{note}")
    print(f"{flagged} issue(s) need a close-out decision")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
