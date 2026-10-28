"""Require digest-bound explanations for changed golden outcomes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


OUTCOME_HEADS = {
    # Ingest: final/reopened state, caller input, returns, exceptions and payloads.
    "ingest": (
        "state-after ", "search-after ", "input-after ", "aliases-after ",
        "next-state ", "cold-state ", "return-", "raise-", "next-return ",
        "cold-return ", "next-raise ", "cold-raise ", "aliases-0 ", "aliases-1 ",
        "next-aliases ", "payloads-after ", "payloads-cold ", "quarantine-payloads-after ",
    ),
    # Compress: returned context, propagated failures, assembled messages, stored rows/DAG/lifecycle and cursor.
    "compress": ("return ", "raise ", "message ", "assemble raise ",
                 "attr _store=", "attr _dag=", "attr _lifecycle=", "attr _ingest_cursor="),
    # Reconcile: returned cursor/result, propagated failures, caller input, stored rows/DAG/lifecycle and cursor.
    "reconcile": ("return ", "raise ", "input-after ", "aliases-after ",
                  "attr _store=", "attr _dag=", "attr _lifecycle=", "attr _ingest_cursor="),
    # Assembly: output rows/identity, caller input/identity, persisted payloads and stored rows/DAG.
    "assembly": ("row ", "result aliases ", "input after ", "input aliases after ", "payload ",
                 "after _store ", "after _dag "),
}
ALLOWLIST = "tests/fixtures/golden_outcome_changes.json"


def outcomes(family, events):
    return [event for event in events if event.startswith(OUTCOME_HEADS[family])]


def outcome_sha(events):
    return hashlib.sha256("\n".join(events).encode("utf-8")).hexdigest()


def golden(text):
    value = json.loads(text)
    if not isinstance(value, dict) or any(
        not isinstance(events, list) or any(not isinstance(event, str) for event in events)
        for events in value.values()
    ):
        raise ValueError("golden must map scenario names to lists of event strings")
    return value


def changes(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("allowlist must be an object")
    for family, entries in value.items():
        if family not in OUTCOME_HEADS or not isinstance(entries, dict):
            raise ValueError(f"invalid allowlist family: {family}")
        for scenario, entry in entries.items():
            if not isinstance(entry, dict):
                raise ValueError(f"invalid allowlist entry: {family}/{scenario}")
            digest, reason = entry.get("outcome_sha"), entry.get("reason")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or not isinstance(reason, str) or not reason.strip()
                    or len(reason.splitlines()) != 1 or "\n" in reason or "\r" in reason):
                raise ValueError(f"invalid digest or one-line reason: {family}/{scenario}")
    return value


def compare(family, base, new, entries):
    failed, matched = False, set()
    for scenario in sorted(base.keys() | new.keys()):
        before = outcomes(family, base.get(scenario, []))
        after = outcomes(family, new.get(scenario, []))
        if before == after and (scenario in base) == (scenario in new):
            continue
        digest = outcome_sha(after)
        accepted = entries.get(scenario, {}).get("outcome_sha") == digest
        kind = "added" if scenario not in base else "removed" if scenario not in new else "changed"
        print(f"{family}/{scenario}: {kind} ({'listed' if accepted else 'UNLISTED'})")
        if accepted:
            matched.add(scenario)
        else:
            failed = True
            for index in range(max(len(before), len(after))):
                left = before[index] if index < len(before) else "<end>"
                right = after[index] if index < len(after) else "<end>"
                if left != right:
                    print(f"  base: {left}\n  new:  {right}")
                    break
            else:
                print(f"  base: {'present' if scenario in base else '<absent>'}")
                print(f"  new:  {'present' if scenario in new else '<absent>'}")
            print(f'  "outcome_sha": "{digest}"')
    for scenario in sorted(entries.keys() - matched):
        print(f"{family}/{scenario}: stale allowlist entry")
    return int(failed)


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--families", default=",".join(OUTCOME_HEADS))
    args = parser.parse_args(argv)
    families = args.families.split(",")
    if any(family not in OUTCOME_HEADS for family in families):
        parser.error("--families must be a comma-separated subset of " + ",".join(OUTCOME_HEADS))
    try:
        root = Path(git(None, "rev-parse", "--show-toplevel").strip())
        base = git(root, "rev-parse", "--verify", "--end-of-options", args.base_ref + "^{commit}").strip()
        entries = changes((root / ALLOWLIST).read_text(encoding="utf-8"))
        failed = 0
        for family in dict.fromkeys(families):
            path = f"tests/fixtures/{family}_trace.json"
            old = git(root, "show", f"{base}:{path}") if git(root, "ls-tree", "--name-only", base, "--", path) else "{}"
            new = (root / path).read_text(encoding="utf-8") if (root / path).exists() else "{}"
            failed |= compare(family, golden(old), golden(new), entries.get(family, {}))
        return failed
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"golden outcome gate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
