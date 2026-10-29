#!/usr/bin/env bash
# Print the `run:` script of the first step whose name starts with $2, from the composite
# action or the workflow's `test` job in file $1, without a YAML library. It reads the two
# forms CI's steps use, a one-line plain scalar and a `|` literal block, and fails on any
# other form rather than guess. A missing step is an error.
step_script() {
  python - "$1" "$2" <<'PY'
import re
import sys
import textwrap
from pathlib import Path

path, prefix = sys.argv[1:]
text = Path(path).read_text()
if "jobs:" in text:
    job = re.search(r"(?ms)^  test:.*?(?=^  [\w-]+:|\Z)", text)
    if not job:
        sys.exit(f"ci_step: no test job in {path}")
    text = job[0]


def depth(line):
    return len(line) - len(line.lstrip(" "))


steps = re.split(r"(?m)^([ ]*)- name: ", text)
for i in range(1, len(steps), 2):
    indent, body = steps[i:i + 2]
    if not body.splitlines()[0].startswith(prefix):
        continue
    block = re.search(r"(?m)^" + indent + r"  run: (.*)\n", body)
    if not block:
        break
    head, lines = block[1], body[block.end():].splitlines(True)
    first = next((line for line in lines if line.strip()), "")
    keys = len(indent) + 2
    if head == "|" and depth(first) > keys:
        end = next((j for j, line in enumerate(lines)
                    if line.strip() and depth(line) < depth(first)), len(lines))
        print(textwrap.dedent("".join(lines[:end])), end="")
        sys.exit(0)
    if head and head[0] not in "|>\"'&*!" and depth(first) <= keys:
        print(head)
        sys.exit(0)
    sys.exit(f"ci_step: unsupported run form for {prefix!r} in {path}")
sys.exit(f"ci_step: no run block for {prefix!r} in {path}")
PY
}
